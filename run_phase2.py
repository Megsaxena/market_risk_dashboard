"""
run_phase2.py
=============
Phase 2 entry point — VaR and Expected Shortfall analysis.

Run with: python run_phase2.py

Produces:
  - Console report: all three VaR methods, term structure, backtest
  - outputs/var_comparison.csv
  - outputs/var_term_structure.csv
  - outputs/rolling_var.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

from config import RISK_PARAMS, PORTFOLIO_NOTIONAL
from data.fetcher import fetch_universe, compute_returns
from data.portfolio import build_default_portfolio, compute_portfolio_returns
from risk.var_es import (
    compute_all_var,
    compute_var_term_structure,
    compute_rolling_var,
    backtest_summary,
    compute_var_breaches,
)

logging.basicConfig(
    level=logging.WARNING,       # suppress INFO noise for clean report
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
SEPARATOR = "═" * 62


def print_section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def fmt_eur(val):
    return f"€{val:>12,.0f}"

def fmt_pct(val):
    return f"{val:>8.4%}"


def run_phase2():
    os.makedirs("outputs", exist_ok=True)

    # ── Load data (Phase 1) ──────────────────────────
    print("\nLoading data and building portfolio...")
    portfolio = build_default_portfolio()
    prices    = fetch_universe(tickers=portfolio.tickers)
    log_rets  = compute_returns(prices, method="log")
    port_rets = compute_portfolio_returns(log_rets, portfolio)

    conf    = RISK_PARAMS.confidence_level
    horizon = RISK_PARAMS.var_horizon_days
    notion  = portfolio.notional

    print(f"  Portfolio:  {portfolio.name}")
    print(f"  Notional:   €{notion:>12,.0f}")
    print(f"  Confidence: {conf:.0%}")
    print(f"  Horizon:    {horizon}d")
    print(f"  History:    {len(port_rets)} trading days")

    # ────────────────────────────────────────────────
    # 1. RETURN DISTRIBUTION DIAGNOSTICS
    # ────────────────────────────────────────────────
    print_section("RETURN DISTRIBUTION DIAGNOSTICS")
    r = port_rets.dropna()
    daily_vol = r.std()
    skew      = r.skew()
    kurt      = r.kurtosis()
    worst5    = r.nsmallest(5)

    print(f"\n  Daily vol (σ):          {daily_vol:.4%}")
    print(f"  Annualised vol:         {daily_vol * np.sqrt(252):.2%}")
    print(f"  Skewness:               {skew:.4f}  (normal = 0)")
    print(f"  Excess kurtosis:        {kurt:.4f}  (normal = 0)")
    print(f"\n  Why this matters for VaR:")
    if kurt > 3:
        print(f"    ⚠  Fat tails detected (kurtosis={kurt:.1f}).")
        print(f"       Parametric Normal VaR will UNDERESTIMATE tail risk.")
        print(f"       Historical Simulation and MC t-dist are more reliable.")
    else:
        print(f"    ✓  Kurtosis near normal — parametric VaR is reasonable.")
    if skew < -0.3:
        print(f"    ⚠  Negative skew ({skew:.3f}): losses more extreme than gains.")

    print(f"\n  Worst 5 daily returns:")
    for dt, val in worst5.items():
        print(f"    {dt.date()}   {val:+.4%}   ({val * notion:>+12,.0f} EUR)")

    # ────────────────────────────────────────────────
    # 2. POINT-IN-TIME VAR — ALL METHODS
    # ────────────────────────────────────────────────
    print_section(f"POINT-IN-TIME VaR & ES  |  {conf:.0%}  |  {horizon}d horizon")

    all_var = compute_all_var(port_rets, conf, horizon, notion)

    print(f"\n  {'Method':<28}  {'VaR %':>8}  {'VaR (€)':>14}  {'ES %':>8}  {'ES (€)':>14}  {'ES/VaR':>7}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*14}  {'-'*8}  {'-'*14}  {'-'*7}")

    labels = {
        "hs":           "Historical Simulation",
        "parametric":   "Parametric (Normal)",
        "parametric_t": "Parametric (Student-t)",
        "mc":           "Monte Carlo (t-dist)",
    }
    comparison_rows = []
    for key, res in all_var.items():
        label = labels[key]
        print(
            f"  {label:<28}  {fmt_pct(res.var_pct)}  {fmt_eur(res.var_eur)}  "
            f"{fmt_pct(res.es_pct)}  {fmt_eur(res.es_eur)}  {res.es_var_ratio:>7.3f}"
        )
        comparison_rows.append({
            "Method": label,
            "VaR %": res.var_pct,
            "ES %": res.es_pct,
            "VaR EUR": res.var_eur,
            "ES EUR": res.es_eur,
            "ES/VaR": res.es_var_ratio,
        })

    # Commentary
    hs_var  = all_var["hs"].var_pct
    par_var = all_var["parametric"].var_pct
    mc_var  = all_var["mc"].var_pct
    divergence = (hs_var - par_var) / par_var * 100

    print(f"\n  Interpretation:")
    if abs(divergence) > 20:
        direction = "above" if hs_var > par_var else "below"
        print(f"    ⚠  HS VaR is {abs(divergence):.0f}% {direction} Parametric Normal.")
        if hs_var > par_var:
            print(f"       The historical return distribution has heavier tails than normal.")
            print(f"       Relying on Parametric Normal would understate capital requirements.")
    else:
        print(f"    ✓  All three methods broadly agree (divergence: {divergence:+.0f}%).")

    es_var_ratio = all_var["hs"].es_var_ratio
    print(f"\n    ES/VaR ratio (HS): {es_var_ratio:.3f}")
    if es_var_ratio > 1.4:
        print(f"    ⚠  High ratio ({es_var_ratio:.2f}): the tail beyond VaR is long.")
        print(f"       In stress periods, average losses well exceed the VaR threshold.")

    # Save comparison
    pd.DataFrame(comparison_rows).to_csv("outputs/var_comparison.csv", index=False)

    # ────────────────────────────────────────────────
    # 3. MULTIPLE CONFIDENCE LEVELS
    # ────────────────────────────────────────────────
    print_section("VaR SENSITIVITY TO CONFIDENCE LEVEL  (HS, 1d)")

    conf_levels = [0.90, 0.95, 0.975, 0.99, 0.999]
    print(f"\n  {'Confidence':>11}  {'VaR %':>9}  {'VaR (€)':>14}  {'ES %':>9}  {'ES (€)':>14}")
    print(f"  {'-'*11}  {'-'*9}  {'-'*14}  {'-'*9}  {'-'*14}")
    for cl in conf_levels:
        from risk.var_es import var_historical
        res = var_historical(port_rets, cl, notional=notion)
        tag = "  ← Basel" if abs(cl - 0.975) < 0.001 else ""
        tag2 = "  ← Standard" if abs(cl - 0.95) < 0.001 else ""
        print(
            f"  {cl:>10.1%}  {fmt_pct(res.var_pct)}  {fmt_eur(res.var_eur)}  "
            f"{fmt_pct(res.es_pct)}  {fmt_eur(res.es_eur)}{tag}{tag2}"
        )

    # ────────────────────────────────────────────────
    # 4. TERM STRUCTURE
    # ────────────────────────────────────────────────
    print_section("VaR TERM STRUCTURE  (HS vs Parametric vs MC)")

    term_struct = compute_var_term_structure(port_rets, notional=notion)
    print(f"\n  Horizon   HS VaR      HS ES      Par VaR    Par ES     MC VaR")
    print(f"  --------  ---------  ---------  ---------  ---------  ---------")
    for h, row in term_struct.iterrows():
        print(
            f"  {h:>3}d      "
            f"{row['HS VaR %']:>8.4%}   {row['HS ES %']:>8.4%}   "
            f"{row['Param VaR %']:>8.4%}   {row['Param ES %']:>8.4%}   "
            f"{row['MC VaR %']:>8.4%}"
        )
    print(f"\n  Note: All methods use sqrt-T scaling from 1d. MC uses multi-day path simulation.")
    term_struct.to_csv("outputs/var_term_structure.csv")

    # ────────────────────────────────────────────────
    # 5. ROLLING VAR (HS)
    # ────────────────────────────────────────────────
    print_section("ROLLING VAR (Historical Simulation, 252-day window)")

    rolling = compute_rolling_var(port_rets, method="hs", notional=notion)

    print(f"\n  Rolling VaR summary (pct of portfolio):")
    print(f"    Current VaR ({rolling.index[-1].date()}):   {rolling['rolling_var'].iloc[-1]:.4%}")
    print(f"    1-year avg:                 {rolling['rolling_var'].tail(252).mean():.4%}")
    print(f"    Full-history avg:           {rolling['rolling_var'].mean():.4%}")
    print(f"    Maximum rolling VaR:        {rolling['rolling_var'].max():.4%}  "
          f"({rolling['rolling_var'].idxmax().date()})")
    print(f"    Minimum rolling VaR:        {rolling['rolling_var'].min():.4%}  "
          f"({rolling['rolling_var'].idxmin().date()})")

    rolling.to_csv("outputs/rolling_var.csv")

    # ────────────────────────────────────────────────
    # 6. BACKTEST — KUPIEC TEST
    # ────────────────────────────────────────────────
    print_section("VaR BACKTEST  (Kupiec POF Test)")

    bt = backtest_summary(rolling, conf)
    print()
    for k, v in bt.items():
        print(f"    {k:<28}  {v}")

    print(f"\n  What this means:")
    pval = float(bt["Kupiec p-value"]) if bt["Kupiec p-value"] != "n/a" else None
    if pval is not None:
        if pval > 0.05:
            print(f"    ✓  Kupiec p={pval:.3f} > 0.05: cannot reject H0.")
            print(f"       Breach frequency is statistically consistent with {conf:.0%} confidence.")
        else:
            print(f"    ⚠  Kupiec p={pval:.4f} < 0.05: reject H0.")
            print(f"       Breach frequency is INCONSISTENT with {conf:.0%} VaR model.")

    # ────────────────────────────────────────────────
    # 7. SAVE ALL OUTPUTS
    # ────────────────────────────────────────────────
    print_section("OUTPUTS SAVED")
    print(f"\n  ✓ outputs/var_comparison.csv     — all methods side by side")
    print(f"  ✓ outputs/var_term_structure.csv  — 1d to 10d horizon")
    print(f"  ✓ outputs/rolling_var.csv         — time series of rolling VaR")
    print(f"\n  Phase 2 complete. Ready for Phase 3 (Drawdown + Volatility).")
    print(f"\n{SEPARATOR}\n")

    return {
        "all_var": all_var,
        "term_structure": term_struct,
        "rolling_var": rolling,
        "portfolio_returns": port_rets,
        "portfolio": portfolio,
    }


if __name__ == "__main__":
    run_phase2()
