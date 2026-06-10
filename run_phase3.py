"""
run_phase3.py
=============
Phase 3 entry point — Drawdowns and Volatility.

Run with: python run_phase3.py

Outputs:
  - outputs/drawdown_series.csv
  - outputs/drawdown_episodes.csv
  - outputs/rolling_vol.csv
  - outputs/vol_term_structure.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import numpy as np
import pandas as pd

from data.fetcher import fetch_universe, compute_returns
from data.portfolio import build_default_portfolio, compute_portfolio_returns
from risk.drawdown import (
    compute_drawdown_series,
    compute_drawdown_episodes,
    compute_drawdown_metrics,
    compute_rolling_max_drawdown,
)
from risk.volatility import (
    compute_all_vol,
    classify_vol_regime,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
SEPARATOR = "═" * 62


def print_section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def run_phase3():
    os.makedirs("outputs", exist_ok=True)

    # ── Load data ────────────────────────────
    print("\nLoading portfolio...")
    portfolio = build_default_portfolio()
    prices    = fetch_universe(tickers=portfolio.tickers)
    log_rets  = compute_returns(prices, method="log")
    port_rets = compute_portfolio_returns(log_rets, portfolio)

    print(f"  Loaded {len(port_rets)} trading days  "
          f"({port_rets.index[0].date()} → {port_rets.index[-1].date()})")

    # ════════════════════════════════════════
    # DRAWDOWNS
    # ════════════════════════════════════════

    dd_series = compute_drawdown_series(port_rets)
    episodes  = compute_drawdown_episodes(dd_series, top_n=10)
    metrics   = compute_drawdown_metrics(port_rets, dd_series)
    rolling_mdd = compute_rolling_max_drawdown(port_rets, window=252)

    # ── 1. Current Status ────────────────────
    print_section("DRAWDOWN — CURRENT STATUS")
    current_dd = metrics["Current Drawdown"]
    peak_date  = dd_series[dd_series["drawdown"] == 0].index[-1] \
                 if (dd_series["drawdown"] == 0).any() else "n/a"

    print(f"\n  Current NAV (normalised):  {dd_series['nav'].iloc[-1]:.4f}")
    print(f"  Last peak:                 {peak_date}")
    print(f"  Current drawdown:          {current_dd:.4%}", end="")
    if current_dd < -0.05:
        days_since_peak = (port_rets.index[-1] - pd.Timestamp(peak_date)).days \
                          if peak_date != "n/a" else "n/a"
        print(f"   ⚠  ({days_since_peak} days since peak)")
    else:
        print(f"   ✓  (near peak)")

    # ── 2. Key Metrics ───────────────────────
    print_section("DRAWDOWN METRICS")
    print(f"\n  {'Metric':<30}  {'Value':>12}")
    print(f"  {'-'*30}  {'-'*12}")

    format_map = {
        "Max Drawdown":           lambda v: f"{v:.4%}",
        "Current Drawdown":       lambda v: f"{v:.4%}",
        "Average Drawdown":       lambda v: f"{v:.4%}",
        "Time in Drawdown":       lambda v: f"{v:.1%}",
        "Calmar Ratio":           lambda v: f"{v:.3f}",
        "Sterling Ratio":         lambda v: f"{v:.3f}",
        "Ulcer Index":            lambda v: f"{v:.4%}",
        "Pain Index":             lambda v: f"{v:.4%}",
        "Avg Recovery (days)":    lambda v: f"{v:.0f}d" if not np.isnan(v) else "n/a",
        "Max Recovery (days)":    lambda v: f"{v:.0f}d" if not np.isnan(v) else "n/a",
        "Annualised Return":      lambda v: f"{v:.2%}",
    }
    for metric, val in metrics.items():
        fmt = format_map.get(metric, lambda v: str(v))
        try:
            val_str = fmt(val)
        except Exception:
            val_str = str(val)
        print(f"  {metric:<30}  {val_str:>12}")

    # ── 3. Interpretation ────────────────────
    mdd = metrics["Max Drawdown"]
    calmar = metrics["Calmar Ratio"]
    ulcer  = metrics["Ulcer Index"]
    time_in_dd = metrics["Time in Drawdown"]

    print(f"\n  Interpretation:")
    print(f"    Max Drawdown {mdd:.1%}: ", end="")
    if mdd > -0.20:
        print("moderate — typical for a diversified multi-asset portfolio.")
    elif mdd > -0.35:
        print("significant. Portfolio spent >20% underwater at worst.")
    else:
        print("severe. Consider whether this is acceptable for mandate.")

    print(f"    Calmar Ratio {calmar:.2f}: ", end="")
    if calmar > 0.5:
        print("good — return per unit of max drawdown is healthy.")
    elif calmar > 0.25:
        print("acceptable — in line with a 60/40 benchmark.")
    else:
        print("weak — the drawdown cost is high relative to return earned.")

    print(f"    Ulcer Index {ulcer:.3%}: measures combined depth + duration.")
    print(f"    Time in drawdown: {time_in_dd:.1%} of all trading days "
          f"({'above average' if time_in_dd > 0.5 else 'below average'}).")

    # ── 4. Top Drawdown Episodes ─────────────
    print_section("TOP DRAWDOWN EPISODES")
    if not episodes.empty:
        print(f"\n  {'#':<3}  {'Start':<12}  {'Trough':<12}  {'Depth':>8}  "
              f"{'DD Days':>8}  {'Rec Days':>9}  {'Recovered':>10}")
        print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*10}")

        for rank, row in episodes.iterrows():
            rec_days = f"{int(row['Recovery (days)'])}d" \
                       if pd.notna(row['Recovery (days)']) else "ongoing"
            recovered = "✓ Yes" if row["Recovered"] else "⚠ No"
            print(
                f"  {rank:<3}  {str(row['Start'].date()):<12}  "
                f"{str(row['Trough'].date()):<12}  "
                f"{row['Depth']:>7.2%}  "
                f"{int(row['Drawdown (days)']):>7}d  "
                f"{rec_days:>9}  {recovered:>10}"
            )

    # ── 5. Rolling MDD ───────────────────────
    print_section("ROLLING MAX DRAWDOWN (252-day window)")
    print(f"\n  Current (1-year window):  {rolling_mdd.iloc[-1]:.4%}")
    print(f"  1-year avg:               {rolling_mdd.tail(252).mean():.4%}")
    print(f"  Full-history avg:         {rolling_mdd.mean():.4%}")
    print(f"  Worst rolling period:     {rolling_mdd.min():.4%}  "
          f"({rolling_mdd.idxmin().date()})")

    # Save drawdown outputs
    dd_series.to_csv("outputs/drawdown_series.csv")
    if not episodes.empty:
        episodes.to_csv("outputs/drawdown_episodes.csv")
    rolling_mdd.to_frame("rolling_mdd").to_csv("outputs/rolling_mdd.csv")

    # ════════════════════════════════════════
    # VOLATILITY
    # ════════════════════════════════════════

    vol_data = compute_all_vol(port_rets, run_garch=True)

    # ── 6. Current Vol Snapshot ──────────────
    print_section("VOLATILITY — CURRENT SNAPSHOT")
    regime = vol_data["regime"]

    print(f"\n  Current EWMA Vol (λ=0.94):   {vol_data['current_ewma']:.2%}")
    print(f"  Current 21d Realised Vol:    {vol_data['current_21d']:.2%}")
    print(f"  Current 252d Realised Vol:   {vol_data['current_252d']:.2%}")
    print(f"\n  Vol Regime:                  {regime['regime']}")
    print(f"  Percentile Rank:             {regime['percentile_rank']:.1f}th")
    print(f"  Historical Mean (EWMA):      {regime['history_mean']:.2%}")
    print(f"  Historical Median (EWMA):    {regime['history_median']:.2%}")
    print(f"  Historical Max (EWMA):       {regime['history_max']:.2%}")

    # ── 7. Vol Term Structure ────────────────
    print_section("VOL TERM STRUCTURE (current snapshot)")
    ts = vol_data["term_structure"]
    print(f"\n  {'Window':<10}  {'Ann. Vol':>10}  {'vs Long-run':>12}  {'Signal':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}")
    for _, row in ts.iterrows():
        vslr = f"{row['vs Long-run']:.2f}×" if "vs Long-run" in row.index else "—"
        sig  = row.get("Signal", "—")
        print(f"  {row['Window']:<10}  {row['Current Vol']:>9.2%}  {vslr:>12}  {sig:>12}")

    if len(ts) >= 2:
        short = ts.iloc[0]["Current Vol"]
        long_ = ts.iloc[-1]["Current Vol"]
        if short > long_ * 1.15:
            print(f"\n  ⚠  Term structure INVERTED: short-term vol > long-term vol.")
            print(f"     Near-term stress is elevated relative to structural level.")
        elif short < long_ * 0.85:
            print(f"\n  ✓  Term structure NORMAL: short-term vol < long-term vol.")
            print(f"     Current environment is calmer than the long-run average.")
        else:
            print(f"\n  ≈  Term structure FLAT: short and long-term vol broadly equal.")

    # ── 8. GARCH Parameters ──────────────────
    if vol_data["garch_params"]:
        print_section("GARCH(1,1) PARAMETERS")
        gp = vol_data["garch_params"]
        print(f"\n  ω (omega — base variance):          {gp.get('omega', 0):.2e}")
        print(f"  α (alpha — shock sensitivity):      {gp.get('alpha', 0):.4f}")
        print(f"  β (beta  — vol persistence):        {gp.get('beta',  0):.4f}")
        print(f"  α + β   (total persistence):        {gp.get('persistence', 0):.4f}")
        print(f"  Long-run vol (annualised):          {gp.get('long_run_vol', 0):.2%}")
        print(f"  Mean-reversion half-life:           {gp.get('half_life_days', 0):.1f} days")
        persistence = gp.get("persistence", 0)
        print(f"\n  Interpretation:")
        if persistence > 0.97:
            print(f"    ⚠  Very high persistence ({persistence:.3f}): vol shocks "
                  f"dissipate slowly.")
            print(f"       GARCH is close to IGARCH — vol has near-unit-root behaviour.")
        elif persistence > 0.92:
            print(f"    Moderate-high persistence ({persistence:.3f}): "
                  f"half-life {gp.get('half_life_days',0):.0f}d.")
            print(f"       Elevated vol after a shock will persist for weeks.")
        else:
            print(f"    Lower persistence ({persistence:.3f}): vol mean-reverts quickly.")

    # ── 9. Vol of Vol ────────────────────────
    print_section("VOL OF VOL (VoV)")
    vov = vol_data["vol_of_vol"].dropna()
    current_vov = vov.iloc[-1]
    vov_pctile  = (vov < current_vov).mean() * 100
    print(f"\n  Current VoV (63d):    {current_vov:.2%}")
    print(f"  VoV percentile rank:  {vov_pctile:.1f}th")
    print(f"  VoV mean (history):   {vov.mean():.2%}")
    print(f"  VoV max (history):    {vov.max():.2%}  ({vov.idxmax().date()})")
    print(f"\n  Interpretation:")
    if vov_pctile > 80:
        print(f"    ⚠  VoV is in the top 20%. The vol regime itself is unstable.")
        print(f"       VaR estimates are less reliable; hedge ratios should be wider.")
    elif vov_pctile > 60:
        print(f"    The volatility environment is somewhat unsettled.")
    else:
        print(f"    ✓  VoV is subdued — the vol regime is relatively stable.")

    # ── 10. Rolling vol summary ──────────────
    print_section("ROLLING REALISED VOL (summary statistics)")
    rv = vol_data["rolling_vol"].dropna()
    print(f"\n  {'Window':<10}  {'Current':>9}  {'1yr Avg':>9}  {'Min':>9}  {'Max':>9}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    for col in rv.columns:
        label = col.replace("vol_", "").replace("d", "d")
        curr  = rv[col].iloc[-1]
        avg1y = rv[col].tail(252).mean()
        mn    = rv[col].min()
        mx    = rv[col].max()
        print(f"  {label:<10}  {curr:>8.2%}  {avg1y:>8.2%}  {mn:>8.2%}  {mx:>8.2%}")

    # ── Save outputs ─────────────────────────
    print_section("OUTPUTS SAVED")
    vol_data["rolling_vol"].to_csv("outputs/rolling_vol_all_windows.csv")
    vol_data["term_structure"].to_csv("outputs/vol_term_structure.csv", index=False)
    pd.DataFrame({
        "ewma_vol": vol_data["ewma_vol"],
        "vol_of_vol": vol_data["vol_of_vol"],
        **({"garch_vol": vol_data["garch_vol"]} if vol_data["garch_vol"] is not None else {}),
    }).to_csv("outputs/vol_estimators.csv")

    print(f"\n  ✓ outputs/drawdown_series.csv")
    print(f"  ✓ outputs/drawdown_episodes.csv")
    print(f"  ✓ outputs/rolling_mdd.csv")
    print(f"  ✓ outputs/rolling_vol_all_windows.csv")
    print(f"  ✓ outputs/vol_term_structure.csv")
    print(f"  ✓ outputs/vol_estimators.csv")
    print(f"\n  Phase 3 complete. Ready for Phase 4 (Correlation Risk).")
    print(f"\n{SEPARATOR}\n")

    return {
        "portfolio_returns":  port_rets,
        "portfolio":          portfolio,
        "dd_series":          dd_series,
        "dd_episodes":        episodes,
        "dd_metrics":         metrics,
        "rolling_mdd":        rolling_mdd,
        "vol_data":           vol_data,
    }


if __name__ == "__main__":
    run_phase3()
