"""
run_phase4.py
=============
Phase 4 — Correlation Risk.

Run with: python run_phase4.py

Outputs:
  - outputs/correlation_current.csv
  - outputs/avg_corr_timeseries.csv
  - outputs/pca_decomposition.csv
  - outputs/diversification.csv
  - outputs/conditional_correlations.csv
  - outputs/asset_class_block_corr.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import numpy as np
import pandas as pd

from data.fetcher import fetch_universe, compute_returns
from data.portfolio import build_default_portfolio, compute_portfolio_returns
from risk.correlation import (
    compute_rolling_correlation,
    compute_pca_decomposition,
    compute_rolling_pc1,
    compute_diversification_ratio,
    compute_conditional_correlations,
    compute_asset_class_correlations,
    classify_correlation_regime,
)
from config import UNIVERSE

logging.basicConfig(level=logging.WARNING)
SEPARATOR = "═" * 62


def print_section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def corr_fmt(v):
    if isinstance(v, float) and not np.isnan(v):
        bar = "█" * int(abs(v) * 12)
        sign = "+" if v >= 0 else "-"
        return f"{v:+.3f}  {sign}{bar}"
    return "  n/a"


def run_phase4():
    os.makedirs("outputs", exist_ok=True)

    # ── Load data ────────────────────────────
    print("\nLoading portfolio and data...")
    portfolio = build_default_portfolio()
    prices    = fetch_universe(tickers=portfolio.tickers)
    log_rets  = compute_returns(prices, method="log")
    port_rets = compute_portfolio_returns(log_rets, portfolio)

    print(f"  {len(log_rets)} days  ×  {len(log_rets.columns)} assets")

    # ════════════════════════════════════════
    # 1. ROLLING CORRELATION — CURRENT MATRIX
    # ════════════════════════════════════════
    print_section(f"CURRENT CORRELATION MATRIX  ({RISK_PARAMS_window}d window)")

    roll_corr = compute_rolling_correlation(log_rets)
    curr_corr = roll_corr["current_corr"]
    tickers   = roll_corr["tickers"]

    # Print as a neat lower-triangle table
    names = {t: UNIVERSE[t]["name"] if t in UNIVERSE else t for t in tickers}

    print(f"\n  {'':>6}", end="")
    for t in tickers:
        print(f"  {t:>6}", end="")
    print()
    print(f"  {'─'*6}", end="")
    for _ in tickers:
        print(f"  {'─'*6}", end="")
    print()

    for i, t1 in enumerate(tickers):
        print(f"  {t1:>6}", end="")
        for j, t2 in enumerate(tickers):
            if j > i:
                print(f"  {'':>6}", end="")
            else:
                v = curr_corr.loc[t1, t2]
                if i == j:
                    print(f"  {'1.000':>6}", end="")
                elif v > 0.7:
                    print(f"  \033[91m{v:>+.3f}\033[0m", end="")   # red: high positive
                elif v < -0.2:
                    print(f"  \033[94m{v:>+.3f}\033[0m", end="")   # blue: negative
                else:
                    print(f"  {v:>+.3f}", end="")
        print(f"  {t1}")

    curr_corr.to_csv("outputs/correlation_current.csv")

    # ════════════════════════════════════════
    # 2. AVERAGE PAIRWISE CORRELATION + REGIME
    # ════════════════════════════════════════
    print_section("AVERAGE PAIRWISE CORRELATION — REGIME")

    avg_corr_ts = roll_corr["avg_corr_ts"]
    regime      = classify_correlation_regime(avg_corr_ts)

    current_avg = regime["current_avg_corr"]
    pctile      = regime["percentile_rank"]

    print(f"\n  Current avg pairwise correlation:  {current_avg:+.4f}")
    print(f"  Percentile rank:                   {pctile:.1f}th")
    print(f"  Regime:                            {regime['regime']}")
    print(f"  → {regime['interpretation']}")
    print(f"\n  Historical distribution:")
    print(f"    Min:    {regime['history_min']:+.4f}")
    print(f"    Mean:   {regime['history_mean']:+.4f}")
    print(f"    p33:    {regime['p33']:+.4f}   ← Low / Normal boundary")
    print(f"    p66:    {regime['p66']:+.4f}   ← Normal / Elevated boundary")
    print(f"    p85:    {regime['p85']:+.4f}   ← Elevated / Crisis boundary")
    print(f"    Max:    {regime['history_max']:+.4f}")

    avg_corr_ts.to_frame().to_csv("outputs/avg_corr_timeseries.csv")

    # ════════════════════════════════════════
    # 3. PCA DECOMPOSITION
    # ════════════════════════════════════════
    print_section("PCA DECOMPOSITION — FACTOR STRUCTURE")

    pca   = compute_pca_decomposition(log_rets)
    pc1   = pca["pc1_share"]
    effN  = pca["effective_n"]
    n80   = pca["n_pcs_80pct"]

    print(f"\n  PC1 variance share:     {pc1:.1%}")
    print(f"  Effective N:            {effN:.2f}  (out of {pca['n_assets']} assets)")
    print(f"  PCs needed for 80% var: {n80}")

    print(f"\n  Variance explained per PC:")
    print(f"  {'PC':<5}  {'Eigenvalue':>12}  {'Var Exp':>9}  {'Cum Var':>9}  {'Bar'}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*30}")
    for i, (ev, ve, cv) in enumerate(
        zip(pca["eigenvalues"][:8], pca["var_explained"][:8], pca["cum_var_explained"][:8])
    ):
        bar = "█" * int(ve * 60)
        print(f"  PC{i+1:<3}  {ev:>12.4f}  {ve:>8.2%}  {cv:>8.2%}  {bar}")

    print(f"\n  PC1 Factor Loadings (dominant risk factor):")
    print(f"  Which assets load most heavily on the first principal component?")
    print(f"\n  {'Ticker':<8}  {'Name':<25}  {'Loading':>8}  {'Direction'}")
    print(f"  {'─'*8}  {'─'*25}  {'─'*8}  {'─'*20}")
    for ticker, loading in pca["pc1_top_contributors"].items():
        signed = pca["pc1_loadings"][ticker]
        name   = UNIVERSE[ticker]["name"] if ticker in UNIVERSE else ticker
        ac     = UNIVERSE[ticker]["asset_class"] if ticker in UNIVERSE else ""
        direction = "↑ Risk-on" if signed > 0 else "↓ Hedge/Diversifier"
        print(f"  {ticker:<8}  {name:<25}  {abs(loading):>8.4f}  {direction}")

    print(f"\n  Interpretation:")
    if pc1 > 0.50:
        print(f"    ⚠  PC1 explains {pc1:.0%} of variance. Portfolio is dominated by")
        print(f"       a single risk factor. Diversification benefit is limited.")
    elif pc1 > 0.35:
        print(f"    PC1 at {pc1:.0%} reflects meaningful co-movement but")
        print(f"       {n80} factors are needed for 80% of variance —")
        print(f"       the portfolio has genuine multi-factor structure.")
    else:
        print(f"    ✓  PC1 only {pc1:.0%}: portfolio has rich factor diversity.")

    print(f"    Effective N = {effN:.1f} assets: the 12-asset portfolio has the")
    print(f"    diversification equivalent of {effN:.1f} fully independent assets.")

    # Rolling PC1
    print(f"\n  Rolling PC1 (recent history):")
    rolling_pc1 = compute_rolling_pc1(log_rets)
    curr_pc1    = rolling_pc1.iloc[-1]
    pc1_pctile  = (rolling_pc1 < curr_pc1).mean() * 100
    pc1_max     = rolling_pc1.max()
    pc1_max_dt  = rolling_pc1.idxmax().date()
    print(f"    Current PC1 share:     {curr_pc1:.1%}  (pctile: {pc1_pctile:.0f}th)")
    print(f"    Full-history avg:      {rolling_pc1.mean():.1%}")
    print(f"    Crisis peak:           {pc1_max:.1%}  ({pc1_max_dt})")

    pca_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca["eigenvalues"]))],
        "Eigenvalue": pca["eigenvalues"],
        "Var_Explained": pca["var_explained"],
        "Cum_Var": pca["cum_var_explained"],
    })
    pca_df.to_csv("outputs/pca_decomposition.csv", index=False)

    # ════════════════════════════════════════
    # 4. DIVERSIFICATION RATIO
    # ════════════════════════════════════════
    print_section("DIVERSIFICATION RATIO")

    div = compute_diversification_ratio(log_rets, portfolio.weights)
    dr  = div["dr"]

    print(f"\n  Diversification Ratio:     {dr:.4f}")
    print(f"  Portfolio vol:             {div['portfolio_vol'] * np.sqrt(252):.2%}  (annualised)")
    print(f"  Weighted-avg asset vol:    {div['weighted_avg_vol'] * np.sqrt(252):.2%}  (annualised)")
    print(f"  Diversification benefit:   {div['diversification_benefit']:.2f}%  vol reduction")

    print(f"\n  Interpretation:")
    if dr > 1.5:
        print(f"    ✓  Strong diversification (DR={dr:.2f}). Portfolio vol is")
        print(f"       significantly below the weighted average of its parts.")
    elif dr > 1.2:
        print(f"    Moderate diversification (DR={dr:.2f}). Meaningful but")
        print(f"       not exceptional benefit from asset combination.")
    else:
        print(f"    ⚠  Weak diversification (DR={dr:.2f}). Assets are highly")
        print(f"       correlated — the portfolio vol is close to a single-asset bet.")

    # Rolling DR
    dr_roll = div["dr_rolling"].dropna()
    print(f"\n  Rolling DR (63d window):")
    print(f"    Current:       {dr_roll.iloc[-1]:.4f}")
    print(f"    1yr avg:       {dr_roll.tail(252).mean():.4f}")
    print(f"    Full avg:      {dr_roll.mean():.4f}")
    print(f"    Min (crisis):  {dr_roll.min():.4f}  ({dr_roll.idxmin().date()})")
    print(f"    Max (calm):    {dr_roll.max():.4f}  ({dr_roll.idxmax().date()})")

    print(f"\n  Risk Contribution by Asset:")
    print(f"  {'Ticker':<8}  {'Name':<25}  {'% Risk Contrib':>15}  {'Bar'}")
    print(f"  {'─'*8}  {'─'*25}  {'─'*15}  {'─'*25}")
    pct = div["pct_risk_contribution"].sort_values(ascending=False)
    for ticker, contrib in pct.items():
        name = UNIVERSE[ticker]["name"] if ticker in UNIVERSE else ticker
        bar  = "█" * int(contrib * 50)
        print(f"  {ticker:<8}  {name:<25}  {contrib:>14.1%}  {bar}")

    div["dr_rolling"].to_frame().to_csv("outputs/diversification_ratio.csv")

    # ════════════════════════════════════════
    # 5. CONDITIONAL CORRELATION (CRISIS)
    # ════════════════════════════════════════
    print_section("CONDITIONAL CORRELATIONS  (vs SPY, bottom 10% days)")

    cond = compute_conditional_correlations(log_rets, threshold_pct=0.10)

    print(f"\n  Market proxy: {cond['market_proxy']}")
    print(f"  Crisis days (bottom 10%): {cond['n_crisis_days']}")
    print(f"  Calm days (top 10%):      {cond['n_calm_days']}")
    print(f"\n  Avg pairwise corr:")
    print(f"    Unconditional:  {cond['avg_unconditional']:+.4f}")
    print(f"    Crisis:         {cond['avg_crisis']:+.4f}  "
          f"{'↑ WORSE' if cond['avg_crisis'] > cond['avg_unconditional'] else '↓ better'}")

    print(f"\n  {'Asset':<6}  {'Name':<25}  {'Uncond':>8}  {'Crisis':>8}  {'Calm':>8}  {'Asymm':>8}  {'Works?':>10}")
    print(f"  {'─'*6}  {'─'*25}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")

    at = cond["asymmetry_table"]
    for ticker, row in at.iterrows():
        name = UNIVERSE[ticker]["name"] if ticker in UNIVERSE else ticker
        unc  = f"{row['Unconditional']:+.3f}" if pd.notna(row['Unconditional']) else "  n/a"
        cris = f"{row['Crisis Corr']:+.3f}"   if pd.notna(row['Crisis Corr'])   else "  n/a"
        calm = f"{row['Calm Corr']:+.3f}"     if pd.notna(row['Calm Corr'])     else "  n/a"
        asym = f"{row['Asymmetry']:+.3f}"     if pd.notna(row['Asymmetry'])     else "  n/a"
        div_label = row['Diversifies?']
        print(f"  {ticker:<6}  {name:<25}  {unc:>8}  {cris:>8}  {calm:>8}  {asym:>8}  {div_label:>10}")

    print(f"\n  Interpretation:")
    print(f"    Assets with positive asymmetry = correlation RISES in crises.")
    print(f"    These are the assets that fail to diversify when it matters most.")
    print(f"    Negative asymmetry (e.g. TLT, GLD) = correlations fall in stress")
    print(f"    — these are genuine hedges.")

    at.to_csv("outputs/conditional_correlations.csv")

    # ════════════════════════════════════════
    # 6. ASSET CLASS BLOCK CORRELATIONS
    # ════════════════════════════════════════
    print_section("ASSET-CLASS BLOCK CORRELATIONS")

    ac = compute_asset_class_correlations(log_rets)
    bc = ac["block_corr"]

    print(f"\n  Full-history average intra/inter-class correlations:")
    print(f"\n  {'':>12}", end="")
    for col in bc.columns:
        print(f"  {col:>12}", end="")
    print()
    for idx in bc.index:
        print(f"  {idx:>12}", end="")
        for col in bc.columns:
            v = bc.loc[idx, col]
            if isinstance(v, float) and not np.isnan(v):
                if v > 0.5:
                    print(f"  \033[91m{v:>+.3f}\033[0m      ", end="")
                elif v < 0:
                    print(f"  \033[94m{v:>+.3f}\033[0m      ", end="")
                else:
                    print(f"  {v:>+.3f}      ", end="")
        print()

    # Rolling equity-bond correlation
    eq_bond = ac["roll_eq_bond"].dropna()
    eq_comm = ac["roll_eq_comm"].dropna()
    print(f"\n  Rolling Equity-Bond correlation ({RISK_PARAMS_window}d window):")
    print(f"    Current:         {eq_bond.iloc[-1]:+.4f}", end="")
    if eq_bond.iloc[-1] > 0:
        print(f"  ⚠  POSITIVE — bonds NOT providing equity hedge")
    else:
        print(f"  ✓  Negative — bonds hedging equity risk")
    print(f"    Full-history avg: {eq_bond.mean():+.4f}")
    print(f"    Min (2022 shock): {eq_bond.min():+.4f}  ({eq_bond.idxmin().date()})")
    print(f"    Max:              {eq_bond.max():+.4f}  ({eq_bond.idxmax().date()})")

    print(f"\n  Rolling Equity-Commodity correlation:")
    print(f"    Current:          {eq_comm.iloc[-1]:+.4f}")
    print(f"    Full-history avg: {eq_comm.mean():+.4f}")

    bc.to_csv("outputs/asset_class_block_corr.csv")
    pd.DataFrame({
        "equity_bond_corr": eq_bond,
        "equity_commodity_corr": eq_comm,
    }).to_csv("outputs/rolling_class_correlations.csv")

    # ── Save outputs ─────────────────────────
    print_section("OUTPUTS SAVED")
    files = [
        "correlation_current.csv",
        "avg_corr_timeseries.csv",
        "pca_decomposition.csv",
        "diversification_ratio.csv",
        "conditional_correlations.csv",
        "asset_class_block_corr.csv",
        "rolling_class_correlations.csv",
    ]
    for f in files:
        print(f"  ✓ outputs/{f}")
    print(f"\n  Phase 4 complete. Ready for Phase 5 (Stress Tests).")
    print(f"\n{SEPARATOR}\n")

    return {
        "log_returns": log_rets,
        "portfolio":   portfolio,
        "roll_corr":   roll_corr,
        "pca":         pca,
        "div":         div,
        "cond_corr":   cond,
        "ac_corr":     ac,
    }


# ── Import fix for RISK_PARAMS_window ─────────
from config import RISK_PARAMS
RISK_PARAMS_window = RISK_PARAMS.corr_window


if __name__ == "__main__":
    run_phase4()
