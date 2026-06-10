"""
run_phase5.py
=============
Phase 5 — Stress Tests.

Run with: python run_phase5.py

Outputs:
  - outputs/stress_all_scenarios.csv
  - outputs/stress_asset_breakdown.csv
  - outputs/stress_vs_var.csv
  - outputs/reverse_stress_test.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import numpy as np
import pandas as pd

from data.fetcher import fetch_universe, compute_returns
from data.portfolio import build_default_portfolio, compute_portfolio_returns
from risk.var_es import var_historical
from risk.stress import (
    run_all_scenarios,
    run_historical_scenario,
    run_hypothetical_scenario,
    reverse_stress_test,
    compare_scenarios_to_var,
    results_to_dataframe,
    HISTORICAL_SCENARIOS,
    HYPOTHETICAL_SCENARIOS,
    HypotheticalScenario,
)
from config import UNIVERSE, PORTFOLIO_NOTIONAL

logging.basicConfig(level=logging.WARNING)
SEPARATOR = "═" * 70


def print_section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def bar(val, width=30):
    """Signed bar: positive = ▓, negative = ░"""
    if val == 0:
        return ""
    n = min(int(abs(val) * width), width)
    c = "▓" if val > 0 else "░"
    return c * n


def run_phase5():
    os.makedirs("outputs", exist_ok=True)

    # ── Load data ────────────────────────────
    print("\nLoading portfolio...")
    portfolio = build_default_portfolio()
    prices    = fetch_universe(tickers=portfolio.tickers)
    log_rets  = compute_returns(prices, method="log")
    port_rets = compute_portfolio_returns(log_rets, portfolio)

    notion = portfolio.notional
    weights = portfolio.weights

    # Compute current VaR/ES for comparison
    var_res = var_historical(port_rets, notional=notion)
    var_1d  = var_res.var_pct
    es_1d   = var_res.es_pct

    print(f"  Portfolio notional:  €{notion:>12,.0f}")
    print(f"  Current 1d HS VaR:   {var_1d:.4%}  (€{var_1d*notion:>+,.0f})")
    print(f"  Current 1d HS ES:    {es_1d:.4%}  (€{es_1d*notion:>+,.0f})")

    # ── Run all scenarios ────────────────────
    results = run_all_scenarios(log_rets, weights, notion)

    # ════════════════════════════════════════
    # 1. MASTER SCENARIO TABLE
    # ════════════════════════════════════════
    print_section("ALL SCENARIOS — RANKED BY LOSS (worst first)")

    print(f"\n  {'Scenario':<30}  {'Cat':<12}  {'Return':>8}  {'P&L (EUR)':>14}  {'Severity':<20}  Magnitude")
    print(f"  {'─'*30}  {'─'*12}  {'─'*8}  {'─'*14}  {'─'*20}  {'─'*30}")

    for r in results:
        pnl_bar = bar(r.total_return)
        print(
            f"  {r.name:<30}  {r.category:<12}  "
            f"{r.total_return:>+7.2%}  "
            f"€{r.total_pnl:>+13,.0f}  "
            f"{r.severity:<20}  {pnl_bar}"
        )

    # ════════════════════════════════════════
    # 2. HISTORICAL SCENARIOS — DEEP DIVE
    # ════════════════════════════════════════
    print_section("HISTORICAL SCENARIOS — ASSET BREAKDOWN")

    hist_results = [r for r in results if r.category == "Historical"]

    for res in hist_results:
        pnl_k  = res.total_pnl / 1000
        print(f"\n  ┌─ {res.name}  {res.severity}")
        print(f"  │  {res.description}")
        print(f"  │  Portfolio: {res.total_return:+.2%}  (€{res.total_pnl:>+,.0f})")
        print(f"  │  = {res.total_pnl / (-var_1d * notion):.1f}× 1d VaR")
        print(f"  │")
        print(f"  │  {'Asset':<6}  {'Name':<24}  {'Shock':>8}  {'P&L (EUR)':>12}  Contribution")
        print(f"  │  {'─'*6}  {'─'*24}  {'─'*8}  {'─'*12}  {'─'*25}")

        sorted_contribs = res.asset_contributions.sort_values()
        for ticker, pnl in sorted_contribs.items():
            shock = res.asset_shocks.get(ticker, 0)
            name  = UNIVERSE[ticker]["name"] if ticker in UNIVERSE else ticker
            pnl_bar = bar(pnl / notion, width=20)
            print(f"  │  {ticker:<6}  {name:<24}  {shock:>+7.2%}  €{pnl:>+11,.0f}  {pnl_bar}")
        print(f"  └─")

    # ════════════════════════════════════════
    # 3. HYPOTHETICAL SCENARIOS — DEEP DIVE
    # ════════════════════════════════════════
    print_section("HYPOTHETICAL SCENARIOS — ASSET BREAKDOWN")

    hyp_results = [r for r in results if r.category != "Historical"]

    for res in hyp_results:
        print(f"\n  ┌─ {res.name}  [{res.category}]  {res.severity}")
        print(f"  │  {res.description}")
        print(f"  │  Portfolio: {res.total_return:+.2%}  (€{res.total_pnl:>+,.0f})")
        print(f"  │  = {res.total_pnl / (-var_1d * notion):.1f}× 1d VaR")
        print(f"  │")
        print(f"  │  {'Asset':<6}  {'Name':<24}  {'Shock':>8}  {'P&L (EUR)':>12}  Contribution")
        print(f"  │  {'─'*6}  {'─'*24}  {'─'*8}  {'─'*12}  {'─'*25}")

        sorted_contribs = res.asset_contributions.sort_values()
        for ticker, pnl in sorted_contribs.items():
            shock = res.asset_shocks.get(ticker, 0)
            if shock == 0 and pnl == 0:
                continue
            name  = UNIVERSE[ticker]["name"] if ticker in UNIVERSE else ticker
            pnl_bar = bar(pnl / notion, width=20)
            print(f"  │  {ticker:<6}  {name:<24}  {shock:>+7.2%}  €{pnl:>+11,.0f}  {pnl_bar}")
        print(f"  └─")

    # ════════════════════════════════════════
    # 4. SCENARIOS vs VaR COMPARISON
    # ════════════════════════════════════════
    print_section("SCENARIO LOSSES vs CURRENT VaR & ES")

    vs_var = compare_scenarios_to_var(results, var_1d, es_1d, notion)
    neg = vs_var[vs_var["P&L (EUR)"] < 0].sort_values("P&L (EUR)")

    print(f"\n  Current 1d HS VaR: {var_1d:.4%}  (€{var_1d*notion:>,.0f})")
    print(f"  Current 1d HS ES:  {es_1d:.4%}  (€{es_1d*notion:>,.0f})")
    print(f"\n  {'Scenario':<30}  {'Return':>8}  {'× VaR':>7}  {'× ES':>7}  {'Classification'}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*25}")

    for _, row in neg.iterrows():
        xvar = f"{row['× 1d VaR']:>6.1f}×" if pd.notna(row['× 1d VaR']) else "   n/a"
        xes  = f"{row['× 1d ES']:>6.1f}×"  if pd.notna(row['× 1d ES'])  else "   n/a"
        print(
            f"  {row['Scenario']:<30}  {row['Return']:>+7.2%}  "
            f"{xvar}  {xes}  {row['In VaR?']}"
        )

    beyond_var = neg[neg["× 1d VaR"] < -1]
    print(f"\n  {len(beyond_var)} of {len(neg)} loss scenarios exceed 1d VaR.")
    print(f"  This is expected — stress tests cover multi-day and tail events.")
    print(f"  The × VaR column shows severity relative to the daily risk budget.")

    # ════════════════════════════════════════
    # 5. REVERSE STRESS TEST
    # ════════════════════════════════════════
    print_section("REVERSE STRESS TEST")

    target_losses = [0.05, 0.10, 0.15, 0.20]
    modes = [
        ("equity_parallel", "Uniform equity shock"),
        ("all_parallel",    "Uniform all-asset shock"),
    ]

    for mode_key, mode_label in modes:
        print(f"\n  {mode_label}:")
        print(f"  {'Target Loss':>12}  {'Target (EUR)':>14}  {'Required Shock':>16}  {'Plausible?':>12}")
        print(f"  {'─'*12}  {'─'*14}  {'─'*16}  {'─'*12}")

        for tl in target_losses:
            rst = reverse_stress_test(weights, tl, notion, mode_key)
            if mode_key == "equity_parallel":
                shock = rst.get("required_equity_shock", np.nan)
                plaus = "✓ Yes" if rst.get("plausible", False) else "⚠ Extreme"
            else:
                shock = rst.get("required_uniform_shock", np.nan)
                plaus = "✓ Yes" if abs(shock) < 0.30 else "⚠ Extreme"
            print(
                f"  {tl:>11.0%}  "
                f"€{tl*notion:>13,.0f}  "
                f"{shock:>+15.1%}  "
                f"{plaus:>12}"
            )

    # Single-asset: what shock to SPY alone causes 5% portfolio loss?
    print(f"\n  Single-asset shocks to cause 5% portfolio loss:")
    print(f"\n  {'Ticker':<6}  {'Name':<25}  {'Weight':>8}  {'Required Shock':>16}  {'Plausible?'}")
    print(f"  {'─'*6}  {'─'*25}  {'─'*8}  {'─'*16}  {'─'*12}")

    rst_single = reverse_stress_test(weights, 0.05, notion, "single_asset")
    shocks = rst_single["asset_shocks"]
    for ticker, info in sorted(shocks.items(), key=lambda x: abs(x[1]["required_shock"])):
        name  = UNIVERSE[ticker]["name"] if ticker in UNIVERSE else ticker
        shock = info["required_shock"]
        plaus = "✓ Yes" if info["plausible"] else "⚠ Exceeds 100%"
        print(f"  {ticker:<6}  {name:<25}  {info['weight']:>7.1%}  {shock:>+15.1%}  {plaus}")

    # ════════════════════════════════════════
    # 6. SCENARIO RISK SUMMARY
    # ════════════════════════════════════════
    print_section("STRESS TEST RISK SUMMARY")

    loss_results = [r for r in results if r.total_pnl < 0]
    gain_results = [r for r in results if r.total_pnl >= 0]
    worst        = min(results, key=lambda x: x.total_return)
    best         = max(results, key=lambda x: x.total_return)

    print(f"\n  Scenarios tested:        {len(results)}")
    print(f"  Loss scenarios:          {len(loss_results)}")
    print(f"  Gain scenarios:          {len(gain_results)}")
    print(f"\n  Worst scenario:          {worst.name}")
    print(f"    Return:                {worst.total_return:+.2%}")
    print(f"    P&L:                   €{worst.total_pnl:>+,.0f}")
    print(f"    Severity:              {worst.severity}")
    print(f"    Multiples of 1d VaR:   {worst.total_pnl / (-var_1d * notion):.1f}×")
    print(f"\n  Best scenario:           {best.name}")
    print(f"    Return:                {best.total_return:+.2%}")
    print(f"    P&L:                   €{best.total_pnl:>+,.0f}")
    print(f"\n  Severity distribution:")
    from collections import Counter
    sev_counts = Counter(r.severity for r in results)
    for sev, count in sorted(sev_counts.items(), key=lambda x: x[0]):
        print(f"    {sev:<25}  {count} scenario(s)")

    # ── Save outputs ─────────────────────────
    print_section("OUTPUTS SAVED")

    # All scenarios summary
    results_to_dataframe(results).to_csv("outputs/stress_all_scenarios.csv", index=False)

    # Asset-level breakdown per scenario
    breakdown_rows = []
    for res in results:
        for ticker, pnl in res.asset_contributions.items():
            breakdown_rows.append({
                "Scenario": res.name,
                "Category": res.category,
                "Asset":    ticker,
                "Shock":    res.asset_shocks.get(ticker, 0),
                "P&L_EUR":  pnl,
                "Return":   res.total_return,
            })
    pd.DataFrame(breakdown_rows).to_csv("outputs/stress_asset_breakdown.csv", index=False)

    # VaR comparison
    vs_var.to_csv("outputs/stress_vs_var.csv", index=False)

    print(f"\n  ✓ outputs/stress_all_scenarios.csv")
    print(f"  ✓ outputs/stress_asset_breakdown.csv")
    print(f"  ✓ outputs/stress_vs_var.csv")
    print(f"\n  Phase 5 complete. All analytical modules built.")
    print(f"  Next: Phase 6 — Interactive Dashboard (Plotly).")
    print(f"\n{SEPARATOR}\n")

    return {
        "results":   results,
        "vs_var":    vs_var,
        "portfolio": portfolio,
        "var_1d":    var_1d,
        "es_1d":     es_1d,
    }


if __name__ == "__main__":
    run_phase5()
