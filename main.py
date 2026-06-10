"""
main.py
=======
Phase 1 entry point — validates data pipeline and portfolio layer.

Run with: python main.py

Produces:
  - Console output: universe summary, portfolio stats, data quality checks
  - outputs/phase1_summary.csv: P&L and return series for inspection
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# Make sure imports resolve from project root
sys.path.insert(0, os.path.dirname(__file__))

from config import UNIVERSE, RISK_PARAMS
from data.fetcher import (
    fetch_universe,
    compute_returns,
    get_asset_classes,
    get_display_names,
)
from data.portfolio import (
    build_default_portfolio,
    compute_portfolio_returns,
    compute_pnl,
    compute_asset_contributions,
    compute_portfolio_stats,
    compute_asset_class_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEPARATOR = "═" * 60


def print_section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def run_phase1():
    os.makedirs("outputs", exist_ok=True)

    # ────────────────────────────────────────────
    # 1. BUILD PORTFOLIO
    # ────────────────────────────────────────────
    print_section("PORTFOLIO DEFINITION")
    portfolio = build_default_portfolio()
    names = get_display_names()

    print(f"\n  Portfolio: {portfolio.name}")
    print(f"  Notional:  €{portfolio.notional:>12,.0f}")
    print(f"\n  {'Ticker':<6}  {'Name':<25}  {'Weight':>8}  {'EUR Value':>12}")
    print(f"  {'-'*6}  {'-'*25}  {'-'*8}  {'-'*12}")
    for ticker, w in sorted(portfolio.weights.items(), key=lambda x: -x[1]):
        eur_val = w * portfolio.notional
        name = names.get(ticker, ticker)
        print(f"  {ticker:<6}  {name:<25}  {w:>7.1%}  {eur_val:>12,.0f}")

    print(f"\n  Asset Class Breakdown:")
    for ac, w in compute_asset_class_weights(portfolio).items():
        bar = "█" * int(w * 40)
        print(f"    {ac:<12} {w:>6.1%}  {bar}")

    # ────────────────────────────────────────────
    # 2. FETCH PRICES
    # ────────────────────────────────────────────
    print_section("DATA FETCH")
    prices = fetch_universe(tickers=portfolio.tickers)

    print(f"\n  Date range:  {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Trading days: {len(prices)}")
    print(f"  Assets:       {len(prices.columns)}")
    print(f"\n  Latest prices (USD):")
    latest = prices.iloc[-1].sort_values(ascending=False)
    for ticker, price in latest.items():
        name = names.get(ticker, ticker)
        print(f"    {ticker:<6}  {name:<25}  ${price:>8.2f}")

    # ────────────────────────────────────────────
    # 3. DATA QUALITY CHECK
    # ────────────────────────────────────────────
    print_section("DATA QUALITY")
    print(f"\n  {'Ticker':<6}  {'Missing':>8}  {'Start Date':>12}  {'YTD Return':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*10}")
    ytd_start = prices[prices.index.year == prices.index[-1].year].index[0]
    for ticker in prices.columns:
        missing = prices[ticker].isnull().sum()
        start   = prices[ticker].first_valid_index().date()
        ytd_ret = (prices[ticker].iloc[-1] / prices.loc[ytd_start, ticker] - 1)
        flag    = " ⚠" if missing > 5 else ""
        print(f"  {ticker:<6}  {missing:>8}  {str(start):>12}  {ytd_ret:>+9.2%}{flag}")

    # ────────────────────────────────────────────
    # 4. RETURNS
    # ────────────────────────────────────────────
    print_section("RETURN COMPUTATION")
    log_rets    = compute_returns(prices, method="log")
    simple_rets = compute_returns(prices, method="simple")

    print(f"\n  Annualised Volatility by Asset:")
    print(f"\n  {'Ticker':<6}  {'Name':<25}  {'Ann. Vol':>9}  {'Ann. Return':>12}")
    print(f"  {'-'*6}  {'-'*25}  {'-'*9}  {'-'*12}")
    ann_vol = log_rets.std() * np.sqrt(252)
    ann_ret = log_rets.mean() * 252
    asset_classes = get_asset_classes()
    # Sort by asset class then vol
    ac_order = {"Equity": 0, "Bond": 1, "Commodity": 2}
    tickers_sorted = sorted(
        log_rets.columns,
        key=lambda t: (ac_order.get(asset_classes.get(t, ""), 9), -ann_vol[t])
    )
    for ticker in tickers_sorted:
        name = names.get(ticker, ticker)
        ac   = asset_classes.get(ticker, "")
        print(f"  {ticker:<6}  {name:<25}  {ann_vol[ticker]:>8.2%}  {ann_ret[ticker]:>+11.2%}")

    # ────────────────────────────────────────────
    # 5. PORTFOLIO P&L
    # ────────────────────────────────────────────
    print_section("PORTFOLIO P&L")
    port_rets  = compute_portfolio_returns(log_rets, portfolio)
    cum_pnl    = compute_pnl(port_rets, portfolio.notional, method="cumulative")
    daily_pnl  = compute_pnl(port_rets, portfolio.notional, method="daily")
    contribs   = compute_asset_contributions(log_rets, portfolio)

    total_return = np.exp(port_rets.sum()) - 1
    current_nav  = portfolio.notional + cum_pnl.iloc[-1]
    print(f"\n  Total return (full history):  {total_return:+.2%}")
    print(f"  Current NAV:                  €{current_nav:>12,.0f}")
    print(f"  Cumulative P&L:               €{cum_pnl.iloc[-1]:>+12,.0f}")

    print(f"\n  Last 5 daily P&L:")
    for dt, pnl_val in daily_pnl.tail().items():
        sign = "▲" if pnl_val > 0 else "▼"
        print(f"    {dt.date()}  {sign} €{pnl_val:>+12,.0f}")

    # ────────────────────────────────────────────
    # 6. PORTFOLIO STATISTICS
    # ────────────────────────────────────────────
    print_section("PORTFOLIO STATISTICS (FULL HISTORY)")
    stats = compute_portfolio_stats(port_rets)
    for k, v in stats.items():
        print(f"    {k:<25}  {v}")

    # ────────────────────────────────────────────
    # 7. CONTRIBUTION ANALYSIS
    # ────────────────────────────────────────────
    print_section("P&L ATTRIBUTION (FULL HISTORY)")
    total_contrib = contribs.sum().sort_values(ascending=False)
    total_port_pnl = daily_pnl.sum()
    print(f"\n  {'Ticker':<6}  {'Name':<25}  {'Total P&L':>12}  {'Share':>7}")
    print(f"  {'-'*6}  {'-'*25}  {'-'*12}  {'-'*7}")
    for ticker, pnl_val in total_contrib.items():
        name  = names.get(ticker, ticker)
        share = pnl_val / total_port_pnl if total_port_pnl != 0 else 0
        bar   = "+" if pnl_val > 0 else "-"
        print(f"  {ticker:<6}  {name:<25}  €{pnl_val:>+11,.0f}  {share:>+6.1%}")

    # ────────────────────────────────────────────
    # 8. SAVE OUTPUTS
    # ────────────────────────────────────────────
    print_section("SAVING OUTPUTS")
    summary = pd.DataFrame({
        "portfolio_log_return": port_rets,
        "daily_pnl_eur": daily_pnl,
        "cumulative_pnl_eur": cum_pnl,
    })
    summary = pd.concat([summary, log_rets.add_suffix("_return")], axis=1)
    summary.to_csv("outputs/phase1_summary.csv")
    prices.to_csv("outputs/prices.csv")
    contribs.to_csv("outputs/asset_contributions.csv")

    print(f"\n  ✓ outputs/phase1_summary.csv")
    print(f"  ✓ outputs/prices.csv")
    print(f"  ✓ outputs/asset_contributions.csv")
    print(f"\n  Phase 1 complete. Ready for Phase 2 (VaR + ES).")
    print(f"\n{SEPARATOR}\n")

    return {
        "prices": prices,
        "log_returns": log_rets,
        "simple_returns": simple_rets,
        "portfolio": portfolio,
        "portfolio_returns": port_rets,
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cum_pnl,
        "contributions": contribs,
    }


if __name__ == "__main__":
    data = run_phase1()
