"""
data/portfolio.py
=================
Portfolio object: holds weights, computes P&L series, and produces
attribution breakdowns. This is the object all risk modules consume.

Key outputs:
  - portfolio_returns     : weighted daily log return series
  - portfolio_pnl         : cumulative P&L in notional currency terms
  - asset_pnl_contrib     : per-asset P&L contribution (attribution)
  - summary stats         : Sharpe, Sortino, Calmar, annualised vol/return
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import PORTFOLIO_WEIGHTS, PORTFOLIO_NOTIONAL, UNIVERSE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PORTFOLIO DATACLASS
# ─────────────────────────────────────────────

@dataclass
class Portfolio:
    """
    Represents a multi-asset portfolio.

    Parameters
    ----------
    weights : dict
        {ticker: weight} — must sum to ~1.0. Negative = short.
    notional : float
        Portfolio value in base currency (EUR).
    name : str
        Display name for charts and reports.
    """
    weights: Dict[str, float]
    notional: float = PORTFOLIO_NOTIONAL
    name: str = "Multi-Asset Portfolio"

    def __post_init__(self):
        self.weights = {k: v for k, v in self.weights.items() if v != 0}
        total = sum(self.weights.values())
        if not (0.98 <= total <= 1.02):
            logger.warning(
                f"Weights sum to {total:.4f} — expected ~1.0. "
                f"Consider normalising."
            )
        self.tickers = list(self.weights.keys())
        self.weight_series = pd.Series(self.weights, name="weight")

    def normalise_weights(self) -> "Portfolio":
        """Returns a new Portfolio with weights normalised to sum to 1."""
        total = sum(self.weights.values())
        new_weights = {k: v / total for k, v in self.weights.items()}
        return Portfolio(new_weights, self.notional, self.name)


# ─────────────────────────────────────────────
# P&L ENGINE
# ─────────────────────────────────────────────

def compute_portfolio_returns(
    asset_returns: pd.DataFrame,
    portfolio: Portfolio,
) -> pd.Series:
    """
    Computes daily portfolio log returns as weighted sum of asset returns.

    Note: Strictly, portfolio log returns ≠ sum of weighted log returns
    (Jensen's inequality). For daily returns this approximation is excellent.
    For multi-period, convert to simple returns first, aggregate, then convert back.

    Returns: pd.Series, name='portfolio_return'
    """
    # Align tickers — only use assets present in both weights and returns
    common = [t for t in portfolio.tickers if t in asset_returns.columns]
    missing = [t for t in portfolio.tickers if t not in asset_returns.columns]
    if missing:
        logger.warning(f"Tickers in portfolio but not in returns data: {missing}")

    weights = np.array([portfolio.weights[t] for t in common])
    rets = asset_returns[common]

    port_returns = rets @ weights
    port_returns.name = "portfolio_return"
    return port_returns


def compute_pnl(
    portfolio_returns: pd.Series,
    notional: float,
    method: str = "cumulative",
) -> pd.Series:
    """
    Converts return series to P&L in currency terms.

    method:
        "cumulative" — compound growth of the notional
        "daily"      — daily P&L in currency (notional × daily return)

    Returns: pd.Series in EUR (or whatever notional currency)
    """
    if method == "cumulative":
        # Compound: NAV_t = Notional × exp(sum of log returns)
        cum_log_ret = portfolio_returns.cumsum()
        nav = notional * np.exp(cum_log_ret)
        pnl = nav - notional
        pnl.name = "cumulative_pnl"
        return pnl
    elif method == "daily":
        daily_pnl = notional * portfolio_returns
        daily_pnl.name = "daily_pnl"
        return daily_pnl
    else:
        raise ValueError(f"Unknown method '{method}'")


def compute_asset_contributions(
    asset_returns: pd.DataFrame,
    portfolio: Portfolio,
) -> pd.DataFrame:
    """
    Computes each asset's daily P&L contribution.

    contribution_i = weight_i × return_i × notional

    Returns: DataFrame with one column per asset, values in EUR
    """
    common = [t for t in portfolio.tickers if t in asset_returns.columns]
    contribs = pd.DataFrame(index=asset_returns.index)

    for ticker in common:
        contribs[ticker] = (
            portfolio.weights[ticker]
            * asset_returns[ticker]
            * portfolio.notional
        )
    return contribs


# ─────────────────────────────────────────────
# PORTFOLIO ANALYTICS
# ─────────────────────────────────────────────

def compute_portfolio_stats(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.03,  # ECB deposit rate ~3% (adjust as needed)
    trading_days: int = 252,
) -> dict:
    """
    Computes a full set of portfolio-level performance statistics.

    Returns a dict of labelled metrics ready for display.
    """
    r = portfolio_returns.dropna()
    rf_daily = risk_free_rate / trading_days

    ann_return = r.mean() * trading_days
    ann_vol    = r.std()  * np.sqrt(trading_days)

    # ── Sharpe ───────────────────────────────
    excess_daily = r - rf_daily
    sharpe = (excess_daily.mean() / r.std()) * np.sqrt(trading_days)

    # ── Sortino ──────────────────────────────
    downside = r[r < 0]
    downside_vol = downside.std() * np.sqrt(trading_days)
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol != 0 else np.nan

    # ── Max Drawdown (for Calmar) ─────────────
    cum = np.exp(r.cumsum())
    rolling_max = cum.cummax()
    dd_series = (cum - rolling_max) / rolling_max
    max_dd = dd_series.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    # ── Skew & Kurtosis ───────────────────────
    skew     = r.skew()
    kurt     = r.kurtosis()   # excess kurtosis (normal = 0)

    # ── Hit Rate ─────────────────────────────
    hit_rate = (r > 0).mean()

    # ── Gain/Loss Ratio ──────────────────────
    avg_gain = r[r > 0].mean() if (r > 0).any() else 0
    avg_loss = r[r < 0].mean() if (r < 0).any() else 0
    gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else np.nan

    return {
        "Annualised Return":    f"{ann_return:.2%}",
        "Annualised Vol":       f"{ann_vol:.2%}",
        "Sharpe Ratio":         f"{sharpe:.2f}",
        "Sortino Ratio":        f"{sortino:.2f}",
        "Calmar Ratio":         f"{calmar:.2f}",
        "Max Drawdown":         f"{max_dd:.2%}",
        "Skewness":             f"{skew:.3f}",
        "Excess Kurtosis":      f"{kurt:.3f}",
        "Hit Rate":             f"{hit_rate:.1%}",
        "Gain/Loss Ratio":      f"{gain_loss_ratio:.2f}",
        "Observations":         len(r),
        "Start Date":           str(r.index[0].date()),
        "End Date":             str(r.index[-1].date()),
    }


def compute_asset_class_weights(portfolio: Portfolio) -> pd.Series:
    """Aggregates weights by asset class (Equity / Bond / Commodity)."""
    ac_weights = {}
    for ticker, weight in portfolio.weights.items():
        if ticker in UNIVERSE:
            ac = UNIVERSE[ticker]["asset_class"]
            ac_weights[ac] = ac_weights.get(ac, 0) + weight
    return pd.Series(ac_weights, name="asset_class_weight").sort_values(ascending=False)


def compute_rolling_correlation_to_benchmark(
    asset_returns: pd.DataFrame,
    portfolio: Portfolio,
    benchmark: str = "SPY",
    window: int = 63,
) -> pd.DataFrame:
    """
    Computes rolling correlation of each asset to a benchmark.
    Useful for tracking diversification over time.
    """
    if benchmark not in asset_returns.columns:
        raise ValueError(f"Benchmark {benchmark} not in returns data.")

    bench = asset_returns[benchmark]
    others = [t for t in portfolio.tickers if t != benchmark and t in asset_returns.columns]

    rolling_corrs = pd.DataFrame(index=asset_returns.index)
    for ticker in others:
        rolling_corrs[ticker] = (
            asset_returns[ticker].rolling(window).corr(bench)
        )
    return rolling_corrs


# ─────────────────────────────────────────────
# FACTORY: build default portfolio
# ─────────────────────────────────────────────

def build_default_portfolio() -> Portfolio:
    return Portfolio(
        weights=PORTFOLIO_WEIGHTS,
        notional=PORTFOLIO_NOTIONAL,
        name="Multi-Asset Portfolio",
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    from data.fetcher import fetch_universe, compute_returns

    print("Building portfolio...")
    portfolio = build_default_portfolio()

    print(f"\nPortfolio: {portfolio.name}")
    print(f"Notional: €{portfolio.notional:,.0f}")
    print(f"\nWeights:\n{portfolio.weight_series.to_string()}")
    print(f"\nAsset Class Allocation:\n{compute_asset_class_weights(portfolio).to_string()}")

    print("\nFetching data...")
    prices  = fetch_universe()
    returns = compute_returns(prices, method="log")

    port_rets = compute_portfolio_returns(returns, portfolio)
    pnl       = compute_pnl(port_rets, portfolio.notional)
    daily_pnl = compute_pnl(port_rets, portfolio.notional, method="daily")
    contribs  = compute_asset_contributions(returns, portfolio)

    print(f"\nPortfolio Returns (last 5 days):\n{port_rets.tail().round(4).to_string()}")
    print(f"\nCumulative P&L (last 5 days, EUR):\n{pnl.tail().round(0).to_string()}")
    print(f"\nRecent Daily P&L (EUR):\n{daily_pnl.tail().round(0).to_string()}")

    print("\n" + "="*50)
    print("PORTFOLIO STATISTICS")
    print("="*50)
    stats = compute_portfolio_stats(port_rets)
    for k, v in stats.items():
        print(f"  {k:<25} {v}")
