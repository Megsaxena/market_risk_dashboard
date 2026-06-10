"""
risk/drawdown.py
================
Drawdown analytics — the risk dimension VaR completely ignores.

VaR answers: "how bad can one day get?"
Drawdown answers: "how deep can a losing streak go, and how long does recovery take?"

These are fundamentally different questions. A portfolio with low daily VaR
can still have catastrophic drawdowns if losses are serially correlated.

Outputs:
  - Full drawdown time series (the "underwater" curve)
  - Maximum drawdown (MDD): peak-to-trough over full history
  - Current drawdown: distance from the most recent peak
  - Drawdown table: top N episodes with depth, duration, recovery
  - Calmar, Sterling, Burke ratios (drawdown-adjusted return metrics)
  - Pain Index and Ulcer Index (continuous drawdown measures)
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CORE DRAWDOWN SERIES
# ─────────────────────────────────────────────

def compute_drawdown_series(
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """
    Computes the full drawdown time series from a return series.

    Returns a DataFrame with:
      nav          : normalised NAV (starts at 1.0)
      peak         : rolling maximum NAV achieved so far
      drawdown     : (NAV - peak) / peak  — always ≤ 0
      drawdown_pct : same, as percentage string label
      underwater   : True if currently in drawdown
    """
    r = portfolio_returns.dropna()

    # Compound NAV from log returns
    nav  = np.exp(r.cumsum())
    nav.name = "nav"

    # Running peak
    peak = nav.cummax()
    peak.name = "peak"

    # Drawdown: negative number, 0 = at peak
    dd = (nav - peak) / peak
    dd.name = "drawdown"

    df = pd.DataFrame({"nav": nav, "peak": peak, "drawdown": dd})
    df["underwater"] = df["drawdown"] < 0

    return df


# ─────────────────────────────────────────────
# DRAWDOWN EPISODES TABLE
# ─────────────────────────────────────────────

def compute_drawdown_episodes(
    dd_series: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Identifies discrete drawdown episodes and returns a summary table.

    Each episode is defined as:
      - Start: first day below the prior peak
      - Trough: day of maximum loss within the episode
      - Recovery: first day back at or above the prior peak
      - Duration: start → recovery (or start → today if not yet recovered)

    Returns a DataFrame sorted by drawdown depth (worst first).
    """
    dd = dd_series["drawdown"].copy()
    nav = dd_series["nav"].copy()

    episodes = []
    in_drawdown = False
    start_idx   = None
    trough_idx  = None
    trough_val  = 0.0

    for i, (date, val) in enumerate(dd.items()):
        if not in_drawdown and val < 0:
            # Starting a new drawdown episode
            in_drawdown = True
            start_idx   = date
            trough_idx  = date
            trough_val  = val

        elif in_drawdown:
            if val < trough_val:
                # Deeper trough
                trough_val = val
                trough_idx = date

            if val >= 0:
                # Recovery
                recovery_idx = date
                duration_days = (recovery_idx - start_idx).days
                recovery_days = (recovery_idx - trough_idx).days

                episodes.append({
                    "Start":          start_idx,
                    "Trough":         trough_idx,
                    "Recovery":       recovery_idx,
                    "Depth":          trough_val,
                    "Duration (days)": duration_days,
                    "Drawdown (days)": (trough_idx - start_idx).days,
                    "Recovery (days)": recovery_days,
                    "Recovered":      True,
                })
                in_drawdown = False
                trough_val  = 0.0

    # Handle open drawdown (not yet recovered)
    if in_drawdown:
        last_date = dd.index[-1]
        episodes.append({
            "Start":           start_idx,
            "Trough":          trough_idx,
            "Recovery":        None,
            "Depth":           trough_val,
            "Duration (days)": (last_date - start_idx).days,
            "Drawdown (days)": (trough_idx - start_idx).days,
            "Recovery (days)": None,
            "Recovered":       False,
        })

    if not episodes:
        return pd.DataFrame()

    df = pd.DataFrame(episodes)
    df = df.sort_values("Depth").head(top_n)   # worst first
    df = df.reset_index(drop=True)
    df.index += 1   # 1-indexed ranking
    return df


# ─────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────

def compute_drawdown_metrics(
    portfolio_returns: pd.Series,
    dd_series: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> dict:
    """
    Computes a full set of drawdown-adjusted performance metrics.

    Includes:
      - Max Drawdown (MDD)
      - Current Drawdown
      - Average Drawdown (mean of all negative values)
      - Calmar Ratio  = Ann Return / |MDD|
      - Sterling Ratio = Ann Return / |Avg of top 3 drawdowns|
      - Ulcer Index   = sqrt(mean of squared drawdowns) — penalises prolonged dd
      - Pain Index    = mean of |drawdowns| — simpler continuous measure
      - Time in Drawdown (% of days spent below peak)
    """
    r = portfolio_returns.dropna()
    if dd_series is None:
        dd_series = compute_drawdown_series(r)

    dd = dd_series["drawdown"]
    ann_return = r.mean() * trading_days

    # ── Core drawdown stats ───────────────────
    mdd         = dd.min()                       # most negative value
    current_dd  = dd.iloc[-1]
    avg_dd      = dd[dd < 0].mean() if (dd < 0).any() else 0.0
    time_in_dd  = (dd < 0).mean()               # fraction of time underwater

    # ── Calmar ───────────────────────────────
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan

    # ── Sterling ─────────────────────────────
    # Uses average of the 3 worst drawdowns (more robust than MDD alone)
    episodes = compute_drawdown_episodes(dd_series, top_n=3)
    if not episodes.empty:
        avg_top3_dd = episodes["Depth"].mean()
        sterling = ann_return / abs(avg_top3_dd) if avg_top3_dd != 0 else np.nan
    else:
        sterling = np.nan

    # ── Ulcer Index ──────────────────────────
    # Penalises both depth AND duration — preferred by some practitioners
    # UI = sqrt(sum(dd²) / N)
    ulcer_index = np.sqrt((dd**2).mean())

    # ── Pain Index ───────────────────────────
    pain_index = abs(dd[dd < 0]).mean() if (dd < 0).any() else 0.0

    # ── Recovery stats ───────────────────────
    episodes_all = compute_drawdown_episodes(dd_series, top_n=100)
    recovered = episodes_all[episodes_all["Recovered"] == True]
    if not recovered.empty:
        avg_recovery_days = recovered["Recovery (days)"].dropna().mean()
        max_recovery_days = recovered["Recovery (days)"].dropna().max()
    else:
        avg_recovery_days = np.nan
        max_recovery_days = np.nan

    return {
        "Max Drawdown":              mdd,
        "Current Drawdown":          current_dd,
        "Average Drawdown":          avg_dd,
        "Time in Drawdown":          time_in_dd,
        "Calmar Ratio":              calmar,
        "Sterling Ratio":            sterling,
        "Ulcer Index":               ulcer_index,
        "Pain Index":                pain_index,
        "Avg Recovery (days)":       avg_recovery_days,
        "Max Recovery (days)":       max_recovery_days,
        "Annualised Return":         ann_return,
    }


# ─────────────────────────────────────────────
# ROLLING MAX DRAWDOWN
# ─────────────────────────────────────────────

def compute_rolling_max_drawdown(
    portfolio_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Computes rolling maximum drawdown over a sliding window.

    Useful for detecting whether drawdown risk is increasing or decreasing
    over time — a rising rolling MDD is an early warning signal.
    """
    r = portfolio_returns.dropna()
    rolling_mdd = pd.Series(index=r.index[window:], dtype=float, name="rolling_mdd")

    for i in range(window, len(r)):
        window_rets = r.iloc[i - window : i]
        nav  = np.exp(window_rets.cumsum())
        peak = nav.cummax()
        dd   = (nav - peak) / peak
        rolling_mdd.iloc[i - window] = dd.min()

    return rolling_mdd
