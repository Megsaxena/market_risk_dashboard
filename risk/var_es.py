"""
risk/var_es.py
==============
Value at Risk and Expected Shortfall — three methods, one module.

Methods implemented:
  1. Historical Simulation (HS)       — non-parametric, reads the empirical tail
  2. Parametric (Variance-Covariance) — assumes normality, closed-form
  3. Monte Carlo (MC)                 — simulated paths, handles non-linearity

For each method we compute:
  - VaR   : loss not exceeded with probability (1 - confidence_level)
  - ES    : mean loss conditional on exceeding VaR (CVaR)
  - VaR term structure: 1d through 10d via sqrt-T scaling
  - Rolling VaR: how risk has evolved over time

Key design choices:
  - All outputs in both % return and EUR notional terms
  - VaR is expressed as a positive number (loss magnitude)
  - ES >= VaR always; if not, something is wrong
  - Rolling window for HS-VaR uses config.RISK_PARAMS.historical_window
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

from config import RISK_PARAMS, PORTFOLIO_NOTIONAL

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────

class VaRResult:
    """
    Holds all outputs for one VaR/ES calculation.
    Keeps results structured and easy to pass into the dashboard.
    """
    def __init__(
        self,
        method: str,
        confidence: float,
        horizon: int,
        var_pct: float,
        es_pct: float,
        notional: float,
    ):
        self.method      = method
        self.confidence  = confidence
        self.horizon     = horizon
        self.var_pct     = var_pct          # e.g. 0.0215  (positive = loss)
        self.es_pct      = es_pct           # always >= var_pct
        self.notional    = notional
        self.var_eur     = var_pct * notional
        self.es_eur      = es_pct  * notional
        self.es_var_ratio = es_pct / var_pct if var_pct > 0 else np.nan

    def __repr__(self):
        return (
            f"VaRResult({self.method} | {self.confidence:.0%} | {self.horizon}d)\n"
            f"  VaR: {self.var_pct:.4%}  (€{self.var_eur:>12,.0f})\n"
            f"  ES:  {self.es_pct:.4%}  (€{self.es_eur:>12,.0f})\n"
            f"  ES/VaR ratio: {self.es_var_ratio:.3f}"
        )


# ─────────────────────────────────────────────
# 1. HISTORICAL SIMULATION VAR
# ─────────────────────────────────────────────

def var_historical(
    portfolio_returns: pd.Series,
    confidence: float     = None,
    window: int           = None,
    horizon: int          = None,
    notional: float       = None,
) -> VaRResult:
    """
    Historical Simulation VaR and ES.

    Takes the most recent `window` daily returns, sorts them,
    and reads off the empirical quantile.

    No distributional assumption — the historical return distribution
    is taken as-is, fat tails and all.

    Horizon scaling: VaR(T) = VaR(1d) × sqrt(T)
    This is the Basel sqrt-T approximation. Strictly valid only under
    i.i.d. returns, but universally used in practice.
    """
    conf     = confidence or RISK_PARAMS.confidence_level
    win      = window     or RISK_PARAMS.historical_window
    hor      = horizon    or RISK_PARAMS.var_horizon_days
    notion   = notional   or PORTFOLIO_NOTIONAL

    r = portfolio_returns.dropna()

    # Use only the most recent `window` observations
    if len(r) > win:
        r = r.iloc[-win:]

    if len(r) < 50:
        raise ValueError(f"Insufficient history: {len(r)} obs (need ≥ 50).")

    # VaR: empirical quantile at the loss tail
    # (1 - conf) quantile of returns = conf-th percentile of losses
    var_1d = float(-np.percentile(r, (1 - conf) * 100))

    # ES: mean of returns below the VaR threshold
    tail   = r[r < -var_1d]
    es_1d  = float(-tail.mean()) if len(tail) > 0 else var_1d

    # Scale to horizon
    var_Td = var_1d * np.sqrt(hor)
    es_Td  = es_1d  * np.sqrt(hor)

    logger.debug(
        f"HS-VaR ({conf:.0%}, {hor}d, window={win}): "
        f"VaR={var_Td:.4%}, ES={es_Td:.4%}"
    )
    return VaRResult("Historical Simulation", conf, hor, var_Td, es_Td, notion)


# ─────────────────────────────────────────────
# 2. PARAMETRIC VAR (VARIANCE-COVARIANCE)
# ─────────────────────────────────────────────

def var_parametric(
    portfolio_returns: pd.Series,
    confidence: float     = None,
    horizon: int          = None,
    notional: float       = None,
    distribution: str     = "normal",   # "normal" or "t"
    t_df: int             = 5,          # degrees of freedom if distribution="t"
) -> VaRResult:
    """
    Parametric (Variance-Covariance) VaR and ES.

    Assumes returns are drawn from a specified distribution.
    Uses sample mean and std of the full return history.

    Normal distribution:
        VaR = -(μ + z_α × σ)   where z_α = norm.ppf(1-conf)
        ES  = -(μ - σ × φ(z_α) / (1-conf))
              φ = normal PDF, evaluated at the VaR quantile

    Student-t distribution (better fat-tail fit):
        VaR = -(μ + t_α × σ × sqrt((df-2)/df))
        ES  = -(μ - σ × t_pdf(t_α) × (df + t_α²) / ((df-1) × (1-conf)))

    The t-distribution variant will give higher VaR than normal — the gap
    is exactly the "parametric underestimation" risk.
    """
    conf   = confidence or RISK_PARAMS.confidence_level
    hor    = horizon    or RISK_PARAMS.var_horizon_days
    notion = notional   or PORTFOLIO_NOTIONAL

    r  = portfolio_returns.dropna()
    mu = r.mean()
    sigma = r.std(ddof=1)

    if distribution == "normal":
        z_alpha = norm.ppf(1 - conf)          # negative (left tail)
        var_1d  = -(mu + z_alpha * sigma)

        # ES for normal: E[X | X < VaR] = μ - σ × φ(z_α) / (1-conf)
        es_1d = -(mu - sigma * norm.pdf(z_alpha) / (1 - conf))

        dist_label = "Normal"

    elif distribution == "t":
        df = t_df
        t_alpha = t_dist.ppf(1 - conf, df=df)

        # t-VaR: scale by sqrt((df-2)/df) to match variance
        var_1d = -(mu + t_alpha * sigma * np.sqrt((df - 2) / df))

        # ES for t-distribution
        t_pdf_at_alpha = t_dist.pdf(t_alpha, df=df)
        es_1d = -(mu - sigma * (t_pdf_at_alpha * (df + t_alpha**2))
                  / ((df - 1) * (1 - conf)))

        dist_label = f"Student-t(df={df})"
    else:
        raise ValueError(f"Unknown distribution '{distribution}'")

    # Scale to horizon
    var_Td = var_1d * np.sqrt(hor)
    es_Td  = es_1d  * np.sqrt(hor)

    logger.debug(
        f"Parametric-VaR ({dist_label}, {conf:.0%}, {hor}d): "
        f"VaR={var_Td:.4%}, ES={es_Td:.4%}"
    )
    return VaRResult(f"Parametric ({dist_label})", conf, hor, var_Td, es_Td, notion)


# ─────────────────────────────────────────────
# 3. MONTE CARLO VAR
# ─────────────────────────────────────────────

def var_monte_carlo(
    portfolio_returns: pd.Series,
    confidence: float     = None,
    horizon: int          = None,
    n_simulations: int    = None,
    notional: float       = None,
    seed: int             = 42,
    use_t_dist: bool      = True,
    t_df: int             = 5,
) -> VaRResult:
    """
    Monte Carlo VaR and ES.

    Fits distribution parameters to the portfolio return history,
    then simulates n_simulations × horizon paths and reads the tail.

    For horizon > 1: simulates full multi-day paths (compounding),
    not just sqrt-T scaling. This is the methodological advantage
    of MC over HS and parametric for longer horizons.

    If use_t_dist=True: fits a t-distribution (better for fat tails).
    If use_t_dist=False: uses normal (matches parametric method).
    """
    conf   = confidence  or RISK_PARAMS.confidence_level
    hor    = horizon     or RISK_PARAMS.var_horizon_days
    n_sim  = n_simulations or RISK_PARAMS.n_simulations
    notion = notional    or PORTFOLIO_NOTIONAL

    rng = np.random.default_rng(seed)
    r   = portfolio_returns.dropna()
    mu  = r.mean()
    sigma = r.std(ddof=1)

    if use_t_dist:
        # Fit t-distribution: MLE via scipy
        df_fit, loc_fit, scale_fit = t_dist.fit(r, floc=mu)
        df_fit = max(df_fit, 2.1)   # enforce finite variance

        # Simulate: for horizon > 1, simulate path-by-path
        if hor == 1:
            sim_rets = t_dist.rvs(
                df=df_fit, loc=loc_fit, scale=scale_fit,
                size=n_sim, random_state=rng
            )
        else:
            # Simulate multi-day paths: sum of T daily returns
            daily_sims = t_dist.rvs(
                df=df_fit, loc=loc_fit, scale=scale_fit,
                size=(n_sim, hor), random_state=rng
            )
            sim_rets = daily_sims.sum(axis=1)   # total T-day return per path

        dist_label = f"t-dist(df={df_fit:.1f})"

    else:
        if hor == 1:
            sim_rets = rng.normal(loc=mu, scale=sigma, size=n_sim)
        else:
            daily_sims = rng.normal(loc=mu, scale=sigma, size=(n_sim, hor))
            sim_rets = daily_sims.sum(axis=1)
        dist_label = "Normal"

    # VaR: empirical quantile of simulated P&L distribution
    var_Td = float(-np.percentile(sim_rets, (1 - conf) * 100))

    # ES: mean of simulated returns in the tail
    tail  = sim_rets[sim_rets < -var_Td]
    es_Td = float(-tail.mean()) if len(tail) > 0 else var_Td

    logger.debug(
        f"MC-VaR ({dist_label}, {conf:.0%}, {hor}d, n={n_sim:,}): "
        f"VaR={var_Td:.4%}, ES={es_Td:.4%}"
    )
    return VaRResult(f"Monte Carlo ({dist_label})", conf, hor, var_Td, es_Td, notion)


# ─────────────────────────────────────────────
# ALL THREE — COMPARISON TABLE
# ─────────────────────────────────────────────

def compute_all_var(
    portfolio_returns: pd.Series,
    confidence: float = None,
    horizon: int      = None,
    notional: float   = None,
) -> dict:
    """
    Runs all three VaR methods and returns a structured dict of results.
    This is the primary entry point for the dashboard.
    """
    conf   = confidence or RISK_PARAMS.confidence_level
    hor    = horizon    or RISK_PARAMS.var_horizon_days
    notion = notional   or PORTFOLIO_NOTIONAL

    results = {
        "hs":           var_historical(portfolio_returns,  conf, horizon=hor, notional=notion),
        "parametric":   var_parametric(portfolio_returns,  conf, horizon=hor, notional=notion, distribution="normal"),
        "parametric_t": var_parametric(portfolio_returns,  conf, horizon=hor, notional=notion, distribution="t"),
        "mc":           var_monte_carlo(portfolio_returns, conf, horizon=hor, notional=notion),
    }
    return results


# ─────────────────────────────────────────────
# VAR TERM STRUCTURE
# ─────────────────────────────────────────────

def compute_var_term_structure(
    portfolio_returns: pd.Series,
    horizons: list    = None,
    confidence: float = None,
    notional: float   = None,
) -> pd.DataFrame:
    """
    Computes VaR and ES across multiple horizons (1d through 10d).

    Returns a DataFrame: rows = horizons, columns = [method × metric].
    Shows how risk scales with time — and where sqrt-T breaks down.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10]

    conf   = confidence or RISK_PARAMS.confidence_level
    notion = notional   or PORTFOLIO_NOTIONAL

    rows = []
    for h in horizons:
        hs  = var_historical(portfolio_returns,  conf, horizon=h, notional=notion)
        par = var_parametric(portfolio_returns,  conf, horizon=h, notional=notion)
        mc  = var_monte_carlo(portfolio_returns, conf, horizon=h, notional=notion)

        rows.append({
            "Horizon (days)":   h,
            "HS VaR %":         hs.var_pct,
            "HS ES %":          hs.es_pct,
            "Param VaR %":      par.var_pct,
            "Param ES %":       par.es_pct,
            "MC VaR %":         mc.var_pct,
            "MC ES %":          mc.es_pct,
            "HS VaR (€)":       hs.var_eur,
            "Param VaR (€)":    par.var_eur,
            "MC VaR (€)":       mc.var_eur,
        })

    return pd.DataFrame(rows).set_index("Horizon (days)")


# ─────────────────────────────────────────────
# ROLLING VAR
# ─────────────────────────────────────────────

def compute_rolling_var(
    portfolio_returns: pd.Series,
    method: str       = "hs",
    confidence: float = None,
    window: int       = None,
    notional: float   = None,
) -> pd.DataFrame:
    """
    Computes rolling 1-day VaR and ES over time.

    Produces a time series showing how tail risk has evolved —
    spikes during crisis periods are clearly visible.

    method: "hs" | "parametric" | "mc"
    window: rolling lookback for HS; for parametric uses expanding then rolling
    """
    conf   = confidence or RISK_PARAMS.confidence_level
    win    = window     or RISK_PARAMS.historical_window
    notion = notional   or PORTFOLIO_NOTIONAL

    r = portfolio_returns.dropna()
    dates = r.index[win:]           # start only when we have a full window

    var_series = []
    es_series  = []

    for i in range(win, len(r)):
        window_rets = r.iloc[i - win : i]

        if method == "hs":
            res = var_historical(window_rets, conf, window=win, notional=notion)
        elif method == "parametric":
            res = var_parametric(window_rets, conf, notional=notion)
        elif method == "mc":
            # MC is expensive in a loop — use smaller n_sim for rolling
            res = var_monte_carlo(
                window_rets, conf, n_simulations=2000, notional=notion, seed=i
            )
        else:
            raise ValueError(f"Unknown method '{method}'")

        var_series.append(res.var_pct)
        es_series.append(res.es_pct)

    rolling = pd.DataFrame({
        "rolling_var": var_series,
        "rolling_es":  es_series,
        "realised_return": r.iloc[win:].values,
    }, index=dates)

    return rolling


# ─────────────────────────────────────────────
# VaR BREACHES (BACKTESTING)
# ─────────────────────────────────────────────

def compute_var_breaches(
    rolling_var: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> pd.DataFrame:
    """
    Identifies days when the realised loss exceeded the VaR estimate.
    These are "VaR breaches" or "exceptions".

    At 95% confidence over N days, we expect 5% × N breaches.
    Too many → model underestimates risk.
    Too few  → model overestimates risk (capital inefficiency).

    Basel traffic light:
      Green  : 0–4 breaches per year
      Yellow : 5–9 breaches per year
      Red    : ≥10 breaches per year
    """
    df = rolling_var.copy()
    df["breach"] = df["realised_return"] < -df["rolling_var"]
    df["breach_magnitude"] = np.where(
        df["breach"],
        df["realised_return"] + df["rolling_var"],   # negative = how far beyond VaR
        0
    )
    return df


def backtest_summary(
    rolling_var: pd.DataFrame,
    confidence: float = None,
) -> dict:
    """
    Summarises VaR backtest quality metrics.
    """
    conf = confidence or RISK_PARAMS.confidence_level
    df   = compute_var_breaches(rolling_var, rolling_var["realised_return"])

    n_total   = len(df)
    n_breaches = df["breach"].sum()
    expected  = (1 - conf) * n_total
    breach_rate = n_breaches / n_total

    # Kupiec POF test: is the observed breach rate consistent with the model?
    # H0: breach probability = (1 - conf)
    # Test stat: LR = -2 * ln(L0/L1) ~ chi2(1)
    p = 1 - conf
    p_hat = breach_rate
    if 0 < p_hat < 1:
        lr_stat = -2 * (
            n_breaches * np.log(p / p_hat) +
            (n_total - n_breaches) * np.log((1 - p) / (1 - p_hat))
        )
    else:
        lr_stat = np.nan

    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lr_stat, df=1) if not np.isnan(lr_stat) else np.nan

    # Per-year breach count (Basel traffic light)
    years = n_total / 252
    annual_breaches = n_breaches / years if years > 0 else 0
    if annual_breaches <= 4:
        traffic_light = "🟢 Green"
    elif annual_breaches <= 9:
        traffic_light = "🟡 Yellow"
    else:
        traffic_light = "🔴 Red"

    return {
        "Total observations":    n_total,
        "VaR breaches":          int(n_breaches),
        "Expected breaches":     f"{expected:.1f}",
        "Observed breach rate":  f"{breach_rate:.2%}",
        "Expected breach rate":  f"{1-conf:.2%}",
        "Annual breach rate":    f"{annual_breaches:.1f}/year",
        "Kupiec LR statistic":   f"{lr_stat:.3f}",
        "Kupiec p-value":        f"{p_value:.4f}" if not np.isnan(p_value) else "n/a",
        "Basel Traffic Light":   traffic_light,
    }
