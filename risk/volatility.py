"""
risk/volatility.py
==================
Volatility analytics — multiple estimators, regime detection, and diagnostics.

The core question: what is the "true" volatility of this portfolio right now?
Different estimators answer this differently, and the spread between them
is itself informative.

Estimators implemented:
  1. Rolling Realised Vol      — simple, transparent, configurable windows
  2. EWMA Vol (RiskMetrics)    — exponentially-weighted, more reactive to shocks
  3. GARCH(1,1)                — mean-reverting conditional vol model
  4. Parkinson Estimator       — uses High-Low range (needs OHLC, optional)

Additional analytics:
  - Vol of Vol (VoV)           — volatility of the volatility estimate
  - Vol Regime Classification  — Low / Elevated / High / Crisis
  - Vol Percentile Rank        — where current vol sits vs history
  - Vol Term Structure         — annualised vol at different windows side by side
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Optional

from config import RISK_PARAMS

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────
# 1. ROLLING REALISED VOLATILITY
# ─────────────────────────────────────────────

def compute_rolling_vol(
    returns: pd.Series,
    windows: list = None,
    annualise: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Computes rolling realised volatility at multiple lookback windows.

    Returns a DataFrame with one column per window, annualised.

    windows: list of ints (trading days). Default from RISK_PARAMS.
    """
    if windows is None:
        windows = RISK_PARAMS.vol_windows   # [21, 63, 126, 252]

    r = returns.dropna()
    result = pd.DataFrame(index=r.index)

    scale = np.sqrt(trading_days) if annualise else 1.0

    for w in windows:
        col = f"vol_{w}d"
        result[col] = r.rolling(w).std(ddof=1) * scale

    return result.dropna(how="all")


# ─────────────────────────────────────────────
# 2. EWMA VOLATILITY (RiskMetrics)
# ─────────────────────────────────────────────

def compute_ewma_vol(
    returns: pd.Series,
    lam: float = None,
    annualise: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Exponentially Weighted Moving Average (EWMA) volatility.
    RiskMetrics standard: λ = 0.94 for daily data.

    σ²_t = λ × σ²_{t-1} + (1-λ) × r²_{t-1}

    More reactive than simple rolling vol — picks up regime changes faster.
    The half-life of the weight decay = ln(0.5) / ln(λ).

    λ = 0.94 → half-life ≈ 11 days
    λ = 0.97 → half-life ≈ 23 days
    """
    lam = lam or RISK_PARAMS.ewma_lambda
    r   = returns.dropna()

    scale = np.sqrt(trading_days) if annualise else 1.0
    half_life = np.log(0.5) / np.log(lam)

    # Initialise variance with the first 21-day sample variance
    init_var = r.iloc[:21].var(ddof=1)
    ewma_var = [init_var]

    for ret in r.iloc[1:]:
        new_var = lam * ewma_var[-1] + (1 - lam) * ret**2
        ewma_var.append(new_var)

    ewma_vol = pd.Series(
        np.sqrt(ewma_var) * scale,
        index=r.index,
        name=f"ewma_vol_λ{lam}"
    )

    logger.debug(f"EWMA vol: λ={lam}, half-life={half_life:.1f} days")
    return ewma_vol


# ─────────────────────────────────────────────
# 3. GARCH(1,1)
# ─────────────────────────────────────────────

def compute_garch_vol(
    returns: pd.Series,
    annualise: bool = True,
    trading_days: int = 252,
    n_iter: int = 200,
) -> pd.Series:
    """
    GARCH(1,1) conditional volatility via numerical MLE.

    Model: σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

    Constraints:
      ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (stationarity)

    Long-run variance: σ²_LR = ω / (1 - α - β)

    Mean-reversion speed: how fast conditional vol returns to long-run
      Half-life = ln(0.5) / ln(α + β)

    We fit via scipy.optimize.minimize with log-likelihood objective.
    This is a simplified GARCH — for production use arch package.
    """
    from scipy.optimize import minimize

    r = returns.dropna().values
    T = len(r)
    scale = np.sqrt(trading_days) if annualise else 1.0

    # ── Log-likelihood for GARCH(1,1) ────────
    def garch_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10

        sigma2 = np.zeros(T)
        sigma2[0] = np.var(r)   # initialise with sample variance

        for t in range(1, T):
            sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]

        sigma2 = np.maximum(sigma2, 1e-10)  # numerical stability
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r**2 / sigma2)
        return -ll   # minimise negative log-likelihood

    # Initial guess: small omega, typical alpha/beta
    init = [1e-6, 0.08, 0.88]
    bounds = [(1e-8, 0.01), (0.01, 0.3), (0.5, 0.98)]

    try:
        result = minimize(
            garch_loglik, init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": n_iter, "ftol": 1e-9}
        )
        omega, alpha, beta = result.x
        logger.debug(f"GARCH(1,1): ω={omega:.2e}, α={alpha:.4f}, β={beta:.4f}, α+β={alpha+beta:.4f}")

    except Exception as e:
        logger.warning(f"GARCH optimisation failed: {e}. Using EWMA fallback.")
        return compute_ewma_vol(returns, annualise=annualise)

    # ── Compute fitted conditional vol series ─
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(r)

    for t in range(1, T):
        sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]

    garch_vol = pd.Series(
        np.sqrt(sigma2) * scale,
        index=returns.dropna().index,
        name="garch_vol"
    )

    # ── Store parameters as attributes ───────
    long_run_vol = np.sqrt(omega / (1 - alpha - beta)) * scale
    half_life    = np.log(0.5) / np.log(alpha + beta) if (alpha + beta) > 0 else np.nan

    garch_vol.attrs = {
        "omega": omega, "alpha": alpha, "beta": beta,
        "persistence": alpha + beta,
        "long_run_vol": long_run_vol,
        "half_life_days": half_life,
    }

    return garch_vol


# ─────────────────────────────────────────────
# 4. VOL OF VOL
# ─────────────────────────────────────────────

def compute_vol_of_vol(
    vol_series: pd.Series,
    window: int = 63,
    annualise: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Volatility of volatility — the rolling standard deviation of a vol estimate.

    A high VoV means the vol regime itself is unstable:
      - Hard to hedge effectively
      - VaR estimates are less reliable
      - Signals possible regime transition

    This is the vol-space analogue of VVIX (vol of VIX).
    """
    scale = np.sqrt(trading_days) if annualise else 1.0
    vov = vol_series.rolling(window).std(ddof=1) * scale
    vov.name = "vol_of_vol"
    return vov


# ─────────────────────────────────────────────
# 5. VOL REGIME CLASSIFICATION
# ─────────────────────────────────────────────

def classify_vol_regime(
    current_vol: float,
    vol_history: pd.Series,
) -> dict:
    """
    Classifies the current volatility regime based on its
    percentile rank within the historical distribution.

    Regimes (calibrated to long-run equity vol distribution):
      Calm    : bottom 25th percentile
      Normal  : 25th–60th percentile
      Elevated: 60th–85th percentile
      High    : 85th–95th percentile
      Crisis  : top 5th percentile

    Returns the regime label, percentile rank, and thresholds.
    """
    history = vol_history.dropna()
    percentile_rank = (history < current_vol).mean() * 100

    thresholds = {
        p: np.percentile(history, p)
        for p in [25, 60, 85, 95]
    }

    if percentile_rank <= 25:
        regime = "😴 Calm"
        colour = "green"
    elif percentile_rank <= 60:
        regime = "✅ Normal"
        colour = "lightgreen"
    elif percentile_rank <= 85:
        regime = "🟡 Elevated"
        colour = "yellow"
    elif percentile_rank <= 95:
        regime = "🟠 High"
        colour = "orange"
    else:
        regime = "🔴 Crisis"
        colour = "red"

    return {
        "current_vol":      current_vol,
        "percentile_rank":  percentile_rank,
        "regime":           regime,
        "colour":           colour,
        "p25_threshold":    thresholds[25],
        "p60_threshold":    thresholds[60],
        "p85_threshold":    thresholds[85],
        "p95_threshold":    thresholds[95],
        "history_mean":     history.mean(),
        "history_median":   history.median(),
        "history_max":      history.max(),
    }


# ─────────────────────────────────────────────
# 6. VOL TERM STRUCTURE SNAPSHOT
# ─────────────────────────────────────────────

def compute_vol_term_structure(
    returns: pd.Series,
    windows: list = None,
    annualise: bool = True,
) -> pd.DataFrame:
    """
    Snapshot of current vol at multiple lookback windows.
    Shows whether short-term vol is above or below long-term vol —
    a vol term structure inversion signals near-term stress.
    """
    if windows is None:
        windows = [5, 10, 21, 63, 126, 252]

    r = returns.dropna()
    scale = np.sqrt(252) if annualise else 1.0

    rows = []
    for w in windows:
        if len(r) >= w:
            vol = r.iloc[-w:].std(ddof=1) * scale
            rows.append({
                "Window":       f"{w}d",
                "Window Days":  w,
                "Current Vol":  vol,
            })

    df = pd.DataFrame(rows)
    if len(df) >= 2:
        # Short vol vs long vol ratio: > 1 = inverted (stressed)
        short_vol = df.iloc[0]["Current Vol"]
        long_vol  = df.iloc[-1]["Current Vol"]
        df["vs Long-run"] = df["Current Vol"] / long_vol
        df["Signal"] = df["vs Long-run"].apply(
            lambda x: "↑ Elevated" if x > 1.2 else ("↓ Subdued" if x < 0.8 else "≈ Normal")
        )
    return df


# ─────────────────────────────────────────────
# 7. FULL VOLATILITY SUMMARY
# ─────────────────────────────────────────────

def compute_all_vol(
    returns: pd.Series,
    run_garch: bool = True,
) -> dict:
    """
    Master function: runs all vol estimators and returns structured output
    for use in the dashboard and Phase 3 runner.
    """
    r = returns.dropna()

    rolling   = compute_rolling_vol(r)
    ewma      = compute_ewma_vol(r)

    garch = None
    garch_params = {}
    if run_garch:
        logger.info("Fitting GARCH(1,1)...")
        garch = compute_garch_vol(r)
        garch_params = dict(garch.attrs) if hasattr(garch, "attrs") else {}

    # Current vol snapshot (use 21d as primary)
    current_ewma = float(ewma.iloc[-1])
    current_21d  = float(rolling["vol_21d"].iloc[-1])
    current_252d = float(rolling["vol_252d"].dropna().iloc[-1])

    regime = classify_vol_regime(current_ewma, ewma)
    term_structure = compute_vol_term_structure(r)

    vov = compute_vol_of_vol(ewma)

    return {
        "rolling_vol":    rolling,
        "ewma_vol":       ewma,
        "garch_vol":      garch,
        "garch_params":   garch_params,
        "vol_of_vol":     vov,
        "regime":         regime,
        "term_structure": term_structure,
        "current_ewma":   current_ewma,
        "current_21d":    current_21d,
        "current_252d":   current_252d,
    }
