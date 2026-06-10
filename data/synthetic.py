"""
data/synthetic.py
=================
Generates realistic synthetic price history for the multi-asset universe.

Why synthetic data is valuable:
  1. Offline / demo use when Bloomberg/yfinance is unavailable
  2. Stress testing with fully controlled scenarios
  3. Unit testing risk modules against known inputs

Design:
  - Uses a multivariate t-distribution (df=5) for fat-tailed returns
  - Correlation structure calibrated to approximate historical 2015-2024 values
  - Volatility targets match long-run realised vols per asset class
  - Includes a "crisis regime" (approx. March 2020) with vol spike + correlation spike
  - Includes a "rate shock regime" (approx. 2022) affecting bonds negatively
  - GBM price path reconstruction from generated returns
  - Seeded for reproducibility
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from datetime import date
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CALIBRATION PARAMETERS
# ─────────────────────────────────────────────

# Annual parameters — approximate 2015-2024 historical values
ANNUAL_PARAMS = {
    #         mu      sigma   (all in annual terms)
    "SPY":  ( 0.115,  0.175),
    "QQQ":  ( 0.155,  0.215),
    "EFA":  ( 0.060,  0.155),
    "EEM":  ( 0.045,  0.185),
    "TLT":  ( 0.020,  0.140),
    "IEF":  ( 0.015,  0.070),
    "HYG":  ( 0.055,  0.085),
    "EMB":  ( 0.045,  0.090),
    "GLD":  ( 0.075,  0.140),
    "SLV":  ( 0.050,  0.280),
    "USO":  (-0.005,  0.380),
    "DBC":  ( 0.025,  0.190),
}

TICKERS = list(ANNUAL_PARAMS.keys())
N = len(TICKERS)

# ─────────────────────────────────────────────
# CORRELATION MATRIX
# Approximate long-run realised correlation structure.
# Layout order matches TICKERS list above.
# ─────────────────────────────────────────────
#           SPY   QQQ   EFA   EEM   TLT   IEF   HYG   EMB   GLD   SLV   USO   DBC
CORR_MATRIX = np.array([
    # SPY
    [ 1.00, 0.92, 0.82, 0.72,-0.30,-0.22, 0.68, 0.62, 0.04, 0.10, 0.30, 0.38],
    # QQQ
    [ 0.92, 1.00, 0.75, 0.65,-0.28,-0.20, 0.60, 0.55, 0.02, 0.08, 0.25, 0.33],
    # EFA
    [ 0.82, 0.75, 1.00, 0.80,-0.25,-0.18, 0.65, 0.70, 0.08, 0.12, 0.35, 0.42],
    # EEM
    [ 0.72, 0.65, 0.80, 1.00,-0.18,-0.12, 0.60, 0.72, 0.10, 0.15, 0.40, 0.48],
    # TLT
    [-0.30,-0.28,-0.25,-0.18, 1.00, 0.92,-0.20,-0.10, 0.18, 0.05,-0.12,-0.05],
    # IEF
    [-0.22,-0.20,-0.18,-0.12, 0.92, 1.00,-0.15,-0.08, 0.15, 0.03,-0.10,-0.03],
    # HYG
    [ 0.68, 0.60, 0.65, 0.60,-0.20,-0.15, 1.00, 0.75, 0.08, 0.12, 0.35, 0.40],
    # EMB
    [ 0.62, 0.55, 0.70, 0.72,-0.10,-0.08, 0.75, 1.00, 0.10, 0.15, 0.38, 0.42],
    # GLD
    [ 0.04, 0.02, 0.08, 0.10, 0.18, 0.15, 0.08, 0.10, 1.00, 0.72, 0.18, 0.30],
    # SLV
    [ 0.10, 0.08, 0.12, 0.15, 0.05, 0.03, 0.12, 0.15, 0.72, 1.00, 0.22, 0.35],
    # USO
    [ 0.30, 0.25, 0.35, 0.40,-0.12,-0.10, 0.35, 0.38, 0.18, 0.22, 1.00, 0.82],
    # DBC
    [ 0.38, 0.33, 0.42, 0.48,-0.05,-0.03, 0.40, 0.42, 0.30, 0.35, 0.82, 1.00],
])

# Verify symmetry and positive definiteness (important for Cholesky)
assert np.allclose(CORR_MATRIX, CORR_MATRIX.T), "Correlation matrix must be symmetric"


def _make_positive_definite(C: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Nudges a correlation matrix to be positive definite via eigenvalue floor.
    Handles floating-point imprecision in hand-crafted matrices.
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, epsilon)
    C_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalise diagonal to 1
    diag_sqrt = np.sqrt(np.diag(C_pd))
    C_pd = C_pd / np.outer(diag_sqrt, diag_sqrt)
    return C_pd


# ─────────────────────────────────────────────
# REGIME DEFINITIONS
# Dates where we inject stress/regime shifts
# ─────────────────────────────────────────────
REGIMES = {
    "covid_crash": {
        "start": "2020-02-20",
        "end":   "2020-04-30",
        "vol_multiplier": 3.5,
        "corr_boost": 0.35,          # all cross-correlations pushed higher
        "equity_drift_shock": -0.15, # annualised extra negative drift
        "bond_drift_shock":   +0.08, # flight to quality
    },
    "rate_shock_2022": {
        "start": "2022-01-01",
        "end":   "2022-10-31",
        "vol_multiplier": 1.8,
        "corr_boost": 0.10,
        "equity_drift_shock": -0.18,
        "bond_drift_shock":   -0.25,  # bonds sell off with equities
    },
}


# ─────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────

def generate_synthetic_prices(
    start_date: str = "2015-01-01",
    end_date: str   = None,
    trading_days: int = None,
    seed: int = 42,
    t_df: int = 5,             # degrees of freedom for t-distribution (fat tails)
    starting_price: float = 100.0,
) -> pd.DataFrame:
    """
    Generates synthetic daily price paths for the full universe.

    Parameters
    ----------
    start_date : str
        First date in the price series.
    end_date : str or None
        Last date. If None, uses today's date or trading_days.
    trading_days : int or None
        Overrides end_date if provided.
    seed : int
        Random seed for reproducibility.
    t_df : int
        Degrees of freedom for multivariate t (lower = fatter tails).
    starting_price : float
        Initial price for all assets (normalised to 100 for comparability).

    Returns
    -------
    pd.DataFrame : Adj-close equivalent prices, indexed by date.
    """
    rng = np.random.default_rng(seed)

    # Build date index
    start = pd.Timestamp(start_date)
    if end_date is not None:
        end = pd.Timestamp(end_date)
        biz_days = pd.bdate_range(start=start, end=end)
    elif trading_days is not None:
        biz_days = pd.bdate_range(start=start, periods=trading_days + 1)
    else:
        end = pd.Timestamp("today").normalize()
        biz_days = pd.bdate_range(start=start, end=end)

    T = len(biz_days) - 1  # number of return observations
    logger.info(f"Generating synthetic data: {T} trading days, {N} assets")

    # ── Daily drift and vol ───────────────────
    daily_mu    = np.array([p[0] / 252 for p in ANNUAL_PARAMS.values()])
    daily_sigma = np.array([p[1] / np.sqrt(252) for p in ANNUAL_PARAMS.values()])

    # ── Correlation Cholesky ──────────────────
    C_pd = _make_positive_definite(CORR_MATRIX)
    L    = np.linalg.cholesky(C_pd)          # lower triangular Cholesky factor

    # ── Generate base returns ──────────────────
    # Multivariate t via Normal-variance mixture:
    # Z ~ N(0,I), V ~ chi2(df)/df; X = Z / sqrt(V) → t_df marginals, correlated via L
    Z = rng.standard_normal((T, N))          # independent standard normals
    V = rng.chisquare(df=t_df, size=T) / t_df
    Z_t = Z / np.sqrt(V[:, None])            # t-distributed marginals
    Z_corr = Z_t @ L.T                       # apply correlation structure

    # Scale to calibrated vol: r = mu + sigma * z
    returns = daily_mu + daily_sigma * Z_corr  # shape (T, N)

    # ── Inject regimes ───────────────────────
    returns = _inject_regimes(returns, biz_days[1:], rng, t_df)

    # ── Build price paths via GBM ─────────────
    # P_t = P_0 * exp(sum of log returns)
    prices = starting_price * np.exp(np.cumsum(returns, axis=0))
    prices = np.vstack([np.full((1, N), starting_price), prices])  # prepend P_0

    df = pd.DataFrame(prices, index=biz_days, columns=TICKERS)
    logger.info(
        f"Synthetic prices generated: "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    return df


def _inject_regimes(
    returns: np.ndarray,
    date_index: pd.DatetimeIndex,
    rng: np.random.Generator,
    t_df: int,
) -> np.ndarray:
    """
    Modifies return array in place to inject crisis regime behaviour.
    Each regime period gets elevated vol, correlation boost, and drift shocks.
    """
    equity_idx    = [TICKERS.index(t) for t in ["SPY","QQQ","EFA","EEM"]]
    bond_idx      = [TICKERS.index(t) for t in ["TLT","IEF","HYG","EMB"]]
    commodity_idx = [TICKERS.index(t) for t in ["GLD","SLV","USO","DBC"]]

    daily_sigma = np.array([p[1] / np.sqrt(252) for p in ANNUAL_PARAMS.values()])

    for regime_name, regime in REGIMES.items():
        mask = (
            (date_index >= pd.Timestamp(regime["start"])) &
            (date_index <= pd.Timestamp(regime["end"]))
        )
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        logger.debug(f"Injecting regime '{regime_name}': {len(idx)} days")

        # Boosted correlation matrix
        C_regime = CORR_MATRIX.copy()
        boost = regime["corr_boost"]
        C_regime = np.clip(C_regime + boost * (1 - np.eye(N)), -1, 1)
        np.fill_diagonal(C_regime, 1.0)
        C_regime_pd = _make_positive_definite(C_regime)
        L_regime = np.linalg.cholesky(C_regime_pd)

        # Regenerate returns for this regime
        n_days = len(idx)
        Z   = rng.standard_normal((n_days, N))
        V   = rng.chisquare(df=t_df, size=n_days) / t_df
        Z_t = Z / np.sqrt(V[:, None])
        Z_r = Z_t @ L_regime.T

        vol_mult = regime["vol_multiplier"]
        r_regime = daily_sigma * vol_mult * Z_r

        # Apply drift shocks
        eq_shock   = regime.get("equity_drift_shock", 0) / 252
        bond_shock = regime.get("bond_drift_shock",   0) / 252

        r_regime[:, equity_idx]    += eq_shock
        r_regime[:, bond_idx]      += bond_shock

        returns[idx] = r_regime

    return returns


# ─────────────────────────────────────────────
# FETCH WRAPPER (drop-in replacement for fetcher.fetch_universe)
# ─────────────────────────────────────────────

def fetch_universe_synthetic(
    tickers: list = None,
    start: str = "2015-01-01",
    **kwargs,
) -> pd.DataFrame:
    """
    Drop-in replacement for fetcher.fetch_universe() using synthetic data.
    Returns a DataFrame with the same shape and structure.
    """
    prices = generate_synthetic_prices(start_date=start)
    if tickers is not None:
        available = [t for t in tickers if t in prices.columns]
        prices = prices[available]
    return prices


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices = generate_synthetic_prices()
    print(f"\nSynthetic price DataFrame: {prices.shape}")
    print(prices.tail(3).round(2).to_string())

    # Quick sanity check on realised vols
    log_rets = np.log(prices / prices.shift(1)).dropna()
    ann_vol = log_rets.std() * np.sqrt(252)
    print(f"\nRealised vs Target Annual Vol:")
    for t in TICKERS:
        target = ANNUAL_PARAMS[t][1]
        realised = ann_vol[t]
        diff = realised - target
        print(f"  {t:<5}  target={target:.1%}  realised={realised:.1%}  diff={diff:+.1%}")
