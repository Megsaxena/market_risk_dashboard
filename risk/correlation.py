"""
risk/correlation.py
===================
Correlation risk analytics.

The central insight this module is built around:
  Diversification is not a free lunch — it is a regime-dependent phenomenon.
  In calm markets, assets have low cross-correlations and the portfolio
  benefits from diversification. In crises, correlations spike toward 1,
  all assets fall together, and diversification evaporates exactly when
  it is most needed.

This module measures:
  1. Rolling Correlation Matrix       — how pairwise correlations evolve
  2. Average Pairwise Correlation     — single time series, regime signal
  3. Eigenvalue Decomposition (PCA)   — how many independent risk factors exist
  4. Diversification Ratio            — how much benefit diversification provides
  5. Effective N                      — equivalent number of uncorrelated assets
  6. Conditional Correlations         — correlations conditional on bad days
  7. Asset-Class Block Correlations   — Equity-Bond, Equity-Commodity, etc.
  8. Correlation Regime Detection     — normal vs crisis correlation environment
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List

from config import UNIVERSE, RISK_PARAMS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. ROLLING CORRELATION MATRIX
# ─────────────────────────────────────────────

def compute_rolling_correlation(
    returns: pd.DataFrame,
    window: int = None,
) -> dict:
    """
    Computes rolling pairwise correlation matrices over time.

    Returns a dict: {date: pd.DataFrame (correlation matrix)}
    Stored at monthly frequency (every 21 trading days) to keep
    memory manageable for 12 assets × 2983 days.

    Also returns:
      - current_corr  : most recent full-window correlation matrix
      - avg_corr_ts   : time series of average pairwise correlation
    """
    win = window or RISK_PARAMS.corr_window
    r   = returns.dropna()

    tickers     = list(r.columns)
    n           = len(tickers)
    dates       = r.index[win:]

    # ── Average pairwise correlation time series ──
    avg_corr_ts = pd.Series(index=dates, dtype=float, name="avg_pairwise_corr")

    # Store matrices monthly (every 21 days) for memory efficiency
    snapshot_matrices = {}
    snapshot_step = 21

    for i, date in enumerate(dates):
        window_rets = r.iloc[i : i + win]
        corr = window_rets.corr()

        # Average of upper triangle (excluding diagonal)
        upper = corr.values[np.triu_indices(n, k=1)]
        avg_corr_ts.iloc[i] = upper.mean()

        # Store snapshot every 21 days and the last day
        if i % snapshot_step == 0 or i == len(dates) - 1:
            snapshot_matrices[date] = corr

    current_corr = r.iloc[-win:].corr()

    return {
        "current_corr":       current_corr,
        "avg_corr_ts":        avg_corr_ts,
        "snapshot_matrices":  snapshot_matrices,
        "window":             win,
        "tickers":            tickers,
    }


# ─────────────────────────────────────────────
# 2. EIGENVALUE DECOMPOSITION (PCA)
# ─────────────────────────────────────────────

def compute_pca_decomposition(
    returns: pd.DataFrame,
    window: int = None,
    n_components: int = None,
) -> dict:
    """
    PCA decomposition of the return covariance matrix.

    Key outputs:
      - Eigenvalues: variance explained by each principal component
      - Eigenvectors: factor loadings (which assets drive each PC)
      - PC1 variance share: the fraction of total variance in PC1
        → High PC1 share = the portfolio acts like one risk factor
        → Low PC1 share = genuine diversification across factors
      - Effective N: equivalent number of uncorrelated assets
        = 1 / sum(λ_i² / (sum λ_i)²)
        Ranges from 1 (all assets move identically) to N (all independent)

    PCA of returns in a window captures the current factor structure,
    not just the unconditional structure. This is the dynamic version.
    """
    win  = window       or RISK_PARAMS.corr_window
    r    = returns.dropna()
    tickers = list(r.columns)
    n    = len(tickers)
    n_comp = n_components or n

    # Use the most recent window
    r_win = r.iloc[-win:]

    # Standardise: demeaned, unit variance (correlation PCA, not covariance PCA)
    r_std = (r_win - r_win.mean()) / r_win.std(ddof=1)
    cov   = r_std.cov().values   # correlation matrix from standardised returns

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    idx     = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_var    = eigvals.sum()
    var_explained = eigvals / total_var     # fraction each PC explains
    cum_var       = np.cumsum(var_explained)

    # PC1 variance share
    pc1_share = var_explained[0]

    # Effective number of assets
    # Marchenko-Pastur-inspired: how many PCs explain 80% of variance?
    n_80pct = int(np.searchsorted(cum_var, 0.80)) + 1

    # Effective N via entropy: exp(H) where H = -sum(p_i * log(p_i))
    p = var_explained
    entropy = -np.sum(p * np.log(p + 1e-12))
    effective_n = np.exp(entropy)

    # PC1 loadings: which assets load heavily on the dominant factor?
    pc1_loadings = pd.Series(eigvecs[:, 0], index=tickers, name="PC1_loading")
    pc2_loadings = pd.Series(eigvecs[:, 1], index=tickers, name="PC2_loading")

    # Top contributors to PC1
    pc1_abs = pc1_loadings.abs().sort_values(ascending=False)

    return {
        "eigenvalues":        eigvals,
        "eigenvectors":       eigvecs,
        "var_explained":      var_explained,
        "cum_var_explained":  cum_var,
        "pc1_share":          pc1_share,
        "pc1_loadings":       pc1_loadings,
        "pc2_loadings":       pc2_loadings,
        "pc1_top_contributors": pc1_abs,
        "effective_n":        effective_n,
        "n_pcs_80pct":        n_80pct,
        "tickers":            tickers,
        "n_assets":           n,
    }


def compute_rolling_pc1(
    returns: pd.DataFrame,
    window: int = None,
) -> pd.Series:
    """
    Rolling PC1 variance share time series.

    A rising PC1 share → assets becoming more correlated → less diversification.
    Sharp spikes in PC1 share mark crisis onset (COVID, GFC, 2022 rate shock).
    """
    win  = window or RISK_PARAMS.corr_window
    r    = returns.dropna()
    tickers = list(r.columns)
    n    = len(tickers)
    dates = r.index[win:]

    pc1_shares = []

    for i in range(len(dates)):
        r_win = r.iloc[i : i + win]
        r_std = (r_win - r_win.mean()) / (r_win.std(ddof=1) + 1e-10)
        cov   = r_std.cov().values
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        pc1_shares.append(eigvals[0] / eigvals.sum())

    return pd.Series(pc1_shares, index=dates, name="pc1_variance_share")


# ─────────────────────────────────────────────
# 3. DIVERSIFICATION RATIO
# ─────────────────────────────────────────────

def compute_diversification_ratio(
    returns: pd.DataFrame,
    weights: dict,
    window: int = None,
) -> dict:
    """
    Diversification Ratio = Weighted Average Vol / Portfolio Vol

    DR = (Σ w_i σ_i) / σ_portfolio

    DR = 1: no diversification benefit (all assets perfectly correlated)
    DR > 1: diversification is reducing portfolio vol below the weighted average
    DR → √N for equal-weighted, uncorrelated assets

    Also computes:
      - Diversification benefit: (DR - 1) as a percentage
      - Rolling DR time series
      - Marginal diversification contribution per asset
    """
    win = window or RISK_PARAMS.corr_window
    r   = returns.dropna()

    tickers = [t for t in weights.keys() if t in r.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()   # normalise

    # ── Point-in-time DR (full history) ──────
    r_sub   = r[tickers]
    sigma_i = r_sub.std(ddof=1).values             # individual vols
    cov_mat = r_sub.cov().values
    sigma_p = np.sqrt(w @ cov_mat @ w)             # portfolio vol
    weighted_avg_vol = (w * sigma_i).sum()

    dr = weighted_avg_vol / sigma_p if sigma_p > 0 else 1.0

    # ── Rolling DR ───────────────────────────
    dates  = r.index[win:]
    dr_rolling = pd.Series(index=dates, dtype=float, name="diversification_ratio")

    for i in range(len(dates)):
        r_win   = r_sub.iloc[i : i + win]
        sig_i   = r_win.std(ddof=1).values
        cov_w   = r_win.cov().values
        sig_p   = np.sqrt(w @ cov_w @ w)
        wav     = (w * sig_i).sum()
        dr_rolling.iloc[i] = wav / sig_p if sig_p > 0 else 1.0

    # ── Per-asset marginal contribution ──────
    # MCR_i = (Cov(r_i, r_p) / σ_p²) × w_i
    cov_vec = cov_mat @ w
    mcr     = (cov_vec * w) / (sigma_p**2) if sigma_p > 0 else w
    mcr_series = pd.Series(mcr, index=tickers, name="marginal_risk_contribution")
    pct_contrib = mcr_series / mcr_series.sum()

    return {
        "dr":                    dr,
        "dr_rolling":            dr_rolling,
        "portfolio_vol":         sigma_p,
        "weighted_avg_vol":      weighted_avg_vol,
        "diversification_benefit": (dr - 1) * 100,   # percent
        "marginal_risk_contribution": mcr_series,
        "pct_risk_contribution": pct_contrib,
        "asset_vols":            pd.Series(sigma_i, index=tickers),
    }


# ─────────────────────────────────────────────
# 4. CONDITIONAL CORRELATION (CRISIS CORRELATION)
# ─────────────────────────────────────────────

def compute_conditional_correlations(
    returns: pd.DataFrame,
    threshold_pct: float = 0.10,   # bottom 10% of market days = "bad days"
    market_proxy: str = "SPY",
) -> dict:
    """
    Correlations conditional on market stress.

    Compares:
      - Unconditional correlation (full sample)
      - Crisis correlation (days when market proxy is in bottom X%)
      - Calm correlation (days when market proxy is in top X%)

    The difference (crisis corr - calm corr) is the "correlation asymmetry":
    the degree to which diversification breaks down during drawdowns.

    This is what portfolio managers call "the dirty secret of diversification."
    """
    r = returns.dropna()
    tickers = [t for t in r.columns if t != market_proxy]

    if market_proxy not in r.columns:
        logger.warning(f"Market proxy {market_proxy} not in returns. Using first column.")
        market_proxy = r.columns[0]
        tickers = [t for t in r.columns if t != market_proxy]

    market_rets = r[market_proxy]
    threshold_low  = market_rets.quantile(threshold_pct)
    threshold_high = market_rets.quantile(1 - threshold_pct)

    crisis_mask = market_rets <= threshold_low
    calm_mask   = market_rets >= threshold_high

    unconditional_corr = r.corr()
    crisis_corr = r[crisis_mask].corr() if crisis_mask.sum() >= 20 else None
    calm_corr   = r[calm_mask].corr()   if calm_mask.sum()   >= 20 else None

    # Build asymmetry table: per-pair crisis vs calm correlation
    pairs = []
    for t in tickers:
        unc  = unconditional_corr.loc[market_proxy, t]
        cris = crisis_corr.loc[market_proxy, t]  if crisis_corr is not None else np.nan
        calm = calm_corr.loc[market_proxy, t]    if calm_corr   is not None else np.nan
        asym = cris - calm if not (np.isnan(cris) or np.isnan(calm)) else np.nan

        pairs.append({
            "Asset":             t,
            "Unconditional":     unc,
            "Crisis Corr":       cris,
            "Calm Corr":         calm,
            "Asymmetry":         asym,
            "Diversifies?":      "✗ Fails" if (not np.isnan(cris) and cris > 0.5) else "✓ Holds",
        })

    asymmetry_table = pd.DataFrame(pairs).set_index("Asset")

    # Average crisis correlation across all assets
    if crisis_corr is not None:
        upper_crisis = crisis_corr.values[np.triu_indices(len(r.columns), k=1)]
        avg_crisis_corr = upper_crisis.mean()
    else:
        avg_crisis_corr = np.nan

    upper_unc = unconditional_corr.values[np.triu_indices(len(r.columns), k=1)]
    avg_unc_corr = upper_unc.mean()

    return {
        "unconditional_corr":    unconditional_corr,
        "crisis_corr":           crisis_corr,
        "calm_corr":             calm_corr,
        "asymmetry_table":       asymmetry_table,
        "avg_unconditional":     avg_unc_corr,
        "avg_crisis":            avg_crisis_corr,
        "n_crisis_days":         int(crisis_mask.sum()),
        "n_calm_days":           int(calm_mask.sum()),
        "market_proxy":          market_proxy,
        "threshold_pct":         threshold_pct,
    }


# ─────────────────────────────────────────────
# 5. ASSET-CLASS BLOCK CORRELATIONS
# ─────────────────────────────────────────────

def compute_asset_class_correlations(
    returns: pd.DataFrame,
    window: int = None,
) -> dict:
    """
    Aggregates correlations by asset class block.

    Computes:
      - Intra-class correlations (e.g., equity-equity avg)
      - Inter-class correlations (e.g., equity-bond avg)
      - Rolling inter-class correlation time series

    The equity-bond correlation is particularly important:
    negative → bonds hedge equity drawdowns (traditional safe haven)
    positive → bonds and equities sell off together (2022 regime)
    """
    win = window or RISK_PARAMS.corr_window
    r   = returns.dropna()

    # Group tickers by asset class
    ac_map = {t: UNIVERSE[t]["asset_class"] for t in r.columns if t in UNIVERSE}
    classes = {}
    for ticker, ac in ac_map.items():
        classes.setdefault(ac, []).append(ticker)

    # Full-history correlation matrix
    full_corr = r.corr()

    # Block average correlations
    ac_names = list(classes.keys())
    block_corr = pd.DataFrame(index=ac_names, columns=ac_names, dtype=float)

    for ac1 in ac_names:
        for ac2 in ac_names:
            t1 = classes[ac1]
            t2 = classes[ac2]
            pairs = []
            for a in t1:
                for b in t2:
                    if a != b and a in full_corr.index and b in full_corr.columns:
                        pairs.append(full_corr.loc[a, b])
            block_corr.loc[ac1, ac2] = np.mean(pairs) if pairs else np.nan

    # Rolling equity-bond correlation (key regime indicator)
    eq_tickers   = [t for t in classes.get("Equity", [])   if t in r.columns]
    bond_tickers = [t for t in classes.get("Bond", [])     if t in r.columns]
    comm_tickers = [t for t in classes.get("Commodity", []) if t in r.columns]

    dates = r.index[win:]
    roll_eq_bond = pd.Series(index=dates, dtype=float, name="equity_bond_corr")
    roll_eq_comm = pd.Series(index=dates, dtype=float, name="equity_commodity_corr")

    for i in range(len(dates)):
        r_win = r.iloc[i : i + win]

        # Equity-Bond
        if eq_tickers and bond_tickers:
            pairs = []
            for a in eq_tickers:
                for b in bond_tickers:
                    pairs.append(r_win[a].corr(r_win[b]))
            roll_eq_bond.iloc[i] = np.mean(pairs)

        # Equity-Commodity
        if eq_tickers and comm_tickers:
            pairs = []
            for a in eq_tickers:
                for b in comm_tickers:
                    pairs.append(r_win[a].corr(r_win[b]))
            roll_eq_comm.iloc[i] = np.mean(pairs)

    return {
        "block_corr":        block_corr,
        "full_corr":         full_corr,
        "asset_classes":     classes,
        "roll_eq_bond":      roll_eq_bond,
        "roll_eq_comm":      roll_eq_comm,
    }


# ─────────────────────────────────────────────
# 6. CORRELATION REGIME CLASSIFICATION
# ─────────────────────────────────────────────

def classify_correlation_regime(
    avg_corr_ts: pd.Series,
) -> dict:
    """
    Classifies the current correlation regime using percentile thresholds.

    Regime interpretation:
      Low     (< p33)  : genuine diversification, portfolio benefits
      Normal  (p33-p66): typical market environment
      Elevated(p66-p85): risk-off early warning, correlations rising
      Crisis  (> p85)  : correlation breakdown, diversification failing
    """
    history = avg_corr_ts.dropna()
    current = float(history.iloc[-1])
    pctile  = (history < current).mean() * 100

    p33 = np.percentile(history, 33)
    p66 = np.percentile(history, 66)
    p85 = np.percentile(history, 85)

    if pctile < 33:
        regime = "🟢 Low Correlation"
        interpretation = "Diversification is working well. Assets moving independently."
    elif pctile < 66:
        regime = "✅ Normal"
        interpretation = "Typical correlation environment. Moderate diversification benefit."
    elif pctile < 85:
        regime = "🟡 Elevated"
        interpretation = "Correlations rising. Diversification benefit reducing. Monitor closely."
    else:
        regime = "🔴 Crisis Correlation"
        interpretation = "High systemic correlation. Portfolio behaving like a single risk factor."

    return {
        "current_avg_corr": current,
        "percentile_rank":  pctile,
        "regime":           regime,
        "interpretation":   interpretation,
        "p33":  p33, "p66": p66, "p85": p85,
        "history_mean":     history.mean(),
        "history_min":      history.min(),
        "history_max":      history.max(),
    }
