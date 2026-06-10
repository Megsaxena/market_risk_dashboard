"""
config.py
=========
Single source of truth for the entire dashboard.
Change tickers, weights, or risk parameters here — nothing else needs touching.
"""

from dataclasses import dataclass, field
from typing import Dict

# ─────────────────────────────────────────────
# ASSET UNIVERSE
# ─────────────────────────────────────────────
# Each ticker maps to metadata used in labels, grouping, and stress scenarios.
UNIVERSE: Dict[str, dict] = {
    # ── Equities ──────────────────────────────
    "SPY":  {"name": "S&P 500",          "asset_class": "Equity",    "region": "US"},
    "QQQ":  {"name": "Nasdaq 100",        "asset_class": "Equity",    "region": "US"},
    "EFA":  {"name": "MSCI EAFE",         "asset_class": "Equity",    "region": "Intl"},
    "EEM":  {"name": "MSCI EM",           "asset_class": "Equity",    "region": "EM"},
    # ── Fixed Income ──────────────────────────
    "TLT":  {"name": "US 20Y Treasury",   "asset_class": "Bond",      "region": "US"},
    "IEF":  {"name": "US 7-10Y Treasury", "asset_class": "Bond",      "region": "US"},
    "HYG":  {"name": "US High Yield",     "asset_class": "Bond",      "region": "US"},
    "EMB":  {"name": "EM USD Bonds",      "asset_class": "Bond",      "region": "EM"},
    # ── Commodities ───────────────────────────
    "GLD":  {"name": "Gold",              "asset_class": "Commodity", "region": "Global"},
    "SLV":  {"name": "Silver",            "asset_class": "Commodity", "region": "Global"},
    "USO":  {"name": "WTI Crude Oil",     "asset_class": "Commodity", "region": "Global"},
    "DBC":  {"name": "Broad Commodities", "asset_class": "Commodity", "region": "Global"},
}

# ─────────────────────────────────────────────
# PORTFOLIO WEIGHTS
# ─────────────────────────────────────────────
# A representative 60/30/10 multi-asset allocation.
# Weights must sum to 1.0. Negative weights = short positions.
PORTFOLIO_WEIGHTS: Dict[str, float] = {
    # Equities — 50%
    "SPY":  0.25,
    "QQQ":  0.10,
    "EFA":  0.10,
    "EEM":  0.05,
    # Fixed Income — 35%
    "TLT":  0.15,
    "IEF":  0.10,
    "HYG":  0.05,
    "EMB":  0.05,
    # Commodities — 15%
    "GLD":  0.07,
    "SLV":  0.03,
    "USO":  0.02,
    "DBC":  0.03,
}

# Portfolio notional (EUR equivalent — adjust to your book size)
PORTFOLIO_NOTIONAL: float = 10_000_000  # €10M

# ─────────────────────────────────────────────
# DATA PARAMETERS
# ─────────────────────────────────────────────
DATA_START_DATE: str = "2015-01-01"   # gives us GFC aftermath + COVID + 2022 rate shock
CACHE_DIR: str = "./cache"            # local price cache (CSV)
PRICE_FIELD: str = "Close"            # Adj Close accounts for dividends/splits

# ─────────────────────────────────────────────
# RISK PARAMETERS
# ─────────────────────────────────────────────
@dataclass
class RiskParams:
    # VaR / ES
    confidence_level: float = 0.95      # 95% default; Basel = 0.975
    var_horizon_days: int = 1           # 1-day VaR; scale by sqrt(T) for multi-day
    historical_window: int = 504        # 2 trading years for HS-VaR
    n_simulations: int = 10_000         # Monte Carlo paths

    # Volatility
    ewma_lambda: float = 0.94           # RiskMetrics daily decay factor
    vol_windows: list = field(default_factory=lambda: [21, 63, 126, 252])

    # Correlation
    corr_window: int = 63               # rolling correlation window (3 months)

    # Stress scenarios (historical date ranges)
    stress_scenarios: dict = field(default_factory=lambda: {
        "GFC 2008–09":        ("2008-09-01", "2009-03-31"),
        "EU Debt Crisis 2011": ("2011-07-01", "2011-12-31"),
        "COVID Crash 2020":   ("2020-02-19", "2020-03-23"),
        "2022 Rate Shock":    ("2022-01-01", "2022-10-31"),
        "SVB Crisis 2023":    ("2023-03-01", "2023-03-31"),
    })


# Global risk parameter instance — import this everywhere
RISK_PARAMS = RiskParams()
