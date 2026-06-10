"""
risk/stress.py
==============
Stress testing — two flavours, one reverse test.

Flavour 1 — Historical Scenario Replay:
  Takes a named crisis period, extracts the actual return path from
  the data, and applies it to current portfolio weights.
  Answer: "If 2020 COVID happened again tomorrow, how much do we lose?"

Flavour 2 — Hypothetical Shock Scenarios:
  User-defined instantaneous shocks to each asset class.
  Answer: "If equities fall 25% and rates rise 150bps simultaneously,
           what is the P&L and which assets drive it?"

Reverse Stress Test:
  Works backwards from a target loss to find what market move
  would cause it. Answers: "What single equity shock wipes 10%
  of the portfolio?" and "What combined shock hits the mandate limit?"

All outputs include:
  - Total P&L (EUR and %)
  - Per-asset P&L contribution
  - Comparison to current VaR and ES
  - Severity classification (Moderate / Severe / Extreme / Catastrophic)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import UNIVERSE, RISK_PARAMS, PORTFOLIO_NOTIONAL

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SCENARIO DEFINITIONS
# ─────────────────────────────────────────────

@dataclass
class HistoricalScenario:
    name: str
    start: str
    end: str
    description: str

@dataclass
class HypotheticalScenario:
    """
    Defines a set of instantaneous asset-level shocks.
    shocks: {ticker: fractional return}  e.g. {"SPY": -0.25} = SPY falls 25%
    Assets not listed are assumed to have zero shock.
    """
    name: str
    shocks: Dict[str, float]
    description: str
    category: str = "Hypothetical"   # label for grouping in output


# ── Built-in Historical Scenarios ─────────────
HISTORICAL_SCENARIOS = [
    HistoricalScenario(
        "GFC 2008–09",
        "2008-09-15", "2009-03-31",
        "Lehman collapse through S&P trough. Equity -57%, credit spreads explode."
    ),
    HistoricalScenario(
        "EU Debt Crisis 2011",
        "2011-07-01", "2011-12-31",
        "Sovereign spread blowout, Italy/Spain on the edge. Risk-off globally."
    ),
    HistoricalScenario(
        "Taper Tantrum 2013",
        "2013-05-22", "2013-09-05",
        "Bernanke hints at tapering. Rates spike, EM assets sell off sharply."
    ),
    HistoricalScenario(
        "China/Oil Shock 2015–16",
        "2015-08-01", "2016-02-15",
        "CNY devaluation fear, oil collapse to $26. Global risk-off."
    ),
    HistoricalScenario(
        "COVID Crash 2020",
        "2020-02-19", "2020-03-23",
        "Fastest -34% drawdown in S&P history. 23 trading days."
    ),
    HistoricalScenario(
        "COVID Recovery 2020",
        "2020-03-24", "2020-08-31",
        "V-shaped recovery. Risk assets rebound sharply."
    ),
    HistoricalScenario(
        "2022 Rate Shock",
        "2022-01-03", "2022-10-13",
        "Fed hikes 425bps in 9 months. Both equities AND bonds sell off."
    ),
    HistoricalScenario(
        "SVB / Banking Crisis 2023",
        "2023-03-08", "2023-03-24",
        "SVB collapse, contagion fears. Rates volatile, financials hit hard."
    ),
]

# ── Built-in Hypothetical Scenarios ───────────
HYPOTHETICAL_SCENARIOS = [
    HypotheticalScenario(
        name="Equity Bear Market",
        shocks={
            "SPY": -0.30, "QQQ": -0.38, "EFA": -0.28, "EEM": -0.32,
            "HYG": -0.12, "EMB": -0.10,
            "TLT": +0.06, "IEF": +0.04,
            "GLD": +0.05, "SLV": -0.05,
            "USO": -0.25, "DBC": -0.15,
        },
        description="Full bear market. Equities -30%, flight to quality in govts.",
        category="Equity Shock"
    ),
    HypotheticalScenario(
        name="Rate Shock +200bps",
        shocks={
            "TLT": -0.22, "IEF": -0.13,
            "HYG": -0.08, "EMB": -0.10,
            "SPY": -0.12, "QQQ": -0.15, "EFA": -0.10, "EEM": -0.14,
            "GLD": -0.05,
            "USO": +0.03, "DBC": +0.02,
        },
        description="+200bps parallel shift in rates. 2022-style joint selloff.",
        category="Rate Shock"
    ),
    HypotheticalScenario(
        name="Stagflation",
        shocks={
            "SPY": -0.20, "QQQ": -0.25, "EFA": -0.18, "EEM": -0.20,
            "TLT": -0.15, "IEF": -0.09,
            "HYG": -0.10, "EMB": -0.12,
            "GLD": +0.15, "SLV": +0.12,
            "USO": +0.25, "DBC": +0.20,
        },
        description="Inflation stays high: equities and bonds fall, commodities rally.",
        category="Macro Regime"
    ),
    HypotheticalScenario(
        name="Credit Crisis",
        shocks={
            "HYG": -0.25, "EMB": -0.22,
            "SPY": -0.18, "QQQ": -0.15, "EFA": -0.16, "EEM": -0.24,
            "TLT": +0.04, "IEF": +0.03,
            "GLD": +0.03, "SLV": -0.08,
            "USO": -0.30, "DBC": -0.18,
        },
        description="Credit spreads blow out. HY -25%, EM debt -22%. 2008 flavour.",
        category="Credit Shock"
    ),
    HypotheticalScenario(
        name="USD Surge / EM Crisis",
        shocks={
            "EEM": -0.28, "EMB": -0.20, "EFA": -0.15,
            "SPY": -0.06, "QQQ": -0.07,
            "TLT": +0.02, "IEF": +0.01,
            "GLD": -0.08, "SLV": -0.12,
            "USO": -0.15, "DBC": -0.10,
        },
        description="Dollar rally, EM capital flight. EM equity/bonds hardest hit.",
        category="FX / EM Shock"
    ),
    HypotheticalScenario(
        name="Flash Crash (1-day)",
        shocks={
            "SPY": -0.10, "QQQ": -0.12, "EFA": -0.09, "EEM": -0.11,
            "HYG": -0.05, "EMB": -0.04,
            "TLT": +0.02, "IEF": +0.01,
            "GLD": +0.02, "SLV": -0.01,
            "USO": -0.05, "DBC": -0.03,
        },
        description="Single-day flash crash. Equities -10%, brief panic.",
        category="Tail Event"
    ),
    HypotheticalScenario(
        name="Soft Landing",
        shocks={
            "SPY": +0.12, "QQQ": +0.15, "EFA": +0.10, "EEM": +0.12,
            "HYG": +0.04, "EMB": +0.03,
            "TLT": +0.03, "IEF": +0.02,
            "GLD": -0.02, "SLV": -0.01,
            "USO": +0.02, "DBC": +0.02,
        },
        description="Goldilocks: inflation tamed, no recession. Risk-on across the board.",
        category="Positive Scenario"
    ),
]


# ─────────────────────────────────────────────
# SCENARIO RESULT CONTAINER
# ─────────────────────────────────────────────

@dataclass
class ScenarioResult:
    name: str
    category: str
    description: str
    total_return: float          # portfolio-level fractional return
    total_pnl: float             # EUR P&L
    notional: float
    asset_contributions: pd.Series    # per-asset P&L
    asset_shocks: pd.Series           # per-asset shock applied
    severity: str = ""

    def __post_init__(self):
        self.severity = _classify_severity(self.total_return)

    @property
    def total_return_pct(self):
        return f"{self.total_return:+.2%}"

    @property
    def total_pnl_eur(self):
        return f"€{self.total_pnl:>+,.0f}"


def _classify_severity(ret: float) -> str:
    if ret >= 0:
        return "✅ Positive"
    elif ret > -0.05:
        return "🟡 Moderate"
    elif ret > -0.10:
        return "🟠 Severe"
    elif ret > -0.20:
        return "🔴 Extreme"
    else:
        return "💀 Catastrophic"


# ─────────────────────────────────────────────
# 1. HISTORICAL SCENARIO ENGINE
# ─────────────────────────────────────────────

def run_historical_scenario(
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    scenario: HistoricalScenario,
    notional: float = None,
) -> ScenarioResult:
    """
    Replays a historical crisis period against current portfolio weights.

    Steps:
      1. Slice the return data to the scenario date range
      2. Compute cumulative return for each asset over that period
      3. Apply current portfolio weights to get total portfolio return
      4. Compute per-asset P&L contribution

    Note: This answers "what would our *current* portfolio have returned
    during that period?" — not what an actual portfolio would have done,
    since weights drift over time. Frozen-weight attribution.
    """
    notion = notional or PORTFOLIO_NOTIONAL
    r = asset_returns.copy()

    # Slice to scenario window
    mask = (r.index >= pd.Timestamp(scenario.start)) & \
           (r.index <= pd.Timestamp(scenario.end))
    r_window = r[mask]

    if len(r_window) == 0:
        logger.warning(
            f"No data for scenario '{scenario.name}' "
            f"({scenario.start} → {scenario.end}). "
            f"Data range: {r.index[0].date()} → {r.index[-1].date()}"
        )
        return None

    # Cumulative simple return per asset over the window
    # exp(sum of log returns) - 1 = total simple return
    cum_log_rets = r_window.sum()
    cum_simple   = np.exp(cum_log_rets) - 1

    # Apply weights
    tickers = [t for t in weights.keys() if t in cum_simple.index]
    w_vec   = np.array([weights[t] for t in tickers])
    w_vec   = w_vec / w_vec.sum()

    shocks_series = cum_simple[tickers]
    asset_pnl     = pd.Series(
        {t: weights[t] * cum_simple[t] * notion for t in tickers}
    )

    total_ret = float((w_vec * shocks_series.values).sum())
    total_pnl = total_ret * notion

    return ScenarioResult(
        name=scenario.name,
        category="Historical",
        description=scenario.description,
        total_return=total_ret,
        total_pnl=total_pnl,
        notional=notion,
        asset_contributions=asset_pnl,
        asset_shocks=shocks_series,
    )


# ─────────────────────────────────────────────
# 2. HYPOTHETICAL SHOCK ENGINE
# ─────────────────────────────────────────────

def run_hypothetical_scenario(
    weights: Dict[str, float],
    scenario: HypotheticalScenario,
    notional: float = None,
) -> ScenarioResult:
    """
    Applies instantaneous asset-level shocks to portfolio weights.

    For each asset:
      P&L_i = weight_i × shock_i × notional

    Portfolio P&L = Σ P&L_i
    """
    notion  = notional or PORTFOLIO_NOTIONAL
    tickers = list(weights.keys())

    asset_pnl  = {}
    asset_shocks = {}
    for t in tickers:
        shock = scenario.shocks.get(t, 0.0)
        asset_pnl[t]    = weights[t] * shock * notion
        asset_shocks[t] = shock

    total_pnl = sum(asset_pnl.values())
    total_ret = total_pnl / notion

    return ScenarioResult(
        name=scenario.name,
        category=scenario.category,
        description=scenario.description,
        total_return=total_ret,
        total_pnl=total_pnl,
        notional=notion,
        asset_contributions=pd.Series(asset_pnl),
        asset_shocks=pd.Series(asset_shocks),
    )


# ─────────────────────────────────────────────
# 3. RUN ALL SCENARIOS
# ─────────────────────────────────────────────

def run_all_scenarios(
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    notional: float = None,
    extra_hypothetical: List[HypotheticalScenario] = None,
) -> List[ScenarioResult]:
    """
    Runs all built-in historical + hypothetical scenarios.
    Returns a list of ScenarioResult objects sorted by P&L (worst first).
    """
    notion  = notional or PORTFOLIO_NOTIONAL
    results = []

    # Historical
    for scenario in HISTORICAL_SCENARIOS:
        res = run_historical_scenario(asset_returns, weights, scenario, notion)
        if res is not None:
            results.append(res)

    # Hypothetical
    all_hyp = HYPOTHETICAL_SCENARIOS + (extra_hypothetical or [])
    for scenario in all_hyp:
        res = run_hypothetical_scenario(weights, scenario, notion)
        results.append(res)

    # Sort: worst (most negative) first
    results.sort(key=lambda x: x.total_return)
    return results


def results_to_dataframe(results: List[ScenarioResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Scenario":      r.name,
            "Category":      r.category,
            "Return":        r.total_return,
            "P&L (EUR)":     r.total_pnl,
            "Severity":      r.severity,
            "Description":   r.description,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 4. REVERSE STRESS TEST
# ─────────────────────────────────────────────

def reverse_stress_test(
    weights: Dict[str, float],
    target_loss: float,
    notional: float = None,
    shock_mode: str = "single_asset",
    target_asset: str = "SPY",
) -> dict:
    """
    Reverse stress test: find the market move that causes a specified loss.

    Modes:
      "single_asset":    what shock to one asset causes the target loss?
      "equity_parallel": what uniform equity shock causes the target loss?
      "all_parallel":    what uniform shock to all assets causes target loss?

    target_loss: fractional loss (e.g. 0.10 = 10% portfolio loss).
                 Pass as a positive number — the function converts to negative P&L.

    Returns the required shock levels and whether they seem plausible.
    """
    notion = notional or PORTFOLIO_NOTIONAL
    loss_eur = target_loss * notion    # target loss in EUR

    results = {}

    if shock_mode == "single_asset":
        # For each asset: solve weight_i × shock_i × notion = -loss_eur
        for ticker, w in weights.items():
            if abs(w) < 1e-6:
                continue
            required_shock = -loss_eur / (w * notion)
            results[ticker] = {
                "required_shock": required_shock,
                "plausible": abs(required_shock) <= 1.0,
                "weight": w,
                "comment": (
                    f"{required_shock:+.1%} shock to {ticker} alone "
                    f"({'plausible' if abs(required_shock) <= 1.0 else 'implausible — exceeds 100%'})"
                )
            }
        return {
            "mode": "single_asset",
            "target_loss_pct": target_loss,
            "target_loss_eur": loss_eur,
            "asset_shocks": results,
        }

    elif shock_mode == "equity_parallel":
        equity_tickers = [t for t in weights if UNIVERSE.get(t, {}).get("asset_class") == "Equity"]
        total_eq_weight = sum(weights[t] for t in equity_tickers)
        if total_eq_weight == 0:
            return {"error": "No equity positions"}
        required_shock = -target_loss / total_eq_weight
        return {
            "mode": "equity_parallel",
            "target_loss_pct": target_loss,
            "target_loss_eur": loss_eur,
            "equity_weight": total_eq_weight,
            "required_equity_shock": required_shock,
            "plausible": abs(required_shock) <= 0.60,
            "comment": (
                f"Equities need to fall {abs(required_shock):.1%} uniformly "
                f"to lose {target_loss:.0%} of portfolio "
                f"({'plausible' if abs(required_shock) <= 0.60 else 'severe — exceeds historical extremes'})"
            )
        }

    elif shock_mode == "all_parallel":
        total_weight = sum(abs(w) for w in weights.values())
        net_weight   = sum(weights.values())
        if net_weight == 0:
            return {"error": "Net weight is zero"}
        required_shock = -target_loss / net_weight
        return {
            "mode": "all_parallel",
            "target_loss_pct": target_loss,
            "target_loss_eur": loss_eur,
            "required_uniform_shock": required_shock,
            "comment": f"All assets fall {abs(required_shock):.1%} uniformly."
        }

    else:
        raise ValueError(f"Unknown shock_mode: {shock_mode}")


# ─────────────────────────────────────────────
# 5. SCENARIO vs VaR COMPARISON
# ─────────────────────────────────────────────

def compare_scenarios_to_var(
    results: List[ScenarioResult],
    var_1d_hs: float,
    es_1d_hs: float,
    notional: float = None,
) -> pd.DataFrame:
    """
    Shows how each scenario loss compares to current 1d VaR and ES.
    Scenarios much larger than VaR are "beyond VaR" tail events.
    """
    notion = notional or PORTFOLIO_NOTIONAL
    var_eur = var_1d_hs * notion
    es_eur  = es_1d_hs  * notion

    rows = []
    for r in results:
        multiples_of_var = r.total_pnl / (-var_eur) if var_eur != 0 else np.nan
        multiples_of_es  = r.total_pnl / (-es_eur)  if es_eur  != 0 else np.nan
        rows.append({
            "Scenario":          r.name,
            "P&L (EUR)":         r.total_pnl,
            "Return":            r.total_return,
            "× 1d VaR":          multiples_of_var,
            "× 1d ES":           multiples_of_es,
            "Severity":          r.severity,
        })

    df = pd.DataFrame(rows)
    df["In VaR?"] = df["× 1d VaR"].apply(
        lambda x: "Within VaR" if (not np.isnan(x) and abs(x) <= 1) else "Beyond VaR"
        if x < 0 else "Gain"
    )
    return df
