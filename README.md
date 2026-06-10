# Market Risk Dashboard

A quantitative multi-asset market risk system built in Python, covering the six core dimensions of portfolio risk management: Value at Risk, Expected Shortfall, Drawdowns, Volatility, Correlation Risk, and Stress Testing.

Built as a portfolio project to demonstrate applied risk methodology, from raw price data through to an interactive Plotly Dash dashboard.

---

## Live Demo

> Run locally following the [Quick Start](#quick-start) below.  
> The dashboard runs on `http://localhost:8050` and requires no cloud deployment.

---

## What It Does

| Module | Methods | Key Output |
|---|---|---|
| **VaR & ES** | Historical Simulation, Parametric (Normal + Student-t), Monte Carlo | Point-in-time & rolling 1d–10d VaR, Kupiec backtest |
| **Drawdown** | Peak-to-trough, Ulcer Index, Pain Index | Underwater curve, episode table, rolling MDD |
| **Volatility** | Rolling (21/63/126/252d), EWMA (RiskMetrics λ=0.94), GARCH(1,1) | Regime classification, vol term structure, vol-of-vol |
| **Correlation** | Rolling matrix, PCA eigendecomposition, conditional correlation | Diversification ratio, effective N, crisis correlation breakdown |
| **Stress Tests** | Historical scenario replay, hypothetical shocks, reverse stress test | Tornado chart, × VaR multiples, per-asset attribution |
| **Portfolio** | Weighted log returns, cumulative P&L, risk contribution | Sharpe, Sortino, Calmar, attribution by asset and asset class |

---

## Project Structure

```
market_risk_dashboard/
│
├── config.py                  # Universe, weights, all risk parameters
├── main.py                    # Phase 1: data pipeline validation
├── run_phase2.py              # VaR & ES report
├── run_phase3.py              # Drawdown & volatility report
├── run_phase4.py              # Correlation risk report
├── run_phase5.py              # Stress test report
│
├── data/
│   ├── fetcher.py             # Price fetch: yfinance → Bloomberg → synthetic
│   ├── portfolio.py           # Portfolio class, P&L engine, attribution
│   └── synthetic.py           # Multivariate-t synthetic data with crisis regimes
│
├── risk/
│   ├── var_es.py              # VaR (3 methods), ES, rolling VaR, Kupiec test
│   ├── drawdown.py            # Drawdown series, episodes, Calmar, Ulcer Index
│   ├── volatility.py          # Realised, EWMA, GARCH(1,1), vol regime
│   └── correlation.py         # Rolling matrix, PCA, diversification ratio
│   └── stress.py              # Historical replay, hypothetical shocks, reverse test
│
├── dashboard/
│   └── app.py                 # Interactive Plotly Dash app (6 tabs)
│
├── outputs/                   # CSV outputs from each phase run
└── requirements.txt
```

---

## Quick Start

**1. Clone the repo**

```bash
git clone https://github.com/YOUR_USERNAME/market_risk_dashboard.git
cd market_risk_dashboard
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the dashboard**

```bash
python dashboard/app.py
```

Open `http://localhost:8050` in your browser.

**4. Run individual phase reports** (optional — console output)

```bash
python main.py          # Phase 1: data pipeline
python run_phase2.py    # VaR & ES
python run_phase3.py    # Drawdown & Volatility
python run_phase4.py    # Correlation Risk
python run_phase5.py    # Stress Tests
```

---

## Asset Universe

A 12-asset multi-class portfolio (€10M default notional):

| Asset Class | Tickers | Weight |
|---|---|---|
| **Equity** | SPY, QQQ, EFA, EEM | 50% |
| **Fixed Income** | TLT, IEF, HYG, EMB | 35% |
| **Commodities** | GLD, SLV, USO, DBC | 15% |

Weights, tickers, notional, and all risk parameters are configured in a single file: `config.py`.

---

## Methodology Notes

### Why three VaR methods?

Each method makes different assumptions. The spread between them is itself a risk signal:

- **Historical Simulation** — non-parametric; reads the empirical tail directly. Accurate but slow to react to regime changes and constrained by the lookback window.
- **Parametric (Normal)** — closed-form; fast and clean but underestimates tail risk when kurtosis > 3 (which it almost always is). Useful as a lower-bound benchmark.
- **Parametric (Student-t)** — same structure but fits a t-distribution, capturing fat tails. ES diverges substantially from the Normal case when df is low — the correct way to penalise heavy tails.
- **Monte Carlo** — fits distribution parameters, simulates 10,000 paths. For horizons > 1 day, MC compounds daily returns into full paths rather than using the sqrt-T approximation, which is the correct treatment for non-linear portfolios.

### Why GARCH over simple rolling vol?

Rolling volatility treats all observations in a window equally. GARCH(1,1) weights recent observations more heavily through the persistence parameter (α + β), producing a mean-reverting conditional volatility that spikes after shocks and decays with a calibrated half-life. For this portfolio, α + β ≈ 0.955 implying a half-life of ~15 days — elevated volatility after a shock persists for several weeks.

### Why eigenvalue decomposition for correlation risk?

Average pairwise correlation tells you the level of co-movement. PCA decomposition tells you the *structure*: how many independent risk factors actually drive the portfolio. If PC1 explains 70%+ of variance (as it did in March 2020), the 12-asset portfolio is behaving like a single risk factor — diversification has collapsed. The Effective N metric (entropy of the eigenvalue distribution) summarises this in one number: it ranged from ~5.5 in normal markets to ~2.5 at the COVID peak.

### Synthetic data design

When Bloomberg/yfinance is unavailable, the system generates synthetic prices using a multivariate t-distribution (df=5) with:
- Correlation structure calibrated to approximate 2015–2024 long-run realised correlations
- Per-asset volatility targets matched to historical realised vol
- Two injected crisis regimes: COVID March 2020 (vol ×3.5, correlation boost +0.35) and 2022 rate shock (joint equity-bond selloff)
- Seeded for reproducibility

This is not a substitute for real data but makes the dashboard fully demonstrable offline.

---

## Data Sources

The fetcher tries sources in this order:

```
1. Bloomberg Terminal (xbbg) — requires terminal running, pip install xbbg
2. yfinance              — free, works on any internet-connected machine
3. Synthetic generator   — offline fallback, clearly flagged in output
```

You can force a specific source:

```python
from data.fetcher import fetch_universe
prices = fetch_universe(source="bloomberg")   # or "yfinance" or "synthetic"
```

On startup the dashboard prints the active data source and the last available date. If prices are synthetic, a warning is raised.

---

## Configuration

Everything in `config.py`:

```python
# Change the portfolio
PORTFOLIO_WEIGHTS = {
    "SPY": 0.25, "TLT": 0.15, ...
}

# Change risk parameters
RISK_PARAMS = RiskParams(
    confidence_level=0.975,    # Basel ES standard
    var_horizon_days=10,       # regulatory 10-day VaR
    historical_window=504,     # 2 years
    n_simulations=50_000,      # MC paths
    ewma_lambda=0.94,          # RiskMetrics daily
)

# Change notional
PORTFOLIO_NOTIONAL = 50_000_000   # €50M
```

---

## Dashboard Tabs

| Tab | Charts |
|---|---|
| **Overview** | NAV curve, daily P&L bar chart, portfolio stats, risk contribution vs weight |
| **VaR & ES** | Four-method comparison, term structure (1d–10d), rolling VaR with breach markers |
| **Drawdown** | Underwater curve, metrics panel (Calmar, Ulcer, Pain Index), top episodes table |
| **Volatility** | Four estimators on one chart, vol term structure, GARCH parameters |
| **Correlation** | Live heatmap, avg pairwise corr + PC1 share time series, diversification stats |
| **Stress Tests** | Tornado chart (worst→best), per-scenario asset attribution, × VaR multiples |

---

## Tech Stack

| Layer | Library |
|---|---|
| Data | yfinance, pandas, numpy |
| Risk math | scipy, numpy |
| Visualisation | plotly, dash |
| Export | openpyxl |
| Bloomberg (optional) | xbbg |

Python 3.10+ required.

---

## Outputs

Each phase run writes CSVs to `outputs/`:

```
prices.csv                      raw price matrix
phase1_summary.csv              portfolio returns + P&L series
var_comparison.csv              all VaR methods side by side
var_term_structure.csv          1d–10d VaR across methods
rolling_var.csv                 rolling VaR time series + breach flags
drawdown_series.csv             NAV, peak, drawdown series
drawdown_episodes.csv           top 10 drawdown episodes with recovery
rolling_vol_all_windows.csv     21/63/126/252d realised vol
vol_estimators.csv              EWMA + GARCH time series
correlation_current.csv         current correlation matrix
avg_corr_timeseries.csv         rolling avg pairwise correlation
pca_decomposition.csv           eigenvalues and variance explained
conditional_correlations.csv    crisis vs calm correlation table
stress_all_scenarios.csv        all scenario P&L summary
stress_asset_breakdown.csv      per-asset attribution per scenario
```

---

## Background

Built to demonstrate applied quantitative risk management across a multi-asset portfolio. The methodology follows industry-standard approaches used in front-office risk teams:

- VaR and ES as defined in **Basel III/IV** (97.5% ES for regulatory capital)
- Kupiec POF backtest for model validation
- RiskMetrics EWMA (λ = 0.94) as the industry-standard reactive vol estimator
- Sqrt-T horizon scaling with MC multi-day path simulation as the correct alternative
- Conditional correlation analysis to assess whether diversification holds under stress

---

## License

MIT — free to use, adapt, and build on.
