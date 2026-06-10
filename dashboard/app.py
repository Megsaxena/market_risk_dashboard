"""
dashboard/app.py
================
Market Risk Dashboard — Interactive Plotly Dash app.

Run with:  python dashboard/app.py
Then open: http://localhost:8050

Six tabs:
  1. Overview       — NAV, daily P&L, portfolio stats
  2. VaR & ES       — three methods, term structure, rolling VaR, backtest
  3. Drawdown       — underwater curve, episodes, rolling MDD
  4. Volatility     — realised, EWMA, GARCH, vol regime, term structure
  5. Correlation    — heatmap, avg pairwise, PC1 share, diversification ratio
  6. Stress Tests   — tornado chart, scenario table, reverse stress
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

# ── Risk modules ─────────────────────────────
from config import UNIVERSE, RISK_PARAMS, PORTFOLIO_NOTIONAL
from data.fetcher import fetch_universe, compute_returns
from data.portfolio import (
    build_default_portfolio, compute_portfolio_returns,
    compute_pnl, compute_portfolio_stats, compute_asset_class_weights,
)
from risk.var_es import compute_all_var, compute_var_term_structure, compute_rolling_var
from risk.drawdown import compute_drawdown_series, compute_drawdown_episodes, compute_drawdown_metrics, compute_rolling_max_drawdown
from risk.volatility import compute_all_vol
from risk.correlation import (
    compute_rolling_correlation, compute_pca_decomposition,
    compute_rolling_pc1, compute_diversification_ratio,
    compute_conditional_correlations, compute_asset_class_correlations,
    classify_correlation_regime,
)
from risk.stress import run_all_scenarios, compare_scenarios_to_var

# ═══════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════
DARK   = "#0d1117"
CARD   = "#161b22"
CARD2  = "#1c2128"
BORDER = "#30363d"
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
ORANGE = "#d29922"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

# Base layout — NO xaxis/yaxis here to avoid duplicate keyword conflicts
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD2,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
)

# Axis style applied separately in each chart
AXIS = dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER, tickcolor=MUTED)

def base_layout(**overrides):
    """Merges base layout with per-chart overrides cleanly."""
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(overrides)
    return layout

def card(children, style=None):
    base = dict(
        background=CARD, border=f"1px solid {BORDER}",
        borderRadius="8px", padding="20px", marginBottom="16px",
    )
    if style:
        base.update(style)
    return html.Div(children, style=base)

def stat_box(label, value, color=TEXT, sublabel=None):
    return html.Div([
        html.Div(label, style=dict(color=MUTED, fontSize="11px", textTransform="uppercase", letterSpacing="0.05em", marginBottom="4px")),
        html.Div(value, style=dict(color=color, fontSize="22px", fontWeight="700", fontFamily="monospace")),
        html.Div(sublabel or "", style=dict(color=MUTED, fontSize="11px", marginTop="2px")),
    ], style=dict(
        background=CARD2, border=f"1px solid {BORDER}",
        borderRadius="6px", padding="14px 18px", flex="1", minWidth="140px",
    ))

def section_title(text):
    return html.H3(text, style=dict(fontSize="13px", fontWeight="600",
        textTransform="uppercase", letterSpacing="0.08em", color=MUTED,
        marginBottom="12px", marginTop="4px"))

# ═══════════════════════════════════════════════
# DATA LOADING  (runs once at startup)
# ═══════════════════════════════════════════════
print("Loading data and computing risk metrics...")

portfolio  = build_default_portfolio()
prices     = fetch_universe(tickers=portfolio.tickers)
log_rets   = compute_returns(prices, method="log")
port_rets  = compute_portfolio_returns(log_rets, portfolio)
cum_pnl    = compute_pnl(port_rets, portfolio.notional)
daily_pnl  = compute_pnl(port_rets, portfolio.notional, method="daily")

stats      = compute_portfolio_stats(port_rets)
var_data   = compute_all_var(port_rets)
term_str   = compute_var_term_structure(port_rets)
roll_var   = compute_rolling_var(port_rets, method="hs")
dd_series  = compute_drawdown_series(port_rets)
dd_eps     = compute_drawdown_episodes(dd_series, top_n=10)
dd_metrics = compute_drawdown_metrics(port_rets, dd_series)
roll_mdd   = compute_rolling_max_drawdown(port_rets)
vol_data   = compute_all_vol(port_rets, run_garch=True)
roll_corr  = compute_rolling_correlation(log_rets)
pca        = compute_pca_decomposition(log_rets)
roll_pc1   = compute_rolling_pc1(log_rets)
div_data   = compute_diversification_ratio(log_rets, portfolio.weights)
ac_corr    = compute_asset_class_correlations(log_rets)
corr_regime = classify_correlation_regime(roll_corr["avg_corr_ts"])
stress_res = run_all_scenarios(log_rets, portfolio.weights, portfolio.notional)
vs_var     = compare_scenarios_to_var(stress_res, var_data["hs"].var_pct, var_data["hs"].es_pct)

notion = portfolio.notional
nav    = portfolio.notional + cum_pnl

print("✓ All metrics computed. Starting dashboard...")

# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════
def pct(v):  return f"{v:+.2%}"
def eur(v):  return f"€{v:>+,.0f}"
def num(v):  return f"{v:.3f}"
def col(v):  return GREEN if v >= 0 else RED

# ═══════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════

def make_nav_chart():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav.values,
        line=dict(color=BLUE, width=1.8),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        name="NAV", hovertemplate="€%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    colors = [GREEN if v >= 0 else RED for v in daily_pnl.values]
    fig.add_trace(go.Bar(
        x=daily_pnl.index, y=daily_pnl.values,
        marker_color=colors, name="Daily P&L",
        hovertemplate="€%{y:,.0f}<extra></extra>",
    ), row=2, col=1)
    fig.update_layout(**base_layout(height=440, showlegend=False,
    title=dict(text="Portfolio NAV & Daily P&L", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS), yaxis=dict(**AXIS, tickprefix="€", tickformat=",.0f"),
    yaxis2=dict(**AXIS),
))
    return fig



def make_rolling_var_chart():
    rv = roll_var.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rv.index, y=rv["rolling_var"] * notion,
        name="1d HS VaR (€)", line=dict(color=ORANGE, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=rv.index, y=rv["rolling_es"] * notion,
        name="1d HS ES (€)", line=dict(color=RED, width=1.5, dash="dot"),
    ))
    breach_days = rv[rv["realised_return"] < -rv["rolling_var"]]
    fig.add_trace(go.Scatter(
        x=breach_days.index, y=abs(breach_days["realised_return"]) * notion,
        mode="markers", name="VaR Breach",
        marker=dict(color=RED, size=5, symbol="x"),
    ))
    fig.update_layout(**base_layout(height=320,
    title=dict(text="Rolling 1d HS VaR & ES", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS), yaxis=dict(**AXIS, tickprefix="€", tickformat=",.0f"),
))
    return fig


def make_var_comparison_chart():
    labels  = ["Historical\nSimulation", "Parametric\n(Normal)", "Parametric\n(Student-t)", "Monte Carlo\n(t-dist)"]
    var_vals = [var_data["hs"].var_pct, var_data["parametric"].var_pct,
                var_data["parametric_t"].var_pct, var_data["mc"].var_pct]
    es_vals  = [var_data["hs"].es_pct,  var_data["parametric"].es_pct,
                var_data["parametric_t"].es_pct, var_data["mc"].es_pct]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="VaR", x=labels, y=[v * notion for v in var_vals],
        marker_color=ORANGE, text=[f"€{v*notion:,.0f}" for v in var_vals], textposition="outside"))
    fig.add_trace(go.Bar(name="ES",  x=labels, y=[v * notion for v in es_vals],
        marker_color=RED,    text=[f"€{v*notion:,.0f}" for v in es_vals], textposition="outside"))
    fig.update_layout(**base_layout(height=320, barmode="group",
    title=dict(text="VaR & ES — All Methods (1d, 95%)", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS), yaxis=dict(**AXIS, tickprefix="€", tickformat=",.0f"),
))
    return fig


def make_term_structure_chart():
    ts = term_str
    horizons = ts.index.tolist()
    fig = go.Figure()
    for col_name, color, label in [
        ("HS VaR %", BLUE, "HS VaR"),
        ("Param VaR %", ORANGE, "Parametric"),
        ("MC VaR %", GREEN, "Monte Carlo"),
    ]:
        fig.add_trace(go.Scatter(
            x=horizons, y=(ts[col_name] * notion).tolist(),
            name=label, line=dict(width=2), mode="lines+markers",
        ))
    fig.update_layout(**base_layout(height=280,
    title=dict(text="VaR Term Structure (1d→10d)", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS, title="Horizon (days)"),
    yaxis=dict(**AXIS, title="VaR (€)", tickprefix="€", tickformat=",.0f"),
))
    return fig


def make_drawdown_chart():
    dd = dd_series["drawdown"] * 100
    nav_norm = dd_series["nav"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.55], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(
        x=nav_norm.index, y=nav_norm.values,
        line=dict(color=BLUE, width=1.5), name="NAV",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", fillcolor="rgba(248,81,73,0.25)",
        line=dict(color=RED, width=1.2), name="Drawdown %",
        hovertemplate="%{y:.2f}%<extra></extra>",
    ), row=2, col=1)
    if not dd_eps.empty:
        for _, ep in dd_eps.head(3).iterrows():
            fig.add_vline(x=ep["Trough"], line=dict(color=RED, width=1, dash="dot"), row=2, col=1)
    fig.update_layout(**base_layout(height=400, showlegend=False,
    title=dict(text="Portfolio NAV & Underwater Drawdown Curve", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS), yaxis=dict(**AXIS), xaxis2=dict(**AXIS), yaxis2=dict(**AXIS, ticksuffix="%"),
))
    return fig


def make_vol_chart():
    rv = vol_data["rolling_vol"].dropna()
    ew = vol_data["ewma_vol"].dropna()
    gc = vol_data["garch_vol"]
    fig = go.Figure()
    for col_name, color, dash, label in [
        ("vol_21d", "#7ee8a2", "solid", "21d Realised"),
        ("vol_63d", BLUE, "solid", "63d Realised"),
        ("vol_252d","#a78bfa", "solid", "252d Realised"),
    ]:
        if col_name in rv.columns:
            fig.add_trace(go.Scatter(x=rv.index, y=rv[col_name]*100,
                name=label, line=dict(color=color, width=1.5, dash=dash)))
    fig.add_trace(go.Scatter(x=ew.index, y=ew.values*100,
        name="EWMA (λ=0.94)", line=dict(color=ORANGE, width=2)))
    if gc is not None:
        fig.add_trace(go.Scatter(x=gc.index, y=gc.values*100,
            name="GARCH(1,1)", line=dict(color=RED, width=1.5, dash="dot")))
    fig.update_layout(**base_layout(height=360,
    title=dict(text="Volatility Estimators (Annualised %)", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS), yaxis=dict(**AXIS, ticksuffix="%"),
))

    return fig


def make_corr_heatmap():
    corr = roll_corr["current_corr"]
    tickers = list(corr.columns)
    z = corr.values
    fig = go.Figure(go.Heatmap(
        z=z, x=tickers, y=tickers,
        colorscale=[[0, "#1e3a5f"], [0.5, CARD2], [1, "#7f1d1d"]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(z, 2), texttemplate="%{text}",
        textfont=dict(size=9), colorbar=dict(thickness=12),
        hovertemplate="%{x} / %{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(height=380,
    title=dict(text=f"Current Correlation Matrix ({RISK_PARAMS.corr_window}d window)", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS, side="bottom"), yaxis=dict(**AXIS),
))
    return fig


def make_avg_corr_chart():
    ac = roll_corr["avg_corr_ts"].dropna()
    pc1 = roll_pc1.dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.06,
                        subplot_titles=["Avg Pairwise Correlation", "PC1 Variance Share"])
    fig.add_trace(go.Scatter(x=ac.index, y=ac.values,
        line=dict(color=BLUE, width=1.5), name="Avg Corr",
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"), row=1, col=1)
    fig.add_hline(y=float(corr_regime["p85"]), line=dict(color=RED, dash="dash", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=pc1.index, y=pc1.values*100,
        line=dict(color=ORANGE, width=1.5), name="PC1 Share %"), row=2, col=1)
    fig.add_hline(y=50, line=dict(color=RED, dash="dash", width=1), row=2, col=1)
    fig.update_layout(**base_layout(height=380, showlegend=False,
    title=dict(text="Correlation Regime Indicators", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS), yaxis=dict(**AXIS), yaxis2=dict(**AXIS, ticksuffix="%"),
))
    return fig


def make_stress_tornado():
    res = sorted(stress_res, key=lambda r: r.total_pnl)
    names  = [r.name for r in res]
    values = [r.total_pnl / 1000 for r in res]  # in thousands
    colors = [RED if v < 0 else GREEN for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"€{v*1000:>+,.0f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}: €%{x:.0f}k<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=BORDER, width=1))
    fig.update_layout(**base_layout(height=420,
    title=dict(text="Stress Scenario P&L (€k)", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS, tickprefix="€", ticksuffix="k"),
    yaxis=dict(**AXIS),
    margin=dict(l=200, r=80, t=40, b=40),
))
    return fig


def make_risk_contribution_chart():
    pct_risk = div_data["pct_risk_contribution"].sort_values(ascending=False)
    weights  = pd.Series(portfolio.weights)
    names    = [UNIVERSE[t]["name"] for t in pct_risk.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Risk Contribution", x=pct_risk.values*100, y=names,
        orientation="h", marker_color=RED))
    fig.add_trace(go.Bar(name="Portfolio Weight",  x=[weights.get(t,0)*100 for t in pct_risk.index], y=names,
        orientation="h", marker_color=BLUE, opacity=0.6))
    fig.update_layout(**base_layout(height=360, barmode="overlay",
    title=dict(text="Risk Contribution vs Portfolio Weight (%)", font=dict(size=13, color=MUTED)),
    xaxis=dict(**AXIS, ticksuffix="%"), yaxis=dict(**AXIS),
    margin=dict(l=180, r=30, t=40, b=40),
))
    return fig


# ═══════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════

# Computed header stats
current_var  = var_data["hs"].var_pct
current_es   = var_data["hs"].es_pct
current_dd   = dd_metrics["Current Drawdown"]
current_vol  = vol_data["current_ewma"]
current_nav  = nav.iloc[-1]
total_ret    = (current_nav / notion) - 1

app = dash.Dash(__name__, title="Market Risk Dashboard")
app.layout = html.Div(style=dict(background=DARK, minHeight="100vh", fontFamily="Inter, sans-serif", color=TEXT), children=[

    # ── Header ─────────────────────────────────
    html.Div(style=dict(
        background=CARD, borderBottom=f"1px solid {BORDER}",
        padding="16px 28px", display="flex", alignItems="center", justifyContent="space-between",
    ), children=[
        html.Div([
            html.Span("Market Risk Dashboard", style=dict(fontSize="18px", fontWeight="700", color=TEXT)),
            html.Span(f"  ·  {portfolio.name}", style=dict(fontSize="13px", color=MUTED, marginLeft="8px")),
        ]),
        html.Div(f"€{notion/1e6:.0f}M Notional  ·  {len(log_rets)} trading days  ·  {log_rets.index[-1].date()}", 
                 style=dict(fontSize="12px", color=MUTED)),
    ]),

    # ── KPI Strip ──────────────────────────────
    html.Div(style=dict(display="flex", gap="12px", padding="16px 28px", flexWrap="wrap"), children=[
        stat_box("Current NAV",       f"€{current_nav/1e6:.3f}M", BLUE),
        stat_box("Total Return",      pct(total_ret),   col(total_ret)),
        stat_box("1d VaR (95% HS)",   pct(-current_var), RED,    f"€{current_var*notion:,.0f}"),
        stat_box("1d ES (95% HS)",    pct(-current_es),  RED,    f"€{current_es*notion:,.0f}"),
        stat_box("Current Drawdown",  pct(current_dd),   col(current_dd), "from peak"),
        stat_box("EWMA Vol",          f"{current_vol:.1%}", ORANGE, vol_data["regime"]["regime"]),
        stat_box("Corr Regime",       corr_regime["regime"].split(" ",1)[-1], BLUE,
                 f"avg ρ = {corr_regime['current_avg_corr']:.3f}"),
        stat_box("Diversif. Ratio",   f"{div_data['dr']:.3f}",  GREEN, f"DR = {div_data['dr']:.2f}"),
    ]),

    # ── Tabs ───────────────────────────────────
    html.Div(style=dict(padding="0 28px 28px"), children=[
        dcc.Tabs(id="tabs", value="overview", style=dict(marginBottom="0"),
                 colors=dict(border=BORDER, primary=BLUE, background=CARD),
            children=[

            # ── TAB 1: OVERVIEW ───────────────
            dcc.Tab(label="Overview", value="overview", style=dict(color=MUTED), selected_style=dict(color=TEXT, borderTop=f"2px solid {BLUE}", background=CARD), children=[
                html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="16px", marginTop="16px"), children=[
                    card(dcc.Graph(figure=make_nav_chart(), config=dict(displayModeBar=False))),
                    card([
                        section_title("Portfolio Statistics"),
                        html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="8px"), children=[
                            html.Div([
                                html.Div(k, style=dict(color=MUTED, fontSize="11px")),
                                html.Div(v, style=dict(color=TEXT, fontFamily="monospace", fontSize="13px")),
                            ]) for k, v in list(stats.items())[:12]
                        ]),
                        html.Div(style=dict(marginTop="16px"), children=[
                            section_title("Asset Class Weights"),
                            *[html.Div(style=dict(display="flex", alignItems="center", gap="8px", marginBottom="6px"), children=[
                                html.Div(ac, style=dict(color=MUTED, fontSize="12px", width="100px")),
                                html.Div(style=dict(
                                    width=f"{w*280:.0f}px", height="14px",
                                    background=BLUE if ac == "Equity" else (GREEN if ac == "Bond" else ORANGE),
                                    borderRadius="2px",
                                )),
                                html.Div(f"{w:.1%}", style=dict(color=TEXT, fontSize="12px", fontFamily="monospace")),
                            ]) for ac, w in compute_asset_class_weights(portfolio).items()],
                        ]),
                    ]),
                ]),
            ]),

            # ── TAB 2: VaR & ES ───────────────
            dcc.Tab(label="VaR & ES", value="var", style=dict(color=MUTED), selected_style=dict(color=TEXT, borderTop=f"2px solid {BLUE}", background=CARD), children=[
                html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="16px", marginTop="16px"), children=[
                    card(dcc.Graph(figure=make_var_comparison_chart(), config=dict(displayModeBar=False))),
                    card(dcc.Graph(figure=make_term_structure_chart(), config=dict(displayModeBar=False))),
                ]),
                card(dcc.Graph(figure=make_rolling_var_chart(), config=dict(displayModeBar=False))),
            ]),

            # ── TAB 3: DRAWDOWN ───────────────
            dcc.Tab(label="Drawdown", value="drawdown", style=dict(color=MUTED), selected_style=dict(color=TEXT, borderTop=f"2px solid {BLUE}", background=CARD), children=[
                card(dcc.Graph(figure=make_drawdown_chart(), config=dict(displayModeBar=False)), style=dict(marginTop="16px")),
                html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="16px"), children=[
                    card([
                        section_title("Drawdown Metrics"),
                        html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="8px"), children=[
                            html.Div([
                                html.Div(k, style=dict(color=MUTED, fontSize="11px")),
                                html.Div(
                                    f"{v:.4%}" if isinstance(v, float) and abs(v) < 10 else
                                    f"{v:.2f}" if isinstance(v, float) else str(v),
                                    style=dict(color=col(v) if isinstance(v, float) and k in
                                        ["Max Drawdown","Current Drawdown","Average Drawdown","Annualised Return"] else TEXT,
                                        fontFamily="monospace", fontSize="13px")),
                            ]) for k, v in dd_metrics.items()
                        ]),
                    ]),
                    card([
                        section_title("Top Drawdown Episodes"),
                        dash_table.DataTable(
                            data=dd_eps.reset_index(drop=True).rename(columns={
                                "Start": "Start", "Trough": "Trough", "Depth": "Depth",
                                "Duration (days)": "Days", "Recovered": "Rec.",
                            }).assign(
                                Depth=lambda df: df["Depth"].apply(lambda x: f"{x:.2%}"),
                                Start=lambda df: df["Start"].apply(lambda x: str(x.date())),
                                Trough=lambda df: df["Trough"].apply(lambda x: str(x.date())),
                            )[["Start","Trough","Depth","Days","Rec."]].to_dict("records"),
                            columns=[{"name": c, "id": c} for c in ["Start","Trough","Depth","Days","Rec."]],
                            style_table=dict(overflowX="auto"),
                            style_header=dict(background=CARD2, color=MUTED, border=f"1px solid {BORDER}", fontSize="11px"),
                            style_cell=dict(background=CARD, color=TEXT, border=f"1px solid {BORDER}",
                                fontSize="12px", fontFamily="monospace", padding="6px 10px"),
                            style_data_conditional=[{
                                "if": {"filter_query": '{Rec.} = "False"'},
                                "color": ORANGE,
                            }],
                        ),
                    ]),
                ]),
            ]),

            # ── TAB 4: VOLATILITY ─────────────
            dcc.Tab(label="Volatility", value="volatility", style=dict(color=MUTED), selected_style=dict(color=TEXT, borderTop=f"2px solid {BLUE}", background=CARD), children=[
                card(dcc.Graph(figure=make_vol_chart(), config=dict(displayModeBar=False)), style=dict(marginTop="16px")),
                html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="16px"), children=[
                    card([
                        section_title("Vol Term Structure (current)"),
                        dash_table.DataTable(
                            data=vol_data["term_structure"].to_dict("records"),
                            columns=[{"name": c, "id": c} for c in ["Window","Current Vol","Signal"]
                                     if c in vol_data["term_structure"].columns],
                            style_header=dict(background=CARD2, color=MUTED, border=f"1px solid {BORDER}", fontSize="11px"),
                            style_cell=dict(background=CARD, color=TEXT, border=f"1px solid {BORDER}",
                                fontSize="12px", fontFamily="monospace", padding="6px 10px"),
                        ),
                    ]),
                    card([
                        section_title("GARCH(1,1) Parameters"),
                        *([
                            html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="8px"), children=[
                                html.Div([
                                    html.Div(k.replace("_"," ").title(), style=dict(color=MUTED, fontSize="11px")),
                                    html.Div(f"{v:.4f}" if isinstance(v, float) else str(v),
                                        style=dict(color=TEXT, fontFamily="monospace", fontSize="13px")),
                                ]) for k, v in vol_data["garch_params"].items()
                            ])
                        ] if vol_data["garch_params"] else [html.Div("GARCH not available", style=dict(color=MUTED))]),
                    ]),
                ]),
            ]),

            # ── TAB 5: CORRELATION ────────────
            dcc.Tab(label="Correlation", value="correlation", style=dict(color=MUTED), selected_style=dict(color=TEXT, borderTop=f"2px solid {BLUE}", background=CARD), children=[
                html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="16px", marginTop="16px"), children=[
                    card(dcc.Graph(figure=make_corr_heatmap(),    config=dict(displayModeBar=False))),
                    card(dcc.Graph(figure=make_avg_corr_chart(), config=dict(displayModeBar=False))),
                ]),
                html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="16px"), children=[
                    card(dcc.Graph(figure=make_risk_contribution_chart(), config=dict(displayModeBar=False))),
                    card([
                        section_title("PCA Factor Structure"),
                        html.Div(style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="10px"), children=[
                            html.Div([html.Div("PC1 Variance Share", style=dict(color=MUTED, fontSize="11px")),
                                html.Div(f"{pca['pc1_share']:.1%}", style=dict(color=ORANGE, fontFamily="monospace", fontSize="20px", fontWeight="700"))]),
                            html.Div([html.Div("Effective N (entropy)", style=dict(color=MUTED, fontSize="11px")),
                                html.Div(f"{pca['effective_n']:.2f}", style=dict(color=BLUE, fontFamily="monospace", fontSize="20px", fontWeight="700"))]),
                            html.Div([html.Div("PCs for 80% variance", style=dict(color=MUTED, fontSize="11px")),
                                html.Div(f"{pca['n_pcs_80pct']}", style=dict(color=TEXT, fontFamily="monospace", fontSize="20px", fontWeight="700"))]),
                            html.Div([html.Div("Diversification Ratio", style=dict(color=MUTED, fontSize="11px")),
                                html.Div(f"{div_data['dr']:.3f}", style=dict(color=GREEN, fontFamily="monospace", fontSize="20px", fontWeight="700"))]),
                        ]),
                        html.Div(style=dict(marginTop="16px"), children=[
                            section_title("PC1 Loadings (top contributors)"),
                            *[html.Div(style=dict(display="flex", gap="8px", alignItems="center", marginBottom="4px"), children=[
                                html.Div(t, style=dict(color=MUTED, fontSize="11px", width="36px", fontFamily="monospace")),
                                html.Div(style=dict(
                                    width=f"{abs(float(pca['pc1_top_contributors'][t]))*180:.0f}px",
                                    height="10px", background=BLUE if pca["pc1_loadings"][t] > 0 else RED,
                                    borderRadius="2px",
                                )),
                                html.Div(f"{pca['pc1_top_contributors'][t]:.3f}", style=dict(color=TEXT, fontSize="11px", fontFamily="monospace")),
                            ]) for t in list(pca["pc1_top_contributors"].index[:8])],
                        ]),
                    ]),
                ]),
            ]),

            # ── TAB 6: STRESS TESTS ───────────
            dcc.Tab(label="Stress Tests", value="stress", style=dict(color=MUTED), selected_style=dict(color=TEXT, borderTop=f"2px solid {BLUE}", background=CARD), children=[
                card(dcc.Graph(figure=make_stress_tornado(), config=dict(displayModeBar=False)), style=dict(marginTop="16px")),
                card([
                    section_title("Scenario Summary Table"),
                    dash_table.DataTable(
                        data=[{
                            "Scenario": r.name,
                            "Category": r.category,
                            "Return": f"{r.total_return:+.2%}",
                            "P&L (EUR)": f"€{r.total_pnl:>+,.0f}",
                            "× VaR": f"{r.total_pnl/(-var_data['hs'].var_pct*notion):.1f}×",
                            "Severity": r.severity,
                        } for r in sorted(stress_res, key=lambda x: x.total_pnl)],
                        columns=[{"name": c, "id": c} for c in ["Scenario","Category","Return","P&L (EUR)","× VaR","Severity"]],
                        style_table=dict(overflowX="auto"),
                        style_header=dict(background=CARD2, color=MUTED, border=f"1px solid {BORDER}", fontSize="11px"),
                        style_cell=dict(background=CARD, color=TEXT, border=f"1px solid {BORDER}",
                            fontSize="12px", fontFamily="monospace", padding="7px 12px"),
                        style_data_conditional=[
                            {"if": {"filter_query": '{Return} contains "-"'}, "color": RED},
                            {"if": {"filter_query": '{Return} contains "+"'}, "color": GREEN},
                        ],
                    ),
                ]),
            ]),
        ]),
    ]),
])

if __name__ == "__main__":
    app.run(debug=False, port=8050)
