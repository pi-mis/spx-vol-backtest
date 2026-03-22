"""
backtest.py
===========
Vectorised backtest on VXX daily returns.

P&L mechanic:
  We trade VXX — a real, liquid ETN that tracks the S&P 500
  VIX Short-Term Futures Index.

  Going LONG  VXX: profit when VIX rises (long volatility).
  Going SHORT VXX: profit when VIX falls (short volatility, harvesting VRP).

  Daily return of our position:
    pnl_t = effective_pos_{t-1} × VXX_return_t

  Where VXX_return_t = (VXX_t − VXX_{t-1}) / VXX_{t-1}

  1-day execution lag: signal from close of day t → position held day t+1.

Transaction costs:
  Applied as a percentage of the trade value:
    cost_t = |effective_pos_t − effective_pos_{t-1}| × TRANSACTION_COST_PCT × portfolio_value

  This is the same 5 bps (0.05%) assumption used in Zarattini et al.

Portfolio:
  Starts at $1.00 (normalised).
  position_value_t = effective_pos_{t-1} × portfolio_value_{t-1}
  P&L is compounded (we re-invest gains, losses reduce capital).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from config import TRANSACTION_COST_PCT, SIGNAL_THRESHOLD


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    signals_df: pd.DataFrame,
    vxx:        pd.Series,
    spy:        pd.Series,
    verbose:    bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    signals_df : output of signals.generate_signals()
    vxx        : VXX daily adjusted close prices
    spy        : SPY daily adjusted close prices (for benchmark)
    verbose    : print sample P&L

    Returns
    -------
    DataFrame with full daily P&L breakdown.
    """
    # Daily returns (fill_method=None avoids pandas FutureWarning)
    vxx_ret = vxx.pct_change(fill_method=None)
    spy_ret = spy.pct_change(fill_method=None)

    bt = signals_df.copy()
    bt["vxx_ret"] = vxx_ret
    bt["spy_ret"] = spy_ret
    bt = bt.dropna(subset=["vxx_ret"])

    # 1-day execution lag
    bt["held_pos"] = bt["effective_pos"].shift(1).fillna(0.0)

    # Gross P&L (% of capital)
    bt["gross_pnl"] = bt["held_pos"] * bt["vxx_ret"]

    # Transaction cost: charged on position changes
    bt["pos_change"]       = bt["held_pos"].diff().abs().fillna(0.0)
    bt["transaction_cost"] = bt["pos_change"] * TRANSACTION_COST_PCT

    bt["net_pnl"] = bt["gross_pnl"] - bt["transaction_cost"]

    # Compounded equity curve
    bt["equity"] = (1 + bt["net_pnl"]).cumprod()

    # SPY benchmark (buy and hold, normalised to 1 at backtest start)
    spy_ret_clean = bt["spy_ret"].fillna(0)
    bt["spy_equity"] = (1 + spy_ret_clean).cumprod()

    # Drawdown
    bt["peak"]     = bt["equity"].cummax()
    bt["drawdown"] = (bt["equity"] - bt["peak"]) / bt["peak"]

    if verbose:
        _print_sample(bt)

    return bt


def _print_sample(bt: pd.DataFrame) -> None:
    active = bt[bt["held_pos"].abs() > 0.01].head(5)
    if active.empty:
        return
    print("\n  Sample P&L (first 5 active days):")
    print(f"  {'Date':12}  {'Pos':>7}  {'VIX':>6}  "
          f"{'VXX ret':>8}  {'Gross':>8}  {'Cost':>7}  {'Net':>8}")
    print("  " + "─" * 65)
    for dt, r in active.iterrows():
        print(f"  {str(dt.date()):12}  "
              f"{r['held_pos']:>+7.3f}  "
              f"{r['vix']:>6.1f}  "
              f"{r['vxx_ret']*100:>+7.2f}%  "
              f"{r['gross_pnl']*100:>+7.2f}%  "
              f"{r['transaction_cost']*100:>6.3f}%  "
              f"{r['net_pnl']*100:>+7.2f}%")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    bt:          pd.DataFrame,
    annual_days: int = 252,
) -> dict:
    pnl    = bt["net_pnl"].dropna()
    equity = bt["equity"].dropna()

    # Returns
    total_ret    = float(equity.iloc[-1] - 1)
    n_years      = len(pnl) / annual_days
    cagr         = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    # Risk
    ann_vol  = pnl.std() * np.sqrt(annual_days)
    sharpe   = cagr / ann_vol if ann_vol > 0 else np.nan

    down_vol = pnl[pnl < 0].std() * np.sqrt(annual_days)
    sortino  = cagr / down_vol if down_vol > 0 else np.nan

    max_dd   = float(bt["drawdown"].min())
    calmar   = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Hit rate and profit factor
    hit_rate = (pnl > 0).mean()
    gp = pnl[pnl > 0].sum()
    gl = pnl[pnl < 0].abs().sum()
    pf = gp / gl if gl > 0 else np.nan

    # SPY benchmark
    spy_ret    = bt["spy_ret"].dropna()
    spy_equity = (1 + spy_ret).cumprod()
    spy_total  = float(spy_equity.iloc[-1] - 1)
    spy_years  = len(spy_ret) / annual_days
    spy_cagr   = (1 + spy_total) ** (1 / spy_years) - 1 if spy_years > 0 else np.nan
    spy_vol    = spy_ret.std() * np.sqrt(annual_days)
    spy_sharpe = spy_cagr / spy_vol if spy_vol > 0 else np.nan

    # Correlation with SPY
    corr = pnl.corr(bt["spy_ret"].reindex(pnl.index).fillna(0))

    n_trades  = int((bt["pos_change"] > 0.005).sum())
    avg_hold  = len(bt[bt["held_pos"].abs() > 0.01]) / n_trades if n_trades > 0 else 0

    return {
        "Period":                   f"{bt.index[0].date()} → {bt.index[-1].date()}",
        "Total days":               len(pnl),
        "─── Strategy ──────────":  "──────────────",
        "Total return":             f"{total_ret*100:.2f}%",
        "CAGR":                     f"{cagr*100:.2f}%",
        "Ann. volatility":          f"{ann_vol*100:.2f}%",
        "Sharpe ratio":             round(sharpe, 3),
        "Sortino ratio":            round(sortino, 3),
        "Max drawdown":             f"{max_dd*100:.2f}%",
        "Calmar ratio":             round(calmar, 3),
        "Hit rate":                 f"{hit_rate*100:.1f}%",
        "Profit factor":            round(pf, 3),
        "N trades":                 n_trades,
        "Avg holding (days)":       round(avg_hold, 1),
        "Corr. with SPY":           f"{corr*100:.1f}%",
        "─── Benchmark (SPY) ───":  "──────────────",
        "SPY total return":         f"{spy_total*100:.2f}%",
        "SPY CAGR":                 f"{spy_cagr*100:.2f}%",
        "SPY Sharpe":               round(spy_sharpe, 3),
    }


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_results(
    bt:        pd.DataFrame,
    metrics:   dict,
    save_path: str = "backtest_results.png",
) -> None:
    """
    6-panel chart:
      1. VIX spot + NS forecast + VIX3M
      2. Forecast spread with threshold lines
      3. Position over time
      4. VXX daily returns coloured by position
      5. Equity curve vs SPY
      6. Rolling 252-day Sharpe
    """
    BG, PANEL, GRID = "#0d1117", "#161b22", "#30363d"
    TEXT, MUTED     = "#e6edf3", "#8b949e"
    BLUE, ORANGE    = "#58a6ff", "#f0883e"
    GREEN, RED      = "#3fb950", "#f85149"
    PURPLE, YELLOW  = "#bc8cff", "#d29922"

    fig = plt.figure(figsize=(16, 26), facecolor=BG)
    gs  = gridspec.GridSpec(6, 1, figure=fig, hspace=0.48,
                            top=0.95, bottom=0.04, left=0.08, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    def _style(ax, title=""):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=0, ha="center", color=MUTED)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.5)
        if title:
            ax.set_title(title, color=MUTED, fontsize=9, pad=4)

    fig.suptitle(
        "S&P 500 Volatility Strategy — Nelson-Siegel Term Structure Forecasting\n"
        "NS + ARIMA forecast  |  VIX term structure signal  |  VXX execution",
        fontsize=13, color=TEXT, fontweight="bold",
    )

    thr = SIGNAL_THRESHOLD

    # 1. VIX term structure ────────────────────────────────────────────────────
    ax = axes[0]
    _style(ax, "VIX Term Structure: Spot (30d), Forecast (30d), VIX3M (90d)")
    ax.plot(bt.index, bt["vix"],          color=TEXT,   lw=1.2, label="VIX (30d, spot)")
    ax.plot(bt.index, bt["vix_forecast"], color=BLUE,   lw=0.9,
            ls="--", alpha=0.85, label="NS+ARIMA forecast (30d)")
    ax.plot(bt.index, bt["vix3m"],        color=PURPLE, lw=0.8,
            alpha=0.7, label="VIX3M (90d)")
    ax.fill_between(bt.index, bt["vix"], bt["vix3m"],
                    where=(bt["vix"] < bt["vix3m"]),
                    color=GREEN, alpha=0.07, label="Contango")
    ax.fill_between(bt.index, bt["vix"], bt["vix3m"],
                    where=(bt["vix"] >= bt["vix3m"]),
                    color=RED, alpha=0.07, label="Backwardation")
    ax.set_ylabel("VIX (%)", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8,
              framealpha=0.8, ncol=3, loc="upper right")

    # 2. Forecast spread ───────────────────────────────────────────────────────
    ax = axes[1]
    _style(ax, f"Forecast Spread = NS_VIX_forecast − VIX_spot  (±{thr} threshold)")
    ax.plot(bt.index, bt["spread"], color=ORANGE, lw=0.9, alpha=0.9)
    ax.fill_between(bt.index, bt["spread"],  thr,
                    where=(bt["spread"] < -thr), color=GREEN, alpha=0.25)
    ax.fill_between(bt.index, bt["spread"], -thr,
                    where=(bt["spread"] >  thr), color=RED,   alpha=0.25)
    ax.axhline( thr, color=GREEN, lw=1.0, ls="--", alpha=0.7)
    ax.axhline(-thr, color=RED,   lw=1.0, ls="--", alpha=0.7)
    ax.axhline(0,    color=GRID,  lw=0.6)
    ax.set_ylabel("Spread (vol pts)", color=TEXT)

    # 3. Position ──────────────────────────────────────────────────────────────
    ax = axes[2]
    _style(ax, "Position in VXX  (negative = short, positive = long, 1-day lag)")
    ax.fill_between(bt.index, bt["held_pos"],
                    where=(bt["held_pos"] < -0.01),
                    color=GREEN, alpha=0.8, step="post", label="Short VXX")
    ax.fill_between(bt.index, bt["held_pos"],
                    where=(bt["held_pos"] >  0.01),
                    color=RED,   alpha=0.8, step="post", label="Long VXX")
    ax.axhline(0, color=GRID, lw=0.6)
    ax.set_ylabel("Position", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)

    # 4. Daily VXX returns coloured by position ────────────────────────────────
    ax = axes[3]
    _style(ax, "Daily VXX Returns  (colour = position direction)")
    colors = []
    for hp, ret in zip(bt["held_pos"], bt["vxx_ret"].fillna(0)):
        if hp < -0.01:   colors.append(GREEN if ret < 0 else RED)
        elif hp > 0.01:  colors.append(GREEN if ret > 0 else RED)
        else:            colors.append(MUTED)
    ax.bar(bt.index, bt["vxx_ret"].fillna(0) * 100, color=colors,
           width=1.5, alpha=0.7)
    ax.axhline(0, color=GRID, lw=0.6)
    ax.set_ylabel("VXX return (%)", color=TEXT)

    # 5. Equity vs SPY ─────────────────────────────────────────────────────────
    ax = axes[4]
    _style(ax, "Equity Curve vs SPY Buy-and-Hold")
    ax.plot(bt.index, bt["equity"],     color=ORANGE, lw=1.5, label="Strategy", zorder=3)
    ax.plot(bt.index, bt["spy_equity"], color=BLUE,   lw=1.0,
            alpha=0.7, label="SPY (buy & hold)", zorder=2)
    ax.fill_between(bt.index, bt["peak"], bt["equity"],
                    color=RED, alpha=0.2, label="Drawdown", zorder=1)
    ax.axhline(1.0, color=GRID, lw=0.8, ls="--")
    ax.set_ylabel("Equity (normalised)", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)

    # 6. Rolling Sharpe ────────────────────────────────────────────────────────
    ax = axes[5]
    _style(ax, "Rolling 252-day Sharpe Ratio")
    pnl         = bt["net_pnl"]
    roll_mean   = pnl.rolling(252).mean() * 252
    roll_std    = pnl.rolling(252).std()  * np.sqrt(252)
    roll_sharpe = roll_mean / roll_std.replace(0, np.nan)
    ax.plot(bt.index, roll_sharpe, color=YELLOW, lw=1.0)
    ax.axhline(0,   color=GRID,  lw=0.6, ls="--")
    ax.axhline(1.0, color=GREEN, lw=0.6, ls=":", alpha=0.5, label="Sharpe=1")
    ax.axhline(-1.0,color=RED,   lw=0.6, ls=":", alpha=0.5)
    ax.set_ylabel("Sharpe (252d)", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)

    # Footer ───────────────────────────────────────────────────────────────────
    m = metrics
    footer = (
        f"CAGR: {m['CAGR']}   Sharpe: {m['Sharpe ratio']}   "
        f"Max DD: {m['Max drawdown']}   Sortino: {m['Sortino ratio']}   "
        f"Hit rate: {m['Hit rate']}   Corr(SPY): {m['Corr. with SPY']}"
    )
    fig.text(0.5, 0.010, footer, ha="center", fontsize=9, color=MUTED,
             bbox=dict(facecolor=PANEL, edgecolor=GRID, boxstyle="round,pad=0.4"))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"\n  ✓ Chart saved → {save_path}")
