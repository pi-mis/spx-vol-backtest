"""
backtest.py
===========
Vectorised backtest using dual instruments:
  Short vol → LONG SVXY (ProShares -0.5x VIX ETF)
  Long  vol → LONG VXX  (iPath +1x VIX ETN)

P&L: gross_pnl_t = held_pos_{t-1} x instrument_return_t
     1-day execution lag (signal at close t → position next day)
     Transaction cost = 5 bps per trade (same as Zarattini et al.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from config import TRANSACTION_COST_PCT, SIGNAL_THRESHOLD


def run_backtest(
    signals_df: pd.DataFrame,
    combined:   pd.DataFrame,
    verbose:    bool = True,
) -> pd.DataFrame:
    vxx  = combined["vxx"].loc[~combined["vxx"].index.duplicated(keep="last")]
    spy  = combined["spy"].loc[~combined["spy"].index.duplicated(keep="last")]
    svxy = combined["svxy"].loc[~combined["svxy"].index.duplicated(keep="last")] \
           if "svxy" in combined.columns else pd.Series(dtype=float)

    vxx_ret  = vxx.pct_change(fill_method=None)
    spy_ret  = spy.pct_change(fill_method=None)
    svxy_ret = svxy.pct_change(fill_method=None) if len(svxy) > 0 else pd.Series(dtype=float)

    bt = signals_df.copy()
    bt["vxx_ret"]  = vxx_ret
    bt["svxy_ret"] = svxy_ret
    bt["spy_ret"]  = spy_ret

    # 1-day lag
    bt["held_pos"]    = bt["effective_pos"].shift(1).fillna(0.0)
    bt["held_instr"]  = bt["instrument"].shift(1).fillna("")

    # P&L: use svxy_ret when instrument is 'svxy', vxx_ret when 'vxx'
    # For SVXY: held_pos is negative (short vol direction = long SVXY)
    #   → we hold abs(held_pos) of SVXY, so pnl = abs(held_pos) × svxy_ret
    #   → since held_pos < 0 for short vol and we flip the sign: pnl = -held_pos × svxy_ret
    # For VXX: held_pos is positive (long vol = long VXX)
    #   → pnl = held_pos × vxx_ret
    instr_ret = pd.Series(0.0, index=bt.index)
    svxy_mask = bt["held_instr"] == "svxy"
    vxx_mask  = bt["held_instr"] == "vxx"

    instr_ret[svxy_mask] = bt.loc[svxy_mask, "svxy_ret"].fillna(0)
    instr_ret[vxx_mask]  = bt.loc[vxx_mask,  "vxx_ret"].fillna(0)

    # Short vol: held_pos < 0, we're long SVXY → negate
    gross_pnl = bt["held_pos"].copy() * 0.0
    gross_pnl[svxy_mask] = (-bt.loc[svxy_mask, "held_pos"]) * instr_ret[svxy_mask]
    gross_pnl[vxx_mask]  = ( bt.loc[vxx_mask,  "held_pos"]) * instr_ret[vxx_mask]

    bt["gross_pnl"]        = gross_pnl
    bt["pos_change"]       = bt["held_pos"].diff().abs().fillna(0.0)
    bt["transaction_cost"] = bt["pos_change"] * TRANSACTION_COST_PCT
    bt["net_pnl"]          = bt["gross_pnl"] - bt["transaction_cost"]
    bt["equity"]           = (1 + bt["net_pnl"]).cumprod()
    bt["spy_equity"]       = (1 + bt["spy_ret"].fillna(0)).cumprod()
    bt["peak"]             = bt["equity"].cummax()
    bt["drawdown"]         = (bt["equity"] - bt["peak"]) / bt["peak"]

    if verbose:
        active = bt[(bt["held_pos"].abs() > 0.01)].head(5)
        if not active.empty:
            print("\n  Sample P&L (first 5 active days):")
            print(f"  {'Date':12}  {'Pos':>7}  {'VIX':>6}  {'Instr':>5}  "
                  f"{'Ret':>7}  {'Gross':>7}  {'Net':>7}")
            print("  " + "─" * 62)
            for dt, r in active.iterrows():
                ret = instr_ret.loc[dt] * 100 if dt in instr_ret.index else 0
                print(f"  {str(dt.date()):12}  "
                      f"{r['held_pos']:>+7.3f}  "
                      f"{r['vix']:>6.1f}  "
                      f"{r['held_instr']:>5}  "
                      f"{ret:>+6.2f}%  "
                      f"{r['gross_pnl']*100:>+6.2f}%  "
                      f"{r['net_pnl']*100:>+6.2f}%")

    return bt


def compute_metrics(bt: pd.DataFrame, annual_days: int = 252) -> dict:
    pnl    = bt["net_pnl"].dropna()
    equity = bt["equity"].dropna()

    total_ret = float(equity.iloc[-1] - 1)
    n_years   = len(pnl) / annual_days
    cagr      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol   = pnl.std() * np.sqrt(annual_days)
    sharpe    = cagr / ann_vol if ann_vol > 0 else np.nan
    down_vol  = pnl[pnl < 0].std() * np.sqrt(annual_days)
    sortino   = cagr / down_vol if down_vol > 0 else np.nan
    max_dd    = float(bt["drawdown"].min())
    calmar    = cagr / abs(max_dd) if max_dd != 0 else np.nan
    hit_rate  = (pnl > 0).mean()
    gp = pnl[pnl > 0].sum(); gl = pnl[pnl < 0].abs().sum()
    pf = gp / gl if gl > 0 else np.nan
    corr = pnl.corr(bt["spy_ret"].reindex(pnl.index).fillna(0))
    n_trades  = int((bt["pos_change"] > 0.005).sum())
    avg_hold  = len(bt[bt["held_pos"].abs() > 0.01]) / n_trades if n_trades > 0 else 0

    spy_ret   = bt["spy_ret"].dropna()
    spy_eq    = (1 + spy_ret).cumprod()
    spy_tot   = float(spy_eq.iloc[-1] - 1) if len(spy_eq) > 0 else np.nan
    spy_yrs   = len(spy_ret) / annual_days
    spy_cagr  = (1 + spy_tot) ** (1 / spy_yrs) - 1 if spy_yrs > 0 else np.nan
    spy_vol   = spy_ret.std() * np.sqrt(annual_days)
    spy_sharpe= spy_cagr / spy_vol if spy_vol > 0 else np.nan

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
        "SPY total return":         f"{spy_tot*100:.2f}%",
        "SPY CAGR":                 f"{spy_cagr*100:.2f}%",
        "SPY Sharpe":               round(spy_sharpe, 3),
    }


def plot_results(bt, metrics, save_path="backtest_results.png"):
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
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", color=MUTED)
        for s in ax.spines.values(): s.set_color(GRID)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.5)
        if title: ax.set_title(title, color=MUTED, fontsize=9, pad=4)

    fig.suptitle(
        "S&P 500 Volatility Strategy — NS+ARIMA + eVRP + Term Structure\n"
        "Short vol via SVXY  |  Long vol via VXX  |  Dynamic sizing",
        fontsize=13, color=TEXT, fontweight="bold")

    # 1. VIX term structure
    ax = axes[0]; _style(ax, "VIX Term Structure: Spot (30d), Forecast, VIX3M (90d)")
    ax.plot(bt.index, bt["vix"],          color=TEXT,   lw=1.2, label="VIX (30d)")
    ax.plot(bt.index, bt["vix_forecast"], color=BLUE,   lw=0.9, ls="--", alpha=0.85, label="NS+ARIMA forecast")
    ax.plot(bt.index, bt["vix3m"],        color=PURPLE, lw=0.8, alpha=0.7, label="VIX3M (90d)")
    ax.fill_between(bt.index, bt["vix"], bt["vix3m"],
                    where=(bt["vix"] < bt["vix3m"]), color=GREEN, alpha=0.07)
    ax.fill_between(bt.index, bt["vix"], bt["vix3m"],
                    where=(bt["vix"] >= bt["vix3m"]), color=RED, alpha=0.07)
    ax.set_ylabel("VIX (%)", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8, ncol=3)

    # 2. eVRP
    ax = axes[1]; _style(ax, "eVRP = VIX − 10d Realized Vol of SPY")
    evrp = bt.get("evrp", bt["vix"] * 0)
    ax.plot(bt.index, evrp, color=ORANGE, lw=0.9, alpha=0.9)
    ax.fill_between(bt.index, evrp, 0, where=(evrp > 0), color=GREEN, alpha=0.2, label="eVRP>0 (short edge)")
    ax.fill_between(bt.index, evrp, 0, where=(evrp <= 0), color=RED,   alpha=0.2, label="eVRP≤0 (no short)")
    ax.axhline(0, color=GRID, lw=0.6)
    ax.set_ylabel("Vol pts", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)

    # 3. Position
    ax = axes[2]; _style(ax, "Position  (negative = short vol via SVXY, positive = long vol via VXX)")
    ax.fill_between(bt.index, bt["held_pos"], where=(bt["held_pos"] < -0.01),
                    color=GREEN, alpha=0.8, step="post", label="Short vol (SVXY)")
    ax.fill_between(bt.index, bt["held_pos"], where=(bt["held_pos"] >  0.01),
                    color=RED,   alpha=0.8, step="post", label="Long vol (VXX)")
    ax.axhline(0, color=GRID, lw=0.6)
    ax.set_ylabel("Position", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)

    # 4. Daily P&L
    ax = axes[3]; _style(ax, "Daily Net P&L")
    colors = [GREEN if v >= 0 else RED for v in bt["net_pnl"].fillna(0)]
    ax.bar(bt.index, bt["net_pnl"].fillna(0) * 100, color=colors, width=1.5, alpha=0.85)
    ax.axhline(0, color=GRID, lw=0.6)
    ax.set_ylabel("Return (%)", color=TEXT)

    # 5. Equity
    ax = axes[4]; _style(ax, "Equity Curve vs SPY Buy-and-Hold")
    ax.plot(bt.index, bt["equity"],     color=ORANGE, lw=1.5, label="Strategy", zorder=3)
    ax.plot(bt.index, bt["spy_equity"], color=BLUE,   lw=1.0, alpha=0.7, label="SPY", zorder=2)
    ax.fill_between(bt.index, bt["peak"], bt["equity"], color=RED, alpha=0.2, label="Drawdown")
    ax.axhline(1.0, color=GRID, lw=0.8, ls="--")
    ax.set_ylabel("Equity (norm.)", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)

    # 6. Rolling Sharpe
    ax = axes[5]; _style(ax, "Rolling 252-day Sharpe")
    roll_sharpe = (bt["net_pnl"].rolling(252).mean() * 252 /
                   (bt["net_pnl"].rolling(252).std() * np.sqrt(252)).replace(0, np.nan))
    ax.plot(bt.index, roll_sharpe, color=YELLOW, lw=1.0)
    ax.axhline(0,   color=GRID,  lw=0.6, ls="--")
    ax.axhline(1.0, color=GREEN, lw=0.6, ls=":", alpha=0.5, label="Sharpe=1")
    ax.set_ylabel("Sharpe (252d)", color=TEXT)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)

    m = metrics
    footer = (f"CAGR: {m['CAGR']}   Sharpe: {m['Sharpe ratio']}   "
              f"Max DD: {m['Max drawdown']}   Sortino: {m['Sortino ratio']}   "
              f"Hit rate: {m['Hit rate']}   Corr(SPY): {m['Corr. with SPY']}")
    fig.text(0.5, 0.010, footer, ha="center", fontsize=9, color=MUTED,
             bbox=dict(facecolor=PANEL, edgecolor=GRID, boxstyle="round,pad=0.4"))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"\n  Chart saved -> {save_path}")
