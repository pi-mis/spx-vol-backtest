"""
main.py
=======
S&P 500 Volatility Strategy — Nelson-Siegel Term Structure Forecasting.

Based on:
  Chen et al. (2018), "Forecasting the Term Structure of Option Implied
  Volatility: The Power of an Adaptive Method", Journal of Empirical Finance.
  + Diebold-Li (2006) Nelson-Siegel parameterisation.
  + Zarattini, Mele & Aziz (2025) term structure + dynamic sizing framework.

Strategy:
  1. Fit Nelson-Siegel daily to VIX term structure (VIX9D, VIX, VIX3M, VIX6M)
  2. Extract β₁, β₂, β₃ time series
  3. Rolling ARIMA on each β → 1-step-ahead forecast
  4. Reconstruct predicted 30-day VIX from forecasted βs
  5. Signal: compare forecast vs spot + VIX term structure slope
  6. Execute via VXX (long vol ETN) with dynamic sizing

Data sources (all free, no API key):
  CBOE public CSV: VIX9D, VIX, VIX3M, VIX6M (back to 2007)
  Yahoo Finance:   VXX, SPY

Usage:
  python main.py            → full backtest
  python main.py --noplot   → backtest without chart
"""

import sys
import textwrap
import pandas as pd

from config         import BACKTEST_START, MATURITIES, SIGNAL_THRESHOLD
from cboe_client    import download_all as download_cboe
from yahoo_client   import download_all as download_yahoo
from nelson_siegel  import fit_ns_panel
from arima_forecaster import rolling_beta_forecasts, reconstruct_vix_forecast
from signals        import generate_signals, compute_evrp
from backtest       import run_backtest, compute_metrics, plot_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(t): print("\n" + "═"*66 + f"\n  {t}\n" + "═"*66)
def _step(n, tot, t): print(f"\n[Step {n}/{tot}]  {t}\n" + "─"*50)
def _note(t): [print(f"  {l}") for l in textwrap.wrap(t, 64)]; print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(plot: bool = True):
    _banner("S&P 500 Vol Strategy — Nelson-Siegel Term Structure Backtest")

    from config import (ARIMA_TRAIN_WINDOW, ARIMA_REFIT_EVERY,
                        SIZING_DIVISOR, REBAL_THRESHOLD, TRANSACTION_COST_PCT)
    print(f"""
  Configuration:
    Data start        : {BACKTEST_START}
    NS maturities     : VIX9D (9d), VIX (30d), VIX3M (90d), VIX6M (180d)
    ARIMA window      : {ARIMA_TRAIN_WINDOW} days
    ARIMA refit       : every {ARIMA_REFIT_EVERY} days
    Signal threshold  : ±{SIGNAL_THRESHOLD} vol pts
    Sizing divisor    : VIX / {SIZING_DIVISOR}
    Rebal threshold   : ±{REBAL_THRESHOLD*100:.0f}%
    Transaction cost  : {TRANSACTION_COST_PCT*10000:.0f} bps per trade
    """)

    # ── Step 1: Download data ─────────────────────────────────────────────────
    _step(1, 6, "Download VIX term structure from CBOE")
    _note(
        "Four VIX indices published daily by CBOE — free, no account needed. "
        "These are the 4 points of the implied vol term structure: "
        "VIX9D (9d), VIX (30d), VIX3M (90d), VIX6M (180d). "
        "History: VIX from 1990, VIX3M/VIX6M from 2007, VIX9D from 2014."
    )
    vix_ts = download_cboe(verbose=True)

    _step(1, 6, "Download VXX and SPY from Yahoo Finance")
    _note(
        "VXX is the iPath S&P 500 VIX Short-Term Futures ETN — "
        "the liquid instrument we trade. Long VXX = long vol. "
        "SPY is used as benchmark."
    )
    price_df = download_yahoo(verbose=True)

    # ── Step 2: Align data ────────────────────────────────────────────────────
    _step(2, 6, "Align and filter data")
    combined = vix_ts.join(price_df, how="inner").dropna(
        subset=["vix", "vix3m"]  # require at least VIX + VIX3M
    )
    combined = combined[combined.index >= pd.Timestamp(BACKTEST_START)]

    print(f"  Combined dataset: {len(combined)} trading days")
    print(f"  Date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    vix9d_from = combined["vix9d"].first_valid_index()
    print(f"  VIX9D available from: {vix9d_from.date() if vix9d_from else 'N/A'}")

    # ── Step 3: Nelson-Siegel fitting ─────────────────────────────────────────
    _step(3, 6, "Fit Nelson-Siegel on each daily term structure")
    _note(
        "OLS fit of IV(τ) = β₁ + β₂·L(τ) + β₃·C(τ) for each day. "
        "λ is optimised via Brent's method to minimise RMSE. "
        "Days with fewer than 3 valid quotes or R² < 0.5 are skipped. "
        "This extracts the 3-factor representation of the vol curve each day."
    )
    betas_df = fit_ns_panel(
        term_structure=combined[list(MATURITIES.keys())],
        maturities=MATURITIES,
        verbose=True,
    )

    # ── Step 4: ARIMA rolling forecasts on βs ────────────────────────────────
    _step(4, 6, "Rolling ARIMA walk-forward on β₁, β₂, β₃")
    _note(
        "For each day t (starting at day 120), we: "
        "(1) fit best ARIMA(p,d,q) on trailing 120 days of each β, "
        "(2) produce 1-step-ahead forecast, "
        "(3) advance window by 1 day. "
        "No look-ahead: forecast at t uses only data from days 0…t-1. "
        "ARIMA spec is selected by AIC; d is set by ADF test."
    )
    forecasts_df = rolling_beta_forecasts(betas_df, verbose=True)

    # Use the fixed lambda from the NS panel (all rows have the same value)
    median_lam = float(betas_df["lam"].iloc[0])
    print(f"\n  λ = {median_lam} (fixed, from NS panel)")

    vix_forecast = reconstruct_vix_forecast(forecasts_df, median_lam)

    # ── Step 5: Generate signals ──────────────────────────────────────────────
    _step(5, 6, "Generate trading signals")
    _note(
        f"Three filters combined: "
        f"(1) NS spread < -{SIGNAL_THRESHOLD} → forecast predicts vol falling. "
        f"(2) VIX < VIX3M → contango confirms normal market. "
        f"(3) eVRP = VIX − 10d realized vol of SPY > 0 → premium exists. "
        f"All three must agree for a full short. "
        f"Long vol only when spread > +{SIGNAL_THRESHOLD}, backwardation, and eVRP ≤ 0. "
        f"Execution: short vol via SVXY, long vol via VXX."
    )
    signals_df = generate_signals(
        vix=combined["vix"],
        vix3m=combined["vix3m"],
        vix_forecast=vix_forecast,
        spy_prices=combined["spy"],
        verbose=True,
    )

    # ── Step 6: Backtest ──────────────────────────────────────────────────────
    _step(6, 6, "Backtest on VXX returns")
    _note(
        "P&L per day = held_position_{t-1} × VXX_return_t. "
        "Returns are compounded (equity reinvested). "
        "Transaction cost = 5 bps per trade (same as Zarattini et al.)."
    )
    bt = run_backtest(
        signals_df=signals_df,
        combined=combined,
        verbose=True,
    )
    bt["vix_forecast"] = vix_forecast.loc[~vix_forecast.index.duplicated(keep="last")]
    bt["evrp"]         = signals_df["evrp"]

    metrics = compute_metrics(bt)

    # Print results table
    print()
    W = 32
    print("  ┌" + "─"*(W+24) + "┐")
    print(f"  │  {'Metric':<{W}} {'Value':>20}  │")
    print("  ├" + "─"*(W+24) + "┤")
    for k, v in metrics.items():
        if str(v).startswith("──"):
            print("  ├" + "─"*(W+24) + "┤")
            print(f"  │  {k:<{W}} {str(v):>20}  │")
        else:
            print(f"  │  {k:<{W}} {str(v):>20}  │")
    print("  └" + "─"*(W+24) + "┘")

    if plot:
        print("\n  Generating chart → backtest_results.png ...")
        plot_results(bt, metrics, save_path="backtest_results.png")

    print("\n" + "✓ " * 22)


if __name__ == "__main__":
    plot = "--noplot" not in sys.argv
    main(plot=plot)
