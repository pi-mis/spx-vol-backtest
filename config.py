"""
config.py
=========
Single source of truth for all parameters.
"""

from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Data files ────────────────────────────────────────────────────────────────
VIX9D_FILE  = DATA_DIR / "vix9d.csv"
VIX_FILE    = DATA_DIR / "vix.csv"
VIX3M_FILE  = DATA_DIR / "vix3m.csv"
VIX6M_FILE  = DATA_DIR / "vix6m.csv"
VXX_FILE    = DATA_DIR / "vxx.csv"
SPY_FILE    = DATA_DIR / "spy.csv"

# ── Term structure maturities (years) ─────────────────────────────────────────
# Matches: VIX9D=9d, VIX=30d, VIX3M=90d, VIX6M=180d
MATURITIES = {
    "vix9d": 9   / 365.25,
    "vix":   30  / 365.25,
    "vix3m": 90  / 365.25,
    "vix6m": 180 / 365.25,
}

# ── Nelson-Siegel ─────────────────────────────────────────────────────────────
NS_LAMBDA_BOUNDS = (0.5, 50.0)   # search range for λ

# ── ARIMA rolling forecasting ─────────────────────────────────────────────────
ARIMA_TRAIN_WINDOW = 120          # days of history per fit
ARIMA_REFIT_EVERY  = 5            # refit ARIMA spec every N days
ARIMA_MAX_P        = 3            # max AR order to search
ARIMA_MAX_Q        = 3            # max MA order to search

# ── Signal ────────────────────────────────────────────────────────────────────
# Forecast horizon: we predict the 30-day VIX (= standard VIX maturity)
FORECAST_MATURITY  = 30 / 365.25

# Minimum |forecasted_VIX − current_VIX| to open a position (vol pts)
SIGNAL_THRESHOLD   = 2.0

# Dynamic sizing: VIX / SIZING_DIVISOR = position size fraction
SIZING_DIVISOR     = 100

# Rebalancing band (2% of capital — same as Zarattini et al.)
REBAL_THRESHOLD    = 0.02

# ── Backtest ──────────────────────────────────────────────────────────────────
# Transaction costs in % of trade value (0.05% = 5 bps, same as the paper)
TRANSACTION_COST_PCT = 0.0005

# Start date for backtest.
# SVXY was restructured from -1x to -0.5x on March 5, 2018 after Volmageddon.
# Starting from 2019-01-01 ensures:
#   - SVXY is consistently -0.5x for the entire backtest period
#   - Includes COVID 2020 and the 2022 bear market as real stress tests
#   - All P&L uses real prices, no proxies, no structural breaks
BACKTEST_START = "2019-01-01"
