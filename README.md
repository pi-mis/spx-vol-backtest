# S&P 500 Volatility Strategy
## Nelson-Siegel Term Structure Forecasting

Systematic backtest of a volatility trading strategy based on forecasting
the implied volatility term structure using the Nelson-Siegel / Diebold-Li model.

---

### Strategy overview

1. **Download** four VIX term structure indices from CBOE (free):
   `VIX9D` (9d), `VIX` (30d), `VIX3M` (90d), `VIX6M` (180d)

2. **Fit Nelson-Siegel** daily — extract β₁ (level), β₂ (slope), β₃ (curvature)

3. **Rolling ARIMA** on each β → 1-step-ahead forecast of tomorrow's vol curve

4. **Signal**: compare forecasted 30d VIX vs spot VIX  
   + confirm with VIX term structure slope (contango / backwardation)

5. **Execute** via VXX (long vol ETN). Size = VIX/100 (dynamic sizing).

---

### Based on

- Chen, Y. et al. (2018). *Forecasting the Term Structure of Option Implied Volatility*. Journal of Empirical Finance.
- Nelson, C. & Siegel, A. (1987). *Parsimonious Modeling of Yield Curves*. Journal of Business.
- Diebold, F. & Li, C. (2006). *Forecasting the Term Structure of Government Bond Yields*. Journal of Econometrics.
- Zarattini, C., Mele, A. & Aziz, A. (2025). *The Volatility Edge*. Concretum Research.

---

### Data sources (all free, no API key)

| Data | Source | History |
|------|--------|---------|
| VIX9D, VIX, VIX3M, VIX6M | CBOE public CSV | From 2007 |
| VXX (trading instrument) | Yahoo Finance | From 2009 |
| SPY (benchmark) | Yahoo Finance | From 1993 |

---

### Quickstart

```bash
pip install -r requirements.txt
python main.py
```

Runtime: ~5–10 minutes (ARIMA rolling on ~17 years of data).

---

### Files

```
main.py             ← entry point
config.py           ← all parameters
cboe_client.py      ← downloads VIX term structure from CBOE
yahoo_client.py     ← downloads VXX and SPY from Yahoo Finance
nelson_siegel.py    ← NS model fitting (OLS + λ optimisation)
arima_forecaster.py ← rolling ARIMA on β₁, β₂, β₃
signals.py          ← signal generation (NS forecast + term structure)
backtest.py         ← P&L simulation, metrics, 6-panel chart

data/               ← created on first run (in .gitignore)
```

---

### Key parameters (`config.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ARIMA_TRAIN_WINDOW` | 120 | Days of history per ARIMA fit |
| `SIGNAL_THRESHOLD` | 2.0 | Min |forecast − spot| to trade |
| `SIZING_DIVISOR` | 100 | VIX/100 = position size |
| `REBAL_THRESHOLD` | 0.02 | 2% band before rebalancing |
| `TRANSACTION_COST_PCT` | 0.0005 | 5 bps per trade |
