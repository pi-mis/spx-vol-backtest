"""
arima_forecaster.py
===================
Rolling walk-forward ARIMA forecasting on the Nelson-Siegel β time series.

Each β (β₁, β₂, β₃) is modelled independently as ARIMA(p, d, q).
We search the grid of (p, d, q) candidates on a trailing window and
pick the spec with the lowest AIC, then produce a 1-step-ahead forecast.

The 1-step-ahead forecasts of β₁, β₂, β₃ are then plugged back into
the Nelson-Siegel formula to reconstruct the predicted VIX at 30 days —
the core forecasting signal of the strategy.

No-look-ahead discipline:
  The forecast for day t uses ONLY data from days 0 … t-1.
  The ARIMA model is fit on a rolling window ending at t-1.
  This mirrors exactly what would have been possible in real time.
"""

import warnings
import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from config import (
    ARIMA_TRAIN_WINDOW, ARIMA_REFIT_EVERY,
    ARIMA_MAX_P, ARIMA_MAX_Q, FORECAST_MATURITY,
)
from nelson_siegel import NSFit, predict_iv


# ── Stationarity test ─────────────────────────────────────────────────────────

def _needs_diff(series: pd.Series, alpha: float = 0.05) -> bool:
    """ADF test — True if series is non-stationary (needs d=1)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_value, *_ = adfuller(series.dropna(), autolag="AIC")
    return p_value > alpha


# ── Best-AIC ARIMA ────────────────────────────────────────────────────────────

def _best_arima(series: pd.Series) -> ARIMA:
    """Fit ARIMA candidates and return the fitted model with lowest AIC."""
    d = 1 if _needs_diff(series) else 0

    best_aic, best_model = np.inf, None
    for p, q in product(range(ARIMA_MAX_P + 1), range(ARIMA_MAX_Q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(series, order=(p, d, q)).fit()
            if m.aic < best_aic:
                best_aic, best_model = m.aic, m
        except Exception:
            continue

    if best_model is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model = ARIMA(series, order=(1, d, 0)).fit()

    return best_model


# ── Rolling beta forecasts ────────────────────────────────────────────────────

def rolling_beta_forecasts(
    betas_df: pd.DataFrame,
    verbose:  bool = True,
) -> pd.DataFrame:
    """
    Produce rolling 1-step-ahead forecasts for β₁, β₂, β₃.

    Parameters
    ----------
    betas_df : DataFrame with columns [beta1, beta2, beta3], daily index.
               Output of nelson_siegel.fit_ns_panel().

    Returns
    -------
    DataFrame with columns [beta1_fwd, beta2_fwd, beta3_fwd]
    indexed by the prediction date (= day after last training obs).
    Length = len(betas_df) − ARIMA_TRAIN_WINDOW.
    """
    betas  = betas_df[["beta1", "beta2", "beta3"]].dropna()
    dates  = betas.index.tolist()
    n      = len(dates)
    w      = ARIMA_TRAIN_WINDOW
    every  = ARIMA_REFIT_EVERY

    if verbose:
        print(f"\n  Rolling ARIMA  |  window={w}d  "
              f"|  refit every {every}d  "
              f"|  forecasting {n - w} days")
        print(f"  {'%':>5}  {'Date':>12}  "
              f"{'β₁ fwd':>9}  {'β₂ fwd':>9}  {'β₃ fwd':>9}")
        print("  " + "─" * 50)

    results   = []
    models    = {c: None for c in ["beta1", "beta2", "beta3"]}
    last_fit  = {c: -999  for c in ["beta1", "beta2", "beta3"]}

    for i in range(w, n):
        pred_date   = dates[i]
        train_slice = betas.iloc[i - w : i]
        row         = {"date": pred_date}

        for col in ["beta1", "beta2", "beta3"]:
            if i - last_fit[col] >= every or models[col] is None:
                models[col]   = _best_arima(train_slice[col])
                last_fit[col] = i

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fc = float(models[col].forecast(steps=1).iloc[0])
            except Exception:
                fc = float(train_slice[col].iloc[-1])   # naive fallback

            row[f"{col}_fwd"] = fc

        results.append(row)

        if verbose and (i % 50 == 0 or i == n - 1):
            pct = (i - w + 1) / (n - w) * 100
            print(f"  {pct:4.0f}%  {pred_date.date()}  "
                  f"{row['beta1_fwd']:>9.3f}  "
                  f"{row['beta2_fwd']:>9.3f}  "
                  f"{row['beta3_fwd']:>9.3f}")

    if not results:
        raise ValueError("No forecasts produced — need more data.")

    df = pd.DataFrame(results).set_index("date").sort_index()

    if verbose:
        print(f"\n  ✓ {len(df)} beta forecasts generated")

    return df


# ── Reconstruct forecasted VIX30 ─────────────────────────────────────────────

def reconstruct_vix_forecast(
    beta_forecasts: pd.DataFrame,
    median_lam:     float,
) -> pd.Series:
    """
    Plug forecasted βs into the NS formula to get predicted VIX at 30 days.

    Parameters
    ----------
    beta_forecasts : output of rolling_beta_forecasts()
    median_lam     : median λ from the NS panel (use betas_df['lam'].median())

    Returns pd.Series named 'vix_forecast'.
    """
    ivs = []
    for _, row in beta_forecasts.iterrows():
        ns = NSFit(
            beta1=row["beta1_fwd"],
            beta2=row["beta2_fwd"],
            beta3=row["beta3_fwd"],
            lam=median_lam,
            r2=np.nan, rmse=np.nan, n_points=0,
        )
        ivs.append(float(predict_iv(FORECAST_MATURITY, ns)[0]))

    return pd.Series(ivs, index=beta_forecasts.index, name="vix_forecast")
