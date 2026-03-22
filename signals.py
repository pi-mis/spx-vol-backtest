"""
signals.py
==========
Combines the Nelson-Siegel forecast with the term structure signal
(VIX vs VIX3M) and dynamic sizing to produce daily trading signals.

Signal logic:
─────────────────────────────────────────────────────────────────
The strategy has two layers, matching the paper's progression:

  Layer 1 — NS forecast:
    vix_forecast_t = NS(β₁_fwd, β₂_fwd, β₃_fwd) at τ=30d
    spread = vix_forecast − vix_spot
    spread >  +threshold  →  expect vol to rise  →  LONG VXX
    spread <  −threshold  →  expect vol to fall  →  SHORT VXX

  Layer 2 — Term structure confirmation (VIX vs VIX3M):
    VIX < VIX3M  →  contango (normal)      →  short vol favoured
    VIX > VIX3M  →  backwardation (stress) →  long vol favoured

  Combined (4 cases):
    spread < −thresh AND contango      →  SHORT VXX  (full conviction)
    spread > +thresh AND backwardation →  LONG  VXX  (full conviction)
    spread < −thresh AND backwardation →  SHORT VXX  (half — conflicting)
    spread > +thresh AND contango      →  LONG  VXX  (half — conflicting)
    |spread| ≤ thresh                  →  FLAT

  Dynamic sizing: VIX / SIZING_DIVISOR
    Higher VIX → bigger position (premium richer, mean reversion stronger)
    Lower VIX  → smaller position (steamroller risk)

  Rebalancing: only trade when |new − old| > REBAL_THRESHOLD (2%)
─────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

from config import SIGNAL_THRESHOLD, SIZING_DIVISOR, REBAL_THRESHOLD


def generate_signals(
    vix:          pd.Series,
    vix3m:        pd.Series,
    vix_forecast: pd.Series,
    verbose:      bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    vix          : daily VIX closing values (%)
    vix3m        : daily VIX3M closing values (%)
    vix_forecast : 1-step-ahead VIX forecast from NS+ARIMA (%)
    verbose      : print signal breakdown

    Returns
    -------
    DataFrame with columns:
      vix, vix3m, vix_forecast, spread, contango,
      raw_direction, raw_size, raw_position, effective_pos, direction, size
    """
    # Remove duplicate dates before joining (can arise from VXX stitching)
    vix          = vix.loc[~vix.index.duplicated(keep="last")]
    vix3m        = vix3m.loc[~vix3m.index.duplicated(keep="last")]
    vix_forecast = vix_forecast.loc[~vix_forecast.index.duplicated(keep="last")]

    df = pd.DataFrame({
        "vix":          vix,
        "vix3m":        vix3m,
        "vix_forecast": vix_forecast,
    }).dropna()

    df["spread"]   = df["vix_forecast"] - df["vix"]
    df["contango"] = df["vix"] < df["vix3m"]

    base_size = df["vix"] / SIZING_DIVISOR

    raw_dir  = pd.Series(0.0, index=df.index)
    raw_size = pd.Series(0.0, index=df.index)

    # Case 1: forecast lower + contango → Short full conviction
    mask_short_full = (df["spread"] < -SIGNAL_THRESHOLD) & df["contango"]

    # Case 2: forecast lower + backwardation → Short half (conflicting)
    mask_short_half = (df["spread"] < -SIGNAL_THRESHOLD) & ~df["contango"]

    # Case 3: forecast higher + backwardation → Long full conviction
    # (term structure already inverted, model confirms stress → long vol)
    mask_long_full  = (df["spread"] >  SIGNAL_THRESHOLD) & ~df["contango"]

    # Case 4: forecast higher + contango → FLAT
    # The NS model predicts vol will rise but the term structure is normal
    # (contango). In practice, long VXX in contango is chronically loss-making
    # because the negative carry from rolling front-month futures erodes returns
    # faster than the rare vol spikes can compensate.
    # This is the key asymmetry vs the original code: we suppress long signals
    # in contango entirely, making the strategy correctly short-biased.
    # mask_long_half_contango → 0 (flat, not traded)

    raw_dir[mask_short_full]  = -1.0
    raw_size[mask_short_full] = base_size[mask_short_full]

    raw_dir[mask_short_half]  = -1.0
    raw_size[mask_short_half] = 0.5 * base_size[mask_short_half]

    raw_dir[mask_long_full]   =  1.0
    raw_size[mask_long_full]  = base_size[mask_long_full]

    # Case 4 (spread > +t AND contango) → stays at 0 (flat) by default

    df["raw_direction"] = raw_dir
    df["raw_size"]      = raw_size
    df["raw_position"]  = raw_dir * raw_size

    # Rebalancing threshold
    actual_pos   = df["raw_position"].copy()
    current_held = 0.0
    for i in range(len(actual_pos)):
        desired = float(df["raw_position"].iloc[i])
        if abs(desired - current_held) > REBAL_THRESHOLD:
            current_held = desired
        actual_pos.iloc[i] = current_held

    df["effective_pos"] = actual_pos
    df["direction"]     = np.sign(df["effective_pos"])
    df["size"]          = df["effective_pos"].abs()

    if verbose:
        n = len(df)
        n_sf   = mask_short_full.reindex(df.index, fill_value=False).sum()
        n_sh   = mask_short_half.reindex(df.index, fill_value=False).sum()
        n_lf   = mask_long_full.reindex(df.index, fill_value=False).sum()
        n_lhc  = ((df["spread"] > SIGNAL_THRESHOLD) & df["contango"]).sum()
        n_flat = (df["raw_position"] == 0).sum()

        print(f"\n  Raw signal breakdown  (threshold=±{SIGNAL_THRESHOLD}):")
        print(f"    Short full  (spread<-t, contango)     : "
              f"{n_sf:4d}  ({n_sf/n*100:.1f}%)")
        print(f"    Short half  (spread<-t, backwardation): "
              f"{n_sh:4d}  ({n_sh/n*100:.1f}%)")
        print(f"    Long  full  (spread>+t, backwardation): "
              f"{n_lf:4d}  ({n_lf/n*100:.1f}%)")
        print(f"    FLAT        (spread>+t, contango)     : "
              f"{n_lhc:4d}  ({n_lhc/n*100:.1f}%)  ← suppressed (carry)")

        after_short = (df["direction"] == -1).sum()
        after_long  = (df["direction"] ==  1).sum()
        after_flat  = (df["direction"] ==  0).sum()
        avg_sz = df["size"][df["size"] > 0].mean() * 100

        print(f"\n  After rebalancing threshold (±{REBAL_THRESHOLD*100:.0f}%):")
        print(f"    Short VXX : {after_short:4d}  ({after_short/n*100:.1f}%)")
        print(f"    Long  VXX : {after_long:4d}  ({after_long/n*100:.1f}%)")
        print(f"    Flat      : {after_flat:4d}  ({after_flat/n*100:.1f}%)")
        print(f"    Avg size  : {avg_sz:.1f}% of capital")

    return df
