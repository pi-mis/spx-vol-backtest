"""
signals.py
==========
Three signals combined:
  1. NS+ARIMA spread: forecasted VIX vs current VIX
  2. Term structure:  VIX vs VIX3M (contango / backwardation)
  3. eVRP filter:     VIX vs 10-day realized vol of SPY

Signal logic:
  Short SVXY (full): spread < -t  AND  contango      AND  eVRP > 0
  Short SVXY (half): spread < -t  AND  backwardation AND  eVRP > 0
  Long  VXX  (full): spread > +t  AND  backwardation AND  eVRP <= 0
  Flat: everything else (no long in contango; no short without premium)
"""

import numpy as np
import pandas as pd
from config import SIGNAL_THRESHOLD, SIZING_DIVISOR, REBAL_THRESHOLD


def compute_evrp(spy_prices: pd.Series, vix: pd.Series, window: int = 10) -> pd.Series:
    """eVRP = VIX - annualised 10-day realized vol of SPY."""
    log_ret = np.log(spy_prices / spy_prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252) * 100
    return (vix - rv).rename("evrp")


def generate_signals(
    vix: pd.Series,
    vix3m: pd.Series,
    vix_forecast: pd.Series,
    spy_prices: pd.Series,
    verbose: bool = True,
) -> pd.DataFrame:
    # Deduplicate
    vix          = vix.loc[~vix.index.duplicated(keep="last")]
    vix3m        = vix3m.loc[~vix3m.index.duplicated(keep="last")]
    vix_forecast = vix_forecast.loc[~vix_forecast.index.duplicated(keep="last")]
    spy_prices   = spy_prices.loc[~spy_prices.index.duplicated(keep="last")]

    evrp = compute_evrp(spy_prices, vix)

    df = pd.DataFrame({
        "vix":          vix,
        "vix3m":        vix3m,
        "vix_forecast": vix_forecast,
        "evrp":         evrp,
    }).dropna()

    df["spread"]   = df["vix_forecast"] - df["vix"]
    df["contango"] = df["vix"] < df["vix3m"]
    base_size      = df["vix"] / SIZING_DIVISOR

    raw_dir    = pd.Series(0.0, index=df.index)
    raw_size   = pd.Series(0.0, index=df.index)
    instrument = pd.Series("",  index=df.index, dtype=str)

    mask_sf = (df["spread"] < -SIGNAL_THRESHOLD) &  df["contango"] & (df["evrp"] > 0)
    mask_sh = (df["spread"] < -SIGNAL_THRESHOLD) & ~df["contango"] & (df["evrp"] > 0)
    mask_lf = (df["spread"] >  SIGNAL_THRESHOLD) & ~df["contango"] & (df["evrp"] <= 0)

    raw_dir[mask_sf]  = -1.0;  raw_size[mask_sf]  = base_size[mask_sf];      instrument[mask_sf]  = "svxy"
    raw_dir[mask_sh]  = -1.0;  raw_size[mask_sh]  = 0.5*base_size[mask_sh];  instrument[mask_sh]  = "svxy"
    raw_dir[mask_lf]  =  1.0;  raw_size[mask_lf]  = base_size[mask_lf];      instrument[mask_lf]  = "vxx"

    df["raw_direction"] = raw_dir
    df["raw_size"]      = raw_size
    df["raw_position"]  = raw_dir * raw_size
    df["instrument"]    = instrument

    # Rebalancing threshold
    actual_pos = df["raw_position"].copy()
    current = 0.0
    for i in range(len(actual_pos)):
        desired = float(df["raw_position"].iloc[i])
        if abs(desired - current) > REBAL_THRESHOLD:
            current = desired
        actual_pos.iloc[i] = current

    df["effective_pos"] = actual_pos
    df["direction"]     = np.sign(df["effective_pos"])
    df["size"]          = df["effective_pos"].abs()

    if verbose:
        n = len(df)
        n_sf  = int(mask_sf.sum())
        n_sh  = int(mask_sh.sum())
        n_lf  = int(mask_lf.sum())
        n_nep = int(((df["spread"] < -SIGNAL_THRESHOLD) & (df["evrp"] <= 0)).sum())
        n_lco = int(((df["spread"] >  SIGNAL_THRESHOLD) &  df["contango"]).sum())
        a_s   = int((df["direction"] == -1).sum())
        a_l   = int((df["direction"] ==  1).sum())
        a_f   = int((df["direction"] ==  0).sum())
        avg_z = df["size"][df["size"] > 0].mean() * 100

        print(f"\n  Raw signal breakdown  (threshold ±{SIGNAL_THRESHOLD}):")
        print(f"    Short SVXY full (spread<-t, contango, eVRP>0)  : {n_sf:4d}  ({n_sf/n*100:.1f}%)")
        print(f"    Short SVXY half (spread<-t, backw.,  eVRP>0)  : {n_sh:4d}  ({n_sh/n*100:.1f}%)")
        print(f"    Long  VXX  full (spread>+t, backw.,  eVRP<=0) : {n_lf:4d}  ({n_lf/n*100:.1f}%)")
        print(f"    FLAT — no premium (spread<-t, eVRP<=0)        : {n_nep:4d}  ({n_nep/n*100:.1f}%)  <- suppressed")
        print(f"    FLAT — carry     (spread>+t, contango)        : {n_lco:4d}  ({n_lco/n*100:.1f}%)  <- suppressed")
        print(f"\n  After rebalancing (+-{REBAL_THRESHOLD*100:.0f}%):")
        print(f"    Short vol (SVXY) : {a_s:4d}  ({a_s/n*100:.1f}%)")
        print(f"    Long  vol (VXX)  : {a_l:4d}  ({a_l/n*100:.1f}%)")
        print(f"    Flat             : {a_f:4d}  ({a_f/n*100:.1f}%)")
        print(f"    Avg active size  : {avg_z:.1f}% of capital")
        print(f"    eVRP positive    : {(df['evrp']>0).mean()*100:.1f}% of days")

    return df
