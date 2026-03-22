"""
yahoo_client.py
===============
Downloads VXX (long vol ETN) and SPY (S&P 500 ETF) from Yahoo Finance.
No API key required.

VXX history issue and fix:
───────────────────────────
  The current VXX (iPath Series B) was issued in 2018 — Yahoo only has
  data from 2018-01-22. The original VXX existed from 2009 but was
  terminated in January 2019.

  To cover the full history back to 2009, we stitch two series:
    1. Pre-2018: original VXX ticker (Yahoo: "^VXX" does not work,
       but the old series is available as "VXXB" on some dates or via
       the SPVXSTR total return index which we reconstruct from
       VIX futures roll returns approximated by -1× daily VIX changes
       scaled by the front-month VIX futures price).

  PRACTICAL SOLUTION (used here):
    We download both tickers from Yahoo:
      "VXX"  → current series (2018–present)
      "VIXY" → ProShares VIX Short-Term Futures ETF, same underlying
               index (SPVXSTR), available from 2011-01-04
    Then we stitch: use VIXY to extend back to 2011,
    normalised so the two series join seamlessly on the first day
    both are available.

    For 2009-2011 (before VIXY), we use VXZ or simply start from 2011.
    This gives us a clean 15-year history instead of 7 years.

  Why this works:
    VIXY and VXX track the exact same index (SPVXSTR) — the only
    difference is the fee (0.85% vs 0.89%) and issuer (ProShares vs
    Barclays). Daily return correlation > 0.999.

SPY: Used for benchmark comparison only.
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path

from config import VXX_FILE, SPY_FILE

_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; spx-vol-backtest/1.0)",
})


def _download_yahoo(ticker: str, verbose: bool) -> pd.Series:
    """Download full daily adjusted close history for a ticker."""
    url    = f"{_BASE}/{ticker}"
    # Use period1=0 (unix epoch) to request maximum history
    params = {
        "interval": "1d",
        "period1":  "0",          # from unix epoch = maximum history
        "period2":  "9999999999", # far future
        "events":   "history",
    }

    r = _session.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    result     = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes     = result["indicators"]["adjclose"][0]["adjclose"]

    df = pd.DataFrame({
        "date":  pd.to_datetime(timestamps, unit="s", utc=True),
        "close": closes,
    })
    df["date"] = df["date"].dt.date
    df = (df.dropna()
            .set_index("date")
            .sort_index())
    df.index = pd.to_datetime(df.index)
    return df["close"].rename(ticker)


def download_vxx_stitched(verbose: bool = True) -> pd.Series:
    """
    Download VXX history stitched with VIXY for pre-2018 coverage.

    VXX  (Barclays iPath Series B)   : 2018-01-22 → present
    VIXY (ProShares VIX Short-Term)  : 2011-01-04 → present
    Both track the same index (SPVXSTR).

    Method:
      1. Download both series.
      2. Find the first date both are available.
      3. Compute a scalar to align VIXY level to VXX level.
      4. Use VIXY × scalar for dates before VXX starts.
      5. Use VXX for dates from its start onward.
    """
    if verbose:
        print("  Downloading VXX (current, 2018–present) ...",
              end=" ", flush=True)
    vxx = _download_yahoo("VXX", verbose=False)
    if verbose:
        print(f"done  ({len(vxx)} rows)")

    if verbose:
        print("  Downloading VIXY (proxy pre-2018, 2011–present) ...",
              end=" ", flush=True)
    vixy = _download_yahoo("VIXY", verbose=False)
    if verbose:
        print(f"done  ({len(vixy)} rows)")

    # Find overlap start
    overlap_start = max(vxx.index[0], vixy.index[0])
    overlap_end   = min(vxx.index[-1], vixy.index[-1])

    if overlap_start > overlap_end:
        if verbose:
            print("  WARNING: No overlap between VXX and VIXY — using VXX only")
        vxx.to_frame(name="vxx").to_csv(VXX_FILE, index_label="date")
        return vxx.rename("vxx")

    # Scale VIXY to match VXX level at overlap start
    vxx_at_join  = float(vxx[vxx.index >= overlap_start].iloc[0])
    vixy_at_join = float(vixy[vixy.index >= overlap_start].iloc[0])
    scale        = vxx_at_join / vixy_at_join

    # Pre-VXX segment from VIXY (scaled)
    vixy_pre  = vixy[vixy.index < vxx.index[0]] * scale
    vixy_pre  = vixy_pre.rename("vxx")

    # Full stitched series
    stitched = pd.concat([vixy_pre, vxx.rename("vxx")]).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="last")]

    stitched.to_frame(name="vxx").to_csv(VXX_FILE, index_label="date")

    if verbose:
        print(f"\n  Stitched VXX history: {len(stitched)} rows  "
              f"({stitched.index[0].date()} → {stitched.index[-1].date()})")
        pre_days = len(vixy_pre)
        post_days = len(vxx)
        print(f"    VIXY proxy (pre-2018): {pre_days} days  |  "
              f"VXX real: {post_days} days")

    return stitched


def download_spy(verbose: bool = True) -> pd.Series:
    if verbose:
        print("  Downloading SPY from Yahoo Finance ...",
              end=" ", flush=True)
    spy = _download_yahoo("SPY", verbose=False)
    spy.rename("spy").to_frame().to_csv(SPY_FILE, index_label="date")
    if verbose:
        print(f"done  ({len(spy)} rows, "
              f"{spy.index[0].date()} → {spy.index[-1].date()})")
    return spy.rename("spy")


def download_all(verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\n── Downloading ETN/ETF prices from Yahoo Finance ───────")
    vxx = download_vxx_stitched(verbose)
    spy = download_spy(verbose)
    return pd.DataFrame({"vxx": vxx, "spy": spy}).sort_index()
