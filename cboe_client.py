"""
cboe_client.py
==============
Downloads the four VIX term structure indices directly from CBOE.
No API key, no account — all public.

Indices and their maturities:
  VIX9D  →   9 days  (since 2014)
  VIX    →  30 days  (since 1990)
  VIX3M  →  90 days  (since 2007)
  VIX6M  → 180 days  (since 2007)

These four points are the term structure we feed into Nelson-Siegel.

CBOE URL format (verified March 2026):
  https://cdn.cboe.com/api/global/us_indices/daily_prices/{INDEX}_History.csv

Each CSV has columns: DATE, OPEN, HIGH, LOW, CLOSE
We keep only DATE and CLOSE, renamed to 'date' and the index name.
"""

import requests
import pandas as pd
import io
from pathlib import Path

from config import VIX9D_FILE, VIX_FILE, VIX3M_FILE, VIX6M_FILE

_BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices"

_INDICES = {
    "vix9d": (f"{_BASE}/VIX9D_History.csv", VIX9D_FILE),
    "vix":   (f"{_BASE}/VIX_History.csv",   VIX_FILE),
    "vix3m": (f"{_BASE}/VIX3M_History.csv", VIX3M_FILE),
    "vix6m": (f"{_BASE}/VIX6M_History.csv", VIX6M_FILE),
}

_session = requests.Session()
_session.headers.update({"User-Agent": "spx-vol-backtest/1.0"})


def _download_one(name: str, url: str, path: Path, verbose: bool) -> pd.Series:
    """Download one VIX index CSV and save it locally."""
    if verbose:
        print(f"  Downloading {name.upper()} from CBOE ...", end=" ", flush=True)

    r = _session.get(url, timeout=20)
    r.raise_for_status()

    # CBOE sometimes includes a header comment line starting with '"'
    # We skip rows until we find the one starting with DATE
    lines = r.text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("DATE"))
    csv_text = "\n".join(lines[start:])

    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip().upper() for c in df.columns]

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = (df[["DATE", "CLOSE"]]
            .rename(columns={"DATE": "date", "CLOSE": name})
            .set_index("date")
            .sort_index()
            .dropna())

    # Save
    df.to_csv(path, index_label="date")

    if verbose:
        print(f"done  ({len(df)} rows, "
              f"{df.index[0].date()} → {df.index[-1].date()})")

    return df[name]


def download_all(verbose: bool = True) -> pd.DataFrame:
    """
    Download all four VIX indices from CBOE and return a merged DataFrame.

    Columns: vix9d, vix, vix3m, vix6m
    Index:   date (daily)

    Rows are kept only where all four indices have data.
    VIX9D starts in 2014, so the full 4-column DataFrame begins then.
    For 2007-2014 we still keep rows with VIX, VIX3M, VIX6M available
    — Nelson-Siegel works with 3 points too.
    """
    if verbose:
        print("\n── Downloading VIX term structure from CBOE ────────────")

    series = {}
    for name, (url, path) in _INDICES.items():
        series[name] = _download_one(name, url, path, verbose)

    df = pd.DataFrame(series).sort_index()

    # Forward-fill small gaps (holidays, missing days) up to 3 days
    df = df.ffill(limit=3)

    if verbose:
        full = df.dropna()
        print(f"\n  Full 4-index overlap: {len(full)} days "
              f"({full.index[0].date()} → {full.index[-1].date()})")

    return df
