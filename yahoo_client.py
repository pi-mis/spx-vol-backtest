"""
yahoo_client.py
===============
Downloads VXX (long vol ETN) and SPY (S&P 500 ETF) from Yahoo Finance.
No API key required.

VXX: iPath Series B S&P 500 VIX Short-Term Futures ETN
  Tracks a rolling long position in front-month VIX futures.
  Going LONG VXX = going long volatility.
  Going SHORT VXX = going short volatility (harvesting VRP).
  Available from 2018 (reissued); pre-2018 we use the original VXX.

SPY: Used for benchmark comparison only.

We use Yahoo Finance's public download endpoint:
  https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}
"""

import requests
import pandas as pd
from pathlib import Path

from config import VXX_FILE, SPY_FILE

_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; spx-vol-backtest/1.0)",
})


def _download_yahoo(ticker: str, path: Path, verbose: bool) -> pd.Series:
    """Download full daily history for a ticker from Yahoo Finance."""
    if verbose:
        print(f"  Downloading {ticker} from Yahoo Finance ...",
              end=" ", flush=True)

    url = f"{_BASE}/{ticker}"
    params = {
        "interval":  "1d",
        "range":     "max",
        "events":    "history",
    }

    r = _session.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    result    = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes     = result["indicators"]["adjclose"][0]["adjclose"]

    df = pd.DataFrame({
        "date":  pd.to_datetime(timestamps, unit="s", utc=True),
        ticker:  closes,
    })
    df["date"] = df["date"].dt.date
    df = (df.dropna()
            .set_index("date")
            .sort_index())
    df.index = pd.to_datetime(df.index)

    df.to_csv(path, index_label="date")

    if verbose:
        print(f"done  ({len(df)} rows, "
              f"{df.index[0].date()} → {df.index[-1].date()})")

    return df[ticker]


def download_vxx(verbose: bool = True) -> pd.Series:
    return _download_yahoo("VXX", VXX_FILE, verbose)


def download_spy(verbose: bool = True) -> pd.Series:
    return _download_yahoo("SPY", SPY_FILE, verbose)


def download_all(verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\n── Downloading ETN/ETF prices from Yahoo Finance ───────")
    vxx = download_vxx(verbose)
    spy = download_spy(verbose)
    return pd.DataFrame({"vxx": vxx, "spy": spy}).sort_index()
