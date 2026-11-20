"""
Step 1 - Data loading.

Two regimes (see README):
  * download : fetch fresh data from yfinance and FRED, then cache it locally
  * file     : read the cached files only (fast, reproducible, no internet needed)
"""
import io
import os
import time

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Storage helpers: parquet if pyarrow is installed, otherwise compressed CSV
# ---------------------------------------------------------------------------
def save_df(df: pd.DataFrame, path_no_ext: str) -> str:
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    try:
        path = path_no_ext + ".parquet"
        df.to_parquet(path, index=False)
    except ImportError:
        path = path_no_ext + ".csv.gz"
        df.to_csv(path, index=False)
    return path


def load_df(path_no_ext: str) -> pd.DataFrame:
    if os.path.exists(path_no_ext + ".parquet"):
        df = pd.read_parquet(path_no_ext + ".parquet")
    elif os.path.exists(path_no_ext + ".csv.gz"):
        df = pd.read_csv(path_no_ext + ".csv.gz")
    else:
        raise FileNotFoundError(f"No cached file for {path_no_ext}. Run with --source download first.")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------------------------------------------------------------------
# Prices (yfinance)
# ---------------------------------------------------------------------------
def download_prices(tickers, start: str, pause: float = 0.5) -> pd.DataFrame:
    """
    Download daily adjusted close prices.

    Returns a LONG table: Date | Ticker | Close | Volume
    Adjusted prices (auto_adjust=True) include dividends, so buy-and-hold
    returns already contain reinvested dividends.
    """
    import yfinance as yf            # imported here so 'file' mode works without yfinance

    frames, failed = [], []
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
            if hist.empty:           # delisted / wrong ticker -> empty frame, not an exception
                failed.append(ticker)
                continue
            idx = hist.index
            # Drop the timezone but KEEP the local trading date.
            # (tz_convert(None) would shift European dates to the previous day in UTC.)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            frames.append(pd.DataFrame({
                "Date": idx.normalize(),
                "Ticker": ticker,
                "Close": hist["Close"].to_numpy(),
                "Volume": hist["Volume"].to_numpy(),
            }))
        except Exception as exc:     # network errors etc. -> skip ticker, report later
            print(f"  ! {ticker}: {type(exc).__name__}")
            failed.append(ticker)
        time.sleep(pause)            # be polite to Yahoo (avoid rate limits)

    if failed:
        print(f"  Failed tickers: {failed}")
    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Macro (FRED CSV endpoint + VIX from yfinance)
# ---------------------------------------------------------------------------
def download_fred(series_ids, start: str) -> pd.DataFrame:
    """Download FRED series as CSV (no API key). Missing values ('.') become NaN."""
    out = []
    for sid in series_ids:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["Date", sid]                       # header name changed over time
        df["Date"] = pd.to_datetime(df["Date"])
        df[sid] = pd.to_numeric(df[sid], errors="coerce")
        out.append(df.set_index("Date"))
    macro = pd.concat(out, axis=1).sort_index()
    return macro[macro.index >= start].reset_index()


def download_macro(fred_series, vix_ticker: str, start: str) -> pd.DataFrame:
    """Daily macro table: Date | DGS10 | DGS1 | VIX"""
    fred = download_fred(fred_series, start)
    vix = download_prices([vix_ticker], start)[["Date", "Close"]].rename(columns={"Close": "VIX"})
    macro = fred.merge(vix, on="Date", how="outer").sort_values("Date")
    return macro.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point used by the pipeline
# ---------------------------------------------------------------------------
def load_data(source: str, cfg):
    """source='download' -> fetch + cache;  source='file' -> read cache only."""
    prices_path = os.path.join(cfg.DATA_DIR, "prices")
    macro_path = os.path.join(cfg.DATA_DIR, "macro")

    if source == "download":
        print("Downloading prices ...")
        prices = download_prices(cfg.ALL_TICKERS, cfg.START_DATE)
        print("Downloading macro ...")
        macro = download_macro(cfg.FRED_SERIES, cfg.VIX_TICKER, cfg.START_DATE)
        print("  saved:", save_df(prices, prices_path), "|", save_df(macro, macro_path))
    elif source == "file":
        prices, macro = load_df(prices_path), load_df(macro_path)
    else:
        raise ValueError("source must be 'download' or 'file'")

    print(f"Prices: {len(prices):,} rows, {prices.Ticker.nunique()} tickers, "
          f"{prices.Date.min().date()} -> {prices.Date.max().date()}")
    return prices, macro
