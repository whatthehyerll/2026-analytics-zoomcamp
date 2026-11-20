"""
Step 2 - Feature engineering and the monthly modelling table.

Pipeline:
  daily prices -> per-ticker daily features -> + benchmark-relative features
  -> + macro features -> sample the FIRST trading day of each month
  -> + cross-sectional ranks and dummies -> one modelling DataFrame

Lessons from the course applied here:
  * all shift/rolling operations run inside groupby('Ticker')   (Module 2)
  * the target stays NaN when the future is unknown (no fake zeros) (Module 3)
  * rates enter as changes / spreads, not only raw levels        (Module 3)
"""
import numpy as np
import pandas as pd

GROWTH_WINDOWS = [21, 63, 126, 252]   # ~1, 3, 6, 12 months of trading days


# ---------------------------------------------------------------------------
# Per-ticker daily features
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI implemented with pandas (avoids the TA-Lib C dependency)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _as_ns(s: pd.Series) -> pd.Series:
    """Unify datetime resolution (merge_asof requires identical key dtypes)."""
    return pd.to_datetime(s).astype("datetime64[ns]")


def add_price_features(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = prices.copy()
    df["Date"] = _as_ns(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    g = df.groupby("Ticker")["Close"]

    for n in GROWTH_WINDOWS:                                    # momentum features (past)
        df[f"growth_{n}d"] = g.transform(lambda s, n=n: s / s.shift(n))

    daily_ret = g.transform(lambda s: s.pct_change())
    # Volatility on RETURNS (not on price levels), annualised with sqrt(252)
    df["vol_63d"] = daily_ret.groupby(df["Ticker"]).transform(lambda s: s.rolling(63).std() * np.sqrt(252))
    df["rsi_14"] = g.transform(_rsi)
    df["dist_52w_high"] = g.transform(lambda s: s / s.rolling(252, min_periods=60).max() - 1)
    df["dist_52w_low"] = g.transform(lambda s: s / s.rolling(252, min_periods=60).min() - 1)

    # Future growth = what we want to predict (NaN for the last `horizon` days)
    df[f"future_growth_{horizon}d"] = g.transform(lambda s: s.shift(-horizon) / s)
    return df


# ---------------------------------------------------------------------------
# Benchmark-relative features and target
# ---------------------------------------------------------------------------
def add_benchmark_features(df: pd.DataFrame, benchmark: str, horizon: int) -> pd.DataFrame:
    fut = f"future_growth_{horizon}d"
    bench = df.loc[df.Ticker == benchmark, ["Date", "Close", "growth_126d", fut]].copy()
    # All-time-high drawdown of the benchmark (used by strategy D and as a feature)
    bench["bench_drawdown"] = bench["Close"] / bench["Close"].cummax() - 1
    bench = bench.rename(columns={"growth_126d": "bench_growth_126d", fut: "bench_" + fut}).drop(columns="Close")

    # merge_asof: each asset row gets the latest benchmark value on/before its date
    # (handles different exchange holidays between US and European listings)
    df = pd.merge_asof(df.sort_values("Date"), bench.sort_values("Date"), on="Date", direction="backward")

    df["rel_growth_126d"] = df["growth_126d"] - df["bench_growth_126d"]
    df["excess_future_growth"] = df[fut] - df["bench_" + fut]

    # TARGET: 1 if the asset beats the benchmark over the next 6 months.
    # Kept as NaN when the future is unknown -> such rows are excluded from training.
    known = df["excess_future_growth"].notna()
    df["is_outperform_6m"] = np.where(known, (df["excess_future_growth"] > 0).astype(float), np.nan)
    # The benchmark cannot beat itself -> no label (keeps it out of training)
    df.loc[df["Ticker"] == benchmark, "is_outperform_6m"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Macro features
# ---------------------------------------------------------------------------
def add_macro_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.copy()
    m["Date"] = _as_ns(m["Date"])
    m = m.sort_values("Date")
    m[["DGS10", "DGS1", "VIX"]] = m[["DGS10", "DGS1", "VIX"]].ffill()   # forward-fill only (no look-ahead)
    m["term_spread"] = m["DGS10"] - m["DGS1"]              # >0 normal, <0 inverted curve
    m["dgs10_chg_63d"] = m["DGS10"] - m["DGS10"].shift(63)  # 3-month change in long rates
    m["vix_chg_21d"] = m["VIX"] / m["VIX"].shift(21) - 1
    keep = ["Date", "DGS10", "term_spread", "dgs10_chg_63d", "VIX", "vix_chg_21d"]
    return pd.merge_asof(df.sort_values("Date"), m[keep], on="Date", direction="backward")


# ---------------------------------------------------------------------------
# Monthly table
# ---------------------------------------------------------------------------
RANK_COLS = ["growth_126d", "growth_252d", "rsi_14", "vol_63d", "dist_52w_high"]


def build_monthly_table(prices, macro, cfg):
    """Return (monthly_df, feature_sets dict)."""
    df = add_price_features(prices, cfg.HORIZON_DAYS)
    df = add_benchmark_features(df, cfg.BENCHMARK, cfg.HORIZON_DAYS)
    df = add_macro_features(df, macro)

    # One row per ticker and month: the first trading day = our monthly buying day
    df["ym"] = df["Date"].dt.to_period("M")
    first_idx = df.groupby(["Ticker", "ym"])["Date"].idxmin()
    monthly = df.loc[first_idx].sort_values(["ym", "Ticker"]).reset_index(drop=True)

    # Cross-sectional ranks: position of each asset vs. the others in the same month (0..1)
    for c in RANK_COLS:
        monthly[f"rank_{c}"] = monthly.groupby("ym")[c].rank(pct=True)

    # Dummies: calendar month + asset group
    monthly["month_name"] = monthly["Date"].dt.month_name()
    monthly["asset_group"] = np.where(monthly["Ticker"].isin(cfg.CORE), "core", "satellite")
    dummies = pd.get_dummies(monthly[["month_name", "asset_group"]], dtype="int8")
    monthly = pd.concat([monthly, dummies], axis=1)

    feature_sets = {
        "GROWTH": [f"growth_{n}d" for n in GROWTH_WINDOWS] + ["rel_growth_126d"],
        "TECHNICAL": ["vol_63d", "rsi_14", "dist_52w_high", "dist_52w_low"],
        "MACRO": ["DGS10", "term_spread", "dgs10_chg_63d", "VIX", "vix_chg_21d", "bench_drawdown"],
        "RANKS": [f"rank_{c}" for c in RANK_COLS],
        "DUMMIES": dummies.columns.tolist(),
        "TO_PREDICT": ["is_outperform_6m", "excess_future_growth", f"future_growth_{cfg.HORIZON_DAYS}d"],
    }
    feature_sets["NUMERICAL"] = (feature_sets["GROWTH"] + feature_sets["TECHNICAL"]
                                 + feature_sets["MACRO"] + feature_sets["RANKS"])
    feature_sets["FEATURES"] = feature_sets["NUMERICAL"] + feature_sets["DUMMIES"]
    return monthly, feature_sets
