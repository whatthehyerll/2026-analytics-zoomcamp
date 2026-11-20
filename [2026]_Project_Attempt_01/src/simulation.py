"""
Step 4 - Buy-only monthly investing simulation (dollar-cost averaging).

Every month a fixed contribution is split across assets according to a weight
table (month x ticker). Nothing is ever sold. The simulation is fully
vectorised and exact: units bought = money / price, holdings accumulate, and
dividends are included because prices are dividend-adjusted.

Metrics
  * IRR (money-weighted, annualised): the right return measure when money is
    added every month (a simple CAGR assumes one initial deposit)
  * TWR CAGR and max drawdown on a time-weighted index (removes the effect of
    the inflows themselves)
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def price_matrix(monthly: pd.DataFrame) -> pd.DataFrame:
    """Month x ticker table of buying prices (close on the first trading day)."""
    return monthly.pivot_table(index="ym", columns="Ticker", values="Close", aggfunc="first").sort_index()


def contribution_series(monthly, pm, cfg, buy_the_dip: bool) -> pd.Series:
    """Fixed monthly amount; doubled when the benchmark is in a drawdown (strategy D)."""
    cash = pd.Series(float(cfg.MONTHLY_CONTRIBUTION), index=pm.index)
    if buy_the_dip:
        dd = (monthly[monthly.Ticker == cfg.BENCHMARK].set_index("ym")["bench_drawdown"]
              .reindex(pm.index))
        cash[dd <= cfg.DRAWDOWN_TRIGGER] *= cfg.DRAWDOWN_MULTIPLIER
    return cash


# ---------------------------------------------------------------------------
# Weight tables (each row sums to 1)
# ---------------------------------------------------------------------------
def _equal(pm, tickers, share=1.0):
    w = pd.DataFrame(0.0, index=pm.index, columns=pm.columns)
    cols = [t for t in tickers if t in pm.columns]
    avail = pm[cols].notna()                                  # only assets with a price that month
    w[cols] = avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * share
    return w


def weights_passive(pm, cfg):
    return _equal(pm, [cfg.BENCHMARK])


def weights_core(pm, cfg):
    return _equal(pm, cfg.CORE)


def weights_core_satellite(monthly, pm, cfg, use_model: bool = True):
    """
    Core share split equally across core ETFs.
    Satellite share: top-K satellite stocks by predicted probability (>= threshold).
    If fewer than K qualify, the unused part goes to the benchmark.
    use_model=False -> satellite share split equally across ALL satellite stocks
    (a 'no-ML' baseline that isolates the value of the model).
    """
    sat_share = 1 - cfg.CORE_SHARE
    w = _equal(pm, cfg.CORE, share=cfg.CORE_SHARE)
    if not use_model:
        return w + _equal(pm, cfg.SATELLITE, share=sat_share)

    cand = monthly[monthly.Ticker.isin(cfg.SATELLITE) & (monthly.proba >= cfg.PROBA_THRESHOLD)]
    for ym, grp in cand.groupby("ym"):
        picks = grp.nlargest(cfg.TOP_K, "proba")["Ticker"].tolist()
        for t in picks:
            w.loc[ym, t] += sat_share / cfg.TOP_K
        w.loc[ym, cfg.BENCHMARK] += sat_share * (cfg.TOP_K - len(picks)) / cfg.TOP_K
    no_pick = ~w.index.isin(cand["ym"].unique())               # months without any pick
    w.loc[no_pick, cfg.BENCHMARK] += sat_share
    return w


# ---------------------------------------------------------------------------
# Simulation and metrics
# ---------------------------------------------------------------------------
def _irr_annual(cash_flows: np.ndarray) -> float:
    """Monthly IRR by bisection (no extra dependency), annualised."""
    t = np.arange(len(cash_flows))
    npv = lambda r: np.sum(cash_flows / (1 + r) ** t)
    lo, hi = -0.5, 0.5
    if npv(lo) * npv(hi) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (1 + mid) ** 12 - 1


def run_dca(pm: pd.DataFrame, weights: pd.DataFrame, cash: pd.Series, fee_rate: float = 0.0):
    """Vectorised buy-only simulation. Returns (summary dict, value series)."""
    w = weights.where(pm.notna(), 0)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)   # renormalise if a price is missing
    units = w.mul(cash * (1 - fee_rate), axis=0) / pm               # units bought each month (after fees)
    holdings = units.fillna(0).cumsum()
    px = pm.ffill()                                                  # carry last price for valuation

    value = (holdings * px).sum(axis=1)                              # after this month's purchase
    value_before = (holdings.shift(1).fillna(0) * px).sum(axis=1)    # before this month's purchase
    ret = (value_before / value.shift(1) - 1).iloc[1:]               # time-weighted monthly returns
    twr = (1 + ret).cumprod()
    years = len(ret) / 12

    flows = -cash.to_numpy().copy()
    flows[-1] += value.iloc[-1]                                      # sell everything at the end (valuation only)
    summary = {
        "months": len(cash),
        "invested": cash.sum(),
        "final_value": value.iloc[-1],
        "profit_multiple": value.iloc[-1] / cash.sum(),
        "IRR": _irr_annual(flows),
        "TWR_CAGR": twr.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "max_drawdown": (twr / twr.cummax() - 1).min(),
    }
    return summary, value


def compare_strategies(monthly, cfg, period_mask=None):
    """Run all strategies on the same months and return (summary table, value curves)."""
    data = monthly if period_mask is None else monthly[period_mask]
    pm = price_matrix(data)
    plain = contribution_series(data, pm, cfg, buy_the_dip=False)
    dip = contribution_series(data, pm, cfg, buy_the_dip=True)

    strategies = {
        "A_passive_all_world": (weights_passive(pm, cfg), plain),
        "B_core_equal": (weights_core(pm, cfg), plain),
        "C0_core_satellite_no_ml": (weights_core_satellite(data, pm, cfg, use_model=False), plain),
        "C_core_satellite_ml": (weights_core_satellite(data, pm, cfg, use_model=True), plain),
        "D_ml_plus_buy_the_dip": (weights_core_satellite(data, pm, cfg, use_model=True), dip),
    }
    rows, curves = [], {}
    for name, (w, c) in strategies.items():
        s, v = run_dca(pm, w, c, fee_rate=cfg.FEE_RATE)
        rows.append({"strategy": name, **s})
        curves[name] = v
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ---------------------------------------------------------------------------
# "What do I buy this month?"
# ---------------------------------------------------------------------------
def latest_buy_list(monthly, cfg) -> pd.DataFrame:
    last = monthly["ym"].max()
    data = monthly[monthly.ym == last]
    pm = price_matrix(data)
    w = weights_core_satellite(data, pm, cfg, use_model=True).iloc[0]
    cash = contribution_series(data, pm, cfg, buy_the_dip=True).iloc[0]
    out = (w[w > 0].rename("weight").to_frame()
           .assign(amount=lambda d: (d.weight * cash).round(2))
           .join(data.set_index("Ticker")[["Close", "proba"]]))
    out.index.name = f"buy list {last}"
    return out.sort_values("amount", ascending=False)
