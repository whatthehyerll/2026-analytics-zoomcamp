# Buy-only Monthly Investing with an ML Tilt

**Can a machine learning model improve *where* my monthly savings go, compared with simply buying a global index fund every month?**

Capstone project for the [Stock Markets Analytics Zoomcamp](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp) (2026 cohort).
Status: **Project attempt 1** (working end-to-end pipeline; results and deeper analysis will be extended in attempt 2).

---

## 1. Problem description

I am a long-term, **buy-only** investor: every month I invest a fixed amount and I never sell.
The course strategies buy a stock and sell it 30 days later. That does not fit how I (and many private savers) invest,
so this project reframes the problem:

> On the first trading day of each month, the model estimates for each asset the probability that it will
> **outperform the FTSE All-World index over the next 6 months**. The monthly contribution is then split
> between a fixed **core** of index ETFs and a **satellite** of large US stocks chosen by the model.

**How the solution is used:** once a month the pipeline downloads the latest data and prints *this month's buy list*
(tickers and amounts). The backtest answers whether this beats the passive alternative.

**Why this is a new problem definition (vs. the course setup):**
* long-term, monthly decisions with a 6-month horizon instead of 30-day trades
* buy-only accumulation (dollar-cost averaging) instead of buy-and-sell
* mixed universe across exchanges (US ETFs/stocks, Euronext Amsterdam, Xetra)
* target is *relative* performance vs. a benchmark, not absolute growth

---

## 2. Data sources

| Source | Data | Access |
|---|---|---|
| yfinance | Daily dividend-adjusted prices for 15 assets, VIX | `yf.Ticker().history(auto_adjust=True)` |
| FRED | 10Y and 1Y US Treasury yields (`DGS10`, `DGS1`) | public CSV endpoint, no API key |

**Universe** (`config.py`)

| Group | Tickers |
|---|---|
| Benchmark | `VWRL.AS` — Vanguard FTSE All-World (listed since 2012) |
| Core | `VWRL.AS`, `SPY` (S&P 500), `QQQ` (Nasdaq 100), `EXS1.DE` (DAX) |
| Satellite | `MSFT AAPL GOOG NVDA AMZN META BRK-B LLY AVGO V JPM` |

> ⚠️ European tickers must be verified on the first download (alternatives: `VWRL.L`, `VGWL.DE`, `^GDAXI`).
> Failed tickers are reported and skipped, not silently dropped.

**Features** (34 = 20 numerical + 2 dummy sets)

| Set | Feature | Description |
|---|---|---|
| GROWTH | `growth_21d/63d/126d/252d` | Price ratio vs. 1, 3, 6, 12 months ago (momentum) |
| GROWTH | `rel_growth_126d` | 6-month growth minus the benchmark's 6-month growth |
| TECHNICAL | `vol_63d` | Annualised volatility of **daily returns** (3 months) |
| TECHNICAL | `rsi_14` | Relative Strength Index (Wilder, implemented in pandas) |
| TECHNICAL | `dist_52w_high`, `dist_52w_low` | Distance from the 52-week high / low |
| MACRO | `DGS10`, `term_spread` | 10Y yield and 10Y−1Y spread (yield-curve shape) |
| MACRO | `dgs10_chg_63d` | 3-month change in the 10Y yield (rates enter as *changes*, not only levels) |
| MACRO | `VIX`, `vix_chg_21d` | Market fear gauge and its 1-month change |
| MACRO | `bench_drawdown` | All-World distance from its all-time high |
| RANKS | `rank_*` | Each asset's percentile among all assets in the same month (growth, RSI, volatility, 52w-high distance) |
| DUMMIES | `month_name_*`, `asset_group_*` | Calendar month; core vs. satellite |
| TO_PREDICT | `is_outperform_6m` | **Target**: 1 if the asset beats All-World over the next 126 trading days |

Data files are **not** committed to GitHub; they are cached locally in `data/` (see §6).

---

## 3. Data transformations

1. Long table of daily prices (`Date | Ticker | Close | Volume`)
2. Per-ticker features — every `shift`/`rolling` runs **inside `groupby('Ticker')`** so tickers never mix
3. Benchmark features joined with `merge_asof` (handles different holiday calendars of US and European exchanges)
4. Macro data forward-filled only (no backward fill → no look-ahead)
5. **Monthly sampling**: first trading day of each month = buying day
6. Cross-sectional ranks per month and dummies → one modelling DataFrame

The target stays **NaN** when the 6-month future is unknown (latest months), instead of being labelled 0.
These rows are excluded from training but still receive predictions (that is this month's buy list).

---

## 4. Modelling

| | |
|---|---|
| Split | TRAIN ≤ 2019-12, VALIDATION 2020-01…2021-12, TEST ≥ 2022-01 (temporal) |
| Embargo | 6 months before each split boundary (6-month labels would otherwise overlap the next period) |
| Models | `DecisionTreeClassifier`, `RandomForestClassifier` |
| Tuning | `max_depth` 2–8 for both models, selected by **validation precision** (min. 20 positive calls) |
| Final model | Refit on TRAIN+VALIDATION, predict every month incl. the latest |
| Decision rule | Satellite pick only if probability ≥ 0.55, top-3 per month |

**Test report includes `n_months`** (distinct months with a signal). Because every asset is compared with the
*same* benchmark, all assets tend to "outperform" together in months when the benchmark does badly.
A synthetic-noise smoke test produced 0.80 precision from signals in only 17 months — pure luck.
Precision is therefore always read together with `n_months`.

---

## 5. Trading simulation (buy-only)

Fully vectorised and exact: units bought = money / price, holdings accumulate, nothing is sold.
Prices are dividend-adjusted, so dividends are reinvested. A 0.1% fee applies to each purchase.

| Strategy | Each month's contribution |
|---|---|
| **A** (benchmark) | 100% All-World |
| B | 25% each core ETF |
| C0 | 70% core + 30% equally over all satellite stocks (**no ML** — isolates the model's value) |
| **C** | 70% core + 30% to the top-3 satellite stocks by predicted probability |
| **D** | C, and the contribution is doubled when All-World is ≥5% below its high |

**Metrics:** IRR (money-weighted, the correct measure when money is added monthly), time-weighted CAGR,
max drawdown. Strategy D invests more money in total, so it is compared by IRR.

### Results (TEST period, out-of-sample)

> To be filled after the first run on real data — copy `outputs/strategy_summary.csv` here.

| Strategy | IRR | TWR CAGR | Max drawdown | Final value / invested |
|---|---|---|---|---|
| A_passive_all_world | | | | |
| B_core_equal | | | | |
| C0_core_satellite_no_ml | | | | |
| C_core_satellite_ml | | | | |
| D_ml_plus_buy_the_dip | | | | |

---

## 6. How to run

```bash
git clone https://github.com/whatthehyerll/2026-analytics-zoomcamp.git
cd "2026-analytics-zoomcamp/[2026]_Project_Attempt_01"   # quotes needed: [ ] are special characters in some shells

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py --source download   # regime 1: fetch fresh data, cache it in data/
python run_pipeline.py --source file       # regime 2: reuse cached data (fast, reproducible)
```

**Quickest option (Google Colab):** open `notebooks/all_in_one_colab.ipynb` and choose *Runtime → Run all*.
It contains the same code as the `.py` files in a single notebook.

Or open `notebooks/main_workflow.ipynb` (works in Google Colab), which calls the same functions step by step.

**Monthly automation (cron):** runs on the first weekdays of each month and prints the buy list.
```cron
0 8 1-3 * 1-5  cd "/path/to/2026-analytics-zoomcamp/[2026]_Project_Attempt_01" && .venv/bin/python run_pipeline.py --source download >> outputs/cron.log 2>&1
```

Outputs in `outputs/`: `tuning.csv`, `strategy_summary.csv`, `value_curves.csv`, `latest_buy_list.csv`.

### Repository structure
```
config.py                 # universe, dates, strategy parameters
run_pipeline.py           # end-to-end script (CLI)
src/data_loader.py        # step 1: download / cache / load
src/features.py           # step 2: features, monthly table, feature sets
src/modeling.py           # step 3: split + embargo, tuning, final model, test report
src/simulation.py         # step 4: buy-only simulation, metrics, latest buy list
notebooks/main_workflow.ipynb      # step-by-step workflow calling the src/ functions
notebooks/all_in_one_colab.ipynb   # same code in one notebook (easiest in Colab)
requirements.txt
```

---

## 7. Known limitations (honest list)

* **Survivorship bias:** satellite stocks are today's large caps; an investor in 2013 could not know this list. Results are likely optimistic.
* **Currency ignored:** EUR- and USD-listed assets are mixed without FX conversion (same simplification as Homework 1).
* **Short history:** All-World ETF data starts in 2012 → about 14 years, ~170 months.
* **Few independent observations:** 6-month labels overlap and all assets share one benchmark.
* **Execution:** signals use the close of the buying day and the purchase happens at that close.

## 8. Plan for attempt 2

* Rolling start dates: run the 5-year plan from every starting month and report how often C/D beat A
* Walk-forward retraining (yearly) instead of a single split
* EUR conversion with ECB exchange rates (new data source)
* Additional model (XGBoost / logistic regression) and threshold analysis
* Quantify survivorship bias with a historical large-cap list
* Simple dashboard of the monthly buy list and portfolio value

---

*Code was smoke-tested end-to-end on synthetic data (no leakage: on random data, test precision stayed at the base rate across four seeds). AI assistance was used for code structure and review; design decisions, data choices and interpretation are my own.*
