"""
Project configuration: universe, dates and strategy parameters.

All tunable settings live here so the pipeline scripts stay free of hard-coded values.
"""

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# Benchmark = the asset a passive investor would buy every month.
# VWRL tracks the FTSE All-World index (large + mid caps, developed + emerging).
# NOTE: verify the yfinance ticker on first download (alternatives: 'VWRL.L', 'VGWL.DE').
BENCHMARK = "VWRL.AS"

# Core = fixed part of the monthly plan (index ETFs).
CORE = [
    "VWRL.AS",   # FTSE All-World
    "SPY",       # S&P 500
    "QQQ",       # Nasdaq 100
    "EXS1.DE",   # iShares Core DAX ETF (verify ticker; fallback: '^GDAXI' index)
]

# Satellite = large US stocks the model can tilt towards (same US list as the course).
# Known limitation: chosen with today's knowledge -> survivorship bias (see README).
SATELLITE = ["MSFT", "AAPL", "GOOG", "NVDA", "AMZN", "META",
             "BRK-B", "LLY", "AVGO", "V", "JPM"]

ALL_TICKERS = list(dict.fromkeys(CORE + SATELLITE))   # de-duplicated, order kept

# Macro series from FRED (downloaded as CSV, no API key needed)
FRED_SERIES = ["DGS10", "DGS1"]          # 10Y and 1Y US Treasury yields
VIX_TICKER = "^VIX"                      # market fear gauge (yfinance)

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
START_DATE = "2011-01-01"     # extra history so 252-day features exist from 2012-2013
TRAIN_END = "2019-12"         # last month of TRAIN
VALID_END = "2021-12"         # last month of VALIDATION; TEST starts the month after
EMBARGO_MONTHS = 6            # gap before each split: 6-month labels must not overlap

# ---------------------------------------------------------------------------
# Target and strategy
# ---------------------------------------------------------------------------
HORIZON_DAYS = 126            # ~6 months of trading days
MONTHLY_CONTRIBUTION = 1000   # money invested every month (currency effects ignored)
CORE_SHARE = 0.70             # 70% of each contribution goes to the core ETFs
TOP_K = 3                     # number of satellite stocks bought per month
PROBA_THRESHOLD = 0.55        # minimum predicted probability for a satellite pick
DRAWDOWN_TRIGGER = -0.05      # strategy D: double the contribution below this drawdown
DRAWDOWN_MULTIPLIER = 2.0
FEE_RATE = 0.001              # 0.1% cost on every purchase (no sells in a buy-only plan)

RANDOM_STATE = 42
DATA_DIR = "data"             # local cache (NOT committed to GitHub, see .gitignore)
OUTPUT_DIR = "outputs"
