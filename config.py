import os
from dotenv import load_dotenv

try:
    load_dotenv()
except Exception:
    pass  # .env missing or malformed — fall back to env vars or default below

_FALLBACK_KEY = "69af52dd739130.88973111"
EODHD_API_KEY = os.getenv("EODHD_API_KEY") or _FALLBACK_KEY

SCAN_INTERVAL_SECONDS = 300  # 5 minutes

STARTING_CAPITAL = 50.0
MAX_RISK_PER_TRADE = 0.10   # 10% of current balance per trade
MAX_OPEN_POSITIONS = 5

# Signal thresholds
CP_FADE_ZSCORE = 1.5        # C/P ratio z-score above this → overbullish, fade candidate
CP_MOMENTUM_ZSCORE = -1.0   # C/P ratio z-score below this → put activity dropping, momentum candidate
SENTIMENT_BULLISH_THRESHOLD = 0.25   # TextBlob polarity above this = bullish
SENTIMENT_BEARISH_THRESHOLD = -0.15  # TextBlob polarity below this = bearish
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
CP_LOOKBACK_DAYS = 20       # days of C/P history for z-score baseline

EODHD_BASE_URL = "https://eodhd.com/api"

TOP_100_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "AVGO",
    "JPM", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "ORCL",
    "ABBV", "BAC", "KO", "MRK", "CVX", "WMT", "NFLX", "CRM", "AMD", "PEP",
    "TMO", "ACN", "LIN", "MCD", "CSCO", "ABT", "NKE", "ADBE", "TXN", "DHR",
    "NEE", "PM", "QCOM", "IBM", "INTU", "RTX", "AMGN", "CAT", "SPGI", "UNP",
    "LOW", "ISRG", "GS", "BKNG", "SYK", "MS", "VRTX", "T", "PLD", "AMAT",
    "MDT", "BLK", "AXP", "TJX", "GILD", "REGN", "CI", "ADP", "MMC", "LRCX",
    "SCHW", "C", "PGR", "ZTS", "CB", "BSX", "ETN", "MO", "DE", "EOG",
    "SO", "NOC", "CME", "WM", "ITW", "HUM", "SLB", "F", "GM", "DUK",
    "CL", "MSI", "MCO", "GD", "APD", "FDX", "NSC", "PH", "ICE", "USB",
]
