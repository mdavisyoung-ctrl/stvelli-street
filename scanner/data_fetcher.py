"""
Fetches price data (yfinance) and news (EODHD).
All network calls are isolated here so tests can mock cleanly.
"""
import time
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from config import EODHD_API_KEY, EODHD_BASE_URL

logger = logging.getLogger(__name__)


def get_price_history(ticker: str, days: int = 60) -> pd.DataFrame:
    """OHLCV DataFrame for `ticker` covering the last `days` calendar days."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=f"{days}d", auto_adjust=True)
        if df.empty:
            logger.warning("No price data for %s", ticker)
        return df
    except Exception as e:
        logger.error("Price fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()


def get_current_price(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
        if price is None:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return float(price) if price else None
    except Exception as e:
        logger.error("Current price fetch failed for %s: %s", ticker, e)
        return None


def get_options_chain(ticker: str) -> dict:
    """
    Returns {"calls": DataFrame, "puts": DataFrame, "expiry": str}
    for the nearest expiry with meaningful open interest.
    """
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return {}

        # Use the nearest expiry that is at least 7 days out
        today = datetime.today()
        chosen = None
        for exp in expiries:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            if (exp_dt - today).days >= 7:
                chosen = exp
                break
        if chosen is None:
            chosen = expiries[0]

        chain = t.option_chain(chosen)
        return {"calls": chain.calls, "puts": chain.puts, "expiry": chosen}
    except Exception as e:
        logger.error("Options fetch failed for %s: %s", ticker, e)
        return {}


def get_news(ticker: str, limit: int = 50) -> list[dict]:
    """
    Fetches recent news from EODHD for `ticker`.
    Returns list of {"title": str, "date": str, "content": str}.
    """
    try:
        url = f"{EODHD_BASE_URL}/news"
        params = {
            "s": ticker,
            "limit": limit,
            "api_token": EODHD_API_KEY,
            "fmt": "json",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json()
        return [
            {
                "title": a.get("title", ""),
                "date": a.get("date", ""),
                "content": a.get("content", ""),
                "link": a.get("link", ""),
            }
            for a in articles
        ]
    except Exception as e:
        logger.error("News fetch failed for %s: %s", ticker, e)
        return []


def batch_fetch(tickers: list[str], delay: float = 0.25) -> dict:
    """
    Fetches price history + options + news for all tickers.
    Returns {ticker: {"history": df, "options": dict, "news": list}}.
    Adds a small delay between calls to avoid rate limiting.
    """
    results = {}
    for ticker in tickers:
        results[ticker] = {
            "history": get_price_history(ticker),
            "options": get_options_chain(ticker),
            "news": get_news(ticker),
        }
        time.sleep(delay)
    return results
