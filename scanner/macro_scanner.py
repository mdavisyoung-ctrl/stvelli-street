"""
Specialized scanner for SPY, XRP, and BTC.

These three assets are treated differently from equities:
  - SPY:  represents the S&P 500 macro regime — its signal colors all 100 stock trades
  - XRP/BTC: crypto has no options chain; we use price action + news sentiment only

Fetches data via yfinance (BTC-USD, XRP-USD, SPY all supported).
"""
import logging
import yfinance as yf
import pandas as pd
from scanner.data_fetcher import get_news
from scanner.sentiment import aggregate_sentiment
from scanner.technicals import compute_rsi, compute_atr, find_support_resistance, estimate_momentum_duration, rsi_label
from scanner.options_analyzer import compute_cp_ratio, cp_zscore, classify_cp

logger = logging.getLogger(__name__)

MACRO_TICKERS = {
    "SPY": {"type": "etf",    "yf_symbol": "SPY",     "eodhd_symbol": "SPY"},
    "BTC": {"type": "crypto", "yf_symbol": "BTC-USD",  "eodhd_symbol": "BTC"},
    "XRP": {"type": "crypto", "yf_symbol": "XRP-USD",  "eodhd_symbol": "XRP"},
}


def fetch_macro_data(days: int = 60) -> dict:
    """
    Returns {ticker: {"history": df, "options": dict, "news": list, "type": str}}
    """
    results = {}
    for name, meta in MACRO_TICKERS.items():
        try:
            t = yf.Ticker(meta["yf_symbol"])
            hist = t.history(period=f"{days}d", auto_adjust=True)
            options = {}
            if meta["type"] == "etf":
                try:
                    expiries = t.options
                    if expiries:
                        chosen = expiries[0]
                        chain = t.option_chain(chosen)
                        options = {"calls": chain.calls, "puts": chain.puts, "expiry": chosen}
                except Exception:
                    pass
            news = get_news(meta["eodhd_symbol"], limit=20)
            results[name] = {"history": hist, "options": options, "news": news, "type": meta["type"]}
        except Exception as e:
            logger.error("Macro fetch failed for %s: %s", name, e)
    return results


def analyze_macro(name: str, data: dict, cp_history: list) -> dict:
    """
    Returns a macro analysis dict with regime label and signal.
    """
    hist = data.get("history")
    options = data.get("options", {})
    news = data.get("news", [])
    asset_type = data.get("type", "etf")

    price = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
    rsi = compute_rsi(hist["Close"]) if hist is not None and not hist.empty else None
    atr = compute_atr(hist) if hist is not None and not hist.empty else None
    sr = find_support_resistance(hist) if hist is not None and not hist.empty else {"support": None, "resistance": None}
    momentum = estimate_momentum_duration(atr, rsi, price)
    sent = aggregate_sentiment(news)

    cp_ratio = compute_cp_ratio(options)
    cp_full = cp_history + ([cp_ratio] if cp_ratio is not None else [])
    zscore = cp_zscore(cp_full)
    cp_class = classify_cp(cp_ratio, zscore)

    # Regime classification
    r_label = rsi_label(rsi)
    if name == "SPY":
        if rsi and rsi > 65 and sent["label"] == "BULLISH" and cp_class == "OVERBULLISH":
            regime = "RISK_OFF"   # overextended, fade the market
        elif rsi and rsi < 45 and sent["label"] == "BEARISH":
            regime = "RISK_OFF"
        elif sent["label"] == "BULLISH" and rsi and 45 <= rsi <= 65:
            regime = "RISK_ON"
        else:
            regime = "NEUTRAL"
    else:
        # Crypto: purely momentum / sentiment driven
        if rsi and rsi > 70:
            regime = "OVERBOUGHT"
        elif rsi and rsi < 30:
            regime = "OVERSOLD"
        elif sent["label"] == "BULLISH":
            regime = "BULLISH"
        else:
            regime = "NEUTRAL"

    return {
        "ticker": name,
        "type": asset_type,
        "price": round(price, 4) if price else None,
        "rsi": rsi,
        "rsi_label": r_label,
        "atr": atr,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "sentiment_score": sent["score"],
        "sentiment_label": sent["label"],
        "top_headline": sent["top_headline"],
        "cp_ratio": round(cp_ratio, 3) if cp_ratio else None,
        "cp_class": cp_class,
        "regime": regime,
        "momentum": momentum,
    }
