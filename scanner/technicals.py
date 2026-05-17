"""
Technical indicators: RSI, ATR, support/resistance, momentum duration estimate.
"""
import numpy as np
import pandas as pd
from config import RSI_OVERBOUGHT, RSI_OVERSOLD


def compute_rsi(closes: pd.Series, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(float(100 - 100 / (1 + rs)), 2)


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """Average True Range — measures recent volatility."""
    if len(df) < period + 1:
        return None
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 4)


def find_support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Simple pivot-based support/resistance from recent highs and lows.
    Returns {"support": float, "resistance": float}.
    """
    if len(df) < lookback:
        return {"support": None, "resistance": None}
    recent = df.tail(lookback)
    return {
        "support": round(float(recent["Low"].min()), 4),
        "resistance": round(float(recent["High"].max()), 4),
    }


def estimate_momentum_duration(atr: float | None, rsi: float | None, price: float | None) -> dict:
    """
    Rough estimate of how many bars a momentum move may last and a price target.
    Based on ATR multiples and RSI distance from 50.

    Logic:
      - RSI further from 50 → move has more room
      - ATR → volatility-adjusted target band
    Returns {"bars_estimate": int, "target_price": float | None, "direction": str}
    """
    if atr is None or rsi is None or price is None:
        return {"bars_estimate": 0, "target_price": None, "direction": "UNKNOWN"}

    rsi_distance = abs(rsi - 50)
    # Each 10 RSI points from center ≈ 1–2 bars of continuation
    bars = max(1, int(rsi_distance / 10))

    if rsi > 50:
        direction = "UP"
        target = round(price + atr * bars * 0.5, 2)
    else:
        direction = "DOWN"
        target = round(price - atr * bars * 0.5, 2)

    return {"bars_estimate": bars, "target_price": target, "direction": direction}


def rsi_label(rsi: float | None) -> str:
    if rsi is None:
        return "UNKNOWN"
    if rsi >= RSI_OVERBOUGHT:
        return "OVERBOUGHT"
    if rsi <= RSI_OVERSOLD:
        return "OVERSOLD"
    return "NEUTRAL"
