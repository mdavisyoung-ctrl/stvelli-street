"""
Risk management: stop-loss and take-profit levels per position.
"""
from config import MAX_RISK_PER_TRADE


def stop_loss_price(entry: float, atr: float | None, signal_type: str) -> float:
    """
    ATR-based stop loss.
    LONG:  entry - 1.5 * ATR
    FADE:  entry + 1.5 * ATR (stop out if it rips higher)
    Falls back to 5% hard stop if ATR unavailable.
    """
    if atr and atr > 0:
        multiplier = 1.5
        if signal_type == "LONG":
            return round(entry - multiplier * atr, 4)
        else:  # FADE
            return round(entry + multiplier * atr, 4)
    # Hard fallback
    if signal_type == "LONG":
        return round(entry * 0.95, 4)
    return round(entry * 1.05, 4)


def take_profit_price(entry: float, atr: float | None, signal_type: str, bars: int = 2) -> float:
    """
    2:1 reward-to-risk target using ATR.
    LONG:  entry + 3.0 * ATR
    FADE:  entry - 3.0 * ATR
    Falls back to 10% if ATR unavailable.
    """
    multiplier = max(2.0, bars * 1.5)
    if atr and atr > 0:
        if signal_type == "LONG":
            return round(entry + multiplier * atr, 4)
        else:
            return round(entry - multiplier * atr, 4)
    if signal_type == "LONG":
        return round(entry * 1.10, 4)
    return round(entry * 0.90, 4)


def should_stop(entry: float, current: float, stop: float, signal_type: str) -> bool:
    if signal_type == "LONG":
        return current <= stop
    return current >= stop


def should_take_profit(entry: float, current: float, target: float, signal_type: str) -> bool:
    if signal_type == "LONG":
        return current >= target
    return current <= target
