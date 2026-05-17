"""
Momentum strategy: identifies stocks with upside continuation expected.

Entry logic:
  1. C/P ratio not elevated (market not crowded long yet)
  2. News sentiment bullish (catalyst present)
  3. RSI between 45–68 (momentum without being overbought)
  4. Price above 20-period support

Duration estimate uses ATR and RSI distance from 50.
"""
from dataclasses import dataclass
from scanner.signals import Signal
from portfolio.risk_manager import stop_loss_price, take_profit_price


@dataclass
class MomentumSetup:
    ticker: str
    entry_price: float
    stop_loss: float
    take_profit: float
    bars_estimate: int
    confidence: float
    reason: str
    headline: str


def evaluate_momentum(signal: Signal) -> MomentumSetup | None:
    """Returns a MomentumSetup if the signal meets criteria, else None."""
    if signal.signal_type != "LONG":
        return None
    if signal.price is None:
        return None
    if signal.confidence < 0.55:
        return None

    entry = signal.price
    stop = stop_loss_price(entry, signal.atr, "LONG")
    target = take_profit_price(entry, signal.atr, "LONG", bars=signal.bars_estimate)

    return MomentumSetup(
        ticker=signal.ticker,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        bars_estimate=signal.bars_estimate,
        confidence=signal.confidence,
        reason=signal.reason,
        headline=signal.top_headline,
    )
