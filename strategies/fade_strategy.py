"""
Fade strategy: identifies crowded-long setups for short-side plays.

Entry logic:
  1. C/P ratio z-score >= CP_FADE_ZSCORE (too many calls vs puts)
  2. News sentiment bullish (crowd is overly optimistic)
  3. RSI >= 65 (not waiting for full overbought, catch it early)
  4. Price near resistance

The trade idea surfaced here is informational — the user executes in Robinhood.
This is a STOCK short (or put buy) idea, not a leveraged derivative.
"""
from dataclasses import dataclass
from scanner.signals import Signal
from portfolio.risk_manager import stop_loss_price, take_profit_price


@dataclass
class FadeSetup:
    ticker: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    headline: str


def evaluate_fade(signal: Signal) -> FadeSetup | None:
    """Returns a FadeSetup if the signal meets fade criteria, else None."""
    if signal.signal_type != "FADE":
        return None
    if signal.price is None:
        return None
    if signal.confidence < 0.60:
        return None

    entry = signal.price
    stop = stop_loss_price(entry, signal.atr, "FADE")
    target = take_profit_price(entry, signal.atr, "FADE", bars=signal.bars_estimate)

    return FadeSetup(
        ticker=signal.ticker,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        confidence=signal.confidence,
        reason=signal.reason,
        headline=signal.top_headline,
    )
