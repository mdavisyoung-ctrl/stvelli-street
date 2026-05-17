import pytest
from scanner.signals import Signal
from strategies.fade_strategy import evaluate_fade, FadeSetup
from strategies.momentum_strategy import evaluate_momentum, MomentumSetup


def _make_signal(sig_type, confidence=0.75, price=150.0, atr=2.0, bars=2):
    return Signal(
        ticker="AAPL",
        signal_type=sig_type,
        confidence=confidence,
        price=price,
        cp_ratio=1.8,
        cp_zscore=1.7,
        cp_class="OVERBULLISH" if sig_type == "FADE" else "NEUTRAL",
        sentiment_score=0.35,
        sentiment_label="BULLISH",
        rsi=72.0 if sig_type == "FADE" else 58.0,
        rsi_label="OVERBOUGHT" if sig_type == "FADE" else "NEUTRAL",
        atr=atr,
        support=145.0,
        resistance=155.0,
        target_price=144.0 if sig_type == "FADE" else 156.0,
        bars_estimate=bars,
        direction="DOWN" if sig_type == "FADE" else "UP",
        top_headline="Test headline",
        reason="Test reason",
    )


class TestFadeStrategy:
    def test_returns_setup_for_fade_signal(self):
        sig = _make_signal("FADE")
        setup = evaluate_fade(sig)
        assert setup is not None
        assert isinstance(setup, FadeSetup)
        assert setup.stop_loss > setup.entry_price  # fade stop is above entry
        assert setup.take_profit < setup.entry_price

    def test_returns_none_for_long_signal(self):
        sig = _make_signal("LONG")
        assert evaluate_fade(sig) is None

    def test_returns_none_for_low_confidence(self):
        sig = _make_signal("FADE", confidence=0.4)
        assert evaluate_fade(sig) is None

    def test_returns_none_when_no_price(self):
        sig = _make_signal("FADE")
        sig.price = None
        assert evaluate_fade(sig) is None


class TestMomentumStrategy:
    def test_returns_setup_for_long_signal(self):
        sig = _make_signal("LONG")
        setup = evaluate_momentum(sig)
        assert setup is not None
        assert isinstance(setup, MomentumSetup)
        assert setup.stop_loss < setup.entry_price
        assert setup.take_profit > setup.entry_price

    def test_returns_none_for_fade_signal(self):
        sig = _make_signal("FADE")
        assert evaluate_momentum(sig) is None

    def test_returns_none_for_low_confidence(self):
        sig = _make_signal("LONG", confidence=0.3)
        assert evaluate_momentum(sig) is None
