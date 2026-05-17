import pytest
from scanner.signals import Signal, macro_to_signal
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


def _make_macro(regime, ticker="SPY", price=500.0, rsi=55.0):
    return {
        "ticker": ticker,
        "regime": regime,
        "price": price,
        "rsi": rsi,
        "rsi_label": "NEUTRAL",
        "atr": 5.0,
        "support": 490.0,
        "resistance": 510.0,
        "sentiment_score": 0.3,
        "sentiment_label": "BULLISH",
        "top_headline": "Markets rally",
        "cp_ratio": 1.1,
        "cp_class": "NEUTRAL",
        "momentum": {"target_price": 520.0, "bars_estimate": 3, "direction": "UP"},
    }


def _make_ml(label, confidence=0.70, prediction=1):
    return {"label": label, "confidence": confidence, "prediction": prediction}


class TestMacroToSignal:
    def test_long_signal_when_risk_on_and_ml_up(self):
        sig = macro_to_signal(_make_macro("RISK_ON"), _make_ml("UP"))
        assert sig is not None
        assert sig.signal_type == "LONG"
        assert sig.ticker == "SPY"

    def test_fade_signal_when_risk_off_and_ml_down(self):
        sig = macro_to_signal(_make_macro("RISK_OFF"), _make_ml("DOWN", prediction=-1))
        assert sig is not None
        assert sig.signal_type == "FADE"

    def test_returns_none_when_low_ml_confidence(self):
        sig = macro_to_signal(_make_macro("RISK_ON"), _make_ml("UP", confidence=0.4))
        assert sig is None

    def test_returns_none_when_regime_conflicts_ml(self):
        # ML says UP but regime is RISK_OFF → no signal
        sig = macro_to_signal(_make_macro("RISK_OFF"), _make_ml("UP"))
        assert sig is None

    def test_returns_none_when_ml_flat(self):
        sig = macro_to_signal(_make_macro("RISK_ON"), _make_ml("FLAT", confidence=0.8))
        assert sig is None

    def test_btc_xrp_generate_signals(self):
        for ticker in ("BTC", "XRP"):
            macro = _make_macro("BULLISH", ticker=ticker, price=2.5 if ticker == "XRP" else 95000)
            sig = macro_to_signal(macro, _make_ml("UP"))
            assert sig is not None
            assert sig.ticker == ticker

    def test_signal_has_valid_price(self):
        sig = macro_to_signal(_make_macro("RISK_ON"), _make_ml("UP"))
        assert sig.price == 500.0

    def test_source_tagged_in_extra(self):
        sig = macro_to_signal(_make_macro("RISK_ON"), _make_ml("UP"))
        assert sig.extra.get("source") == "macro"
