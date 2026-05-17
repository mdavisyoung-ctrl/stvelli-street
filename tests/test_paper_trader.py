import pytest
from portfolio.paper_trader import (
    _default_paper_state, paper_open, paper_close,
    paper_portfolio_value, model_accuracy_report, should_paper_enter
)


class TestPaperOpenClose:
    def test_open_deducts_cash(self):
        state = _default_paper_state()
        state = paper_open(state, "BTC", 50000.0, "LONG", 0.7, 1)
        assert state["cash"] < 50.0
        assert "BTC" in state["positions"]

    def test_close_returns_cash_on_win(self):
        state = _default_paper_state()
        state = paper_open(state, "BTC", 50000.0, "LONG", 0.7, 1)
        state = paper_close(state, "BTC", 55000.0)
        assert "BTC" not in state["positions"]
        assert state["winning_trades"] == 1

    def test_no_duplicate_open(self):
        state = _default_paper_state()
        state = paper_open(state, "BTC", 50000.0, "LONG", 0.7, 1)
        cash_after_first = state["cash"]
        state = paper_open(state, "BTC", 50000.0, "LONG", 0.7, 1)
        assert state["cash"] == cash_after_first

    def test_fade_position_profits_on_price_drop(self):
        state = _default_paper_state()
        state = paper_open(state, "SPY", 500.0, "FADE", 0.8, -1)
        state = paper_close(state, "SPY", 480.0)
        assert state["winning_trades"] == 1

    def test_model_accuracy_tracked(self):
        state = _default_paper_state()
        state = paper_open(state, "XRP", 2.0, "LONG", 0.7, 1)
        state = paper_close(state, "XRP", 2.5)  # up → actual=1, pred=1 → correct
        acc = model_accuracy_report(state)
        assert "XRP" in acc
        assert acc["XRP"] == 100.0


class TestPortfolioValue:
    def test_value_above_cash_when_position_is_up(self):
        state = _default_paper_state()
        state = paper_open(state, "BTC", 50000.0, "LONG", 0.7, 1)
        val = paper_portfolio_value(state, {"BTC": 60000.0})
        assert val > state["cash"]

    def test_value_equals_cash_when_no_positions(self):
        state = _default_paper_state()
        assert paper_portfolio_value(state, {}) == pytest.approx(state["cash"])


class TestShouldPaperEnter:
    def test_enters_long_on_bullish_risk_on(self):
        macro = {"regime": "RISK_ON", "price": 500.0}
        ml = {"label": "UP", "confidence": 0.7, "prediction": 1}
        enter, direction = should_paper_enter(macro, ml)
        assert enter is True
        assert direction == "LONG"

    def test_enters_fade_on_risk_off_down(self):
        macro = {"regime": "RISK_OFF", "price": 500.0}
        ml = {"label": "DOWN", "confidence": 0.7, "prediction": -1}
        enter, direction = should_paper_enter(macro, ml)
        assert enter is True
        assert direction == "FADE"

    def test_no_entry_low_confidence(self):
        macro = {"regime": "RISK_ON", "price": 500.0}
        ml = {"label": "UP", "confidence": 0.4, "prediction": 1}
        enter, _ = should_paper_enter(macro, ml)
        assert enter is False

    def test_no_entry_conflicting_signals(self):
        macro = {"regime": "RISK_OFF", "price": 500.0}
        ml = {"label": "UP", "confidence": 0.8, "prediction": 1}
        enter, _ = should_paper_enter(macro, ml)
        assert enter is False
