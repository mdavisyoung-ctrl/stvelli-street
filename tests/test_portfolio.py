import pytest
from portfolio.tracker import (
    _default_state, position_size, open_position, close_position,
    portfolio_value, win_rate
)


class TestPositionSize:
    def test_ten_percent_of_cash(self):
        state = _default_state()
        state["cash"] = 50.0
        size = position_size(state, 100.0)
        assert size == pytest.approx(5.0)

    def test_minimum_one_dollar(self):
        state = _default_state()
        state["cash"] = 5.0  # 10% = $0.50, but min is $1
        size = position_size(state, 100.0)
        assert size >= 1.0

    def test_cannot_exceed_cash(self):
        state = _default_state()
        state["cash"] = 3.0
        size = position_size(state, 100.0)
        assert size <= state["cash"]


class TestOpenClosePosition:
    def test_open_deducts_cash(self):
        state = _default_state()
        state["cash"] = 50.0
        state = open_position(state, "AAPL", 150.0, "LONG")
        assert state["cash"] < 50.0
        assert "AAPL" in state["positions"]

    def test_close_returns_cash(self):
        state = _default_state()
        state["cash"] = 50.0
        state = open_position(state, "AAPL", 100.0, "LONG")
        cash_after_open = state["cash"]
        state = close_position(state, "AAPL", 110.0)
        assert state["cash"] > cash_after_open
        assert "AAPL" not in state["positions"]

    def test_winning_trade_increments_counter(self):
        state = _default_state()
        state = open_position(state, "AAPL", 100.0, "LONG")
        state = close_position(state, "AAPL", 110.0)
        assert state["winning_trades"] == 1

    def test_losing_trade_does_not_increment_winner(self):
        state = _default_state()
        state = open_position(state, "AAPL", 100.0, "LONG")
        state = close_position(state, "AAPL", 90.0)
        assert state["winning_trades"] == 0

    def test_cannot_open_duplicate_position(self):
        state = _default_state()
        state = open_position(state, "AAPL", 100.0, "LONG")
        cash_before = state["cash"]
        state = open_position(state, "AAPL", 100.0, "LONG")
        assert state["cash"] == cash_before  # no second deduction

    def test_close_nonexistent_position_safe(self):
        state = _default_state()
        cash_before = state["cash"]
        state = close_position(state, "FAKE", 100.0)
        assert state["cash"] == cash_before


class TestPortfolioValue:
    def test_value_includes_open_positions(self):
        state = _default_state()
        state = open_position(state, "AAPL", 100.0, "LONG")
        pos = state["positions"]["AAPL"]
        # Mark up 10%
        val = portfolio_value(state, {"AAPL": 110.0})
        assert val > state["cash"]

    def test_value_equals_cash_when_no_positions(self):
        state = _default_state()
        assert portfolio_value(state, {}) == pytest.approx(state["cash"])


class TestWinRate:
    def test_zero_trades(self):
        state = _default_state()
        assert win_rate(state) == 0.0

    def test_all_wins(self):
        state = _default_state()
        state["total_trades"] = 4
        state["winning_trades"] = 4
        assert win_rate(state) == 100.0

    def test_mixed(self):
        state = _default_state()
        state["total_trades"] = 4
        state["winning_trades"] = 3
        assert win_rate(state) == 75.0
