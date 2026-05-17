import pytest
from portfolio.risk_manager import stop_loss_price, take_profit_price, should_stop, should_take_profit


class TestStopLoss:
    def test_long_stop_below_entry(self):
        stop = stop_loss_price(100.0, 2.0, "LONG")
        assert stop < 100.0

    def test_fade_stop_above_entry(self):
        stop = stop_loss_price(100.0, 2.0, "FADE")
        assert stop > 100.0

    def test_no_atr_fallback_long(self):
        stop = stop_loss_price(100.0, None, "LONG")
        assert stop == pytest.approx(95.0)

    def test_no_atr_fallback_fade(self):
        stop = stop_loss_price(100.0, None, "FADE")
        assert stop == pytest.approx(105.0)


class TestTakeProfit:
    def test_long_target_above_entry(self):
        target = take_profit_price(100.0, 2.0, "LONG")
        assert target > 100.0

    def test_fade_target_below_entry(self):
        target = take_profit_price(100.0, 2.0, "FADE")
        assert target < 100.0

    def test_reward_greater_than_risk_long(self):
        stop = stop_loss_price(100.0, 2.0, "LONG")
        target = take_profit_price(100.0, 2.0, "LONG")
        risk = 100.0 - stop
        reward = target - 100.0
        assert reward > risk  # 2:1+

    def test_no_atr_fallback(self):
        target = take_profit_price(100.0, None, "LONG")
        assert target == pytest.approx(110.0)


class TestShouldStop:
    def test_long_stops_when_below_stop(self):
        assert should_stop(100.0, 93.0, 95.0, "LONG") is True

    def test_long_does_not_stop_above_stop(self):
        assert should_stop(100.0, 98.0, 95.0, "LONG") is False

    def test_fade_stops_when_above_stop(self):
        assert should_stop(100.0, 107.0, 105.0, "FADE") is True


class TestShouldTakeProfit:
    def test_long_takes_profit_above_target(self):
        assert should_take_profit(100.0, 115.0, 110.0, "LONG") is True

    def test_long_does_not_take_profit_below_target(self):
        assert should_take_profit(100.0, 105.0, 110.0, "LONG") is False

    def test_fade_takes_profit_below_target(self):
        assert should_take_profit(100.0, 88.0, 90.0, "FADE") is True
