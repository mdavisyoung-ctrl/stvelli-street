import pytest
import pandas as pd
import numpy as np
from scanner.technicals import compute_rsi, compute_atr, find_support_resistance, estimate_momentum_duration, rsi_label


def _make_price_series(values):
    return pd.Series(values, dtype=float)


def _make_ohlcv(n=30, trend="up"):
    base = 100.0
    closes = [base + (i if trend == "up" else -i) * 0.5 for i in range(n)]
    df = pd.DataFrame({
        "Open": closes,
        "High": [c + 1 for c in closes],
        "Low": [c - 1 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n,
    })
    return df


class TestComputeRsi:
    def test_uptrend_rsi_above_50(self):
        df = _make_ohlcv(30, "up")
        rsi = compute_rsi(df["Close"])
        assert rsi is not None
        assert rsi > 50

    def test_downtrend_rsi_below_50(self):
        df = _make_ohlcv(30, "down")
        rsi = compute_rsi(df["Close"])
        assert rsi is not None
        assert rsi < 50

    def test_insufficient_data_returns_none(self):
        assert compute_rsi(_make_price_series([100, 101, 102])) is None

    def test_flat_prices_rsi(self):
        df = _make_ohlcv(30)
        closes = pd.Series([100.0] * 30)
        rsi = compute_rsi(closes)
        # With no gains/losses, RSI is edge-case — should return something or handle gracefully
        assert rsi is None or (0 <= rsi <= 100)


class TestComputeAtr:
    def test_atr_positive(self):
        df = _make_ohlcv(30)
        atr = compute_atr(df)
        assert atr is not None
        assert atr > 0

    def test_insufficient_data_returns_none(self):
        df = _make_ohlcv(5)
        assert compute_atr(df) is None


class TestSupportResistance:
    def test_returns_correct_range(self):
        df = _make_ohlcv(30)
        sr = find_support_resistance(df)
        assert sr["support"] is not None
        assert sr["resistance"] is not None
        assert sr["resistance"] > sr["support"]

    def test_insufficient_data(self):
        df = _make_ohlcv(5)
        sr = find_support_resistance(df, lookback=20)
        assert sr["support"] is None


class TestMomentumDuration:
    def test_upward_direction_when_rsi_above_50(self):
        result = estimate_momentum_duration(atr=1.5, rsi=65.0, price=100.0)
        assert result["direction"] == "UP"
        assert result["target_price"] > 100.0
        assert result["bars_estimate"] >= 1

    def test_downward_direction_when_rsi_below_50(self):
        result = estimate_momentum_duration(atr=1.5, rsi=35.0, price=100.0)
        assert result["direction"] == "DOWN"
        assert result["target_price"] < 100.0

    def test_none_inputs_returns_unknown(self):
        result = estimate_momentum_duration(None, None, None)
        assert result["direction"] == "UNKNOWN"
        assert result["bars_estimate"] == 0


class TestRsiLabel:
    def test_overbought(self):
        assert rsi_label(75) == "OVERBOUGHT"

    def test_oversold(self):
        assert rsi_label(25) == "OVERSOLD"

    def test_neutral(self):
        assert rsi_label(50) == "NEUTRAL"

    def test_none(self):
        assert rsi_label(None) == "UNKNOWN"
