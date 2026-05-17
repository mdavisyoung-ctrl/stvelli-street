import pytest
import pandas as pd
import numpy as np
from scanner.options_analyzer import compute_cp_ratio, cp_zscore, classify_cp, compute_put_call_skew


def _make_options(call_oi, put_oi, call_iv=None, put_iv=None):
    calls = pd.DataFrame({"openInterest": [call_oi], "impliedVolatility": [call_iv or 0.3]})
    puts = pd.DataFrame({"openInterest": [put_oi], "impliedVolatility": [put_iv or 0.3]})
    return {"calls": calls, "puts": puts}


class TestComputeCpRatio:
    def test_basic_ratio(self):
        opts = _make_options(1000, 500)
        assert compute_cp_ratio(opts) == pytest.approx(2.0)

    def test_zero_put_oi_returns_none(self):
        opts = _make_options(1000, 0)
        assert compute_cp_ratio(opts) is None

    def test_empty_options_returns_none(self):
        assert compute_cp_ratio({}) is None

    def test_balanced_ratio(self):
        opts = _make_options(750, 750)
        assert compute_cp_ratio(opts) == pytest.approx(1.0)


class TestCpZscore:
    def test_high_recent_value_positive_zscore(self):
        history = [1.0, 1.0, 1.0, 1.0, 3.0]
        z = cp_zscore(history)
        assert z > 1.5

    def test_too_short_history_returns_none(self):
        assert cp_zscore([1.0, 1.0]) is None

    def test_constant_history_zero_zscore(self):
        history = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert cp_zscore(history) == pytest.approx(0.0)


class TestClassifyCp:
    def test_overbullish_by_zscore(self):
        assert classify_cp(2.5, 1.8) == "OVERBULLISH"

    def test_bearish_pressure_by_zscore(self):
        assert classify_cp(0.4, -1.5) == "BEARISH_PRESSURE"

    def test_neutral(self):
        assert classify_cp(1.0, 0.3) == "NEUTRAL"

    def test_none_ratio_returns_neutral(self):
        assert classify_cp(None, None) == "NEUTRAL"

    def test_absolute_threshold_overbullish(self):
        assert classify_cp(2.5, None) == "OVERBULLISH"

    def test_absolute_threshold_bearish(self):
        assert classify_cp(0.3, None) == "BEARISH_PRESSURE"


class TestSkew:
    def test_positive_skew_when_put_iv_higher(self):
        calls = pd.DataFrame({"openInterest": [100, 100, 100, 100], "impliedVolatility": [0.2, 0.2, 0.2, 0.2]})
        puts = pd.DataFrame({"openInterest": [100, 100, 100, 100], "impliedVolatility": [0.4, 0.4, 0.4, 0.4]})
        opts = {"calls": calls, "puts": puts}
        skew = compute_put_call_skew(opts)
        assert skew > 0

    def test_no_iv_column_returns_none(self):
        calls = pd.DataFrame({"openInterest": [100]})
        puts = pd.DataFrame({"openInterest": [100]})
        assert compute_put_call_skew({"calls": calls, "puts": puts}) is None
