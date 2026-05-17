import pytest
import numpy as np
import pandas as pd
from scanner.ml_pattern import (
    _build_features, _build_labels, _rolling_rsi, _rolling_atr,
    train_model, predict, WINDOW, FORWARD
)


def _make_hist(n=100, trend="up"):
    base = 100.0
    closes = [base + (i if trend == "up" else -i) * 0.3 for i in range(n)]
    df = pd.DataFrame({
        "Open": closes,
        "High": [c + 0.5 for c in closes],
        "Low": [c - 0.5 for c in closes],
        "Close": closes,
        "Volume": [1_000_000] * n,
    })
    return df


class TestRollingHelpers:
    def test_rsi_uptrend_above_50(self):
        closes = np.array([100 + i * 0.5 for i in range(30)])
        rsi = _rolling_rsi(closes)
        assert rsi > 50

    def test_rsi_insufficient_data_returns_50(self):
        assert _rolling_rsi(np.array([100.0, 101.0])) == 50.0

    def test_atr_positive(self):
        df = _make_hist(30)
        atr = _rolling_atr(df)
        assert atr > 0


class TestBuildFeatures:
    def test_shape(self):
        hist = _make_hist(60)
        X = _build_features(hist, [0.0] * 60, [1.0] * 60)
        assert X.shape[0] == 60 - WINDOW
        assert X.shape[1] == 7

    def test_no_nans(self):
        hist = _make_hist(60)
        X = _build_features(hist, [0.0] * 60, [1.0] * 60)
        assert not np.isnan(X).any()


class TestBuildLabels:
    def test_length_matches_features(self):
        hist = _make_hist(60)
        X = _build_features(hist, [], [])
        y = _build_labels(hist)
        assert len(y) == len(X)

    def test_uptrend_generates_positive_labels(self):
        hist = _make_hist(100, "up")
        y = _build_labels(hist)
        # Strong uptrend should produce more +1 than -1
        assert (y == 1).sum() > (y == -1).sum()


class TestTrainPredict:
    def test_train_returns_pipeline(self):
        hist = _make_hist(120)
        pipe = train_model("TEST", hist)
        assert pipe is not None

    def test_predict_returns_dict_with_expected_keys(self):
        hist = _make_hist(120)
        train_model("TEST", hist)
        result = predict("TEST", hist)
        assert "prediction" in result
        assert "label" in result
        assert "confidence" in result
        assert result["label"] in ("UP", "DOWN", "FLAT")

    def test_predict_insufficient_data(self):
        hist = _make_hist(10)
        result = predict("TINY", hist)
        assert result["label"] == "FLAT"
        assert result["confidence"] == 0.0

    def test_confidence_between_0_and_1(self):
        hist = _make_hist(120)
        train_model("TEST2", hist)
        result = predict("TEST2", hist)
        assert 0.0 <= result["confidence"] <= 1.0
