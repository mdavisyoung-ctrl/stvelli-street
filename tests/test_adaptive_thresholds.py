"""
Tests for portfolio/adaptive_thresholds.py
"""
import json
import pytest
from datetime import datetime, timezone

from portfolio.adaptive_thresholds import (
    DEFAULT_FADE_THRESHOLD,
    DEFAULT_LONG_THRESHOLD,
    MIN_THRESHOLD,
    MAX_THRESHOLD,
    ADJUSTMENT_STEP,
    MIN_SAMPLES_TO_ADJUST,
    ThresholdState,
    compute_thresholds,
    update_thresholds,
    load_thresholds,
    save_thresholds,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_labeled(signal_type, outcome, n=1):
    """Create n labeled outcome dicts of the given signal_type and outcome."""
    return [
        {
            "ticker": "AAPL",
            "signal_type": signal_type,
            "outcome": outcome,
            "confidence": 0.7,
        }
    ] * n


def _labeled_mix(signal_type, wins, losses):
    """wins correct, losses incorrect of given type."""
    return (
        _make_labeled(signal_type, 1, wins) +
        _make_labeled(signal_type, -1, losses)
    )


# ─── compute_thresholds ────────────────────────────────────────────────────────

class TestComputeThresholds:

    def test_returns_defaults_when_fewer_than_min_samples(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        labeled = _make_labeled("LONG", 1, MIN_SAMPLES_TO_ADJUST - 1)
        result = compute_thresholds(labeled)
        assert result == {"fade": DEFAULT_FADE_THRESHOLD, "long": DEFAULT_LONG_THRESHOLD}

    def test_returns_defaults_when_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        result = compute_thresholds([])
        assert result == {"fade": DEFAULT_FADE_THRESHOLD, "long": DEFAULT_LONG_THRESHOLD}

    def test_lowers_long_threshold_when_win_rate_above_65pct(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        # 80% win rate on LONG (8 wins, 2 losses = 10 decisive), plus enough samples
        labeled = _labeled_mix("LONG", 8, 2) + _make_labeled("FADE", 1, 5)
        result = compute_thresholds(labeled)
        assert result["long"] == round(DEFAULT_LONG_THRESHOLD - ADJUSTMENT_STEP, 4)
        # FADE should not change (no 30+ samples, only 5 → not enough decisive fade)
        # Actually 5 FADE wins with no losses → 100% win rate → lowers
        # We test LONG specifically changes down
        assert result["long"] < DEFAULT_LONG_THRESHOLD

    def test_raises_long_threshold_when_win_rate_below_40pct(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        # 20% win rate on LONG (2 wins, 8 losses)
        labeled = _labeled_mix("LONG", 2, 8) + _make_labeled("FADE", 1, 5)
        result = compute_thresholds(labeled)
        assert result["long"] == round(DEFAULT_LONG_THRESHOLD + ADJUSTMENT_STEP, 4)
        assert result["long"] > DEFAULT_LONG_THRESHOLD

    def test_lowers_fade_threshold_when_win_rate_above_65pct(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        # 90% win rate on FADE (9 wins, 1 loss)
        labeled = _labeled_mix("FADE", 9, 1) + _make_labeled("LONG", 1, 5)
        result = compute_thresholds(labeled)
        assert result["fade"] == round(DEFAULT_FADE_THRESHOLD - ADJUSTMENT_STEP, 4)

    def test_raises_fade_threshold_when_win_rate_below_40pct(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        # 30% win rate on FADE (3 wins, 7 losses)
        labeled = _labeled_mix("FADE", 3, 7) + _make_labeled("LONG", 1, 5)
        result = compute_thresholds(labeled)
        assert result["fade"] == round(DEFAULT_FADE_THRESHOLD + ADJUSTMENT_STEP, 4)

    def test_clamps_long_threshold_to_min(self, tmp_path, monkeypatch):
        """When current threshold is already at MIN, it shouldn't go lower."""
        thresholds_file = tmp_path / "thresholds.json"
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE", thresholds_file)

        # Write a state with long threshold already at MIN
        state = ThresholdState(
            fade_threshold=DEFAULT_FADE_THRESHOLD,
            long_threshold=MIN_THRESHOLD,
        )
        save_thresholds(state)

        # 90% long win rate → would lower threshold, but already at min
        labeled = _labeled_mix("LONG", 9, 1) + _make_labeled("FADE", 1, 5)
        result = compute_thresholds(labeled)
        assert result["long"] == MIN_THRESHOLD

    def test_clamps_fade_threshold_to_max(self, tmp_path, monkeypatch):
        """When current threshold is already at MAX, it shouldn't go higher."""
        thresholds_file = tmp_path / "thresholds.json"
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE", thresholds_file)

        # Write a state with fade threshold already at MAX
        state = ThresholdState(
            fade_threshold=MAX_THRESHOLD,
            long_threshold=DEFAULT_LONG_THRESHOLD,
        )
        save_thresholds(state)

        # 10% fade win rate → would raise threshold, but already at max
        labeled = _labeled_mix("FADE", 1, 9) + _make_labeled("LONG", 1, 5)
        result = compute_thresholds(labeled)
        assert result["fade"] == MAX_THRESHOLD

    def test_no_change_when_win_rate_between_40_and_65(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        # 50% win rate — neither adjustment zone
        labeled = _labeled_mix("LONG", 5, 5) + _make_labeled("FADE", 1, 5)
        result = compute_thresholds(labeled)
        assert result["long"] == DEFAULT_LONG_THRESHOLD


# ─── update_thresholds ────────────────────────────────────────────────────────

class TestUpdateThresholds:

    def test_returns_threshold_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        labeled = _labeled_mix("LONG", 5, 5) + _make_labeled("FADE", 1, 10)
        state = update_thresholds(labeled)
        assert isinstance(state, ThresholdState)

    def test_saves_thresholds_to_file(self, tmp_path, monkeypatch):
        thresholds_file = tmp_path / "thresholds.json"
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE", thresholds_file)

        labeled = _labeled_mix("LONG", 5, 5) + _make_labeled("FADE", 1, 10)
        state = update_thresholds(labeled)

        assert thresholds_file.exists()
        data = json.loads(thresholds_file.read_text())
        assert data["fade_threshold"] == state.fade_threshold
        assert data["long_threshold"] == state.long_threshold

    def test_state_has_win_rates(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        labeled = _labeled_mix("LONG", 8, 2) + _labeled_mix("FADE", 3, 7)
        state = update_thresholds(labeled)
        assert 0 <= state.long_win_rate <= 1.0
        assert 0 <= state.fade_win_rate <= 1.0

    def test_state_has_sample_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        labeled = _make_labeled("LONG", 1, MIN_SAMPLES_TO_ADJUST)
        state = update_thresholds(labeled)
        assert state.sample_count == len(labeled)

    def test_state_has_last_updated(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        labeled = _make_labeled("LONG", 1, MIN_SAMPLES_TO_ADJUST)
        state = update_thresholds(labeled)
        assert state.last_updated != ""
        # Should be parseable as ISO datetime
        dt = datetime.fromisoformat(state.last_updated)
        assert dt is not None


# ─── load_thresholds ──────────────────────────────────────────────────────────

class TestLoadThresholds:

    def test_returns_defaults_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE",
                            tmp_path / "thresholds.json")
        state = load_thresholds()
        assert state.fade_threshold == DEFAULT_FADE_THRESHOLD
        assert state.long_threshold == DEFAULT_LONG_THRESHOLD
        assert state.sample_count == 0
        assert state.fade_win_rate == 0.0
        assert state.long_win_rate == 0.0

    def test_loads_saved_state(self, tmp_path, monkeypatch):
        thresholds_file = tmp_path / "thresholds.json"
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE", thresholds_file)

        original = ThresholdState(
            fade_threshold=0.57,
            long_threshold=0.52,
            last_updated="2024-01-15T10:00:00+00:00",
            fade_win_rate=0.72,
            long_win_rate=0.64,
            sample_count=42,
        )
        save_thresholds(original)

        loaded = load_thresholds()
        assert loaded.fade_threshold == 0.57
        assert loaded.long_threshold == 0.52
        assert loaded.fade_win_rate == 0.72
        assert loaded.long_win_rate == 0.64
        assert loaded.sample_count == 42

    def test_returns_defaults_when_file_corrupt(self, tmp_path, monkeypatch):
        thresholds_file = tmp_path / "thresholds.json"
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE", thresholds_file)
        thresholds_file.write_text("not valid json{{")

        state = load_thresholds()
        assert state.fade_threshold == DEFAULT_FADE_THRESHOLD
        assert state.long_threshold == DEFAULT_LONG_THRESHOLD

    def test_partial_file_uses_defaults_for_missing_fields(self, tmp_path, monkeypatch):
        thresholds_file = tmp_path / "thresholds.json"
        monkeypatch.setattr("portfolio.adaptive_thresholds.THRESHOLDS_FILE", thresholds_file)
        thresholds_file.write_text(json.dumps({"fade_threshold": 0.58}))

        state = load_thresholds()
        assert state.fade_threshold == 0.58
        assert state.long_threshold == DEFAULT_LONG_THRESHOLD
