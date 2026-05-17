"""
Tests for scanner/outcome_tracker.py
"""
import json
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path

from scanner.signals import Signal


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_signal(signal_type="LONG", ticker="AAPL", confidence=0.75, price=150.0):
    return Signal(
        ticker=ticker,
        signal_type=signal_type,
        confidence=confidence,
        price=price,
        cp_ratio=1.2,
        cp_zscore=0.8,
        cp_class="NEUTRAL",
        sentiment_score=0.3,
        sentiment_label="BULLISH",
        rsi=58.0,
        rsi_label="NEUTRAL",
        atr=2.25,           # atr_pct = 2.25/150 = 0.015
        support=145.0,
        resistance=155.0,
        target_price=155.0,
        bars_estimate=5,
        direction="UP",
        top_headline="AAPL beats earnings",
        reason="test reason",
        extra={"skew": 0.02},
    )


def _empty_outcomes():
    return {"pending": [], "labeled": []}


# ─── record_signal ─────────────────────────────────────────────────────────────

class TestRecordSignal:

    def test_records_long_signal_to_pending(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        import scanner.outcome_tracker as ot
        sig = _make_signal("LONG")
        result_id = ot.record_signal(sig)

        assert result_id is not None
        assert len(result_id) == 36  # UUID format

        data = json.loads(outcomes_file.read_text())
        assert len(data["pending"]) == 1
        assert len(data["labeled"]) == 0
        entry = data["pending"][0]
        assert entry["ticker"] == "AAPL"
        assert entry["signal_type"] == "LONG"
        assert entry["confidence"] == 0.75
        assert entry["price_at_signal"] == 150.0
        assert entry["id"] == result_id

    def test_records_fade_signal_to_pending(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        import scanner.outcome_tracker as ot
        sig = _make_signal("FADE", ticker="TSLA", confidence=0.65, price=200.0)
        result_id = ot.record_signal(sig)

        assert result_id is not None
        data = json.loads(outcomes_file.read_text())
        assert len(data["pending"]) == 1
        assert data["pending"][0]["signal_type"] == "FADE"
        assert data["pending"][0]["ticker"] == "TSLA"

    def test_skips_pass_signals(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        import scanner.outcome_tracker as ot
        sig = _make_signal("PASS")
        result_id = ot.record_signal(sig)

        assert result_id is None
        # File should not have been created (or if it exists, has no pending entries)
        if outcomes_file.exists():
            data = json.loads(outcomes_file.read_text())
            assert len(data["pending"]) == 0

    def test_returns_uuid_string(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        import scanner.outcome_tracker as ot
        sig = _make_signal("LONG")
        result_id = ot.record_signal(sig)

        # Should be a valid UUID
        parsed = uuid.UUID(result_id)
        assert str(parsed) == result_id

    def test_resolve_after_is_25_minutes_from_now(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        import scanner.outcome_tracker as ot
        before = datetime.now(timezone.utc)
        sig = _make_signal("LONG")
        ot.record_signal(sig)
        after = datetime.now(timezone.utc)

        data = json.loads(outcomes_file.read_text())
        resolve_after = datetime.fromisoformat(data["pending"][0]["resolve_after"])
        expected_min = before + timedelta(minutes=25)
        expected_max = after + timedelta(minutes=25)

        assert expected_min <= resolve_after <= expected_max

    def test_features_stored_correctly(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        import scanner.outcome_tracker as ot
        sig = _make_signal("LONG")
        ot.record_signal(sig)

        data = json.loads(outcomes_file.read_text())
        features = data["pending"][0]["features"]
        assert features["cp_ratio"] == 1.2
        assert features["cp_zscore"] == 0.8
        assert features["sentiment_score"] == 0.3
        assert features["rsi"] == 58.0
        assert features["skew"] == 0.02
        # atr_pct = atr / price = 2.25 / 150 = 0.015
        assert abs(features["atr_pct"] - 0.015) < 1e-6


# ─── resolve_pending ───────────────────────────────────────────────────────────

class TestResolvePending:

    def _write_pending(self, outcomes_file, entries):
        data = {"pending": entries, "labeled": []}
        outcomes_file.write_text(json.dumps(data))

    def _make_pending_entry(self, ticker="AAPL", signal_type="LONG",
                             price=150.0, resolve_offset_minutes=-1):
        """resolve_offset_minutes < 0 means resolve_after is in the past."""
        resolve_after = datetime.now(timezone.utc) + timedelta(minutes=resolve_offset_minutes)
        return {
            "id": str(uuid.uuid4()),
            "ticker": ticker,
            "signal_type": signal_type,
            "confidence": 0.75,
            "price_at_signal": price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": {
                "cp_ratio": 1.2, "cp_zscore": 0.8,
                "sentiment_score": 0.3, "rsi": 58.0,
                "atr_pct": 0.015, "skew": 0.02,
            },
            "resolve_after": resolve_after.isoformat(),
        }

    # ── LONG outcomes ──

    def test_long_correct_when_price_up_more_than_half_pct(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        entry = self._make_pending_entry("AAPL", "LONG", 150.0)
        self._write_pending(outcomes_file, [entry])

        import scanner.outcome_tracker as ot
        newly = ot.resolve_pending({"AAPL": 151.0})  # +0.67% > 0.5%

        assert len(newly) == 1
        assert newly[0]["outcome"] == 1
        assert newly[0]["exit_price"] == 151.0

    def test_long_incorrect_when_price_down_more_than_half_pct(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        entry = self._make_pending_entry("AAPL", "LONG", 150.0)
        self._write_pending(outcomes_file, [entry])

        import scanner.outcome_tracker as ot
        newly = ot.resolve_pending({"AAPL": 148.5})  # -1% < -0.5%

        assert len(newly) == 1
        assert newly[0]["outcome"] == -1

    def test_long_flat_when_move_less_than_half_pct(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        entry = self._make_pending_entry("AAPL", "LONG", 150.0)
        self._write_pending(outcomes_file, [entry])

        import scanner.outcome_tracker as ot
        newly = ot.resolve_pending({"AAPL": 150.3})  # +0.2% — flat

        assert len(newly) == 1
        assert newly[0]["outcome"] == 0

    # ── FADE outcomes ──

    def test_fade_correct_when_price_down_more_than_half_pct(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        entry = self._make_pending_entry("TSLA", "FADE", 200.0)
        self._write_pending(outcomes_file, [entry])

        import scanner.outcome_tracker as ot
        newly = ot.resolve_pending({"TSLA": 198.5})  # -0.75% < -0.5%

        assert len(newly) == 1
        assert newly[0]["outcome"] == 1

    def test_fade_incorrect_when_price_up_more_than_half_pct(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        entry = self._make_pending_entry("TSLA", "FADE", 200.0)
        self._write_pending(outcomes_file, [entry])

        import scanner.outcome_tracker as ot
        newly = ot.resolve_pending({"TSLA": 202.0})  # +1% > 0.5%

        assert len(newly) == 1
        assert newly[0]["outcome"] == -1

    def test_fade_flat_when_move_less_than_half_pct(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        entry = self._make_pending_entry("TSLA", "FADE", 200.0)
        self._write_pending(outcomes_file, [entry])

        import scanner.outcome_tracker as ot
        newly = ot.resolve_pending({"TSLA": 200.2})  # +0.1% — flat

        assert len(newly) == 1
        assert newly[0]["outcome"] == 0

    # ── Lifecycle: moves from pending to labeled ──

    def test_resolved_entry_moves_to_labeled(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        entry = self._make_pending_entry("AAPL", "LONG", 150.0)
        self._write_pending(outcomes_file, [entry])

        import scanner.outcome_tracker as ot
        ot.resolve_pending({"AAPL": 151.0})

        data = json.loads(outcomes_file.read_text())
        assert len(data["pending"]) == 0
        assert len(data["labeled"]) == 1

    def test_future_entries_stay_pending(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        past_entry = self._make_pending_entry("AAPL", "LONG", 150.0, resolve_offset_minutes=-5)
        future_entry = self._make_pending_entry("MSFT", "LONG", 300.0, resolve_offset_minutes=20)
        self._write_pending(outcomes_file, [past_entry, future_entry])

        import scanner.outcome_tracker as ot
        newly = ot.resolve_pending({"AAPL": 151.0, "MSFT": 305.0})

        assert len(newly) == 1
        assert newly[0]["ticker"] == "AAPL"

        data = json.loads(outcomes_file.read_text())
        assert len(data["pending"]) == 1
        assert data["pending"][0]["ticker"] == "MSFT"
        assert len(data["labeled"]) == 1

    def test_entry_stays_pending_when_price_missing(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)

        entry = self._make_pending_entry("AAPL", "LONG", 150.0)
        self._write_pending(outcomes_file, [entry])

        import scanner.outcome_tracker as ot
        newly = ot.resolve_pending({})  # AAPL not in prices

        assert len(newly) == 0
        data = json.loads(outcomes_file.read_text())
        assert len(data["pending"]) == 1

    def test_returns_empty_when_nothing_to_resolve(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        self._write_pending(outcomes_file, [])

        import scanner.outcome_tracker as ot
        newly = ot.resolve_pending({"AAPL": 150.0})
        assert newly == []


# ─── win_rate_by_type ──────────────────────────────────────────────────────────

class TestWinRateByType:

    def _write_labeled(self, outcomes_file, labeled):
        data = {"pending": [], "labeled": labeled}
        outcomes_file.write_text(json.dumps(data))

    def _labeled_entry(self, signal_type, outcome, ticker="AAPL"):
        return {
            "ticker": ticker, "signal_type": signal_type,
            "outcome": outcome, "confidence": 0.7,
            "price_at_signal": 150.0, "exit_price": 151.0,
            "pnl_pct": 0.67, "labeled_at": datetime.now(timezone.utc).isoformat(),
        }

    def test_empty_data_returns_zeros(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        self._write_labeled(outcomes_file, [])

        import scanner.outcome_tracker as ot
        result = ot.win_rate_by_type()
        assert result == {"LONG": 0.0, "FADE": 0.0, "overall": 0.0}

    def test_all_long_wins(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        labeled = [self._labeled_entry("LONG", 1) for _ in range(4)]
        self._write_labeled(outcomes_file, labeled)

        import scanner.outcome_tracker as ot
        result = ot.win_rate_by_type()
        assert result["LONG"] == 1.0
        assert result["FADE"] == 0.0
        assert result["overall"] == 1.0

    def test_mixed_long_win_rate(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        labeled = (
            [self._labeled_entry("LONG", 1)] * 3 +
            [self._labeled_entry("LONG", -1)] * 1
        )
        self._write_labeled(outcomes_file, labeled)

        import scanner.outcome_tracker as ot
        result = ot.win_rate_by_type()
        assert abs(result["LONG"] - 0.75) < 1e-6

    def test_flat_outcomes_excluded_from_denominator(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        labeled = [
            self._labeled_entry("LONG", 1),
            self._labeled_entry("LONG", 0),  # flat — should not count
            self._labeled_entry("LONG", -1),
        ]
        self._write_labeled(outcomes_file, labeled)

        import scanner.outcome_tracker as ot
        result = ot.win_rate_by_type()
        # Only 2 decisive: 1 win → 50%
        assert abs(result["LONG"] - 0.5) < 1e-6

    def test_fade_win_rate_separate_from_long(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        labeled = [
            self._labeled_entry("LONG", 1),
            self._labeled_entry("LONG", -1),
            self._labeled_entry("FADE", 1),
            self._labeled_entry("FADE", 1),
        ]
        self._write_labeled(outcomes_file, labeled)

        import scanner.outcome_tracker as ot
        result = ot.win_rate_by_type()
        assert abs(result["LONG"] - 0.5) < 1e-6
        assert abs(result["FADE"] - 1.0) < 1e-6


# ─── get_labeled ──────────────────────────────────────────────────────────────

class TestGetLabeled:

    def _write_labeled(self, outcomes_file, labeled):
        data = {"pending": [], "labeled": labeled}
        outcomes_file.write_text(json.dumps(data))

    def _entry(self, ticker, outcome=1):
        return {
            "ticker": ticker, "signal_type": "LONG",
            "outcome": outcome, "confidence": 0.7,
            "price_at_signal": 150.0, "exit_price": 151.0,
            "pnl_pct": 0.67, "labeled_at": datetime.now(timezone.utc).isoformat(),
        }

    def test_returns_all_labeled(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        labeled = [self._entry("AAPL"), self._entry("MSFT"), self._entry("TSLA")]
        self._write_labeled(outcomes_file, labeled)

        import scanner.outcome_tracker as ot
        result = ot.get_labeled()
        assert len(result) == 3

    def test_filters_by_ticker(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        labeled = [
            self._entry("AAPL"),
            self._entry("AAPL"),
            self._entry("MSFT"),
        ]
        self._write_labeled(outcomes_file, labeled)

        import scanner.outcome_tracker as ot
        result = ot.get_labeled(ticker="AAPL")
        assert len(result) == 2
        assert all(r["ticker"] == "AAPL" for r in result)

    def test_filters_by_ticker_returns_empty_for_unknown(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        labeled = [self._entry("AAPL")]
        self._write_labeled(outcomes_file, labeled)

        import scanner.outcome_tracker as ot
        result = ot.get_labeled(ticker="TSLA")
        assert result == []

    def test_limits_to_last_n(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        labeled = [self._entry("AAPL") for _ in range(10)]
        labeled[-1]["outcome"] = -1  # mark the last one distinctly
        self._write_labeled(outcomes_file, labeled)

        import scanner.outcome_tracker as ot
        result = ot.get_labeled(last_n=3)
        assert len(result) == 3
        # Should be the last 3 entries
        assert result[-1]["outcome"] == -1

    def test_ticker_and_last_n_combined(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        labeled = (
            [self._entry("AAPL")] * 5 +
            [self._entry("MSFT")] * 5
        )
        self._write_labeled(outcomes_file, labeled)

        import scanner.outcome_tracker as ot
        result = ot.get_labeled(ticker="AAPL", last_n=2)
        assert len(result) == 2
        assert all(r["ticker"] == "AAPL" for r in result)

    def test_returns_empty_when_no_labeled(self, tmp_path, monkeypatch):
        outcomes_file = tmp_path / "outcomes.json"
        monkeypatch.setattr("scanner.outcome_tracker.OUTCOMES_FILE", outcomes_file)
        self._write_labeled(outcomes_file, [])

        import scanner.outcome_tracker as ot
        result = ot.get_labeled()
        assert result == []
