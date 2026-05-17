"""
Adaptive Thresholds — dynamically adjusts signal confidence thresholds
based on recent win rates from labeled outcomes.

Thresholds are persisted to thresholds.json in the project root.
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

THRESHOLDS_FILE = Path("thresholds.json")

DEFAULT_FADE_THRESHOLD = 0.60
DEFAULT_LONG_THRESHOLD = 0.55
MIN_THRESHOLD = 0.45
MAX_THRESHOLD = 0.80
ADJUSTMENT_STEP = 0.025
MIN_SAMPLES_TO_ADJUST = 15   # need at least 15 labeled signals before adjusting

RECENT_WINDOW = 30           # number of recent labeled outcomes to use
WIN_RATE_HIGH = 0.65         # lower threshold (catch more signals)
WIN_RATE_LOW = 0.40          # raise threshold (be more selective)


# ─────────────────────────────────────────────────────────────────────────────
# State dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThresholdState:
    fade_threshold: float = DEFAULT_FADE_THRESHOLD
    long_threshold: float = DEFAULT_LONG_THRESHOLD
    last_updated: str = ""
    fade_win_rate: float = 0.0
    long_win_rate: float = 0.0
    sample_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_thresholds() -> ThresholdState:
    """Read from thresholds.json; return defaults if missing or corrupt."""
    if THRESHOLDS_FILE.exists():
        try:
            with open(THRESHOLDS_FILE) as f:
                d = json.load(f)
            return ThresholdState(
                fade_threshold=d.get("fade_threshold", DEFAULT_FADE_THRESHOLD),
                long_threshold=d.get("long_threshold", DEFAULT_LONG_THRESHOLD),
                last_updated=d.get("last_updated", ""),
                fade_win_rate=d.get("fade_win_rate", 0.0),
                long_win_rate=d.get("long_win_rate", 0.0),
                sample_count=d.get("sample_count", 0),
            )
        except Exception as e:
            logger.error("Failed to load thresholds.json: %s", e)
    return ThresholdState()


def save_thresholds(state: ThresholdState) -> None:
    """Write ThresholdState to thresholds.json."""
    try:
        with open(THRESHOLDS_FILE, "w") as f:
            json.dump(asdict(state), f, indent=2)
    except Exception as e:
        logger.error("Failed to save thresholds.json: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold computation
# ─────────────────────────────────────────────────────────────────────────────

def _decisive_win_rate(entries: list[dict]) -> float | None:
    """Win rate over decisive outcomes (1 or -1). Returns None if no data."""
    decisive = [e for e in entries if e.get("outcome") in (1, -1)]
    if not decisive:
        return None
    wins = sum(1 for e in decisive if e.get("outcome") == 1)
    return wins / len(decisive)


def compute_thresholds(labeled_outcomes: list[dict]) -> dict:
    """
    Returns {"fade": float, "long": float} thresholds.

    For each signal type, uses the last RECENT_WINDOW outcomes:
      - win_rate > 65% → lower threshold by ADJUSTMENT_STEP (catch more)
      - win_rate < 40% → raise threshold by ADJUSTMENT_STEP (be selective)
      - Clamp to [MIN_THRESHOLD, MAX_THRESHOLD]

    If fewer than MIN_SAMPLES_TO_ADJUST total samples, return defaults.
    """
    if len(labeled_outcomes) < MIN_SAMPLES_TO_ADJUST:
        return {
            "fade": DEFAULT_FADE_THRESHOLD,
            "long": DEFAULT_LONG_THRESHOLD,
        }

    # Load current thresholds as starting point
    current = load_thresholds()
    fade_t = current.fade_threshold
    long_t = current.long_threshold

    # Use the most recent RECENT_WINDOW outcomes
    recent = labeled_outcomes[-RECENT_WINDOW:]

    fade_entries = [e for e in recent if e.get("signal_type") == "FADE"]
    long_entries = [e for e in recent if e.get("signal_type") == "LONG"]

    fade_wr = _decisive_win_rate(fade_entries)
    if fade_wr is not None:
        if fade_wr > WIN_RATE_HIGH:
            fade_t = max(MIN_THRESHOLD, fade_t - ADJUSTMENT_STEP)
        elif fade_wr < WIN_RATE_LOW:
            fade_t = min(MAX_THRESHOLD, fade_t + ADJUSTMENT_STEP)

    long_wr = _decisive_win_rate(long_entries)
    if long_wr is not None:
        if long_wr > WIN_RATE_HIGH:
            long_t = max(MIN_THRESHOLD, long_t - ADJUSTMENT_STEP)
        elif long_wr < WIN_RATE_LOW:
            long_t = min(MAX_THRESHOLD, long_t + ADJUSTMENT_STEP)

    return {
        "fade": round(fade_t, 4),
        "long": round(long_t, 4),
    }


def update_thresholds(labeled_outcomes: list[dict]) -> ThresholdState:
    """Compute new thresholds, save them, and return the updated ThresholdState."""
    thresholds = compute_thresholds(labeled_outcomes)

    recent = labeled_outcomes[-RECENT_WINDOW:]
    fade_wr = _decisive_win_rate([e for e in recent if e.get("signal_type") == "FADE"]) or 0.0
    long_wr = _decisive_win_rate([e for e in recent if e.get("signal_type") == "LONG"]) or 0.0

    state = ThresholdState(
        fade_threshold=thresholds["fade"],
        long_threshold=thresholds["long"],
        last_updated=datetime.now(timezone.utc).isoformat(),
        fade_win_rate=round(fade_wr, 4),
        long_win_rate=round(long_wr, 4),
        sample_count=len(labeled_outcomes),
    )
    save_thresholds(state)
    logger.info(
        "Thresholds updated: FADE=%.3f (wr=%.1f%%) LONG=%.3f (wr=%.1f%%)",
        state.fade_threshold, fade_wr * 100,
        state.long_threshold, long_wr * 100,
    )
    return state


def get_current_thresholds() -> dict:
    """
    Returns {"fade": float, "long": float} from thresholds.json.
    Falls back to defaults if file missing.
    """
    state = load_thresholds()
    return {
        "fade": state.fade_threshold,
        "long": state.long_threshold,
    }
