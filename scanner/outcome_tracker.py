"""
Outcome Tracker — records signals and labels them correct/incorrect after
25 minutes (5 scan intervals at 5 min each).

State is persisted to outcomes.json in the project root.
"""
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from scanner.signals import Signal

logger = logging.getLogger(__name__)

OUTCOMES_FILE = Path("outcomes.json")
RESOLVE_DELAY_MINUTES = 25
CORRECT_THRESHOLD_PCT = 0.005   # 0.5%


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_outcomes() -> dict:
    """Load outcomes from JSON file, returning empty structure if missing."""
    if OUTCOMES_FILE.exists():
        try:
            with open(OUTCOMES_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load outcomes.json: %s", e)
    return {"pending": [], "labeled": []}


def save_outcomes(data: dict) -> None:
    """Persist outcomes dict to JSON file."""
    try:
        with open(OUTCOMES_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.error("Failed to save outcomes.json: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Recording
# ─────────────────────────────────────────────────────────────────────────────

def record_signal(signal: Signal) -> str | None:
    """
    Save a LONG or FADE signal to the pending list.
    PASS signals are skipped (returns None).
    Returns the generated uuid string for the recorded signal.
    """
    if signal.signal_type not in ("LONG", "FADE"):
        return None

    now = datetime.now(timezone.utc)
    resolve_after = now + timedelta(minutes=RESOLVE_DELAY_MINUTES)

    skew = signal.extra.get("skew") if signal.extra else None
    atr_pct = None
    if signal.atr is not None and signal.price and signal.price > 0:
        atr_pct = round(signal.atr / signal.price, 6)

    entry = {
        "id": str(uuid.uuid4()),
        "ticker": signal.ticker,
        "signal_type": signal.signal_type,
        "confidence": signal.confidence,
        "price_at_signal": signal.price,
        "timestamp": now.isoformat(),
        "features": {
            "cp_ratio": signal.cp_ratio,
            "cp_zscore": signal.cp_zscore,
            "sentiment_score": signal.sentiment_score,
            "rsi": signal.rsi,
            "atr_pct": atr_pct,
            "skew": skew,
        },
        "resolve_after": resolve_after.isoformat(),
    }

    data = load_outcomes()
    data["pending"].append(entry)
    save_outcomes(data)
    logger.debug("Recorded %s signal for %s (id=%s)", signal.signal_type, signal.ticker, entry["id"])
    return entry["id"]


# ─────────────────────────────────────────────────────────────────────────────
# Resolving
# ─────────────────────────────────────────────────────────────────────────────

def resolve_pending(current_prices: dict) -> list[dict]:
    """
    Check all pending signals whose resolve_after has passed.
    Labels each one based on price movement and moves them to labeled.
    Returns list of newly labeled outcome dicts.
    """
    now = datetime.now(timezone.utc)
    data = load_outcomes()
    newly_labeled = []
    still_pending = []

    for entry in data["pending"]:
        resolve_after_str = entry.get("resolve_after", "")
        try:
            resolve_after = datetime.fromisoformat(resolve_after_str)
            # Ensure timezone-aware
            if resolve_after.tzinfo is None:
                resolve_after = resolve_after.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            still_pending.append(entry)
            continue

        if now < resolve_after:
            still_pending.append(entry)
            continue

        ticker = entry["ticker"]
        if ticker not in current_prices:
            # Can't resolve without current price; keep pending
            still_pending.append(entry)
            continue

        entry_price = entry.get("price_at_signal")
        exit_price = current_prices[ticker]

        if entry_price is None or entry_price == 0:
            still_pending.append(entry)
            continue

        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4)
        sig_type = entry["signal_type"]
        price_change = (exit_price - entry_price) / entry_price

        if sig_type == "LONG":
            if price_change > CORRECT_THRESHOLD_PCT:
                outcome = 1
            elif price_change < -CORRECT_THRESHOLD_PCT:
                outcome = -1
            else:
                outcome = 0
        elif sig_type == "FADE":
            if price_change < -CORRECT_THRESHOLD_PCT:
                outcome = 1
            elif price_change > CORRECT_THRESHOLD_PCT:
                outcome = -1
            else:
                outcome = 0
        else:
            outcome = 0

        labeled_entry = dict(entry)
        labeled_entry["exit_price"] = exit_price
        labeled_entry["outcome"] = outcome
        labeled_entry["pnl_pct"] = pnl_pct
        labeled_entry["labeled_at"] = now.isoformat()

        newly_labeled.append(labeled_entry)
        data["labeled"].append(labeled_entry)
        logger.info(
            "Resolved %s %s: entry=%.4f exit=%.4f pnl=%.2f%% outcome=%d",
            sig_type, ticker, entry_price, exit_price, pnl_pct, outcome,
        )

    data["pending"] = still_pending
    save_outcomes(data)
    return newly_labeled


# ─────────────────────────────────────────────────────────────────────────────
# Querying
# ─────────────────────────────────────────────────────────────────────────────

def get_labeled(ticker: str = None, last_n: int = None) -> list[dict]:
    """
    Return labeled outcomes, optionally filtered by ticker or limited to last N.
    """
    data = load_outcomes()
    results = data.get("labeled", [])
    if ticker is not None:
        results = [r for r in results if r.get("ticker") == ticker]
    if last_n is not None:
        results = results[-last_n:]
    return results


def win_rate_by_type() -> dict:
    """
    Returns {"LONG": float, "FADE": float, "overall": float} win rates (0.0–1.0).
    Only counts outcome=1 as wins; outcome=-1 as losses; outcome=0 skipped in denominator.
    Returns 0.0 for types with no data.
    """
    data = load_outcomes()
    labeled = data.get("labeled", [])

    def _win_rate(entries):
        decisive = [e for e in entries if e.get("outcome") in (1, -1)]
        if not decisive:
            return 0.0
        wins = sum(1 for e in decisive if e.get("outcome") == 1)
        return round(wins / len(decisive), 4)

    long_entries = [e for e in labeled if e.get("signal_type") == "LONG"]
    fade_entries = [e for e in labeled if e.get("signal_type") == "FADE"]

    return {
        "LONG": _win_rate(long_entries),
        "FADE": _win_rate(fade_entries),
        "overall": _win_rate(labeled),
    }
