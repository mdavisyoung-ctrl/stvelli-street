"""
Paper trading engine for SPY, BTC, and XRP.

Runs a simulated portfolio in parallel with the real one.
Uses the ML model predictions and macro signals to enter/exit paper positions.
Tracks P&L separately so we can measure model accuracy before going live.

Paper positions are held in paper_state.json (separate from portfolio_state.json).
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from config import STARTING_CAPITAL

logger = logging.getLogger(__name__)
PAPER_STATE_FILE = Path("paper_state.json")

PAPER_CAPITAL = 50.0  # mirrors real portfolio


def _default_paper_state() -> dict:
    return {
        "cash": PAPER_CAPITAL,
        "positions": {},
        "history": [],
        "total_trades": 0,
        "winning_trades": 0,
        "model_accuracy": {},  # {ticker: {"correct": int, "total": int}}
    }


def load_paper_state() -> dict:
    if PAPER_STATE_FILE.exists():
        try:
            with open(PAPER_STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.error("Paper state load failed: %s", e)
    return _default_paper_state()


def save_paper_state(state: dict) -> None:
    try:
        with open(PAPER_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error("Paper state save failed: %s", e)


def paper_open(state: dict, ticker: str, price: float, direction: str,
               confidence: float, ml_prediction: int) -> dict:
    """
    Opens a paper trade. direction = "LONG" or "FADE".
    """
    if ticker in state["positions"]:
        return state

    risk = min(state["cash"] * 0.15, state["cash"])  # 15% risk on paper trades
    shares = risk / price if price > 0 else 0

    state["positions"][ticker] = {
        "entry_price": round(price, 6),
        "shares": round(shares, 8),
        "cost_basis": round(risk, 4),
        "direction": direction,
        "confidence": confidence,
        "ml_prediction": ml_prediction,
        "opened_at": datetime.utcnow().isoformat(),
    }
    state["cash"] = round(state["cash"] - risk, 4)
    state["total_trades"] += 1
    save_paper_state(state)
    return state


def paper_close(state: dict, ticker: str, price: float) -> dict:
    pos = state["positions"].get(ticker)
    if not pos:
        return state

    proceeds = pos["shares"] * price
    pnl = proceeds - pos["cost_basis"]
    if pos["direction"] == "FADE":
        pnl = -pnl  # fade = short, profit when price drops

    won = pnl > 0
    if won:
        state["winning_trades"] += 1

    # Track model accuracy
    ml_pred = pos.get("ml_prediction", 0)
    actual = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
    acc = state["model_accuracy"].setdefault(ticker, {"correct": 0, "total": 0})
    acc["total"] += 1
    if ml_pred != 0 and ml_pred == actual:
        acc["correct"] += 1

    state["cash"] = round(state["cash"] + max(0, proceeds), 4)
    state["history"].append({
        "ticker": ticker,
        **pos,
        "exit_price": round(price, 6),
        "pnl": round(pnl, 4),
        "won": won,
        "closed_at": datetime.utcnow().isoformat(),
    })
    del state["positions"][ticker]
    save_paper_state(state)
    return state


def paper_portfolio_value(state: dict, prices: dict) -> float:
    total = state["cash"]
    for ticker, pos in state["positions"].items():
        cur = prices.get(ticker, pos["entry_price"])
        total += pos["shares"] * cur
    return round(total, 4)


def model_accuracy_report(state: dict) -> dict:
    """Returns per-ticker accuracy % for ML predictions."""
    report = {}
    for ticker, acc in state["model_accuracy"].items():
        if acc["total"] > 0:
            report[ticker] = round(acc["correct"] / acc["total"] * 100, 1)
    return report


def should_paper_enter(macro_result: dict, ml_result: dict) -> tuple[bool, str]:
    """
    Decides whether to open a paper trade based on combined macro + ML signal.
    Returns (should_enter: bool, direction: str).
    """
    regime = macro_result.get("regime", "NEUTRAL")
    ml_label = ml_result.get("label", "FLAT")
    ml_conf = ml_result.get("confidence", 0.0)

    # Need ML confidence > 55% and regime alignment
    if ml_conf < 0.55:
        return False, "PASS"

    if ml_label == "UP" and regime in ("RISK_ON", "BULLISH", "OVERSOLD"):
        return True, "LONG"
    if ml_label == "DOWN" and regime in ("RISK_OFF", "OVERBOUGHT"):
        return True, "FADE"

    return False, "PASS"
