"""
Local portfolio state backed by a JSON file.
Tracks cash, open positions, and closed trade history.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from config import STARTING_CAPITAL, MAX_RISK_PER_TRADE, MAX_OPEN_POSITIONS

logger = logging.getLogger(__name__)
STATE_FILE = Path("portfolio_state.json")


def _default_state() -> dict:
    return {
        "cash": STARTING_CAPITAL,
        "positions": {},   # {ticker: {entry_price, shares, cost_basis, opened_at, signal_type}}
        "history": [],     # closed trades
        "total_trades": 0,
        "winning_trades": 0,
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.error("State load failed: %s", e)
    return _default_state()


def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error("State save failed: %s", e)


def position_size(state: dict, price: float) -> float:
    """
    Returns dollar amount to risk on the next trade.
    Caps at MAX_RISK_PER_TRADE * cash, and never more than cash itself.
    Minimum $1 so we can always make a move.
    """
    risk_dollars = state["cash"] * MAX_RISK_PER_TRADE
    return max(1.0, min(risk_dollars, state["cash"]))


def open_position(state: dict, ticker: str, entry_price: float, signal_type: str) -> dict:
    """
    Opens a position sized by risk rules.
    Returns the updated state dict.
    """
    if ticker in state["positions"]:
        logger.warning("Already have an open position in %s", ticker)
        return state

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        logger.warning("Max open positions reached")
        return state

    dollars = position_size(state, entry_price)
    shares = dollars / entry_price  # fractional for tracking purposes

    state["positions"][ticker] = {
        "entry_price": round(entry_price, 4),
        "shares": round(shares, 6),
        "cost_basis": round(dollars, 4),
        "opened_at": datetime.utcnow().isoformat(),
        "signal_type": signal_type,
    }
    state["cash"] = round(state["cash"] - dollars, 4)
    state["total_trades"] += 1
    save_state(state)
    logger.info("Opened %s %s @ %.4f (${:.2f})", signal_type, ticker, entry_price, dollars)
    return state


def close_position(state: dict, ticker: str, exit_price: float) -> dict:
    """
    Closes an open position and records the trade in history.
    """
    pos = state["positions"].get(ticker)
    if not pos:
        logger.warning("No open position for %s", ticker)
        return state

    proceeds = pos["shares"] * exit_price
    pnl = proceeds - pos["cost_basis"]
    pnl_pct = (pnl / pos["cost_basis"]) * 100 if pos["cost_basis"] else 0

    state["cash"] = round(state["cash"] + proceeds, 4)
    if pnl > 0:
        state["winning_trades"] += 1

    closed = {
        **pos,
        "ticker": ticker,
        "exit_price": round(exit_price, 4),
        "proceeds": round(proceeds, 4),
        "pnl": round(pnl, 4),
        "pnl_pct": round(pnl_pct, 2),
        "closed_at": datetime.utcnow().isoformat(),
    }
    state["history"].append(closed)
    del state["positions"][ticker]
    save_state(state)
    logger.info("Closed %s @ %.4f  PnL: $%.2f (%.1f%%)", ticker, exit_price, pnl, pnl_pct)
    return state


def portfolio_value(state: dict, current_prices: dict) -> float:
    """Total portfolio value: cash + mark-to-market open positions."""
    total = state["cash"]
    for ticker, pos in state["positions"].items():
        price = current_prices.get(ticker, pos["entry_price"])
        total += pos["shares"] * price
    return round(total, 2)


def win_rate(state: dict) -> float:
    if state["total_trades"] == 0:
        return 0.0
    return round(state["winning_trades"] / state["total_trades"] * 100, 1)
