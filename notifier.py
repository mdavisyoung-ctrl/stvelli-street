"""
Telegram notification module.
Sends trade alerts to your phone when signals fire.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Message @userinfobot on Telegram → copy your chat_id
  3. Add to .env:
       TELEGRAM_TOKEN=your_bot_token
       TELEGRAM_CHAT_ID=your_chat_id
"""
import os
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_BASE = "https://api.telegram.org"


def _enabled() -> bool:
    return bool(_TOKEN and _CHAT_ID)


def send(text: str) -> bool:
    if not _enabled():
        return False
    try:
        url = f"{_BASE}/bot{_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        return resp.ok
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def alert_signal(ticker: str, signal_type: str, price: float,
                 stop: float, target: float, confidence: float,
                 reason: str, headline: str = "") -> bool:
    direction = "🔴 FADE (Short)" if signal_type == "FADE" else "🟢 LONG (Buy)"
    ts = datetime.now().strftime("%H:%M:%S")
    text = (
        f"<b>Stvelli Street Signal</b>  {ts}\n\n"
        f"{direction}  <b>${ticker}</b>\n"
        f"Price:      ${price:.2f}\n"
        f"Stop loss:  ${stop:.2f}\n"
        f"Target:     ${target:.2f}\n"
        f"Confidence: {confidence:.0%}\n"
        f"Reason:     {reason}\n"
    )
    if headline:
        text += f"\n📰 {headline[:120]}"
    text += f"\n\n<i>→ enter {ticker} &lt;fill_price&gt;</i>"
    return send(text)


def alert_stop_hit(ticker: str, price: float, pnl: float) -> bool:
    icon = "🛑"
    text = (
        f"{icon} <b>STOP HIT — Close in Robinhood NOW</b>\n\n"
        f"<b>{ticker}</b> @ ${price:.2f}\n"
        f"P&amp;L: ${pnl:+.2f}"
    )
    return send(text)


def alert_target_hit(ticker: str, price: float, pnl: float) -> bool:
    text = (
        f"🎯 <b>TARGET HIT — Consider closing</b>\n\n"
        f"<b>{ticker}</b> @ ${price:.2f}\n"
        f"P&amp;L: ${pnl:+.2f}"
    )
    return send(text)


def alert_portfolio(cash: float, value: float, win_rate: float) -> bool:
    pnl = value - 50.0
    icon = "📈" if pnl >= 0 else "📉"
    text = (
        f"{icon} <b>Portfolio Update</b>\n\n"
        f"Value:    ${value:.2f}  ({pnl:+.2f})\n"
        f"Cash:     ${cash:.2f}\n"
        f"Win rate: {win_rate:.1f}%"
    )
    return send(text)
