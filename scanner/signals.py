"""
Combines C/P ratio, sentiment, and technicals into actionable trade signals.

Signal types:
  FADE  – overbullish crowd, look for short-side reversal
  LONG  – momentum setup, upside continuation expected
  HOLD  – existing position, nothing new to act on
  PASS  – no edge detected
"""
import logging
from dataclasses import dataclass, field
from scanner.options_analyzer import compute_cp_ratio, cp_zscore, classify_cp, compute_put_call_skew
from scanner.sentiment import aggregate_sentiment
from scanner.technicals import compute_rsi, compute_atr, find_support_resistance, estimate_momentum_duration, rsi_label
from config import SENTIMENT_BULLISH_THRESHOLD, SENTIMENT_BEARISH_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    ticker: str
    signal_type: str          # "FADE" | "LONG" | "PASS"
    confidence: float         # 0.0 – 1.0
    price: float | None
    cp_ratio: float | None
    cp_zscore: float | None
    cp_class: str
    sentiment_score: float
    sentiment_label: str
    rsi: float | None
    rsi_label: str
    atr: float | None
    support: float | None
    resistance: float | None
    target_price: float | None
    bars_estimate: int
    direction: str
    top_headline: str
    reason: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _confidence(factors: list[bool]) -> float:
    """Fraction of confirming factors, rounded to 2 dp."""
    if not factors:
        return 0.0
    return round(sum(factors) / len(factors), 2)


def analyze_ticker(ticker: str, data: dict, cp_history: list) -> Signal:
    """
    data = {"history": pd.DataFrame, "options": dict, "news": list}
    cp_history = list of past C/P ratios for this ticker (append current one after this call)
    """
    history = data.get("history")
    options = data.get("options", {})
    news = data.get("news", [])

    # --- Price ---
    price = None
    if history is not None and not history.empty:
        price = float(history["Close"].iloc[-1])

    # --- Options ---
    cp_ratio = compute_cp_ratio(options)
    cp_history_full = cp_history + ([cp_ratio] if cp_ratio is not None else [])
    zscore = cp_zscore(cp_history_full)
    cp_class = classify_cp(cp_ratio, zscore)
    skew = compute_put_call_skew(options)

    # --- Sentiment ---
    sent = aggregate_sentiment(news)

    # --- Technicals ---
    rsi = None
    atr = None
    sr = {"support": None, "resistance": None}
    if history is not None and not history.empty:
        rsi = compute_rsi(history["Close"])
        atr = compute_atr(history)
        sr = find_support_resistance(history)

    momentum = estimate_momentum_duration(atr, rsi, price)
    r_label = rsi_label(rsi)

    # ------------------------------------------------------------------ #
    # FADE signal: crowded long + overbullish sentiment + overbought RSI  #
    # ------------------------------------------------------------------ #
    fade_factors = [
        cp_class == "OVERBULLISH",
        sent["score"] >= SENTIMENT_BULLISH_THRESHOLD,
        r_label == "OVERBOUGHT",
        (skew is not None and skew < -0.05),  # call IV > put IV = speculative
    ]
    fade_conf = _confidence(fade_factors)

    # ------------------------------------------------------------------ #
    # LONG signal: put activity dropping + positive news + not overbought #
    # ------------------------------------------------------------------ #
    long_factors = [
        cp_class in ("NEUTRAL", "BEARISH_PRESSURE"),
        sent["score"] >= SENTIMENT_BULLISH_THRESHOLD,
        rsi is not None and rsi < RSI_OVERBOUGHT_LOCAL,
        r_label != "OVERBOUGHT",
        (skew is not None and skew > 0.02),  # put IV elevated = smart hedging done
    ]
    long_conf = _confidence(long_factors)

    if fade_conf >= 0.60:
        sig_type = "FADE"
        confidence = fade_conf
        reason = (
            f"C/P={_fmt(cp_ratio)} ({cp_class}), "
            f"Sentiment={sent['label']} ({sent['score']:+.2f}), "
            f"RSI={_fmt(rsi)} ({r_label})"
        )
    elif long_conf >= 0.55:
        sig_type = "LONG"
        confidence = long_conf
        reason = (
            f"C/P={_fmt(cp_ratio)} ({cp_class}), "
            f"Sentiment={sent['label']} ({sent['score']:+.2f}), "
            f"RSI={_fmt(rsi)} ({r_label})"
        )
    else:
        sig_type = "PASS"
        confidence = max(fade_conf, long_conf)
        reason = "No edge: insufficient signal confluence"

    return Signal(
        ticker=ticker,
        signal_type=sig_type,
        confidence=confidence,
        price=round(price, 2) if price else None,
        cp_ratio=round(cp_ratio, 3) if cp_ratio else None,
        cp_zscore=round(zscore, 3) if zscore else None,
        cp_class=cp_class,
        sentiment_score=sent["score"],
        sentiment_label=sent["label"],
        rsi=rsi,
        rsi_label=r_label,
        atr=atr,
        support=sr["support"],
        resistance=sr["resistance"],
        target_price=momentum["target_price"],
        bars_estimate=momentum["bars_estimate"],
        direction=momentum["direction"],
        top_headline=sent["top_headline"],
        reason=reason,
        extra={"skew": skew},
    )


RSI_OVERBOUGHT_LOCAL = 70  # avoid circular import


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"
