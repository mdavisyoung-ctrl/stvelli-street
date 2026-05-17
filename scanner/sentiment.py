"""
News sentiment scoring using FinBERT (primary) with TextBlob fallback.
Includes source credibility weighting, deduplication, time decay,
volume spike detection, and PR domination flagging.
"""
import logging
import math
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FinBERT lazy loading
# ---------------------------------------------------------------------------
_finbert_pipeline = None
_finbert_available = None  # None = not yet attempted; True/False after attempt


def _load_finbert():
    """Attempt to load the FinBERT pipeline; sets module-level flags."""
    global _finbert_pipeline, _finbert_available
    if _finbert_available is not None:
        return  # already attempted
    try:
        from transformers import pipeline  # noqa: PLC0415
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=-1,
            top_k=None,  # return all labels
        )
        _finbert_available = True
        logger.info("FinBERT pipeline loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.info("FinBERT unavailable (%s); falling back to TextBlob.", exc)
        _finbert_available = False


# ---------------------------------------------------------------------------
# Source credibility
# ---------------------------------------------------------------------------

# (domain_fragment → weight)  — checked with 'in' against the netloc
_SOURCE_WEIGHTS: list[tuple[str, float]] = [
    ("reuters.com", 3.0),
    ("bloomberg.com", 3.0),
    ("wsj.com", 3.0),
    ("cnbc.com", 2.5),
    ("marketwatch.com", 2.5),
    ("seekingalpha.com", 1.5),
    ("benzinga.com", 1.5),
    # Press-release wires — low credibility, flagged as PR
    ("businesswire.com", 0.5),
    ("prnewswire.com", 0.5),
    ("globenewswire.com", 0.5),
    ("accesswire.com", 0.5),
]

_PR_DOMAINS = {"businesswire.com", "prnewswire.com", "globenewswire.com", "accesswire.com"}

_DEFAULT_WEIGHT = 1.0


def get_source_weight(link: str) -> tuple[float, bool]:
    """
    Returns (weight, is_pr) for the given article URL.
    is_pr is True for known press-release wire domains.
    """
    if not link:
        return _DEFAULT_WEIGHT, False
    try:
        netloc = urlparse(link).netloc.lower()
        # Strip www. prefix for matching
        netloc = re.sub(r"^www\.", "", netloc)
    except Exception:  # noqa: BLE001
        return _DEFAULT_WEIGHT, False

    for domain, weight in _SOURCE_WEIGHTS:
        if domain in netloc:
            return weight, domain in _PR_DOMAINS

    return _DEFAULT_WEIGHT, False


# ---------------------------------------------------------------------------
# Time decay
# ---------------------------------------------------------------------------

_HALF_LIFE_HOURS = 12.0


def _time_decay_weight(date_str: str) -> float:
    """
    Exponential decay with a 12-hour half-life.
    Returns a weight in (0, 1] — 1.0 for 'now', ~0.5 for 12 h ago, etc.
    Returns 1.0 if the date cannot be parsed.
    """
    if not date_str:
        return 1.0
    try:
        # EODHD format: "2024-01-15T10:00:00+00:00"
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        hours_ago = max((now - dt).total_seconds() / 3600.0, 0.0)
        return math.pow(0.5, hours_ago / _HALF_LIFE_HOURS)
    except Exception:  # noqa: BLE001
        return 1.0


# ---------------------------------------------------------------------------
# Deduplication via Jaccard similarity
# ---------------------------------------------------------------------------

_JACCARD_THRESHOLD = 0.70


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """
    Remove near-duplicate headlines (Jaccard similarity > 70%).
    When two articles are duplicates, keep the one with the higher
    source credibility weight. Preserves ordering of kept articles.
    """
    kept: list[dict] = []
    kept_tokens: list[set] = []

    for article in articles:
        tokens = _tokenize(article.get("title", ""))
        is_dup = False
        for i, existing_tokens in enumerate(kept_tokens):
            union = existing_tokens | tokens
            if not union:
                continue
            similarity = len(existing_tokens & tokens) / len(union)
            if similarity > _JACCARD_THRESHOLD:
                # Duplicate found — keep higher-credibility source
                existing_weight, _ = get_source_weight(kept[i].get("link", ""))
                new_weight, _ = get_source_weight(article.get("link", ""))
                if new_weight > existing_weight:
                    kept[i] = article
                    kept_tokens[i] = tokens
                is_dup = True
                break
        if not is_dup:
            kept.append(article)
            kept_tokens.append(tokens)

    return kept


# ---------------------------------------------------------------------------
# Volume spike detection
# ---------------------------------------------------------------------------

def detect_volume_spike(article_count: int, volume_history: list[int]) -> dict:
    """
    Returns {"is_spike": bool, "z_score": float}.
    A spike is defined as z_score >= 2.0.
    If history is insufficient (< 2 samples), z_score is 0.0 and is_spike False.
    """
    if not volume_history or len(volume_history) < 2:
        return {"is_spike": False, "z_score": 0.0}

    n = len(volume_history)
    mean = sum(volume_history) / n
    variance = sum((x - mean) ** 2 for x in volume_history) / n
    std = math.sqrt(variance)

    if std == 0.0:
        return {"is_spike": False, "z_score": 0.0}

    z = (article_count - mean) / std
    return {"is_spike": z >= 2.0, "z_score": round(z, 4)}


# ---------------------------------------------------------------------------
# Article scoring
# ---------------------------------------------------------------------------

def _score_with_finbert(title: str, content: str) -> float:
    """Score using FinBERT. Returns polarity in [-1, 1]."""
    text_title = title.strip() if title else ""
    text_content = (content or "")[:300].strip()

    def _score_text(text: str) -> float:
        if not text:
            return 0.0
        results = _finbert_pipeline(text, truncation=True, max_length=512)
        # top_k=None returns [[{...}, ...]] — unwrap outer list if nested
        if results and isinstance(results[0], list):
            results = results[0]
        label_map = {r["label"].lower(): r["score"] for r in results}
        pos = label_map.get("positive", 0.0)
        neg = label_map.get("negative", 0.0)
        return pos - neg

    title_score = _score_text(text_title)
    content_score = _score_text(text_content)
    # Title weighted 2x over content
    if text_content:
        return (2 * title_score + content_score) / 3
    return title_score


def _score_with_textblob(title: str, content: str) -> float:
    """Score using TextBlob. Returns polarity in [-1, 1]."""
    from textblob import TextBlob  # noqa: PLC0415
    title_score = TextBlob(title).sentiment.polarity if title else 0.0
    content_score = TextBlob((content or "")[:500]).sentiment.polarity if content else 0.0
    return (2 * title_score + content_score) / 3


def score_article(title: str, content: str) -> float:
    """
    Returns polarity in [-1, 1].
    Uses FinBERT if available, falls back to TextBlob.
    Title weighted 2x over first 300 chars (FinBERT) / 500 chars (TextBlob) of content.
    """
    _load_finbert()
    if _finbert_available:
        return _score_with_finbert(title, content)
    return _score_with_textblob(title, content)


# ---------------------------------------------------------------------------
# Aggregate sentiment
# ---------------------------------------------------------------------------

def aggregate_sentiment(articles: list[dict], volume_history: list[int] = None) -> dict:
    """
    Aggregate news sentiment with credibility weighting, time decay,
    deduplication, volume spike detection, and PR domination flagging.

    Parameters
    ----------
    articles       : list of dicts with keys: title, content, date, link
    volume_history : historical article counts for z-score spike detection

    Returns
    -------
    dict with keys:
        score         – weighted polarity in [-1, 1]
        label         – "BULLISH" / "BEARISH" / "NEUTRAL"
        article_count – raw article count before dedup
        clean_count   – article count after dedup
        top_headline  – headline from the most credible + recent source
        volume_spike  – {"is_spike": bool, "z_score": float}
        pr_dominated  – True if >50% of raw articles are from PR wires
        scorer        – "finbert" or "textblob"
    """
    # Ensure FinBERT loading has been attempted
    _load_finbert()
    scorer_name = "finbert" if _finbert_available else "textblob"

    article_count = len(articles)

    if not articles:
        return {
            "score": 0.0,
            "label": "NEUTRAL",
            "article_count": 0,
            "clean_count": 0,
            "top_headline": "",
            "volume_spike": detect_volume_spike(0, volume_history or []),
            "pr_dominated": False,
            "scorer": scorer_name,
        }

    # --- PR domination check (on raw articles) ---
    pr_count = sum(1 for a in articles if get_source_weight(a.get("link", ""))[1])
    pr_dominated = (pr_count / article_count) > 0.5

    # --- Deduplication ---
    deduped = deduplicate_articles(articles)
    clean_count = len(deduped)

    # --- Score each article ---
    scored: list[tuple[float, float, float, dict]] = []  # (raw_score, source_w, time_w, article)
    for article in deduped:
        raw_score = score_article(article.get("title", ""), article.get("content", ""))
        source_w, _ = get_source_weight(article.get("link", ""))
        time_w = _time_decay_weight(article.get("date", ""))
        scored.append((raw_score, source_w, time_w, article))

    # --- Weighted average ---
    total_weight = sum(sw * tw for _, sw, tw, _ in scored)
    if total_weight > 0:
        weighted_score = sum(rs * sw * tw for rs, sw, tw, _ in scored) / total_weight
    else:
        weighted_score = 0.0
    weighted_score = round(weighted_score, 4)

    # --- Label ---
    if weighted_score > 0.15:
        label = "BULLISH"
    elif weighted_score < -0.10:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    # --- Top headline: highest source_weight * time_weight article ---
    best = max(scored, key=lambda x: x[1] * x[2])
    top_headline = best[3].get("title", "")

    # --- Volume spike ---
    volume_spike = detect_volume_spike(article_count, volume_history or [])

    return {
        "score": weighted_score,
        "label": label,
        "article_count": article_count,
        "clean_count": clean_count,
        "top_headline": top_headline,
        "volume_spike": volume_spike,
        "pr_dominated": pr_dominated,
        "scorer": scorer_name,
    }
