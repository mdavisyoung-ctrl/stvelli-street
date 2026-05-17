"""
News sentiment scoring using TextBlob.
Aggregates headline + content polarity into a single score per ticker.
"""
import logging
from textblob import TextBlob

logger = logging.getLogger(__name__)

RECENCY_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]


def score_article(title: str, content: str) -> float:
    """Returns polarity in [-1, 1]. Title weighted 2x over content."""
    title_score = TextBlob(title).sentiment.polarity if title else 0.0
    content_score = TextBlob(content[:500]).sentiment.polarity if content else 0.0
    return (2 * title_score + content_score) / 3


def aggregate_sentiment(articles: list[dict]) -> dict:
    """
    Returns:
        score       – weighted polarity in [-1, 1]
        label       – "BULLISH" / "BEARISH" / "NEUTRAL"
        article_count
        top_headline – most recent headline
    """
    if not articles:
        return {"score": 0.0, "label": "NEUTRAL", "article_count": 0, "top_headline": ""}

    scores = []
    for i, article in enumerate(articles[:10]):  # cap at 10 most recent
        raw = score_article(article.get("title", ""), article.get("content", ""))
        weight = RECENCY_WEIGHTS[i] if i < len(RECENCY_WEIGHTS) else 0.1
        scores.append((raw, weight))

    total_weight = sum(w for _, w in scores)
    weighted_score = sum(s * w for s, w in scores) / total_weight if total_weight else 0.0
    weighted_score = round(weighted_score, 4)

    if weighted_score > 0.15:
        label = "BULLISH"
    elif weighted_score < -0.10:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "score": weighted_score,
        "label": label,
        "article_count": len(articles),
        "top_headline": articles[0].get("title", "") if articles else "",
    }
