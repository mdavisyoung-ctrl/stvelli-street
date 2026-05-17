import pytest
from scanner.sentiment import score_article, aggregate_sentiment


class TestScoreArticle:
    def test_positive_headline(self):
        score = score_article("Stock surges to record high on amazing earnings", "")
        assert score > 0

    def test_negative_headline(self):
        score = score_article("Terrible disaster destroys horrible worthless company", "")
        assert score < 0

    def test_empty_inputs(self):
        score = score_article("", "")
        assert score == pytest.approx(0.0)


class TestAggregateSentiment:
    def test_empty_articles(self):
        result = aggregate_sentiment([])
        assert result["score"] == 0.0
        assert result["label"] == "NEUTRAL"
        assert result["article_count"] == 0

    def test_bullish_articles(self):
        articles = [
            {"title": "Massive rally as profits soar to record levels", "content": ""},
            {"title": "Excellent growth beats expectations again", "content": ""},
        ]
        result = aggregate_sentiment(articles)
        assert result["label"] == "BULLISH"
        assert result["score"] > 0

    def test_bearish_articles(self):
        articles = [
            {"title": "Stock plunges on terrible earnings miss and bad outlook", "content": ""},
            {"title": "Company faces catastrophic losses and crisis", "content": ""},
        ]
        result = aggregate_sentiment(articles)
        assert result["label"] == "BEARISH"

    def test_recency_weight_applied(self):
        # Most recent article should have highest weight
        articles = [
            {"title": "Exceptional extraordinary gains today", "content": ""},
        ] + [{"title": "neutral report", "content": ""} for _ in range(9)]
        result = aggregate_sentiment(articles)
        assert result["article_count"] == 10
        assert result["top_headline"] == "Exceptional extraordinary gains today"
