"""
ML pattern matching engine.

Approach: k-Nearest Neighbors on a feature vector built from the last N bars.
We find historical windows that most resemble the current market environment
(price action + sentiment + regime), then look at what happened next in those
analogous periods to generate a forward projection.

Why KNN instead of a neural net:
  - Interpretable: you can literally see the matched historical windows
  - No training data pipeline needed beyond historical price CSVs
  - Fast enough for 5-minute scans
  - Works on small datasets (≥100 bars)

Feature vector per window (20 bars):
  [rsi_norm, atr_pct, price_change_5d, price_change_10d, price_change_20d,
   sentiment_score, cp_ratio_norm (or 0 for crypto)]

Labels (what happened next):
  +1 if close[window+5] > close[window] * 1.01  (up >1% in 5 bars)
  -1 if close[window+5] < close[window] * 0.99  (down >1%)
   0 otherwise
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

WINDOW = 20      # bars of history to encode per sample
FORWARD = 5      # bars ahead to label
N_NEIGHBORS = 7


def _build_features(hist: pd.DataFrame, sentiment_series: list, cp_series: list) -> np.ndarray:
    """
    Builds feature matrix X where each row is a WINDOW-bar snapshot.
    Returns array of shape (n_samples, n_features).
    """
    closes = hist["Close"].values
    n = len(closes)
    rows = []
    for i in range(WINDOW, n):
        window_closes = closes[i - WINDOW: i]
        rsi_norm = _rolling_rsi(window_closes) / 100.0
        pct_5 = (closes[i - 1] - closes[max(0, i - 5)]) / closes[max(0, i - 5)]
        pct_10 = (closes[i - 1] - closes[max(0, i - 10)]) / closes[max(0, i - 10)]
        pct_20 = (closes[i - 1] - closes[max(0, i - 20)]) / closes[max(0, i - 20)]
        atr_pct = _rolling_atr(hist.iloc[max(0, i - 20): i]) / closes[i - 1] if closes[i - 1] > 0 else 0
        sent = sentiment_series[i] if i < len(sentiment_series) else 0.0
        cp = cp_series[i] if i < len(cp_series) else 1.0
        rows.append([rsi_norm, atr_pct, pct_5, pct_10, pct_20, sent, cp])
    return np.array(rows, dtype=float)


def _build_labels(hist: pd.DataFrame) -> np.ndarray:
    closes = hist["Close"].values
    n = len(closes)
    labels = []
    for i in range(WINDOW, n):
        if i + FORWARD >= n:
            labels.append(0)
        else:
            future = closes[i + FORWARD]
            now = closes[i - 1]
            if future > now * 1.01:
                labels.append(1)
            elif future < now * 0.99:
                labels.append(-1)
            else:
                labels.append(0)
    return np.array(labels)


def _rolling_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = gain[-period:].mean()
    avg_loss = loss[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def _rolling_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < 2:
        return 0.0
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    return float(tr[-period:].mean()) if len(tr) >= period else float(tr.mean())


def train_model(ticker: str, hist: pd.DataFrame,
                sentiment_series: list = None,
                cp_series: list = None) -> Pipeline | None:
    """
    Trains a KNN model on historical data and saves it to disk.
    Returns the trained pipeline or None if insufficient data.
    """
    if len(hist) < WINDOW + FORWARD + 20:
        logger.warning("Not enough data to train model for %s", ticker)
        return None

    sent = sentiment_series or [0.0] * len(hist)
    cp = cp_series or [1.0] * len(hist)

    X = _build_features(hist, sent, cp)
    y = _build_labels(hist)

    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=min(N_NEIGHBORS, len(X) - 1), weights="distance")),
    ])
    pipe.fit(X, y)

    model_path = MODEL_DIR / f"{ticker}_knn.joblib"
    joblib.dump(pipe, model_path)
    logger.info("Trained and saved model for %s (%d samples)", ticker, len(X))
    return pipe


def load_model(ticker: str) -> Pipeline | None:
    model_path = MODEL_DIR / f"{ticker}_knn.joblib"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error("Model load failed for %s: %s", ticker, e)
    return None


def retrain_on_outcomes(ticker: str, hist: pd.DataFrame, labeled_outcomes: list[dict]) -> bool:
    """
    Retrain the KNN model for *ticker* by appending real-world labeled examples
    (from outcome_tracker) to the historical training set.

    Returns True if the model was retrained, False if insufficient data.
    Requires at least 20 labeled outcomes for this ticker.
    """
    ticker_outcomes = [o for o in labeled_outcomes if o.get("ticker") == ticker]
    if len(ticker_outcomes) < 20:
        logger.debug("Not enough labeled outcomes for %s (%d < 20), skipping retrain",
                     ticker, len(ticker_outcomes))
        return False

    # Build feature matrix from historical data (same as train_model)
    if len(hist) < WINDOW + FORWARD + 20:
        logger.warning("Not enough history to retrain model for %s", ticker)
        return False

    X_hist = _build_features(hist, [0.0] * len(hist), [1.0] * len(hist))
    y_hist = _build_labels(hist)
    min_len = min(len(X_hist), len(y_hist))
    X_hist, y_hist = X_hist[:min_len], y_hist[:min_len]

    # Build feature vectors from labeled real-world outcomes
    outcome_rows = []
    outcome_labels = []
    for outcome in ticker_outcomes:
        features = outcome.get("features", {})
        if not features:
            continue
        cp_ratio = features.get("cp_ratio") or 1.0
        cp_zscore_val = features.get("cp_zscore") or 0.0
        sentiment_score = features.get("sentiment_score") or 0.0
        rsi = features.get("rsi") or 50.0
        atr_pct = features.get("atr_pct") or 0.0
        # Map to the 7-feature vector: [rsi_norm, atr_pct, pct_5, pct_10, pct_20, sent, cp]
        # We only have aggregated features, so use zeros for price-change fields
        row = [rsi / 100.0, atr_pct, 0.0, 0.0, 0.0, sentiment_score, cp_ratio]
        outcome_rows.append(row)
        # Map outcome (1=correct, -1=incorrect, 0=flat) to label
        raw_outcome = outcome.get("outcome", 0)
        sig_type = outcome.get("signal_type", "")
        if raw_outcome == 1:
            label = 1 if sig_type == "LONG" else -1
        elif raw_outcome == -1:
            label = -1 if sig_type == "LONG" else 1
        else:
            label = 0
        outcome_labels.append(label)

    if not outcome_rows:
        return False

    X_outcomes = np.array(outcome_rows, dtype=float)
    y_outcomes = np.array(outcome_labels)

    X_combined = np.vstack([X_hist, X_outcomes])
    y_combined = np.concatenate([y_hist, y_outcomes])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=min(N_NEIGHBORS, len(X_combined) - 1),
            weights="distance",
        )),
    ])
    pipe.fit(X_combined, y_combined)

    model_path = MODEL_DIR / f"{ticker}_knn.joblib"
    joblib.dump(pipe, model_path)
    logger.info(
        "Retrained model for %s with %d historical + %d outcome samples",
        ticker, len(X_hist), len(outcome_rows),
    )
    return True


def predict(ticker: str, hist: pd.DataFrame,
            current_sentiment: float = 0.0,
            current_cp: float = 1.0) -> dict:
    """
    Returns a prediction dict:
      {
        "prediction": 1 / -1 / 0,
        "label": "UP" / "DOWN" / "FLAT",
        "confidence": float,
        "matched_windows": int,
        "model_trained": bool,
      }
    """
    pipe = load_model(ticker)
    if pipe is None:
        # Auto-train if we have enough data
        pipe = train_model(ticker, hist)
        if pipe is None:
            return {"prediction": 0, "label": "FLAT", "confidence": 0.0,
                    "matched_windows": 0, "model_trained": False}

    # Build current feature vector (just the most recent window)
    n = len(hist)
    if n < WINDOW:
        return {"prediction": 0, "label": "FLAT", "confidence": 0.0,
                "matched_windows": 0, "model_trained": True}

    closes = hist["Close"].values
    window_closes = closes[n - WINDOW: n]
    rsi_norm = _rolling_rsi(window_closes) / 100.0
    pct_5 = (closes[-1] - closes[max(0, n - 5)]) / closes[max(0, n - 5)]
    pct_10 = (closes[-1] - closes[max(0, n - 10)]) / closes[max(0, n - 10)]
    pct_20 = (closes[-1] - closes[max(0, n - 20)]) / closes[max(0, n - 20)]
    atr_pct = _rolling_atr(hist.tail(20)) / closes[-1] if closes[-1] > 0 else 0

    x = np.array([[rsi_norm, atr_pct, pct_5, pct_10, pct_20, current_sentiment, current_cp]])

    try:
        pred = int(pipe.predict(x)[0])
        proba = pipe.predict_proba(x)[0]
        confidence = round(float(proba.max()), 3)
        label_map = {1: "UP", -1: "DOWN", 0: "FLAT"}
        return {
            "prediction": pred,
            "label": label_map.get(pred, "FLAT"),
            "confidence": confidence,
            "matched_windows": N_NEIGHBORS,
            "model_trained": True,
        }
    except Exception as e:
        logger.error("Prediction failed for %s: %s", ticker, e)
        return {"prediction": 0, "label": "FLAT", "confidence": 0.0,
                "matched_windows": 0, "model_trained": True}
