"""
Computes call/put ratio signals from options chain data.
"""
import logging
import numpy as np
import pandas as pd
from config import CP_FADE_ZSCORE, CP_MOMENTUM_ZSCORE

logger = logging.getLogger(__name__)


def compute_cp_ratio(options: dict) -> float | None:
    """
    Returns call/put open interest ratio for the given options chain snapshot.
    Returns None if data is insufficient.
    """
    if not options:
        return None
    calls: pd.DataFrame = options.get("calls", pd.DataFrame())
    puts: pd.DataFrame = options.get("puts", pd.DataFrame())
    if calls.empty or puts.empty:
        return None

    call_oi = calls["openInterest"].fillna(0).sum()
    put_oi = puts["openInterest"].fillna(0).sum()
    if put_oi == 0:
        return None
    return float(call_oi / put_oi)


def cp_zscore(cp_history: list) -> float | None:
    """
    Given a list of recent C/P ratios (oldest first), returns the z-score
    of the most recent value vs. the prior window.
    """
    if len(cp_history) < 5:
        return None
    arr = np.array(cp_history, dtype=float)
    mean = arr[:-1].mean()
    std = arr[:-1].std()
    if std == 0:
        # All prior values identical — return directional extreme rather than 0
        return 5.0 if arr[-1] > mean else (-5.0 if arr[-1] < mean else 0.0)
    return float((arr[-1] - mean) / std)


def classify_cp(cp_ratio: float | None, zscore: float | None) -> str:
    """
    Returns one of: "OVERBULLISH", "BEARISH_PRESSURE", "NEUTRAL"
    """
    if cp_ratio is None:
        return "NEUTRAL"

    if zscore is not None:
        if zscore >= CP_FADE_ZSCORE:
            return "OVERBULLISH"
        if zscore <= CP_MOMENTUM_ZSCORE:
            return "BEARISH_PRESSURE"
        return "NEUTRAL"

    # Fallback absolute thresholds when no history exists yet
    if cp_ratio > 2.0:
        return "OVERBULLISH"
    if cp_ratio < 0.5:
        return "BEARISH_PRESSURE"
    return "NEUTRAL"


def compute_put_call_skew(options: dict) -> float | None:
    """
    Median IV of ATM puts minus median IV of ATM calls.
    Positive = fear premium on puts (protective hedging).
    Negative = call buying dominates (speculative bullishness).
    """
    if not options:
        return None
    calls: pd.DataFrame = options.get("calls", pd.DataFrame())
    puts: pd.DataFrame = options.get("puts", pd.DataFrame())
    if calls.empty or puts.empty:
        return None
    if "impliedVolatility" not in calls.columns:
        return None

    call_iv = calls["impliedVolatility"].dropna()
    put_iv = puts["impliedVolatility"].dropna()
    if call_iv.empty or put_iv.empty:
        return None

    n_c, n_p = len(call_iv), len(put_iv)
    call_atm = float(call_iv.iloc[n_c // 4: 3 * n_c // 4].mean())
    put_atm = float(put_iv.iloc[n_p // 4: 3 * n_p // 4].mean())
    return round(put_atm - call_atm, 4)
