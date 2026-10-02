"""Representative public feature engineering for Delta Patterns.

This module intentionally demonstrates only conventional OHLCV transformations
and a standard Wilder RSI feature. The private research implementation contains
additional engineered market-state inputs that are not distributed publicly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


REQUIRED_OHLCV = {"open", "high", "low", "close", "volume"}


def _validate_ohlcv(df: pd.DataFrame) -> None:
    missing = REQUIRED_OHLCV - set(df.columns)
    if missing:
        raise KeyError(f"Missing required OHLCV columns: {sorted(missing)}")


def wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Wilder-style Relative Strength Index on a closing-price series."""
    close = pd.to_numeric(close, errors="coerce").astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / float(window),
        adjust=False,
        min_periods=window,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / float(window),
        adjust=False,
        min_periods=window,
    ).mean()

    rs = avg_gain / (avg_loss + float(config.EPSILON))
    rsi = 100.0 - (100.0 / (1.0 + rs))

    both_flat = (avg_gain <= config.EPSILON) & (avg_loss <= config.EPSILON)
    only_gains = (avg_gain > config.EPSILON) & (avg_loss <= config.EPSILON)
    only_losses = (avg_loss > config.EPSILON) & (avg_gain <= config.EPSILON)

    rsi = rsi.mask(both_flat, 50.0)
    rsi = rsi.mask(only_gains, 100.0)
    rsi = rsi.mask(only_losses, 0.0)
    return rsi.clip(lower=0.0, upper=100.0)


def add_public_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small, intentionally non-proprietary feature set."""
    _validate_ohlcv(df)
    out = df.copy()

    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(out["open"], errors="coerce").astype(float)
    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(out["volume"], errors="coerce").astype(float)

    out["simple_return"] = close.pct_change()
    out["log_return"] = np.log((close + config.EPSILON) / (close.shift(1) + config.EPSILON))
    out["intrabar_return"] = (close - open_) / (open_.abs() + config.EPSILON)
    out["high_low_range_pct"] = (high - low) / (close.abs() + config.EPSILON)
    out["volume_change"] = volume.pct_change()
    out["rsi_14"] = wilder_rsi(close, window=int(config.RSI_WINDOW))

    return out.replace([np.inf, -np.inf], np.nan)
