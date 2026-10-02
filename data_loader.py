"""Session-aware SPY market-data loader used by the public demonstration.

The loader keeps regular U.S. trading hours, normalizes timestamps to
America/New_York, and records the actual duration represented by each bar.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import yfinance as yf

import config


def _parse_clock(value: str) -> time:
    hour, minute = value.split(":")
    return time(hour=int(hour), minute=int(minute))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out.columns = [str(column).strip().lower() for column in out.columns]

    if "date" in out.columns and "datetime" not in out.columns:
        out = out.rename(columns={"date": "datetime"})

    return out


def _normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "datetime" not in out.columns:
        raise KeyError("Market data must contain a datetime column.")

    dt = pd.to_datetime(out["datetime"], errors="coerce")
    if dt.isna().any():
        raise ValueError("Market data contains unparseable timestamps.")

    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(config.MARKET_TIMEZONE)
    else:
        dt = dt.dt.tz_convert(config.MARKET_TIMEZONE)

    out["datetime"] = dt
    return out


def _keep_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    start = _parse_clock(config.REGULAR_SESSION_START)
    end = _parse_clock(config.REGULAR_SESSION_END)
    clock = df["datetime"].dt.time
    return df.loc[(clock >= start) & (clock < end)].copy()


def _add_session_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["session_date"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out["session_time"] = out["datetime"].dt.strftime("%H:%M")
    out["session_bar_number"] = out.groupby("session_date", sort=False).cumcount()
    return out


def _add_bar_duration_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Record the duration represented by each regular-session observation.

    Yahoo's 1-hour SPY regular-session series normally includes six 60-minute
    bars followed by a final 15:30-16:00 ET 30-minute bar. Keeping this metadata
    avoids silently treating the closing half-hour as a full hour.
    """
    out = df.copy()

    if config.INTERVAL.lower() not in {"1h", "60m"}:
        out["bar_duration_minutes"] = np.nan
        out["bar_duration_hours"] = np.nan
        out["is_partial_session_bar"] = False
        return out

    out["bar_duration_minutes"] = 60

    session_counts = out.groupby("session_date", sort=False)["datetime"].transform("size")
    final_mask = (
        out.groupby("session_date", sort=False).cumcount()
        == (session_counts - 1)
    )

    normal_close_mask = out["session_time"].eq("15:30")
    early_close_candidate = (
        final_mask
        & (session_counts < int(config.EXPECTED_BARS_PER_NORMAL_SESSION))
        & out["datetime"].dt.minute.eq(30)
    )
    partial_mask = normal_close_mask | early_close_candidate

    if bool(config.KEEP_FINAL_PARTIAL_BAR):
        out.loc[partial_mask, "bar_duration_minutes"] = int(
            config.EXPECTED_FINAL_BAR_MINUTES
        )
    else:
        out = out.loc[~partial_mask].copy()

    out["bar_duration_minutes"] = pd.to_numeric(
        out["bar_duration_minutes"], errors="raise"
    ).astype(int)
    out["bar_duration_hours"] = out["bar_duration_minutes"] / 60.0
    out["is_partial_session_bar"] = out["bar_duration_minutes"] < 60
    return out


def load_market_data() -> pd.DataFrame:
    """Download and normalize intraday SPY OHLCV data."""
    df = yf.download(
        tickers=config.TICKER,
        period=config.PERIOD,
        interval=config.INTERVAL,
        auto_adjust=False,
        prepost=False,
        actions=False,
        progress=False,
    )

    if df.empty:
        raise ValueError("No data returned from yfinance.")

    df = df.reset_index()
    df = _normalize_columns(df)
    df = _normalize_datetime(df)
    df = _keep_regular_session(df)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = _add_session_metadata(df)
    df = _add_bar_duration_metadata(df)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = _add_session_metadata(df)

    df["market_bar_index"] = np.arange(len(df), dtype=np.int64)

    required = {
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "session_date",
        "session_time",
        "bar_duration_minutes",
        "bar_duration_hours",
        "market_bar_index",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("NaN values found in required OHLCV columns.")

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate market timestamps found after normalization.")

    return df
