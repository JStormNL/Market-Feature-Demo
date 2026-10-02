"""Build the public Delta Patterns feature demonstration."""

from __future__ import annotations

import pandas as pd

import config
from data_loader import load_market_data
from example_feature_engineering import add_public_features


PUBLIC_FEATURE_COLUMNS = [
    "datetime",
    "session_date",
    "session_time",
    "market_bar_index",
    "bar_duration_minutes",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "simple_return",
    "log_return",
    "intrabar_return",
    "high_low_range_pct",
    "volume_change",
    "rsi_14",
]


def build_public_feature_frame() -> pd.DataFrame:
    market = load_market_data()
    featured = add_public_features(market)

    missing = [column for column in PUBLIC_FEATURE_COLUMNS if column not in featured.columns]
    if missing:
        raise RuntimeError(f"Public feature build is missing columns: {missing}")

    return featured[PUBLIC_FEATURE_COLUMNS].copy()


def main() -> None:
    frame = build_public_feature_frame()
    frame.to_csv(config.PUBLIC_OUTPUT_PATH, index=False)

    print("DELTA PATTERNS — PUBLIC FEATURE DEMONSTRATION")
    print(f"Rows: {len(frame)}")
    print(f"Saved: {config.PUBLIC_OUTPUT_PATH}")
    print()
    print("Latest rows:")
    print(frame.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
