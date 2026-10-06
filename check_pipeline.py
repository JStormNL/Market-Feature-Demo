"""Lightweight structural checks for the public demonstration pipeline."""

from __future__ import annotations

import numpy as np

from data_loader import load_market_data
from example_feature_engineering import add_public_features


def main() -> None:
    print("PUBLIC PIPELINE CHECK")

    market = load_market_data()
    required_market = {
        "datetime", "open", "high", "low", "close", "volume",
        "session_date", "session_time", "bar_duration_minutes",
    }
    missing_market = required_market - set(market.columns)
    if missing_market:
        raise RuntimeError(f"Missing market columns: {sorted(missing_market)}")
    print(f"[PASS] Loaded {len(market)} normalized market observations.")

    closing = market.loc[market["session_time"].eq("15:30")]
    if not closing.empty:
        if not np.all(closing["bar_duration_minutes"].to_numpy() == 30):
            raise RuntimeError("15:30 closing observations are not marked as 30 minutes.")
        print("[PASS] Closing half-hour duration metadata is preserved.")

    featured = add_public_features(market)
    required_features = {
        "simple_return", "log_return", "intrabar_return",
        "high_low_range_pct", "volume_change", "rsi_14",
    }
    missing_features = required_features - set(featured.columns)
    if missing_features:
        raise RuntimeError(f"Missing public features: {sorted(missing_features)}")
    print("[PASS] Representative public features were generated.")

    rsi = featured["rsi_14"].dropna()
    if not rsi.empty and not rsi.between(0.0, 100.0, inclusive="both").all():
        raise RuntimeError("RSI values escaped the expected [0, 100] range.")
    print("[PASS] RSI remains within [0, 100].")

    if market["datetime"].duplicated().any():
        raise RuntimeError("Duplicate timestamps found.")
    print("[PASS] Timestamps are unique.")

    print("PIPELINE STATUS: PASS")


if __name__ == "__main__":
    main()
