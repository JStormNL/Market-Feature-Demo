"""Public configuration for the Delta Patterns portfolio demonstration.

This file intentionally contains only the parameters needed to reproduce the
public data-ingestion and representative feature-engineering workflow.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_OUTPUT_PATH = RESULTS_DIR / "public_feature_demo.csv"

TICKER = "SPY"
PERIOD = "2y"
INTERVAL = "1h"

MARKET_TIMEZONE = "America/New_York"
REGULAR_SESSION_START = "09:30"
REGULAR_SESSION_END = "16:00"

KEEP_FINAL_PARTIAL_BAR = True
EXPECTED_BARS_PER_NORMAL_SESSION = 7
EXPECTED_FINAL_BAR_MINUTES = 30

RSI_WINDOW = 14
EPSILON = 1e-12
