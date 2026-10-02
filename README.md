# Market Feature Demo

A compact Python project that demonstrates financial time-series data collection, cleaning, feature engineering, and validation using intraday SPY market data.

## Project Overview

Market Feature Demo pulls OHLCV data from Yahoo Finance, normalizes the intraday session, engineers several conventional market features, and exports a clean tabular dataset that can be used in later machine-learning experiments.

This repository is a **public portfolio demonstration**. It includes representative feature-engineering work while intentionally omitting additional research-specific inputs and transformations from the larger private project.

## Objective

The goal is to demonstrate a reproducible workflow for converting raw financial time-series data into structured, model-ready features.

```
SPY OHLCV
    ↓
Data Cleaning & Session Handling
    ↓
Return / Range / Volume Features
    ↓
 RSI
    ↓
Pipeline Validation
    ↓
Feature Dataset
```

## Data

- **Instrument:** SPDR S&P 500 ETF Trust (`SPY`)
- **Source:** Yahoo Finance via `yfinance`
- **Interval:** 1-hour intraday data
- **Session:** U.S. regular trading hours
- **Timezone:** `America/New_York`

The loader also records bar duration so the final 15:30–16:00 ET half-hour observation is not treated as a full 60-minute bar.

## Feature Engineering

The public pipeline includes:

- simple close-to-close returns;
- log returns;
- intrabar open-to-close returns;
- high-low price range;
- volume change; and
- 14-period Relative Strength Index (RSI).

RSI is included as a representative example of transforming raw price history into a stateful quantitative feature. The private research project contains additional engineered inputs that are intentionally not published here.

## Validation

`check_pipeline.py` verifies key data and feature properties, including:

- required OHLCV columns;
- unique timestamps;
- regular-session handling;
- closing-bar duration metadata;
- successful feature construction; and
- RSI values within the expected `[0, 100]` range.

This repository does **not** train a predictive model or claim trading profitability. Its purpose is to demonstrate the data and feature-engineering layer that can support later machine-learning work.

## Technologies

- Python
- pandas
- NumPy
- yfinance
- Financial time-series processing
- Feature engineering

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── data_loader.py
├── example_feature_engineering.py
├── build_demo_features.py
├── check_pipeline.py
└── results/
    └── README.md
```

## How to Run

Python 3.10 or 3.11 is recommended.

pip install -r requirements.txt
python check_pipeline.py
python build_demo_features.py
```

The generated feature dataset is written to:

```text
results/public_feature_demo.csv
```

## Public Repository Scope

This repository intentionally exposes only a representative subset of a larger private financial time-series research project. Selected custom feature definitions and transformation logic are withheld from the public version.

## Disclaimer

This project is for research, educational, and portfolio purposes only. 
