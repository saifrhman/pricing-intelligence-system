# Pricing Intelligence System

A production-style Python project for **risk-aware pricing intelligence** and financial decision support.

This repository simulates an analytics workflow used by pricing and portfolio teams. Instead of predicting raw prices, it predicts **next-day returns**, then combines model output with risk, anomaly, sentiment, and explainability signals to produce an interpretable recommendation narrative.

## Why This Project
Pricing and financial analytics teams rarely rely on a single forecast number. They need a full context layer:
- What is the expected directional move?
- How risky is the current regime?
- Are market patterns unusual?
- Is external sentiment aligned or conflicting?
- What features drove the prediction?

This system packages those questions into a modular, testable pipeline.

## Core Capabilities
- Time-series market data ingestion using `yfinance`
- Return-focused feature engineering (lags, momentum, volatility, RSI, MACD)
- Time-aware forecasting with:
  - Naive previous-return baseline
  - Linear regression baseline
  - XGBoost main model
- Evaluation metrics: RMSE, MAE, R2, directional accuracy
- Risk scoring from volatility and drawdown
- Isolation Forest anomaly detection
- Optional sentiment analysis (FinBERT via transformers, plus fallback)
- SHAP explainability (global and latest local drivers)
- Lightweight multi-agent reasoning layer
- Streamlit dashboard for end-to-end exploration

## Repository Structure
```text
pricing-intelligence-system/
├── app/
│   └── streamlit_app.py
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── plots/
│   ├── models/
│   ├── shap/
│   └── reports/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   ├── sentiment.py
│   ├── explainability.py
│   ├── agents.py
│   ├── decision_engine.py
│   ├── pipeline.py
│   ├── utils.py
│   └── schemas.py
├── tests/
│   ├── test_features.py
│   ├── test_forecasting.py
│   └── test_decision_engine.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Architecture
1. Data Ingestion Agent
- Downloads OHLCV data and validates schema.

2. Feature Engineering Agent
- Generates return-based and technical/risk features.

3. Quant Agent
- Trains baseline and ML models with time-aware splits.

4. Risk Agent
- Produces normalized risk score and risk level.

5. Anomaly Agent
- Flags abnormal behavior using Isolation Forest.

6. Sentiment Agent (Optional)
- Aggregates headline sentiment signal.

7. Explanation Agent
- Uses SHAP to explain model behavior.

8. Decision Agent
- Combines all signals into a business-style recommendation.

## Why Predict Returns Instead of Price?
Predicting raw prices can be scale-sensitive and less stationary. Predicting returns:
- aligns better with risk and performance analysis,
- makes cross-asset comparisons cleaner,
- supports directional decision-making,
- reduces model drift caused by long-term price level changes.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
### 1) Run CLI Pipeline
```bash
python main.py --ticker AAPL --start-date 2020-01-01 --end-date 2025-01-01
```

Optional flags:
```bash
python main.py --disable-sentiment
python main.py --no-transformer
python main.py --config config/config.yaml
```

### 2) Run Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

## Example Outputs
- `outputs/models/`: trained model artifact (`.joblib`)
- `outputs/plots/`: prediction and anomaly plots
- `outputs/shap/`: SHAP global/local charts
- `outputs/reports/`: JSON and markdown decision reports

A sample report is included at `outputs/reports/sample_report.md`.

## Testing
```bash
pytest -q
```

## Limitations
- Uses daily OHLCV data; no intraday microstructure modeling.
- Sentiment uses mock headlines by default unless external headlines are supplied.
- This is a single-asset workflow; portfolio optimization is out of scope.
- Regime shifts can degrade short-horizon model quality.

## Future Improvements
- Add robust backtesting and walk-forward retraining.
- Include transaction-cost-aware thresholding.
- Add confidence intervals and probabilistic forecasts.
- Add multi-ticker batch analysis and dashboard comparisons.
- Integrate external financial news APIs for live sentiment.

## Disclaimer
This project is for educational and research purposes only. It is **not** investment advice, financial advice, or a recommendation to trade securities.
