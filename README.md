# Intraday Indian Stock Market Predictor

Predicts buy/sell signals for Indian stocks (NSE/BSE) using Moving Average Crossover or ML models.

## Features
- Fetches real-time intraday data using broker API (e.g., Zerodha Kite)
- Calculates technical indicators (e.g., SMA, EMA, RSI)
- Generates buy/sell signals
- (Optional) Backtesting and performance metrics

## Setup
- Python 3.8+
- Install requirements: `pip install -r requirements.txt`
- Configure your API keys in `config.py`

## Usage
- Run: `python main.py`