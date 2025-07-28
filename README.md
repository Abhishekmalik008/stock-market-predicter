# 📈 Stock Market Prediction Software

A comprehensive stock market prediction application built with Python, featuring multiple machine learning models and an interactive web interface.

## Features

- **Real-time Stock Data**: Fetch live stock data using Yahoo Finance API
- **Multiple ML Models**: LSTM, Random Forest, and Linear Regression models
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Interactive Dashboard**: Built with Streamlit for easy visualization
- **Price Predictions**: Predict stock prices up to 30 days ahead
- **Performance Metrics**: Model evaluation with RMSE, MAE, and R² scores
- **Trading Signals**: Generate BUY/SELL/HOLD recommendations

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run main.py
```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

3. Use the sidebar to navigate between different features:
   - **Stock Analysis**: View historical data, charts, and basic metrics
   - **Price Prediction**: Generate predictions using different ML models
   - **Model Performance**: Compare model accuracy and performance

## Supported Models

### LSTM (Long Short-Term Memory)
- Deep learning model ideal for time series prediction
- Uses 60-day sequences to predict future prices
- Best for capturing long-term patterns and trends

### Random Forest
- Ensemble learning method using multiple decision trees
- Good for handling non-linear relationships
- Provides feature importance insights

### Linear Regression
- Simple baseline model for comparison
- Fast training and prediction
- Good for understanding linear relationships

## Technical Indicators

The software automatically calculates and uses the following technical indicators:

- **Moving Averages**: 5, 10, 20, 50-day periods
- **RSI**: Relative Strength Index for momentum analysis
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Price volatility indicators
- **Volume Indicators**: Volume ratios and trends
- **Price Ratios**: High/Low, Open/Close relationships

## Example Usage

```python
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from predictor import StockPredictor

# Initialize components
processor = DataProcessor()
trainer = ModelTrainer()
predictor = StockPredictor()

# Fetch and process data
data = processor.fetch_stock_data("AAPL", period="2y")
processed_data = processor.prepare_features(data)

# Train model
model = trainer.train_model(processed_data, "lstm")

# Make predictions
predictions = predictor.predict(model, processed_data, days_ahead=7)
```

## Disclaimer

This software is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals and do your own research before making investment choices.

## Requirements

- Python 3.8+
- Internet connection for fetching stock data
- Minimum 4GB RAM recommended for LSTM training

## Contributing

Feel free to contribute by:
- Adding new technical indicators
- Implementing additional ML models
- Improving the user interface
- Adding more comprehensive backtesting features

## License

This project is open source and available under the MIT License.
