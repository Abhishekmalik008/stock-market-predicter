# 📈 Stock Market Prediction Software

![CI/CD Status](https://github.com/Abhishekmalik008/stock-market-predicter/actions/workflows/ci-cd.yml/badge.svg?branch=main)

A comprehensive stock market prediction application built with Python, featuring multiple machine learning models and an interactive web interface.

## 🚀 Features

- **Real-time Stock Data**: Fetch live stock data using Yahoo Finance API
- **Multiple ML Models**: LSTM, Random Forest, XGBoost, LightGBM, and more
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and 50+ others
- **Interactive Dashboard**: Built with Streamlit for easy visualization
- **Price Predictions**: Predict stock prices with confidence intervals
- **Performance Metrics**: Comprehensive model evaluation
- **Trading Signals**: Generate BUY/SELL/HOLD recommendations
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Cloud Ready**: Easy deployment to cloud platforms

## 🛠️ System Requirements

- **Python**: 3.9, 3.10, or 3.11
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 500MB free space
- **Internet Connection**: Required for data fetching

## 🚀 Installation

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-market-predictor.git
cd stock-market-predictor

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with core dependencies (minimum required)
pip install -e .

# For machine learning features
pip install -e ".[ml]"

# For web deployment
pip install -e ".[web]"

# Install all optional dependencies
pip install -e ".[all]"
```

### Option 2: Using requirements.txt

```bash
# For basic functionality
pip install -r requirements.txt

# For additional ML features
pip install -r requirements-ml.txt  # If you have this file
```

## 🔧 Compatibility Notes

- **Python Version**: Tested with Python 3.9, 3.10, and 3.11
- **Operating Systems**:
  - Windows 10/11 (64-bit)
  - macOS 10.15+
  - Most Linux distributions (Ubuntu 20.04+, CentOS 8+)
- **Known Issues**:
  - Some visualization features may require additional system libraries on Linux
  - TensorFlow/Keras models might have compatibility issues on ARM-based Macs (M1/M2)

## 🚀 Quick Start

1. Install the package as shown above
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
3. Open your browser to `http://localhost:8501`

## 📦 Dependencies

### Core Dependencies
- Python 3.9+
- Streamlit (web interface)
- Pandas & NumPy (data manipulation)
- yfinance (stock data)
- scikit-learn (machine learning)
- Plotly (visualizations)

### Optional Dependencies
- XGBoost, LightGBM (advanced ML models)
- TensorFlow/PyTorch (deep learning)
- Qiskit (quantum computing features)
- TA-Lib (technical analysis)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Code Coverage with Codecov

This project uses Codecov to track test coverage. To enable code coverage reporting:

1. Sign in to [Codecov](https://about.codecov.io/) with your GitHub account
2. Add your repository to Codecov
3. Get the repository upload token from Codecov
4. Add the token to your GitHub repository secrets:
   - Go to your repository Settings > Secrets > Actions
   - Click "New repository secret"
   - Name: `CODECOV_TOKEN`
   - Value: [paste your Codecov token here]

Code coverage reports will now be automatically uploaded when tests pass on the main branch.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For support or questions, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).
```

## Local Usage

1. Run the application:
```bash
streamlit run main.py
```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

## Cloud Deployment (Streamlit Cloud)

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and select your repository
4. Set the following configuration:
   - **Repository**: Your GitHub repository
   - **Branch**: main (or your preferred branch)
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.9
5. Click "Deploy!"

## Features in the Cloud Version

The cloud version includes a streamlined interface with:
- Real-time stock data visualization
- Interactive candlestick charts
- Basic stock information

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
