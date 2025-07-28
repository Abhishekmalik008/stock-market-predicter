import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def fetch_stock_data(self, symbol, period="2y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def add_technical_indicators(self, data):
        """Add technical indicators to the dataset"""
        df = data.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price change indicators
        df['Price_change'] = df['Close'].pct_change()
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Open_Close_ratio'] = df['Open'] / df['Close']
        
        return df
    
    def prepare_features(self, data):
        """Prepare features for machine learning"""
        df = self.add_technical_indicators(data)
        
        # Select features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Volume_ratio', 'Price_change', 'High_Low_ratio', 'Open_Close_ratio'
        ]
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Create target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        
        return df
    
    def create_sequences(self, data, sequence_length=60):
        """Create sequences for LSTM model"""
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
            targets.append(data[i])
        
        return np.array(sequences), np.array(targets)
    
    def scale_data(self, data, fit_scaler=True):
        """Scale data using MinMaxScaler"""
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)
        
        return scaled_data
    
    def inverse_scale(self, data):
        """Inverse scale the data"""
        return self.scaler.inverse_transform(data)
