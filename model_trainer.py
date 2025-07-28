import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def train_lstm_model(self, data, sequence_length=60):
        """LSTM model not available - TensorFlow not installed"""
        raise NotImplementedError("LSTM model requires TensorFlow. Please use Random Forest or Linear Regression.")
    
    def train_random_forest(self, data):
        """Train Random Forest model"""
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Volume_ratio', 'Price_change', 'High_Low_ratio', 'Open_Close_ratio'
        ]
        
        X = data[feature_columns].dropna()
        y = data['Target'].dropna()
        
        # Align X and y
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Store model
        self.models['random_forest'] = model
        
        return model
    
    def train_linear_regression(self, data):
        """Train Linear Regression model"""
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Volume_ratio', 'Price_change', 'High_Low_ratio', 'Open_Close_ratio'
        ]
        
        X = data[feature_columns].dropna()
        y = data['Target'].dropna()
        
        # Align X and y
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Store model
        self.models['linear_regression'] = model
        
        return model
    
    def train_model(self, data, model_type='lstm'):
        """Train specified model type"""
        if model_type == 'lstm':
            return self.train_lstm_model(data)
        elif model_type == 'random_forest':
            return self.train_random_forest(data)
        elif model_type == 'linear_regression':
            return self.train_linear_regression(data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def evaluate_model(self, model, X_test, y_test, model_type='lstm'):
        """Evaluate model performance"""
        if model_type == 'lstm':
            predictions = model.predict(X_test)
            # Inverse scale predictions and actual values
            processor = self.scalers['lstm']
            predictions = processor.inverse_scale(predictions)
            y_test = processor.inverse_scale(y_test.reshape(-1, 1))
        else:
            predictions = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
