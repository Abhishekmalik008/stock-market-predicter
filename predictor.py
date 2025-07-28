import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from future_analytics import FutureAnalytics

class StockPredictor:
    def __init__(self):
        self.future_analytics = FutureAnalytics()
    
    def predict_lstm(self, model, data, days_ahead=7, sequence_length=60):
        """Make predictions using LSTM model"""
        from data_processor import DataProcessor
        processor = DataProcessor()
        
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Volume_ratio', 'Price_change', 'High_Low_ratio', 'Open_Close_ratio'
        ]
        
        # Get the last sequence_length days of data
        features = data[feature_columns].values
        scaled_features = processor.scale_data(features, fit_scaler=False)
        
        predictions = []
        current_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, len(feature_columns))
        
        for _ in range(days_ahead):
            # Predict next value
            pred = model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            # For simplicity, we'll use the last row and update only the Close price
            next_row = current_sequence[0, -1, :].copy()
            next_row[3] = pred[0, 0]  # Update Close price (index 3)
            
            # Shift sequence and add new prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = next_row
        
        # Inverse scale predictions (assuming Close price scaling)
        predictions = np.array(predictions).reshape(-1, 1)
        # Create dummy array for inverse scaling
        dummy_array = np.zeros((len(predictions), len(feature_columns)))
        dummy_array[:, 3] = predictions.flatten()  # Close price is at index 3
        
        inverse_scaled = processor.inverse_scale(dummy_array)
        return inverse_scaled[:, 3]  # Return only Close prices
    
    def predict_sklearn(self, model, data, days_ahead=7):
        """Make predictions using sklearn models"""
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Volume_ratio', 'Price_change', 'High_Low_ratio', 'Open_Close_ratio'
        ]
        
        predictions = []
        current_data = data[feature_columns].iloc[-1:].copy()
        
        for _ in range(days_ahead):
            # Make prediction
            pred = model.predict(current_data)[0]
            predictions.append(pred)
            
            # Update features for next prediction
            # For simplicity, we'll update some key features
            current_data.iloc[0, current_data.columns.get_loc('Open')] = pred
            current_data.iloc[0, current_data.columns.get_loc('High')] = pred * 1.02
            current_data.iloc[0, current_data.columns.get_loc('Low')] = pred * 0.98
        
        return np.array(predictions)
    
    def predict(self, model, data, days_ahead=7, model_type='lstm'):
        """Make predictions based on model type"""
        if model_type == 'lstm':
            return self.predict_lstm(model, data, days_ahead)
        else:
            return self.predict_sklearn(model, data, days_ahead)
    
    def calculate_confidence_intervals(self, predictions, confidence_level=0.95):
        """Calculate confidence intervals for predictions"""
        # Simple approach using standard deviation
        std = np.std(predictions)
        margin = 1.96 * std  # 95% confidence interval
        
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        
        return lower_bound, upper_bound
    
    def generate_trading_signals(self, current_price, predicted_prices, threshold=0.05):
        """Generate trading signals based on predictions"""
        signals = []
        
        for pred_price in predicted_prices:
            change_percent = (pred_price - current_price) / current_price
            
            if change_percent > threshold:
                signals.append("BUY")
            elif change_percent < -threshold:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        return signals
