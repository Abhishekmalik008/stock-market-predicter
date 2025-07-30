import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from ta import add_all_ta_features

class ClassicStockPredictor:
    """Classical Machine Learning Stock Predictor"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.fitted = False
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(21).std() * np.sqrt(252)
        
        # Add technical indicators
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['Close'].rolling(window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Add lagged features
        for lag in [1, 2, 3, 5, 7, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        
        # Add volume features
        df['volume_change'] = df['Volume'].pct_change()
        
        return df.dropna()
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close', forecast_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        df = self._add_features(df)
        
        # Create target (future price movement)
        df['target'] = df[target_col].shift(-forecast_horizon)
        df = df.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_col, 'target', 'Date']]
        X = df[feature_cols]
        y = df['target']
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the ensemble of models"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize base models
        models = {
            'xgb': XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, n_jobs=-1),
            'lgbm': LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, n_jobs=-1),
            'catboost': CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=6, verbose=False)
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y)
            self.models[name] = model
        
        # Create and train meta-learner
        estimators = [(name, model) for name, model in models.items()]
        self.meta_learner = StackingRegressor(
            estimators=estimators,
            final_estimator=XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, n_jobs=-1),
            n_jobs=-1
        )
        self.meta_learner.fit(X_scaled, y)
        self.fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble"""
        if not self.fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.meta_learner.predict(X_scaled)

# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Example usage
    print("Loading data...")
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
    
    predictor = ClassicStockPredictor()
    X, y = predictor.prepare_data(data)
    
    # Split data (80% train, 20% test)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print("Training model...")
    predictor.train(X_train, y_train)
    
    print("Making predictions...")
    predictions = predictor.predict(X_test)
    
    # Calculate and print accuracy
    mse = ((predictions - y_test) ** 2).mean()
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Predictions: {predictions[:5]}")
    print(f"Actual: {y_test.values[:5]}")
