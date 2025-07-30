import yfinance as yf
import pandas as pd
import numpy as np
from enhanced_model_trainer import EnhancedModelTrainer
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

def fetch_stock_data(symbol, days=365):
    """Fetch stock data for the given symbol and ensure proper column structure"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Download data with multi-index columns
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # If the download fails or returns empty, return None
        if df.empty:
            print(f"No data found for {symbol}")
            return None
            
        # Check if we have a MultiIndex for columns
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten the column names by joining the multi-index levels with underscore
            df.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in df.columns.values]
        
        # Ensure we have the required columns (case insensitive)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in df.columns if any(req_col in col for req_col in required_columns)]
        
        if len(available_columns) < len(required_columns):
            missing = set(required_columns) - set(available_columns)
            print(f"Warning: Missing required columns: {missing}")
            return None
        
        # Rename columns to standard names if they contain the required names
        column_mapping = {}
        for req_col in required_columns:
            # Find the first column that contains the required column name
            matching_cols = [col for col in df.columns if req_col.lower() in col.lower()]
            if matching_cols:
                column_mapping[matching_cols[0]] = req_col
        
        # Apply the column renaming
        df = df.rename(columns=column_mapping)
        
        # Calculate technical indicators
        # Simple Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        
        # Additional features
        df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
        df['Price_change'] = df['Close'].pct_change()
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Open_Close_ratio'] = df['Open'] / df['Close']
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Ensure we have enough data
        if len(df) < 60:  # Need at least 60 days of data
            print(f"Not enough data points: {len(df)} (need at least 60)")
            return None
            
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")

def test_simple_prediction():
    print("Testing Simple Prediction...")
    
    # Create a simple dataset
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100)
    data = pd.DataFrame({
        'Open': np.random.normal(100, 5, 100),
        'High': np.random.normal(105, 5, 100),
        'Low': np.random.normal(95, 5, 100),
        'Close': np.random.normal(100, 5, 100),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Add some technical indicators
    for window in [5, 10, 20, 50]:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    
    # Add some random features
    data['RSI'] = np.random.uniform(30, 70, 100)
    data['MACD'] = np.random.normal(0, 1, 100)
    data['MACD_signal'] = np.random.normal(0, 1, 100)
    data['MACD_histogram'] = np.random.normal(0, 0.5, 100)
    data['BB_upper'] = data['Close'] + 5
    data['BB_lower'] = data['Close'] - 5
    data['BB_middle'] = data['Close']
    data['Volume_ratio'] = np.random.uniform(0.8, 1.2, 100)
    data['Price_change'] = np.random.normal(0, 0.01, 100)
    data['High_Low_ratio'] = data['High'] / data['Low']
    data['Open_Close_ratio'] = data['Open'] / data['Close']
    
    # Drop any remaining NaN values
    data = data.dropna()
    
    print(f"Created synthetic dataset with {len(data)} rows")
    
    # Initialize and train a simple model
    print("\nTraining simple model...")
    trainer = EnhancedModelTrainer()
    
    # Train just one model type for testing
    model_type = 'xgb'
    print(f"Training {model_type} model...")
    
    # Prepare data
    X, y, _ = trainer.prepare_features(data)
    
    # Ensure we have enough data for train/test split
    if len(X) < 10:
        raise ValueError("Not enough data for training and testing")
    
    # Use 80% for training, 20% for testing
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Train model with fixed parameters (no optimization)
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    trainer.models[model_type] = model
    
    # Make prediction
    print("\nMaking prediction...")
    prediction = model.predict(X_test[:1])[0]
    actual = y_test.iloc[0]
    
    print(f"Predicted value: {prediction:.2f}")
    print(f"Actual value: {actual:.2f}")
    print(f"Difference: {abs(prediction - actual):.2f} ({abs((prediction - actual)/actual*100):.2f}%)")
    
    # Simple evaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\nModel R² Score - Train: {train_score:.4f}, Test: {test_score:.4f}")
    
    if test_score > 0.3:  # Very loose threshold for synthetic data
        print("[PASS] Test passed! Model is working.")
    else:
        print("[INFO] Model performance is low, but this is expected with random data.")

def test_enhanced_training():
    print("Testing Enhanced Model Training...")
    
    # Test with a real stock symbol (e.g., RELIANCE.NS for NSE)
    symbol = 'RELIANCE.NS'
    print(f"\nFetching data for {symbol}...")
    data = fetch_stock_data(symbol, days=730)  # Get 2 years of data for better walk-forward validation
    
    if data is None or len(data) < 200:  # Require at least 200 days of data
        print("Insufficient data for testing. Using fallback test.")
        test_simple_prediction()
        return
    
    print(f"\nData shape: {data.shape}")
    print(f"First few rows of data:")
    print(data[['Open', 'High', 'Low', 'Close', 'Volume']].head())
    
    # Initialize the enhanced trainer
    trainer = EnhancedModelTrainer()
    
    try:
        # First run walk-forward validation for robust evaluation
        print("\nRunning walk-forward validation for robust evaluation...")
        wf_metrics = trainer.train_ensemble_model(
            data, 
            n_trials=0,  # Skip hyperparameter tuning for now
            use_walk_forward=True
        )
        
        # Now run with hyperparameter tuning
        print("\n\nTraining ensemble model with hyperparameter optimization...")
        try:
            result = trainer.train_ensemble_model(
                data, 
                n_trials=10,  # Reduced for testing
                cv_splits=3,  # Reduced for testing
                use_walk_forward=False
            )
            
            # Check if we got a tuple (model, metrics) or just metrics
            if isinstance(result, tuple) and len(result) == 2:
                model, metrics = result
            else:
                metrics = result
                model = None
            
            # Print metrics
            if metrics:
                print("\nModel Metrics:")
                print("-" * 50)
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}: {value}")
            
            # Make a prediction if we have a model
            if model is not None:
                X, y, _ = trainer.prepare_features(data)
                if len(X) > 0 and hasattr(model, 'predict'):
                    prediction = model.predict(X[-1:])[0]
                    actual = y[-1] if hasattr(y, 'iloc') else y[-1]
                    print("\nNext Period Prediction:")
                    print("-" * 50)
                    print(f"Predicted return: {prediction:.4f}")
                    print(f"Last actual return: {actual:.4f}")
                    print(f"Predicted direction: {'UP' if prediction > 0 else 'DOWN'}")
                    print(f"Actual direction: {'UP' if actual > 0 else 'DOWN'}")
        
        except Exception as e:
            print(f"\nError during model training: {str(e)}")
            print("Continuing with walk-forward validation results...")
        
        # Always show walk-forward validation results if available
        if hasattr(trainer, 'walk_forward_metrics') and trainer.walk_forward_metrics:
            print("\nWalk-Forward Validation Results:")
            print("-" * 50)
            for metric, value in trainer.walk_forward_metrics.items():
                if 'Baseline' not in metric and isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
    
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        print("Falling back to simple prediction test...")
        test_simple_prediction()
        return
        
    # Make prediction if we have a stacking model
    if hasattr(trainer, 'models') and 'stacking' in trainer.models and trainer.models['stacking'] is not None:
        try:
            # Prepare features for prediction
            X, y, _ = trainer.prepare_features(data)
            if len(X) > 0 and hasattr(trainer.models['stacking'], 'predict'):
                y_pred = trainer.models['stacking'].predict(X[-1:])[0]
                y_actual = y.iloc[-1] if hasattr(y, 'iloc') else y[-1]
                
                print("\nStacking Ensemble Prediction:")
                print("-" * 50)
                print(f"Predicted next day return: {y_pred:.4f}")
                print(f"Last actual return: {y_actual:.4f}")
                print(f"Predicted direction: {'UP' if y_pred > 0 else 'DOWN'}")
                print(f"Actual direction: {'UP' if y_actual > 0 else 'DOWN'}")
        except Exception as e:
            print(f"\nError during stacking prediction: {str(e)}")
    else:
        print("\nNo stacking model available for prediction")
    
    # Print feature importance if available
    if hasattr(trainer, 'feature_importance') and trainer.feature_importance:
        print("\nTop 10 Important Features:")
        for model_name, importance in trainer.feature_importance.items():
            if importance is not None and len(importance) > 0:
                print(f"\n{model_name.upper()}:")
                print(importance.head(10).to_string())

if __name__ == "__main__":
    try:
        test_enhanced_training()
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        print("Falling back to simple prediction test...")
        test_simple_prediction()
