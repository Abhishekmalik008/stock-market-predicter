import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from enhanced_model_trainer import EnhancedModelTrainer
import matplotlib.pyplot as plt
import ta  # Technical analysis library
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def print_with_border(text, padding=1, width=80, char='='):
    """Print text with a border around it for better visibility"""
    lines = str(text).split('\n')
    max_len = max(len(line) for line in lines) if lines else 0
    border = char * min(max(max_len, len(text) + 4) + padding * 2, width)
    
    print(f"\n{border}")
    for line in lines:
        print(f"{char}{' ' * padding}{line.ljust(max_len)}{' ' * padding}{char}")
    print(f"{border}\n")

# Set style for plots
plt.style.use('default')  # Use default style
import seaborn as sns
sns.set_theme(style='whitegrid')  # Use whitegrid style from seaborn

def add_technical_indicators(df):
    """
    Add comprehensive technical indicators to the dataframe
    Includes: Moving Averages, RSI, Bollinger Bands, MACD, ADX, ATR, OBV, VWAP, and more
    """
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # If we have MultiIndex columns (from yfinance), handle them
        if isinstance(df.columns, pd.MultiIndex):
            # Create a new DataFrame with single-level columns
            new_data = {}
            for col in df.columns:
                if col[0] in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    new_data[col[0]] = df[col].values
            df = pd.DataFrame(new_data, index=df.index)
        
        # Ensure all columns are numeric and handle missing values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Price Action Features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log1p(df['Returns'])
        df['Range'] = (df['High'] - df['Low']) / df['Close'].shift(1)
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # 2. Volatility Features
        for window in [5, 10, 20, 30]:
            # Historical volatility (annualized)
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
            
            # Average True Range (ATR) based volatility
            tr = pd.DataFrame()
            tr['h-l'] = df['High'] - df['Low']
            tr['h-pc'] = abs(df['High'] - df['Close'].shift(1))
            tr['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
            tr['TR'] = tr.max(axis=1)
            df[f'ATR_{window}'] = tr['TR'].rolling(window=window).mean() / df['Close'] * 100
        
        # 3. Moving Averages & Trends
        windows = [5, 20, 50, 100, 200]
        for window in windows:
            # Simple Moving Averages
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            # Exponential Moving Averages
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
            
            # Price relative to MA
            df[f'Close_MA{window}_Ratio'] = df['Close'] / df[f'MA_{window}'] - 1
            
            # MA crossovers
            if window > 5:
                df[f'MA_Cross_{window}_5'] = (df[f'MA_5'] > df[f'MA_{window}']).astype(int)
        
        # MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 4. Momentum Indicators
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Rate of Change (ROC)
        for window in [5, 10, 20]:
            df[f'ROC_{window}'] = df['Close'].pct_change(window)
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # 5. Volume Analysis
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volume Moving Averages
        for window in [5, 20, 50]:
            df[f'Volume_MA{window}'] = df['Volume'].rolling(window=window).mean()
            df[f'Volume_MA{window}_Ratio'] = df['Volume'] / df[f'Volume_MA{window}']
        
        # VWAP (Volume Weighted Average Price)
        cum_vol = df['Volume'].cumsum()
        cum_price_vol = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum()
        df['VWAP'] = cum_price_vol / cum_vol
        
        # 6. Clean up
        # Replace infinite values with NaN and then drop them
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with any NaN values
        initial_rows = len(df)
        df = df.dropna()
        rows_dropped = initial_rows - len(df)
        
        # Ensure all numeric columns are properly typed
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        
        print(f"Successfully added {len(df.columns) - len(['Open', 'High', 'Low', 'Close', 'Volume'])} technical indicators")
        print(f"Dropped {rows_dropped} rows with NaN or infinite values")
        
        return df
        
    except Exception as e:
        print(f"Error in add_technical_indicators: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def fetch_stock_data(symbol, days=365):
    """Fetch stock data and calculate technical indicators"""
    try:
        print(f"Downloading data for {symbol}...")
        # Download stock data
        df = yf.download(symbol, period=f"{days}d", progress=False)
        
        if df.empty:
            print("Error: No data found for the given symbol.")
            return pd.DataFrame()
            
        print(f"Successfully downloaded {len(df)} days of data")
        print("\nInitial data sample:")
        print(df.head())
        
        # Check data types
        print("\nData types:")
        print(df.dtypes)
        
        # Add technical indicators
        print("\nAdding technical indicators...")
        df = add_technical_indicators(df)
        
        if df is None or df.empty:
            print("Error: Failed to add technical indicators")
            return pd.DataFrame()
            
        # Add target variable (next day's close price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop any remaining NaN values
        initial_rows = len(df)
        df = df.dropna()
        
        print(f"\nDropped {initial_rows - len(df)} rows with missing values")
        print(f"Final dataset has {len(df)} rows and {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        print(f"\nError in fetch_stock_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def plot_predictions(dates, y_true, y_pred, symbol):
    """Plot actual vs predicted values with confidence intervals"""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Calculate errors and metrics
    errors = y_true - y_pred
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    r2 = r2_score(y_true, y_pred)
    
    # Plot 1: Actual vs Predicted
    ax1.plot(dates, y_true, label='Actual', color='#2ecc71', linewidth=2)
    ax1.plot(dates, y_pred, '--', label='Predicted', color='#e74c3c', linewidth=2)
    
    # Add confidence interval
    ci = 1.96 * np.std(errors)
    ax1.fill_between(dates, 
                   y_pred - ci, 
                   y_pred + ci, 
                   color='#3498db', alpha=0.1, label='95% Confidence')
    
    # Add indicators
    ax1.axhline(y_true.mean(), color='#7f8c8d', linestyle='--', alpha=0.7, 
               label=f'Mean: {y_true.mean():.2f}')
    
    # Formatting
    ax1.set_title(f'{symbol} - Stock Price Prediction\nRMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}', 
                 fontsize=14, pad=20)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_true - y_pred
    ax2.bar(dates, residuals, width=1, alpha=0.6, color='#9b59b6')
    ax2.axhline(0, color='#7f8c8d', linestyle='--', alpha=0.7)
    ax2.set_title('Prediction Residuals', fontsize=12, pad=10)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save the plot
    filename = f'{symbol}_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    plt.close()
    
    return filename

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance with multiple metrics"""
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'R2': r2_score(y_train, y_train_pred)
        },
        'test': {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred)
        }
    }
    
    # Calculate directional accuracy
    train_direction = np.sign(y_train_pred[1:] - y_train_pred[:-1]) == np.sign(y_train.values[1:] - y_train.values[:-1])
    test_direction = np.sign(y_test_pred[1:] - y_test_pred[:-1]) == np.sign(y_test.values[1:] - y_test.values[:-1])
    
    metrics['train']['Direction_Accuracy'] = np.mean(train_direction) * 100
    metrics['test']['Direction_Accuracy'] = np.mean(test_direction) * 100
    
    return metrics, y_test_pred

def predict_stock(symbol='TCS.NS', days=365, test_size=0.2):
    """Predict stock prices using the enhanced model"""
    print(f"\n{'='*50}")
    print(f"PREDICTING STOCK: {symbol}")
    print(f"{'='*50}\n")
        
    # Fetch and prepare data
    print("1. Fetching and preparing data...")
    data = fetch_stock_data(symbol, days)
    
    if data.empty:
        print("Failed to fetch data. Please check the symbol and try again.")
        return None
    
    # Initialize the trainer
    trainer = EnhancedModelTrainer()
    
    # Prepare features
    print("\n2. Engineering features...")
    X, y, feature_names = trainer.prepare_features(data)
    
    # Split data using time series split
    print("\n3. Splitting data into train/test sets...")
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Get dates for plotting
    dates = data.index[split_idx:split_idx + len(X_test)]
    
    print(f"   [TRAIN] Training samples: {len(X_train)}")
    print(f"   [TEST]  Testing samples: {len(X_test)}\n")
    
    # Train XGBoost model
    print("\n4. Training XGBoost model...")
    from xgboost import XGBRegressor
    
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )
    
    # Evaluate model
    print("\n5. Evaluating model performance...")
    metrics, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Convert predictions back to price space for evaluation
    last_prices = data['Close'].iloc[-len(y_test)-1:-1].values  # Get prices corresponding to test set
    y_true_prices = last_prices * (1 + y_test)  # Convert returns back to prices
    y_pred_prices = last_prices * (1 + y_pred)  # Convert predicted returns to prices
    
    # Calculate direction accuracy in price space
    direction_correct = ((y_test > 0) & (y_pred > 0)) | ((y_test <= 0) & (y_pred <= 0))
    direction_accuracy = np.mean(direction_correct) * 100
    
    # Print enhanced evaluation metrics
    print_with_border("Model Performance (Return Space)")
    print(f"  [RMSE] {metrics['test']['RMSE']:.4f} ({(metrics['test']['RMSE'] * 100):.2f}%)")
    print(f"  [MAE]  {metrics['test']['MAE']:.4f} ({(metrics['test']['MAE'] * 100):.2f}%)")
    print(f"  [R²]   {metrics['test']['R2']:.4f}")
    print(f"  [DIR]  {metrics['test']['Direction_Accuracy']:.1f}% Direction Accuracy")
    
    # Print price-based metrics
    price_rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
    price_mae = mean_absolute_error(y_true_prices, y_pred_prices)
    
    print_with_border("Model Performance (Price Space)")
    print(f"  [RMSE] {price_rmse:.2f} ({(price_rmse / data['Close'].mean() * 100):.2f}% of avg price)")
    print(f"  [MAE]  {price_mae:.2f} ({(price_mae / data['Close'].mean() * 100):.2f}% of avg price)")
    
    # Make next day prediction
    last_data = X[-1].reshape(1, -1)
    predicted_return = model.predict(last_data)[0]
    last_price = data['Close'].iloc[-1]
    predicted_price = last_price * (1 + predicted_return)
    price_change_pct = predicted_return * 100
    
    print_with_border("Next Day Prediction")
    print(f"  [LAST] Last available price: {last_price:.2f}")
    print(f"  [PRED] Next day prediction: {predicted_price:.2f}")
    print(f"  [CHNG] Predicted change: {price_change_pct:+.2f}%")
    
    # Generate visualizations
    print("\n6. Generating visualizations...")
    plot_dates = data.index[-len(y_test):]
    plot_predictions(plot_dates, y_test, y_pred, symbol)
    
    # Save predictions to CSV
    results = pd.DataFrame({
        'Date': plot_dates,
        'Actual_Return': y_test,
        'Predicted_Return': y_pred,
        'Actual_Price': y_true_prices,
        'Predicted_Price': y_pred_prices
    })
    results.to_csv(f'{symbol}_predictions.csv', index=False)
    
    # Feature importance
    print("\n7. Analyzing feature importance...")
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print("-" * 30)
        print(feature_importance.head(10).to_string(index=False))
    else:
        print("Feature importance not available for this model.")
    
    # Save feature importance
    if 'feature_importance' in locals():
        feature_importance.to_csv(f'{symbol}_feature_importance.csv', index=False)
        print(f"\nFeature importance saved to {symbol}_feature_importance.csv")
    
    # Print metrics
    print("\nModel Performance:")
    print("-" * 30)
    print_with_border("Training Set", char='*')
    print(f"  [RMSE] {metrics['train']['RMSE']:.2f}")
    print(f"  [MAE]  {metrics['train']['MAE']:.2f}")
    print(f"  [R2]   {metrics['train']['R2']:.4f}")
    print(f"  [DIR]  {metrics['train']['Direction_Accuracy']:.1f}% Direction Accuracy")
    
    print_with_border("Test Set", char='*')
    print(f"  [RMSE] {metrics['test']['RMSE']:.2f}")
    print(f"  [MAE]  {metrics['test']['MAE']:.2f}")
    print(f"  [R2]   {metrics['test']['R2']:.4f}")
    print(f"  [DIR]  {metrics['test']['Direction_Accuracy']:.1f}% Direction Accuracy")
    print(f"  [RMSE] {metrics['test']['RMSE']:.2f}")
    print(f"  [MAE]  {metrics['test']['MAE']:.2f}")
    print(f"  [R2]   {metrics['test']['R2']:.4f}")
    print(f"  [DIR]  {metrics['test']['Direction_Accuracy']:.1f}% Direction Accuracy")
    
    # Get prediction details
    last_price = float(data['Close'].iloc[-1])
    last_prediction = float(y_pred[-1])
    price_change = ((last_prediction - last_price) / last_price) * 100
    
    print("\nPrediction Summary:")
    print("-" * 30)
    print(f"  [LAST] Last available price: {last_price:.2f}")
    print(f"  [PRED] Next day prediction: {last_prediction:.2f}")
    print(f"  [CHNG] Predicted change: {price_change:+.2f}%")
    
    # Plot results
    print("\n6. Generating visualizations...")
    plot_filename = plot_predictions(dates, y_test, y_pred, symbol)
    
    # Feature importance
    print("\n7. Analyzing feature importance...")
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print("-"*30)
    # Convert DataFrame to string and replace any problematic characters
    importance_str = feature_importance.to_string(index=False)
    # Replace any remaining Unicode characters
    importance_str = importance_str.replace('\u2705', '[OK]').replace('\u274c', '[X]')
    print(importance_str)
    
    # Also save the feature importance to a CSV file
    feature_importance.to_csv(f"{symbol}_feature_importance.csv", index=False)
    print(f"\nFeature importance saved to {symbol}_feature_importance.csv")
    
    # Save results
    result = {
        'symbol': symbol,
        'model': model,
        'data': data,
        'X_test': X_test,
        'y_test': y_test,
        'predictions': y_pred,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'plot_file': plot_filename
    }
    
    print("\n[SUCCESS] Prediction completed successfully!")
    return result

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stock Price Prediction using Machine Learning')
    parser.add_argument('--symbol', type=str, default='TCS.NS',
                       help='Stock symbol (e.g., TCS.NS, RELIANCE.NS)')
    parser.add_argument('--days', type=int, default=365,
                       help='Number of days of historical data to fetch')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for testing (0.1-0.3)')
    
    args = parser.parse_args()
    
    # Run prediction
    try:
        results = predict_stock(
            symbol=args.symbol,
            days=args.days,
            test_size=args.test_size
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your input parameters and try again.")
