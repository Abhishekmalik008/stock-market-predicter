import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Import our ultimate predictor
from quantum_enhanced_predictor import MetaEnsemblePredictor
from advanced_indicators import AdvancedIndicators

# Set style for plots
plt.style.use('seaborn')
sns.set_palette('viridis')

class UltimateStockPredictor:
    """End-to-end stock prediction system with the ultimate predictor"""
    
    def __init__(self, ticker: str = 'AAPL', start_date: str = '2010-01-01'):
        """
        Initialize the predictor with a stock ticker and start date
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date for historical data (YYYY-MM-DD)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.predictor = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
    
    def fetch_data(self) -> None:
        """Fetch historical stock data"""
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        # Add technical indicators
        self.data = AdvancedIndicators.calculate_all_indicators(self.data)
        print(f"Fetched {len(self.data)} days of data")
    
    def prepare_features(self, forecast_horizon: int = 5) -> None:
        """Prepare features and target variable"""
        if self.data is None:
            raise ValueError("Please fetch data first using fetch_data()")
        
        # Create target (future price movement)
        self.data['target'] = self.data['Close'].shift(-forecast_horizon)
        self.data = self.data.dropna()
        
        # Drop non-feature columns
        feature_cols = [col for col in self.data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
        X = self.data[feature_cols]
        y = self.data['target']
        
        # Split into train and test sets (time-based split)
        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Prepared {len(feature_cols)} features")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
    
    def train_predictor(self, use_quantum: bool = True) -> None:
        """Train the ultimate predictor"""
        print("\nTraining the ultimate predictor...")
        self.predictor = MetaEnsemblePredictor(use_quantum=use_quantum)
        
        # Optimize hyperparameters
        print("Optimizing hyperparameters...")
        best_params = self.predictor.optimize_hyperparameters(
            self.X_train, 
            self.y_train,
            n_trials=50
        )
        
        # Train with best parameters
        print("Training models with optimized parameters...")
        self.predictor.models['xgb1'].set_params(**best_params)
        self.predictor.train(self.X_train, self.y_train)
        
        print("Training complete!")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the predictor on test data"""
        if self.predictor is None:
            raise ValueError("Please train the predictor first using train_predictor()")
        
        # Make predictions
        y_pred = self.predictor.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Calculate directional accuracy
        direction_acc = np.mean((np.sign(y_pred[1:] - y_pred[:-1]) == 
                               np.sign(self.y_test.values[1:] - self.y_test.values[:-1]))) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Direction Accuracy (%)': direction_acc
        }
        
        # Print metrics
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_predictions(self) -> None:
        """Plot actual vs predicted prices"""
        if self.predictor is None:
            raise ValueError("Please train the predictor first using train_predictor()")
        
        # Make predictions
        y_pred = self.predictor.predict(self.X_test)
        
        # Create figure
        plt.figure(figsize=(14, 7))
        
        # Plot actual prices
        plt.plot(self.X_test.index, self.y_test, label='Actual', linewidth=2)
        
        # Plot predicted prices
        plt.plot(self.X_test.index, y_pred, label='Predicted', linestyle='--', linewidth=2)
        
        # Add labels and title
        plt.title(f'{self.ticker} Stock Price Prediction', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'{self.ticker}_prediction.png', dpi=300, bbox_inches='tight')
        print(f"\nPrediction plot saved as '{self.ticker}_prediction.png'")
    
    def predict_future(self, days_ahead: int = 30) -> pd.DataFrame:
        """Predict future stock prices"""
        if self.predictor is None:
            raise ValueError("Please train the predictor first using train_predictor()")
        
        # Get the most recent data point
        last_data = self.X_test.iloc[-1:].copy()
        
        # Create a DataFrame to store predictions
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='B'  # Business days
        )
        
        predictions = []
        current_data = last_data.copy()
        
        # Generate predictions for each future day
        for i in range(days_ahead):
            # Predict next day's price
            pred_price = self.predictor.predict(current_data)[0]
            predictions.append(pred_price)
            
            # Update features for next prediction
            if i < days_ahead - 1:
                # Shift lagged features
                for col in current_data.columns:
                    if 'lag' in col:
                        current_data[col] = current_data[col].shift(1).fillna(pred_price)
                
                # Update other features based on prediction
                current_data['Close'] = pred_price
                
                # Recalculate technical indicators (simplified)
                current_data['daily_returns'] = pred_price / current_data['Close'].shift(1) - 1
                current_data['sma_5'] = current_data['Close'].rolling(5).mean()
                current_data['sma_10'] = current_data['Close'].rolling(10).mean()
        
        # Create DataFrame with predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions
        }).set_index('Date')
        
        return future_df

def main():
    # Initialize with a stock ticker (e.g., AAPL, MSFT, GOOGL)
    ticker = 'AAPL'
    print(f"=== {ticker} Stock Price Prediction ===\n")
    
    # Create predictor instance
    predictor = UltimateStockPredictor(ticker=ticker, start_date='2015-01-01')
    
    # 1. Fetch and prepare data
    predictor.fetch_data()
    predictor.prepare_features(forecast_horizon=5)
    
    # 2. Train the ultimate predictor
    use_quantum = True  # Set to False if you don't have Qiskit installed
    predictor.train_predictor(use_quantum=use_quantum)
    
    # 3. Evaluate performance
    metrics = predictor.evaluate()
    
    # 4. Plot predictions
    predictor.plot_predictions()
    
    # 5. Predict future prices
    future_days = 30
    future_prices = predictor.predict_future(days_ahead=future_days)
    
    print("\nFuture Price Predictions:")
    print(future_prices.tail())
    
    # Plot future predictions
    plt.figure(figsize=(14, 7))
    plt.plot(predictor.data['Close'][-100:], label='Historical Prices', linewidth=2)
    plt.plot(future_prices['Predicted_Close'], label='Predicted Future Prices', 
             linestyle='--', linewidth=2, color='red')
    
    plt.title(f'{ticker} Stock Price Prediction (Next {future_days} Days)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    future_plot_path = f'{ticker}_future_prediction.png'
    plt.savefig(future_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nFuture prediction plot saved as '{future_plot_path}'")

if __name__ == "__main__":
    main()
