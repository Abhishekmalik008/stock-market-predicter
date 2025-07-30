from classic_predictor import ClassicStockPredictor
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Download data
    print("Downloading stock data...")
    data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
    
    # Initialize and prepare data
    print("Preparing data...")
    predictor = ClassicStockPredictor()
    X, y = predictor.prepare_data(data)
    
    # Split data (80% train, 20% test)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Train model
    print("Training model...")
    predictor.train(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(X_test)
    
    # Calculate metrics
    mse = ((predictions - y_test) ** 2).mean()
    mae = abs(predictions - y_test).mean()
    
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', color='blue')
    plt.plot(y_test.index, predictions, label='Predicted', color='red', alpha=0.7)
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
