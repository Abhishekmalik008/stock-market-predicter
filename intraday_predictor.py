import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
from quantum_predictor import QuantumStockPredictor, HybridQuantumPredictor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class IntradayPredictor:
    def __init__(self, use_quantum=False):
        """
        Initialize the IntradayPredictor.
        
        Args:
            use_quantum (bool): Whether to use quantum-enhanced predictions
        """
        self.models = {}
        self.use_quantum = use_quantum
        self.quantum_predictor = None
        self.classical_classifier = None
        self.hybrid_predictor = None
        self.scaler = StandardScaler()

    def fetch_intraday_data(self, symbol, period="5d", interval="5m"):
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        market_status = "OPEN"
        is_fallback = False
        if data.empty:
            market_status = "CLOSED (or no data)"
            is_fallback = True
            # Fallback to daily data and simulate intraday
            daily_data = stock.history(period="1mo")
            if not daily_data.empty:
                data = self._simulate_intraday_from_daily(daily_data, interval)

        return data, is_fallback, market_status

    def _simulate_intraday_from_daily(self, daily_data, interval):
        # A simple simulation for when markets are closed
        intraday_frames = []
        interval_minutes = int(interval[:-1])
        for idx, row in daily_data.iterrows():
            # Create a full trading day
            start_time = idx.replace(hour=9, minute=30)
            end_time = idx.replace(hour=16, minute=0)
            time_range = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}T')
            
            # Simulate OHLCV
            price_path = np.linspace(row['Open'], row['Close'], len(time_range))
            simulated_df = pd.DataFrame(index=time_range)
            simulated_df['Open'] = price_path
            simulated_df['High'] = price_path * (1 + np.random.uniform(0, 0.005, len(time_range)))
            simulated_df['Low'] = price_path * (1 - np.random.uniform(0, 0.005, len(time_range)))
            simulated_df['Close'] = price_path
            simulated_df['Volume'] = np.random.randint(low=row['Volume']//len(time_range)*0.8, high=row['Volume']//len(time_range)*1.2, size=len(time_range))
            intraday_frames.append(simulated_df)

        return pd.concat(intraday_frames) if intraday_frames else pd.DataFrame()

    def prepare_intraday_features(self, data):
        df = data.copy()
        n_rows = len(df)

        # If we have very little data, we can't generate meaningful features.
        if n_rows < 5:
            return pd.DataFrame() # Return empty frame to be handled by the UI

        # Adapt window sizes based on available data
        win_5 = min(5, n_rows - 1)
        win_10 = min(10, n_rows - 1)
        win_20 = min(20, n_rows - 1)
        win_14 = min(14, n_rows - 1)
        win_slow_macd = min(26, n_rows - 1)
        win_fast_macd = min(12, win_slow_macd - 1) # Ensure fast is smaller than slow

        df['MA_5'] = ta.trend.sma_indicator(df['Close'], window=win_5)
        df['MA_10'] = ta.trend.sma_indicator(df['Close'], window=win_10)
        df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=win_20)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=win_14)
        
        if n_rows > 26: # MACD needs sufficient data
            macd = ta.trend.MACD(df['Close'], window_slow=win_slow_macd, window_fast=win_fast_macd, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_histogram'] = macd.macd_diff()
        else:
            df['MACD'], df['MACD_signal'], df['MACD_histogram'] = 0, 0, 0

        if n_rows > 20: # Bollinger Bands need sufficient data
            bollinger = ta.volatility.BollingerBands(df['Close'], window=win_20)
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_lower'] = bollinger.bollinger_lband()
            df['BB_middle'] = bollinger.bollinger_mavg()
        else:
            df['BB_upper'], df['BB_lower'], df['BB_middle'] = df['Close'], df['Close'], df['Close']

        df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=win_20).mean()
        df['Price_change'] = df['Close'].pct_change()
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Open_Close_ratio'] = df['Open'] / df['Close']
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['Day_of_week'] = df.index.dayofweek
        df['Resistance'] = df['High'].rolling(window=win_14).max()
        df['Support'] = df['Low'].rolling(window=win_14).min()
        df['Volatility'] = df['Close'].rolling(window=win_14).std()
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=win_14)
        
        # Create target columns for different time horizons (percentage change in price)
        # We'll predict the price movement over the next N periods
        df['Target_5min'] = df['Close'].pct_change(1).shift(-1)  # Next period's return
        df['Target_15min'] = df['Close'].pct_change(3).shift(-3)  # 3 periods ahead for 15min
        df['Target_30min'] = df['Close'].pct_change(6).shift(-6)  # 6 periods ahead for 30min
        
        # Handle NaNs from indicator calculations and target creation
        df = df.fillna(method='ffill')  # Forward-fill first
        df = df.fillna(method='bfill')  # Back-fill to handle initial NaNs
        df = df.fillna(0)  # Fill any remaining NaNs with 0 as a fallback
        df = df.dropna() # Drop any rows that might still have NaNs, just in case
        
        # Ensure we have enough future data points for the target
        # Drop the last few rows where we don't have future target values
        df = df.iloc[:-6] if len(df) > 6 else df
        
        return df

    def train_intraday_model(self, data, target_column='Target_5min', model_type='random_forest'):
        """
        Train a model for intraday prediction with optional quantum enhancement.
        
        Args:
            data (pd.DataFrame): Input data with features and target
            target_column (str): Name of the target column
            model_type (str): Type of model to train ('random_forest', 'linear_regression', or 'quantum')
            
        Returns:
            tuple: (trained_model, X_test, y_test)
        """
        if data.empty:
            return None, None, None
            
        # Prepare features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Select only 4 key features for quantum model to match number of qubits
        if model_type == 'quantum':
            # Choose 4 important features that are likely to be available
            selected_features = []
            
            # Try to get these features in order of importance
            for feature in ['Close', 'RSI', 'Volume', 'ATR']:
                if feature in X.columns:
                    selected_features.append(feature)
                    if len(selected_features) >= 4:
                        break
            
            # If we couldn't find enough features, take the first 4 available
            if len(selected_features) < 4:
                additional_features = [col for col in X.columns if col not in selected_features]
                selected_features.extend(additional_features[:4 - len(selected_features)])
            
            X = X[selected_features]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Train-test split (use most recent 20% for testing)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if model_type == 'quantum':
            # Train quantum model with exactly 4 qubits
            self.quantum_predictor = QuantumStockPredictor(
                n_qubits=4,  # Fixed to 4 qubits to match our feature selection
                n_layers=2,
                max_iter=50  # Reduced for faster training
            )
            
            # Train a classical classifier for the hybrid approach
            self.classical_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.classical_classifier.fit(X_train, (y_train > 0).astype(int))
            
            # Train quantum model
            train_score, test_score = self.quantum_predictor.train(np.array(X_train), np.array(y_train))
            
            # Create hybrid predictor
            self.hybrid_predictor = HybridQuantumPredictor(
                self.quantum_predictor,
                self.classical_classifier
            )
            
            # Store the hybrid predictor as the model
            model = self.hybrid_predictor
            
            # Evaluate hybrid model
            y_pred = model.predict(np.array(X_test))
            accuracy = accuracy_score((y_test > 0).astype(int), y_pred)
            print(f"Hybrid model test accuracy: {accuracy:.4f}")
            
        else:
            # Train classical model
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:  # linear_regression
                model = LinearRegression()
                
            model.fit(X_train, y_train)
            
            # Evaluate classical model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"{model_type} model - MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Store the trained model
        self.models[model_type] = model
        
        return model, X_test, y_test

    def predict_intraday(self, model, data, periods_ahead=12):
        """
        Predict future intraday prices using the specified model.
        
        Args:
            model: Trained model (classical or hybrid quantum-classical)
            data (pd.DataFrame): Input data for prediction
            periods_ahead (int): Number of periods to predict ahead
            
        Returns:
            list: Predicted prices
        """
        if data.empty:
            return []
            
        # Get the most recent data point
        last_row = data.iloc[-1:]
        
        # For quantum/hybrid models, use the same features as during training
        if self.use_quantum and hasattr(self, 'quantum_predictor'):
            # Use the same feature selection as in train_intraday_model
            selected_features = []
            for feature in ['Close', 'RSI', 'Volume', 'ATR']:
                if feature in last_row.columns:
                    selected_features.append(feature)
                    if len(selected_features) >= 4:
                        break
            
            # If we couldn't find enough features, take the first 4 available
            if len(selected_features) < 4:
                additional_features = [col for col in last_row.select_dtypes(include=[np.number]).columns 
                                    if col not in selected_features]
                selected_features.extend(additional_features[:4 - len(selected_features)])
            
            current_features = last_row[selected_features].copy()
            feature_columns = selected_features
        # For classical models, use the same features as during training
        elif hasattr(model, 'feature_names_in_'):
            feature_columns = model.feature_names_in_
            current_features = last_row[list(feature_columns)].copy()
        else:
            # Fallback: use all numeric columns
            feature_columns = last_row.select_dtypes(include=[np.number]).columns
            current_features = last_row[feature_columns].copy()
        
        # Scale features if using quantum/hybrid model
        if self.use_quantum and hasattr(self, 'scaler'):
            try:
                # Ensure we have the same features as during training
                current_features = current_features[feature_columns]
                
                # Scale the features using the same scaler as during training
                current_features_scaled = self.scaler.transform(current_features)
                
                # Convert back to DataFrame with correct column names
                current_features = pd.DataFrame(
                    current_features_scaled,
                    columns=feature_columns,
                    index=current_features.index
                )
            except Exception as e:
                print(f"Error scaling features: {e}")
                # If scaling fails, try to continue with unscaled features
                pass
        
        predictions = []
        
        for _ in range(periods_ahead):
            # Make prediction for next time step
            if self.use_quantum and isinstance(model, HybridQuantumPredictor):
                # For hybrid quantum-classical model
                pred_class = model.predict(np.array(current_features))
                # For simplicity, use the class prediction as a multiplier
                # In a real application, you might want to predict actual price changes
                last_price = current_features['Close'].iloc[0] if 'Close' in current_features else current_features.iloc[0, 0]
                pred = last_price * (1.001 if pred_class[0] == 1 else 0.999)
            else:
                # For classical models
                pred = model.predict(current_features[feature_columns])[0]
            
            predictions.append(pred)
            
            # Update features for next prediction
            if len(predictions) < periods_ahead:  # Don't update on last iteration
                # Update the relevant features with the predicted price
                for col in ['Open', 'Close', 'High', 'Low']:
                    if col in current_features.columns:
                        current_features[col] = pred
                
                # Update technical indicators based on the new price
                # This is a simplified example - you might need to adjust based on your actual features
                if 'MA_5' in current_features.columns:
                    current_features['MA_5'] = pred
                if 'RSI' in current_features.columns:
                    current_features['RSI'] = 50  # Neutral RSI
                if 'MACD' in current_features.columns:
                    current_features['MACD'] = 0  # Neutral MACD
        
        return predictions

    def generate_intraday_signals(self, data, predictions):
        signals = []
        current_price = data['Close'].iloc[-1]
        for i, pred_price in enumerate(predictions):
            period = i + 1
            change_percent = ((pred_price - current_price) / current_price) * 100
            signal = "HOLD"
            if change_percent > 0.5:
                signal = "STRONG BUY"
            elif change_percent > 0.2:
                signal = "BUY"
            elif change_percent < -0.5:
                signal = "STRONG SELL"
            elif change_percent < -0.2:
                signal = "SELL"
            signals.append({'period': f'{period*5} min ahead', 'predicted_price': f'{pred_price:.2f}', 'change_%': f'{change_percent:.2f}%', 'signal': signal})
        return signals

    def calculate_intraday_levels(self, data):
        """Calculate key intraday price levels.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and a datetime index.
            
        Returns:
            dict: Dictionary containing key price levels.
        """
        if data.empty:
            return {
                'current_price': 0,
                'day_high': 0,
                'day_low': 0,
                'pivot': 0,
                'resistance_1': 0,
                'support_1': 0,
                'resistance_2': 0,
                'support_2': 0,
                'vwap': 0
            }
            
        try:
            # Get the most recent day's data
            latest_date = data.index.date[-1]
            last_day = data[data.index.date == latest_date]
            
            if last_day.empty:
                # If no data for the latest date, use the entire dataset
                last_day = data
                
            high = last_day['High'].max()
            low = last_day['Low'].min()
            close = last_day['Close'].iloc[-1]
            
            # Ensure we have valid values
            if pd.isna(high) or pd.isna(low) or pd.isna(close):
                raise ValueError("Missing required price data")
                
            pivot = (high + low + close) / 3
            
            # Calculate VWAP if we have volume data
            if 'Volume' in data.columns and not data['Volume'].empty and data['Volume'].sum() > 0:
                vwap = (data['Close'] * data['Volume']).sum() / data['Volume'].sum()
            else:
                vwap = close  # Fallback to close price if no volume data
                
            return {
                'current_price': close,
                'day_high': high,
                'day_low': low,
                'pivot': pivot,
                'resistance_1': (2 * pivot) - low,
                'support_1': (2 * pivot) - high,
                'resistance_2': pivot + (high - low),
                'support_2': pivot - (high - low),
                'vwap': vwap
            }
            
        except Exception as e:
            # Return a default structure with zeros if anything goes wrong
            return {
                'current_price': 0,
                'day_high': 0,
                'day_low': 0,
                'pivot': 0,
                'resistance_1': 0,
                'support_1': 0,
                'resistance_2': 0,
                'support_2': 0,
                'vwap': 0
            }
