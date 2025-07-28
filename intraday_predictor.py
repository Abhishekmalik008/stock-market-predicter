"""
Intraday Stock Prediction Module
Provides minute-by-minute and hourly predictions for day trading
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pytz
import warnings
warnings.filterwarnings('ignore')

class IntradayPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.models = {}
    
    def check_market_status(self, symbol):
        """Check if markets are likely open based on current time and symbol"""
        now = datetime.now()
        
        # Check if it's weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False, "Markets are closed (Weekend)"
        
        # Determine market timezone based on symbol
        if symbol.endswith('.NS') or symbol.endswith('.BO'):
            # Indian market
            ist = pytz.timezone('Asia/Kolkata')
            market_time = now.astimezone(ist)
            market_open = market_time.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = market_time.replace(hour=15, minute=30, second=0, microsecond=0)
            market_name = "Indian (NSE/BSE)"
        else:
            # US market (default)
            est = pytz.timezone('US/Eastern')
            market_time = now.astimezone(est)
            market_open = market_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = market_time.replace(hour=16, minute=0, second=0, microsecond=0)
            market_name = "US"
        
        is_open = market_open <= market_time <= market_close
        status = f"{market_name} market is {'OPEN' if is_open else 'CLOSED'}"
        
        return is_open, status
        
    def fetch_intraday_data(self, symbol, period="5d", interval="5m"):
        """Fetch intraday data with specified interval, with fallback to daily data"""
        try:
            stock = yf.Ticker(symbol)
            # Check market status first
            is_open, market_status = self.check_market_status(symbol)
            
            # Try to fetch intraday data
            data = stock.history(period=period, interval=interval)
            
            # If no intraday data, try fallback approaches
            if data.empty or len(data) < 10:
                # Try longer period first
                if period == "5d":
                    data = stock.history(period="1mo", interval=interval)
                
                # If still insufficient, try daily data as fallback
                if data.empty or len(data) < 10:
                    daily_data = stock.history(period="3mo", interval="1d")
                    if not daily_data.empty and len(daily_data) >= 20:
                        # Convert daily data to simulate intraday
                        data = self.simulate_intraday_from_daily(daily_data, interval)
                        return data, True, market_status  # Return flag indicating fallback used
                    else:
                        raise ValueError(f"No sufficient data available for {symbol}. {market_status}. " +
                                       "This could be due to:\n" +
                                       "• Invalid stock symbol\n" +
                                       "• Newly listed stock with limited history\n" +
                                       "• Delisted or inactive stock")
            
            return data, False, market_status  # Return flag indicating real intraday data
            
        except Exception as e:
            if "No data found" in str(e) or "No timezone found" in str(e):
                raise Exception(f"No data found for {symbol}. {market_status}. Please check:\n" +
                              "• Stock symbol is correct (use .NS for Indian stocks)\n" +
                              "• Stock is actively traded\n" +
                              "• Try a different symbol")
            else:
                raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def simulate_intraday_from_daily(self, daily_data, target_interval="5m"):
        """Simulate intraday data from daily data when real intraday data is unavailable"""
        simulated_data = []
        
        # Determine number of intervals per day based on target interval
        intervals_per_day = {
            "1m": 390,  # 6.5 hours * 60 minutes
            "5m": 78,   # 6.5 hours * 12 intervals
            "15m": 26,  # 6.5 hours * 4 intervals
            "30m": 13,  # 6.5 hours * 2 intervals
            "1h": 7     # 6.5 hours
        }
        
        intervals = intervals_per_day.get(target_interval, 78)
        
        for _, day in daily_data.iterrows():
            # Create intraday points for each day
            day_start = day.name.replace(hour=9, minute=30)  # Market open
            
            for i in range(intervals):
                # Calculate time for this interval
                if target_interval == "1m":
                    time_offset = timedelta(minutes=i)
                elif target_interval == "5m":
                    time_offset = timedelta(minutes=i * 5)
                elif target_interval == "15m":
                    time_offset = timedelta(minutes=i * 15)
                elif target_interval == "30m":
                    time_offset = timedelta(minutes=i * 30)
                else:  # 1h
                    time_offset = timedelta(hours=i)
                
                current_time = day_start + time_offset
                
                # Simulate price movement within the day
                progress = i / intervals  # 0 to 1 throughout the day
                
                # Create realistic intraday movement
                volatility = np.random.normal(0, 0.01)  # 1% random volatility
                trend = (day['Close'] - day['Open']) * progress  # Linear trend toward close
                
                simulated_open = day['Open'] + trend + (day['High'] - day['Low']) * volatility
                simulated_high = simulated_open * (1 + abs(volatility))
                simulated_low = simulated_open * (1 - abs(volatility))
                simulated_close = simulated_open + (day['Close'] - day['Open']) * 0.1  # Small movement
                simulated_volume = day['Volume'] / intervals  # Distribute volume
                
                simulated_data.append({
                    'Open': max(simulated_open, 0.01),
                    'High': max(simulated_high, simulated_open),
                    'Low': min(simulated_low, simulated_open),
                    'Close': max(simulated_close, 0.01),
                    'Volume': max(simulated_volume, 1)
                })
        
        # Create DataFrame with proper datetime index
        result_df = pd.DataFrame(simulated_data)
        
        # Create datetime index
        start_date = daily_data.index[0]
        if target_interval == "5m":
            freq = '5T'
        elif target_interval == "15m":
            freq = '15T'
        elif target_interval == "30m":
            freq = '30T'
        elif target_interval == "1h":
            freq = '1H'
        else:
            freq = '5T'
        
        # Generate business hours index
        date_range = pd.date_range(
            start=start_date,
            periods=len(result_df),
            freq=freq
        )
        
        result_df.index = date_range[:len(result_df)]
        return result_df
    
    def add_intraday_indicators(self, data):
        """Add technical indicators optimized for intraday trading"""
        df = data.copy()
        
        # Short-term moving averages for intraday
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI with shorter period for intraday
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD for intraday
        macd = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        # Bollinger Bands with shorter period
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price momentum indicators
        df['Price_change'] = df['Close'].pct_change()
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Open_Close_ratio'] = df['Open'] / df['Close']
        
        # Intraday specific indicators
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['Day_of_week'] = df.index.dayofweek
        
        # Support and resistance levels
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        # Volatility indicators
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        return df
    
    def prepare_intraday_features(self, data):
        """Prepare features for intraday prediction"""
        df = self.add_intraday_indicators(data)
        
        # Select features optimized for intraday trading
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Volume_ratio', 'Price_change', 'High_Low_ratio', 'Open_Close_ratio',
            'Hour', 'Minute', 'Day_of_week',
            'Resistance', 'Support', 'Volatility', 'ATR'
        ]
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Create multiple targets for different time horizons
        df['Target_5min'] = df['Close'].shift(-1)  # Next 5-minute price
        df['Target_15min'] = df['Close'].shift(-3)  # Next 15-minute price
        df['Target_30min'] = df['Close'].shift(-6)  # Next 30-minute price
        df['Target_1hour'] = df['Close'].shift(-12)  # Next 1-hour price
        
        df = df.dropna()
        
        return df
    
    def train_intraday_model(self, data, target_column='Target_5min', model_type='random_forest'):
        """Train model for intraday prediction"""
        # Check if we have sufficient data
        if len(data) < 50:
            raise ValueError(f"Insufficient data for intraday prediction. Need at least 50 data points, got {len(data)}. This usually happens when markets are closed or for stocks with limited intraday data.")
        
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Volume_ratio', 'Price_change', 'High_Low_ratio', 'Open_Close_ratio',
            'Hour', 'Minute', 'Day_of_week',
            'Resistance', 'Support', 'Volatility', 'ATR'
        ]
        
        # Filter available feature columns (some might be missing)
        available_features = [col for col in feature_columns if col in data.columns]
        if len(available_features) < 10:
            raise ValueError(f"Insufficient features available. Need at least 10 features, got {len(available_features)}.")
        
        # Prepare features and target
        X = data[available_features].dropna()
        y = data[target_column].dropna()
        
        # Align X and y by common indices
        common_indices = X.index.intersection(y.index)
        X = X.loc[common_indices]
        y = y.loc[common_indices]
        
        # Final validation
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid data samples after preprocessing. This typically occurs during market closed hours or for stocks with insufficient intraday trading data.")
        
        if len(X) < 20:
            raise ValueError(f"Insufficient samples for training. Need at least 20 samples, got {len(X)}. Try using a different time interval or check if markets are open.")
        
        # Split data (use more recent data for testing in intraday)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:  # linear_regression
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        
        # Store model
        self.models[f'{model_type}_{target_column}'] = model
        
        return model, X_test, y_test
    
    def predict_intraday(self, model, data, periods=12):
        """Make intraday predictions for next N periods"""
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'Volume_ratio', 'Price_change', 'High_Low_ratio', 'Open_Close_ratio',
            'Hour', 'Minute', 'Day_of_week',
            'Resistance', 'Support', 'Volatility', 'ATR'
        ]
        
        predictions = []
        current_data = data[feature_columns].iloc[-1:].copy()
        
        for i in range(periods):
            # Make prediction
            pred = model.predict(current_data)[0]
            predictions.append(pred)
            
            # Update features for next prediction
            # Simulate time progression
            current_time = data.index[-1] + timedelta(minutes=5 * (i + 1))
            current_data.iloc[0, current_data.columns.get_loc('Hour')] = current_time.hour
            current_data.iloc[0, current_data.columns.get_loc('Minute')] = current_time.minute
            
            # Update price-based features
            current_data.iloc[0, current_data.columns.get_loc('Open')] = pred
            current_data.iloc[0, current_data.columns.get_loc('High')] = pred * 1.01
            current_data.iloc[0, current_data.columns.get_loc('Low')] = pred * 0.99
        
        return np.array(predictions)
    
    def generate_intraday_signals(self, data, predictions):
        """Generate intraday trading signals"""
        current_price = data['Close'].iloc[-1]
        signals = []
        
        for i, pred_price in enumerate(predictions):
            time_horizon = (i + 1) * 5  # minutes
            change_percent = (pred_price - current_price) / current_price * 100
            
            if change_percent > 1.0:  # 1% gain threshold for intraday
                signal = f"STRONG BUY ({time_horizon}min)"
            elif change_percent > 0.5:
                signal = f"BUY ({time_horizon}min)"
            elif change_percent < -1.0:
                signal = f"STRONG SELL ({time_horizon}min)"
            elif change_percent < -0.5:
                signal = f"SELL ({time_horizon}min)"
            else:
                signal = f"HOLD ({time_horizon}min)"
            
            signals.append({
                'time_horizon': f"{time_horizon} min",
                'predicted_price': pred_price,
                'change_percent': change_percent,
                'signal': signal
            })
        
        return signals
    
    def calculate_intraday_levels(self, data):
        """Calculate key intraday levels"""
        current_price = data['Close'].iloc[-1]
        
        # Today's high and low
        today_data = data[data.index.date == data.index[-1].date()]
        day_high = today_data['High'].max()
        day_low = today_data['Low'].min()
        
        # Pivot points (classic method)
        yesterday_data = data[data.index.date == (data.index[-1].date() - timedelta(days=1))]
        if not yesterday_data.empty:
            prev_high = yesterday_data['High'].max()
            prev_low = yesterday_data['Low'].min()
            prev_close = yesterday_data['Close'].iloc[-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = 2 * pivot - prev_low
            r2 = pivot + (prev_high - prev_low)
            s1 = 2 * pivot - prev_high
            s2 = pivot - (prev_high - prev_low)
        else:
            pivot = current_price
            r1 = r2 = s1 = s2 = current_price
        
        # VWAP (Volume Weighted Average Price)
        today_data['VWAP'] = (today_data['Close'] * today_data['Volume']).cumsum() / today_data['Volume'].cumsum()
        vwap = today_data['VWAP'].iloc[-1] if not today_data.empty else current_price
        
        return {
            'current_price': current_price,
            'day_high': day_high,
            'day_low': day_low,
            'pivot': pivot,
            'resistance_1': r1,
            'resistance_2': r2,
            'support_1': s1,
            'support_2': s2,
            'vwap': vwap
        }
