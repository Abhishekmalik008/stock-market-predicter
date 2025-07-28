"""
Advanced Stock Price Predictor with Improved Accuracy
Uses ensemble methods, feature engineering, and advanced algorithms
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedStockPredictor:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers
        self.models = {}
        self.feature_importance = {}
        
    def fetch_enhanced_data(self, symbol, period="2y"):
        """Fetch comprehensive stock data with multiple sources"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get historical data
            data = stock.history(period=period)
            
            # Get additional info
            info = stock.info
            
            # Add market cap and sector information if available
            if 'marketCap' in info:
                data['Market_Cap'] = info['marketCap']
            if 'sector' in info:
                data['Sector'] = hash(info['sector']) % 100  # Convert to numeric
                
            return data, info
        except Exception as e:
            raise Exception(f"Error fetching enhanced data for {symbol}: {str(e)}")
    
    def create_advanced_features(self, data):
        """Create comprehensive feature set for better predictions"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        try:
            # Price-based features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            
            # Multi-timeframe moving averages (adjust periods based on data length)
            max_period = min(200, len(df) // 2)
            periods = [p for p in [5, 10, 20, 50, 100, 200] if p <= max_period]
            
            for period in periods:
                if len(df) >= period:
                    df[f'MA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
                    if len(df) >= period + 5:
                        df[f'MA_{period}_slope'] = df[f'MA_{period}'].diff(5)
                    else:
                        df[f'MA_{period}_slope'] = 0
        except Exception as e:
            print(f"Warning: Error creating moving averages: {e}")
        
        try:
            # Multiple RSI periods (adjust based on data length)
            rsi_periods = [p for p in [14, 21, 30] if p <= len(df) // 2]
            for period in rsi_periods:
                if len(df) >= period:
                    rsi_indicator = ta.momentum.RSIIndicator(df['Close'], window=period)
                    df[f'RSI_{period}'] = rsi_indicator.rsi()
        except Exception as e:
            print(f"Warning: Error creating RSI: {e}")
            # Fallback: create simple RSI
            if len(df) >= 14:
                df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        try:
            # MACD variations
            if len(df) >= 26:  # MACD needs at least 26 periods
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_signal'] = macd.macd_signal()
                df['MACD_histogram'] = macd.macd_diff()
            else:
                df['MACD'] = 0
                df['MACD_signal'] = 0
                df['MACD_histogram'] = 0
        except Exception as e:
            print(f"Warning: Error creating MACD: {e}")
            df['MACD'] = 0
            df['MACD_signal'] = 0
            df['MACD_histogram'] = 0
        
        try:
            # Bollinger Bands - use consistent naming with data_processor.py
            if len(df) >= 20:
                bb = ta.volatility.BollingerBands(df['Close'], window=20)
                df['BB_upper'] = bb.bollinger_hband()
                df['BB_lower'] = bb.bollinger_lband()
                df['BB_middle'] = bb.bollinger_mavg()
                
                # Safe calculation of BB width and position
                middle_vals = df['BB_middle']
                upper_vals = df['BB_upper']
                lower_vals = df['BB_lower']
                
                df['BB_width'] = np.where(
                    middle_vals != 0,
                    (upper_vals - lower_vals) / middle_vals,
                    0
                )
                
                df['BB_position'] = np.where(
                    (upper_vals - lower_vals) != 0,
                    (df['Close'] - lower_vals) / (upper_vals - lower_vals),
                    0.5
                )
            else:
                # Fallback for insufficient data
                df['BB_upper'] = df['Close'] * 1.02
                df['BB_lower'] = df['Close'] * 0.98
                df['BB_middle'] = df['Close']
                df['BB_width'] = 0.04
                df['BB_position'] = 0.5
        except Exception as e:
            print(f"Warning: Error creating Bollinger Bands: {e}")
            # Fallback values
            df['BB_upper'] = df['Close'] * 1.02
            df['BB_lower'] = df['Close'] * 0.98
            df['BB_middle'] = df['Close']
            df['BB_width'] = 0.04
            df['BB_position'] = 0.5
        
        try:
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
        except Exception as e:
            print(f"Warning: Error creating Stochastic Oscillator: {e}")
        
        try:
            # Williams %R
            df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        except Exception as e:
            print(f"Warning: Error creating Williams %R: {e}")
        
        # Williams %R
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        # Commodity Channel Index
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        
        # Average True Range and volatility
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_30'] = df['Returns'].rolling(window=30).std()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['Volume_Price_Trend'] = ta.volume.VolumePriceTrendIndicator(df['Close'], df['Volume']).volume_price_trend()
        
        # Price patterns and momentum
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Seasonal and cyclical features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Month'] = df.index.day
        df['Week_of_Year'] = df.index.isocalendar().week
        
        # Market structure features
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Distance_to_Support'] = (df['Close'] - df['Support']) / df['Close']
        df['Distance_to_Resistance'] = (df['Resistance'] - df['Close']) / df['Close']
        
        # Lagged features (important for time series)
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        
        return df
    
    def select_best_features(self, data, target_col, top_k=20):
        """Select best features using correlation analysis with fallbacks for limited data"""
        # Only use numeric columns
        numeric_df = data.select_dtypes(include=[np.number])
        
        if target_col not in numeric_df.columns:
            # Return all available numeric columns if target not found
            available_cols = [col for col in numeric_df.columns if col != target_col]
            return available_cols[:min(len(available_cols), top_k)]
        
        try:
            # Calculate correlation with target
            correlations = abs(numeric_df.corr()[target_col]).sort_values(ascending=False)
            
            # Remove target itself and NaN values
            correlations = correlations.drop(target_col, errors='ignore').dropna()
            
            # Adjust top_k based on available features
            available_features = len(correlations)
            actual_k = min(top_k, available_features, 15)  # Cap at 15 for limited data
            
            if actual_k == 0:
                # Fallback: use basic price and volume features
                basic_features = ['Open', 'High', 'Low', 'Volume']
                feature_cols = [col for col in basic_features if col in numeric_df.columns]
                if not feature_cols:
                    # Last resort: use any available numeric columns
                    feature_cols = [col for col in numeric_df.columns if col != target_col][:5]
            else:
                feature_cols = correlations.head(actual_k).index.tolist()
            
            return feature_cols
            
        except Exception as e:
            print(f"Warning: Feature selection failed ({str(e)}). Using fallback features.")
            # Fallback to basic features
            basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [col for col in basic_features if col in numeric_df.columns and col != target_col]
            return feature_cols[:min(len(feature_cols), 10)]
    
    def create_ensemble_model(self):
        """Create ensemble of different algorithms for better accuracy"""
        # Base models with optimized parameters
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            random_state=42
        )
        
        ridge = Ridge(alpha=1.0)
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        # Create voting ensemble
        ensemble = VotingRegressor([
            ('rf', rf),
            ('gb', gb),
            ('ridge', ridge),
            ('elastic', elastic)
        ])
        
        return ensemble
    
    def train_advanced_model(self, data, target_periods=[1, 5, 10]):
        """Train advanced model with multiple prediction horizons"""
        # Create advanced features
        enhanced_data = self.create_advanced_features(data)
        
        models = {}
        feature_importance = {}
        
        for periods in target_periods:
            # Create target variable
            enhanced_data[f'Target_{periods}'] = enhanced_data['Close'].shift(-periods)
            
            # Remove NaN values more carefully
            clean_data = enhanced_data.dropna()
            
            # Reduce minimum data requirement and provide better feedback
            if len(clean_data) < 30:  # Reduced from 100 to 30
                print(f"Warning: Only {len(clean_data)} data points available for {periods}-day prediction. Skipping.")
                continue
            elif len(clean_data) < 60:
                print(f"Note: Using {len(clean_data)} data points for {periods}-day prediction (limited data).")
            
            # Select best features and ensure no target variables are included
            feature_cols = self.select_best_features(clean_data, f'Target_{periods}')
            
            # Explicitly exclude all target variables from features
            target_cols = [col for col in clean_data.columns if col.startswith('Target_')]
            feature_cols = [col for col in feature_cols if col not in target_cols and col != 'Close']
            
            # Ensure we have some features to work with
            if len(feature_cols) == 0:
                print(f"Warning: No valid features found for {periods}-day prediction. Using basic features.")
                basic_features = ['Open', 'High', 'Low', 'Volume', 'Returns']
                feature_cols = [col for col in basic_features if col in clean_data.columns]
            
            X = clean_data[feature_cols]
            y = clean_data[f'Target_{periods}']
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create and train ensemble model
            model = self.create_ensemble_model()
            
            # Train model
            model.fit(X_scaled, y)
            
            # Store model and feature info
            models[f'model_{periods}'] = {
                'model': model,
                'features': feature_cols,
                'scaler': self.scaler
            }
            
            # Calculate feature importance (from Random Forest component)
            if hasattr(model.named_estimators_['rf'], 'feature_importances_'):
                importance = dict(zip(feature_cols, model.named_estimators_['rf'].feature_importances_))
                feature_importance[f'importance_{periods}'] = importance
        
        self.models = models
        self.feature_importance = feature_importance
        
        return models
    
    def predict_advanced(self, data, symbol, periods_ahead=[1, 5, 10]):
        """Make advanced predictions with confidence intervals"""
        try:
            # Create features for the latest data
            enhanced_data = self.create_advanced_features(data)
            
            predictions = {}
            
            for periods in periods_ahead:
                model_key = f'model_{periods}'
                
                if model_key not in self.models:
                    continue
                
                model_info = self.models[model_key]
                model = model_info['model']
                feature_cols = model_info['features']
                
                # Get latest features with robust error handling
                try:
                    # Check if all required features are available
                    available_features = [col for col in feature_cols if col in enhanced_data.columns]
                    missing_features = [col for col in feature_cols if col not in enhanced_data.columns]
                    
                    if missing_features:
                        print(f"Warning: Missing features for {periods}-day prediction: {missing_features}")
                        # Skip this prediction if too many features are missing
                        if len(missing_features) > len(feature_cols) * 0.5:
                            print(f"Skipping {periods}-day prediction due to too many missing features.")
                            continue
                    
                    # Use only available features
                    if available_features:
                        latest_features = enhanced_data[available_features].iloc[-1:].fillna(method='ffill').fillna(0)
                        
                        # For missing features, create dummy columns with zeros
                        for missing_col in missing_features:
                            latest_features[missing_col] = 0
                        
                        # Reorder columns to match training order
                        latest_features = latest_features[feature_cols]
                    else:
                        print(f"No valid features available for {periods}-day prediction. Skipping.")
                        continue
                    
                    # Scale features
                    latest_scaled = model_info['scaler'].transform(latest_features)
                    
                except Exception as e:
                    print(f"Error preparing features for {periods}-day prediction: {e}")
                    continue
                
                # Make prediction
                pred = model.predict(latest_scaled)[0]
                
                # Calculate confidence based on recent volatility
                recent_volatility = enhanced_data['Volatility_10'].iloc[-1]
                confidence_interval = pred * recent_volatility * np.sqrt(periods)
                
                predictions[f'{periods}_day'] = {
                    'prediction': pred,
                    'lower_bound': pred - confidence_interval,
                    'upper_bound': pred + confidence_interval,
                    'confidence': max(0.6, min(0.95, 1 - recent_volatility * 2))
                }
            
            return predictions
            
        except Exception as e:
            raise Exception(f"Error making advanced predictions: {str(e)}")
    
    def validate_model_accuracy(self, data, test_size=0.2):
        """Validate model accuracy on historical data with consistent feature handling"""
        results = {}
        
        for periods in [1, 5, 10]:
            model_key = f'model_{periods}'
            
            if model_key not in self.models:
                continue
            
            try:
                # Get model info
                model_info = self.models[model_key]
                model = model_info['model']
                feature_cols = model_info['features']
                scaler = model_info['scaler']
                
                # Create features using the same process as training
                enhanced_data = self.create_advanced_features(data)
                enhanced_data[f'Target_{periods}'] = enhanced_data['Close'].shift(-periods)
                clean_data = enhanced_data.dropna()
                
                if len(clean_data) < 10:
                    continue
                
                # Split data
                split_idx = int(len(clean_data) * (1 - test_size))
                test_data = clean_data.iloc[split_idx:]
                
                # Ensure we have the exact features that were used in training
                available_features = [col for col in feature_cols if col in test_data.columns]
                if len(available_features) < len(feature_cols) * 0.8:  # Need at least 80% of features
                    print(f"Warning: Missing too many features for {periods}-day validation. Skipping.")
                    continue
                
                # Use only available features and fill missing ones
                X_test = test_data[available_features].fillna(method='ffill').fillna(0)
                y_test = test_data[f'Target_{periods}']
                
                if len(X_test) == 0 or len(y_test) == 0:
                    continue
                
                # For missing features, create dummy columns with zeros
                missing_features = [col for col in feature_cols if col not in available_features]
                for missing_col in missing_features:
                    X_test[missing_col] = 0
                
                # Reorder columns to match training order
                X_test = X_test[feature_cols]
                
                # Scale and predict
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                
            except Exception as e:
                print(f"Error in validation for {periods}-day model: {str(e)}")
                continue
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Calculate percentage accuracy
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            accuracy = max(0, 100 - mape)
            
            results[f'{periods}_day'] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'accuracy_percent': accuracy,
                'mape': mape
            }
        
        return results
    
    def get_prediction_explanation(self, symbol, periods=1):
        """Explain why the model made specific predictions"""
        model_key = f'model_{periods}'
        
        if model_key not in self.models or f'importance_{periods}' not in self.feature_importance:
            return "Model explanation not available"
        
        importance = self.feature_importance[f'importance_{periods}']
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        explanation = f"Top factors influencing {periods}-day prediction for {symbol}:\n"
        for i, (feature, score) in enumerate(top_features, 1):
            explanation += f"{i}. {feature}: {score:.3f}\n"
        
        return explanation
