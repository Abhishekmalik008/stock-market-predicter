import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_params = {}
        self.feature_importance = {}
        
    def prepare_features(self, data):
        """
        Prepare features and target variable with enhanced feature selection
        Uses mutual information and correlation analysis to select the most predictive features
        """
        # Check if we have multi-index columns
        is_multi_index = isinstance(data.columns, pd.MultiIndex)
        
        # Handle both single and multi-index column formats
        if is_multi_index:
            # For multi-index, we'll work with the first level (price data)
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = []
            
            # Find the first occurrence of each required column in the multi-index
            for col in price_columns:
                for c in data.columns:
                    if c[1] == col or (isinstance(c, str) and col in c):
                        available_columns.append(c)
                        break
            
            if len(available_columns) != len(price_columns):
                missing = set(price_columns) - set(str(c) for c in available_columns)
                raise ValueError(f"Missing required columns in MultiIndex data: {missing}")
                
            # Create a copy to avoid modifying the original data
            data = data.copy()
            
            # Extract the close price column for target calculation
            close_col = [c for c in available_columns if c[1] == 'Close' or (isinstance(c, str) and 'Close' in c)][0]
            
            # Set target variable (next day's return)
            data[('Target', 'Target')] = data[close_col].pct_change().shift(-1)
            data = data.dropna()
            
            # Ensure we have valid data
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Initialize X with basic price and volume data
            X = data[available_columns].copy()
            y = data[('Target', 'Target')].copy()
            
            # Create a mapping for easier column access
            col_map = {}
            for col in available_columns:
                if isinstance(col, tuple):
                    # Get the base column name (second part of the tuple)
                    base_col = col[1] if len(col) > 1 else col[0]
                    col_map[base_col] = col
                else:
                    col_map[col] = col
        else:
            # Original single-index handling
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = [col for col in required_columns if col in data.columns]
            
            if len(available_columns) != len(required_columns):
                missing = set(required_columns) - set(available_columns)
                raise ValueError(f"Missing required columns: {missing}")
                
            # Create a copy to avoid modifying the original data
            data = data.copy()
            
            # Set target variable (next day's return)
            data['Target'] = data['Close'].pct_change().shift(-1)
            data = data.dropna()
            
            # Ensure we have valid data
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Initialize X with basic price and volume data
            X = data[available_columns].copy()
            y = data['Target'].copy()
            
            # For single index, column names are the same as the keys
            col_map = {col: col for col in available_columns}
        
        # Helper function to get column with fallback
        def get_col(name, df=X):
            if name in df.columns:
                return df[name]
            elif name in col_map and col_map[name] in df.columns:
                return df[col_map[name]]
            elif any(name in str(c) for c in df.columns):
                # Try to find a column that contains the name
                for c in df.columns:
                    if name in str(c):
                        return df[c]
            raise KeyError(f"Could not find column: {name} in DataFrame")
            
        # 1. Basic price transformations
        close = get_col('Close')
        X['Returns'] = close.pct_change()
        X['Log_Returns'] = np.log1p(X['Returns'])
        
        # 2. Volatility features
        X['Volatility_5D'] = X['Returns'].rolling(5).std()
        X['Volatility_21D'] = X['Returns'].rolling(21).std()
        X['Volatility_Ratio'] = X['Volatility_5D'] / X['Volatility_21D'].replace(0, np.nan)
        
        # 3. Price momentum features
        X['Momentum_5D'] = close.pct_change(5)
        X['Momentum_21D'] = close.pct_change(21)
        X['Momentum_Ratio'] = X['Momentum_5D'] / X['Momentum_21D'].replace(0, np.nan)
        
        # 4. Volume features
        volume = get_col('Volume')
        X['Volume_MA5'] = volume.rolling(5).mean()
        X['Volume_MA21'] = volume.rolling(21).mean()
        X['Volume_Ratio'] = volume / X['Volume_MA5'].replace(0, np.nan)
        
        # 5. Price range features
        open_prices = get_col('Open')
        high = get_col('High')
        low = get_col('Low')
        
        X['Range'] = (high - low) / close.replace(0, np.nan)
        X['Body'] = (close - open_prices).abs() / close.replace(0, np.nan)
        
        # Calculate max/min of open/close for shadows
        max_oc = pd.concat([open_prices, close], axis=1).max(axis=1)
        min_oc = pd.concat([open_prices, close], axis=1).min(axis=1)
        
        X['Upper_Shadow'] = (high - max_oc) / close.replace(0, np.nan)
        X['Lower_Shadow'] = (min_oc - low) / close.replace(0, np.nan)
        
        # 6. Market regime features
        X['MA50'] = close.rolling(50).mean()
        X['MA200'] = close.rolling(200).mean()
        X['Trend_Strength'] = (close - X['MA50']) / X['MA50'].replace(0, np.nan)
        X['Trend_Regime'] = np.where(close > X['MA200'], 1, -1)
        
        # 7. Mean reversion features
        rolling_mean_21 = close.rolling(21).mean()
        rolling_std_21 = close.rolling(21).std()
        rolling_mean_63 = close.rolling(63).mean()
        rolling_std_63 = close.rolling(63).std()
        
        X['Z_Score_21D'] = (close - rolling_mean_21) / rolling_std_21.replace(0, np.nan)
        X['Z_Score_63D'] = (close - rolling_mean_63) / rolling_std_63.replace(0, np.nan)
        
        # 8. Lagged features
        for lag in [1, 2, 5]:
            X[f'Returns_lag{lag}'] = X['Returns'].shift(lag)
            X[f'Volume_lag{lag}'] = volume.pct_change(lag)
        
        # Handle infinite and missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill().fillna(0)
        
        # Ensure all values are finite
        if not np.all(np.isfinite(X)):
            print("[WARNING] Non-finite values detected in features after processing")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.ffill().bfill().fillna(0)
        
        # 2. Volatility features
        X['Volatility_5D'] = X['Returns'].rolling(5).std()
        X['Volatility_21D'] = X['Returns'].rolling(21).std()
        X['Volatility_Ratio'] = X['Volatility_5D'] / X['Volatility_21D']
        
        # 3. Price momentum features
        X['Momentum_5D'] = X['Close'].pct_change(5)
        X['Momentum_21D'] = X['Close'].pct_change(21)
        X['Momentum_Ratio'] = X['Momentum_5D'] / X['Momentum_21D']
        
        # 4. Volume features
        X['Volume_MA5'] = X['Volume'].rolling(5).mean()
        X['Volume_MA21'] = X['Volume'].rolling(21).mean()
        X['Volume_Ratio'] = X['Volume'] / X['Volume_MA21']
        
        # 5. Price range features
        X['Range'] = (X['High'] - X['Low']) / X['Close']
        X['Body'] = (X['Close'] - X['Open']).abs() / X['Close']
        X['Upper_Shadow'] = (X['High'] - X[['Open', 'Close']].max(axis=1)) / X['Close']
        X['Lower_Shadow'] = (X[['Open', 'Close']].min(axis=1) - X['Low']) / X['Close']
        
        # 6. Market regime features
        X['MA50'] = X['Close'].rolling(50).mean()
        X['MA200'] = X['Close'].rolling(200).mean()
        X['Trend_Strength'] = (X['Close'] - X['MA50']) / X['MA50']
        X['Trend_Regime'] = np.where(X['Close'] > X['MA200'], 1, -1)
        
        # 7. Mean reversion features
        X['Z_Score_21D'] = (X['Close'] - X['Close'].rolling(21).mean()) / X['Close'].rolling(21).std()
        X['Z_Score_63D'] = (X['Close'] - X['Close'].rolling(63).mean()) / X['Close'].rolling(63).std()
        
        # 8. Lagged features
        for lag in [1, 2, 5]:
            X[f'Returns_lag{lag}'] = X['Returns'].shift(lag)
            X[f'Volume_lag{lag}'] = X['Volume'].pct_change(lag)
        
        # Handle infinite and missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill().fillna(0)
        
        # Ensure all values are finite
        if not np.all(np.isfinite(X)):
            print("[WARNING] Non-finite values detected in features after processing")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.ffill().bfill().fillna(0)
        
        # Remove highly correlated features
        X_filtered = self._remove_highly_correlated(X, threshold=0.85)  # More aggressive correlation threshold
        
        # Feature selection using mutual information - keep top 30 features
        selected_features = self._select_features_mutual_info(X_filtered, y, k=30)
        
        # Scale features with robust scaler to handle outliers
        from sklearn.preprocessing import RobustScaler
        self.scalers['robust'] = RobustScaler()
        X_scaled = self.scalers['robust'].fit_transform(X_filtered[selected_features])
        
        print(f"\n[INFO] Using {len(selected_features)} selected features from {len(X.columns)} initial features")
        print(f"      Training samples: {len(X_scaled)}")
        
        return X_scaled, y, selected_features
    
    def _remove_highly_correlated(self, X, threshold=0.95):
        # Calculate correlation matrix
        corr_matrix = X.corr()
        
        # Get upper triangular matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find index of feature columns with correlation greater than threshold
        to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) > threshold)]
        
        # Drop features 
        X_filtered = X.drop(to_drop, axis=1)
        
        return X_filtered
    
    def _select_features_mutual_info(self, X, y, k=30):
        from sklearn.feature_selection import mutual_info_regression
        mutual_info = mutual_info_regression(X, y)
        mutual_info_series = pd.Series(mutual_info, index=X.columns)
        selected_features = mutual_info_series.nlargest(k).index.tolist()
        
        return selected_features
    
    def objective(self, trial, X, y, model_type):
        """Objective function for Optuna hyperparameter optimization with enhanced search space"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'gamma': trial.suggest_float('gamma', 0, 5, step=0.1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, step=0.1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, step=0.1),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50)
        }
        
        # Model-specific parameters and setup
        if model_type == 'xgb':
            model = XGBRegressor(
                **params, 
                random_state=42, 
                n_jobs=-1,
                enable_categorical=False  # Add this for XGBoost 2.0+
            )
            fit_params = {
                'eval_metric': 'rmse',
                'verbose': 0
            }
        elif model_type == 'lgbm':
            model = LGBMRegressor(
                **params, 
                random_state=42, 
                n_jobs=-1,
                force_row_wise=True  # Add this for LightGBM
            )
            fit_params = {
                'eval_metric': 'rmse',
                'early_stopping_rounds': 50,
                'verbose': 0
            }
        elif model_type == 'catboost':
            model = CatBoostRegressor(
                **{k: v for k, v in params.items() if k != 'n_estimators'},
                iterations=params['n_estimators'],
                random_seed=42,
                verbose=0,
                allow_writing_files=False  # Disable writing files during training
            )
            fit_params = {
                'early_stopping_rounds': 50,
                'verbose': 0
            }
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        # Convert to numpy arrays for consistent handling
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        for train_idx, val_idx in tscv.split(X_array):
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]
            
            # Handle model fitting with appropriate parameters
            if model_type == 'xgb':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    **fit_params
                )
            else:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    **fit_params
                )
            
            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        return np.mean(scores)
    
    def _create_baseline_predictions(self, y_train, y_test):
        """
        Create simple baseline predictions for comparison.
        
        Args:
            y_train: Training target values
            y_test: Test target values
            
        Returns:
            Dictionary of baseline predictions
        """
        # Mean baseline
        mean_pred = np.full_like(y_test, y_train.mean())
        
        # Last value baseline
        last_value = y_train.iloc[-1] if hasattr(y_train, 'iloc') else y_train[-1]
        last_value_pred = np.full_like(y_test, last_value)
        
        # Naive forecast (last observed value)
        naive_pred = np.roll(np.concatenate(([y_train[-1]], y_test)), 1)[1:]
        
        return {
            'mean': mean_pred,
            'last_value': last_value_pred,
            'naive': naive_pred
        }
    
    def _evaluate_predictions(self, y_true, y_pred, baseline_preds=None):
        """
        Evaluate predictions using multiple metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            baseline_preds: Dictionary of baseline predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'Direction_Accuracy': np.mean((y_true * y_pred) > 0) if len(y_true) > 0 else 0
        }
        
        if baseline_preds:
            for name, pred in baseline_preds.items():
                metrics[f'Baseline_{name}_RMSE'] = np.sqrt(mean_squared_error(y_true, pred))
                metrics[f'Baseline_{name}_Direction'] = np.mean((y_true * pred) > 0) if len(y_true) > 0 else 0
        
        return metrics
    
    def walk_forward_validation(self, X, y, train_size=0.8, step=1):
        """
        Perform walk-forward validation on time series data.
        
        Args:
            X: Features
            y: Target
            train_size: Proportion of data to use for initial training
            step: Number of steps to move forward in each iteration
            
        Returns:
            Dictionary of evaluation metrics and predictions
        """
        # Ensure we're working with numpy arrays
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        n_samples = len(X_array)
        train_size = int(n_samples * train_size)
        
        all_preds = []
        all_actuals = []
        
        for i in range(train_size, n_samples, step):
            # Split data
            X_train, X_test = X_array[:i], X_array[i:i+step]
            y_train, y_test = y_array[:i], y_array[i:i+step]
            
            if len(X_test) == 0:
                continue
            
            try:
                # Ensure we have data to train on
                if len(X_train) < 5 or len(y_train) < 5:
                    continue
                    
                # Train model on current window
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Convert to numpy arrays if they're pandas objects
                X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
                y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
                X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
                
                model.fit(X_train_array, y_train_array)
                
                # Make predictions
                y_pred = model.predict(X_test_array)
                
                # Store results
                all_preds.extend(y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred)
                all_actuals.extend(y_test.tolist() if hasattr(y_test, 'tolist') else y_test)
                
            except Exception as e:
                print(f"Warning: Error in walk-forward iteration: {str(e)}")
                continue
        
        # Calculate metrics
        baseline_preds = self._create_baseline_predictions(y_array[:train_size], np.array(all_actuals))
        metrics = self._evaluate_predictions(np.array(all_actuals), np.array(all_preds), baseline_preds)
        
        return {
            'metrics': metrics,
            'predictions': all_preds,
            'actuals': all_actuals,
            'baseline_preds': baseline_preds
        }
    
    def train_ensemble_model(self, data, n_trials=50, cv_splits=5, use_walk_forward=False):
        """
        Train an ensemble of XGBoost, LightGBM, and CatBoost with enhanced regularization,
        cross-validation, and hyperparameter optimization using Optuna.
        
        Args:
            data: Input data with features and target
            n_trials: Number of Optuna trials for hyperparameter optimization
            cv_splits: Number of cross-validation splits
            use_walk_forward: Whether to use walk-forward validation
            
        Returns:
            Trained ensemble model and performance metrics
        """
        def create_objective(X, y, model_type, tscv):
            """Create an objective function for Optuna optimization"""
            def objective(trial):
                # Define parameter search space
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
                    'gamma': trial.suggest_float('gamma', 0, 5, step=0.1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, step=0.1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, step=0.1),
                    'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 50)
                }
                
                # Create a fresh model instance for each trial
                if model_type == 'xgb':
                    model = XGBRegressor(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        subsample=params['subsample'],
                        colsample_bytree=params['colsample_bytree'],
                        min_child_weight=params['min_child_weight'],
                        gamma=params['gamma'],
                        reg_alpha=params['reg_alpha'],
                        reg_lambda=params['reg_lambda'],
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_type == 'lgbm':
                    model = LGBMRegressor(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        subsample=params['subsample'],
                        colsample_bytree=params['colsample_bytree'],
                        reg_alpha=params['reg_alpha'],
                        reg_lambda=params['reg_lambda'],
                        min_child_samples=params['min_child_samples'],
                        random_state=42,
                        n_jobs=-1,
                        early_stopping_round=50
                    )
                else:  # catboost
                    model = CatBoostRegressor(
                        iterations=params['n_estimators'],
                        depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        subsample=params['subsample'],
                        colsample_bylevel=params['colsample_bytree'],
                        l2_leaf_reg=params['reg_lambda'],
                        random_seed=42,
                        verbose=0,
                        thread_count=-1,
                        early_stopping_rounds=50
                    )
                
                # Perform cross-validation
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Handle model fitting based on model type
                    if hasattr(model, 'fit'):
                        try:
                            # Set common parameters
                            if hasattr(model, 'set_params'):
                                model_params = model.get_params()
                                # Disable verbosity for all models
                                if 'verbose' in model_params:
                                    model.set_params(verbose=0)
                                # Disable early stopping if eval_set is not supported
                                if 'early_stopping_rounds' in model_params and not hasattr(model, 'eval_set'):
                                    model.set_params(early_stopping_rounds=None)
                            
                            # For models that support eval_set (XGBoost, LightGBM, CatBoost)
                            if hasattr(model, 'eval_set'):
                                try:
                                    # Try with eval_set if the model supports it
                                    model.fit(
                                        X_train, y_train,
                                        eval_set=[(X_val, y_val)],
                                        verbose=False
                                    )
                                except (TypeError, ValueError) as e:
                                    # If eval_set fails, try without it
                                    model.fit(X_train, y_train)
                            else:
                                # For models that don't support eval_set
                                model.fit(X_train, y_train)
                                
                        except Exception as e:
                            # If any other error occurs, try with minimal parameters
                            try:
                                model = model.__class__()  # Create new instance with default params
                                model.fit(X_train, y_train)
                            except Exception as e2:
                                print(f"Error fitting model: {str(e2)}")
                                raise
                    
                    y_pred = model.predict(X_val)
                    cv_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
                
                return np.mean(cv_scores)
            return objective
            
        try:
            # Prepare features and target
            X, y, feature_names = self.prepare_features(data)
            
            # Convert to numpy arrays for consistent handling
            X = X.values if hasattr(X, 'values') else np.array(X)
            y = y.values if hasattr(y, 'values') else np.array(y)
            
            # Store feature names for later use
            self.feature_names = feature_names
            
            # If using walk-forward validation
            if use_walk_forward:
                print("\nPerforming walk-forward validation...")
                wf_results = self.walk_forward_validation(X, y)
                
                print("\nWalk-Forward Validation Results:")
                print("-" * 50)
                for metric, value in wf_results['metrics'].items():
                    if 'Baseline' in metric:
                        continue
                    print(f"{metric}: {value:.4f}")
                    
                    # Print baseline comparison if available
                    baseline_metric = f"Baseline_{metric}_RMSE"
                    if baseline_metric in wf_results['metrics']:
                        print(f"  - Best Baseline RMSE: {wf_results['metrics'][baseline_metric]:.4f}")
                
                # Add walk-forward metrics to results
                self.walk_forward_metrics = wf_results['metrics']
                
                # If we're only doing walk-forward, return early
                if not n_trials:
                    return wf_results['metrics']
            
            # Define base models with default parameters and ensure no early stopping
            base_models = {
                'xgb': XGBRegressor(
                    n_estimators=1000,
                    max_depth=6,
                    learning_rate=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=1,
                    gamma=0,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=None,
                    eval_metric=None
                ),
                'lgbm': LGBMRegressor(
                    n_estimators=2000,
                    max_depth=-1,
                    learning_rate=0.01,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_round=None,  # Note: LightGBM uses singular 'round'
                    verbose=-1  # Disable all LightGBM output
                ),
                'catboost': CatBoostRegressor(
                    iterations=2000,
                    depth=6,
                    learning_rate=0.01,
                    l2_leaf_reg=3,
                    random_seed=42,
                    thread_count=-1,
                    verbose=0,
                    early_stopping_rounds=None
                )
            }
            
            # Initialize evaluation metrics
            evaluation = {}
            
            # Create time series cross-validator
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            
            # Train each model with hyperparameter optimization
            self.models = {}
            for model_name, model in base_models.items():
                if not hasattr(model, 'fit'):
                    print(f"Skipping {model_name} as it doesn't have a fit method")
                    continue
                
                # Optimize hyperparameters using Optuna
                study = optuna.create_study(direction='minimize')
                study.optimize(
                    create_objective(X, y, model_name, tscv),
                    n_trials=n_trials,
                    n_jobs=-1
                )
                
                # Store best parameters
                self.best_params[model_name] = study.best_params
                
                # Update model with best parameters
                if model_name == 'xgb':
                    model = XGBRegressor(
                        n_estimators=study.best_params['n_estimators'],
                        max_depth=study.best_params['max_depth'],
                        learning_rate=study.best_params['learning_rate'],
                        subsample=study.best_params['subsample'],
                        colsample_bytree=study.best_params['colsample_bytree'],
                        min_child_weight=study.best_params['min_child_weight'],
                        gamma=study.best_params['gamma'],
                        reg_alpha=study.best_params['reg_alpha'],
                        reg_lambda=study.best_params['reg_lambda'],
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_name == 'lgbm':
                    model = LGBMRegressor(
                        n_estimators=study.best_params['n_estimators'],
                        max_depth=study.best_params['max_depth'],
                        learning_rate=study.best_params['learning_rate'],
                        subsample=study.best_params['subsample'],
                        colsample_bytree=study.best_params['colsample_bytree'],
                        reg_alpha=study.best_params['reg_alpha'],
                        reg_lambda=study.best_params['reg_lambda'],
                        min_child_samples=study.best_params['min_child_samples'],
                        random_state=42,
                        n_jobs=-1,
                        early_stopping_round=50
                    )
                else:  # catboost
                    model = CatBoostRegressor(
                        iterations=study.best_params['n_estimators'],
                        depth=study.best_params['max_depth'],
                        learning_rate=study.best_params['learning_rate'],
                        subsample=study.best_params['subsample'],
                        colsample_bylevel=study.best_params['colsample_bytree'],
                        l2_leaf_reg=study.best_params['reg_lambda'],
                        random_seed=42,
                        verbose=0,
                        thread_count=-1,
                        early_stopping_rounds=50
                    )
                
                # Store the trained model
                self.models[model_name] = model
                
                # Evaluate the model with cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Simplified model fitting without early stopping
                    try:
                        # For models that support verbose parameter
                        if hasattr(model, 'set_params') and hasattr(model, 'get_params') and 'verbose' in model.get_params():
                            model.set_params(verbose=0)
                        
                        # Fit the model
                        model.fit(X_train, y_train)
                        
                    except Exception as e:
                        print(f"Warning: Error fitting model: {str(e)}")
                        # If fitting fails, try with default parameters
                        try:
                            model = model.__class__()
                            model.fit(X_train, y_train)
                        except Exception as e2:
                            print(f"Error refitting model with default parameters: {str(e2)}")
                            raise
                    y_pred = model.predict(X_val)
                    
                    # Calculate direction accuracy
                    if len(y_val) > 1:  # Need at least 2 points to calculate direction
                        direction_acc = np.mean((y_pred[1:] * np.diff(y_val)) > 0)
                    else:
                        direction_acc = 0.0
                    
                    cv_scores.append({
                        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
                        'MAE': mean_absolute_error(y_val, y_pred),
                        'R2': r2_score(y_val, y_pred),
                        'Direction_Accuracy': direction_acc
                    })
                
                # Store evaluation metrics
                evaluation[model_name] = {
                    'RMSE': np.mean([s['RMSE'] for s in cv_scores]),
                    'RMSE_std': np.std([s['RMSE'] for s in cv_scores]),
                    'MAE': np.mean([s['MAE'] for s in cv_scores]),
                    'R2': np.mean([s['R2'] for s in cv_scores]),
                    'R2_std': np.std([s['R2'] for s in cv_scores]),
                    'Direction_Accuracy': np.mean([s['Direction_Accuracy'] for s in cv_scores])
                }
                
                print(f"{model_name.upper()} - "
                      f"RMSE: {evaluation[model_name]['RMSE']:.4f} ± {evaluation[model_name]['RMSE_std']:.4f}, "
                      f"R²: {evaluation[model_name]['R2']:.4f} ± {evaluation[model_name]['R2_std']:.4f}, "
                      f"Dir. Acc: {evaluation[model_name]['Direction_Accuracy']:.2%}")
            
            # Create stacking ensemble if we have at least 2 models
            valid_models = {name: model for name, model in self.models.items() 
                          if name != 'stacking' and hasattr(model, 'predict')}
            
            if len(valid_models) >= 2:
                print("\nCreating stacking ensemble...")
                try:
                    # Prepare base models list for stacking
                    base_models_list = []
                    for name, model in valid_models.items():
                        # Create a fresh instance with safe defaults
                        if 'xgb' in name.lower() and hasattr(model, 'set_params'):
                            model_params = model.get_params()
                            safe_params = {k: v for k, v in model_params.items() 
                                        if k not in ['early_stopping_rounds', 'eval_metric', 'verbose']}
                            model = XGBRegressor(**safe_params)
                            model.set_params(early_stopping_rounds=None, eval_metric=None, verbose=0)
                        elif 'lgbm' in name.lower() and hasattr(model, 'set_params'):
                            model_params = model.get_params()
                            safe_params = {k: v for k, v in model_params.items() 
                                        if k not in ['early_stopping_round', 'eval_metric', 'verbose']}
                            model = LGBMRegressor(**safe_params)
                            model.set_params(early_stopping_round=None, verbose=-1)
                        base_models_list.append((name, model))
                    
                    if len(base_models_list) >= 2:  # Double-check we still have enough models
                        # Create meta-model (simpler than base models) with all verbosity and early stopping disabled
                        meta_model = XGBRegressor(
                            n_estimators=100,
                            max_depth=3,
                            learning_rate=0.1,
                            random_state=42,
                            n_jobs=-1,
                            early_stopping_rounds=None,
                            eval_metric=None,
                            verbose=0
                        )
                        
                        # Create stacking ensemble
                        self.models['stacking'] = StackingRegressor(
                            estimators=base_models_list,
                            final_estimator=meta_model,
                            cv=3,
                            n_jobs=-1,
                            verbose=0
                        )
                        
                        # Train the stacking ensemble
                        self.models['stacking'].fit(X, y)
                        print("Stacking ensemble trained successfully")
                    else:
                        print("Warning: Not enough valid models for stacking ensemble")
                        
                        cv_scores.append({
                            'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
                            'MAE': mean_absolute_error(y_val, y_pred),
                            'R2': r2_score(y_val, y_pred),
                            'Direction_Accuracy': direction_acc
                        })
                    
                    # Store stacking model evaluation
                    evaluation['stacking'] = {
                        'RMSE': np.mean([s['RMSE'] for s in cv_scores]),
                        'RMSE_std': np.std([s['RMSE'] for s in cv_scores]),
                        'MAE': np.mean([s['MAE'] for s in cv_scores]),
                        'R2': np.mean([s['R2'] for s in cv_scores]),
                        'R2_std': np.std([s['R2'] for s in cv_scores]),
                        'Direction_Accuracy': np.mean([s['Direction_Accuracy'] for s in cv_scores])
                    }
                    
                    print(f"\nSTACKING - "
                          f"RMSE: {evaluation['stacking']['RMSE']:.4f} ± {evaluation['stacking']['RMSE_std']:.4f}, "
                          f"R²: {evaluation['stacking']['R2']:.4f} ± {evaluation['stacking']['R2_std']:.4f}, "
                          f"Dir. Acc: {evaluation['stacking']['Direction_Accuracy']:.2%}")
                    
                    # Fit stacking model on full dataset
                    self.models['stacking'].fit(X, y)
                    
                except Exception as e:
                    print(f"Error creating stacking ensemble: {str(e)}")
                    evaluation['stacking'] = {
                        'RMSE': np.nan,
                        'RMSE_std': np.nan,
                        'MAE': np.nan,
                        'R2': np.nan,
                        'R2_std': np.nan,
                        'Direction_Accuracy': np.nan
                    }
            else:
                print("Not enough base models available for stacking. Need at least 2 models.")
                evaluation['stacking'] = {
                    'RMSE': np.nan,
                    'RMSE_std': np.nan,
                    'MAE': np.nan,
                    'R2': np.nan,
                    'R2_std': np.nan,
                    'Direction_Accuracy': np.nan
                }
            
            # Evaluate stacking model with cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                self.models['stacking'].fit(X_train, y_train)
                y_pred = self.models['stacking'].predict(X_val)
                
                # Calculate direction accuracy
                if len(y_val) > 1:  # Need at least 2 points to calculate direction
                    direction_acc = np.mean((y_pred[1:] * np.diff(y_val)) > 0)
                else:
                    direction_acc = 0.0
                
                cv_scores.append({
                    'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
                    'MAE': mean_absolute_error(y_val, y_pred),
                    'R2': r2_score(y_val, y_pred),
                    'Direction_Accuracy': direction_acc
                })
            
            # Store stacking model evaluation
            evaluation['stacking'] = {
                'RMSE': np.mean([s['RMSE'] for s in cv_scores]),
                'RMSE_std': np.std([s['RMSE'] for s in cv_scores]),
                'MAE': np.mean([s['MAE'] for s in cv_scores]),
                'R2': np.mean([s['R2'] for s in cv_scores]),
                'R2_std': np.std([s['R2'] for s in cv_scores]),
                'Direction_Accuracy': np.mean([s['Direction_Accuracy'] for s in cv_scores])
            }
            
            print(f"\nSTACKING - "
                  f"RMSE: {evaluation['stacking']['RMSE']:.4f} ± {evaluation['stacking']['RMSE_std']:.4f}, "
                  f"R²: {evaluation['stacking']['R2']:.4f} ± {evaluation['stacking']['R2_std']:.4f}, "
                  f"Dir. Acc: {evaluation['stacking']['Direction_Accuracy']:.2%}")
            
            # Train final models on full dataset
            print("\nTraining final models on full dataset...")
            for name, model in self.models.items():
                if name != 'stacking' and hasattr(model, 'fit'):
                    # Standard model fitting with error handling
                    try:
                        # Handle model fitting based on model type
                        if hasattr(model, 'fit'):
                            # Set common parameters
                            if hasattr(model, 'set_params'):
                                model_params = model.get_params()
                                # Disable verbosity for all models
                                if 'verbose' in model_params:
                                    model.set_params(verbose=0)
                                # Disable early stopping for final fit
                                if 'early_stopping_rounds' in model_params:
                                    model.set_params(early_stopping_rounds=None)
                            
                            # Fit the model on the entire dataset
                            model.fit(X, y)
                        
                    except Exception as e:
                        print(f"Warning: Error fitting model on full dataset: {str(e)}")
                        # If fitting fails, try with default parameters
                        try:
                            # Create a new instance with default parameters
                            model = model.__class__()
                            model.fit(X, y)
                        except Exception as e2:
                            print(f"Error refitting model with default parameters: {str(e2)}")
                            raise
            
            return evaluation
            
        except Exception as e:
            print(f"Error in train_ensemble_model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _remove_highly_correlated(self, X, threshold=0.95):
        """Remove highly correlated features to reduce redundancy"""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            print(f"[INFO] Removing {len(to_drop)} highly correlated features: {', '.join(to_drop)}")
            return X.drop(columns=to_drop)
        return X
    
    def _select_features_mutual_info(self, X, y, k=30):
        """Select top k features using mutual information"""
        from sklearn.feature_selection import mutual_info_regression
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        
        # Select top k features
        selected_features = mi_scores.head(k).index.tolist()
        
        # Print top 10 features by importance
        print("\nTop 10 most predictive features:")
        for i, (feature, score) in enumerate(mi_scores.head(10).items(), 1):
            print(f"  {i}. {feature}: {score:.4f}")
        
        return selected_features
    
    def predict(self, data, model_type='stacking'):
        """Make predictions using the specified model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Available models: {list(self.models.keys())}")
            
        X, _, _ = self.prepare_features(data)
        return self.models[model_type].predict(X[-1].reshape(1, -1))[0]
