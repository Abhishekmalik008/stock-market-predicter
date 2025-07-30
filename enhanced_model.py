import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    StackingRegressor, 
    VotingRegressor,
    RandomForestRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from advanced_indicators import AdvancedIndicators

class EnhancedStockPredictor:
    """Enhanced stock price predictor with advanced features and models"""
    
    def __init__(self, n_trials: int = 50, cv_splits: int = 5):
        """
        Initialize the enhanced stock predictor
        
        Args:
            n_trials: Number of optimization trials
            cv_splits: Number of cross-validation splits
        """
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.models: Dict = {}
        self.scaler = StandardScaler()
        self.best_params: Dict = {}
        self.feature_importance: Dict = {}
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            Tuple of (features, target)
        """
        # Add advanced technical indicators
        df = AdvancedIndicators.calculate_all_indicators(df)
        
        # Calculate target (next day's close price)
        target = df['Close'].shift(-1).dropna()
        
        # Remove rows with NaN values
        df = df.iloc[:-1]  # Remove last row since we don't have target for it
        df = df[df.index.isin(target.index)]
        
        # Drop non-feature columns
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        features = df[feature_cols]
        
        return features, target
    
    def optimize_hyperparameters(self, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              model_type: str) -> Dict:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X: Features
            y: Target
            model_type: Type of model ('xgb', 'lgbm', 'catboost')
            
        Returns:
            Dictionary of best parameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, step=0.1)
            }
            
            # Model-specific parameters
            if model_type == 'xgb':
                params.update({
                    'gamma': trial.suggest_float('gamma', 0, 5, step=0.1),
                    'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                    'tree_method': 'hist',
                    'enable_categorical': True,
                    'verbosity': 0
                })
                model = XGBRegressor(**params)
            elif model_type == 'lgbm':
                params.update({
                    'num_leaves': trial.suggest_int('num_leaves', 20, 3000, step=20),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100, step=5),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0, step=0.05),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0, step=0.05),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'verbosity': -1
                })
                model = LGBMRegressor(**params)
            elif model_type == 'catboost':
                params.update({
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10, log=True),
                    'random_strength': trial.suggest_float('random_strength', 1e-5, 10, log=True),
                    'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                    'od_type': 'Iter',
                    'od_wait': 50,
                    'verbose': False
                })
                model = CatBoostRegressor(**params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                # Train model
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                # Predict and score
                y_pred = model.predict(X_val_scaled)
                score = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, n_jobs=-1)
        
        return study.best_params
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train an ensemble of models
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary of trained models and their performance
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize base models
        base_models = {
            'xgb': XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42
            ),
            'lgbm': LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42
            ),
            'catboost': CatBoostRegressor(
                iterations=1000,
                learning_rate=0.01,
                depth=6,
                random_seed=42,
                verbose=False
            )
        }
        
        # Train base models with cross-validation
        cv_scores = {}
        
        for name, model in base_models.items():
            print(f"Training {name}...")
            
            # Optimize hyperparameters
            print(f"Optimizing {name} hyperparameters...")
            best_params = self.optimize_hyperparameters(X, y, name)
            
            # Update model with best parameters
            model.set_params(**best_params)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            scores = cross_val_score(
                model, X_scaled, y,
                cv=tscv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            
            cv_scores[name] = {
                'mean_rmse': -scores.mean(),
                'std_rmse': scores.std(),
                'params': best_params
            }
            
            # Train final model on full dataset
            model.fit(X_scaled, y)
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.Series(
                    model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
            
            # Store model
            self.models[name] = model
        
        # Create ensemble
        print("Training ensemble model...")
        
        # Stacking ensemble
        estimators = [
            ('xgb', self.models['xgb']),
            ('lgbm', self.models['lgbm']),
            ('catboost', self.models['catboost'])
        ]
        
        self.models['stacking'] = StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
            n_jobs=-1
        )
        
        # Train stacking model
        self.models['stacking'].fit(X_scaled, y)
        
        # Evaluate ensemble
        y_pred = self.models['stacking'].predict(X_scaled)
        ensemble_score = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        cv_scores['stacking'] = {
            'mean_rmse': ensemble_score['rmse'],
            'std_rmse': 0,
            'params': {}
        }
        
        return cv_scores
    
    def predict(self, X: pd.DataFrame, model: str = 'stacking') -> np.ndarray:
        """
        Make predictions using the specified model
        
        Args:
            X: Features
            model: Model to use for prediction ('xgb', 'lgbm', 'catboost', 'stacking')
            
        Returns:
            Array of predictions
        """
        if model not in self.models:
            raise ValueError(f"Model {model} not found. Available models: {list(self.models.keys())}")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.models[model].predict(X_scaled)
    
    def get_feature_importance(self, model: str = 'stacking') -> pd.Series:
        """
        Get feature importance from the specified model
        
        Args:
            model: Model to get feature importance from
            
        Returns:
            Series of feature importances
        """
        if model not in self.feature_importance:
            raise ValueError(f"Feature importance not available for model {model}")
        return self.feature_importance[model]
