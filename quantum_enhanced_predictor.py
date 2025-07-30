import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Quantum Computing
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.utils import QuantumInstance, algorithm_globals

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensemble Learning
from sklearn.ensemble import StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Time Series
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Feature Engineering
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator

# Advanced Models
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import optuna

# Custom imports
from advanced_indicators import AdvancedIndicators

class QuantumNeuralNetwork(nn.Module):
    """Quantum-inspired Neural Network for time series prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super(QuantumNeuralNetwork, self).__init__()
        
        # Quantum circuit
        self.n_qubits = min(8, input_dim)  # Limit number of qubits
        self.qc = QuantumCircuit(self.n_qubits)
        
        # Add parameters for each qubit
        self.theta = [Parameter(f'theta_{i}') for i in range(self.n_qubits * 2)]
        
        # Create quantum circuit
        for i in range(self.n_qubits):
            self.qc.ry(self.theta[i], i)
        
        for i in range(self.n_qubits - 1):
            self.qc.cx(i, i + 1)
        
        for i in range(self.n_qubits):
            self.qc.ry(self.theta[i + self.n_qubits], i)
        
        # Classical neural network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical part
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class MetaEnsemblePredictor:
    """Ultimate Stock Market Predictor with Quantum Enhancement"""
    
    def __init__(self, n_models: int = 5, use_quantum: bool = True, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_models = n_models
        self.use_quantum = use_quantum
        self.device = device
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.quantum_models = {}
        
        # Initialize random seed for reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        algorithm_globals.random_seed = self.seed
    
    def _create_quantum_circuit(self, input_dim: int) -> QuantumCircuit:
        """Create a parameterized quantum circuit"""
        n_qubits = min(8, input_dim)
        qc = QuantumCircuit(n_qubits)
        
        # Add parameters
        params = [Parameter(f'θ_{i}') for i in range(n_qubits * 2)]
        
        # Add gates
        for i in range(n_qubits):
            qc.ry(params[i], i)
        
        # Add entanglement
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Add final rotation
        for i in range(n_qubits):
            qc.ry(params[i + n_qubits], i)
        
        return qc
    
    def _create_quantum_model(self, input_dim: int) -> NeuralNetworkClassifier:
        """Create a quantum neural network model"""
        # Create quantum instance
        quantum_instance = QuantumInstance(
            Aer.get_backend('qasm_simulator'),
            shots=1024,
            seed_simulator=self.seed,
            seed_transpiler=self.seed
        )
        
        # Create quantum circuit
        qc = self._create_quantum_circuit(input_dim)
        
        # Create QNN
        qnn = TwoLayerQNN(
            n_qubits=min(8, input_dim),
            feature_map=qc,
            ansatz=qc,
            quantum_instance=quantum_instance
        )
        
        return qnn
    
    def _create_ensemble_models(self, input_dim: int) -> Dict:
        """Create an ensemble of diverse models"""
        models = {}
        
        # 1. XGBoost with different hyperparameters
        models['xgb1'] = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=self.seed
        )
        
        models['xgb2'] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.7,
            n_jobs=-1,
            random_state=self.seed + 1
        )
        
        # 2. LightGBM models
        models['lgbm1'] = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=self.seed
        )
        
        # 3. CatBoost model
        models['catboost'] = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=6,
            random_seed=self.seed,
            verbose=False
        )
        
        # 4. Quantum Neural Network
        if self.use_quantum and input_dim > 0:
            qnn = self._create_quantum_model(input_dim)
            self.quantum_models['qnn'] = qnn
        
        return models
    
    def _create_meta_learner(self) -> StackingRegressor:
        """Create a meta-learner to combine base models"""
        estimators = [
            ('xgb1', self.models['xgb1']),
            ('xgb2', self.models['xgb2']),
            ('lgbm1', self.models['lgbm1']),
            ('catboost', self.models['catboost'])
        ]
        
        return StackingRegressor(
            estimators=estimators,
            final_estimator=XGBRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=self.seed
            ),
            n_jobs=-1
        )
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators to the dataframe"""
        # Basic indicators
        df = add_all_ta_features(
            df, 
            open="Open", 
            high="High", 
            low="Low", 
            close="Close", 
            volume="Volume",
            fillna=True
        )
        
        # Advanced indicators
        df = AdvancedIndicators.calculate_all_indicators(df)
        
        # Add more advanced features
        df['price_volume'] = df['Close'] * df['Volume']
        df['high_div_low'] = df['High'] / df['Low']
        df['close_div_open'] = df['Close'] / df['Open']
        
        # Add lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Add moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Add volatility measures
        df['daily_returns'] = df['Close'].pct_change()
        df['volatility'] = df['daily_returns'].rolling(window=21).std() * np.sqrt(252)
        
        # Add RSI
        rsi = RSIIndicator(close=df['Close'])
        df['rsi'] = rsi.rsi()
        
        # Add Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        
        # Add MACD
        macd = MACD(close=df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Close', forecast_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Create target (future price movement)
        df['target'] = df[target_col].shift(-forecast_horizon)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_col, 'target', 'Date']]
        X = df[feature_cols]
        y = df['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return pd.DataFrame(X_scaled, columns=feature_cols, index=df.index), y
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the ensemble of models"""
        # Create base models
        self.models = self._create_ensemble_models(X.shape[1])
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y)
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.Series(
                    model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
        
        # Train meta-learner
        print("Training meta-learner...")
        self.meta_learner = self._create_meta_learner()
        self.meta_learner.fit(X, y)
        
        # Train quantum model if enabled
        if self.use_quantum and 'qnn' in self.quantum_models:
            print("Training quantum model...")
            X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32).to(self.device)
            
            # Convert to PyTorch dataset
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Initialize model and optimizer
            model = QuantumNeuralNetwork(input_dim=X.shape[1]).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            n_epochs = 100
            for epoch in range(n_epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
            
            self.quantum_models['qnn'] = model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble"""
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from base models
        base_predictions = {}
        for name, model in self.models.items():
            base_predictions[name] = model.predict(X_scaled)
        
        # Get meta-learner prediction
        meta_features = np.column_stack(list(base_predictions.values()))
        meta_prediction = self.meta_learner.predict(meta_features)
        
        # Get quantum prediction if available
        if self.use_quantum and 'qnn' in self.quantum_models:
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                quantum_pred = self.quantum_models['qnn'](X_tensor).cpu().numpy().flatten()
            
            # Combine with meta-learner prediction
            final_prediction = 0.7 * meta_prediction + 0.3 * quantum_pred
        else:
            final_prediction = meta_prediction
        
        return final_prediction
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict:
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
                'gamma': trial.suggest_float('gamma', 0, 5, step=0.1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, step=0.1)
            }
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model = XGBRegressor(**params, n_jobs=-1, random_state=self.seed)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                score = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        
        return study.best_params
