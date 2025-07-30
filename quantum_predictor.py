import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

class QuantumStockPredictor:
    """A quantum-enhanced stock price predictor using Qiskit's VQC (Variational Quantum Classifier)."""
    
    def __init__(self, n_qubits=4, n_layers=2, max_iter=100):
        """
        Initialize the quantum predictor.
        
        Args:
            n_qubits (int): Number of qubits to use in the quantum circuit
            n_layers (int): Number of layers in the variational circuit
            max_iter (int): Maximum number of optimization iterations
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.max_iter = max_iter
        self.scaler = MinMaxScaler()
        self.label_binarizer = LabelBinarizer()
        self.model = None
        # Initialize the Sampler without backend parameter (uses default simulator)
        self.sampler = Sampler()
        self.optimizer = COBYLA(maxiter=max_iter)
    
    def _create_quantum_circuit(self):
        """Create a parameterized quantum circuit for the model."""
        # Create a quantum circuit with n_qubits
        qc = QuantumCircuit(self.n_qubits)
        
        # Add feature map
        feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=2)
        qc.compose(feature_map, inplace=True)
        
        # Add variational form (ansatz)
        ansatz = RealAmplitudes(self.n_qubits, reps=self.n_layers)
        qc.compose(ansatz, inplace=True)
        
        # Create the quantum neural network
        qnn = SamplerQNN(
            circuit=qc,
            sampler=self.sampler,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
            interpret=lambda x: np.argmax(x, axis=1),
            output_shape=2  # Binary classification output (0 or 1)
        )
        
        return feature_map, ansatz, qnn
    
    def prepare_data(self, X, y):
        """
        Prepare data for quantum model training.
        
        Args:
            X (np.array): Features
            y (np.array): Target values
            
        Returns:
            tuple: (X_scaled, y_binary) scaled features and binarized labels
        """
        # Scale features to [0, 1] range
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert continuous target to binary classification (up/down movement)
        y_binary = np.sign(np.diff(y, prepend=y[0]))
        y_binary = (y_binary > 0).astype(int)  # 1 for up, 0 for down
        y_binary = self.label_binarizer.fit_transform(y_binary.reshape(-1, 1)).flatten()
        
        return X_scaled, y_binary
    
    def train(self, X, y):
        """
        Train the quantum model.
        
        Args:
            X (np.array): Training features
            y (np.array): Training target values
        """
        try:
            # Prepare data
            X_scaled, y_binary = self.prepare_data(X, y)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_binary, test_size=0.2, random_state=42
            )
            
            # Create quantum circuit and QNN
            feature_map, ansatz, qnn = self._create_quantum_circuit()
            
            # Initialize VQC (Variational Quantum Classifier)
            # For Qiskit Machine Learning 0.8.3, we need to pass the sampler and other parameters
            self.model = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                loss='cross_entropy',
                optimizer=self.optimizer,
                sampler=self.sampler,
                callback=None
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            print(f"Training accuracy: {train_score:.4f}")
            print(f"Test accuracy: {test_score:.4f}")
            
            return train_score, test_score
            
        except Exception as e:
            print(f"Error during quantum model training: {str(e)}")
            raise
        
        return train_score, test_score
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Predicted probabilities for each class
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
                
            X_scaled = self.scaler.transform(X)
            
            # For VQC, we need to handle the prediction output format
            raw_predictions = self.model.predict_proba(X_scaled)
            
            # Ensure we have probabilities in the correct shape (n_samples, n_classes)
            if isinstance(raw_predictions, list):
                # Convert list of arrays to 2D array
                return np.array(raw_predictions)
            elif len(raw_predictions.shape) == 1:
                # If we have a 1D array, convert to 2D with shape (n_samples, 1)
                return np.column_stack((1 - raw_predictions, raw_predictions))
            return raw_predictions
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Predicted class labels (0 or 1)
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
                
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def get_quantum_circuit(self):
        """Get the quantum circuit used in the model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.circuit


class HybridQuantumPredictor:
    """A hybrid quantum-classical predictor that combines quantum and classical models."""
    
    def __init__(self, quantum_predictor, classical_predictor):
        """
        Initialize the hybrid predictor.
        
        Args:
            quantum_predictor: An instance of QuantumStockPredictor
            classical_predictor: A trained classical predictor with predict_proba method
        """
        self.quantum_predictor = quantum_predictor
        self.classical_predictor = classical_predictor
        self.quantum_weight = 0.5  # Initial weight for quantum predictions
    
    def predict(self, X, quantum_weight=None):
        """
        Make predictions using both quantum and classical models.
        
        Args:
            X (np.array): Input features
            quantum_weight (float, optional): Weight for quantum predictions (0-1). 
                                           If None, uses the current weight.
                                           
        Returns:
            np.array: Combined predictions
        """
        if quantum_weight is not None:
            self.quantum_weight = max(0, min(1, quantum_weight))
            
        # Get predictions from both models
        q_preds = self.quantum_predictor.predict_proba(X)[:, 1]  # Probability of class 1 (up movement)
        c_preds = self.classical_predictor.predict_proba(X)[:, 1]
        
        # Combine predictions with weighted average
        combined = (self.quantum_weight * q_preds + 
                   (1 - self.quantum_weight) * c_preds)
        
        return (combined > 0.5).astype(int)  # Convert to binary predictions
    
    def set_quantum_weight(self, weight):
        """Set the weight for quantum predictions (0-1)."""
        self.quantum_weight = max(0, min(1, weight))
        return self.quantum_weight
