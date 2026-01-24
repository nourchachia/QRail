# File: src/models/outcome_predictor_xgb.py

"""
Outcome Predictor (XGBoost-based)

Predicts success probability (0-1) based on:
- Historical Actions Taken
- Prior Outcomes
- Context Bias (Weather, Maintenance Notes)

This model helps evaluate proposed resolutions by predicting their likelihood of success.
"""

import xgboost as xgb
import numpy as np
from pathlib import Path
import pickle
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutcomePredictor:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the XGBoost-based outcome predictor.
        
        Args:
            model_path: Optional path to load a pre-trained model
        """
        self.model = xgb.XGBRegressor(
            objective='reg:logistic',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.is_trained = False
        
        if model_path:
            # Check if the .json model file exists
            model_file = Path(model_path).with_suffix('.json')
            if model_file.exists():
                self.load(model_path)
                logger.info(f"Loaded pre-trained model from {model_path}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train the outcome predictor.
        
        Args:
            X: Training features [n_samples, n_features]
               Features = [Context Embedding + Proposed Action Features]
            y: Training labels - outcome_score (0-1) from historical data
            X_val: Optional validation features
            y_val: Optional validation labels
        
        Returns:
            Training history with metrics
        """
        logger.info(f"Training XGBoost model on {X.shape[0]} samples...")
        
        # Simple training without eval set for compatibility
        self.model.fit(X, y, verbose=True)
        
        self.is_trained = True
        
        # Evaluate on validation set if provided
        val_score = None
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_predictions = np.clip(val_predictions, 0.0, 1.0)
            # Calculate MSE
            val_score = np.mean((val_predictions - y_val) ** 2)
            logger.info(f"Validation MSE: {val_score:.4f}")
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        history = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_importance': feature_importance,
            'val_mse': val_score
        }
        
        logger.info(f"Training complete.")
        return history
    
    def predict(self, incident_vec: np.ndarray, resolution_vec: np.ndarray) -> np.ndarray:
        """
        Predict outcome probability for given incident and resolution.
        
        Args:
            incident_vec: Incident context embedding [batch_size, incident_dim]
            resolution_vec: Proposed resolution features [batch_size, resolution_dim]
        
        Returns:
            Predicted success probability [batch_size] in range [0, 1]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        # Combine vectors
        input_vec = np.concatenate([incident_vec, resolution_vec], axis=1)
        
        # Predict
        predictions = self.model.predict(input_vec)
        
        # Ensure predictions are in [0, 1] range
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outcomes for a batch of pre-combined features.
        
        Args:
            X: Combined features [batch_size, n_features]
        
        Returns:
            Predicted success probabilities [batch_size]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        predictions = self.model.predict(X)
        return np.clip(predictions, 0.0, 1.0)
    
    def save(self, filepath: str):
        """Save the trained model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = filepath.with_suffix('.json')
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'is_trained': self.is_trained,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate
        }
        
        metadata_path = filepath.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, filepath: str):
        """Load a trained model from disk."""
        filepath = Path(filepath)
        
        # Load XGBoost model
        model_path = filepath.with_suffix('.json')
        if model_path.exists():
            self.model.load_model(str(model_path))
            self.is_trained = True  # Set as trained after loading model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load metadata if available
        metadata_path = filepath.with_suffix('.pkl')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            # Override with saved metadata
            self.is_trained = metadata.get('is_trained', True)
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> dict:
        """
        Get feature importance scores.
        
        Args:
            feature_names: Optional list of feature names
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance.")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return dict(zip(feature_names, importance))


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Testing OutcomePredictor")
    print("=" * 70)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    incident_dim = 64
    resolution_dim = 32
    
    # Simulate incident vectors and resolution vectors
    X_incident = np.random.randn(n_samples, incident_dim)
    X_resolution = np.random.randn(n_samples, resolution_dim)
    
    # Combine features
    X = np.concatenate([X_incident, X_resolution], axis=1)
    
    # Simulate outcome scores (0-1)
    # Higher correlation with some features for testing
    y = np.clip(
        0.5 + 0.3 * X[:, 0] + 0.2 * X[:, incident_dim] + np.random.randn(n_samples) * 0.1,
        0, 1
    )
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Initialize and train
    predictor = OutcomePredictor()
    history = predictor.train(X_train, y_train, X_val, y_val)
    
    print("\nTraining History:")
    print(f"  Samples: {history['n_samples']}")
    print(f"  Features: {history['n_features']}")
    print(f"  Validation MSE: {history['val_mse']:.4f}")
    
    # Test prediction
    test_incident = X_incident[split_idx:split_idx+5]
    test_resolution = X_resolution[split_idx:split_idx+5]
    
    predictions = predictor.predict(test_incident, test_resolution)
    
    print("\nTest Predictions:")
    for i, (pred, actual) in enumerate(zip(predictions, y_val[:5])):
        print(f"  Sample {i+1}: Predicted={pred:.3f}, Actual={actual:.3f}")
    
    # Save model
    predictor.save("checkpoints/outcome_predictor/model")
    print("\n[OK] Model saved to checkpoints/outcome_predictor/model.json")
    
    # Test loading
    predictor2 = OutcomePredictor("checkpoints/outcome_predictor/model")
    predictions2 = predictor2.predict(test_incident, test_resolution)
    
    print("\n[OK] Loaded model predictions match:", np.allclose(predictions, predictions2))
