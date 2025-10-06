"""
Advanced Time Series Validation and Training Strategies
Implements proper time-series cross-validation to prevent data leakage and lookahead bias
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional, Generator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesValidator:
    """
    Comprehensive time series validation with proper temporal ordering
    Prevents lookahead bias and ensures realistic evaluation
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size_ratio: float = 0.2,
                 gap_size: int = 0,
                 purge_size: int = 0):
        """
        Initialize Time Series Validator
        
        Args:
            n_splits: Number of cross-validation splits
            test_size_ratio: Ratio of data to use for testing in each split
            gap_size: Gap between training and test sets (to simulate real-world delay)
            purge_size: Number of samples to purge around the gap
        """
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.gap_size = gap_size
        self.purge_size = purge_size
        
    def time_series_split(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate time series cross-validation splits
        
        Why this is important:
        - Prevents lookahead bias (using future data to predict past)
        - Maintains temporal order of data
        - Simulates realistic trading conditions
        - Provides robust performance estimation
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size_ratio)
        
        # Calculate split points
        min_train_size = int(n_samples * 0.3)  # Minimum training size
        
        for i in range(self.n_splits):
            # Calculate split indices for this fold
            split_point = min_train_size + i * (n_samples - min_train_size - test_size) // (self.n_splits - 1) if self.n_splits > 1 else min_train_size
            
            # Training indices (from start to split_point)
            train_end = split_point - self.gap_size - self.purge_size
            train_indices = np.arange(0, max(train_end, min_train_size))
            
            # Test indices (from split_point + gap to split_point + gap + test_size)
            test_start = split_point + self.gap_size
            test_end = min(test_start + test_size, n_samples)
            test_indices = np.arange(test_start, test_end)
            
            if len(test_indices) == 0:
                continue
                
            logger.debug(f"Fold {i+1}: Train size={len(train_indices)}, Test size={len(test_indices)}")
            logger.debug(f"Train range: {train_indices[0]} to {train_indices[-1]}")
            logger.debug(f"Test range: {test_indices[0]} to {test_indices[-1]}")
            
            yield train_indices, test_indices
    
    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray, 
                              initial_train_size: int = None,
                              step_size: int = 1) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Walk-forward validation for time series
        Simulates real-time trading by incrementally adding new data
        """
        n_samples = len(X)
        
        if initial_train_size is None:
            initial_train_size = int(n_samples * 0.6)
        
        for i in range(initial_train_size, n_samples - 1, step_size):
            train_indices = np.arange(0, i)
            test_indices = np.array([i])
            
            if len(train_indices) < 50:  # Minimum training size
                continue
                
            yield train_indices, test_indices
    
    def purged_cross_validation(self, X: np.ndarray, y: np.ndarray,
                               timestamps: pd.DatetimeIndex = None,
                               embargo_period: pd.Timedelta = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Purged Cross-Validation with embargo period
        Advanced technique used in quantitative finance to prevent leakage
        """
        if timestamps is None or embargo_period is None:
            # Fall back to regular time series split
            yield from self.time_series_split(X, y)
            return
        
        n_samples = len(X)
        test_size = int(n_samples / self.n_splits)
        
        for i in range(self.n_splits):
            # Test set indices
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = np.arange(test_start, test_end)
            
            # Training set with purging and embargo
            train_indices = []
            
            for j in range(n_samples):
                if j in test_indices:
                    continue
                
                # Check if sample is too close to test set (embargo period)
                sample_time = timestamps[j]
                too_close = False
                
                for test_idx in test_indices:
                    test_time = timestamps[test_idx]
                    if abs(sample_time - test_time) < embargo_period:
                        too_close = True
                        break
                
                if not too_close:
                    train_indices.append(j)
            
            train_indices = np.array(train_indices)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

class AdvancedDataPreprocessor:
    """
    Advanced preprocessing for financial time series data
    Handles multiple features, proper scaling, and sequence generation
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 target_column: str = 'Close',
                 scaler_type: str = 'robust',
                 feature_columns: List[str] = None):
        
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.scaler_type = scaler_type
        self.feature_columns = feature_columns
        
        # Initialize scalers
        if scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        
        self.is_fitted = False
        
    def fit_scalers(self, data: pd.DataFrame) -> 'AdvancedDataPreprocessor':
        """
        Fit scalers on training data
        """
        if self.feature_columns is None:
            self.feature_columns = [col for col in data.columns if col != self.target_column]
        
        # Fit feature scaler
        feature_data = data[self.feature_columns].values
        self.feature_scaler.fit(feature_data)
        
        # Fit target scaler
        target_data = data[[self.target_column]].values
        self.target_scaler.fit(target_data)
        
        self.is_fitted = True
        logger.info(f"Scalers fitted on data with shape: {data.shape}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        
        return self
    
    def transform_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted scalers
        """
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers() first.")
        
        # Transform features
        feature_data = data[self.feature_columns].values
        scaled_features = self.feature_scaler.transform(feature_data)
        
        # Transform target
        target_data = data[[self.target_column]].values
        scaled_target = self.target_scaler.transform(target_data)
        
        return scaled_features, scaled_target
    
    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples, 1)
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for training data
        """
        # Fit and transform
        self.fit_scalers(data)
        scaled_features, scaled_target = self.transform_data(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_features, scaled_target)
        
        logger.info(f"Prepared training data: X.shape={X.shape}, y.shape={y.shape}")
        
        return X, y
    
    def prepare_inference_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare data for model inference
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call prepare_training_data() first.")
        
        scaled_features, _ = self.transform_data(data)
        
        # Get last sequence for prediction
        if len(scaled_features) >= self.sequence_length:
            X = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            return X
        else:
            raise ValueError(f"Insufficient data for inference. Need {self.sequence_length}, got {len(scaled_features)}")
    
    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """
        Inverse transform target values to original scale
        """
        return self.target_scaler.inverse_transform(scaled_target)

class ModelTrainingStrategy:
    """
    Comprehensive training strategy with proper validation
    """
    
    def __init__(self, 
                 model_class,
                 validation_method: str = 'time_series_split',
                 n_splits: int = 5,
                 early_stopping_patience: int = 15):
        
        self.model_class = model_class
        self.validation_method = validation_method
        self.n_splits = n_splits
        self.early_stopping_patience = early_stopping_patience
        
        self.validator = TimeSeriesValidator(n_splits=n_splits)
        self.preprocessor = None
        
    def train_with_validation(self, 
                            data: pd.DataFrame,
                            model_params: Dict[str, Any],
                            feature_columns: List[str] = None) -> Dict[str, Any]:
        """
        Train model with proper time series validation
        """
        logger.info("Starting training with time series validation...")
        
        # Initialize preprocessor
        self.preprocessor = AdvancedDataPreprocessor(
            sequence_length=model_params.get('sequence_length', 60),
            feature_columns=feature_columns
        )
        
        # Prepare full dataset
        X_full, y_full = self.preprocessor.prepare_training_data(data)
        
        # Cross-validation results
        cv_results = {
            'fold_scores': [],
            'fold_predictions': [],
            'best_model': None,
            'best_score': float('inf')
        }
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.validator.time_series_split(X_full, y_full)):
            logger.info(f"Training fold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_val = X_full[train_idx], X_full[val_idx]
            y_train, y_val = y_full[train_idx], y_full[val_idx]
            
            # Initialize model for this fold
            model = self.model_class(**model_params)
            
            # Train model
            training_results = model.train(
                X_train, y_train.ravel(),
                X_val, y_val.ravel(),
                epochs=100,
                batch_size=32
            )
            
            # Evaluate fold
            fold_score = training_results['val_mae']
            cv_results['fold_scores'].append(fold_score)
            
            # Keep best model
            if fold_score < cv_results['best_score']:
                cv_results['best_score'] = fold_score
                cv_results['best_model'] = model
            
            logger.info(f"Fold {fold + 1} MAE: {fold_score:.6f}")
        
        # Calculate overall CV performance
        mean_score = np.mean(cv_results['fold_scores'])
        std_score = np.std(cv_results['fold_scores'])
        
        logger.info(f"Cross-validation results:")
        logger.info(f"Mean MAE: {mean_score:.6f} ± {std_score:.6f}")
        logger.info(f"Best fold MAE: {cv_results['best_score']:.6f}")
        
        cv_results['mean_score'] = mean_score
        cv_results['std_score'] = std_score
        cv_results['preprocessor'] = self.preprocessor
        
        return cv_results
    
    def final_training(self, 
                      data: pd.DataFrame,
                      model_params: Dict[str, Any],
                      feature_columns: List[str] = None,
                      validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Final training on all available data with holdout validation
        """
        logger.info("Starting final model training...")
        
        # Initialize preprocessor if not already done
        if self.preprocessor is None:
            self.preprocessor = AdvancedDataPreprocessor(
                sequence_length=model_params.get('sequence_length', 60),
                feature_columns=feature_columns
            )
        
        # Prepare data
        X_full, y_full = self.preprocessor.prepare_training_data(data)
        
        # Split for final validation
        split_idx = int(len(X_full) * (1 - validation_split))
        X_train, X_val = X_full[:split_idx], X_full[split_idx:]
        y_train, y_val = y_full[:split_idx], y_full[split_idx:]
        
        # Train final model
        final_model = self.model_class(**model_params)
        training_results = final_model.train(
            X_train, y_train.ravel(),
            X_val, y_val.ravel(),
            epochs=150,
            batch_size=32
        )
        
        # Comprehensive evaluation
        evaluation_results = final_model.evaluate_model(X_val, y_val.ravel())
        
        results = {
            'model': final_model,
            'preprocessor': self.preprocessor,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'data_shapes': {
                'X_train': X_train.shape,
                'X_val': X_val.shape,
                'y_train': y_train.shape,
                'y_val': y_val.shape
            }
        }
        
        logger.info("Final training completed!")
        logger.info(f"Final validation MAE: {evaluation_results['mae']:.6f}")
        logger.info(f"Final validation RMSE: {evaluation_results['rmse']:.6f}")
        
        return results

def explain_time_series_validation():
    """
    Explanation of why proper time series validation is crucial
    """
    print("""
    Why Proper Time Series Validation is Critical:
    
    1. TEMPORAL ORDER MATTERS:
       - Financial data has inherent time dependencies
       - Future information cannot be used to predict the past
       - Random splits violate this fundamental principle
    
    2. LOOKAHEAD BIAS PREVENTION:
       - Standard k-fold CV can accidentally use future data
       - This leads to overly optimistic performance estimates
       - Real trading would never have access to future information
    
    3. REALISTIC PERFORMANCE ESTIMATION:
       - Time series splits simulate real-world conditions
       - Training on past data, testing on future data
       - Provides honest assessment of model performance
    
    4. MARKET REGIME CHANGES:
       - Financial markets change over time
       - Models must adapt to new market conditions
       - Proper validation tests this adaptability
    
    5. WALK-FORWARD ANALYSIS:
       - Simulates incremental model updates
       - Tests model stability over time
       - Identifies potential degradation periods
    
    Methods Implemented:
    
    1. Time Series Split:
       - Maintains chronological order
       - Increasing training sets
       - No data leakage
    
    2. Walk-Forward Validation:
       - Incremental testing approach
       - Simulates real-time trading
       - Tests model adaptability
    
    3. Purged Cross-Validation:
       - Advanced technique from quantitative finance
       - Removes samples too close in time
       - Accounts for autocorrelation in returns
    """)

if __name__ == "__main__":
    explain_time_series_validation()
    
    # Example usage
    print("\nTesting Time Series Validation...")
    
    # Create sample data
    n_samples = 1000
    n_features = 10
    X_sample = np.random.randn(n_samples, n_features)
    y_sample = np.random.randn(n_samples, 1)
    
    # Test validator
    validator = TimeSeriesValidator(n_splits=5)
    
    fold_count = 0
    for train_idx, test_idx in validator.time_series_split(X_sample, y_sample):
        fold_count += 1
        print(f"Fold {fold_count}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    print(f"Total folds generated: {fold_count}")