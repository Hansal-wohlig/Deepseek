import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Optional
from config.settings import config

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, scaler_type: str = "minmax"):
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        self.is_fitted = False
        self.feature_range = (0, 1) if scaler_type == "minmax" else None
        
    def save_scaler(self, ticker: str, version: str = "latest"):
        """Save scaler state to file"""
        try:
            scaler_path = config.get_scaler_path(ticker, version)
            with open(scaler_path, 'wb') as f:
                pickle.dump({
                    'scaler_type': self.scaler_type,
                    'scaler_state': self.scaler,
                    'is_fitted': self.is_fitted,
                    'feature_range': self.feature_range
                }, f)
            logger.debug(f"Scaler saved for {ticker}")
        except Exception as e:
            logger.error(f"Error saving scaler for {ticker}: {e}")
    
    def load_scaler(self, ticker: str, version: str = "latest") -> bool:
        """Load scaler state from file"""
        try:
            scaler_path = config.get_scaler_path(ticker, version)
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
            
            self.scaler_type = scaler_data['scaler_type']
            self.scaler = scaler_data['scaler_state']
            self.is_fitted = scaler_data['is_fitted']
            self.feature_range = scaler_data['feature_range']
            
            logger.debug(f"Scaler loaded for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error loading scaler for {ticker}: {e}")
            return False
    
    def partial_fit(self, new_data: np.ndarray):
        """Partially update scaler with new data"""
        try:
            if hasattr(self.scaler, 'partial_fit'):
                self.scaler.partial_fit(new_data)
            else:
                # For MinMaxScaler, we need to manually update min/max
                if hasattr(self.scaler, 'data_min_'):
                    self.scaler.data_min_ = np.minimum(self.scaler.data_min_, new_data.min(axis=0))
                    self.scaler.data_max_ = np.maximum(self.scaler.data_max_, new_data.max(axis=0))
                    self.scaler.scale_ = (self.feature_range[1] - self.feature_range[0]) / (
                        self.scaler.data_max_ - self.scaler.data_min_ + 1e-8)
                    self.scaler.min_ = self.feature_range[0] - self.scaler.data_min_ * self.scaler.scale_
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in partial fit: {e}")
    
    def create_sequences(self, data: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i])
            y.append(data[i, 0])  # Assuming Close price is the target
        return np.array(X), np.array(y)
    
    def prepare_training_data(self, 
                            data: pd.DataFrame, 
                            time_steps: int = None,
                            save_scaler: bool = True,
                            ticker: str = None) -> Tuple[Tuple, Tuple, pd.DataFrame]:
        """Prepare data for model training"""
        if time_steps is None:
            time_steps = config.model.TIME_STEPS
            
        # Use only Close price for simplicity
        price_data = data[['Close']].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(price_data)
        self.is_fitted = True
        
        # Save scaler if requested
        if save_scaler and ticker:
            self.save_scaler(ticker)
        
        # Split data
        split_idx = int(len(scaled_data) * config.model.TRAIN_TEST_SPLIT)
        train_data = scaled_data[:split_idx]
        test_data = scaled_data[split_idx:]
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data, time_steps)
        X_test, y_test = self.create_sequences(test_data, time_steps)
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return (X_train, y_train), (X_test, y_test), data
    
    def prepare_online_data(self, 
                          historical_data: pd.DataFrame, 
                          new_data: pd.DataFrame,
                          time_steps: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for online learning"""
        if time_steps is None:
            time_steps = config.model.TIME_STEPS
            
        # Combine historical and new data
        combined_data = pd.concat([historical_data, new_data], axis=0)
        price_data = combined_data[['Close']].values
        
        # Update scaler with new data
        self.partial_fit(new_data[['Close']].values)
        
        # Scale the combined data
        scaled_data = self.scaler.transform(price_data)
        
        # Use only the new portion for training (to avoid retraining on old data)
        new_start_idx = len(historical_data)
        training_data = scaled_data[new_start_idx - time_steps:]
        
        # Create sequences from the new data region
        X_online, y_online = self.create_sequences(training_data, time_steps)
        X_online = X_online.reshape((X_online.shape[0], X_online.shape[1], 1))
        
        return X_online, y_online
    
    def prepare_inference_data(self, recent_data: pd.DataFrame, time_steps: int = None) -> np.ndarray:
        """Prepare recent data for inference"""
        if time_steps is None:
            time_steps = config.model.TIME_STEPS
            
        if len(recent_data) < time_steps:
            logger.warning(f"Insufficient data: {len(recent_data)} < {time_steps}")
            return np.array([])
            
        price_data = recent_data[['Close']].values

        logger.info(f"Inference data shape: {price_data.shape}")
        logger.info(f"Inference data contains NaN: {np.isnan(price_data).any()}")
        logger.info(f"Scaler is fitted: {self.is_fitted}")
        
        if self.is_fitted:
            scaled_data = self.scaler.transform(price_data)
        else:
            logger.warning("Scaler not fitted during inference. Fitting on recent data.")
            scaled_data = self.scaler.fit_transform(price_data)
            self.is_fitted = True

        logger.info(f"Scaled inference data contains NaN: {np.isnan(scaled_data).any()}")
            
        # Create sequence for prediction
        sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)
        return sequence
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data to original scale"""
        if self.is_fitted:
            return self.scaler.inverse_transform(scaled_data)
        return scaled_data