import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import shutil
from typing import List
from typing import Tuple, Dict, Any, Optional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from data.collectors import RealTimeDataCollector
from utils.preprocessor import DataPreprocessor
from models.online_learning import OnlineLearningManager
from config.settings import config

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.data_collector = RealTimeDataCollector(ticker)
        self.online_learning_manager = OnlineLearningManager()
        self.is_online_ready = False
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=config.model.LSTM_UNITS[0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(0.2))
        
        # Additional LSTM layers
        for i, units in enumerate(config.model.LSTM_UNITS[1:], 1):
            model.add(LSTM(units=units, return_sequences=(i < len(config.model.LSTM_UNITS[1:]))))
            model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def train(self, 
              data: pd.DataFrame = None,
              save_model: bool = True) -> Dict[str, Any]:
        """Train the LSTM model"""
        try:
            # Get data if not provided
            if data is None:
                data = self.data_collector.get_historical_data()
            
            if data.empty:
                raise ValueError(f"No data available for {self.ticker}")
            
            # Prepare data
            (X_train, y_train), (X_test, y_test), _ = self.preprocessor.prepare_training_data(
                data, save_scaler=save_model, ticker=self.ticker
            )
            
            # Build model
            self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    config.get_model_path(self.ticker, "best"),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=config.model.EPOCHS,
                batch_size=config.model.BATCH_SIZE,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save final model and scaler
            if save_model:
                model_path = config.get_model_path(self.ticker)
                self.model.save(model_path)
                self.preprocessor.save_scaler(self.ticker)
                logger.info(f"Model and scaler saved to {model_path}")
            
            # Initialize online learning
            self.is_online_ready = True
            self.online_learning_manager.initialize_ticker(self.ticker)
            
            # Evaluate model
            test_loss, test_mae, test_mse = self.model.evaluate(X_test, y_test, verbose=0)
            
            return {
                'history': history.history,
                'test_loss': test_loss,
                'test_mae': test_mae,
                'test_mse': test_mse
            }
            
        except Exception as e:
            logger.error(f"Error training model for {self.ticker}: {e}")
            raise
    
    def online_update(self, new_data: pd.DataFrame) -> bool:
        """Update model with new data using online learning"""
        if not self.is_online_ready or self.model is None:
            logger.warning(f"Model not ready for online learning: {self.ticker}")
            return False
        
        try:
            # Add data to buffer
            self.online_learning_manager.add_data(self.ticker, new_data)
            
            # Check if we should update the model
            if not self.online_learning_manager.should_update_model(self.ticker):
                return False
            
            logger.info(f"Starting online learning update for {self.ticker}")
            
            # Prepare online training data
            X_online, y_online = self.online_learning_manager.prepare_online_data(
                self.ticker, self.preprocessor
            )
            
            if X_online is None or y_online is None:
                logger.warning(f"No online data prepared for {self.ticker}")
                return False
            
            # Create a backup of current model
            if config.online_learning.BACKUP_MODELS:
                backup_path = config.get_model_path(self.ticker, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                self.model.save(backup_path)
                logger.debug(f"Model backup created: {backup_path}")
            
            # Configure optimizer for fine-tuning
            fine_tune_optimizer = Adam(learning_rate=config.online_learning.LEARNING_RATE)
            self.model.compile(
                optimizer=fine_tune_optimizer,
                loss='mean_squared_error',
                metrics=['mae', 'mse']
            )
            
            # Fine-tune model with new data
            online_history = self.model.fit(
                X_online, y_online,
                epochs=config.online_learning.EPOCHS_PER_UPDATE,
                batch_size=config.online_learning.BATCH_SIZE,
                verbose=0,  # Quiet training for online updates
                shuffle=True
            )
            
            # Save updated model and scaler
            online_model_path = config.get_online_model_path(self.ticker)
            self.model.save(online_model_path)
            self.preprocessor.save_scaler(self.ticker, config.online_learning.ONLINE_MODEL_SUFFIX)
            
            # Reset buffer
            self.online_learning_manager.reset_buffer(self.ticker)
            
            logger.info(f"Online learning completed for {ticker}. Final loss: {online_history.history['loss'][-1]:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Error in online learning for {self.ticker}: {e}")
            return False
    
    def load_model(self, version: str = "latest", enable_online: bool = True) -> bool:
        """Load pre-trained model and scaler"""
        try:
            model_path = config.get_model_path(self.ticker, version)
            scaler_loaded = self.preprocessor.load_scaler(self.ticker, version)
            
            if os.path.exists(model_path) and scaler_loaded:
                self.model = load_model(model_path)
                
                # Initialize online learning if enabled
                if enable_online:
                    self.is_online_ready = True
                    self.online_learning_manager.initialize_ticker(self.ticker)
                
                logger.info(f"Model and scaler loaded for {self.ticker}")
                return True
            else:
                logger.warning(f"No model or scaler found for {self.ticker}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model for {self.ticker}: {e}")
            return False
    
    def get_online_status(self) -> Dict[str, Any]:
        """Get online learning status"""
        return {
            'is_online_ready': self.is_online_ready,
            'buffer_info': self.online_learning_manager.get_buffer_info(self.ticker),
            'model_loaded': self.model is not None
        }
    
class ModelTrainer:
    """Orchestrates training for multiple tickers"""
    
    def __init__(self):
        self.models = {}
        
    def train_all_models(self, tickers: List[str]) -> Dict[str, Any]:
        """Train models for all specified tickers"""
        results = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Training model for {ticker}")
                
                model = LSTMModel(ticker)
                training_result = model.train()
                
                results[ticker] = {
                    'status': 'success',
                    'test_loss': training_result['test_loss'],
                    'test_mae': training_result['test_mae'],
                    'model': model
                }
                
                logger.info(f"Successfully trained model for {ticker}")
                
            except Exception as e:
                logger.error(f"Failed to train model for {ticker}: {e}")
                results[ticker] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results