import numpy as np
import pandas as pd
import logging
import os
import shutil
from typing import Dict, Any, Optional, List
from data.collectors import RealTimeDataCollector, DataBuffer
from utils.preprocessor import DataPreprocessor
from models.trainer import LSTMModel
from config.settings import config

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, ticker: str, enable_online_learning: bool = True):
        self.ticker = ticker
        self.model_wrapper = LSTMModel(ticker)
        self.data_collector = RealTimeDataCollector(ticker)
        self.data_buffer = DataBuffer(ticker, buffer_size=config.model.TIME_STEPS * 2)
        self.preprocessor = self.model_wrapper.preprocessor
        self.enable_online_learning = enable_online_learning
        self.is_ready = False

    def initialize(self) -> bool:
        try:
            # Try to load online model first
            online_loaded = self.model_wrapper.load_model(
                config.online_learning.ONLINE_MODEL_SUFFIX, self.enable_online_learning
            )

            if not online_loaded:
                logger.info(f"Online model for {self.ticker} not found, attempting to load latest model.")
                latest_loaded = self.model_wrapper.load_model("latest", self.enable_online_learning)
                if not latest_loaded:
                    logger.error(f"No pre-trained model or scaler found for {self.ticker}")
                    return False

            # Load initial historical data
            historical_data = self.data_collector.get_latest_data(period="1d", interval="1m")
            if historical_data.empty:
                logger.error(f"No historical data available for {self.ticker}")
                return False

            # Initialize data buffer with recent data
            recent_data = historical_data.tail(config.model.TIME_STEPS * 2)
            self.data_buffer.update_buffer(recent_data)

            self.is_ready = True
            logger.info(f"Predictor initialized for {self.ticker}. Online learning: {self.enable_online_learning}")
            return True

        except Exception as e:
            logger.error(f"Error initializing predictor for {self.ticker}: {e}")
            return False
    
    def update_data(self, new_data: pd.DataFrame) -> bool:
        """Update data buffer and trigger online learning if enabled"""
        try:
            if not new_data.empty:
                # Update data buffer
                self.data_buffer.update_buffer(new_data)
                
                # Trigger online learning if enabled
                if self.enable_online_learning and self.model_wrapper.is_online_ready:
                    success = self.model_wrapper.online_update(new_data)
                    if success:
                        logger.debug(f"Online learning triggered for {self.ticker}")
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating data for {self.ticker}: {e}")
            return False
    
    def predict(self) -> Optional[Dict[str, Any]]:
        """Make prediction using current data"""
        if not self.is_ready:
            logger.warning(f"Predictor not ready for {self.ticker}")
            return None
            
        try:
            recent_data = self.data_buffer.get_recent_data(config.model.TIME_STEPS)
            if len(recent_data) < config.model.TIME_STEPS:
                logger.warning(f"Insufficient data for prediction: {len(recent_data)} < {config.model.TIME_STEPS}")
                return None
            
            sequence = self.preprocessor.prepare_inference_data(recent_data)
            if sequence.size == 0:
                return None
            
            scaled_prediction = self.model_wrapper.model.predict(sequence, verbose=0)
            prediction = self.preprocessor.inverse_transform(scaled_prediction)[0][0]
            
            # Get current price for comparison
            current_price = self.data_collector.get_current_price()
            
            # Get online learning status
            online_status = self.model_wrapper.get_online_status()
            
            price_change_percent = (
                float((prediction - current_price) / current_price * 100)
                if current_price != 0
                else 0.0
            )
            
            return {
                'ticker': self.ticker,
                'timestamp': pd.Timestamp.now(),
                'current_price': current_price,
                'predicted_price': float(prediction),
                'price_difference': float(prediction - current_price),
                'price_change_percent': price_change_percent,
                'online_learning': {
                    'enabled': self.enable_online_learning,
                    'buffer_samples': online_status['buffer_info'].get('sample_count', 0),
                    'needs_update': online_status['buffer_info'].get('needs_update', False)
                }
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {self.ticker}: {e}")
            return None

class PredictionManager:
    """Manages predictions for multiple tickers"""
    
    def __init__(self, enable_online_learning: bool = True):
        self.predictors = {}
        self.enable_online_learning = enable_online_learning
        
    def initialize_predictors(self, tickers: List[str] = None) -> Dict[str, bool]:
        """Initialize predictors for multiple tickers"""
        if tickers is None:
            tickers = config.data.DEFAULT_TICKERS
            
        results = {}
        
        for ticker in tickers:
            predictor = StockPredictor(ticker, self.enable_online_learning)
            success = predictor.initialize()
            if success:
                self.predictors[ticker] = predictor
            results[ticker] = success
            
        return results
    
    def update_all_data(self, data_updates: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Update data for all predictors"""
        results = {}
        
        for ticker, data in data_updates.items():
            if ticker in self.predictors:
                results[ticker] = self.predictors[ticker].update_data(data)
            else:
                results[ticker] = False
                
        return results
    
    def predict_all(self) -> Dict[str, Any]:
        """Make predictions for all initialized predictors"""
        predictions = {}
        
        for ticker, predictor in self.predictors.items():
            prediction = predictor.predict()
            if prediction:
                predictions[ticker] = prediction
                
        return predictions
    
    def get_online_status(self) -> Dict[str, Any]:
        """Get online learning status for all predictors"""
        status = {}
        for ticker, predictor in self.predictors.items():
            if hasattr(predictor.model_wrapper, 'get_online_status'):
                status[ticker] = predictor.model_wrapper.get_online_status()
        return status