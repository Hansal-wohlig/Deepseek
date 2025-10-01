import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
from data.collectors import RealTimeDataCollector
from utils.preprocessor import DataPreprocessor
from config.settings import config

logger = logging.getLogger(__name__)

class OnlineLearningManager:
    def __init__(self):
        self.data_buffers: Dict[str, pd.DataFrame] = {}
        self.sample_counts: Dict[str, int] = {}
        self.last_update: Dict[str, datetime] = {}
        
    def initialize_ticker(self, ticker: str):
        """Initialize online learning for a ticker"""
        self.data_buffers[ticker] = pd.DataFrame()
        self.sample_counts[ticker] = 0
        self.last_update[ticker] = datetime.now()
        
    def add_data(self, ticker: str, new_data: pd.DataFrame) -> bool:
        """Add new data to buffer for online learning"""
        try:
            if ticker not in self.data_buffers:
                self.initialize_ticker(ticker)
            
            # Add new data to buffer
            if self.data_buffers[ticker].empty:
                self.data_buffers[ticker] = new_data
            else:
                self.data_buffers[ticker] = pd.concat([
                    self.data_buffers[ticker], new_data
                ]).tail(config.online_learning.UPDATE_FREQUENCY * 2)  # Keep limited history
            
            self.sample_counts[ticker] += len(new_data)
            
            logger.debug(f"Added {len(new_data)} samples to {ticker} buffer. Total: {self.sample_counts[ticker]}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding data to buffer for {ticker}: {e}")
            return False
    
    def should_update_model(self, ticker: str) -> bool:
        """Check if model should be updated"""
        if ticker not in self.sample_counts:
            return False
        
        # Check sample count threshold
        if self.sample_counts[ticker] >= config.online_learning.UPDATE_FREQUENCY:
            return True
        
        # Check time threshold (update at least once per day)
        time_since_update = datetime.now() - self.last_update.get(ticker, datetime.now())
        if time_since_update.days >= 1:
            return True
        
        return False
    
    def prepare_online_data(self, ticker: str, preprocessor: DataPreprocessor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare data for online training"""
        try:
            if ticker not in self.data_buffers or self.data_buffers[ticker].empty:
                return None, None
            
            buffer_data = self.data_buffers[ticker]
            
            if len(buffer_data) < config.online_learning.MIN_SAMPLES_FOR_UPDATE:
                logger.debug(f"Insufficient data for online training: {len(buffer_data)} < {config.online_learning.MIN_SAMPLES_FOR_UPDATE}")
                return None, None
            
            # Get recent historical data for context
            collector = RealTimeDataCollector(ticker)
            historical_data = collector.get_historical_data(
                start_date=(datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            )
            
            if historical_data.empty:
                logger.warning(f"No historical data available for {ticker}")
                return None, None
            
            # Prepare online training data
            X_online, y_online = preprocessor.prepare_online_data(
                historical_data, buffer_data
            )
            
            return X_online, y_online
            
        except Exception as e:
            logger.error(f"Error preparing online data for {ticker}: {e}")
            return None, None
    
    def reset_buffer(self, ticker: str):
        """Reset buffer after update"""
        if ticker in self.data_buffers:
            # Keep only the most recent data for sequence continuity
            self.data_buffers[ticker] = self.data_buffers[ticker].tail(config.model.TIME_STEPS)
            self.sample_counts[ticker] = 0
            self.last_update[ticker] = datetime.now()
            logger.debug(f"Reset buffer for {ticker}")
    
    def get_buffer_info(self, ticker: str) -> Dict:
        """Get information about data buffer"""
        if ticker in self.data_buffers:
            return {
                'sample_count': self.sample_counts[ticker],
                'buffer_size': len(self.data_buffers[ticker]),
                'last_update': self.last_update[ticker],
                'needs_update': self.should_update_model(ticker)
            }
        return {}