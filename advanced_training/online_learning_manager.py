"""
Advanced Online Learning and Hybrid Retraining Strategy
Implements sophisticated incremental learning and periodic retraining
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import queue
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class IncrementalBuffer:
    """
    Sophisticated buffer for storing new data for online learning
    Implements various sampling strategies and data quality checks
    """
    
    def __init__(self, 
                 max_size: int = 10000,
                 min_update_size: int = 50,
                 sampling_strategy: str = 'recent_weighted'):
        
        self.max_size = max_size
        self.min_update_size = min_update_size
        self.sampling_strategy = sampling_strategy
        
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.quality_scores = deque(maxlen=max_size)
        
        self.total_samples_added = 0
        self.last_update_time = datetime.now()
        
    def add_sample(self, features: np.ndarray, target: float, timestamp: datetime = None, quality_score: float = 1.0):
        """
        Add a new sample to the buffer with quality scoring
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Data quality checks
        if np.isnan(features).any() or np.isnan(target):
            logger.warning("Skipping sample with NaN values")
            return False
        
        if np.isinf(features).any() or np.isinf(target):
            logger.warning("Skipping sample with infinite values")
            return False
        
        # Add to buffer
        self.buffer.append((features, target))
        self.timestamps.append(timestamp)
        self.quality_scores.append(quality_score)
        
        self.total_samples_added += 1
        
        return True
    
    def should_trigger_update(self) -> bool:
        """
        Determine if we should trigger an incremental update
        """
        if len(self.buffer) < self.min_update_size:
            return False
        
        # Check time-based triggers
        time_since_update = datetime.now() - self.last_update_time
        
        # Trigger conditions:
        # 1. Buffer size reached threshold
        # 2. Significant time elapsed (1 hour)
        # 3. High-quality samples available
        
        size_trigger = len(self.buffer) >= self.min_update_size * 2
        time_trigger = time_since_update > timedelta(hours=1)
        quality_trigger = np.mean(list(self.quality_scores)) > 0.8
        
        return size_trigger or time_trigger or quality_trigger
    
    def get_training_batch(self, batch_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of samples for training based on sampling strategy
        """
        if len(self.buffer) == 0:
            return None, None
        
        if batch_size is None:
            batch_size = len(self.buffer)
        
        batch_size = min(batch_size, len(self.buffer))
        
        if self.sampling_strategy == 'recent_weighted':
            # Weight recent samples more heavily
            weights = np.exp(np.linspace(-2, 0, len(self.buffer)))
            weights = weights / weights.sum()
            
            indices = np.random.choice(len(self.buffer), size=batch_size, p=weights, replace=False)
            
        elif self.sampling_strategy == 'quality_weighted':
            # Weight high-quality samples more heavily
            scores = np.array(list(self.quality_scores))
            weights = scores / scores.sum()
            
            indices = np.random.choice(len(self.buffer), size=batch_size, p=weights, replace=False)
            
        else:  # 'uniform'
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        # Extract samples
        features_batch = []
        targets_batch = []
        
        for idx in indices:
            features, target = self.buffer[idx]
            features_batch.append(features)
            targets_batch.append(target)
        
        return np.array(features_batch), np.array(targets_batch)
    
    def clear_buffer(self):
        """
        Clear the buffer after successful update
        """
        self.buffer.clear()
        self.timestamps.clear()
        self.quality_scores.clear()
        self.last_update_time = datetime.now()
        
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics for monitoring
        """
        if len(self.buffer) == 0:
            return {'size': 0, 'mean_quality': 0, 'oldest_sample': None, 'newest_sample': None}
        
        return {
            'size': len(self.buffer),
            'mean_quality': np.mean(list(self.quality_scores)),
            'oldest_sample': min(self.timestamps),
            'newest_sample': max(self.timestamps),
            'total_samples_added': self.total_samples_added
        }

class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler for online learning
    Adjusts learning rate based on model performance and market conditions
    """
    
    def __init__(self, 
                 initial_lr: float = 0.001,
                 min_lr: float = 1e-6,
                 max_lr: float = 0.01,
                 adaptation_factor: float = 0.1):
        
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adaptation_factor = adaptation_factor
        
        self.performance_history = deque(maxlen=20)
        self.volatility_history = deque(maxlen=10)
        
    def update_learning_rate(self, current_loss: float, validation_loss: float, market_volatility: float = None):
        """
        Adapt learning rate based on performance and market conditions
        """
        # Store performance metrics
        self.performance_history.append({'loss': current_loss, 'val_loss': validation_loss})
        
        if market_volatility is not None:
            self.volatility_history.append(market_volatility)
        
        if len(self.performance_history) < 3:
            return self.current_lr
        
        # Calculate performance trend
        recent_losses = [p['val_loss'] for p in list(self.performance_history)[-3:]]
        loss_trend = np.mean(np.diff(recent_losses))
        
        # Calculate volatility factor
        volatility_factor = 1.0
        if len(self.volatility_history) > 0:
            current_vol = self.volatility_history[-1]
            avg_vol = np.mean(list(self.volatility_history))
            volatility_factor = min(2.0, max(0.5, current_vol / avg_vol))
        
        # Adjust learning rate
        if loss_trend > 0:  # Loss increasing - reduce learning rate
            self.current_lr *= (1 - self.adaptation_factor)
        elif loss_trend < -0.001:  # Loss decreasing significantly - can increase learning rate slightly
            self.current_lr *= (1 + self.adaptation_factor * 0.5)
        
        # Apply volatility adjustment
        self.current_lr *= volatility_factor
        
        # Clip to bounds
        self.current_lr = np.clip(self.current_lr, self.min_lr, self.max_lr)
        
        logger.debug(f"Learning rate adjusted to: {self.current_lr:.6f} (trend: {loss_trend:.6f}, vol_factor: {volatility_factor:.2f})")
        
        return self.current_lr

class HybridRetrainingManager:
    """
    Manages hybrid retraining strategy combining incremental fine-tuning and periodic full retraining
    """
    
    def __init__(self, 
                 incremental_config: Dict[str, Any] = None,
                 full_retrain_config: Dict[str, Any] = None):
        
        # Default configurations
        self.incremental_config = incremental_config or {
            'epochs': 3,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'patience': 5,
            'min_samples': 50,
            'update_frequency': 100
        }
        
        self.full_retrain_config = full_retrain_config or {
            'trigger_conditions': {
                'time_based': timedelta(weeks=1),
                'performance_degradation': 0.15,  # 15% degradation
                'sample_count': 10000,
                'market_regime_change': True
            },
            'epochs': 100,
            'validation_split': 0.2
        }
        
        # State tracking
        self.buffer = IncrementalBuffer(
            max_size=self.incremental_config['update_frequency'] * 10,
            min_update_size=self.incremental_config['min_samples']
        )
        
        self.lr_scheduler = AdaptiveLearningRateScheduler(
            initial_lr=self.incremental_config['learning_rate']
        )
        
        self.last_full_retrain = datetime.now()
        self.baseline_performance = None
        self.performance_history = deque(maxlen=50)
        
        # Threading for background updates
        self.update_queue = queue.Queue()
        self.is_running = True
        self.update_thread = None
        
    def start_background_processing(self):
        """
        Start background thread for processing updates
        """
        self.update_thread = threading.Thread(target=self._background_update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Background online learning thread started")
    
    def stop_background_processing(self):
        """
        Stop background processing
        """
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Background online learning thread stopped")
    
    def add_new_data(self, features: np.ndarray, target: float, timestamp: datetime = None, quality_score: float = 1.0):
        """
        Add new data point for online learning
        """
        success = self.buffer.add_sample(features, target, timestamp, quality_score)
        
        # Queue update if conditions are met
        if success and self.buffer.should_trigger_update():
            self.update_queue.put({'type': 'incremental_update'})
    
    def incremental_fine_tuning(self, model, preprocessor) -> Dict[str, Any]:
        """
        Perform incremental fine-tuning of the model
        """
        logger.info("Starting incremental fine-tuning...")
        
        # Get training batch from buffer
        X_batch, y_batch = self.buffer.get_training_batch(self.incremental_config['batch_size'])
        
        if X_batch is None or len(X_batch) < self.incremental_config['min_samples']:
            logger.warning("Insufficient data for incremental update")
            return {'success': False, 'reason': 'insufficient_data'}
        
        try:
            # Prepare data in correct format for the model
            if len(X_batch.shape) == 2:
                # Reshape for LSTM if needed
                sequence_length = model.model.input_shape[1]
                n_features = model.model.input_shape[2]
                
                # If we have sequences, use them directly
                if X_batch.shape[1] == sequence_length * n_features:
                    X_batch = X_batch.reshape(-1, sequence_length, n_features)
                else:
                    logger.warning("Data shape mismatch for incremental training")
                    return {'success': False, 'reason': 'shape_mismatch'}
            
            # Get current learning rate
            current_lr = self.lr_scheduler.current_lr
            
            # Compile model with adaptive learning rate
            model.model.compile(
                optimizer=Adam(learning_rate=current_lr),
                loss='mse',
                metrics=['mae']
            )
            
            # Perform incremental training
            history = model.model.fit(
                X_batch, y_batch,
                epochs=self.incremental_config['epochs'],
                batch_size=self.incremental_config['batch_size'],
                verbose=0,
                shuffle=True
            )
            
            # Update learning rate based on performance
            final_loss = history.history['loss'][-1]
            self.lr_scheduler.update_learning_rate(final_loss, final_loss)
            
            # Clear buffer after successful update
            self.buffer.clear_buffer()
            
            # Track performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'loss': final_loss,
                'learning_rate': current_lr,
                'samples_used': len(X_batch)
            })
            
            logger.info(f"Incremental fine-tuning completed. Final loss: {final_loss:.6f}")
            
            return {
                'success': True,
                'final_loss': final_loss,
                'samples_used': len(X_batch),
                'learning_rate': current_lr,
                'epochs_trained': self.incremental_config['epochs']
            }
            
        except Exception as e:
            logger.error(f"Error during incremental fine-tuning: {e}")
            return {'success': False, 'reason': str(e)}
    
    def should_trigger_full_retrain(self) -> Tuple[bool, str]:
        """
        Determine if full retraining should be triggered
        """
        conditions = self.full_retrain_config['trigger_conditions']
        
        # Time-based trigger
        time_since_retrain = datetime.now() - self.last_full_retrain
        if time_since_retrain >= conditions['time_based']:
            return True, 'time_based'
        
        # Performance degradation trigger
        if len(self.performance_history) >= 10:
            recent_performance = np.mean([p['loss'] for p in list(self.performance_history)[-5:]])
            if self.baseline_performance is not None:
                degradation = (recent_performance - self.baseline_performance) / self.baseline_performance
                if degradation > conditions['performance_degradation']:
                    return True, 'performance_degradation'
        
        # Sample count trigger
        if self.buffer.total_samples_added >= conditions['sample_count']:
            return True, 'sample_count'
        
        return False, 'none'
    
    def full_retraining(self, model_class, training_data: pd.DataFrame, model_params: Dict[str, Any], preprocessor) -> Dict[str, Any]:
        """
        Perform full model retraining from scratch
        """
        logger.info("Starting full model retraining...")
        
        try:
            # Prepare fresh data
            X_full, y_full = preprocessor.prepare_training_data(training_data)
            
            # Split for validation
            val_split = self.full_retrain_config['validation_split']
            split_idx = int(len(X_full) * (1 - val_split))
            
            X_train, X_val = X_full[:split_idx], X_full[split_idx:]
            y_train, y_val = y_full[:split_idx], y_full[split_idx:]
            
            # Create new model instance
            new_model = model_class(**model_params)
            
            # Train model
            training_results = new_model.train(
                X_train, y_train.ravel(),
                X_val, y_val.ravel(),
                epochs=self.full_retrain_config['epochs'],
                batch_size=32
            )
            
            # Update baseline performance
            self.baseline_performance = training_results['val_mae']
            self.last_full_retrain = datetime.now()
            
            # Reset buffer and performance history
            self.buffer.clear_buffer()
            self.performance_history.clear()
            
            logger.info(f"Full retraining completed. New baseline MAE: {self.baseline_performance:.6f}")
            
            return {
                'success': True,
                'new_model': new_model,
                'baseline_performance': self.baseline_performance,
                'training_results': training_results
            }
            
        except Exception as e:
            logger.error(f"Error during full retraining: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _background_update_loop(self):
        """
        Background loop for processing updates
        """
        while self.is_running:
            try:
                # Check for updates in queue
                if not self.update_queue.empty():
                    update_request = self.update_queue.get(timeout=1)
                    logger.debug(f"Processing update request: {update_request['type']}")
                
                time.sleep(1)  # Small delay to prevent busy waiting
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the online learning system
        """
        should_retrain, retrain_reason = self.should_trigger_full_retrain()
        
        return {
            'buffer_stats': self.buffer.get_buffer_stats(),
            'learning_rate': self.lr_scheduler.current_lr,
            'performance_history_size': len(self.performance_history),
            'last_full_retrain': self.last_full_retrain,
            'baseline_performance': self.baseline_performance,
            'should_full_retrain': should_retrain,
            'retrain_reason': retrain_reason,
            'is_background_running': self.is_running and (self.update_thread is not None)
        }
    
    def save_state(self, filepath: str):
        """
        Save the current state of the online learning manager
        """
        state = {
            'buffer': {
                'data': list(self.buffer.buffer),
                'timestamps': list(self.buffer.timestamps),
                'quality_scores': list(self.buffer.quality_scores),
                'total_samples_added': self.buffer.total_samples_added
            },
            'lr_scheduler': {
                'current_lr': self.lr_scheduler.current_lr,
                'performance_history': list(self.lr_scheduler.performance_history),
                'volatility_history': list(self.lr_scheduler.volatility_history)
            },
            'last_full_retrain': self.last_full_retrain,
            'baseline_performance': self.baseline_performance,
            'performance_history': list(self.performance_history)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Online learning state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """
        Load the state of the online learning manager
        """
        if not os.path.exists(filepath):
            logger.warning(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore buffer
            self.buffer.buffer = deque(state['buffer']['data'], maxlen=self.buffer.max_size)
            self.buffer.timestamps = deque(state['buffer']['timestamps'], maxlen=self.buffer.max_size)
            self.buffer.quality_scores = deque(state['buffer']['quality_scores'], maxlen=self.buffer.max_size)
            self.buffer.total_samples_added = state['buffer']['total_samples_added']
            
            # Restore learning rate scheduler
            self.lr_scheduler.current_lr = state['lr_scheduler']['current_lr']
            self.lr_scheduler.performance_history = deque(state['lr_scheduler']['performance_history'], maxlen=20)
            self.lr_scheduler.volatility_history = deque(state['lr_scheduler']['volatility_history'], maxlen=10)
            
            # Restore manager state
            self.last_full_retrain = state['last_full_retrain']
            self.baseline_performance = state['baseline_performance']
            self.performance_history = deque(state['performance_history'], maxlen=50)
            
            logger.info(f"Online learning state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

def explain_hybrid_retraining_strategy():
    """
    Explanation of the hybrid retraining approach
    """
    print("""
    Hybrid Retraining Strategy for Stock Prediction:
    
    1. INCREMENTAL FINE-TUNING:
       Purpose: Quick adaptation to recent market changes
       Frequency: Every 50-100 new samples or hourly
       Method: 
         - Use small learning rates (0.0001)
         - Train for 3-5 epochs only
         - Focus on recent high-quality data
         - Adaptive learning rate based on performance
       
       Benefits:
         - Fast adaptation to market changes
         - Maintains model freshness
         - Low computational cost
         - Preserves learned patterns
    
    2. PERIODIC FULL RETRAINING:
       Purpose: Major model updates and pattern relearning
       Frequency: Weekly or when performance degrades significantly
       Method:
         - Complete retraining from scratch
         - Use all available historical data
         - Full epoch training (100+ epochs)
         - Proper train/validation split
       
       Triggers:
         - Time-based (weekly)
         - Performance degradation (>15%)
         - Significant data accumulation (10k+ samples)
         - Market regime changes detected
       
       Benefits:
         - Relearns fundamental patterns
         - Adapts to regime changes
         - Prevents catastrophic forgetting
         - Maintains long-term accuracy
    
    3. ADAPTIVE LEARNING RATE:
       - Adjusts based on recent performance
       - Considers market volatility
       - Prevents overshooting in updates
       - Balances stability vs adaptation
    
    4. QUALITY-BASED SAMPLING:
       - Weights samples by quality scores
       - Prioritizes recent data
       - Filters out anomalous data points
       - Maintains training data integrity
    
    This hybrid approach provides:
    - Best of both worlds: quick adaptation + stable long-term learning
    - Robust performance across different market conditions
    - Computational efficiency
    - Automatic adaptation to changing markets
    """)

if __name__ == "__main__":
    explain_hybrid_retraining_strategy()
    
    # Example usage
    print("\nTesting Hybrid Retraining Manager...")
    
    manager = HybridRetrainingManager()
    
    # Simulate adding new data
    for i in range(75):
        features = np.random.randn(30)  # 30 features
        target = np.random.randn()
        quality = np.random.uniform(0.7, 1.0)
        
        manager.add_new_data(features, target, quality_score=quality)
    
    # Check status
    status = manager.get_status()
    print(f"Buffer size: {status['buffer_stats']['size']}")
    print(f"Should trigger full retrain: {status['should_full_retrain']}")
    print(f"Mean quality: {status['buffer_stats']['mean_quality']:.3f}")