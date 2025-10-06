"""
Advanced LSTM Model with Bidirectional layers and Attention mechanism
Implements state-of-the-art architecture for financial time series forecasting
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, LayerNormalization,
    Attention, MultiHeadAttention, GlobalAveragePooling1D, Concatenate,
    BatchNormalization, Activation, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AttentionLayer(tf.keras.layers.Layer):
    """
    Custom Attention Layer for LSTM outputs
    Implements scaled dot-product attention mechanism
    """
    
    def __init__(self, units: int, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W_q = Dense(units)
        self.W_k = Dense(units)
        self.W_v = Dense(units)
        self.dense = Dense(1)
        
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        query = self.W_q(inputs)
        key = self.W_k(inputs)
        value = self.W_v(inputs)
        
        # Calculate attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention weights to values
        context = tf.matmul(attention_weights, value)
        
        # Global average pooling to get final representation
        output = tf.reduce_mean(context, axis=1)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

class AdvancedLSTMModel:
    """
    Advanced LSTM Model with Bidirectional layers and Attention mechanism
    
    Architecture Benefits:
    1. Bidirectional LSTM: Processes sequences in both forward and backward directions,
       capturing patterns that depend on future context
    2. Attention Mechanism: Allows model to focus on most relevant time steps,
       improving long-term dependency modeling
    3. Multi-layer architecture: Hierarchical feature learning
    4. Regularization: Dropout, L1/L2, BatchNorm for better generalization
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 n_features: int = 1,
                 lstm_units: list = [128, 64, 32],
                 attention_units: int = 64,
                 dropout_rate: float = 0.2,
                 l1_reg: float = 0.001,
                 l2_reg: float = 0.001):
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the advanced Bidirectional LSTM + Attention model
        
        Architecture:
        Input -> Bidirectional LSTM Stack -> Attention -> Dense Layers -> Output
        
        Why this architecture is superior:
        1. Bidirectional processing captures both past and future context
        2. Stacked layers learn hierarchical representations
        3. Attention mechanism focuses on relevant time periods
        4. Regularization prevents overfitting
        """
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input_layer')
        
        # First Bidirectional LSTM layer
        x = Bidirectional(
            LSTM(self.lstm_units[0], 
                 return_sequences=True, 
                 dropout=self.dropout_rate,
                 recurrent_dropout=self.dropout_rate,
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)),
            name='bidirectional_lstm_1'
        )(inputs)
        x = BatchNormalization(name='batch_norm_1')(x)
        
        # Second Bidirectional LSTM layer
        if len(self.lstm_units) > 1:
            x = Bidirectional(
                LSTM(self.lstm_units[1], 
                     return_sequences=True,
                     dropout=self.dropout_rate,
                     recurrent_dropout=self.dropout_rate,
                     kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)),
                name='bidirectional_lstm_2'
            )(x)
            x = BatchNormalization(name='batch_norm_2')(x)
        
        # Third Bidirectional LSTM layer (if specified)
        if len(self.lstm_units) > 2:
            x = Bidirectional(
                LSTM(self.lstm_units[2], 
                     return_sequences=True,
                     dropout=self.dropout_rate,
                     recurrent_dropout=self.dropout_rate,
                     kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)),
                name='bidirectional_lstm_3'
            )(x)
            x = BatchNormalization(name='batch_norm_3')(x)
        
        # Attention mechanism
        attention_output = AttentionLayer(self.attention_units, name='attention_layer')(x)
        
        # Also get global average pooling as alternative representation
        global_avg = GlobalAveragePooling1D(name='global_avg_pooling')(x)
        
        # Combine attention and global average pooling
        combined = Concatenate(name='combine_representations')([attention_output, global_avg])
        
        # Dense layers with residual connections
        dense1 = Dense(128, 
                      kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                      name='dense_1')(combined)
        dense1 = BatchNormalization(name='batch_norm_dense_1')(dense1)
        dense1 = Activation('relu', name='relu_1')(dense1)
        dense1 = Dropout(self.dropout_rate, name='dropout_1')(dense1)
        
        # Second dense layer
        dense2 = Dense(64,
                      kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                      name='dense_2')(dense1)
        dense2 = BatchNormalization(name='batch_norm_dense_2')(dense2)
        dense2 = Activation('relu', name='relu_2')(dense2)
        dense2 = Dropout(self.dropout_rate, name='dropout_2')(dense2)
        
        # Third dense layer
        dense3 = Dense(32,
                      kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                      name='dense_3')(dense2)
        dense3 = BatchNormalization(name='batch_norm_dense_3')(dense3)
        dense3 = Activation('relu', name='relu_3')(dense3)
        dense3 = Dropout(self.dropout_rate, name='dropout_3')(dense3)
        
        # Output layer
        outputs = Dense(1, activation='linear', name='output_layer')(dense3)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='AdvancedLSTMAttentionModel')
        
        # Compile model with advanced optimizer
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0  # Gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        # Print model architecture
        logger.info("Advanced LSTM + Attention Model Architecture:")
        model.summary(print_fn=logger.info)
        
        return model
    
    def get_callbacks(self, model_path: str, monitor: str = 'val_loss') -> list:
        """
        Get training callbacks for better training control
        """
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                model_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            )
        ]
        return callbacks
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray, 
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              model_path: str = None) -> Dict[str, Any]:
        """
        Train the advanced LSTM model
        """
        if self.model is None:
            self.build_model()
        
        # Update feature count based on input
        self.n_features = X_train.shape[2] if len(X_train.shape) == 3 else 1
        
        # Rebuild model if feature count changed
        if self.model.input_shape[2] != self.n_features:
            logger.info(f"Rebuilding model for {self.n_features} features")
            self.build_model()
        
        logger.info(f"Training model with input shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Get callbacks
        callbacks = []
        if model_path:
            callbacks = self.get_callbacks(model_path)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        # Evaluate on validation set
        val_loss, val_mae, val_mse = self.model.evaluate(X_val, y_val, verbose=0)
        val_rmse = np.sqrt(val_mse)
        
        # Make predictions for additional metrics
        y_pred = self.model.predict(X_val, verbose=0)
        
        results = {
            'val_loss': val_loss,
            'val_mae': val_mae,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'history': self.history.history,
            'model': self.model
        }
        
        logger.info(f"Training completed - Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}, Val RMSE: {val_rmse:.6f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        
        # Direction Accuracy (up/down prediction accuracy)
        y_test_diff = np.diff(y_test)
        y_pred_diff = np.diff(y_pred.flatten())
        direction_accuracy = np.mean((y_test_diff * y_pred_diff) > 0) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info("Model Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.6f}")
        
        return metrics
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a pre-trained model
        """
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info(f"Model loaded from {filepath}")

def compare_architectures():
    """
    Demonstration of why Bidirectional LSTM + Attention is superior
    """
    print("""
    Architecture Comparison for Financial Time Series:
    
    1. Basic LSTM:
       - Processes sequence in one direction only
       - May miss future context that affects current predictions
       - Limited ability to focus on relevant time periods
       
    2. Bidirectional LSTM:
       - Processes sequence in both directions
       - Captures patterns that depend on future context
       - Better for understanding market cycles and trends
       
    3. Bidirectional LSTM + Attention:
       - All benefits of bidirectional processing
       - Attention mechanism highlights important time periods
       - Can focus on specific market events or pattern breakouts
       - Superior performance on long sequences
       - Better interpretability through attention weights
    
    Key Benefits for Stock Prediction:
    - Market events often have delayed effects (forward context matters)
    - Attention helps identify key support/resistance levels
    - Better handling of market volatility and trend changes
    - Improved performance on longer prediction horizons
    """)

if __name__ == "__main__":
    # Create sample data for testing
    sequence_length = 60
    n_features = 25  # Multiple features from feature engineering
    n_samples = 1000
    
    # Generate sample data
    X_sample = np.random.randn(n_samples, sequence_length, n_features)
    y_sample = np.random.randn(n_samples, 1)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_val = X_sample[:train_size], X_sample[train_size:]
    y_train, y_val = y_sample[:train_size], y_sample[train_size:]
    
    print("Testing Advanced LSTM + Attention Model...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Create and train model
    model = AdvancedLSTMModel(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=[128, 64, 32],
        attention_units=64
    )
    
    # Build model to see architecture
    model.build_model()
    
    compare_architectures()