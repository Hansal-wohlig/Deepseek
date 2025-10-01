import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional
import logging
from config.settings import config

logger = logging.getLogger(__name__)

class PredictionVisualizer:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.figures_dir = os.path.join(config.paths.LOGS_DIR, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def plot_training_history(self, history, save_path: Optional[str] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{self.ticker} - Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot MAE
        ax2.plot(history['mae'], label='Training MAE')
        ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title(f'{self.ticker} - Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.figures_dir, f"{self.ticker}_training_history.png")
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Training history plot saved to {save_path}")
    
    def plot_predictions(self, 
                        actual_prices: np.ndarray, 
                        predicted_prices: np.ndarray,
                        dates: pd.DatetimeIndex,
                        save_path: Optional[str] = None):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(dates, actual_prices, label='Actual Prices', color='blue', alpha=0.7)
        plt.plot(dates, predicted_prices, label='Predicted Prices', color='red', alpha=0.7)
        
        plt.title(f'{self.ticker} - Actual vs Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.figures_dir, f"{self.ticker}_predictions.png")
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Prediction plot saved to {save_path}")