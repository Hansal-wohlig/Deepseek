"""
Advanced Evaluation Metrics and Prediction Functions
Implements comprehensive evaluation for financial time series forecasting
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FinancialMetrics:
    """
    Comprehensive financial metrics for stock prediction evaluation
    Goes beyond standard ML metrics to include finance-specific measures
    """
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE)
        Measures average absolute difference between predictions and actual values
        Lower is better. Same scale as target variable.
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error (RMSE)
        Penalizes larger errors more heavily than MAE
        Lower is better. Same scale as target variable.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error (MAPE)
        Scale-independent metric expressed as percentage
        Lower is better. 0-100% scale.
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error (sMAPE)
        More stable version of MAPE, bounded between 0-200%
        """
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Directional Accuracy (DA)
        Percentage of correct direction predictions (up/down)
        Critical for trading applications - predicting direction matters more than exact price
        """
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction) * 100
    
    @staticmethod
    def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.02) -> float:
        """
        Hit Rate - percentage of predictions within threshold
        Measures how often predictions are 'close enough' to actual values
        """
        relative_errors = np.abs((y_true - y_pred) / y_true)
        return np.mean(relative_errors <= threshold) * 100
    
    @staticmethod
    def maximum_drawdown_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Maximum Drawdown of prediction errors
        Measures worst-case prediction performance
        """
        errors = y_pred - y_true
        cumulative_errors = np.cumsum(errors)
        running_max = np.maximum.accumulate(cumulative_errors)
        drawdowns = cumulative_errors - running_max
        return np.min(drawdowns)
    
    @staticmethod
    def profit_accuracy(y_true: np.ndarray, y_pred: np.ndarray, transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Trading-based accuracy metrics
        Simulates simple trading strategy based on predictions
        """
        if len(y_true) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0, 'win_rate': 0}
        
        # Calculate returns
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # Simple strategy: go long if predicted return > 0, short if < 0
        positions = np.sign(pred_returns)
        strategy_returns = positions * true_returns - np.abs(np.diff(positions)) * transaction_cost
        
        # Calculate metrics
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        win_rate = np.mean(strategy_returns > 0) * 100
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_return': np.mean(strategy_returns) * 100
        }
    
    @staticmethod
    def prediction_interval_coverage(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_lower: np.ndarray, y_pred_upper: np.ndarray,
                                   confidence_level: float = 0.95) -> float:
        """
        Coverage of prediction intervals
        Measures how often true values fall within predicted confidence intervals
        """
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
        return coverage * 100

class ModelEvaluator:
    """
    Comprehensive model evaluation with financial metrics and visualizations
    """
    
    def __init__(self, include_trading_metrics: bool = True):
        self.include_trading_metrics = include_trading_metrics
        self.metrics = FinancialMetrics()
        
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           timestamps: pd.DatetimeIndex = None,
                           ticker: str = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions
        
        Returns dictionary with all relevant metrics for stock prediction
        """
        # Ensure arrays are 1D
        y_true = y_true.flatten() if y_true.ndim > 1 else y_true
        y_pred = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        # Basic regression metrics
        metrics_dict = {
            'mae': self.metrics.mean_absolute_error(y_true, y_pred),
            'rmse': self.metrics.root_mean_squared_error(y_true, y_pred),
            'mape': self.metrics.mean_absolute_percentage_error(y_true, y_pred),
            'smape': self.metrics.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
        }
        
        # Financial-specific metrics
        metrics_dict.update({
            'directional_accuracy': self.metrics.directional_accuracy(y_true, y_pred),
            'hit_rate_2pct': self.metrics.hit_rate(y_true, y_pred, 0.02),
            'hit_rate_5pct': self.metrics.hit_rate(y_true, y_pred, 0.05),
            'max_drawdown_error': self.metrics.maximum_drawdown_error(y_true, y_pred),
        })
        
        # Trading metrics (if enabled)
        if self.include_trading_metrics:
            trading_metrics = self.metrics.profit_accuracy(y_true, y_pred)
            metrics_dict.update({f'trading_{k}': v for k, v in trading_metrics.items()})
        
        # Statistical metrics
        residuals = y_pred - y_true
        metrics_dict.update({
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'skewness_residual': stats.skew(residuals),
            'kurtosis_residual': stats.kurtosis(residuals),
        })
        
        # Correlation metrics
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics_dict['correlation'] = correlation
        
        # Add metadata
        metrics_dict.update({
            'n_samples': len(y_true),
            'evaluation_timestamp': datetime.now(),
            'ticker': ticker
        })
        
        return metrics_dict
    
    def create_evaluation_report(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               timestamps: pd.DatetimeIndex = None,
                               ticker: str = "Stock",
                               save_path: str = None) -> str:
        """
        Create comprehensive evaluation report
        """
        metrics = self.evaluate_predictions(y_true, y_pred, timestamps, ticker)
        
        report = f"""
Stock Prediction Model Evaluation Report
{'='*50}
Ticker: {ticker}
Evaluation Date: {metrics['evaluation_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Sample Size: {metrics['n_samples']} predictions

REGRESSION METRICS
{'='*30}
Mean Absolute Error (MAE): {metrics['mae']:.4f}
Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}
Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%
Symmetric MAPE (sMAPE): {metrics['smape']:.2f}%
R-squared Score: {metrics['r2_score']:.4f}
Correlation: {metrics['correlation']:.4f}

FINANCIAL METRICS
{'='*30}
Directional Accuracy: {metrics['directional_accuracy']:.2f}%
Hit Rate (2%): {metrics['hit_rate_2pct']:.2f}%
Hit Rate (5%): {metrics['hit_rate_5pct']:.2f}%
Max Drawdown Error: {metrics['max_drawdown_error']:.4f}

STATISTICAL PROPERTIES
{'='*30}
Mean Residual: {metrics['mean_residual']:.4f}
Std Residual: {metrics['std_residual']:.4f}
Residual Skewness: {metrics['skewness_residual']:.4f}
Residual Kurtosis: {metrics['kurtosis_residual']:.4f}
"""
        
        if self.include_trading_metrics:
            report += f"""
TRADING SIMULATION
{'='*30}
Total Return: {metrics['trading_total_return']:.2f}%
Sharpe Ratio: {metrics['trading_sharpe_ratio']:.4f}
Win Rate: {metrics['trading_win_rate']:.2f}%
Average Return: {metrics['trading_avg_return']:.4f}%
"""
        
        report += f"""
INTERPRETATION
{'='*30}
• MAE/RMSE: Lower values indicate better accuracy
• MAPE: {self._interpret_mape(metrics['mape'])}
• Directional Accuracy: {self._interpret_directional_accuracy(metrics['directional_accuracy'])}
• R-squared: {self._interpret_r2(metrics['r2_score'])}
• Hit Rate: {self._interpret_hit_rate(metrics['hit_rate_2pct'])}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def _interpret_mape(self, mape: float) -> str:
        if mape < 5:
            return "Excellent accuracy"
        elif mape < 10:
            return "Good accuracy"
        elif mape < 20:
            return "Reasonable accuracy"
        else:
            return "Poor accuracy - needs improvement"
    
    def _interpret_directional_accuracy(self, da: float) -> str:
        if da > 60:
            return "Excellent directional prediction"
        elif da > 55:
            return "Good directional prediction"
        elif da > 50:
            return "Better than random"
        else:
            return "Poor directional prediction"
    
    def _interpret_r2(self, r2: float) -> str:
        if r2 > 0.8:
            return "Strong explanatory power"
        elif r2 > 0.6:
            return "Moderate explanatory power"
        elif r2 > 0.3:
            return "Weak explanatory power"
        else:
            return "Very weak explanatory power"
    
    def _interpret_hit_rate(self, hit_rate: float) -> str:
        if hit_rate > 80:
            return "Very accurate predictions"
        elif hit_rate > 60:
            return "Good accuracy"
        elif hit_rate > 40:
            return "Moderate accuracy"
        else:
            return "Low accuracy"
    
    def plot_prediction_analysis(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               timestamps: pd.DatetimeIndex = None,
                               ticker: str = "Stock",
                               save_path: str = None) -> plt.Figure:
        """
        Create comprehensive visualization of prediction performance
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{ticker} - Model Prediction Analysis', fontsize=16)
        
        # Flatten arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=len(y_true), freq='D')
        
        # 1. Time series plot
        axes[0, 0].plot(timestamps, y_true, label='Actual', alpha=0.8, linewidth=1)
        axes[0, 0].plot(timestamps, y_pred, label='Predicted', alpha=0.8, linewidth=1)
        axes[0, 0].set_title('Actual vs Predicted Prices')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.6, s=20)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_title('Actual vs Predicted (Scatter)')
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add R² to scatter plot
        r2 = r2_score(y_true, y_pred)
        axes[0, 1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Residuals plot
        residuals = y_pred - y_true
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_title('Residuals vs Predicted')
        axes[1, 0].set_xlabel('Predicted Price')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residual Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction analysis plot saved to {save_path}")
        
        return fig

class AdvancedPredictor:
    """
    Advanced prediction class with uncertainty quantification and multiple prediction types
    """
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        
    def predict_single(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Single point prediction with confidence information
        """
        try:
            # Prepare data
            X = self.preprocessor.prepare_inference_data(recent_data)
            
            # Make prediction
            scaled_prediction = self.model.predict(X)
            prediction = self.preprocessor.inverse_transform_target(scaled_prediction)[0][0]
            
            # Get current price for comparison
            current_price = recent_data['Close'].iloc[-1]
            
            # Calculate change metrics
            price_change = prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            return {
                'predicted_price': float(prediction),
                'current_price': float(current_price),
                'price_change': float(price_change),
                'price_change_percent': float(price_change_pct),
                'prediction_timestamp': datetime.now(),
                'confidence_score': self._calculate_confidence_score(recent_data)
            }
            
        except Exception as e:
            logger.error(f"Error in single prediction: {e}")
            return None
    
    def predict_multi_step(self, recent_data: pd.DataFrame, steps: int = 5) -> Dict[str, Any]:
        """
        Multi-step ahead predictions
        """
        predictions = []
        current_data = recent_data.copy()
        
        for step in range(steps):
            # Make single prediction
            pred_result = self.predict_single(current_data)
            
            if pred_result is None:
                break
                
            predictions.append({
                'step': step + 1,
                'predicted_price': pred_result['predicted_price'],
                'confidence_score': pred_result['confidence_score']
            })
            
            # Update data with prediction for next step
            new_row = current_data.iloc[-1].copy()
            new_row['Close'] = pred_result['predicted_price']
            
            # Append to data (maintaining sequence length)
            current_data = pd.concat([current_data.iloc[1:], new_row.to_frame().T])
        
        return {
            'multi_step_predictions': predictions,
            'prediction_horizon': len(predictions),
            'base_price': float(recent_data['Close'].iloc[-1])
        }
    
    def predict_with_intervals(self, recent_data: pd.DataFrame, n_samples: int = 100) -> Dict[str, Any]:
        """
        Prediction with uncertainty intervals using Monte Carlo dropout
        """
        try:
            X = self.preprocessor.prepare_inference_data(recent_data)
            
            # Multiple predictions with dropout (if model supports it)
            predictions = []
            for _ in range(n_samples):
                scaled_pred = self.model.predict(X)
                pred = self.preprocessor.inverse_transform_target(scaled_pred)[0][0]
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate statistics
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Prediction intervals
            lower_80 = np.percentile(predictions, 10)
            upper_80 = np.percentile(predictions, 90)
            lower_95 = np.percentile(predictions, 2.5)
            upper_95 = np.percentile(predictions, 97.5)
            
            return {
                'predicted_price': float(mean_pred),
                'prediction_std': float(std_pred),
                'confidence_intervals': {
                    '80%': {'lower': float(lower_80), 'upper': float(upper_80)},
                    '95%': {'lower': float(lower_95), 'upper': float(upper_95)}
                },
                'prediction_samples': predictions.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in interval prediction: {e}")
            return self.predict_single(recent_data)
    
    def _calculate_confidence_score(self, recent_data: pd.DataFrame) -> float:
        """
        Calculate confidence score based on data quality and market conditions
        """
        try:
            # Data quality factors
            data_completeness = 1.0 - (recent_data.isnull().sum().sum() / recent_data.size)
            
            # Volatility factor (higher volatility = lower confidence)
            returns = recent_data['Close'].pct_change().dropna()
            volatility = returns.std()
            volatility_factor = max(0.1, 1.0 - min(volatility * 100, 0.8))
            
            # Trend consistency factor
            trend_consistency = abs(returns.mean()) / (volatility + 1e-8)
            trend_factor = min(1.0, trend_consistency)
            
            # Combine factors
            confidence = (data_completeness * 0.3 + volatility_factor * 0.4 + trend_factor * 0.3)
            
            return float(np.clip(confidence, 0.1, 1.0))
            
        except Exception:
            return 0.5  # Default confidence

def demonstrate_evaluation_metrics():
    """
    Demonstrate the evaluation metrics with sample data
    """
    print("""
    Financial Prediction Evaluation Metrics Explained:
    
    1. MEAN ABSOLUTE ERROR (MAE):
       - Average absolute difference between predicted and actual prices
       - Same scale as stock price (e.g., $5.23 MAE means average error is $5.23)
       - Easy to interpret, robust to outliers
    
    2. ROOT MEAN SQUARED ERROR (RMSE):
       - Square root of average squared errors
       - Penalizes large errors more heavily than MAE
       - Useful for detecting models that make occasional large mistakes
    
    3. DIRECTIONAL ACCURACY:
       - Percentage of correct up/down predictions
       - Critical for trading - often more important than exact price
       - >50% is better than random, >60% is very good
    
    4. HIT RATE:
       - Percentage of predictions within X% of actual price
       - Practical metric for trading applications
       - Shows how often predictions are "close enough"
    
    5. TRADING METRICS:
       - Simulate actual trading based on predictions
       - Include transaction costs and realistic constraints
       - Provide real-world performance assessment
    
    Why These Matter More Than Standard ML Metrics:
    - Financial markets are noisy and non-stationary
    - Direction often matters more than exact values
    - Transaction costs and practical constraints affect profitability
    - Risk-adjusted returns are crucial for investment decisions
    """)

if __name__ == "__main__":
    demonstrate_evaluation_metrics()
    
    # Example usage
    print("\nTesting Evaluation Metrics...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    y_true = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)  # Random walk prices
    y_pred = y_true + np.random.randn(n_samples) * 2  # Predictions with noise
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Calculate metrics
    metrics = evaluator.evaluate_predictions(y_true, y_pred, ticker="TEST")
    
    print("Sample Evaluation Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    print(f"Hit Rate (2%): {metrics['hit_rate_2pct']:.2f}%")
    
    if 'trading_total_return' in metrics:
        print(f"Trading Return: {metrics['trading_total_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['trading_sharpe_ratio']:.4f}")