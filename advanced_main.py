"""
Advanced Stock Prediction System - Main Entry Point
Integrates all advanced components for improved accuracy and performance
"""

import argparse
import logging
import sys
import os
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# Add paths for advanced modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'advanced_features'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'advanced_models'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'advanced_training'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'advanced_evaluation'))

# Import advanced modules
from feature_engineering import AdvancedFeatureEngineering
from attention_lstm import AdvancedLSTMModel
from time_series_validation import ModelTrainingStrategy, AdvancedDataPreprocessor
from online_learning_manager import HybridRetrainingManager
from metrics_and_prediction import ModelEvaluator, AdvancedPredictor

# Import original modules
from config.settings import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{config.paths.LOGS_DIR}/advanced_stock_prediction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class AdvancedStockPredictionSystem:
    """
    Main class integrating all advanced components
    """
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineering()
        self.model_evaluator = ModelEvaluator()
        self.training_strategy = ModelTrainingStrategy(AdvancedLSTMModel)
        self.online_learning_manager = HybridRetrainingManager()
        
        self.models = {}
        self.preprocessors = {}
        self.predictors = {}
        
        logger.info("Advanced Stock Prediction System initialized")
    
    def fetch_and_prepare_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Fetch and prepare data with advanced feature engineering
        """
        logger.info(f"Fetching data for {ticker}...")
        
        try:
            # Download stock data
            stock_data = yf.download(ticker, period=period, interval="1d")
            
            if stock_data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            logger.info(f"Downloaded {len(stock_data)} days of data for {ticker}")
            
            # Apply advanced feature engineering
            logger.info(f"Applying feature engineering for {ticker}...")
            engineered_data = self.feature_engineer.engineer_features(stock_data, ticker)
            
            # Get model features
            model_data = self.feature_engineer.get_model_features(engineered_data)
            
            # Clean data
            model_data = model_data.dropna()
            
            logger.info(f"Prepared {len(model_data)} samples with {len(model_data.columns)} features")
            logger.info(f"Feature columns: {list(model_data.columns)}")
            
            return model_data
            
        except Exception as e:
            logger.error(f"Error preparing data for {ticker}: {e}")
            raise
    
    def train_advanced_model(self, ticker: str, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Train advanced model with proper validation
        """
        logger.info(f"Starting advanced training for {ticker}...")
        
        try:
            # Fetch data if not provided
            if data is None:
                data = self.fetch_and_prepare_data(ticker)
            
            # Define model parameters
            model_params = {
                'sequence_length': 60,
                'n_features': len(data.columns),
                'lstm_units': [128, 64, 32],
                'attention_units': 64,
                'dropout_rate': 0.2,
                'l1_reg': 0.001,
                'l2_reg': 0.001
            }
            
            # Get feature columns (excluding target)
            feature_columns = [col for col in data.columns if col != 'Close']
            
            # Train with cross-validation
            cv_results = self.training_strategy.train_with_validation(
                data=data,
                model_params=model_params,
                feature_columns=feature_columns
            )
            
            logger.info(f"Cross-validation completed for {ticker}")
            logger.info(f"Mean CV Score: {cv_results['mean_score']:.6f} ± {cv_results['std_score']:.6f}")
            
            # Final training
            final_results = self.training_strategy.final_training(
                data=data,
                model_params=model_params,
                feature_columns=feature_columns
            )
            
            # Store model and preprocessor
            self.models[ticker] = final_results['model']
            self.preprocessors[ticker] = final_results['preprocessor']
            
            # Create advanced predictor
            self.predictors[ticker] = AdvancedPredictor(
                final_results['model'], 
                final_results['preprocessor']
            )
            
            # Save model
            model_path = config.get_model_path(ticker, "advanced")
            final_results['model'].save_model(model_path)
            
            logger.info(f"Advanced model training completed for {ticker}")
            logger.info(f"Final validation MAE: {final_results['evaluation_results']['mae']:.6f}")
            logger.info(f"Final validation RMSE: {final_results['evaluation_results']['rmse']:.6f}")
            logger.info(f"Directional Accuracy: {final_results['evaluation_results']['directional_accuracy']:.2f}%")
            
            return {
                'ticker': ticker,
                'cv_results': cv_results,
                'final_results': final_results,
                'model_path': model_path,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            return {
                'ticker': ticker,
                'status': 'failed',
                'error': str(e)
            }
    
    def make_predictions(self, ticker: str, prediction_type: str = 'single') -> Dict[str, Any]:
        """
        Make predictions using trained model
        """
        if ticker not in self.predictors:
            logger.error(f"No trained model found for {ticker}")
            return None
        
        try:
            # Get recent data
            recent_data = self.fetch_and_prepare_data(ticker, period="3mo")
            recent_data = recent_data.tail(70)  # Get last 70 days for sequence
            
            predictor = self.predictors[ticker]
            
            if prediction_type == 'single':
                result = predictor.predict_single(recent_data)
            elif prediction_type == 'multi_step':
                result = predictor.predict_multi_step(recent_data, steps=5)
            elif prediction_type == 'intervals':
                result = predictor.predict_with_intervals(recent_data)
            else:
                raise ValueError(f"Unknown prediction type: {prediction_type}")
            
            if result:
                result['ticker'] = ticker
                result['prediction_type'] = prediction_type
                logger.info(f"Prediction made for {ticker}: {prediction_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction for {ticker}: {e}")
            return None
    
    def evaluate_model(self, ticker: str, test_period: str = "3mo") -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        """
        if ticker not in self.models:
            logger.error(f"No trained model found for {ticker}")
            return None
        
        try:
            # Get test data
            test_data = self.fetch_and_prepare_data(ticker, period=test_period)
            
            # Prepare test data using existing preprocessor
            preprocessor = self.preprocessors[ticker]
            scaled_features, scaled_target = preprocessor.transform_data(test_data)
            X_test, y_test = preprocessor.create_sequences(scaled_features, scaled_target)
            
            # Make predictions
            model = self.models[ticker]
            y_pred_scaled = model.predict(X_test)
            y_pred = preprocessor.inverse_transform_target(y_pred_scaled).flatten()
            y_true = preprocessor.inverse_transform_target(y_test).flatten()
            
            # Evaluate predictions
            metrics = self.model_evaluator.evaluate_predictions(y_true, y_pred, ticker=ticker)
            
            # Generate report
            report = self.model_evaluator.create_evaluation_report(
                y_true, y_pred, ticker=ticker,
                save_path=f"{config.paths.LOGS_DIR}/{ticker}_evaluation_report.txt"
            )
            
            # Create visualization
            fig = self.model_evaluator.plot_prediction_analysis(
                y_true, y_pred, ticker=ticker,
                save_path=f"{config.paths.LOGS_DIR}/{ticker}_prediction_analysis.png"
            )
            
            logger.info(f"Model evaluation completed for {ticker}")
            
            return {
                'ticker': ticker,
                'metrics': metrics,
                'report': report,
                'n_test_samples': len(y_test),
                'evaluation_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model for {ticker}: {e}")
            return None
    
    def start_online_learning(self, ticker: str):
        """
        Start online learning for a ticker
        """
        if ticker not in self.models:
            logger.error(f"No trained model found for {ticker}. Train the model first.")
            return False
        
        # Start background processing
        self.online_learning_manager.start_background_processing()
        logger.info(f"Online learning started for {ticker}")
        return True
    
    def stop_online_learning(self):
        """
        Stop online learning
        """
        self.online_learning_manager.stop_background_processing()
        logger.info("Online learning stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status
        """
        return {
            'trained_models': list(self.models.keys()),
            'available_predictors': list(self.predictors.keys()),
            'online_learning_status': self.online_learning_manager.get_status(),
            'system_timestamp': datetime.now()
        }

def main():
    """
    Main entry point for advanced stock prediction system
    """
    parser = argparse.ArgumentParser(description='Advanced Stock Prediction System')
    parser.add_argument('--mode', 
                       choices=['train', 'predict', 'evaluate', 'online', 'demo'], 
                       default='demo',
                       help='Operation mode')
    parser.add_argument('--tickers', nargs='+', 
                       default=['AAPL'],
                       help='Stock tickers to process')
    parser.add_argument('--prediction-type', 
                       choices=['single', 'multi_step', 'intervals'],
                       default='single',
                       help='Type of prediction to make')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AdvancedStockPredictionSystem()
    
    try:
        if args.mode == 'demo':
            # Run comprehensive demo
            logger.info("Starting comprehensive demo...")
            
            ticker = args.tickers[0]
            
            # 1. Train model
            logger.info(f"1. Training advanced model for {ticker}...")
            training_result = system.train_advanced_model(ticker)
            
            if training_result['status'] == 'success':
                logger.info(f"✓ Training successful for {ticker}")
                
                # 2. Make predictions
                logger.info(f"2. Making predictions for {ticker}...")
                prediction = system.make_predictions(ticker, 'single')
                if prediction:
                    logger.info(f"✓ Prediction: ${prediction['predicted_price']:.2f}")
                    logger.info(f"   Current: ${prediction['current_price']:.2f}")
                    logger.info(f"   Change: {prediction['price_change_percent']:.2f}%")
                
                # 3. Evaluate model
                logger.info(f"3. Evaluating model for {ticker}...")
                evaluation = system.evaluate_model(ticker)
                if evaluation:
                    logger.info(f"✓ Evaluation completed")
                    logger.info(f"   MAE: {evaluation['metrics']['mae']:.4f}")
                    logger.info(f"   RMSE: {evaluation['metrics']['rmse']:.4f}")
                    logger.info(f"   Directional Accuracy: {evaluation['metrics']['directional_accuracy']:.2f}%")
                
                # 4. Multi-step predictions
                logger.info(f"4. Multi-step predictions for {ticker}...")
                multi_pred = system.make_predictions(ticker, 'multi_step')
                if multi_pred:
                    logger.info(f"✓ 5-day predictions generated")
                    for pred in multi_pred['multi_step_predictions'][:3]:
                        logger.info(f"   Day {pred['step']}: ${pred['predicted_price']:.2f}")
                
                logger.info("Demo completed successfully!")
            else:
                logger.error(f"Training failed: {training_result['error']}")
        
        elif args.mode == 'train':
            # Train models
            results = {}
            for ticker in args.tickers:
                result = system.train_advanced_model(ticker)
                results[ticker] = result
            
            # Summary
            successful = [t for t, r in results.items() if r['status'] == 'success']
            failed = [t for t, r in results.items() if r['status'] == 'failed']
            
            logger.info(f"Training completed: {len(successful)} successful, {len(failed)} failed")
            
        elif args.mode == 'predict':
            # Make predictions
            for ticker in args.tickers:
                prediction = system.make_predictions(ticker, args.prediction_type)
                if prediction:
                    print(f"\n{ticker} Prediction:")
                    if args.prediction_type == 'single':
                        print(f"  Predicted Price: ${prediction['predicted_price']:.2f}")
                        print(f"  Current Price: ${prediction['current_price']:.2f}")
                        print(f"  Price Change: {prediction['price_change_percent']:.2f}%")
                    elif args.prediction_type == 'multi_step':
                        print(f"  Multi-step predictions:")
                        for pred in prediction['multi_step_predictions']:
                            print(f"    Day {pred['step']}: ${pred['predicted_price']:.2f}")
                else:
                    logger.error(f"Failed to make prediction for {ticker}")
        
        elif args.mode == 'evaluate':
            # Evaluate models
            for ticker in args.tickers:
                evaluation = system.evaluate_model(ticker)
                if evaluation:
                    print(f"\n{ticker} Evaluation:")
                    print(f"  MAE: {evaluation['metrics']['mae']:.4f}")
                    print(f"  RMSE: {evaluation['metrics']['rmse']:.4f}")
                    print(f"  MAPE: {evaluation['metrics']['mape']:.2f}%")
                    print(f"  Directional Accuracy: {evaluation['metrics']['directional_accuracy']:.2f}%")
                else:
                    logger.error(f"Failed to evaluate model for {ticker}")
        
        elif args.mode == 'online':
            # Start online learning
            for ticker in args.tickers:
                system.start_online_learning(ticker)
            
            logger.info("Online learning started. Press Ctrl+C to stop.")
            
            try:
                while True:
                    status = system.get_system_status()
                    logger.info(f"System status: {len(status['trained_models'])} models running")
                    import time
                    time.sleep(60)  # Status update every minute
            except KeyboardInterrupt:
                system.stop_online_learning()
                logger.info("Online learning stopped")
    
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        system.stop_online_learning()
    except Exception as e:
        logger.error(f"Application error: {e}")
        system.stop_online_learning()
        sys.exit(1)

if __name__ == "__main__":
    main()