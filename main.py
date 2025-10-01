import argparse
import logging
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import config
from models.trainer import ModelTrainer
from streaming.producer import StockDataProducer
from streaming.consumer import PredictionConsumer
from data.collectors import RealTimeDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{config.paths.LOGS_DIR}/stock_prediction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup required environment"""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
    
    logger.info("Stock Prediction System Initialized")
    logger.info(f"Models directory: {config.paths.MODELS_DIR}")
    logger.info(f"Data directory: {config.paths.DATA_DIR}")
    logger.info(f"Logs directory: {config.paths.LOGS_DIR}")

def train_models(tickers: list):
    """Train models for specified tickers"""
    logger.info("Starting model training...")
    
    trainer = ModelTrainer()
    results = trainer.train_all_models(tickers)
    
    # Log results
    successful = [ticker for ticker, result in results.items() if result['status'] == 'success']
    failed = [ticker for ticker, result in results.items() if result['status'] == 'failed']
    
    logger.info(f"Training completed: {len(successful)} successful, {len(failed)} failed")
    
    for ticker in successful:
        result = results[ticker]
        logger.info(f"{ticker}: Loss={result['test_loss']:.4f}, MAE={result['test_mae']:.4f}")
    
    for ticker in failed:
        logger.error(f"{ticker}: {results[ticker]['error']}")
    
    return results

def start_producer(tickers: list):
    """Start Kafka producer"""
    logger.info("Starting stock data producer...")
    producer = StockDataProducer()
    
    # Update data collectors with specified tickers
    producer.data_collectors = {
        ticker: RealTimeDataCollector(ticker) 
        for ticker in tickers
    }
    
    producer.run()

def start_consumer(tickers: list, enable_online_learning: bool = True):
    """Start Kafka consumer with prediction"""
    logger.info("Starting prediction consumer...")
    consumer = PredictionConsumer(enable_online_learning=enable_online_learning)
    
    # Initialize with specified tickers
    consumer.initialize_predictors(tickers)
    consumer.run()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Stock Prediction System')
    parser.add_argument('--mode', choices=['train', 'produce', 'consume', 'all'], 
                       default='all', help='Operation mode')
    parser.add_argument('--tickers', nargs='+', 
                       default=config.data.DEFAULT_TICKERS,
                       help='Stock tickers to process')
    parser.add_argument('--no-online-learning', action='store_true',
                       help='Disable online learning (enabled by default)')
    
    args = parser.parse_args()
    
    setup_environment()
    
    # Online learning is enabled by default, can be disabled with flag
    enable_online_learning = not args.no_online_learning
    
    try:
        if args.mode in ['train', 'all']:
            train_models(args.tickers)
        
        if args.mode in ['produce', 'all']:
            start_producer(args.tickers)
        
        if args.mode in ['consume', 'all']:
            start_consumer(args.tickers, enable_online_learning)
            
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()