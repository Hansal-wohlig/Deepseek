import json
import logging
from confluent_kafka import Consumer, KafkaError
from datetime import datetime
import pandas as pd
from models.predictor import PredictionManager
from config.settings import config

logger = logging.getLogger(__name__)

class PredictionConsumer:
    def __init__(self, enable_online_learning: bool = True):
        self.consumer = Consumer({
            'bootstrap.servers': ','.join(config.kafka.BOOTSTRAP_SERVERS),
            'group.id': config.kafka.GROUP_ID,
            'auto.offset.reset': 'latest'
        })
        
        self.prediction_manager = PredictionManager(enable_online_learning)
        self.enable_online_learning = enable_online_learning
        self.initialized = False
        
    def initialize_predictors(self, tickers: list = None):
        """Initialize prediction models for specified tickers"""
        if tickers is None:
            tickers = config.data.DEFAULT_TICKERS
            
        logger.info("Initializing prediction models...")
        results = self.prediction_manager.initialize_predictors(tickers)
        
        successful = [ticker for ticker, success in results.items() if success]
        failed = [ticker for ticker, success in results.items() if not success]
        
        if successful:
            logger.info(f"Successfully initialized predictors for: {', '.join(successful)}")
            if self.enable_online_learning:
                logger.info("Online learning is ENABLED")
            else:
                logger.info("Online learning is DISABLED")
        if failed:
            logger.warning(f"Failed to initialize predictors for: {', '.join(failed)}")
            
        self.initialized = len(successful) > 0
        return self.initialized
    
    def process_market_data(self, message: dict):
        """Process incoming market data and make predictions"""
        try:
            ticker = message['ticker']
            
            # Convert message to DataFrame format
            data_dict = {
                'Open': [message['open']],
                'High': [message['high']],
                'Low': [message['low']],
                'Close': [message['close']],
                'Volume': [message['volume']]
            }
            
            data_df = pd.DataFrame(data_dict, index=[pd.Timestamp(message['timestamp'])])
            
            # Update predictor data
            update_result = self.prediction_manager.update_all_data({ticker: data_df})
            
            if update_result.get(ticker, False):
                # Make prediction
                predictions = self.prediction_manager.predict_all()
                
                if ticker in predictions:
                    prediction = predictions[ticker]
                    
                    # Log prediction
                    logger.info(
                        f"{ticker}: Current=${prediction['current_price']:.2f}, "
                        f"Predicted=${prediction['predicted_price']:.2f}, "
                        f"Change={prediction['price_change_percent']:+.2f}%"
                    )
                    
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def run(self):
        """Main consumer loop"""
        if not self.initialized:
            logger.error("Predictors not initialized. Exiting.")
            return
        
        logger.info("Starting Prediction Consumer")
        
        # Subscribe to topics
        self.consumer.subscribe([config.kafka.STOCK_PRICE_TOPIC])
        
        try:
            while True:
                msg = self.consumer.poll(1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        break
                
                try:
                    # Parse message
                    message_data = json.loads(msg.value().decode('utf-8'))
                    
                    # Process based on message type
                    if message_data.get('data_type') == 'market_data':
                        self.process_market_data(message_data)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            self.consumer.close()

if __name__ == "__main__":
    # Enable online learning by default
    consumer = PredictionConsumer(enable_online_learning=True)
    consumer.initialize_predictors()
    consumer.run()