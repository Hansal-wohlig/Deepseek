import json
import time
import logging
from datetime import datetime
from confluent_kafka import Producer
import pandas as pd
from data.collectors import RealTimeDataCollector
from config.settings import config

logger = logging.getLogger(__name__)

class StockDataProducer:
    def __init__(self):
        self.producer = Producer({
            'bootstrap.servers': ','.join(config.kafka.BOOTSTRAP_SERVERS),
            'client.id': 'stock-data-producer'
        })
        self.data_collectors = {
            ticker: RealTimeDataCollector(ticker) 
            for ticker in config.data.DEFAULT_TICKERS
        }
        
    def delivery_report(self, err, msg):
        """Called once for each message produced to indicate delivery result"""
        if err is not None:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')
    
    def produce_stock_data(self):
        """Collect and produce current stock data"""
        for ticker, collector in self.data_collectors.items():
            try:
                # Get latest data
                latest_data = collector.get_latest_data(period='1d', interval='1m')
                
                if not latest_data.empty:
                    # Get the latest record
                    latest_record = latest_data.iloc[-1]
                    current_price = collector.get_current_price()
                    
                    # Prepare message
                    message = {
                        'ticker': ticker,
                        'timestamp': datetime.now().isoformat(),
                        'open': float(latest_record['Open']),
                        'high': float(latest_record['High']),
                        'low': float(latest_record['Low']),
                        'close': float(latest_record['Close']),
                        'volume': int(latest_record['Volume']),
                        'current_price': current_price,
                        'data_type': 'market_data'
                    }
                    
                    # Produce to Kafka
                    self.producer.produce(
                        config.kafka.STOCK_PRICE_TOPIC,
                        key=ticker,
                        value=json.dumps(message),
                        callback=self.delivery_report
                    )
                    
                    logger.debug(f"Produced data for {ticker}: ${current_price}")
                    
            except Exception as e:
                logger.error(f"Error producing data for {ticker}: {e}")
        
        # Wait for any outstanding messages to be delivered
        self.producer.flush()
    
    def run(self):
        """Main producer loop"""
        logger.info("Starting Stock Data Producer")
        
        while True:
            try:
                self.produce_stock_data()
                time.sleep(config.data.DATA_COLLECTION_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Producer interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in producer loop: {e}")
                time.sleep(10)  # Wait before retrying

if __name__ == "__main__":
    producer = StockDataProducer()
    producer.run()