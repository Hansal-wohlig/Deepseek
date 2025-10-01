import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, Any
from config.settings import config

logger = logging.getLogger(__name__)

class RealTimeDataCollector:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)
        
    def get_company_info(self) -> Dict[str, Any]:
        """Get company information"""
        try:
            return self.yf_ticker.info
        except Exception as e:
            logger.error(f"Error fetching company info for {self.ticker}: {e}")
            return {}
    
    def get_historical_data(self, 
                          start_date: str = None, 
                          end_date: str = None) -> pd.DataFrame:
        """Get historical stock data"""
        if start_date is None:
            start_date = config.data.START_DATE
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            data = yf.download(self.ticker, start=start_date, end=end_date)
            if data.empty:
                logger.warning(f"No data found for {self.ticker}")
                return pd.DataFrame()

            # Clean data: forward-fill missing values and drop any remaining NaNs
            data.ffill(inplace=True)
            data.dropna(inplace=True)

            if data.empty:
                logger.warning(f"Data for {self.ticker} is empty after cleaning.")
                return pd.DataFrame()
                
            return data[config.data.FEATURES]
        except Exception as e:
            logger.error(f"Error downloading data for {self.ticker}: {e}")
            return pd.DataFrame()
    
    def get_latest_data(self, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """Get latest intraday data"""
        try:
            data = self.yf_ticker.history(period=period, interval=interval)
            if data.empty:
                return pd.DataFrame()

            # Clean data
            data.ffill(inplace=True)
            data.dropna(inplace=True)

            if not data.empty:
                return data[config.data.FEATURES]
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching latest data for {self.ticker}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self) -> float:
        """Get current stock price"""
        try:
            data = self.yf_ticker.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching current price for {self.ticker}: {e}")
            return 0.0

class DataBuffer:
    """Buffer to maintain recent data for prediction"""
    def __init__(self, ticker: str, buffer_size: int = 100):
        self.ticker = ticker
        self.buffer_size = buffer_size
        self.data_buffer = pd.DataFrame()
        
    def update_buffer(self, new_data: pd.DataFrame):
        """Update data buffer with new data"""
        if self.data_buffer.empty:
            self.data_buffer = new_data
        else:
            self.data_buffer = pd.concat([self.data_buffer, new_data]).tail(self.buffer_size)
    
    def get_recent_data(self, n_points: int) -> pd.DataFrame:
        """Get recent n points from buffer"""
        if len(self.data_buffer) >= n_points:
            return self.data_buffer.tail(n_points)
        return self.data_buffer