"""
Advanced Feature Engineering Module for Stock Prediction
Implements technical indicators and external data integration
"""

import pandas as pd
import numpy as np
import logging
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import ta  # Technical Analysis library
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Comprehensive technical indicators for stock price prediction.
    Includes trend, momentum, volatility, and volume indicators.
    """
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.DataFrame:
        """
        Relative Strength Index (RSI)
        Measures momentum, oscillates between 0-100
        RSI > 70: Overbought, RSI < 30: Oversold
        """
        df['RSI'] = ta.momentum.RSIIndicator(df[column], window=period).rsi()
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """
        Moving Average Convergence Divergence (MACD)
        Trend-following momentum indicator
        """
        macd = ta.trend.MACD(df[column])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2, column: str = 'Close') -> pd.DataFrame:
        """
        Bollinger Bands
        Volatility indicator with upper and lower bands
        """
        bollinger = ta.volatility.BollingerBands(df[column], window=period, window_dev=std)
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df[column] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50], column: str = 'Close') -> pd.DataFrame:
        """
        Simple and Exponential Moving Averages
        """
        for period in periods:
            df[f'SMA_{period}'] = ta.trend.SMAIndicator(df[column], window=period).sma_indicator()
            df[f'EMA_{period}'] = ta.trend.EMAIndicator(df[column], window=period).ema_indicator()
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator
        Momentum indicator comparing closing price to price range
        """
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 
                                               window=k_period, smooth_window=d_period)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average True Range (ATR)
        Volatility indicator
        """
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=period).average_true_range()
        return df
    
    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index (ADX)
        Trend strength indicator
        """
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=period)
        df['ADX'] = adx.adx()
        df['ADX_POS'] = adx.adx_pos()
        df['ADX_NEG'] = adx.adx_neg()
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based indicators
        """
        # On-Balance Volume
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
        # Volume Price Trend
        df['VPT'] = ta.volume.VolumePriceTrendIndicator(df['Close'], df['Volume']).volume_price_trend()
        
        # Chaikin Money Flow
        df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=20).chaikin_money_flow()
        
        return df

class SentimentAnalyzer:
    """
    Sentiment analysis integration for market sentiment data.
    In a real implementation, this would connect to news APIs, social media sentiment, etc.
    """
    
    def __init__(self):
        self.sentiment_cache = {}
    
    def get_mock_sentiment(self, ticker: str, date: pd.Timestamp) -> float:
        """
        Mock sentiment score for demonstration.
        In production, this would fetch real sentiment data from:
        - News APIs (Alpha Vantage, NewsAPI)
        - Social media (Twitter, Reddit sentiment)
        - Financial news sentiment analysis
        - Fear & Greed Index
        """
        # Generate mock sentiment based on price volatility patterns
        np.random.seed(int(date.timestamp()) % 1000)
        base_sentiment = np.random.normal(0, 0.3)  # Neutral with some variation
        
        # Adjust sentiment based on day of week (markets tend to be different on Mondays/Fridays)
        weekday_factor = {0: -0.1, 1: 0.05, 2: 0.1, 3: 0.05, 4: -0.05, 5: 0, 6: 0}
        sentiment = base_sentiment + weekday_factor.get(date.weekday(), 0)
        
        # Clip to [-1, 1] range
        return np.clip(sentiment, -1, 1)
    
    def add_sentiment_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add sentiment features to the dataset
        """
        sentiments = []
        for date in df.index:
            sentiment = self.get_mock_sentiment(ticker, date)
            sentiments.append(sentiment)
        
        df['Sentiment'] = sentiments
        
        # Add sentiment moving averages
        df['Sentiment_MA_3'] = df['Sentiment'].rolling(window=3).mean()
        df['Sentiment_MA_7'] = df['Sentiment'].rolling(window=7).mean()
        
        return df

class AdvancedFeatureEngineering:
    """
    Main feature engineering class that combines all indicators and external data
    """
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_columns = []
    
    def engineer_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering to stock data
        """
        logger.info(f"Starting feature engineering for {ticker}")
        
        # Make a copy to avoid modifying original data
        engineered_df = df.copy()
        
        # Basic price features
        engineered_df['Price_Change'] = engineered_df['Close'].pct_change()
        engineered_df['Price_Range'] = (engineered_df['High'] - engineered_df['Low']) / engineered_df['Close']
        engineered_df['Volume_Change'] = engineered_df['Volume'].pct_change()
        
        # Technical Indicators
        engineered_df = self.technical_indicators.add_rsi(engineered_df)
        engineered_df = self.technical_indicators.add_macd(engineered_df)
        engineered_df = self.technical_indicators.add_bollinger_bands(engineered_df)
        engineered_df = self.technical_indicators.add_moving_averages(engineered_df)
        engineered_df = self.technical_indicators.add_stochastic(engineered_df)
        engineered_df = self.technical_indicators.add_atr(engineered_df)
        engineered_df = self.technical_indicators.add_adx(engineered_df)
        engineered_df = self.technical_indicators.add_volume_indicators(engineered_df)
        
        # Sentiment Analysis
        engineered_df = self.sentiment_analyzer.add_sentiment_features(engineered_df, ticker)
        
        # Time-based features
        engineered_df['Hour'] = engineered_df.index.hour if hasattr(engineered_df.index, 'hour') else 0
        engineered_df['DayOfWeek'] = engineered_df.index.dayofweek if hasattr(engineered_df.index, 'dayofweek') else 0
        engineered_df['Month'] = engineered_df.index.month if hasattr(engineered_df.index, 'month') else 0
        
        # Cyclical encoding for time features
        engineered_df['Hour_sin'] = np.sin(2 * np.pi * engineered_df['Hour'] / 24)
        engineered_df['Hour_cos'] = np.cos(2 * np.pi * engineered_df['Hour'] / 24)
        engineered_df['DayOfWeek_sin'] = np.sin(2 * np.pi * engineered_df['DayOfWeek'] / 7)
        engineered_df['DayOfWeek_cos'] = np.cos(2 * np.pi * engineered_df['DayOfWeek'] / 7)
        engineered_df['Month_sin'] = np.sin(2 * np.pi * engineered_df['Month'] / 12)
        engineered_df['Month_cos'] = np.cos(2 * np.pi * engineered_df['Month'] / 12)
        
        # Interaction features
        engineered_df['RSI_MACD_Interaction'] = engineered_df['RSI'] * engineered_df['MACD']
        engineered_df['Volume_Price_Interaction'] = engineered_df['Volume_Change'] * engineered_df['Price_Change']
        engineered_df['Sentiment_RSI_Interaction'] = engineered_df['Sentiment'] * engineered_df['RSI']
        
        # Forward fill and backward fill NaN values
        engineered_df = engineered_df.fillna(method='ffill').fillna(method='bfill')
        
        # Store feature columns for later use
        self.feature_columns = [col for col in engineered_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        logger.info(f"Feature engineering completed. Added {len(self.feature_columns)} features")
        logger.info(f"Features: {self.feature_columns}")
        
        return engineered_df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of engineered feature columns
        """
        return self.feature_columns
    
    def get_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get only the features to be used in model training
        """
        # Core features for model training
        model_features = [
            'Close', 'Volume', 'Price_Change', 'Price_Range', 'Volume_Change',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'BB_Width',
            'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20',
            'Stoch_K', 'Stoch_D', 'ATR', 'ADX',
            'OBV', 'CMF', 'Sentiment', 'Sentiment_MA_3',
            'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
            'RSI_MACD_Interaction', 'Volume_Price_Interaction', 'Sentiment_RSI_Interaction'
        ]
        
        # Filter to only include columns that exist in the dataframe
        available_features = [col for col in model_features if col in df.columns]
        
        return df[available_features]

def test_feature_engineering():
    """
    Test function to demonstrate feature engineering capabilities
    """
    # Download sample data
    print("Testing feature engineering on AAPL...")
    ticker = "AAPL"
    data = yf.download(ticker, period="1y", interval="1d")
    
    # Apply feature engineering
    fe = AdvancedFeatureEngineering()
    engineered_data = fe.engineer_features(data, ticker)
    
    print(f"Original data shape: {data.shape}")
    print(f"Engineered data shape: {engineered_data.shape}")
    print(f"Added features: {len(fe.get_feature_columns())}")
    print("\nSample of engineered features:")
    print(engineered_data[fe.get_feature_columns()].head())
    
    # Get model features
    model_features = fe.get_model_features(engineered_data)
    print(f"\nModel features shape: {model_features.shape}")
    print("Model feature columns:", model_features.columns.tolist())
    
    return engineered_data, fe

if __name__ == "__main__":
    test_feature_engineering()