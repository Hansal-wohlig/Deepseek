import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    TIME_STEPS: int = 60
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    TRAIN_TEST_SPLIT: float = 0.8
    LSTM_UNITS: List[int] = field(default_factory=lambda: [100, 50, 50, 50])

@dataclass
class DataConfig:
    START_DATE: str = "2017-01-01"
    DEFAULT_TICKERS: List[str] = field(default_factory=lambda: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])
    DATA_COLLECTION_INTERVAL: int = 60  # seconds
    FEATURES: List[str] = field(default_factory=lambda: ['Open', 'High', 'Low', 'Close', 'Volume'])

@dataclass
class KafkaConfig:
    BOOTSTRAP_SERVERS: List[str] = field(default_factory=lambda: ['localhost:9092'])
    STOCK_PRICE_TOPIC: str = "stock_prices"
    PREDICTION_TOPIC: str = "stock-predictions"
    GROUP_ID: str = "stock-prediction-group"

@dataclass
class OnlineLearningConfig:
    ENABLED: bool = True
    BATCH_SIZE: int = 32
    UPDATE_FREQUENCY: int = 100  # Update every 100 new samples
    LEARNING_RATE: float = 0.0001  # Lower LR for fine-tuning
    EPOCHS_PER_UPDATE: int = 3
    MIN_SAMPLES_FOR_UPDATE: int = 50
    
    # Model versioning for online learning
    ONLINE_MODEL_SUFFIX: str = "online"
    BACKUP_MODELS: bool = True

@dataclass
class PathConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR: str = os.path.join(BASE_DIR, "saved_models")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
    
    def __post_init__(self):
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.kafka = KafkaConfig()
        self.paths = PathConfig()
        self.online_learning = OnlineLearningConfig()
        self.current_date = datetime.now().strftime("%Y%m%d")
    
    def get_model_path(self, ticker: str, version: str = "latest") -> str:
        model_dir = os.path.join(self.paths.MODELS_DIR, ticker, version)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, "model.h5")
    
    def get_online_model_path(self, ticker: str) -> str:
        return self.get_model_path(ticker, self.online_learning.ONLINE_MODEL_SUFFIX)
    
    def get_scaler_path(self, ticker: str, version: str = "latest") -> str:
        model_dir = os.path.join(self.paths.MODELS_DIR, ticker, version)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, "scaler.pkl")

# Global configuration instance
config = Config()