# Stock Price Prediction System

A real-time stock price prediction system that leverages deep learning and real-time data processing to forecast stock prices with high accuracy. The system uses a Bi-LSTM with Attention mechanism for predictions and integrates with Kafka for real-time data streaming.

## Features

- **Real-time Data Collection**: Fetches live stock data using Yahoo Finance API
- **Technical Analysis**: Implements various technical indicators for feature engineering
- **Sentiment Analysis**: Analyzes news sentiment to incorporate market sentiment into predictions
- **Deep Learning Model**: Utilizes Bi-LSTM with Attention mechanism for time-series forecasting
- **Online Learning**: Supports continuous model improvement with new data
- **Kafka Integration**: Real-time data streaming for production deployment
- **Multi-ticker Support**: Train and predict on multiple stock tickers simultaneously

## Prerequisites

- Python 3.8+
- TensorFlow 2.10.0+
- Kafka (for real-time streaming)
- News API key (for sentiment analysis)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Deepseek
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   export NEWS_API_KEY='your_news_api_key_here'
   export KAFKA_BOOTSTRAP_SERVERS='localhost:9092'  # If using Kafka locally
   ```

## Project Structure

```
Deepseek/
├── config/               # Configuration settings
├── data/                 # Data collection and preprocessing
├── models/               # Model architecture and training
├── saved_models/         # Trained model checkpoints
├── streaming/            # Kafka producer and consumer
├── utils/                # Utility functions
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Usage

### 1. Training Models

To train models for specific stock tickers:

```bash
python main.py --mode train --tickers AAPL MSFT GOOGL
```

### 2. Real-time Prediction

Start the Kafka producer to collect real-time data:

```bash
python main.py --mode producer --tickers AAPL
```

In a separate terminal, start the prediction consumer:

```bash
python main.py --mode consumer --tickers AAPL --enable_online_learning
```

### 3. Online Learning

Enable online learning to continuously improve the model with new data:

```bash
python main.py --mode train --tickers AAPL --online_learning
```

## Configuration

Modify `config/settings.py` to adjust:
- Model hyperparameters
- Data collection settings
- File paths
- Kafka configuration
- Training parameters

## Model Architecture

The system uses a Bi-LSTM with Attention mechanism for time-series forecasting:

- **Input Layer**: Historical price data and technical indicators
- **Bi-LSTM Layers**: Capture temporal dependencies in both directions
- **Attention Layer**: Focus on important time steps
- **Dense Layers**: Process the learned features
- **Output Layer**: Predicts the next price movement

## Data Pipeline

1. **Data Collection**: Fetches historical and real-time data from Yahoo Finance
2. **Preprocessing**: Handles missing values, normalizes features
3. **Feature Engineering**: Adds technical indicators and sentiment scores
4. **Windowing**: Creates sequences for time-series prediction
5. **Model Training**: Trains the Bi-LSTM model
6. **Prediction**: Generates price forecasts

## Monitoring

Check the logs in `logs/stock_prediction.log` for training progress and system status.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yahoo Finance for market data
- TensorFlow for deep learning framework
- Apache Kafka for real-time data streaming
- Various Python libraries for data processing and analysis
