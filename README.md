# Stock Price Prediction System

A real-time stock price prediction system that uses deep learning models to forecast stock prices and supports online learning for continuous model improvement.

## Features

- **Real-time Data Collection**: Fetches stock market data in real-time using yfinance
- **Deep Learning Models**: Utilizes LSTM neural networks for time series forecasting
- **Online Learning**: Supports continuous model improvement with new market data
- **Distributed Architecture**: Uses Kafka for scalable data streaming between components
- **Multiple Asset Support**: Configure multiple stock tickers for prediction
- **Model Versioning**: Maintains different versions of trained models (latest, best, online)
- **Technical Analysis**: Integrates technical indicators for improved predictions
- **Sentiment Analysis**: Optional integration with NLP models for market sentiment analysis

## Project Structure

```
.
├── config/               # Configuration settings
├── data/                 # Data collection and processing
├── logs/                 # Application logs
├── models/               # Model training and prediction logic
├── saved_models/         # Trained model checkpoints
├── streaming/            # Kafka producers and consumers
├── utils/                # Utility functions and helpers
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Prerequisites

- Python 3.8+
- Kafka (for real-time streaming)
- TensorFlow 2.10.0+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Deepseek
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Kafka:
   - Download and install Kafka from [https://kafka.apache.org/](https://kafka.apache.org/)
   - Start Zookeeper and Kafka servers
   - Create required topics (stock_prices, stock-predictions)

## Usage

### Training Models

Train models for specific stock tickers:
```bash
python main.py --mode train --tickers AAPL MSFT GOOGL
```

### Start Data Producer

Start collecting and publishing stock data:
```bash
python main.py --mode produce --tickers AAPL MSFT GOOGL
```

### Start Prediction Consumer

Start making predictions with online learning:
```bash
python main.py --mode consume --tickers AAPL MSFT GOOGL
```

### Run All Components

Run the complete system (training, producing, and consuming):
```bash
python main.py --tickers AAPL MSFT GOOGL
```

## Configuration

Modify `config/settings.py` to customize:
- Model architecture and training parameters
- Kafka broker settings
- Data collection intervals
- Feature engineering options
- Online learning behavior

## Model Architecture

The system uses a multi-layer LSTM architecture with the following default configuration:
- Input sequence length: 60 time steps
- LSTM layers: [100, 50, 50, 50] units
- Dropout layers for regularization
- Dense output layer with linear activation

## Data Pipeline

1. **Data Collection**: Fetches historical and real-time market data
2. **Preprocessing**: Handles missing values, normalization, feature engineering
3. **Windowing**: Creates sequential input-output pairs for time series prediction
4. **Training**: Trains LSTM models on historical data
5. **Prediction**: Makes real-time predictions on streaming data
6. **Online Learning**: Continuously updates models with new data

## Monitoring

- Logs are stored in the `logs/` directory
- Model performance metrics are logged during training and evaluation
- Kafka consumer groups allow for monitoring of message processing

## License

[Specify your license here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Support

For issues and feature requests, please use the [issue tracker](https://github.com/yourusername/Deepseek/issues).