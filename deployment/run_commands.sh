#!/bin/bash

# Advanced Stock Prediction System - Deployment Commands
# Complete guide for running the enhanced system

set -e  # Exit on any error

echo "=================================================="
echo "Advanced Stock Prediction System - Command Guide"
echo "=================================================="

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv_advanced" ]; then
        print_error "Virtual environment not found!"
        print_step "Please run setup first: python deployment/setup_advanced.py"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    print_step "Activating virtual environment..."
    
    # Check OS type
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source venv_advanced/Scripts/activate
    else
        # Linux/Mac
        source venv_advanced/bin/activate
    fi
    
    print_success "Virtual environment activated"
}

# System setup and verification
setup_system() {
    print_step "Setting up Advanced Stock Prediction System..."
    
    # Run setup script
    python deployment/setup_advanced.py
    
    if [ $? -eq 0 ]; then
        print_success "System setup completed successfully!"
    else
        print_error "System setup failed!"
        exit 1
    fi
}

# Quick demo - recommended first run
run_demo() {
    print_step "Running comprehensive demo..."
    
    check_venv
    activate_venv
    
    echo "This will:"
    echo "1. Download AAPL data and engineer features"
    echo "2. Train Bidirectional LSTM + Attention model"  
    echo "3. Evaluate with financial metrics"
    echo "4. Make various types of predictions"
    echo ""
    
    python advanced_main.py --mode demo --tickers AAPL
    
    if [ $? -eq 0 ]; then
        print_success "Demo completed successfully!"
        print_step "Check logs/ directory for detailed results"
    else
        print_error "Demo failed!"
    fi
}

# Training functions
train_single_ticker() {
    local ticker=${1:-AAPL}
    
    print_step "Training advanced model for $ticker..."
    
    check_venv
    activate_venv
    
    python advanced_main.py --mode train --tickers $ticker
    
    if [ $? -eq 0 ]; then
        print_success "Training completed for $ticker"
        print_step "Model saved in saved_models/advanced/"
    else
        print_error "Training failed for $ticker"
    fi
}

train_multiple_tickers() {
    local tickers=("${@:-AAPL MSFT GOOGL TSLA AMZN}")
    
    print_step "Training advanced models for: ${tickers[*]}"
    
    check_venv
    activate_venv
    
    python advanced_main.py --mode train --tickers ${tickers[*]}
    
    if [ $? -eq 0 ]; then
        print_success "Training completed for all tickers"
    else
        print_error "Training failed for some tickers"
    fi
}

# Prediction functions
make_predictions() {
    local ticker=${1:-AAPL}
    local pred_type=${2:-single}
    
    print_step "Making $pred_type predictions for $ticker..."
    
    check_venv
    activate_venv
    
    python advanced_main.py --mode predict --tickers $ticker --prediction-type $pred_type
    
    if [ $? -eq 0 ]; then
        print_success "Predictions completed for $ticker"
    else
        print_error "Prediction failed for $ticker"
    fi
}

# Evaluation functions  
evaluate_models() {
    local tickers=("${@:-AAPL}")
    
    print_step "Evaluating models for: ${tickers[*]}"
    
    check_venv
    activate_venv
    
    python advanced_main.py --mode evaluate --tickers ${tickers[*]}
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation completed"
        print_step "Reports saved in logs/ directory"
    else
        print_error "Evaluation failed"
    fi
}

# Online learning
start_online_learning() {
    local tickers=("${@:-AAPL}")
    
    print_step "Starting online learning for: ${tickers[*]}"
    print_warning "Press Ctrl+C to stop online learning"
    
    check_venv
    activate_venv
    
    python advanced_main.py --mode online --tickers ${tickers[*]}
}

# Feature engineering test
test_feature_engineering() {
    local ticker=${1:-AAPL}
    
    print_step "Testing feature engineering for $ticker..."
    
    check_venv
    activate_venv
    
    python -c "
import sys
sys.path.append('advanced_features')
from feature_engineering import test_feature_engineering
test_feature_engineering()
"
    
    if [ $? -eq 0 ]; then
        print_success "Feature engineering test passed"
    else
        print_error "Feature engineering test failed"
    fi
}

# Model architecture test
test_model_architecture() {
    print_step "Testing model architecture..."
    
    check_venv
    activate_venv
    
    python -c "
import sys
sys.path.append('advanced_models')
from attention_lstm import AdvancedLSTMModel, compare_architectures
model = AdvancedLSTMModel(sequence_length=60, n_features=25)
model.build_model()
compare_architectures()
"
    
    if [ $? -eq 0 ]; then
        print_success "Model architecture test passed"
    else
        print_error "Model architecture test failed"
    fi
}

# Performance benchmarking
run_benchmark() {
    print_step "Running performance benchmark..."
    
    check_venv
    activate_venv
    
    echo "Benchmarking on AAPL vs MSFT..."
    
    # Train and evaluate both tickers
    python advanced_main.py --mode train --tickers AAPL MSFT
    python advanced_main.py --mode evaluate --tickers AAPL MSFT
    
    print_success "Benchmark completed - check logs for results"
}

# System diagnostics
run_diagnostics() {
    print_step "Running system diagnostics..."
    
    echo "=== System Information ==="
    python --version
    
    if [ -d "venv_advanced" ]; then
        activate_venv
        
        echo "=== Package Versions ==="
        python -c "
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
print(f'TensorFlow: {tf.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
"
        
        echo "=== GPU Availability ==="
        python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')
for gpu in gpus:
    print(f'  - {gpu}')
"
        
        echo "=== Data Access Test ==="
        python -c "
import yfinance as yf
try:
    data = yf.download('AAPL', period='5d')
    print(f'✓ Downloaded {len(data)} days of data')
except Exception as e:
    print(f'✗ Data access failed: {e}')
"
    else
        print_error "Virtual environment not found!"
    fi
    
    print_success "Diagnostics completed"
}

# Clean up function
cleanup() {
    print_step "Cleaning up system..."
    
    # Remove cache directories
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    
    # Clean up log files older than 7 days
    if [ -d "logs" ]; then
        find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
    fi
    
    print_success "Cleanup completed"
}

# Help function
show_help() {
    echo ""
    echo "Usage: ./run_commands.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Available commands:"
    echo ""
    echo "=== Setup ==="
    echo "  setup                    - Initial system setup"
    echo "  demo                     - Run comprehensive demo (recommended first run)"
    echo ""
    echo "=== Training ==="
    echo "  train [TICKER]           - Train model for single ticker (default: AAPL)"
    echo "  train-multi [TICKERS]    - Train models for multiple tickers"
    echo ""
    echo "=== Predictions ==="
    echo "  predict [TICKER]         - Single point prediction (default: AAPL)"
    echo "  predict-multi [TICKER]   - Multi-step predictions"
    echo "  predict-intervals [TICKER] - Predictions with confidence intervals"
    echo ""
    echo "=== Evaluation ==="
    echo "  evaluate [TICKERS]       - Evaluate trained models"
    echo "  benchmark               - Performance benchmark"
    echo ""
    echo "=== Real-time ==="
    echo "  online [TICKERS]        - Start online learning"
    echo ""
    echo "=== Testing ==="
    echo "  test-features           - Test feature engineering"
    echo "  test-model             - Test model architecture"
    echo "  diagnostics            - Run system diagnostics"
    echo ""
    echo "=== Utilities ==="
    echo "  cleanup                - Clean up cache and old files"
    echo "  help                   - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_commands.sh demo"
    echo "  ./run_commands.sh train AAPL"
    echo "  ./run_commands.sh train-multi AAPL MSFT GOOGL"
    echo "  ./run_commands.sh predict-intervals TSLA"
    echo "  ./run_commands.sh online AAPL MSFT"
    echo ""
}

# Main command dispatcher
case "${1:-help}" in
    "setup")
        setup_system
        ;;
    "demo")
        run_demo
        ;;
    "train")
        train_single_ticker $2
        ;;
    "train-multi")
        shift
        train_multiple_tickers "$@"
        ;;
    "predict")
        make_predictions $2 "single"
        ;;
    "predict-multi")
        make_predictions $2 "multi_step"
        ;;
    "predict-intervals")
        make_predictions $2 "intervals"
        ;;
    "evaluate")
        shift
        evaluate_models "$@"
        ;;
    "online")
        shift
        start_online_learning "$@"
        ;;
    "test-features")
        test_feature_engineering $2
        ;;
    "test-model")
        test_model_architecture
        ;;
    "benchmark")
        run_benchmark
        ;;
    "diagnostics")
        run_diagnostics
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_help
        ;;
esac

echo ""
print_success "Command completed!"
echo "=================================================="