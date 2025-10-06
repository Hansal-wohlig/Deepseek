"""
Advanced Stock Prediction System Setup Script
Automates the complete setup process for the enhanced system
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import shutil
import requests
import zipfile
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedSystemSetup:
    """
    Comprehensive setup for the advanced stock prediction system
    """
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.venv_path = self.base_dir / "venv_advanced"
        self.success_count = 0
        self.total_steps = 8
        
    def run_command(self, command: str, description: str = None) -> bool:
        """
        Execute shell command with error handling
        """
        if description:
            logger.info(f"Running: {description}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=self.base_dir
            )
            logger.info(f"✓ {description or command}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {description or command}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def check_python_version(self) -> bool:
        """
        Check if Python version is compatible
        """
        logger.info("Step 1: Checking Python version...")
        
        python_version = sys.version_info
        if python_version.major >= 3 and python_version.minor >= 8:
            logger.info(f"✓ Python {python_version.major}.{python_version.minor} detected")
            self.success_count += 1
            return True
        else:
            logger.error(f"✗ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
    
    def create_virtual_environment(self) -> bool:
        """
        Create virtual environment for the project
        """
        logger.info("Step 2: Creating virtual environment...")
        
        if self.venv_path.exists():
            logger.info("Virtual environment already exists, removing old one...")
            shutil.rmtree(self.venv_path)
        
        success = self.run_command(
            f"python -m venv {self.venv_path}",
            "Creating virtual environment"
        )
        
        if success:
            self.success_count += 1
        return success
    
    def install_dependencies(self) -> bool:
        """
        Install all required dependencies
        """
        logger.info("Step 3: Installing dependencies...")
        
        # Determine pip path based on OS
        if os.name == 'nt':  # Windows
            pip_path = self.venv_path / "Scripts" / "pip"
        else:  # Linux/Mac
            pip_path = self.venv_path / "bin" / "pip"
        
        # Upgrade pip first
        success = self.run_command(
            f"{pip_path} install --upgrade pip",
            "Upgrading pip"
        )
        
        if not success:
            return False
        
        # Install requirements
        requirements_file = self.base_dir / "requirements_advanced.txt"
        if requirements_file.exists():
            success = self.run_command(
                f"{pip_path} install -r {requirements_file}",
                "Installing advanced requirements"
            )
        else:
            # Install core packages if requirements file doesn't exist
            core_packages = [
                "tensorflow>=2.10.0",
                "scikit-learn>=1.0.0",
                "pandas>=1.5.0",
                "numpy>=1.21.0",
                "yfinance>=0.2.0",
                "ta>=0.10.0",
                "matplotlib>=3.5.0",
                "seaborn>=0.11.0"
            ]
            
            for package in core_packages:
                success = self.run_command(
                    f"{pip_path} install {package}",
                    f"Installing {package}"
                )
                if not success:
                    return False
        
        if success:
            self.success_count += 1
        return success
    
    def setup_directory_structure(self) -> bool:
        """
        Create necessary directories
        """
        logger.info("Step 4: Setting up directory structure...")
        
        directories = [
            "advanced_features",
            "advanced_models", 
            "advanced_training",
            "advanced_evaluation",
            "deployment",
            "saved_models/advanced",
            "logs",
            "data",
            "config"
        ]
        
        try:
            for dir_name in directories:
                dir_path = self.base_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Created directory: {dir_name}")
            
            self.success_count += 1
            return True
        except Exception as e:
            logger.error(f"✗ Error creating directories: {e}")
            return False
    
    def verify_tensorflow_installation(self) -> bool:
        """
        Verify TensorFlow installation and GPU support
        """
        logger.info("Step 5: Verifying TensorFlow installation...")
        
        # Determine python path based on OS
        if os.name == 'nt':  # Windows
            python_path = self.venv_path / "Scripts" / "python"
        else:  # Linux/Mac
            python_path = self.venv_path / "bin" / "python"
        
        # Test TensorFlow import
        test_code = '''
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print("TensorFlow installation verified!")
'''
        
        success = self.run_command(
            f"{python_path} -c \"{test_code}\"",
            "Testing TensorFlow installation"
        )
        
        if success:
            self.success_count += 1
        return success
    
    def test_data_access(self) -> bool:
        """
        Test access to financial data APIs
        """
        logger.info("Step 6: Testing data access...")
        
        # Determine python path based on OS
        if os.name == 'nt':  # Windows
            python_path = self.venv_path / "Scripts" / "python"
        else:  # Linux/Mac
            python_path = self.venv_path / "bin" / "python"
        
        test_code = '''
import yfinance as yf
import pandas as pd
data = yf.download("AAPL", period="5d")
print(f"Downloaded {len(data)} days of AAPL data")
print("Data access verified!")
'''
        
        success = self.run_command(
            f"{python_path} -c \"{test_code}\"",
            "Testing yfinance data access"
        )
        
        if success:
            self.success_count += 1
        return success
    
    def create_activation_scripts(self) -> bool:
        """
        Create convenient activation scripts
        """
        logger.info("Step 7: Creating activation scripts...")
        
        try:
            # Windows activation script
            if os.name == 'nt':
                activate_script = self.base_dir / "activate_advanced.bat"
                with open(activate_script, 'w') as f:
                    f.write(f"@echo off\n")
                    f.write(f"call {self.venv_path}\\Scripts\\activate.bat\n")
                    f.write(f"echo Advanced Stock Prediction Environment Activated!\n")
                    f.write(f"echo Run: python advanced_main.py --mode demo\n")
            else:
                # Linux/Mac activation script
                activate_script = self.base_dir / "activate_advanced.sh"
                with open(activate_script, 'w') as f:
                    f.write(f"#!/bin/bash\n")
                    f.write(f"source {self.venv_path}/bin/activate\n")
                    f.write(f"echo 'Advanced Stock Prediction Environment Activated!'\n")
                    f.write(f"echo 'Run: python advanced_main.py --mode demo'\n")
                
                # Make script executable
                os.chmod(activate_script, 0o755)
            
            logger.info(f"✓ Created activation script: {activate_script.name}")
            self.success_count += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ Error creating activation scripts: {e}")
            return False
    
    def run_system_test(self) -> bool:
        """
        Run a comprehensive system test
        """
        logger.info("Step 8: Running system test...")
        
        # Determine python path based on OS
        if os.name == 'nt':  # Windows
            python_path = self.venv_path / "Scripts" / "python"
        else:  # Linux/Mac
            python_path = self.venv_path / "bin" / "python"
        
        # Test advanced system components
        test_code = '''
import sys
import os
sys.path.append("advanced_features")
sys.path.append("advanced_models")

# Test feature engineering
try:
    from feature_engineering import AdvancedFeatureEngineering
    fe = AdvancedFeatureEngineering()
    print("✓ Feature engineering module loaded")
except Exception as e:
    print(f"✗ Feature engineering test failed: {e}")
    sys.exit(1)

# Test LSTM model
try:
    from attention_lstm import AdvancedLSTMModel
    model = AdvancedLSTMModel(sequence_length=60, n_features=25)
    print("✓ Advanced LSTM model loaded")
except Exception as e:
    print(f"✗ LSTM model test failed: {e}")
    sys.exit(1)

print("All system tests passed!")
'''
        
        success = self.run_command(
            f"{python_path} -c \"{test_code}\"",
            "Running system component tests"
        )
        
        if success:
            self.success_count += 1
        return success
    
    def setup_complete(self) -> None:
        """
        Display setup completion message
        """
        logger.info("\n" + "="*60)
        logger.info("ADVANCED STOCK PREDICTION SYSTEM SETUP")
        logger.info("="*60)
        
        if self.success_count == self.total_steps:
            logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
            logger.info(f"✓ All {self.total_steps} steps completed")
            
            # Provide next steps
            logger.info("\n📋 NEXT STEPS:")
            logger.info("1. Activate the environment:")
            if os.name == 'nt':
                logger.info(f"   activate_advanced.bat")
            else:
                logger.info(f"   source activate_advanced.sh")
            
            logger.info("2. Run the demo:")
            logger.info("   python advanced_main.py --mode demo --tickers AAPL")
            
            logger.info("3. Train models:")
            logger.info("   python advanced_main.py --mode train --tickers AAPL MSFT")
            
            logger.info("4. Make predictions:")
            logger.info("   python advanced_main.py --mode predict --tickers AAPL")
            
            logger.info("\n📚 Documentation: See ADVANCED_README.md for detailed usage")
            
        else:
            logger.error(f"⚠️  SETUP INCOMPLETE: {self.success_count}/{self.total_steps} steps completed")
            logger.error("Please review the errors above and run setup again")
    
    def run_full_setup(self) -> bool:
        """
        Run the complete setup process
        """
        logger.info("Starting Advanced Stock Prediction System Setup...")
        logger.info("="*60)
        
        steps = [
            self.check_python_version,
            self.create_virtual_environment,
            self.install_dependencies,
            self.setup_directory_structure,
            self.verify_tensorflow_installation,
            self.test_data_access,
            self.create_activation_scripts,
            self.run_system_test
        ]
        
        for step in steps:
            if not step():
                logger.error(f"Setup failed at step: {step.__name__}")
                break
        
        self.setup_complete()
        return self.success_count == self.total_steps

def main():
    """
    Main setup entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Stock Prediction System Setup')
    parser.add_argument('--base-dir', help='Base directory for installation', default='.')
    parser.add_argument('--skip-test', action='store_true', help='Skip system tests')
    
    args = parser.parse_args()
    
    # Initialize and run setup
    setup = AdvancedSystemSetup(args.base_dir)
    success = setup.run_full_setup()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()