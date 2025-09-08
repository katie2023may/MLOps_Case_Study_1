#!/bin/bash

echo "Chess App - Optional Dependencies Installer"
echo "==========================================="

echo ""
echo "This script installs optional dependencies for advanced features:"
echo "1. Transformers (for Falcon and other transformer models)"
echo "2. Chess library (for Stockfish integration)"
echo "3. Accelerate & BitsAndBytes (for optimized model loading)"

echo ""
read -p "Install transformer libraries (for Falcon models)? [y/N]: " install_transformers
read -p "Install chess library (for Stockfish engine)? [y/N]: " install_chess
read -p "Install acceleration libraries (for faster models)? [y/N]: " install_accel

echo ""
echo "Installing selected dependencies..."

if [[ $install_transformers =~ ^[Yy]$ ]]; then
    echo "Installing transformers..."
    pip install transformers
fi

if [[ $install_chess =~ ^[Yy]$ ]]; then
    echo "Installing chess library..."
    pip install chess
    
    # Try to install Stockfish engine
    echo "Attempting to install Stockfish engine..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y stockfish
    elif command -v brew &> /dev/null; then
        brew install stockfish
    else
        echo "Please install Stockfish manually for your system"
        echo "Visit: https://stockfishchess.org/download/"
    fi
fi

if [[ $install_accel =~ ^[Yy]$ ]]; then
    echo "Installing acceleration libraries..."
    pip install accelerate bitsandbytes
fi

echo ""
echo "Installation complete!"
echo ""
echo "To configure models, edit the configuration in app.py or run:"
echo "python model_config_examples.py"
