# Chess App - Configurable Model System

## Overview

The chess application now supports flexible model configurations, making it easy to replace `model_100.pth` with Falcon models or other advanced models for both chess piece classification and move prediction.

## Current Configuration

### Default Setup
```python
# Current configuration in app.py
use_local_model = False  # Uses API for chess piece classification
use_local_move_prediction = True  # Uses local model for move prediction

MODEL_CONFIG = {
    "piece_classification": {
        "local": {"type": "cnn", "path": "model_100.pth"},
        "api": {"model": "your-chess-piece-model"}
    },
    "move_prediction": {
        "local": {"type": "stockfish", "depth": 10},
        "api": {"model": "microsoft/DialoGPT-medium"}
    }
}
```

## Supported Model Types

### For Chess Piece Classification:
- **CNN**: Local PyTorch model (current: `model_100.pth`)
- **Transformer/Falcon**: Hugging Face transformer models
- **API**: Any Hugging Face model via API

### For Move Prediction:
- **Stockfish**: Local chess engine (requires `chess` library)
- **Falcon**: Falcon models (local or API)
- **Neural**: Custom neural networks
- **API**: Any Hugging Face model via API

## Quick Configuration Changes

### 1. Switch to Falcon for Move Prediction
```python
# In app.py, uncomment this line:
switch_model_preset("falcon_setup")
```

### 2. Use Specific Falcon Model
```python
# In app.py, uncomment and modify:
configure_falcon_for_moves("tiiuae/falcon-7b-instruct", use_local=True)
```

### 3. Custom Configuration
```python
# Modify MODEL_CONFIG directly:
MODEL_CONFIG["move_prediction"]["local"] = {
    "type": "falcon",
    "model_name": "tiiuae/falcon-40b-instruct"
}
```

## Installation Requirements

### Core Dependencies (Already Installed)
- `gradio`
- `huggingface_hub`
- `torch`
- `opencv-python`

### Optional Dependencies
```bash
# For Falcon and transformer models
pip install transformers accelerate

# For Stockfish chess engine
pip install chess
sudo apt-get install stockfish  # or brew install stockfish

# For optimized model loading
pip install bitsandbytes
```

**Quick Install Script:**
```bash
./install_optional_deps.sh
```

## Configuration Presets

### Available Presets:
1. **`"default_cnn_stockfish"`**: CNN + Stockfish (current default)
2. **`"falcon_setup"`**: CNN + Falcon for moves
3. **`"full_api"`**: API for everything

### Usage:
```python
from app import switch_model_preset
switch_model_preset("falcon_setup")
```

## File Structure

```
/workspaces/MLOps_Case_Study_1/
├── app.py                     # Main application with flexible config
├── model_100.pth             # Current CNN model
├── model_config_examples.py   # Configuration examples
├── install_optional_deps.sh   # Dependency installer
├── requirements.txt           # Core + optional dependencies
└── .env                       # Environment variables
```

## Easy Model Replacement

### Replace model_100.pth with Falcon:

1. **For Chess Piece Classification:**
   ```python
   MODEL_CONFIG["piece_classification"]["local"] = {
       "type": "falcon",
       "model_name": "your-falcon-vision-model"
   }
   ```

2. **For Move Prediction:**
   ```python
   configure_falcon_for_moves("tiiuae/falcon-7b-instruct")
   ```

### Replace with Custom Model:

1. **Add your model file** (e.g., `my_model.pth`)
2. **Update configuration:**
   ```python
   MODEL_CONFIG["piece_classification"]["local"]["path"] = "my_model.pth"
   ```

## Testing Configurations

Run the configuration examples:
```bash
python model_config_examples.py
```

## GUI Features

The Gradio interface includes:
- **Toggle button**: Switch between API and local processing
- **FEN input**: Direct FEN string input
- **Image upload**: Chess board image processing
- **Dual output**: FEN generation + move prediction

## Next Steps

1. **Install optional dependencies** if needed
2. **Choose your preferred configuration** from the presets
3. **Uncomment the desired lines** in `app.py`
4. **Run the application**: `python app.py`

The system is now fully modular and ready for easy model replacement!
