"""
Chess Board to FEN Converter with Move Prediction

This application provides configurable chess analysis with support for multiple model types:

1. Chess Piece Classification (Image → FEN):
   - API model

2. Move Prediction (FEN → Best Move):
   - Local Model

Configuration:
- Edit MODEL_CONFIG to change model settings
- Use MODEL_PRESETS for quick configuration switches
- Use configure_falcon_for_moves() to easily set up Falcon

Dependencies:
- Core: numpy, opencv-python, torch, gradio, huggingface_hub
- Optional: transformers (for Falcon), chess (for Stockfish)
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import urllib.parse
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import InferenceClient

# dotenv path
dotenv_path = find_dotenv()
load_dotenv(dotenv_path=dotenv_path)
hf_token = os.getenv("HF_Token")  # FIXME: Set your actual Hugging Face token in .env file

use_local_model = False # determines whether to use the local model or not (set to False to use API)
use_local_move_prediction = True # determines whether to use local model for move prediction

# Model Configuration
MODEL_CONFIG = {
    # Chess piece classification models
    "piece_classification": {
        "local": {
            "type": "cnn",  # FIXME: Change to "falcon" or "transformer" if replacing with different model
            "path": "model_100.pth",  # FIXME: Update path to your actual model file
            "model_class": "CNNModel"  # FIXME: Update class name if using different model architecture
        },
        "api": {
            "model": "your-chess-piece-model",  # FIXME: Replace with actual HF chess piece classification model ID
            "fallback_to_local": True
        }
    },
    
    # Move prediction models
    "move_prediction": {
        "local": {
            "type": "stockfish",  # options: "stockfish", "neural", "falcon", "transformer"
            "path": None,  # Path for neural models, None for Stockfish
            "model_name": None,  # For Falcon or other HF models
            "depth": 10  # For Stockfish depth
        },
        "api": {
            "model": "microsoft/DialoGPT-medium",  # FIXME: Replace with chess-specific model like Falcon
            "fallback_to_local": True,
            "max_tokens": 10
        }
    }
}

# Preset configurations for easy switching
MODEL_PRESETS = {
    "default_cnn_stockfish": {
        "piece_classification": {"local": {"type": "cnn", "path": "model_100.pth"}},
        "move_prediction": {"local": {"type": "stockfish", "depth": 10}}
    },
    
    "falcon_setup": {
        "piece_classification": {"local": {"type": "cnn", "path": "model_100.pth"}},  # Keep CNN for pieces
        "move_prediction": {
            "local": {"type": "falcon", "model_name": "tiiuae/falcon-7b-instruct"},
            "api": {"model": "tiiuae/falcon-7b-instruct", "max_tokens": 20}
        }
    },
    
    "full_api": {
        "piece_classification": {"api": {"model": "your-chess-piece-model"}},  # FIXME: Replace with actual model ID
        "move_prediction": {"api": {"model": "tiiuae/falcon-7b-instruct", "max_tokens": 20}}
    }
}

def switch_model_preset(preset_name):
    """Switch to a predefined model configuration preset"""
    global MODEL_CONFIG
    if preset_name in MODEL_PRESETS:
        # Update MODEL_CONFIG with preset values
        preset = MODEL_PRESETS[preset_name]
        for category, config in preset.items():
            if category in MODEL_CONFIG:
                MODEL_CONFIG[category].update(config)
        print(f"Switched to model preset: {preset_name}")
        return True
    else:
        print(f"Unknown preset: {preset_name}")
        return False

# Function to easily configure Falcon for move prediction
def configure_falcon_for_moves(model_name="tiiuae/falcon-7b-instruct", use_local=True):
    """Easy function to configure Falcon for move prediction"""
    global MODEL_CONFIG
    falcon_config = {
        "type": "falcon",
        "model_name": model_name
    }
    
    if use_local:
        MODEL_CONFIG["move_prediction"]["local"] = falcon_config
    else:
        MODEL_CONFIG["move_prediction"]["api"]["model"] = model_name
        MODEL_CONFIG["move_prediction"]["api"]["max_tokens"] = 20
    
    print(f"Configured Falcon ({model_name}) for {'local' if use_local else 'API'} move prediction")

# Uncomment and modify these lines to switch to different configurations:
# switch_model_preset("falcon_setup")  # Use Falcon for move prediction
# configure_falcon_for_moves("tiiuae/falcon-40b-instruct")  # Use larger Falcon model

# Set device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN model definition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First Convolutional Layer: 32 features, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        # Second Convolutional Layer: 64 features, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # Fully connected layer
        self.fc1 = nn.Linear(8 * 8 * 64, 1024)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with a probability of 0.5
        # Output layer
        self.fc2 = nn.Linear(1024, 13)

        # Initialize weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights with truncated normal (approximate with normal and clamp)
        nn.init.trunc_normal_(self.conv1.weight, std=0.1)
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.trunc_normal_(self.conv2.weight, std=0.1)
        nn.init.constant_(self.conv2.bias, 0.1)
        nn.init.trunc_normal_(self.fc1.weight, std=0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.trunc_normal_(self.fc2.weight, std=0.1)
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        # Apply first convolutional layer + ReLU activation
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # First pooling

        # Apply second convolutional layer + ReLU activation
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Second pooling

        # Flatten the tensor
        x = x.view(-1, 8 * 8 * 64)

        # Fully connected layer + ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout
        x = self.dropout(x)

        # Output layer (no activation, as CrossEntropyLoss applies Softmax internally)
        x = self.fc2(x)
        return x

# Helper function to convert label index to name
def labelIndex2Name(label_index):
    mapping = {
        0: '1',   # Empty square
        1: 'K',   # White King
        2: 'Q',   # White Queen
        3: 'R',   # White Rook
        4: 'B',   # White Bishop
        5: 'N',   # White Knight
        6: 'P',   # White Pawn
        7: 'k',   # Black King
        8: 'q',   # Black Queen
        9: 'r',   # Black Rook
        10: 'b',  # Black Bishop
        11: 'n',  # Black Knight
        12: 'p'   # Black Pawn
    }
    return mapping.get(label_index, '?')  # '?' for unknown classes

# Load the saved model
def load_model(model_config=None):
    """Load model based on configuration"""
    if model_config is None:
        model_config = MODEL_CONFIG["piece_classification"]["local"]
    
    model_type = model_config.get("type", "cnn")
    model_path = model_config.get("path", "model_100.pth")
    
    if model_type == "cnn":
        model = CNNModel()  # Instantiate the CNN model
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"CNN model file '{model_path}' not found.")
    
    elif model_type == "falcon" or model_type == "transformer":
        # For Falcon or other transformer models
        try:
            from transformers import AutoModel, AutoTokenizer
            model_name = model_config.get("model_name", "tiiuae/falcon-7b")
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return {"model": model, "tokenizer": tokenizer, "type": "transformer"}
        except ImportError:
            print("Transformers library not installed. Falling back to CNN model.")
            return load_model({"type": "cnn", "path": "model_100.pth"})
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Prepare an image for prediction
def prepare_image(image):
    """Prepares an image for prediction by the model"""
    img = Image.fromarray(image).convert('L')  # Convert to grayscale
    img = img.resize((32, 32))  # Resize to 32x32
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add channel dimension
    img_tensor = torch.tensor(img_array, dtype=torch.float32)  # Convert to tensor
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

# Predict the class of the image
def predict_image(image_tensor, local_model:bool):
    """Predict the class of a single image using the loaded model if
    local_model is set to True
    use HF API otherwise"""
    if local_model is True:
        model_config = MODEL_CONFIG["piece_classification"]["local"]
        model = load_model(model_config)
        print(f"Model loaded successfully: {model_config['type']}")
        
        if isinstance(model, dict) and model.get("type") == "transformer":
            # Handle transformer models (like Falcon)
            # This would need specific implementation for chess piece classification
            # For now, fallback to CNN
            print("Transformer model detected, but chess piece classification not implemented. Falling back to CNN.")
            cnn_config = {"type": "cnn", "path": "model_100.pth"}
            model = load_model(cnn_config)
        
        # Standard CNN prediction
        image_tensor = image_tensor.to(device)  # Move image to the same device as the model
        with torch.no_grad():  # Disable gradient computation during inference
            outputs = model(image_tensor)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class index
        return predicted.item()
    else:
        # Use Hugging Face API for chess piece classification
        try:
            # Convert tensor back to PIL Image for API
            img_array = image_tensor.squeeze().numpy()
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            
            # Initialize Hugging Face client
            client = InferenceClient(token=hf_token)  # FIXME: Add timeout and other client configuration options
            
            # Convert PIL image to bytes
            import io
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Make API call (you'll need to replace with your actual model endpoint)
            # For now, using a placeholder - you'll need to specify your chess piece classification model
            result = client.image_classification(
                image=img_bytes.read(),
                model="your-chess-piece-model"  # FIXME: Replace with actual chess piece classification model ID
            )
            
            # Parse result and convert to class index
            # This will depend on your API model's output format
            if result and len(result) > 0:
                predicted_label = result[0]['label']
                # Map label back to class index (you'll need to implement this mapping)
                class_mapping = {  # FIXME: Update this mapping to match your model's output labels
                    'empty': 0, 'white_king': 1, 'white_queen': 2, 'white_rook': 3,
                    'white_bishop': 4, 'white_knight': 5, 'white_pawn': 6,
                    'black_king': 7, 'black_queen': 8, 'black_rook': 9,
                    'black_bishop': 10, 'black_knight': 11, 'black_pawn': 12
                }
                return class_mapping.get(predicted_label, 0)
            else:
                return 0  # Default to empty square
                
        except Exception as e:
            print(f"API prediction failed: {e}")
            # Fallback to local model
            model_config = MODEL_CONFIG["piece_classification"]["local"]
            model = load_model(model_config)
            image_tensor = image_tensor.to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs.data, 1)
            return predicted.item()

# Function to convert the board to FEN
def generate_fen(board_matrix):
    fen_rows = []
    for row in board_matrix:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == '1':  # Empty square
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    # Ensure ranks are ordered from 8 to 1
    fen = "/".join(fen_rows) + " w KQkq - 0 1"  # Default values for active color, castling, en passant, etc.
    return fen

# Chess move prediction functions
def predict_best_move_local(fen):
    """Predict best move using local model (configurable implementation)"""
    move_config = MODEL_CONFIG["move_prediction"]["local"]
    model_type = move_config.get("type", "stockfish")
    
    try:
        print(f"Predicting best move for FEN: {fen} using {model_type}")
        
        if model_type == "stockfish":
            # Use Stockfish engine
            try:
                import chess
                import chess.engine
                board = chess.Board(fen)
                depth = move_config.get("depth", 10)
                
                # Try to find Stockfish binary
                stockfish_path = "stockfish"  # FIXME: Update path to Stockfish binary if not in PATH
                with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                    result = engine.play(board, chess.engine.Limit(depth=depth))
                    return result.move.uci()
            except (ImportError, FileNotFoundError):
                print("Stockfish not available. Using placeholder move.")
                return "e2e4"  # FIXME: Replace with appropriate fallback strategy
        
        elif model_type == "falcon":
            # Use Falcon model for move prediction
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                model_name = move_config.get("model_name", "tiiuae/falcon-7b-instruct")  # FIXME: Choose your preferred Falcon model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
                
                prompt = f"Given this chess position in FEN: {fen}\nWhat is the best move? Respond with only the move in UCI notation (e.g., e2e4):"  # FIXME: Optimize prompt for your specific model
                
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)  # FIXME: Adjust generation parameters as needed
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract move from response
                move = response.split()[-1] if response else "e2e4"  # FIXME: Improve move extraction logic for your model
                return move
                
            except ImportError:
                print("Transformers library not available. Using placeholder.")
                return "e2e4"  # FIXME: Replace with appropriate fallback strategy
        
        elif model_type == "neural":
            # Custom neural network for move prediction
            model_path = move_config.get("path")
            if model_path and os.path.exists(model_path):
                # Load your custom neural network model here
                print(f"Loading neural model from {model_path}")
                # FIXME: Implement loading and inference for your custom neural network
                return "e2e4"  # FIXME: Replace with actual neural network prediction
            else:
                print("Neural model path not found. Using placeholder.")
                return "e2e4"  # FIXME: Replace with appropriate fallback strategy
        
        else:
            print(f"Unknown model type: {model_type}. Using placeholder.")
            return "e2e4"  # FIXME: Replace with appropriate fallback strategy
        
    except Exception as e:
        print(f"Local move prediction failed: {e}")
        return "No move predicted"

def predict_best_move_api(fen):
    """Predict best move using Hugging Face API"""
    api_config = MODEL_CONFIG["move_prediction"]["api"]
    model_name = api_config.get("model", "microsoft/DialoGPT-medium")
    max_tokens = api_config.get("max_tokens", 10)
    
    try:
        # Initialize Hugging Face client
        client = InferenceClient(token=hf_token)  # FIXME: Add timeout and other client configuration options
        
        # Create an appropriate prompt for the model
        if "falcon" in model_name.lower():
            prompt = f"Given this chess position in FEN: {fen}\nWhat is the best move? Answer with only the move in UCI notation:"  # FIXME: Optimize prompt for Falcon models
        else:
            prompt = f"Given the chess position in FEN notation: {fen}, what is the best move?"  # FIXME: Optimize prompt for your specific model
        
        print(f"Using API model: {model_name}")
        
        # Use text generation for move prediction
        result = client.text_generation(
            prompt=prompt,
            model=model_name,
            max_new_tokens=max_tokens,
            temperature=0.1  # FIXME: Adjust temperature and other parameters as needed
        )
        
        # Parse the result to extract the move
        if result:
            # Clean up the response and extract move
            response = result.strip()
            # Try to extract UCI move pattern (e.g., e2e4, Nf3, etc.)
            import re
            move_pattern = r'\b[a-h][1-8][a-h][1-8][qrbnQRBN]?\b'  # FIXME: Adjust regex pattern based on your model's output format
            matches = re.findall(move_pattern, response)
            if matches:
                return matches[0]
            else:
                # Fallback: return last word if it looks like a move
                words = response.split()
                if words and len(words[-1]) >= 4:
                    return words[-1]
        
        return "e2e4"  # FIXME: Replace with appropriate default fallback move
        
    except Exception as e:
        print(f"API move prediction failed: {e}")
        # Fallback to local prediction
        if api_config.get("fallback_to_local", True):
            return predict_best_move_local(fen)
        return "No move predicted"

def predict_best_move(fen, use_local=True):
    """Predict the best move for a given FEN position"""
    if use_local:
        return predict_best_move_local(fen)
    else:
        return predict_best_move_api(fen)

# Gradient and Line Detection Functions
def gradientx(img):
    # Compute gradient in x-direction using larger Sobel kernel
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=31)
    return grad_x

def gradienty(img):
    # Compute gradient in y-direction using larger Sobel kernel
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=31)
    return grad_y

def checkMatch(lineset):
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    for line in linediff:
        if abs(line - x) < 5:  # FIXME: Adjust line spacing tolerance based on board detection precision
            cnt += 1
        else:
            cnt = 0
            x = line
    return cnt == 5

def pruneLines(lineset, image_dim, margin=20):  # FIXME: Adjust margin value based on image resolution and board size
    # Remove lines near the margins
    lineset = [x for x in lineset if x > margin and x < image_dim - margin]
    if not lineset:
        return lineset
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    start_pos = 0
    for i, line in enumerate(linediff):
        if abs(line - x) < 5:  # FIXME: Adjust line spacing tolerance for line pruning accuracy
            cnt += 1
            if cnt == 5:  # FIXME: Verify expected line count matches chess board requirements
                end_pos = i + 2
                return lineset[start_pos:end_pos]
        else:
            cnt = 0
            x = line
            start_pos = i
    return lineset

def skeletonize_1d(arr):
    _arr = arr.copy()
    for i in range(len(_arr) - 1):
        if _arr[i] <= _arr[i + 1]:
            _arr[i] = 0
    for i in range(len(_arr) - 1, 0, -1):
        if _arr[i - 1] > _arr[i]:
            _arr[i] = 0
    return _arr

def getChessLines(hdx, hdy, hdx_thresh, hdy_thresh, image_shape):
    # Generate Gaussian window
    window_size = 21  # FIXME: Tune window size based on board detection accuracy
    sigma = 8.0       # FIXME: Adjust sigma for optimal Gaussian blur based on image quality
    gausswin = cv2.getGaussianKernel(window_size, sigma, cv2.CV_64F)
    gausswin = gausswin.flatten()
    half_size = window_size // 2

    # Threshold signals
    hdx_thresh_binary = np.where(hdx > hdx_thresh, 1.0, 0.0)
    hdy_thresh_binary = np.where(hdy > hdy_thresh, 1.0, 0.0)

    # Blur signals using convolution with Gaussian window
    blur_x = np.convolve(hdx_thresh_binary, gausswin, mode='same')
    blur_y = np.convolve(hdy_thresh_binary, gausswin, mode='same')

    # Skeletonize signals
    skel_x = skeletonize_1d(blur_x)
    skel_y = skeletonize_1d(blur_y)

    # Find line positions
    lines_x = np.where(skel_x > 0)[0].tolist()
    lines_y = np.where(skel_y > 0)[0].tolist()

    # Prune lines
    lines_x = pruneLines(lines_x, image_shape[1])
    lines_y = pruneLines(lines_y, image_shape[0])

    # Check if lines match expected pattern
    is_match = (len(lines_x) == 7) and (len(lines_y) == 7) and \
               checkMatch(lines_x) and checkMatch(lines_y)

    return lines_x, lines_y, is_match

def getChessTiles(img, lines_x, lines_y):
    stepx = int(round(np.mean(np.diff(lines_x))))
    stepy = int(round(np.mean(np.diff(lines_y))))

    # Pad the image if necessary
    padl_x = 0
    padr_x = 0
    padl_y = 0
    padr_y = 0
    if lines_x[0] - stepx < 0:
        padl_x = abs(lines_x[0] - stepx)
    if lines_x[-1] + stepx > img.shape[1] - 1:
        padr_x = lines_x[-1] + stepx - img.shape[1] + 1
    if lines_y[0] - stepy < 0:
        padl_y = abs(lines_y[0] - stepy)
    if lines_y[-1] + stepy > img.shape[0] - 1:
        padr_y = lines_y[-1] + stepy - img.shape[0] + 1

    img_padded = cv2.copyMakeBorder(img, padl_y, padr_y, padl_x, padr_x, cv2.BORDER_REPLICATE)

    setsx = [lines_x[0] - stepx + padl_x] + [x + padl_x for x in lines_x] + [lines_x[-1] + stepx + padl_x]
    setsy = [lines_y[0] - stepy + padl_y] + [y + padl_y for y in lines_y] + [lines_y[-1] + stepy + padl_y]

    squares = []
    for j in range(8):
        for i in range(8):
            x1 = setsx[i]
            x2 = setsx[i + 1]
            y1 = setsy[j]
            y2 = setsy[j + 1]
            # Adjust sizes to ensure squares are of equal size
            if (x2 - x1) != stepx:
                x2 = x1 + stepx
            if (y2 - y1) != stepy:
                y2 = y1 + stepy
            square = img_padded[y1:y2, x1:x2]
            squares.append(square)
    return squares

def process_image_and_generate_fen_local(image):
    """Process image using local model for piece classification"""
    return process_image_and_generate_fen(image, use_local_model=True)

def process_image_and_generate_fen_api(image):
    """Process image using API model for piece classification"""
    return process_image_and_generate_fen(image, use_local_model=False)

def process_image_and_generate_fen(image, use_local_model=True):
    
    # Convert Gradio Image to OpenCV format
    image = np.array(image.convert("RGB"))
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Preprocessing
    equ = cv2.equalizeHist(gray)
    norm_image = equ.astype(np.float32) / 255.0

    # Compute the gradients
    grad_x = gradientx(norm_image)
    grad_y = gradienty(norm_image)

    # Clip the gradients
    Dx_pos = np.clip(grad_x, 0, None)
    Dx_neg = np.clip(-grad_x, 0, None)
    Dy_pos = np.clip(grad_y, 0, None)
    Dy_neg = np.clip(-grad_y, 0, None)

    # Compute the Hough transform
    hough_Dx = (np.sum(Dx_pos, axis=0) * np.sum(Dx_neg, axis=0)) / (norm_image.shape[0] ** 2)
    hough_Dy = (np.sum(Dy_pos, axis=1) * np.sum(Dy_neg, axis=1)) / (norm_image.shape[1] ** 2)

    # Adaptive thresholding
    a = 1
    is_match = False
    lines_x = []
    lines_y = []

    while a < 5:  # FIXME: Adjust maximum threshold iterations based on detection reliability
        threshold_x = np.max(hough_Dx) * (a / 5.0)  # FIXME: Tune threshold multiplier for X-axis detection
        threshold_y = np.max(hough_Dy) * (a / 5.0)  # FIXME: Tune threshold multiplier for Y-axis detection

        lines_x, lines_y, is_match = getChessLines(hough_Dx, hough_Dy, threshold_x, threshold_y, norm_image.shape)

        if is_match:
            break
        else:
            a += 1

    if not is_match:
        print("Retrying with different normalization...")
        # Use alternative normalization
        norm_image = gray.astype(np.float32) / 255.0
        grad_x = gradientx(norm_image)
        grad_y = gradienty(norm_image)

        Dx_pos = np.clip(grad_x, 0, None)
        Dx_neg = np.clip(-grad_x, 0, None)
        Dy_pos = np.clip(grad_y, 0, None)
        Dy_neg = np.clip(-grad_y, 0, None)

        hough_Dx = (np.sum(Dx_pos, axis=0) * np.sum(Dx_neg, axis=0)) / (norm_image.shape[0] ** 2)
        hough_Dy = (np.sum(Dy_pos, axis=1) * np.sum(Dy_neg, axis=1)) / (norm_image.shape[1] ** 2)

        # Repeat the adaptive thresholding
        a = 1
        while a < 5:
            threshold_x = np.max(hough_Dx) * (a / 5.0)
            threshold_y = np.max(hough_Dy) * (a / 5.0)

            lines_x, lines_y, is_match = getChessLines(hough_Dx, hough_Dy, threshold_x, threshold_y, norm_image.shape)

            if is_match:
                break
            else:
                a += 1

    if is_match:
        print("7 horizontal and vertical lines found, slicing up squares")
        squares = getChessTiles(gray, lines_x, lines_y)
        print(f"Tiles generated: ({squares[0].shape[0]}x{squares[0].shape[1]}) * {len(squares)}")

        board_matrix = [[] for _ in range(8)]  # 8x8 board

        for i, square in enumerate(squares):
            # Calculate row and column
            row = i // 8  # Ranks 8 to 1
            col = i % 8

            # Resize to 32x32 for prediction
            resized = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)

            # Predict the piece on this square
            image_tensor = prepare_image(resized)
            predicted_class = predict_image(image_tensor, local_model=use_local_model)
            piece = labelIndex2Name(predicted_class)

            board_matrix[row].append(piece)

        # Generate FEN from board_matrix
        fen = generate_fen(board_matrix)
        print(f"Generated FEN: {fen}")

        return fen
    else:
        print(f"No squares to save for the uploaded image.")
        return "Failed to detect a valid chessboard in the image."

# Initialize and validate model configurations
def validate_model_config():
    """Validate that required model files exist"""
    piece_config = MODEL_CONFIG["piece_classification"]["local"]
    if piece_config["type"] == "cnn":
        model_path = piece_config["path"]
        if not os.path.exists(model_path):
            print(f"Warning: CNN model file '{model_path}' not found. Some features may not work.")
    
    # For move prediction, we'll validate at runtime since Stockfish/API don't need files

# Validate configuration on startup
validate_model_config()

def gradio_predict(image, fen_input, use_api_toggle):
    """
    Main prediction function that handles both image processing and FEN input
    Args:
        image: Chess board image (optional if FEN is provided)
        fen_input: Manual FEN input (optional if image is provided)
        use_api_toggle: Boolean indicating whether to use API for image processing
    """
    
    # Determine the FEN to use
    if fen_input and fen_input.strip():
        # Use manually input FEN
        fen = fen_input.strip()
        processing_method = "Manual FEN Input"
    elif image is not None:
        # Process image to get FEN
        if use_api_toggle:
            # Use API for image processing
            fen = process_image_and_generate_fen_api(image)
            processing_method = "API Image Processing"
        else:
            # Use local model for image processing
            fen = process_image_and_generate_fen_local(image)
            processing_method = "Local Image Processing"
    else:
        return "**Error:** Please provide either an image or a FEN string."
    
    if fen.startswith("Failed"):
        # Return the error message as-is in Markdown
        return f"**Error:** {fen}"
    else:
        # Predict the best move (always using local model for move prediction as specified)
        best_move = predict_best_move(fen, use_local=use_local_move_prediction)
        
        # URL-encode the FEN string to ensure it's safe for URLs
        fen_encoded = urllib.parse.quote(fen)
        
        # Create URLs for Lichess and Chess.com analysis with the encoded FEN
        lichess_url = f"https://lichess.org/editor/{fen_encoded}"
        chesscom_url = f"https://www.chess.com/analysis?fen={fen_encoded}"
        
        # Create a Markdown-formatted string with the FEN, best move, and clickable links
        markdown_output = f"""
            **Processing Method:** {processing_method}
            
            **Generated FEN:**
            {fen}

            **Predicted Best Move:**
            {best_move}
            
            **Analyze Your Position:**
            
            - [Analyze on Lichess]({lichess_url})
            - [Analyze on Chess.com]({chesscom_url})
        """
        return markdown_output

# Create Gradio Interface
with gr.Blocks(title="Chessboard to FEN Converter with Move Prediction") as iface:
    gr.Markdown("""
    # Chessboard to FEN Converter with Move Prediction
    
    Upload an image of a chessboard OR enter a FEN string directly. The system will predict the best move and provide links to analyze the position.
    
    **Two modes:**
    - **API Mode**: Uses API for image processing → FEN → best move prediction
    - **Local Mode**: Uses local model for image processing → FEN → best move prediction
    - **FEN Input**: Skip image processing entirely and go directly from FEN → best move prediction
    """)
    
    with gr.Row():
        with gr.Column():
            # Input components
            image_input = gr.Image(type="pil", label="Upload Chessboard Image (Optional)")
            fen_input = gr.Textbox(
                label="Or Enter FEN String Directly",
                placeholder="e.g., rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                lines=2
            )
            use_api_toggle = gr.Checkbox(
                label="Use API for Image Processing",
                value=False,
                info="When checked, uses API for chess piece classification. When unchecked, uses local model."
            )
            
            # Submit button
            submit_btn = gr.Button("Analyze Position", variant="primary")
            
        with gr.Column():
            # Output component
            output = gr.Markdown(label="Results")
    
    # Example inputs
    gr.Examples(
        examples=[
            [None, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", False],
            [None, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", False],
        ],
        inputs=[image_input, fen_input, use_api_toggle],
        label="Example FEN Positions"
    )
    
    # Connect the function
    submit_btn.click(
        fn=gradio_predict,
        inputs=[image_input, fen_input, use_api_toggle],
        outputs=output
    )

# Launch the interface
iface.launch(share=True)  # FIXME: Configure deployment settings (share=False for production, add auth, set port, etc.)