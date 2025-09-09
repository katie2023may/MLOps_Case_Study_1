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

# FIXME: Add these imports when implementing local LLM move prediction
# import chess
# import chess.engine  
# from transformers import pipeline


# dotenv path
dotenv_path = find_dotenv()
load_dotenv(dotenv_path=dotenv_path)
hf_token = os.getenv("HF_Token")

use_local_model = True # determines whether to use the local model or not

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
def load_model(model_path):
    model = CNNModel()  # Instantiate the model
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model parameters
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

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
    use_local_model is set to True
    use HF API otherwise"""
    local_model=use_local_model
    if local_model is True:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        image_tensor = image_tensor.to(device)  # Move image to the same device as the model
        with torch.no_grad():  # Disable gradient computation during inference
            outputs = model(image_tensor)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class index
        return predicted.item()
    else:
        # FIXME: Fix the API implementation - current code has undefined variables
        try:
            # Convert tensor to PIL Image for API
            image_array = image_tensor.squeeze().numpy() * 255
            image_pil = Image.fromarray(image_array.astype(np.uint8))
            
            # FIXME: Replace with actual Hugging Face model for chess piece classification
            client = InferenceClient(model="FIXME-chess-piece-classification-model", token=hf_token)
            
            # FIXME: Use proper image classification API call
            result = client.image_classification(image_pil)
            
            # FIXME: Map API result to your label indices (0-12)
            # This depends on your API model's output format
            predicted_class = 0  # Placeholder
            return predicted_class
            
        except Exception as e:
            # FIXME: Add proper error logging for production
            print(f"API prediction error: {e}, falling back to local model")
            # Fallback to local model
            model = load_model(MODEL_PATH)
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

# FIXME: Add move prediction functions after generate_fen function
def predict_best_move_api(fen):
    """Use API to predict best move from FEN"""
    try:
        # FIXME: Replace with actual chess move prediction model
        client = InferenceClient(model="FIXME-chess-move-model", token=hf_token)
        
        # FIXME: Optimize this prompt for better chess move prediction
        prompt = f"Given this chess position in FEN notation: {fen}\nWhat is the best move? Reply with only the move in algebraic notation."
        
        # FIXME: Adjust these parameters based on model requirements
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.1,
            return_full_text=False
        )
        
        # FIXME: Improve move extraction logic based on actual model output format
        move = response.strip().split()[0] if response else "No move found"
        return move
    except Exception as e:
        # FIXME: Add proper error logging for production
        print(f"API move prediction error: {e}")
        return "e4"  # FIXME: Replace with better fallback move

def predict_best_move_local(fen):
    """Use local LLM to predict best move from FEN"""
    try:
        # FIXME: Implement local LLM for move prediction (options below)
        # Option 1: Use transformers library with local model
        # from transformers import pipeline
        # generator = pipeline("text-generation", model="FIXME-local-chess-model")
        
        # Option 2: Use chess engine like Stockfish
        # import chess
        # import chess.engine
        # engine = chess.engine.SimpleEngine.popen_uci("FIXME-stockfish-path")
        
        # FIXME: Replace this placeholder with actual local LLM implementation
        # For now, returning a placeholder
        return "Nf3"  # FIXME: Implement actual local move prediction
        
    except Exception as e:
        # FIXME: Add proper error logging for production
        print(f"Local move prediction error: {e}")
        return "e4"  # FIXME: Replace with better fallback move

def predict_best_move(fen, use_api=True):
    """Router function to choose between API and local move prediction"""
    if use_api:
        return predict_best_move_api(fen)
    else:
        return predict_best_move_local(fen)


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
        if abs(line - x) < 5:
            cnt += 1
        else:
            cnt = 0
            x = line
    return cnt == 5

def pruneLines(lineset, image_dim, margin=20):
    # Remove lines near the margins
    lineset = [x for x in lineset if x > margin and x < image_dim - margin]
    if not lineset:
        return lineset
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    start_pos = 0
    for i, line in enumerate(linediff):
        if abs(line - x) < 5:
            cnt += 1
            if cnt == 5:
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
    window_size = 21
    sigma = 8.0
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

def process_image_and_generate_fen(image):
    
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

    while a < 5:
        threshold_x = np.max(hough_Dx) * (a / 5.0)
        threshold_y = np.max(hough_Dy) * (a / 5.0)

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

# Initialize and load the model once
MODEL_PATH = "model_100.pth"  # FIXME Replace with your actual model path

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please ensure the path is correct.")

# model = load_model(MODEL_PATH)
# print("Model loaded successfully.")

def gradio_predict(image, fen_input, use_api_toggle):
    """Main function that handles both image processing and move prediction"""
    
    # Determine which FEN to use
    if fen_input and fen_input.strip():
        # Use direct FEN input
        fen = fen_input.strip()
        processing_method = "Direct FEN Input"
    else:
        # Process image to get FEN
        if image is None:
            return "**Error:** Please provide either an image or a FEN string."
        
        # FIXME: Set the global variable based on toggle
        global use_local_model
        use_local_model = not use_api_toggle  # Toggle inverts: True = use API, False = use local
        
        fen = process_image_and_generate_fen(image)
        processing_method = "API Model" if use_api_toggle else "Local Model"
        
        if fen.startswith("Failed"):
            return f"**Error:** {fen}"
    
    # Predict best move
    # FIXME: Currently always uses API for move prediction - modify logic as needed
    best_move = predict_best_move(fen, use_api=use_api_toggle)
    
    # Create output
    fen_encoded = urllib.parse.quote(fen)
    
    markdown_output = f"""
**Processing Method:** {processing_method}

**Generated FEN:** 
`{fen}`

**Predicted Best Move:** 
`{best_move}`

**Analysis Links:**
- [Analyze on Lichess](https://lichess.org/analysis/{fen_encoded})
- [Analyze on Chess.com](https://www.chess.com/analysis?fen={fen_encoded})
"""
    
    return markdown_output

# Create Gradio Interface with toggle
iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Image(type="pil", label="Upload Chessboard Image (Optional)"),
        gr.Textbox(label="Or Enter FEN Directly (Optional)", placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        gr.Checkbox(label="Use API for Processing", value=True)  # FIXME: Set default API usage preference
    ],
    outputs=gr.Markdown(label="Chess Analysis"),
    title="Chess Position Analyzer",
    description="Upload a chessboard image OR enter FEN notation. Toggle between API and local models. Get FEN notation and best move predictions.",
    examples=[
        [None, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", True],  # FIXME: Add real example FENs
        [None, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", False],  # FIXME: Add more varied examples
    ],
    allow_flagging="never"
)

# FIXME: Configure deployment settings for production
iface.launch(share=True)  # FIXME: Set share=False for production, configure auth if needed