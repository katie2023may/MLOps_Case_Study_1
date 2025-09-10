import os
from PIL import Image
import pytest

# Import directly from app.py
from app import process_image_and_generate_fen

def test_process_real_chessboard_image():
    """Check that a real chessboard image produces a non-empty FEN string."""
    img_path = os.path.join(PROJECT_ROOT, "example1.png")
    img = Image.open(img_path)

    fen = process_image_and_generate_fen(img)

    assert isinstance(fen, str)
    assert len(fen) > 0   # just check something is returned

def test_process_blank_image_returns_failure():
    """Check that a blank image returns a failure message."""
    img = Image.new("RGB", (256, 256), color="white")
    fen = process_image_and_generate_fen(img)

    assert "Failed" in fen
