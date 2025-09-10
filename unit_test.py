import os
from PIL import Image
import pytest
from app import process_image_and_generate_fen

# Go up one level from the tests/ folder to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_process_real_chessboard_image():
    """Check that a real chessboard image produces a non-empty FEN string."""
    img_path = os.path.join(PROJECT_ROOT, "example1.png")
    img = Image.open(img_path)

    fen = process_image_and_generate_fen(img)
    print("\n[TEST OUTPUT] FEN from example1.png:", fen)

    assert isinstance(fen, str)
    assert len(fen) > 0   # just check something is returned

def test_process_blank_image_returns_failure():
    """Check that a blank image returns a failure message."""
    img = Image.new("RGB", (256, 256), color="white")
    fen = process_image_and_generate_fen(img)
    print("\n[TEST OUTPUT] FEN from blank image:", fen)

    assert "Failed" in fen

# Allow running this file directly (not just with pytest)
if __name__ == "__main__":
    # Run the two tests manually and exit
    test_process_real_chessboard_image()
    test_process_blank_image_returns_failure()
    print("\n✅ All tests completed.\n")
c