import os
import torch
import pytest

def test_real_model_loads():
    """Integration test: verify the saved weights produce correct output shape."""
    from app import load_model, MODEL_PATH  # import only when the test runs

    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model weights not available")

    model = load_model(MODEL_PATH)
    dummy_input = torch.randn(1, 1, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 13)
