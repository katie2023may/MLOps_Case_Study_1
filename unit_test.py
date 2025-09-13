import os
import torch
import pytest
from app import load_model, MODEL_PATH

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model weights not available")
def test_real_model_loads():
    """Integration test: verify the saved weights produce correct output shape."""
    model = load_model(MODEL_PATH)
    dummy_input = torch.randn(1, 1, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 13)
