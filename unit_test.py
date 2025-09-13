import torch
from app import CNNModel, labelIndex2Name

def test_model_forward_pass():
    model = CNNModel()
    dummy_input = torch.randn(1, 1, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 13)

def test_label_mapping():
    assert labelIndex2Name(1) == "K"
    assert labelIndex2Name(12) == "p"
