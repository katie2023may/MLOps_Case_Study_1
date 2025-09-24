import torch
import torch.nn as nn
import torch.nn.functional as F

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