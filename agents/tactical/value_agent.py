import torch
import torch.nn as nn
import numpy as np

class ValueNet(nn.Module):
    """
    Neural Network heuristic for DAVI.
    Estimates the distance (number of primitive moves) from the current state to the solved state.
    """
    def __init__(self, input_dim=144, hidden_dim=256):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def state_to_tensor(state, device: str = "cpu") -> torch.Tensor:
    """
    Converts a single 2x2 cube state (24 integers) into a 144-dimensional one-hot tensor.
    """
    if isinstance(state, np.ndarray):
        state_torch = torch.from_numpy(state).long()
    elif isinstance(state, torch.Tensor):
        state_torch = state.long()
    else:
        state_torch = torch.tensor(state, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(state_torch, num_classes=6)
    return one_hot.view(1, -1).to(dtype=torch.float32, device=device)

def batch_to_tensor(states, device: str = "cpu") -> torch.Tensor:
    """
    Converts a batch of 2x2 cube states (shape: batch_size, 24) into a batch of one-hot tensors (shape: batch_size, 144).
    """
    if isinstance(states, np.ndarray):
        states_torch = torch.from_numpy(states).long()
    elif isinstance(states, torch.Tensor):
        states_torch = states.long()
    else:
        states_torch = torch.tensor(states, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(states_torch, num_classes=6)
    return one_hot.view(states_torch.shape[0], -1).to(dtype=torch.float32, device=device)


def save_model(model: nn.Module, filepath: str):
    """Saves the ValueNet model state dict."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Saved ValueNet weights to {filepath}")

def load_model(filepath: str, device: str = "cpu", hidden_dim: int = 256) -> ValueNet:
    """Loads a ValueNet model from the given path."""
    model = ValueNet(hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded ValueNet weights from {filepath}")
    return model
