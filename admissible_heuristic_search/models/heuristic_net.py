import torch
import torch.nn as nn

class HeuristicNet(nn.Module):
    """
    Generalized Feedforward Neural Network estimating cost-to-go.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(HeuristicNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss Function that penalizes overestimation of remaining steps
    more severely than underestimation to encourage learning an admissible heuristic.
    
    L(pred, target) = (target - pred)^2        if pred <= target
                      alpha * (pred - target)^2 if pred > target
    """
    def __init__(self, alpha: float = 100.0):
        super(AsymmetricLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.where(diff <= 0.0, diff ** 2, self.alpha * (diff ** 2))
        return loss.mean()
