import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_class):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, num_class, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(num_class, dtype=torch.float32))
        self.temparature = 0.5
        
    def forward(self, x):
        return (x @ self.W + self.b)/self.temparature