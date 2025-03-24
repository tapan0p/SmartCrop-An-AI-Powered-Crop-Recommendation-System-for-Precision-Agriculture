import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self,input_dim,num_class):
        super().__init__()
        self.input_dim = input_dim
        self.num_class = num_class 
        # weights
        self.W = torch.randn(input_dim,num_class,dtype=torch.float32,requires_grad=True)
        self.b = torch.zeros(num_class,dtype=torch.float32,requires_grad=True)

    def forward(self,x):
        return x@self.W+self.b