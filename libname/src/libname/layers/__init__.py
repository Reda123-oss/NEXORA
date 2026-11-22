import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self , d_model : int , eps : float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameters(torch.ones(d_model))
    def forward(self , x):
        rms = torch.sqrt(torch.mean(x**2 , dim = -1 , keepdim = True) + self.eps)
        rmsx = x/rms
        return self.weight*rmsx




