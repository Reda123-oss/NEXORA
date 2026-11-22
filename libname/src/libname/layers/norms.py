import torch
import torch.nn as nn
import torch.nn.functional as F

#LayerNorm : layernorm(x) = (x - mean(x))/square_root(var(x) + epsilon)
class LayerNorm(nn.Module):
    def __init__(self , d_model , eps = 1e-5 , bias : bool = False):
        super().__init__()
        self.eps = eps

        self.parameter = torch.Parameter(torch.ones(d_model))
        self.bias = bias

        if self.bias == True:
            self.bias = torch.Parameter(torch.ones(d_model))
        else:
            self.register_parameter("bias" , None)

    def forward(self , x):
        mean_x = x.mean(dim = -1 , keepdim = True)
        var_x = x.var(dim = -1 , keepdim = True , unbiased = False)

        x_normalized = (x - mean_x) / torch.sqrt(var_x + self.eps)
        output = self.parameter * x_normalized
        if self.bias == True:
            output = output + self.bias
            
        return output

# RMSNorm  : rmsnorm(x) =( x/rmsx )*y with rmsx = squareroot(1/n * sum(x^2) + epsilon)
class RMSNorm(nn.Module):
    def __init__(self , d_model : int , eps : float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameters(torch.ones(d_model))
    def forward(self , x):
        rms = torch.sqrt(torch.mean(x**2 , dim = -1 , keepdim = True) + self.eps)
        rmsx = x/rms
        return self.weight*rmsx






