import torch 
import torch.nn as nn
import torch.nn.functional as F



class GPT2MLP(nn.Module):
    def __init__(self , d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model , 4*d_model),
            nn.GELU(d_model),
            nn.Linear(4*d_model , d_model)
        )

    def forward(self , x):
        return self.mlp(x)


class SWILGLU(nn.Module):
    def __init__(self, d_model, ff_n):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ff_n, bias=False)
        self.up_proj = nn.Linear(d_model, ff_n, bias=False)
        self.down_proj = nn.Linear(ff_n, d_model, bias=False)

    def forward(self, x):
        gate_proj = F.silu(self.gate_proj(x))
        up_proj = self.up_proj(x)
        return self.down_proj(gate_proj * up_proj)


