import torch.nn as nn 
import torch 
from ..layers.norms import RMSNorm
from ..layers.blocks import TransformerBlock


class Mistral(nn.Module):
    def __init__(self , config):
        super().__ini__()
        self.norm = RMSNorm(config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.lm_head = nn.Linear(config.d_model , config.vocab_size)

    def forward(self , x):
        for layer in self.layers:
            x = layer(x)
        norm = self.norm(x)
        linear = self.lm_head(norm)
        return linear