import torch.nn as nn 
import torch.nn.functional as F 
import torch 
from ..layers.positional import rope
from ..layers.blocks import TransformerBlock
from ..layers.norms import RMSNorm


class Llama(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in config.n_layers
        ])
        self.block = TransformerBlock(config=config)
        self.lm_head = nn.Linear(config.d_model , config.vocab_size)

    def forward(self , x):
        for layer in self.layers:
            x = layer(x)
        norm = self.norm(x)
        linear = self.lm_head(norm)
        return linear
        
