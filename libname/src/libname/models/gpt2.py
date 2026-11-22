import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.blocks import TransformerBlock
from ..layers.norms import LayerNorm , RMSNorm
from ..layers.positional import Positional_encoding


class GPT2(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.positional_encoding = Positional_encoding(config.context_length , config.d_model)
        self.norm = LayerNorm(config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in config.n_layers])
        self.lm_head = nn.Linear(config.d_model , config.vocab_size)


    def forward(self, x):
        positions = self.positional_encoding(x)
        positional_embedding = x + positions
        for layer in self.layers:
            positional_embedding = layer(positional_embedding)
        norm = self.norm(positional_embedding)
        linear = self.lm_head(norm)
        return linear
