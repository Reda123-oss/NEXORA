import torch.nn as nn 
import torch 
from ..layers.norms import RMSNorm
from ..layers.blocks import TransformerBlock
from ..layers.embedding import Embedding


class Mistral(nn.Module):
    def __init__(self , config):
        super().__ini__()
        self.embedding = Embedding(config.vocab_size, config.d_model) 
        self.norm = RMSNorm(config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.lm_head = nn.Linear(config.d_model , config.vocab_size)

    def forward(self , input_ids = None , input_embeds = None):
        if (input_ids == None) == (input_embeds == None):
            raise ValueError("You must provide one input exactly")
        elif len(input_embeds.shape == 3) and input_embeds.shape[-1] != self.d_model:
            raise ValueError(f"Shapes are mismatched d_model is {self.d_model} and x.shape[3] is {input_embeds.shape[3]}")
        else:
            if self.embedding is None:
                raise ValueError("Token Embedding is disabled")
            embed = self.embedding(input_ids)
        for layer in self.layers:
            embed = layer(embed)
        norm = self.norm(embed)
        output = self.lm_head(norm)
        return output