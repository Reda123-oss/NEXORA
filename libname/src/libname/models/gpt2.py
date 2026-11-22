import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.blocks import TransformerBlock
from ..layers.norms import LayerNorm , RMSNorm
from ..layers.positional import Positional_encoding
from ..layers.embedding import Embedding


class GPT2(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.d_model = config.d_model 
        self.embedding = Embedding(config.vocab_size , config.d_model)
        self.positional_encoding = Positional_encoding(config.context_length , config.d_model)
        self.norm = LayerNorm(config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.lm_head = nn.Linear(config.d_model , config.vocab_size)


    def forward(self, input_ids = None , input_embeds = None):
        assert input_ids.shape == 2
        assert input_embeds.shape == 3
        
        if (input_ids == None) == (input_embeds == None):
            raise ValueError("You must provide one input exactly")
        elif len(input_embeds.shape == 3) and input_embeds.shape[-1] != self.d_model:
            raise ValueError(f"Shapes are mismatched d_model is {self.d_model} and x.shape[3] is {input_embeds.shape[3]}")
        else:
            if self.embedding is None:
                raise ValueError("Token Embedding is disabled")
            embed = self.embedding(input_ids)

        positions = self.positional_encoding(embed)
        x = embed + positions
        for layer in self.layers:
            x = layer(x)
        norm = self.norm(x)
        logits = self.lm_head(norm)
        return logits
