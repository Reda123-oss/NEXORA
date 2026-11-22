import torch.nn as nn 
import torch 
from ..layers.norms import RMSNorm
from ..layers.blocks import TransformerBlock
from ..layers.embedding import Embedding



class Mistral(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model

        self.embedding = Embedding(
            config.vocab_size,
            config.d_model
        )

        self.norm = RMSNorm(config.d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size
        )

    def forward(self, input_ids=None, input_embeds=None):

        if (input_ids is None) == (input_embeds is None):
            raise ValueError("Provide exactly one input.")

        if input_ids is not None:

            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [B,T].")

            x = self.embedding(input_ids)

        else:

            if input_embeds.ndim != 3:
                raise ValueError("input_embeds must have shape [B,T,D].")

            if input_embeds.shape[-1] != self.d_model:
                raise ValueError("Invalid embedding dimension.")

            x = input_embeds

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits
