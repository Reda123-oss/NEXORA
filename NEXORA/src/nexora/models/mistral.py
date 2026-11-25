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

    def forward(self, inputs):

        if inputs.ndim == 3:
            # User already provided embeddings: [B, T, D]
            if inputs.shape[-1] != self.d_model:
                raise ValueError("Invalid embedding dimension.")

            x = inputs

        elif inputs.ndim == 2:
            # User provided token IDs: [B, T]
            if inputs.dtype not in (torch.int32, torch.int64):
                raise ValueError("Token IDs must be integers.")

            x = self.embedding(inputs)

        else:
            raise ValueError(
                "Input must have shape [B,T] or [B,T,D]."
            )

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        # Vocabulary projection
        logits = self.lm_head(x)

        return logits
