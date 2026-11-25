import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.blocks import TransformerBlock
from ..layers.norms import LayerNorm , RMSNorm
from ..layers.positional import Positional_encoding
from ..layers.embedding import Embedding



class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model

        self.embedding = Embedding(
            config.vocab_size,
            config.d_model
        )

        self.positional_encoding = Positional_encoding(
            config.context_length,
            config.d_model
        )

        self.norm = LayerNorm(config.d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size
        )

    def forward(self, inputs):

        if inputs.ndim == 2:
            # Token IDs: [B, T]
            embed = self.embedding(inputs)

        elif inputs.ndim == 3:
            # Already-computed embeddings: [B, T, D]
            if inputs.shape[-1] != self.d_model:
                raise ValueError(
                    f"Expected embedding dimension {self.d_model}, "
                    f"got {inputs.shape[-1]}."
                )

            embed = inputs

        else:
            raise ValueError(
                "Input must have shape [B,T] or [B,T,D]."
            )

        positions = self.positional_encoding(embed)

        x = embed + positions

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits
