
import torch
import torch.nn as nn

from attention import Attention
from norms import RMSNorm, LayerNorm
from feedforward import GPT2MLP, SWIGLU


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Detect architecture
        self.use_rope = getattr(config, "rope_theta", None) is not None

        # Attention
        self.attention = Attention(config=config)

        # Normalization
        if self.use_rope:
            # Llama / Mistral
            self.norm1 = RMSNorm(config.d_model)
            self.norm2 = RMSNorm(config.d_model)

            # Feed-forward
            self.ffn = SWIGLU(config.d_model,config.ffn_hidden_dim)

        else:
            # GPT-2
            self.norm1 = LayerNorm(config.d_model)
            self.norm2 = LayerNorm(config.d_model)

            # Feed-forward
            self.ffn = GPT2MLP(config.d_model)

    def forward(self, x):

        # Attention sublayer
        normed_x = self.norm1(x)
        attention_output = self.attention(normed_x)

        # First residual connection
        x = x + attention_output

        # Feed-forward sublayer
        normed_x = self.norm2(x)
        ffn_output = self.ffn(normed_x)

        # Second residual connection
        x = x + ffn_output

        return x
