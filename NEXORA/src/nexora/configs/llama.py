from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig, ConfigError


@dataclass(kw_only=True)
class LlamaConfig(BaseConfig):

    model_type: str = "llama"

    n_kv_heads: Optional[int] = None
    rope_theta: float = 10000.0

    ffn_hidden_dim: Optional[int] = None
    ffn_multiple_of: int = 256

    @property
    def ffn_dim(self) -> int:

        if self.ffn_hidden_dim is not None:
            return self.ffn_hidden_dim

        dim = int((8 / 3) * self.d_model)
        m = self.ffn_multiple_of

        return ((dim + m - 1) // m) * m

    def validate(self):

        super().validate()

        # FFN validation
        if self.ffn_hidden_dim is not None and self.ffn_hidden_dim <= 0:
            raise ConfigError(
                "The Feed Forward hidden dimension should be > 0"
            )

        if self.ffn_multiple_of <= 0:
            raise ConfigError("ffn_multiple_of must be > 0")

        # Grouped Query Attention validation
        if self.n_kv_heads is not None:

            if self.n_kv_heads <= 0:
                raise ConfigError("n_kv_heads must be > 0")

            if self.n_heads % self.n_kv_heads != 0:
                raise ConfigError(
                    "n_heads must be divisible by n_kv_heads"
                )

        # RoPE validation
        if self.head_dim % 2 != 0:
            raise ConfigError("head_dim must be even for RoPE")

        if self.rope_theta <= 0:
            raise ConfigError("rope_theta must be > 0")