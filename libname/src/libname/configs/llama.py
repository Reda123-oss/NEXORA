from dataclasses import dataclass
from base import BaseConfig , ConfigError

from typing import Optional



@dataclass(kw_only=True)
class LlamaConfig(BaseConfig):
    model_type = "llama"
    n_kv_heads : Optional[int] = None
    rope_theta  : float = 10000.0
    ffn_hidden_dim  : Optional[int] = None
    ffn_multiple_of = 256

    def __post_init__(self):
        return self.validate()

    @property
    def ffn_dim(self) -> int:
        if self.ffn_hidden_dim is not None:
            return self.ffn_hidden_dim

        dim = int((4/3)*2*self.d_model)
        m = self.ffn_multiple_of
        return ((dim + m-1)//2) *  m


    def validate(self):
        super().validate()
        if self.ffn_hidden_dim <= 0:
            raise ConfigError("The Feed Forward hidden dimension should be > 0")
        if self.n_heads % self.n_kv_heads != 0 and self.n_heads <= self.n_kv_heads:
            return ConfigError("This head dim should be less than n_heads and also dividable by it")
        if self.head_dim %2 != 0 :
            raise ConfigError("head_dim must be even for RoPE")
        if self.rope_theta <= 0:
            raise ConfigError("Theta in RoPE should be > 0")

