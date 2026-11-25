from dataclasses import dataclass
from typing import Optional

from .base import ConfigError
from .mistral import MistralConfig


@dataclass(kw_only=True)
class MixtralConfig(MistralConfig):
    model_type = "mixtral"
    n_experts: int = 8
    experts_per_token: int = 2
    expert_ffn_dim: Optional[int] = None      
    aux_loss_coef: float = 0.01
    router_noise: float = 0.0

    @property
    def expert_dim(self) -> int:
        if self.expert_ffn_dim is None:
            return self.ffn_dim               
        return self.expert_ffn_dim           

    def validate(self):
        super().validate()
        if self.n_experts < 2:
            raise ConfigError("n_experts must be at least 2")
        if not (1 <= self.experts_per_token <= self.n_experts):
            raise ConfigError(
                f"experts_per_token ({self.experts_per_token}) must be between 1 and n_experts ({self.n_experts})")
        if self.expert_ffn_dim is not None and self.expert_ffn_dim <= 0:
            raise ConfigError("expert_ffn_dim must be positive")
        if self.aux_loss_coef < 0:
            raise ConfigError("aux_loss_coef cannot be negative")
        if self.router_noise < 0:
            raise ConfigError("router_noise cannot be negative")