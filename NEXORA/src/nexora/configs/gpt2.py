
from dataclasses import dataclass
from .base import BaseConfig, ConfigError


@dataclass(kw_only=True)
class GPT2Config(BaseConfig):
    model_type: str = "gpt2"
    ffn_dim: int | None = None

    @property
    def hidden_dim(self) -> int:
        if self.ffn_dim is None:
            return 4 * self.d_model
        return self.ffn_dim

    def validate(self):
        super().validate()

        if self.ffn_dim is not None and self.ffn_dim <= 0:
            raise ConfigError(
                "The Feed Forward Dimension should be > 0"
            )
