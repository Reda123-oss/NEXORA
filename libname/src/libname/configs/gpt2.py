from dataclasses import dataclass
from base import BaseConfig , ConfigError



@dataclass(kw_only=True)
class GPT2Config(BaseConfig):
    model_type = "gpt2"
    ffn_dim = None

    def __post_init__(self):
        self.validate()

    @property
    def ffn_dim(self) -> int:
        if self.ffn_dim == None:
            return 4*self.d_model
        else:
            return self.ffn_dim

    def validate(self):
        super().validate()
        if self.ffn_dim < 0:
            raise ConfigError("the Feed forward Dimension should be > 0")
    

    
