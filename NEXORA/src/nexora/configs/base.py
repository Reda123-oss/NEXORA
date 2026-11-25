from dataclasses import dataclass

class ConfigError(ValueError):
    pass

@dataclass(kw_only=True)
class BaseConfig:
    model_type : str
    vocab_size : int
    context_length : int
    d_model : int
    n_layers : int
    n_heads : int
    dropout : float = 0.0
    bias : bool = False
    tie_embeddings : bool = True 
    norm_eps : float = 1e-5
    init_std : float = 0.02

    def __post_init__(self):
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


    def validate(self):
        if self.vocab_size <= 0:
            raise ConfigError("Vocab size must be > 0")
        if self.context_length <= 0:
            raise ConfigError("Context length must be > 0")
        if self.d_model <=0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ConfigError("d_model or n_heads or n_layers must all be > 0")
        if self.d_model % self.n_heads != 0:
            raise ConfigError("d_model should be dividable by n_heads")
        if not 0.0 <= self.dropout < 1:
            raise ConfigError("Dropout must be between 0 and 1")
        

