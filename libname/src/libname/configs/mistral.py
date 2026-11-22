from dataclasses import dataclass
from .llama import LlamaConfig , ConfigError
from typing import Optional



@dataclass(kw_only=True)
class MistralConfig(LlamaConfig):
    model_type : str = "mistral"
    sliding_window : int = 4096
    prefill_chunk_size : int = None

    @property
    def chunk_size(self) -> int:
        if self.prefill_chunk_size is None:
            return self.sliding_window
        return self.prefill_chunk_size

    def validate(self):
        super().validate()
        if self.sliding_window <= 0:
            raise ConfigError("The sliding Window should be > 0")
        if self.chunk_size <= 0:
            raise ConfigError("The chunk, size should either be > 0 or equal to Window Size")
