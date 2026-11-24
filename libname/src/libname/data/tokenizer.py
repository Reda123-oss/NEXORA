import torch.nn as nn
import torch



class Tokenizer():
    def __init__(self , tokenizer):
        self.tokenizer = tokenizer
        # self.data = data
    def encode(self , data):
        return self.tokenizer.encode(data)
    def decode(self , tokens):
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def unk_token_id(self):
        return self.tokenizer.unk_token_id