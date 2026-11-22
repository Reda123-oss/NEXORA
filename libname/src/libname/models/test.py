
from .llama import Llama
from .mistral import Mistral
import torch
from ..configs.gpt2 import GPT2Config
from ..configs.llama import LlamaConfig
from .gpt2 import GPT2
from ..configs.mistral import MistralConfig


config = GPT2Config(
    vocab_size=100,
    context_length=16,
    d_model=32,
    n_layers=2,
    n_heads=4,
    dropout=0.1
)


import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
    print("CUDA version used by PyTorch:", torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"The current device is  : {device}")


gpt = GPT2(config)

x = torch.randint(0,99 , [70 , 16])

logits = gpt(x)
print(logits.shape)

config_2 = LlamaConfig(
    vocab_size=100,
    context_length=16,
    d_model=32,
    n_layers=2,
    n_heads=4,
    dropout=0.1,
    n_kv_heads = 2
)

llama = Llama(config_2)

x = torch.randint(0,99 , [70 , 16])

logits = llama(x)
print(logits.shape)

config_3 = MistralConfig(
    vocab_size=100,
    context_length=16,
    d_model=32,
    n_layers=2,
    n_heads=4,
    dropout=0.1,
    n_kv_heads = 2,
    sliding_window  = 4

)

mistral = Mistral(config_3)
logits = mistral(x)
print(logits.shape)