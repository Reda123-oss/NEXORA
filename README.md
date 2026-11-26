<div align="center">

<!-- <picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/nexora-wordmark-dark.svg">
  <img src="assets/nexora-wordmark.svg" alt="NEXORA" width="420">
</picture> -->
  <img src="NEXORA/assets/nexora-mark-logo.svg" alt="NEXORA" width="420">


### Define it. Train it. Understand it.

**A PyTorch framework for building and training transformer language models without rewriting every block.**<br/>
GPT-2 · LLaMA · Mistral — and Mixtral on the way.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/status-alpha_0.1.0-orange?style=flat-square)

</div>

---

## Why NEXORA exists

I wrote GPT-2, LLaMA, Mistral and Mixtral from scratch, by hand, more than once. Every time it was the same ritual: attention, normalization, feed-forward, positional encoding, the training loop, the checkpoints. Different model, same blocks, rewritten again.

NEXORA is what I wished I had on day one. You describe the model in a **config**, NEXORA assembles the blocks, and you train it with one `Trainer`. When you want to open the box, every block is a small, readable PyTorch module that you can import and change.

```python
config = LlamaConfig(vocab_size=50257, context_length=256, d_model=256, n_layers=4, n_heads=4, n_kv_heads=2)
model  = Llama(config)
Trainer(model, num_epochs=5, optimizer=opt, loss=nn.CrossEntropyLoss()).fit(train_loader)
```

That is a grouped-query-attention LLaMA, training. No attention code written.

> **Honest status:** NEXORA is early (v0.1.0). The core path (**configs → models → trainer → sampling**) works end to end. Anything marked 🚧 in this README is scaffolded in the repository but not implemented yet. I would rather tell you that here than let you find out at 2 a.m.

---

## What's inside

| Area | What you get | Status |
|---|---|---|
| **Models** | `GPT2`, `Llama`, `Mistral` | ✅ |
| | `Mixtral` (sparse mixture-of-experts) | 🚧 config ready, model in progress |
| **Attention** | Multi-head, multi-query and **grouped-query** attention, **RoPE**, **sliding-window** masks, optional dropout | ✅ |
| **Blocks** | `TransformerBlock`, `RMSNorm`, `LayerNorm`, **SwiGLU**, GPT-2 MLP, learned positional embeddings | ✅ |
| **KV cache** | Growing `KVCache` at the attention-layer level | ✅ |
| | Rolling (fixed-size) cache, model-level cached generation | 🚧 |
| **Training** | `Trainer` with train/validation loops, per-epoch and best-model checkpoints | ✅ |
| | LR schedulers, callbacks, loggers, metrics, gradient clipping, mixed precision | 🚧 |
| **Sampling** | Greedy, temperature, top-k, top-p, repetition penalty | ✅ |
| **Data** | Tokenizer wrapper (Hugging Face compatible), text-file and Hugging Face dataset wrappers | ✅ |
| | Sequence packing, built-in data loaders | 🚧 |

---

## Install

NEXORA is a real Python package. **No cloning needed**: install it straight from GitHub with `pip`, and it becomes importable from any project in that environment.

```bash
pip install git+https://github.com/Reda123-oss/NEXORA.git
```

That is it. You need **Python 3.10+** and **git** on your machine. PyTorch (`torch>=2.0`), NumPy and tqdm are installed automatically.

```
GitHub repository ──▶ pip install git+… ──▶ installed in your environment ──▶ import anywhere
(src/ + pyproject.toml)                      (site-packages/nexora)            from nexora.models.gpt2 import GPT2
```

Use a virtual environment so NEXORA stays tidy:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install git+https://github.com/Reda123-oss/NEXORA.git
```

**Handy variations**

| I want to… | Command |
|---|---|
| Update to the latest version | `pip install --upgrade git+https://github.com/Reda123-oss/NEXORA.git` |
| Pin a release (reproducible runs) | `pip install git+https://github.com/Reda123-oss/NEXORA.git@v0.1.0` |
| Try a branch | `pip install git+https://github.com/Reda123-oss/NEXORA.git@branch-name` |
| Depend on it in `requirements.txt` | `nexora @ git+https://github.com/Reda123-oss/NEXORA.git` |
| Use it in Colab or Kaggle | `!pip install git+https://github.com/Reda123-oss/NEXORA.git` |
| Remove it | `pip uninstall nexora` |

Need a specific CUDA build of PyTorch? Install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/), then run the NEXORA install. For the quickstart below you also need Hugging Face tokenizers (it downloads the GPT-2 tokenizer on first run, so it needs internet once):

```bash
pip install transformers
```

Now, in any project:

```python
from nexora.models.gpt2 import GPT2
```

**Check that everything works:**

```bash
python -m nexora.tests.test
```

It builds a small GPT-2, LLaMA and Mistral and prints the device, then the output shape of each model. You should see three lines reading `torch.Size([70, 16, 100])`.

---

## Quickstart: train a model in five steps

Put some text in a file called `data.txt` (any plain text: a book, your notes, a code dump). A few thousand words is enough to watch the loss move, a few megabytes is better. Then save this as `train.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from nexora.configs.llama import LlamaConfig
from nexora.models.llama import Llama
from nexora.training.trainer import Trainer

# 1. Tokenize your text -------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
ids = tokenizer.encode(open("data.txt", encoding="utf-8").read())

split = int(0.9 * len(ids))
train_ids, val_ids = ids[:split], ids[split:]

# 2. Turn tokens into (input, target) pairs -----------------------------------
class TokenDataset(Dataset):
    def __init__(self, ids, context_length):
        self.ids, self.T = ids, context_length

    def __len__(self):
        return (len(self.ids) - 1) // self.T

    def __getitem__(self, i):
        s = i * self.T
        x = torch.tensor(self.ids[s : s + self.T], dtype=torch.long)
        y = torch.tensor(self.ids[s + 1 : s + self.T + 1], dtype=torch.long)  # shifted by one
        return x, y

CONTEXT = 128
train_loader = DataLoader(TokenDataset(train_ids, CONTEXT), batch_size=8, shuffle=True)
val_loader   = DataLoader(TokenDataset(val_ids,   CONTEXT), batch_size=8)

# 3. Describe the model -------------------------------------------------------
config = LlamaConfig(
    vocab_size=len(tokenizer),   # always match your tokenizer
    context_length=CONTEXT,
    d_model=256,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,                # grouped-query attention: 4 query heads share 2 KV heads
    ffn_hidden_dim=512,
    dropout=0.1,
)
model = Llama(config)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

# 4. Train --------------------------------------------------------------------
trainer = Trainer(
    model=model,
    num_epochs=5,
    optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4),
    loss=nn.CrossEntropyLoss(),
    checkpoint=True,
    best_checkpoint=True,
)
trainer.fit(train_loader, val_loader)
```

```bash
python train.py
```

You will see one line per epoch (your numbers will differ):

```
Training Started...

Epoch 1/5 | Train Loss: 6.8123 | Val Loss: 6.2201
Epoch 2/5 | Train Loss: 5.9120 | Val Loss: 5.8744
...
```

Checkpoints land in `checkpoints/epochs/` (one per epoch) and `checkpoints/best/best_checkpoint.pt` (lowest validation loss).

**5. Generate text** (see [Sampling](#sampling-parameters) for every knob):

```python
from nexora.generations.sampling import sample_next_token

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=60, **sampling):
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    for _ in range(max_new_tokens):
        logits = model(ids[:, -CONTEXT:])[:, -1, :]        # logits for the next token
        next_id = sample_next_token(logits, input_ids=ids, **sampling)
        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].tolist())

print(generate(model, tokenizer, "Once upon a time", temperature=0.8, repetition_penalty=1.1))
```

---

## Pick your architecture

Every model follows the same pattern: **a config class describes it, a model class builds it.**

| I want… | Config | Model | You get |
|---|---|---|---|
| A classic baseline | `GPT2Config` | `GPT2` | Learned positions, LayerNorm, GELU MLP |
| A modern open-model recipe | `LlamaConfig` | `Llama` | RoPE, RMSNorm, SwiGLU, optional grouped-query attention |
| Long context at lower cost | `MistralConfig` | `Mistral` | Everything in LLaMA, plus **sliding-window** attention |
| Sparse experts | `MixtralConfig` | 🚧 | Config is validated, model is being built |

```python
from nexora.configs.gpt2    import GPT2Config
from nexora.configs.llama   import LlamaConfig
from nexora.configs.mistral import MistralConfig
from nexora.models.gpt2     import GPT2
from nexora.models.llama    import Llama
from nexora.models.mistral  import Mistral

common = dict(vocab_size=50257, context_length=256, d_model=256, n_layers=4, n_heads=4)

gpt     = GPT2(GPT2Config(**common))
llama   = Llama(LlamaConfig(**common, n_kv_heads=2))
mistral = Mistral(MistralConfig(**common, n_kv_heads=2, sliding_window=64))
```

All config arguments are **keyword-only**. Invalid combinations (for example a `d_model` that is not divisible by `n_heads`) raise a clear `ConfigError` the moment you build the config, not twenty minutes into training.

**What the models accept.** Token IDs of shape `[batch, time]` (integer tensors), or already-computed embeddings of shape `[batch, time, d_model]`. They return logits of shape `[batch, time, vocab_size]`. The loss is computed outside the model, in the `Trainer`.

---

## Configuration reference

### Shared by every model (`BaseConfig`)

| Parameter | Type | Default | What it does | Status |
|---|---|---|---|---|
| `vocab_size` | int | **required** | Number of tokens. Set it to `len(tokenizer)` | ✅ |
| `context_length` | int | **required** | Maximum sequence length. For **GPT-2** it is the size of the learned position table: longer inputs fail. LLaMA/Mistral use RoPE and have no hard limit, but you should still train at the length you plan to use | ✅ |
| `d_model` | int | **required** | Width of the model (embedding size). **Must be divisible by `n_heads`** | ✅ |
| `n_layers` | int | **required** | Number of transformer blocks | ✅ |
| `n_heads` | int | **required** | Number of attention heads. `head_dim = d_model / n_heads` | ✅ |
| `dropout` | float, `0 ≤ p < 1` | `0.0` | Dropout on attention probabilities, active only in training mode | ✅ |
| `bias` | bool | `False` | Adds a bias to the attention output projection (the Q/K/V projections are always bias-free) | ✅ |
| `tie_embeddings` | bool | `True` | Share input embedding and output head weights | 🚧 accepted, not wired yet |
| `norm_eps` | float | `1e-5` | Normalization epsilon | 🚧 accepted, norms use `1e-5` for now |
| `init_std` | float | `0.02` | Weight initialization scale | 🚧 accepted, not wired yet |

### GPT-2 (`GPT2Config`)

Adds `ffn_dim: int | None = None` (🚧 reserved: the MLP is currently fixed at `4 × d_model`).

### LLaMA (`LlamaConfig`)

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `n_kv_heads` | int or `None` | `None` | Number of key/value heads. `None` = same as `n_heads` (standard multi-head attention). Must divide `n_heads`. Fewer KV heads means a smaller KV cache and faster inference |
| `rope_theta` | float | `10000.0` | Turns RoPE on (a LLaMA-family config always has it). 🚧 The value itself is not passed to the rotation yet, which currently uses the standard `10000`. `head_dim` must be even |
| `ffn_hidden_dim` | int or `None` | `None` | Hidden size of the SwiGLU feed-forward. `None` = automatic (see below) |
| `ffn_multiple_of` | int | `256` | The automatic size is rounded **up** to a multiple of this |

**The automatic feed-forward size** is `⌊8/3 × d_model⌋`, rounded up to a multiple of `ffn_multiple_of`. For large models that is exactly what you want. For tiny models it overshoots (with `d_model=32` you get 256), so **set `ffn_hidden_dim` explicitly when you experiment small.**

**Choosing `n_kv_heads`:**

| Value | Attention type | Trade-off |
|---|---|---|
| `n_kv_heads = n_heads` (or `None`) | Multi-head | Best quality baseline, biggest KV cache |
| `1 < n_kv_heads < n_heads` | **Grouped-query** | Near-baseline quality, KV cache shrinks by `n_heads / n_kv_heads` |
| `n_kv_heads = 1` | Multi-query | Smallest cache, most aggressive sharing |

### Mistral (`MistralConfig`, extends LLaMA)

| Parameter | Type | Default | What it does | Status |
|---|---|---|---|---|
| `sliding_window` | int | `4096` | Each token attends only to itself and the previous `sliding_window - 1` tokens. Smaller windows are cheaper; information still flows further through stacked layers | ✅ |
| `prefill_chunk_size` | int or `None` | `None` (= `sliding_window`) | Prompt chunk size for cached generation | 🚧 reserved |

### Mixtral (`MixtralConfig`, extends Mistral) 🚧

These are validated today and will drive the model once the MoE layer lands.

| Parameter | Default | What it will do |
|---|---|---|
| `n_experts` | `8` | Number of expert feed-forward networks (≥ 2) |
| `experts_per_token` | `2` | How many experts process each token (1 to `n_experts`) |
| `expert_ffn_dim` | `None` | Hidden size per expert (`None` = the LLaMA feed-forward size) |
| `aux_loss_coef` | `0.01` | Weight of the load-balancing loss |
| `router_noise` | `0.0` | Noise added to router logits during training |

---

## Training

```python
Trainer(
    model,                    # any NEXORA model (or your own nn.Module returning [B, T, V] logits)
    num_epochs,               # int
    optimizer,                # any torch optimizer
    loss,                     # e.g. nn.CrossEntropyLoss()
    checkpoint=False,         # save every epoch
    best_checkpoint=False,    # keep the best validation model (requires val_loader)
    device=...,               # defaults to CUDA when available, otherwise CPU
)
trainer.fit(train_loader, val_loader=None)
trainer.evaluate(val_loader)  # returns the average validation loss
```

| Argument | What it does |
|---|---|
| `model` | The model to train. It is moved to `device` for you |
| `num_epochs` | Number of full passes over `train_loader` |
| `optimizer` | Build it yourself: `torch.optim.AdamW(model.parameters(), lr=3e-4)` |
| `loss` | Called as `loss(logits.view(B*T, V), targets.view(B*T))`, so `nn.CrossEntropyLoss()` is the right choice |
| `checkpoint` | `True` writes `checkpoints/epochs/checkpoint_epoch_N.pt` after every epoch |
| `best_checkpoint` | `True` writes `checkpoints/best/best_checkpoint.pt` whenever validation loss improves. Needs a `val_loader` |
| `device` | A `torch.device`. Override it to force CPU or pick a GPU |

**The data contract.** Your `DataLoader` must yield `(x, y)` pairs of `torch.long` tensors, both `[batch, time]`, where `y` is `x` shifted left by one token. The `TokenDataset` in the quickstart does exactly this.

**Checkpoint contents.** Regular checkpoints store `model`, `optimizer` and `epoch`. The best checkpoint stores `model`, `optimizer` and `val_loss`.

**Resume or reload a model:**

```python
from nexora.training.checkpoints import load_checkpoint

epoch = load_checkpoint(
    "checkpoints/epochs/checkpoint_epoch_3.pt",
    model, optimizer, device="cpu",
)
print(f"Loaded epoch {epoch}")
```

> `fit()` always counts epochs from 1, so a resumed run will overwrite `checkpoint_epoch_1.pt`, `2`, and so on. Move your old `checkpoints/` folder aside before resuming, and set `num_epochs` to the number of epochs you still want.

### Knobs that matter

| If you see… | Try |
|---|---|
| Loss does not move | Lower `lr` (`1e-4`), check that `vocab_size` matches the tokenizer, check that targets are shifted by one |
| Loss drops but validation loss climbs (overfitting) | More data, higher `dropout` (`0.1`–`0.2`), fewer `n_layers` / smaller `d_model` |
| Training loss and validation loss are both high | A bigger model (`d_model`, `n_layers`), more epochs, or a higher `lr` |
| `CUDA out of memory` | Lower the `DataLoader` `batch_size`, shorten `context_length`, shrink `d_model`, use `n_kv_heads < n_heads` for inference |
| Training is slow | Use a GPU, raise `batch_size` until memory is nearly full, set `num_workers` in the `DataLoader` |

Sensible starting points for a first run: `AdamW`, `lr` between `1e-4` and `3e-4`, `dropout` `0.0`–`0.1`.

---

## Sampling parameters

`sample_next_token` turns the logits of the last position into the next token.

```python
from nexora.generations.sampling import sample_next_token

next_id = sample_next_token(
    logits,                     # [batch, vocab]
    input_ids=ids,              # tokens so far (needed for repetition_penalty)
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)
```

| Parameter | Default | Effect |
|---|---|---|
| `temperature` | `1.0` | Below 1 = more focused and predictable, above 1 = more random. Must be `> 0` |
| `top_k` | `None` | Sample only among the `k` most likely tokens |
| `top_p` | `None` | Sample only among the smallest set of tokens whose probability adds up to `p` (nucleus sampling). Valid range: `0 < p ≤ 1` |
| `repetition_penalty` | `1.0` | Values above 1 discourage tokens that already appeared. `1.0` disables it |
| `do_sample` | `True` | `False` = greedy decoding (always the single most likely token) |

Order of operations: repetition penalty → (greedy exit) → temperature → top-k → top-p → sample.

> **Performance note:** `top_k` and `top_p` are written for clarity, with plain Python loops over the vocabulary. They are perfect for learning and small vocabularies, and noticeably slow on a 50k-token vocabulary. For big vocabularies, start with `temperature` and `repetition_penalty` only. Vectorized versions are on the roadmap.

---

## Use the blocks directly

Everything NEXORA assembles for you is a normal `nn.Module` you can pull out and rearrange.

**Build your own model from NEXORA's blocks:**

```python
import torch.nn as nn
from nexora.configs.llama import LlamaConfig
from nexora.layers.blocks import TransformerBlock
from nexora.layers.embedding import Embedding
from nexora.layers.norms import RMSNorm

class MyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed  = Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm   = RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
```

It plugs straight into `Trainer`.

**How `TransformerBlock` chooses its style.** If the config has a `rope_theta` (LLaMA / Mistral), the block is **LLaMA-style**: RMSNorm + RoPE attention + SwiGLU. Otherwise (GPT-2) it is **GPT-2-style**: LayerNorm + GELU MLP. Pre-norm residual wiring in both cases.

**Use attention with a KV cache:**

```python
import torch
from nexora.configs.llama import LlamaConfig
from nexora.layers.attention import Attention
from nexora.cache.kv_cache import KVCache

cfg   = LlamaConfig(vocab_size=1000, context_length=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
attn  = Attention(cfg).eval()
cache = KVCache()

with torch.no_grad():
    y = attn(torch.randn(1, 10, cfg.d_model), cache=cache)   # prefill 10 tokens
    y = attn(torch.randn(1, 1,  cfg.d_model), cache=cache)   # then one token at a time

print(cache.get_seq_length())   # 11
cache.reset()                   # start a new sequence
```

The cache stores the original (un-repeated) KV heads, so grouped-query attention gets its full memory saving.

---

## Project layout

```
NEXORA/
├── pyproject.toml
├── assets/                     # logo files: wordmark, mark and icon (SVG + PNG)
├── examples/
│   └── unit.py                 # author's training script (reads a local CSV: adapt the path before running)
└── src/nexora/
    ├── configs/                # base, gpt2, llama, mistral, mixtral
    ├── models/                 # gpt2 ✅  llama ✅  mistral ✅  mixtral 🚧  registry 🚧  base 🚧
    ├── layers/                 # attention ✅  blocks ✅  norms ✅  feedforward ✅
    │                           # positional (RoPE, learned) ✅  embedding ✅  moe 🚧
    ├── cache/                  # kv_cache ✅  rolling_cache 🚧
    ├── training/               # trainer ✅  checkpoints ✅
    │                           # optim 🚧  schedulers 🚧  callbacks 🚧  loggers 🚧  metrics 🚧
    ├── generations/            # sampling ✅
    ├── data/                   # tokenizer ✅  datasets ✅  loader 🚧  packing 🚧
    └── tests/                  # smoke test: python -m nexora.tests.test
```

---

## Troubleshooting

<details>
<summary><b><code>ConfigError: d_model should be dividable by n_heads</code></b></summary>

`head_dim = d_model / n_heads` must be a whole number. For LLaMA/Mistral it must also be **even** (RoPE rotates pairs of values), and `n_heads` must be divisible by `n_kv_heads`.
</details>

<details>
<summary><b><code>IndexError: index out of range in self</code> (embedding)</b></summary>

A token ID is `>= vocab_size`. Build the config with `vocab_size=len(tokenizer)`, not a guess.
</details>

<details>
<summary><b>GPT-2 fails on a long sequence</b></summary>

GPT-2 uses a learned position table of size `context_length`. Keep every training and generation window `<= context_length`.
</details>

<details>
<summary><b><code>ValueError: Expected start_pos=…</code> when using KVCache</b></summary>

The cache enforces strictly sequential positions. Call `cache.reset()` before starting a new sequence, and do not skip or repeat chunks.
</details>

<details>
<summary><b><code>ValueError: A validation loader is required for best checkpoint saving</code></b></summary>

`best_checkpoint=True` needs `trainer.fit(train_loader, val_loader)`.
</details>

<details>
<summary><b>Loss is <code>nan</code></b></summary>

Lower the learning rate first. Then check your data for empty or corrupted text.
</details>

---

## Roadmap

- [ ] **Mixtral**: sparse mixture-of-experts layer (router, top-k experts, load-balancing loss) and the `Mixtral` model
- [ ] Model-level cached generation and a rolling (sliding-window) cache
- [ ] Model registry: build any model from a config name or a JSON file
- [ ] LR schedulers (warmup + cosine), gradient clipping, mixed precision, gradient accumulation
- [ ] Callbacks, loggers and metrics (perplexity, throughput)
- [ ] Sequence packing and built-in data loaders
- [ ] Vectorized sampling
- [ ] Wire the remaining config fields (`rope_theta` value, `tie_embeddings`, `norm_eps`, `init_std`, GPT-2 `ffn_dim`)
- [ ] Distributed training (ZeRO-style optimizer, gradient and parameter sharding)

---

## Contributing

Issues, ideas and pull requests are welcome, especially for anything under 🚧.

To work on NEXORA itself, clone it and install in editable mode, so your changes apply immediately:

```bash
git clone https://github.com/Reda123-oss/NEXORA.git
cd NEXORA
pip install -e .
```
 If you are adding a block, keep it small, readable and tested against a plain PyTorch reference: NEXORA is meant to be a framework you can *understand*, not just use.

## Author

**Reda Malki**: building language models from first principles.
[GitHub](https://github.com/Reda123-oss) · [Medium](https://medium.com/@redamalki2003) · [Hugging Face](https://huggingface.co/reda12Malki) · [LinkedIn](https://linkedin.com/in/reda-malki)

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/nexora-mark-dark.svg">
  <img src="assets/nexora-mark.svg" alt="NEXORA mark" width="56">
</picture>

*If NEXORA saves you from rewriting one more attention block, it did its job.*

</div>