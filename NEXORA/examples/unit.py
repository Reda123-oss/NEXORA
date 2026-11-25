import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ..configs.gpt2 import GPT2Config
from ..models.gpt2 import GPT2
from ..training.trainer import Trainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load and clean data
df = pd.read_csv(
    r"C:\Users\lenovo\OneDrive\Bureau\LLM_Library\validation.csv"
)

df = df.dropna(subset=["text"])
df = df.sample(n=min(3000, len(df)), random_state=42)
df["text"] = df["text"].astype(str)

# 2. Load tokenizer and tokenize text
tokenizer = AutoTokenizer.from_pretrained("gpt2")

all_text = " ".join(df["text"].tolist())
tokens = tokenizer.encode(all_text)

# 3. Dataset
class MyDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length

        x = self.tokens[start : start + self.context_length]
        y = self.tokens[start + 1 : start + self.context_length + 1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )

# 4. DataLoader
dataset = MyDataset(tokens, context_length=128)

train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# 5. Model
config = GPT2Config(
    vocab_size=len(tokenizer),
    context_length=128,
    d_model=32,
    n_layers=2,
    n_heads=4,
    dropout=0.1
)

gpt = GPT2(config)

# 6. Optimizer and loss
optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 7. Trainer
training = Trainer(
    model=gpt,
    num_epochs=10,
    optimizer=optimizer,
    loss=criterion,
    checkpoint=True,
    best_checkpoint=False,
    device=device
)

# 8. Start training
training.fit(train_loader)