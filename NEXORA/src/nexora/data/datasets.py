import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from pathlib import Path


class TextFileDataset(Dataset):
    def __init__(self , path):
        path = Path(path)
        if not path.exists():
            raise ValueError("This path doesn't exist")
        if path.is_file():
            if path.suffix != ".txt":
                raise ValueError("The provided file is not .txt")
            self.path = [path]
        elif path.is_dir():
            self.path = sorted(path.glob("*.txt"))
            if len(self.path) == 0:
                raise ValueError("This folder doesn't have .txt files")
        else:
            raise ValueError("Somehting Wrong you're not passing a valid path")
    def __len__(self):
        return len(self.path)
    def __getitem__(self , index):
        path = self.path[index]
        text = path.read_text(encoding = "utf-8")
        return text

class TextCorpusDataset(Dataset):
    def __init__(self , path):
        path = Path(path)
        if not path.exists():
            raise ValueError("This path doesn't exist")
        if path.is_file():
            if path.suffix != ".txt":
                raise ValueError("The file is not .txt")
            self.path = path
    def __iter__(self):
        path = self.path
        with open(path , "r" , encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                yield text

class HuggingFaceDataset(Dataset):
    def __init__(self , dataset , column_text = "text"):
        self.dataset = dataset
        self.column_text = column_text

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        example = self.dataset[key]
        return example[self.column_text]
        