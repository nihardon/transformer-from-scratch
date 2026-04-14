import torch
from torch.utils.data import DataLoader
from data.dataset import TokenDataset
from tokenizer.bpe import CharTokenizer

with open('data/input.txt', 'r') as f:
    text = f.read()

tokenizer = CharTokenizer(text)


encoded = torch.tensor(tokenizer.encode(text), dtype = torch.long)
dataset = TokenDataset(encoded, block_size=256)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

x, y = next(iter(dataloader))
print(f"x: {x}")
print(f"y: {y}")