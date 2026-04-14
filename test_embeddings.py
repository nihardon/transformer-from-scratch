import torch
from torch.utils.data import DataLoader
from data.dataset import TokenDataset
from tokenizer.bpe import CharTokenizer
from model.embedding import EmbeddingModel

with open('data/input.txt', 'r') as f:
    text = f.read()

tokenizer = CharTokenizer(text)


encoded = torch.tensor(tokenizer.encode(text), dtype = torch.long)
dataset = TokenDataset(encoded, block_size=256)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

x = next(iter(dataloader))[0]

embedding_model = EmbeddingModel(tokenizer.vocab_size, 5, 256)
embeddings = embedding_model.forward(x)

print(embeddings)
assert embeddings.shape == (4,256,5)
