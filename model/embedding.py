import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_embedding = nn.Embedding(num_embeddings=block_size, embedding_dim=d_model)
    
    def forward(self, x):        
        length = x.shape[1]
        
        tokens = self.token_embedding(x)
        positions = torch.arange(length, device = x.device)
        positions = self.positional_embedding(positions)

        final_embeddings = tokens + positions
        return final_embeddings




