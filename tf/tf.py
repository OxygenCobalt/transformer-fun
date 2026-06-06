import torch
import torch.nn as nn
from torch.nn import functional as F

from .block import Block


class Transformer(nn.Module):
    def __init__(
        self, n_layer, vocab_size, embed_size, heads, block_size, dropout, device
    ):
        super().__init__()
        # basically the vocab logits here are the probability of the next token
        # given this token.
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size).to(device)
        self.position_embedding_table = nn.Embedding(block_size, embed_size).to(device)
        self.blocks = nn.Sequential(
            *[
                Block(heads, embed_size, block_size, dropout, device)
                for _ in range(n_layer)
            ]
        )
        self.lm_head = nn.Linear(embed_size, vocab_size).to(device)
        self.block_size = block_size
        self.register_buffer("position", torch.arange(block_size).to(device))

    def forward(self, idx):
        # we get the <token>th row of the embedding
        # [batch, time, channels]
        batch, time = idx.shape
        # token embeddings, simple enough
        tok_emb = self.token_embedding_table(idx)  # (batch, time, embed_dim)
        # apparently this is not usually what's done? like normally we would not
        # be manually passing a fixed timeline [1, 2, 3, 4, ..]
        # into the embedding table? is this where the rotary thing comes in??
        pos_emb = self.position_embedding_table(self.position[:time])  # pyright: ignore[reportIndexIssue]
        # concat them both
        x = tok_emb + pos_emb
        # pass through blocks
        x = self.blocks(x)
        # final lm_head to translate to logits/probs
        x = self.lm_head(x)
        logits = x  # (batch, time, vocab)
        return logits
