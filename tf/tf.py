import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from .block import Block


class Transformer(nn.Module):
    def __init__(
        self, n_layer: int, vocab_size: int, embed_size: int, heads: int, block_size: int, dropout: float, device: str, absolute: bool = False
    ):
        super().__init__()
        # basically the vocab logits here are the probability of the next token
        # given this token.
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size).to(device)
        self.position_embedding_table = None
        self.absolute = absolute
        if self.absolute:
            self.position_embedding_table = nn.Embedding(block_size, embed_size).to(device)
            self.register_buffer("position", torch.arange(block_size).to(device))
        self.blocks = nn.Sequential(
            *[
                Block(heads, embed_size, block_size, dropout, device)
                for _ in range(n_layer)
            ]
        )
        self.lm_head = nn.Linear(embed_size, vocab_size).to(device)
        self.block_size = block_size

    def forward(self, idx: int) -> Tensor:
        # we get the <token>th row of the embedding
        # [batch, time, channels]
        batch, time = idx.shape
        # token embeddings, simple enough
        tok_emb = self.token_embedding_table(idx)  # (batch, time, embed_dim)
        # apparently this is not usually what's done? like normally we would not
        # be manually passing a fixed timeline [1, 2, 3, 4, ..]
        # into the embedding table? is this where the rotary thing comes in??
        if self.absolute:
            pos_emb = self.position_embedding_table(self.position[:time])  # pyright: ignore[reportIndexIssue]
            # concat them both
            x = tok_emb + pos_emb
        else:
            x = tok_emb
        # pass through blocks
        x = self.blocks(x)
        # final lm_head to translate to logits/probs
        x = self.lm_head(x)
        logits = x  # (batch, time, vocab)
        return logits
