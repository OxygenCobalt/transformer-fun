import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from .block import Block


class Transformer(nn.Module):
    def __init__(
        self, n_layer: int, vocab_size: int, embed_size: int, heads: int, block_size: int, dropout: float, positions: str, device: str
    ):
        super().__init__()
        # basically the vocab logits here are the probability of the next token
        # given this token.
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size).to(device)
        self.blocks = nn.ModuleList(
            [
                Block(heads, embed_size, block_size, dropout, positions, device)
                for _ in range(n_layer)
            ]
        )
        self.lm_head = nn.Linear(embed_size, vocab_size).to(device)
        self.block_size = block_size

    def forward(self, idx: Tensor, eval_offset: int) -> Tensor:
        # we get the <token>th row of the embedding
        # [batch, time, channels]
        # token embeddings, simple enough
        x = self.token_embedding_table(idx)  # (batch, time, embed_dim)
        # pass through blocks
        for block in self.blocks:
            x = block(x, eval_offset)
        # final lm_head to translate to logits/probs
        x = self.lm_head(x)
        # logits = x  # (batch, time, vocab)
        return x
