import torch
from torch import Tensor
import torch.nn as nn

from .head import Head


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, block_size: int, dropout: float, device: str):
        super().__init__()
        # actual size of our heads should be the emb size split across all heads
        # this way the matrix math works cuz we just concat them all together
        actual_size = emb_size // num_heads
        # now we just do a bunch of heads honestly so we can pay attention to many different things
        self.heads = nn.ModuleList(
            [
                Head(emb_size, actual_size, block_size, dropout, device)
                for _ in range(num_heads)
            ]
        )
        # proj layer for residual connections
        # heads output is emb size so this is just lateral projection
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # pay attention and concat the logits
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # dropout and funny proj thing
        out = self.dropout(self.proj(out))
        return out
