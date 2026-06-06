import torch
import torch.nn as nn

from .ff import FeedForward
from .mha import MultiHeadAttention


class Block(nn.Module):
    def __init__(self, num_heads, emb_size, block_size, dropout, device):
        super().__init__()
        # attention
        self.sa = MultiHeadAttention(num_heads, emb_size, block_size, dropout, device)
        # ff to analyze
        self.ff = FeedForward(emb_size, dropout)
        # layer norms. i understand they help stabilize variance but i dont know why this helps
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # skip connections! we add here but proj inside the layers actually
        # so we preserve the pre-activation and then add that to the projected output,
        # which helps with deep learning
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
