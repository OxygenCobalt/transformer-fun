from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from .rep import Rep


class Head(nn.Module):
    def __init__(self, emb_size: int, head_size: int, block_size: int, dropout: float, device: str, rope: bool = False):
        super().__init__()
        self.key = Rep(emb_size, head_size, rope=rope, device=device)
        self.query = Rep(emb_size, head_size, rope=rope, device=device)
        self.value = Rep(emb_size, head_size, rope=False, device=device)
        self.dropout_p = dropout
        self.head_size = head_size

    def forward(self, x: Tensor) -> Tensor:
        # THE KEY VECTOR
        # [batch, time, head_size]
        # One side of the attention matrix.
        # "This is what I understand from the input in an abstract sense"
        k = self.key(x)
        # THE QUERY VECTOR
        # The other side of the attention matrix
        # [batch, time, head_size]
        # "This is what I want to know about the input in an abstract sense"
        q = self.query(x)  # B, T, H
        # The actual value, i.e the abstract representation of x
        v = self.value(x)
        # fast sdpa impl
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout_p if self.train else 0.0)
        # x is [batch, time, emb_size]
        # _, time, _ = x.shape
        # Dot them (rearrange k) so that we know what concepts should be paid
        # attention to. Note this goes forward and backward in time
        # [batch, time, head_size] x [batch, head_size, time] -> [batch, time, time]
        # wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        # Temporal masking, we don't want to look into the future.
        # Mask an existing buffer to avoid reallocs
        # wei = wei.masked_fill(self.tril[:time, :time] == 0, float("-inf"))  # pyright: ignore[reportIndexIssue]
        # Softmax for a proability distribution
        # wei = F.softmax(wei, dim=-1)
        # Dropout
        # Done here on the weight-level because ???
        # wei = self.dropout(wei)
        # The attention dot that lets us do the original calculation
        # (make all logits dependent on learnable weighted avg of our and all prior abstract things)
        # This is a communication mechanism based on a dag of token relations according to karpathy
        # out = wei @ v
        # out is dropped out in MHA
        # return out
