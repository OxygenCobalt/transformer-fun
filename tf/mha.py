import torch
from torch import Tensor
import torch.nn as nn

from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, block_size: int, dropout: float, device: str):
        super().__init__()
        # actual size of our heads should be the emb size split across all heads
        # this way the matrix math works cuz we just concat them all together
        self.head_size = emb_size // num_heads
        self.emb_size = emb_size
        self.num_heads = num_heads
        # now we just do a bunch of heads honestly so we can pay attention to many different things
        self.qkv = nn.Linear(emb_size, 3 * emb_size, bias=False, device=device)
        # proj layer for residual connections
        # heads output is emb size so this is just lateral projection
        self.proj = nn.Linear(emb_size, emb_size, device=device)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(self, x: Tensor) -> Tensor:
        # pay attention and concat the logits
        # batch, time, channels = x.shape
        # qkv = self.qkv(x).reshape(
        #     batch,
        #     time,
        #     3,
        #     self.num_heads,
        #     self.head_size,
        # )
        # q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        # out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0)
        # out = out.transpose(1, 2).contiguous()
        # out = out.reshape(batch, time, channels)
        # # dropout and funny proj thing
        # out = self.dropout(self.proj(out))
        # return out

        batch, time, channels = x.shape
        # naively this is:
        # [batch, time, channels] -> [batch, time, channels * 3]
        # this requires us to reshape that last dimen into [3, num heads, head_size] which is equivalent
        qkv = self.qkv(x).reshape(
            batch,
            time,
            3,
            self.num_heads,
            self.head_size,
        )

        # split fused projection into q/k/v
        # sdpa wants [batch, head_size, time, num_heads]
        # but we have [batch, time, head_size, num_heads]
        # so we have to re-arrange accordingly for sdpa to work
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        # after that the output will also be [batch, head_size, time, num_heads] so we have
        # to both transpose this back to what we want [batch, time, head_size, num_heads] and
        # then shape it into our final projected logits
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch, time, channels)

        return self.dropout(self.proj(out))


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
