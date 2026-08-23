import torch
from torch import Tensor
import torch.nn as nn

from torch.nn import functional as F

THETA_BASE = 10_000

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, block_size: int, dropout: float, positions: str, device: str):
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
        self.position_embedding_table = None
        self.positions = positions
        if positions == "abs":
            self.position_embedding_table = nn.Embedding(block_size, emb_size, device=device)
            self.register_buffer("position", torch.arange(block_size, device=device))
        elif positions == "rope":
            i = torch.arange(self.head_size // 2, device=device)
            theta = THETA_BASE ** (-2 * i / self.head_size)
            pos = torch.arange(block_size, device=device)
            angles = pos[:, None] * theta[None, :]
            self.register_buffer("angles", angles)

    def forward(self, x: Tensor, eval_offset: int) -> Tensor:
        batch, time, channels = x.shape
        if self.positions == "abs":
            pos_emb = self.position_embedding_table(self.position[eval_offset:eval_offset + time])  # pyright: ignore[reportIndexIssue]
            x = x + pos_emb
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

        if self.positions == "rope":
            # rope is weird, effectively rather than boosting things by arbitrary
            # absolute positions so that the model learns positions in it's kqv
            # representations, instead we can instead force oscillators into each
            # pair within the kqv as well. these oscillators (ranging from very fast
            # oscillators in context and very slow ones that change little) provide
            # visual information, especially as at the dot product their angles cancel
            # out and thus yield roughly something like a relative position delta
            # (this is my understanding so far, may change?)
            #
            # by doing this we guide the model towards wanting to actually deal with
            # how to represent positions within the k/q representations it decides on
            #
            # slice fused qkvs down to first k/q and then also even/odd for a and b
            # this forms the pairs
            a = qkv[:, :, :2, :, 0::2]
            b = qkv[:, :, :2, :, 1::2]
            # get oscillators, we have to project them across the time axis
            # and then also across the whole head size in order to apply 1
            # oscillator per pair (i think)
            cos = torch.cos(self.angles[eval_offset:eval_offset + time])[None, :, None, None, :]
            sin = torch.sin(self.angles[eval_offset:eval_offset + time])[None, :, None, None, :]   # each (seq_len, head_dim/2)
            # rotate pairs across the curve from their assigned oscillator
            a_rot = a * cos - b * sin
            b_rot = a * sin + b * cos
            new_qkv = qkv.clone() # in-place assignment for autograd
            # update w/rotated points back to pos
            new_qkv[:, :, :2, :, 0::2] = a_rot
            new_qkv[:, :, :2, :, 1::2] = b_rot
            qkv = new_qkv

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
