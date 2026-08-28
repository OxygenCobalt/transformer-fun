import torch
from torch import Tensor
import torch.nn as nn
from lm.config import Config

from torch.nn import functional as F

THETA_BASE = 10_000

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # actual size of our heads should be the emb size split across all heads
        # this way the matrix math works cuz we just concat them all together
        self.head_size = config.hyperparams.embed_size // config.hyperparams.q_heads
        self.embed_size = config.hyperparams.embed_size
        self.q_heads = config.hyperparams.q_heads
        self.kv_heads = config.hyperparams.kv_heads
        self.q_size = self.q_heads * self.head_size
        self.kv_size = self.kv_heads * self.head_size
        # now we just do a bunch of heads honestly so we can pay attention to many different things
        self.qkv = nn.Linear(
            config.hyperparams.embed_size,
             self.q_size + self.kv_size + self.kv_size,
            bias=False,
            device=config.device
        )
        # proj layer for residual connections
        # heads output is emb size so this is just lateral projection
        self.proj = nn.Linear(config.hyperparams.embed_size, config.hyperparams.embed_size, device=config.device)
        self.dropout = nn.Dropout(config.hyperparams.dropout)
        self.dropout_p = config.hyperparams.dropout
        self.positions = config.positions
        assert self.embed_size % self.q_heads == 0
        assert self.q_heads % self.kv_heads == 0
        if self.positions == "rope":
            assert self.head_size % 2 == 0
            i = torch.arange(self.head_size // 2, device=config.device)
            theta = THETA_BASE ** (-2 * i / self.head_size)
            pos = torch.arange(config.hyperparams.block_size, device=config.device)
            angles = pos[:, None] * theta[None, :]
            self.register_buffer("angles", angles)

    def forward(self, x: Tensor, eval_offset: int) -> Tensor:
        batch, time, channels = x.shape
        # naively this is:
        # [batch, time, channels] -> [batch, time, all_heads * channels] -> [batch, time, num_heads * head_size]
        q, k, v = self.qkv(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(batch, time, self.q_heads, self.head_size)
        k = k.reshape(batch, time, self.kv_heads, self.head_size)
        v = v.reshape(batch, time, self.kv_heads, self.head_size)

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
            def rotate(x: Tensor) -> Tensor:
                a = x[..., 0::2]
                b = x[..., 1::2]
                # get oscillators, we have to project them across the time axis
                # and then also across the whole head size in order to apply 1
                # oscillator per pair (i think)
                cos = torch.cos(self.angles[eval_offset:eval_offset + time])[None, :, None, :]
                sin = torch.sin(self.angles[eval_offset:eval_offset + time])[None, :, None, :]   # each (seq_len, head_dim/2)
                # rotate pairs across the curve from their assigned oscillator
                a_rot = a * cos - b * sin
                b_rot = a * sin + b * cos
                new_x = x.clone() # in-place assignment for autograd
                # update w/rotated points back to pos
                new_x[..., 0::2] = a_rot
                new_x[..., 1::2] = b_rot
                return new_x
            q = rotate(q)
            k = rotate(k)

        # fast sdpa
        # uses [batch, head_size, time, num_heads]
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
            enable_gqa=True
        )

        # after that the output will also be [batch, head_size, time, num_heads] so we have
        # to both transpose this back to what we want [batch, time, head_size, num_heads] and
        # then shape it into our final projected logits
        out = out.transpose(1, 2).reshape(batch, time, channels)

        return self.dropout(self.proj(out))
