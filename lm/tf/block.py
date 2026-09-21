from torch import Tensor
import torch.nn as nn

from .ff import FeedForward
from .mha import MultiHeadAttention

from lm.config import Config


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # attention
        self.sa = MultiHeadAttention(config)
        # ff to analyze
        self.ff = FeedForward(config)
        # layer norms
        # basically the skip connections help provide other flows for gradients so they dont
        # have to carry unchanged data or similar useless stuff across the attention evaluation,
        # ff, or similar. instead its more like they can just carry the deltas instead. the issue
        # is in some sense "double-adding x" for these skip conns can amplify existing variance
        # within x and cause training to explode, so the norms help clamp it such that it remains
        # a nice clean delta that wont explode.
        # the projection in the internal ff also does...something with skip connections. don't know
        self.ln1 = nn.LayerNorm(config.hyperparams.embed_size, device=config.device)
        self.ln2 = nn.LayerNorm(config.hyperparams.embed_size, device=config.device)

    def forward(self, x: Tensor, cache: tuple[Tensor, Tensor] | None, eval_offset: int) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # skip connections! we add here but proj inside the layers actually
        # so we preserve the pre-activation and then add that to the projected output,
        # which helps with deep learning
        a, new_cache = self.sa(self.ln1(x), cache, eval_offset)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, new_cache
