from torch import Tensor
import torch.nn as nn

from torch.nn import functional as F
from lm.config import Config

class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hidden_size = int(config.hyperparams.ffn_ratio * config.hyperparams.embed_size)
        if config.gating == "swiglu":
            self.gate_proj = nn.Linear(config.hyperparams.embed_size, hidden_size, device=config.device)
        self.up_proj = nn.Linear(config.hyperparams.embed_size, hidden_size, device=config.device)
        self.down_proj = nn.Linear(hidden_size, config.hyperparams.embed_size, device=config.device)
        self.out_proj = nn.Dropout(config.hyperparams.dropout)
        self.gating = config.gating

    def forward(self, x: Tensor) -> Tensor:
        if self.gating == "relu":
            # relu is the standard validated activation fn
            # this kind of deep-ish neural net in my mind is the "interpreter"
            # of the attention output, in some sense. it needs to go pretty wide
            # hence the out-in-projection
            hidden = F.relu(self.up_proj(x))
            out = self.down_proj(hidden)
            return self.out_proj(out)
        elif self.gating == "swiglu":
            # swiglu is a bit odd, its like our existing ff but we both have a
            # value proj (no activation fn) and a gate proj (silu activation fn).
            # doing elementwise mul on this lets us both determine a first set of
            # learned features a and then another dimension from the input that we
            # can then gate on before evaluating the down-projection
            value = self.up_proj(x)
            gate = F.silu(self.gate_proj(x))
            hidden = gate * value
            out = self.down_proj(hidden)
            return self.out_proj(out)
        else:
            raise ValueError("invalid gating " + self.gating)
