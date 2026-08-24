from torch import Tensor
import torch.nn as nn

from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, embed_size: int, gating: str, ffn_ratio: float,dropout: float):
        super().__init__()
        hidden_size = int(ffn_ratio * embed_size)
        if gating == "swiglu":
            self.gate_proj = nn.Linear(embed_size, hidden_size)
        self.up_proj = nn.Linear(embed_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, embed_size)
        self.out_proj = nn.Dropout(dropout)
        self.gating = gating

    def forward(self, x: Tensor) -> Tensor:
        if self.gating == "relu":
            hidden = F.relu(self.up_proj(x))
            out = self.down_proj(hidden)
            return self.out_proj(out)
        elif self.gating == "swiglu":
            value = self.up_proj(x)
            gate = F.silu(self.gate_proj(x))
            hidden = gate * value
            out = self.down_proj(hidden)
            return self.out_proj(out)
        else:
            raise ValueError("invalid gating " + self.gating)
