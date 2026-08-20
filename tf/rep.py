import torch
from torch import Tensor
import torch.nn as nn

class Rep(nn.Module):
    def __init__(self, emb_size: int, head_size: int, device: str, rope: bool = False):
        super().__init__()
        self.f = nn.Linear(emb_size, head_size, bias=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.f(x)
