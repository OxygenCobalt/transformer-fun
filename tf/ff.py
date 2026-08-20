from torch import Tensor
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            # hidden size of FF is bigger than the output layers
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            # proj + inner-outer layer
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
