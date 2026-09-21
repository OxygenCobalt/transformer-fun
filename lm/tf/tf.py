import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from .block import Block
from lm.config import Config


class Transformer(nn.Module):
    def __init__(
        self, config: Config
    ):
        super().__init__()
        # basically the vocab logits here are the probability of the next token
        # given this token.
        self.token_embedding_table = nn.Embedding(config.tokenizer.vocab(), config.hyperparams.embed_size, device=config.device)
        self.blocks = nn.ModuleList(
            [
                Block(config)
                for _ in range(config.hyperparams.n_layer)
            ]
        )
        self.lm_head = nn.Linear(config.hyperparams.embed_size, config.tokenizer.vocab(), device=config.device)
        self.block_size = config.hyperparams.block_size
        self.positions = config.positions
        if self.positions == "abs":
            self.position_embedding_table = nn.Embedding(config.hyperparams.block_size, config.hyperparams.embed_size, device=config.device)
            self.register_buffer("position", torch.arange(config.hyperparams.block_size, device=config.device))

    def forward(self, idx: Tensor, caches: list[tuple[Tensor, Tensor]] | None, eval_offset: int) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        batch, time = idx.shape
        # token embeddings, simple enough
        x = self.token_embedding_table(idx)  # (batch, time, embed_dim)
        if self.positions == "abs":
            pos_emb = self.position_embedding_table(self.position[eval_offset:eval_offset + time])  # pyright: ignore[reportIndexIssue]
            x = x + pos_emb
        # pass through blocks
        new_caches = []
        if caches:
            for cache, block in zip(caches, self.blocks):
                x, new_cache = block(x, cache, eval_offset)
                new_caches.append(new_cache)
        else:
            for block in self.blocks:
                x, new_cache = block(x, None, eval_offset)
                new_caches.append(new_cache)
        # final lm_head to translate to logits/probs
        x = self.lm_head(x)
        return (x, new_caches)
