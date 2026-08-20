from pydantic import BaseModel, ConfigDict
from torch import nn
from torch import Tensor

import torch
import tqdm
from torch.nn import functional as F

from tf.tf import Transformer
from tok.tok import Tokenizer
import os


class Hyperparams(BaseModel):
    n_layer: int
    embed_size: int
    heads: int
    block_size: int
    dropout: float
    learning_rate: float
    batch_size: int

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hyperparams: Hyperparams
    tokenizer: Tokenizer
    positions: str
    device: str

class LanguageModel:
    def __init__(self, config: Config):
        self.config = config
        self.m = Transformer(
            config.hyperparams.n_layer,
            config.tokenizer.vocab(),
            config.hyperparams.embed_size,
            config.hyperparams.heads,
            config.hyperparams.block_size,
            config.hyperparams.dropout,
            config.device
        )
        self.offsets = torch.arange(0, config.hyperparams.block_size, dtype=torch.long, device=config.device)
        self.optimizer = torch.optim.AdamW(self.m.parameters(), lr=config.hyperparams.learning_rate)

    def forward_sample(self, corpus: Tensor, batch_size: int):
        ixs = torch.randint(0, len(corpus) - self.config.hyperparams.block_size - 1, (batch_size,), device=self.config.device)
        xb = corpus[ixs[:, None] + self.offsets[None, :]]
        yb = corpus[ixs[:, None] + self.offsets[None, :] + 1]
        logits = self.m(xb)
        batch, time, channels = logits.shape
        l_logits = logits.view(batch * time, channels)
        l_targets = yb.view(batch * time)
        loss = F.cross_entropy(l_logits, l_targets)
        return loss

    def train(self, corpus: Tensor, tokens: int):
        prog = tqdm.tqdm(total=tokens, desc="train")
        trained_toks = 0
        while trained_toks < tokens:
            loss = self.forward_sample(corpus, self.config.hyperparams.batch_size)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            trained_toks += self.config.hyperparams.block_size * self.config.hyperparams.batch_size
            prog.n = trained_toks
            prog.refresh()
        prog.close()

    def eval(self, corpus: Tensor, tokens: int, desc: str = "") -> float:
        prog = tqdm.tqdm(total=tokens, desc="test" + f"@{desc}" if desc else "")
        trained_toks = 0
        total_loss = 0
        losses = 0
        while trained_toks < tokens:
            total_loss += self.forward_sample(corpus, self.config.hyperparams.batch_size).item()
            losses += 1
            trained_toks += self.config.hyperparams.block_size * self.config.hyperparams.batch_size
            prog.n = trained_toks
            prog.refresh()
        prog.close()
        return total_loss / losses

    def eval_all(self, corpuses: dict[str, Tensor], tokens: int) -> dict[str, float]:
        losses = {}
        for split, corpus in corpuses.items():
            losses[split] = self.eval(corpus, tokens, split)
        return losses

    def full_train(self, corpuses: dict[str, Tensor], tokens: int, train_tokens: int, eval_tokens: int, checkpoint_path: str):
        print("begin full training loop")
        latest_ckpt = os.path.join(checkpoint_path, "latest.pt")
        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=self.config.device, weights_only=True)
            self.m.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            trained_toks = ckpt["trained_toks"] + 1
            print(f"resumed from {eval_tokens} tokens")
        else:
            print("saving initial model state")
            torch.save(
                {"model": self.m.state_dict(), "optimizer": self.optimizer.state_dict(), "trained_toks": 0},
                os.path.join(checkpoint_path, "init.pt"),
            )

        trained_toks = 0
        while trained_toks < tokens:
            self.train(corpuses["train"], train_tokens)
            losses = self.eval_all(corpuses, eval_tokens)
            print("LOSSES:")
            for split, loss in losses.items():
                print(split, "=", loss)
            trained_toks += train_tokens
