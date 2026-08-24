from torch import Tensor
from datetime import datetime

import torch
import tqdm
from torch.nn import functional as F

from .tf.tf import Transformer
from .config import Config
import os

BETA = 0.99

class LanguageModel:
    def __init__(self, config: Config):
        self.config = config
        self.m = Transformer(config)
        self.offsets = torch.arange(0, config.hyperparams.block_size, dtype=torch.long, device=config.device)
        self.optimizer = torch.optim.AdamW(self.m.parameters(), lr=config.hyperparams.learning_rate)

    def forward_sample(self, corpus: Tensor, batch_size: int, seq_len: int | None = None, eval_offset: int = 0):
        ixs = torch.randint(0, len(corpus) - (seq_len or self.config.hyperparams.block_size) - 1, (batch_size,), device=self.config.device)
        xb = corpus[ixs[:, None] + self.offsets[None, :seq_len]]
        yb = corpus[ixs[:, None] + self.offsets[None, :seq_len] + 1]
        logits = self.m(xb, eval_offset=eval_offset)
        batch, time, channels = logits.shape
        l_logits = logits.view(batch * time, channels)
        l_targets = yb.view(batch * time)
        loss = F.cross_entropy(l_logits, l_targets)
        return loss

    def train(self, corpus: Tensor, tokens: int, seq_len: int | None = None):
        prog = tqdm.tqdm(total=tokens, desc="train")
        trained_toks = 0
        ema = 0
        total_loss = 0
        losses = 0
        while trained_toks <= tokens:
            loss = self.forward_sample(corpus, self.config.hyperparams.batch_size, seq_len=seq_len)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            trained_toks += (seq_len or self.config.hyperparams.block_size) * self.config.hyperparams.batch_size
            prog.n = trained_toks
            ema = (loss * BETA) + (ema * (1 - BETA))
            total_loss += loss
            losses += 1
            avg_loss = total_loss / losses
            prog.postfix = f"avg. loss = {avg_loss:.2f} ema = {ema:.2f}"
            prog.refresh()
        prog.close()

    def eval(self, corpus: Tensor, tokens: int, desc: str = "", seq_len: int | None = None, eval_offset: int = 0) -> float:
        prog = tqdm.tqdm(total=tokens, desc="test" + f"@{desc}" if desc else "")
        trained_toks = 0
        total_loss = 0
        losses = 0
        while trained_toks <= tokens:
            total_loss += self.forward_sample(corpus, self.config.hyperparams.batch_size, seq_len, eval_offset).item()
            losses += 1
            trained_toks += (seq_len or self.config.hyperparams.block_size) * self.config.hyperparams.batch_size
            prog.n = trained_toks
            prog.refresh()
        prog.close()
        return total_loss / losses

    def full_eval(self, corpuses: dict[str, Tensor], tokens: int) -> dict[str, dict[str, float]]:
        losses = {}
        for split, corpus in corpuses.items():
            exps = {"normal": self.eval(corpus, tokens, desc=f"{split}.normal")}
            if self.config.experiments.eval_offsets:
                for i in range(1, self.config.hyperparams.block_size // 2):
                    exps[f"offset@{i}"] = self.eval(corpus, tokens, desc=f"{split}.offset.{i}", seq_len=self.config.hyperparams.block_size // 2, eval_offset=i)
            losses[split] = exps
        return losses

    def full_train(self, corpuses: dict[str, Tensor], tokens: int, train_tokens: int, eval_tokens: int, checkpoint_path: str):
        now = datetime.now()
        print("begin full training loop")
        trained_toks = 0
        latest_ckpt = os.path.join(checkpoint_path, "latest.pt")
        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=self.config.device, weights_only=True)
            self.m.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            trained_toks = ckpt["trained_toks"] + 1
            now = datetime.fromtimestamp(ckpt["now"])
            print(f"resumed from {eval_tokens} tokens")
        else:
            print("saving initial model state")
            torch.save(
                {"model": self.m.state_dict(), "optimizer": self.optimizer.state_dict(), "trained_toks": 0, "now": now.timestamp()},
                os.path.join(checkpoint_path, "init.pt"),
            )
            torch.save(
                {"model": self.m.state_dict(), "optimizer": self.optimizer.state_dict(), "trained_toks": 0, "now": now.timestamp()},
                os.path.join(checkpoint_path, "latest.pt"),
            )

        losses = self.full_eval(corpuses, eval_tokens)
        data_path = f"train_{now}.csv"
        if not os.path.exists(data_path):
            with open(data_path, "xt") as file:
                print("trained_tokens,eval,split,loss", file=file)
        with open(data_path, "wt") as file:
            for split, exp in losses.items():
                for lbl, loss in exp.items():
                    print(f"{trained_toks},{lbl},{split},{loss}", file=file)

        while trained_toks <= tokens:
            if self.config.experiments.eval_offsets:
                self.train(corpuses["train"], train_tokens, seq_len=self.config.hyperparams.block_size // 2)
            else:
                self.train(corpuses["train"], train_tokens)

            losses = self.full_eval(corpuses, eval_tokens)
            with open(data_path, "a") as file:
                for split, exp in losses.items():
                    for lbl, loss in exp.items():
                        print(f"{trained_toks},{lbl},{split},{loss}", file=file)
            trained_toks += train_tokens
            torch.save(
                {"model": self.m.state_dict(), "optimizer": self.optimizer.state_dict(), "trained_toks": trained_toks, "now": now.timestamp()},
                os.path.join(checkpoint_path, f"{trained_toks}.pt"),
            )
            torch.save(
                {"model": self.m.state_dict(), "optimizer": self.optimizer.state_dict(), "trained_toks": trained_toks, "now": now.timestamp()},
                os.path.join(checkpoint_path, "latest.pt"),
            )
