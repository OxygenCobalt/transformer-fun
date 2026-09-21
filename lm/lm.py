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
        if config.optimizer == "adamw":
            self.optimizers = [torch.optim.AdamW(self.m.parameters(), **config.optimizer_hyperparams["adamw"])]
        elif config.optimizer == "muon":
            # muon only works on 2d matrices, so we separate things like biases, norms, embeddings
            # into a separate adam optimizer
            muon_params = []
            adamw_params = []
            for name, param in self.m.named_parameters():
                if name.startswith("blocks.") and param.ndim == 2:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)
            self.optimizers = [
                torch.optim.Muon(muon_params, **config.optimizer_hyperparams["muon"]),
                torch.optim.AdamW(adamw_params, **config.optimizer_hyperparams["adamw"])
            ]

    def forward_sample(self, corpus: Tensor, batch_size: int, seq_len: int | None = None, eval_offset: int = 0):
        ixs = torch.randint(0, len(corpus) - (seq_len or self.config.hyperparams.block_size) - 1, (batch_size,), device=self.config.device)
        xb = corpus[ixs[:, None] + self.offsets[None, :seq_len]]
        yb = corpus[ixs[:, None] + self.offsets[None, :seq_len] + 1]
        logits, _ = self.m(xb, None, eval_offset=eval_offset)
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
        while trained_toks < tokens:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = self.forward_sample(corpus, self.config.hyperparams.batch_size, seq_len=seq_len)
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for optimizer in self.optimizers:
                optimizer.step()
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
        while trained_toks < tokens:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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

    def full_train(self, corpuses: dict[str, Tensor], tokens: int, train_tokens: int, eval_tokens: int, exp_path: str):
        now = datetime.now()
        checkpoint_path = exp_path + "/checkpoints"
        print("begin full training loop")
        trained_toks = 0
        latest_ckpt = os.path.join(checkpoint_path, "latest.pt")
        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=self.config.device, weights_only=True)
            self.m.load_state_dict(ckpt["model"])
            for optimizer, state in zip(self.optimizers, ckpt["optimizers"]):
                optimizer.load_state_dict(state)
            trained_toks = ckpt["trained_toks"] + 1
            now = datetime.fromtimestamp(ckpt["now"])
            print(f"resumed from {eval_tokens} tokens")
        else:
            print("saving initial model state")
            torch.save(
                {"model": self.m.state_dict(), "optimizers": [optimizer.state_dict() for optimizer in self.optimizers], "trained_toks": 0, "now": now.timestamp()},
                os.path.join(checkpoint_path, "init.pt"),
            )
            torch.save(
                {"model": self.m.state_dict(), "optimizers": [optimizer.state_dict() for optimizer in self.optimizers], "trained_toks": 0, "now": now.timestamp()},
                os.path.join(checkpoint_path, "latest.pt"),
            )

        if trained_toks >= tokens:
            print("already trained")
            return

        data_path = exp_path + f"/train_{now}.csv"
        if trained_toks == 0:
            losses = self.full_eval(corpuses, eval_tokens)
            if not os.path.exists(data_path):
                with open(data_path, "xt") as file:
                    print("trained_tokens,eval,split,loss", file=file)
            with open(data_path, "wt") as file:
                for split, exp in losses.items():
                    for lbl, loss in exp.items():
                        print(f"{trained_toks},{lbl},{split},{loss}", file=file)

        while trained_toks < tokens:
            if self.config.experiments.eval_offsets:
                self.train(corpuses["train"], train_tokens, seq_len=self.config.hyperparams.block_size // 2)
            else:
                self.train(corpuses["train"], train_tokens)
            trained_toks += train_tokens

            losses = self.full_eval(corpuses, eval_tokens)
            with open(data_path, "a") as file:
                for split, exp in losses.items():
                    for lbl, loss in exp.items():
                        print(f"{trained_toks},{lbl},{split},{loss}", file=file)
            torch.save(
                {"model": self.m.state_dict(), "optimizers": [optimizer.state_dict() for optimizer in self.optimizers], "trained_toks": trained_toks, "now": now.timestamp()},
                os.path.join(checkpoint_path, "init.pt"),
            )
            torch.save(
                {"model": self.m.state_dict(), "optimizers": [optimizer.state_dict() for optimizer in self.optimizers], "trained_toks": trained_toks, "now": now.timestamp()},
                os.path.join(checkpoint_path, "latest.pt"),
            )

    def complete(self, prompt: str):
        self.m.eval()
        with torch.inference_mode():
            # my code: i want to generate tokens forever
            print(prompt, end="", flush=True)
            idx = torch.tensor([self.config.tokenizer.tokenize([prompt])[:-1]], dtype=torch.long, device=self.config.device)
            # crop to context window (block size)
            # this is why all models are fixed-context
            # oh this is why models can stream token by token
            logits, caches = self.m(idx[:, -self.config.hyperparams.block_size:], None, eval_offset=0)
            while True:
                logits = logits[:, -1, :]  # (B, C): last time step
                probs = F.softmax(logits, dim=-1)  # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                s = self.config.tokenizer.stringify_one(int(idx_next[0]))
                if s is None:
                    break
                print(s, end="", flush=True)
                offset = caches[0][0].shape[-2]
                if offset >= self.config.hyperparams.block_size and self.config.positions == "abs":
                    # cut to position table size for abs positions, not really much use for it
                    # since its a worse experimental mode
                    break
                logits, caches = self.m(idx_next, caches, eval_offset=offset)
            print("--end--")
