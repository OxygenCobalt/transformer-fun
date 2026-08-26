import os
import random

import torch
from torch._refs import Tensor

from lm.tok.bpe import BPE
from lm.lm import LanguageModel
from lm.config import Config, Hyperparams, Experiments
from lm.data import wikitext_103_raw_v1

# seed
torch.manual_seed(1616)
random.seed(1616)

# hyperparams
tokenizer = BPE(512)
config = Config(
    hyperparams = Hyperparams(
        # n_layer = 2,
        # embed_size = 64,
        # heads = 4,
        # block_size = 64,
        # batch_size = 8,
        n_layer = 12,
        embed_size = 768,
        heads = 12,
        block_size = 512,
        batch_size = 64,
        ffn_ratio = 8 / 3,
        learning_rate = 3e-4,
        dropout = 0.2,
    ),
    tokenizer = tokenizer,
    gating = "swiglu",
    positions = "rope",
    device = "cuda" if torch.cuda.is_available() else "cpu",
    experiments = Experiments(
        eval_offsets=False
    )
)
print("go:", config)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
tokenizer_dir = tokenizer.id()
os.makedirs(tokenizer_dir, exist_ok=True)

dataset = wikitext_103_raw_v1("./wikitext/wikitext-103-raw-v1")
print("initializing tokenizer")

tokenizer_path = f"{tokenizer_dir}/{dataset.id}.tok.pt"
loaded = False
if os.path.exists(tokenizer_path):
    loaded = tokenizer.load(tokenizer_path)
if not loaded:
    tokenizer.train(dataset.train)
    tokenizer.save(tokenizer_path)

corpuses_path = f"{tokenizer_dir}/{dataset.id}.corpuses.pt"
corpuses: dict[str, Tensor] = {}
if os.path.exists(corpuses_path):
    corpuses = torch.load(corpuses_path, map_location=config.device)
if not corpuses:
    print("tokenizing data")
    corpuses = dataset.prepare(tokenizer, config.device)
    torch.save(corpuses, corpuses_path)

lm = LanguageModel(config)
lm.full_train(corpuses, tokens = 500_000_000, train_tokens = 50_000_000, eval_tokens = 5_000_000, checkpoint_path = checkpoint_dir)
