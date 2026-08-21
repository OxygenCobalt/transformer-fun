import os
import random

import pandas
import torch
import tqdm
from torch.nn import functional as F

from tok.bpe import BPE
from lm.lm import LanguageModel, Config, Hyperparams

# seed
torch.manual_seed(1616)
random.seed(1616)

train_data = pandas.concat(
    [
        pandas.read_parquet("./wikitext/wikitext-103-raw-v1/train-00000-of-00002.parquet"),
        pandas.read_parquet("./wikitext/wikitext-103-raw-v1/train-00001-of-00002.parquet"),
    ]
)
train_docs = [text for text in train_data["text"]]
test_data = pandas.concat(
    [
        pandas.read_parquet("./wikitext/wikitext-103-raw-v1/test-00000-of-00001.parquet"),
    ]
)
test_docs = [text for text in test_data["text"]]

validation_data = pandas.concat(
    [
        pandas.read_parquet("./wikitext/wikitext-103-raw-v1/validation-00000-of-00001.parquet"),
    ]
)
validation_docs = [text for text in test_data["text"]]

# hyperparams
device = "cuda" if torch.cuda.is_available() else "cpu"
print("use:", device)

train_toks = 5_000_000
eval_toks = 500_000
batch_size = 256
block_size = 256
learning_rate = 3e-4
embed_size = 768 // 2
heads = 6
n_layer = 6
dropout = 0.2

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

print("initializing tokenizer")
tokenizer = BPE(512)
bpe_path = checkpoint_dir + "/bpe.pl"
loaded = False
if os.path.exists(bpe_path):
    with open(bpe_path, "rb") as checkpoint:
        print("reloaded bpe tokenizer checkpoint")
        loaded = tokenizer.load(checkpoint)
if not loaded:
    tokenizer.train(train_docs)
    with open(bpe_path, "xb") as checkpoint:
        print("saved bpe tokenizer")
        tokenizer.save(checkpoint)

config = Config(
    hyperparams = Hyperparams(
        n_layer = 2,
        embed_size = 64,
        heads = 4,
        block_size = 64,
        batch_size = 8,
        # n_layer = 6,
        # embed_size = 384,
        # heads = 6,
        # block_size = 256,
        # batch_size = 256,
        learning_rate = 3e-4,
        dropout = 0.2,
    ),
    tokenizer = tokenizer,
    positions = "rope",
    device = device
)

lm = LanguageModel(config)
corpuses_path = checkpoint_dir + "/corpus.pt"
corpuses = {}
if os.path.exists(corpuses_path):
    corpuses = torch.load(corpuses_path, map_location=device)
if not corpuses:
    print("tokenizing data")

    def tokenize_docs(docs: list[str], desc: str) -> list[int]:
        tokens: list[int] = []
        tokenizer_batch_size = 4096
        with tqdm.tqdm(total=len(docs), desc=desc) as progress:
            for start in range(0, len(docs), tokenizer_batch_size):
                batch = docs[start : start + tokenizer_batch_size]
                tokens.extend(tokenizer.tokenize(batch))
                progress.update(len(batch))
        return tokens

    train = tokenize_docs(train_docs, "train")
    test = tokenize_docs(test_docs, "test")
    validation = tokenize_docs(validation_docs, "validation")

    train = torch.tensor(train, dtype=torch.long).to(device)
    test = torch.tensor(test, dtype=torch.long).to(device)
    validation = torch.tensor(validation, dtype=torch.long).to(device)
    corpuses = {
        "train": train,
        "test": test,
        "validation": validation
    }
    torch.save(corpuses, corpuses_path)

lm.full_train(corpuses, tokens = 5_000_000, train_tokens = 500_000, eval_tokens = 50_000, checkpoint_path = "./checkpoints")
