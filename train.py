import os
import random

import pandas
import torch
import tqdm
from torch.nn import functional as F

from tf.tf import Transformer
from tok.bpe import BPE

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
    tokenizer.train(train_docs + test_docs + validation_docs)
    with open(bpe_path, "xb") as checkpoint:
        print("saved bpe tokenizer")
        tokenizer.save(checkpoint)

train = None
test = None
validation = None
train_path = checkpoint_dir + "/train.pl"
test_path = checkpoint_dir + "/test.pl"
validation_path = checkpoint_dir + "/validation.pl"
loaded_universe = False
if os.path.exists(train_path) and os.path.exists(test_path):
    train = torch.load(train_path, map_location=device)
    test = torch.load(test_path, map_location=device)
    validation = torch.load(test_path, map_location=device)
    loaded_universe = True

if not loaded_universe:
    print("tokenizing data")
    train = []
    test = []
    validation = []
    for doc in tqdm.tqdm(train_docs, desc="train"):
        tokenized_doc = tokenizer.forward(doc)
        train += tokenized_doc
    for doc in tqdm.tqdm(test_docs, desc="test"):
        tokenized_doc = tokenizer.forward(doc)
        test += tokenized_doc
    for doc in tqdm.tqdm(validation_docs, desc="validation"):
        tokenized_doc = tokenizer.forward(doc)
        validation += tokenized_doc

    train = torch.tensor(train, dtype=torch.long).to(device)
    torch.save(train, train_path)
    test = torch.tensor(test, dtype=torch.long).to(device)
    torch.save(test, test_path)
    validation = torch.tensor(validation, dtype=torch.long).to(device)
    torch.save(validation, validation_path)

all_toks = 20_000_000 # train.shape[0]

m = Transformer(
    n_layer, tokenizer.vocab, embed_size, heads, block_size, dropout, device
).to(device)

# adam optimizer! this is really effective but i dont know why nor how it works
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# --claude checkpoint code cuz training is slow--
# resume from latest checkpoint if available
tokens_seen = 0
latest_ckpt = os.path.join(checkpoint_dir, "latest.pt")
if os.path.exists(latest_ckpt):
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=True)
    m.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    tokens_seen = ckpt["tokens_seen"] + 1
    print(f"resumed from {tokens_seen} tokens seen")
else:
    print("saving initial model state")
    torch.save(
        {"model": m.state_dict(), "optimizer": optimizer.state_dict(), "tokens_seen": 0},
        os.path.join(checkpoint_dir, "init.pt"),
    )

o = torch.arange(0, block_size, dtype=torch.long).to(device)

while tokens_seen < all_toks:
    # eval
    out = {}
    m.eval()
    with torch.no_grad():
        map = {
            "train": train,
            "test": test,
            "validation": validation
        }
        for split in ["train", "test", "validation"]:
            evaled_toks = 0
            total_loss = 0
            loss_amts = 0
            set = map[split]
            while evaled_toks < eval_toks:
                ixs = torch.randint(0, len(set) - block_size - 1, (batch_size,)).to(
                    device
                )
                xb = set[ixs[:, None] + o[None, :]]
                yb = set[ixs[:, None] + o[None, :] + 1]
                logits = m(xb)
                batch, time, channels = logits.shape
                l_logits = logits.view(batch * time, channels)
                l_targets = yb.view(batch * time)
                loss = F.cross_entropy(l_logits, l_targets)
                total_loss += loss.item()
                loss_amts += 1
                evaled_toks += block_size * batch_size
            out[split] = total_loss / loss_amts
    m.train()

    print("losses: ", out)

    prog = tqdm.tqdm(total=train_toks, desc="train")
    trained_toks = 0
    while trained_toks < train_toks:
        ixs = torch.randint(0, len(train) - block_size - 1, (batch_size,)).to(device)
        xb = train[ixs[:, None] + o[None, :]]
        yb = train[ixs[:, None] + o[None, :] + 1]
        logits = m(xb)
        # calculate loss
        batch, time, channels = logits.shape
        # crossentropy wants [flat, chan]
        l_logits = logits.view(batch * time, channels)
        l_targets = yb.view(batch * time)
        loss = F.cross_entropy(l_logits, l_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trained_toks += block_size * batch_size
        prog.n = trained_toks
        prog.refresh()
    prog.close()
    tokens_seen += trained_toks

    ckpt_data = {
        "model": m.state_dict(),
        "optimizer": optimizer.state_dict(),
        "tokens_seen": tokens_seen,
    }
    torch.save(ckpt_data, latest_ckpt)
    torch.save(ckpt_data, os.path.join(checkpoint_dir, f"saw_{tokens_seen}.pt"))


# test model now

# my code: i want to generate tokens forever
while True:
    prompt = input("prompt: ")
    print(prompt, end="", flush=True)
    idx = torch.tensor([tokenizer.forward_continue(prompt)], dtype=torch.long).to(device)
    # crop to context window (block size)
    # this is why all models are fixed-context
    # oh this is why models can stream token by token
    while True:
        logits = m(idx[:, -block_size:])
        logits = logits[:, -1, :]  # (B, C): last time step
        probs = F.softmax(logits, dim=-1)  # (B, C)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        s = tokenizer.backward_one(idx_next[0])
        if s is None:
            break
        idx = torch.cat((idx, idx_next), dim=1)
        print(s, end="", flush=True)
    print("--end--")
