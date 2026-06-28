import os
import pickle
import random

import pandas
import torch
import tqdm
from torch.nn import functional as F

from tf.tf import Transformer
from tok.bigram import Bigram
from tok.bpe import BPE

# seed
torch.manual_seed(1616)

pq = pandas.concat(
    [
        pandas.read_parquet("./wikitext/wikitext-103-v1/train-00000-of-00002.parquet"),
        pandas.read_parquet("./wikitext/wikitext-103-v1/train-00001-of-00002.parquet"),
    ]
)

# hyperparams
device = "cuda" if torch.cuda.is_available() else "cpu"
print("use:", device)
batch_size = 64
block_size = 256
learning_rate = 3e-4
eval_interval = 100
eval_iters = 200
epochs = 10
epoch_iters = 500
embed_size = 384
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
    tokenizer.train(docs)
    with open(bpe_path, "xb") as checkpoint:
        print("saved bpe tokenizer")
        tokenizer.save(checkpoint)

examples = []
examples_path = checkpoint_dir + "/examples.pl"
loaded_examples = False
if os.path.exists(examples_path):
    with open(examples_path, "rb") as checkpoint:
        print("reloaded tokenized checkpoint")
        examples = pickle.Unpickler(checkpoint).load()["examples"]
        loaded_examples = True

if not loaded_examples:
    print("tokenizing data")
    for doc in tqdm.tqdm(docs):
        tokenized_doc = tokenizer.forward(doc)
        if len(tokenized_doc) <= block_size:
            continue
        examples.append(tokenized_doc)
    with open(examples_path, "xb") as checkpoint:
        pickle.Pickler(checkpoint).dump({"examples": examples})

n = int(0.9 * len(examples))
train = examples[:n]
test = examples[n:]

m = Transformer(
    n_layer, tokenizer.vocab, embed_size, heads, block_size, dropout, device
).to(device)

# adam optimizer! this is really effective but i dont know why nor how it works
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# --claude checkpoint code cuz training is slow--
# resume from latest checkpoint if available
start_epoch = 0
latest_ckpt = os.path.join(checkpoint_dir, "latest.pt")
if os.path.exists(latest_ckpt):
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=True)
    m.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    print(f"resumed from epoch {start_epoch}")
else:
    print("saving initial model state")
    torch.save(
        {"model": m.state_dict(), "optimizer": optimizer.state_dict(), "epoch": -1},
        os.path.join(checkpoint_dir, "init.pt"),
    )

for epoch in range(start_epoch, epochs):
    # eval
    out = {}
    m.eval()
    with torch.no_grad():
        for split in ["train", "test"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                set = train if split == "train" else test
                xs = []
                ys = []
                for i in range(batch_size):
                    doc = random.choice(set)
                    i = random.randint(0, len(doc) - block_size - 1)
                    xs.append(
                        torch.tensor(doc[i : i + block_size], dtype=torch.long).to(
                            device
                        )
                    )
                    ys.append(
                        torch.tensor(
                            doc[i + 1 : i + block_size + 1], dtype=torch.long
                        ).to(device)
                    )
                xb = torch.stack(xs).to(device)
                yb = torch.stack(ys).to(device)
                logits = m(xb)
                batch, time, channels = logits.shape
                l_logits = logits.view(batch * time, channels)
                l_targets = yb.view(batch * time)
                loss = F.cross_entropy(l_logits, l_targets)
                losses[k] = loss.item()
            out[split] = losses.mean()
    m.train()

    print("losses: ", out)

    prog = tqdm.tqdm(range(epoch_iters))
    prog.desc = f"epoch {epoch}"
    for step in prog:
        xs = []
        ys = []
        for i in range(batch_size):
            doc = random.choice(train)
            i = random.randint(0, len(doc) - block_size - 1)
            xs.append(
                torch.tensor(doc[i : i + block_size], dtype=torch.long).to(device)
            )
            ys.append(
                torch.tensor(doc[i + 1 : i + block_size + 1], dtype=torch.long).to(
                    device
                )
            )
        xb = torch.stack(xs).to(device)
        yb = torch.stack(ys).to(device)
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

    ckpt_data = {
        "model": m.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt_data, latest_ckpt)
    torch.save(ckpt_data, os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))


# test model now

# my code: i want to generate tokens forever
while True:
    prompt = input("prompt: ")
    print(prompt, end="", flush=True)
    idx = torch.tensor([tokenizer.forward(prompt)], dtype=torch.long).to(device)
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
