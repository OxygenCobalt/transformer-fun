import os

import torch
import tqdm
from torch.nn import functional as F

from tf.tf import Transformer
from tok.bigram import Bigram
from tok.bpe import BPE

# seed
torch.manual_seed(1616)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

bpe = BPE(text, 256)

sys.exit(0)
bg = Bigram(text)
data = torch.tensor(bg.forward(text), dtype=torch.long)

n = int(0.9 * len(data))
train = data[:n]
test = data[n:]


def select(data):
    # random displacement idxs for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # crazy freaking stacking holy crap
    # output becomes like:
    # x: access batch yields sliceable block
    # y: access batch yields indexable output
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


# just eval code eh
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in tqdm.tqdm(range(eval_iters)):
            X, Y = select(train if split == "train" else test)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# hyperparams
device = "cuda" if torch.cuda.is_available() else "cpu"
print("use:", device)
batch_size = 64
block_size = 256
learning_rate = 3e-4
eval_interval = 100
eval_iters = 200
max_iters = 5000
embed_size = 384
heads = 6
n_layer = 6
dropout = 0.2

m = Transformer(n_layer, bg.vocab, embed_size, heads, block_size, dropout, device).to(
    device
)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

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
    torch.save(
        {"model": m.state_dict(), "optimizer": optimizer.state_dict(), "epoch": -1},
        os.path.join(checkpoint_dir, "init.pt"),
    )

for step in tqdm.tqdm(range(start_epoch, max_iters)):
    xb, yb = select(train)
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

    if step % eval_interval == 0:
        # --claude checkpoint code cuz training is slow--
        epoch = step // eval_interval
        ckpt_data = {
            "model": m.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckpt_data, latest_ckpt)
        torch.save(ckpt_data, os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))

# test model now
idx = torch.zeros((1, 1), dtype=torch.long).to(device)

# my code: i want to generate tokens forever
while True:
    # crop to context window (block size)
    # this is why all models are fixed-context
    logits, loss = m(idx[:, -block_size:])
    logits = logits[:, -1, :]  # (B, C): last time step
    probs = F.softmax(logits, dim=-1)  # (B, C)
    idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
    # oh this is why models can stream token by token
    print(bg.backward(idx_next[0].tolist()), end="", flush=True)
    idx = torch.cat((idx, idx_next), dim=1)
