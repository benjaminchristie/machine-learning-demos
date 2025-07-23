import torch
import tqdm

from argparse import ArgumentParser

from models import TransformerDecoder
from utils import save_model, load_model


# data loading
def get_batch(data, batch_size, block_size, device):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: TransformerDecoder,
    train_batch,
    valid_batch,
    batch_size,
    block_size,
    device,
    eval_iters=16,
):
    out = {}
    # train
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        X, Y = get_batch(train_batch, batch_size, block_size, device)
        loss = model.update_parameters(X, Y, update_weights=False)
        losses[k] = loss
    out["train"] = losses.mean()
    # valu
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        X, Y = get_batch(valid_batch, batch_size, block_size, device)
        loss = model.update_parameters(X, Y, update_weights=False)
        losses[k] = loss
    out["val"] = losses.mean()
    return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--demo", action="store_true", default=False)
    args = parser.parse_args()
    batch_size = 64
    block_size = 256
    max_iters = 5000
    epochs = 5000
    eval_interval = epochs // 10
    lr = 3e-4
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """prepare dataset"""
    with open("./data/alldata.txt", "rb") as f:
        text = f.read().decode(errors="ignore")
    print(f"loaded {len(text)/1e6}M characters")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long, device=device)
    n = int(0.9 * len(data))
    train_data = data[:n]
    valid_data = data[n:]

    """prepare model"""
    model = TransformerDecoder(
        vocab_size,
        n_embd,
        block_size,
        n_head,
        dropout,
        4 * n_embd,
        n_layer,
        device,
        lr=lr,
    ).to(device)

    """begin training"""

    if not args.demo:
        print(f"training with {sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1e6}M parameters, vocab size {vocab_size}")
        epochs = 5000
        tbar = tqdm.trange(epochs)
        s = []
        for i in tbar:
            if i % eval_interval == 0 or i == epochs - 1:
                losses = estimate_loss(model, train_data, valid_data, batch_size, block_size, device)
                s = [f"ep {i}: {losses['train']:4.4f} {losses['val']:4.4f}", ""]
                tbar.set_description("".join(s))
            xb, yb = get_batch(train_data, batch_size, block_size, device)
            l = model.update_parameters(xb, yb)
            s[-1] = f" | curr: {l:3.3f}"
            tbar.set_description("".join(s))
        save_model(model, "./weights/alldata.pt")
    else:
        load_model(model, "./weights/alldata.pt")
        
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        N = 0
        while True:
        # for _ in range(block_size - 1):
            N += 1
            if N >= block_size:
                context = context[:, 1:block_size]
            new = decode(model.sample(context)[0].cpu().tolist())
            new_t = torch.tensor(encode(new), dtype=torch.long, device=device).unsqueeze(0)
            context = torch.concatenate((context, new_t), dim=1)

            print(new[0], end="")