"""train the two models once, with every seed fixed, and write the weights
plus a json record of how they were trained."""
import argparse, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import data, models, env

SEED = 0


def seed_all(s):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def evaluate(m, x, y, dev, bs=500):
    m.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            correct += (m(x[i:i + bs].to(dev)).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return correct / len(x)


def augment(xb, g):
    """random crop with 4px pad + horizontal flip, on normalised tensors"""
    b = xb.shape[0]
    p = F.pad(xb, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (b,), generator=g); dy = torch.randint(0, 9, (b,), generator=g)
    out = torch.stack([p[i, :, dy[i]:dy[i] + 32, dx[i]:dx[i] + 32] for i in range(b)])
    flip = torch.rand(b, generator=g) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def train_resnet(epochs, bs, dev, log):
    seed_all(SEED)
    xtr, ytr = data.cifar_train(); xte, yte = data.cifar_test()
    m = models.resnet18_cifar().to(dev)
    if dev == "cpu":
        m = m.to(memory_format=torch.channels_last)
    steps = epochs * (len(xtr) // bs)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps, pct_start=0.25)
    g = torch.Generator().manual_seed(SEED)
    rec = {"model": "resnet18_cifar", "seed": SEED, "epochs": epochs, "batch": bs, "device": dev,
           "optimizer": "sgd nesterov lr 0.1 momentum 0.9 wd 5e-4 onecycle", "label_smoothing": 0.1,
           "augment": "random crop pad 4, horizontal flip", "epoch_log": []}
    t0 = time.time()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(xtr), generator=g); tl = 0.0; n = 0
        for i in range(0, len(xtr) - bs + 1, bs):
            idx = perm[i:i + bs]
            xb = augment(xtr[idx], g).to(dev)
            if dev == "cpu":
                xb = xb.contiguous(memory_format=torch.channels_last)
            loss = F.cross_entropy(m(xb), ytr[idx].to(dev), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * bs; n += bs
        acc = evaluate(m, xte, yte, dev)
        rec["epoch_log"].append({"epoch": ep + 1, "train_loss": tl / n, "test_acc": acc, "elapsed_s": time.time() - t0})
        print(f"epoch {ep + 1}/{epochs} loss {tl / n:.3f} test acc {acc:.4f} {time.time() - t0:.0f}s", flush=True)
    rec["test_acc"] = acc; rec["train_s"] = time.time() - t0
    os.makedirs(models.WEIGHTS, exist_ok=True)
    p = os.path.join(models.WEIGHTS, "resnet18_cifar10.pt")
    torch.save({k: v.contiguous() for k, v in m.cpu().state_dict().items()}, p)
    rec["sha256"] = data.sha256(p); rec["env"] = env.capture()
    json.dump(rec, open(log, "w"), indent=1)


def train_text(epochs, bs, dev, log):
    seed_all(SEED)
    xtr, ytr = data.ag_train(); xte, yte = data.ag_test()
    m = models.TinyTransformer().to(dev)
    steps = epochs * (len(xtr) // bs)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(SEED)
    rec = {"model": "tiny_transformer_agnews", "seed": SEED, "epochs": epochs, "batch": bs, "device": dev,
           "optimizer": "adamw lr 1e-3 wd 0.01 onecycle", "epoch_log": []}
    t0 = time.time()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(xtr), generator=g); tl = 0.0; n = 0
        for i in range(0, len(xtr) - bs + 1, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(m(xtr[idx].to(dev)), ytr[idx].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * bs; n += bs
        acc = evaluate(m, xte, yte, dev)
        rec["epoch_log"].append({"epoch": ep + 1, "train_loss": tl / n, "test_acc": acc, "elapsed_s": time.time() - t0})
        print(f"epoch {ep + 1}/{epochs} loss {tl / n:.3f} test acc {acc:.4f} {time.time() - t0:.0f}s", flush=True)
    rec["test_acc"] = acc; rec["train_s"] = time.time() - t0
    os.makedirs(models.WEIGHTS, exist_ok=True)
    p = os.path.join(models.WEIGHTS, "transformer_agnews.pt")
    torch.save(m.cpu().state_dict(), p)
    rec["sha256"] = data.sha256(p); rec["env"] = env.capture()
    json.dump(rec, open(log, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["resnet", "text"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(models.WEIGHTS, exist_ok=True)
    if a.model == "resnet":
        train_resnet(a.epochs or 15, a.batch, a.device, os.path.join(models.WEIGHTS, "resnet18_cifar10.train.json"))
    else:
        train_text(a.epochs or 3, a.batch, a.device, os.path.join(models.WEIGHTS, "transformer_agnews.train.json"))
