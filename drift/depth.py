"""dump the activation of every layer, for a fixed subset of inputs, under one
configuration. forward hooks on every leaf module plus every residual block;
each firing is saved in order as <nnn>_<module>#<call>.pt so the analysis can
compare two dumps one layer at a time.

    python -m drift.depth --label L --model resnet --config b32 --n 64
"""
import argparse, os, shutil, sys
import torch
from . import configs, data, models

SUBSET_SEED = 1


def subset_indices(n_total, n):
    g = torch.Generator().manual_seed(SUBSET_SEED)
    return torch.randperm(n_total, generator=g)[:n].sort().values


def hook_all(m, store):
    """store: list of (name, call_index, output tensor on cpu). returns handles."""
    calls = {}
    handles = []
    leaf_or_block = []
    for name, mod in m.named_modules():
        if name == "":
            continue
        if len(list(mod.children())) == 0 or name.count(".") == 1 and name.startswith("layer") or name.startswith("blocks."):
            leaf_or_block.append((name, mod))

    def make(name):
        def hook(mod, inp, out):
            c = calls.get(name, 0); calls[name] = c + 1
            # keep batch-first outputs only (the positional embedding table is not one)
            if torch.is_tensor(out) and torch.is_tensor(inp[0]) and out.shape[:1] == inp[0].shape[:1] and out.dim() >= 2:
                store.append((name, c, out.detach().clone().cpu()))
        return hook
    for name, mod in leaf_or_block:
        handles.append(mod.register_forward_hook(make(name)))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="resnet", choices=["resnet", "text"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    cfgset = configs.gpu_configs(os.cpu_count()) if torch.cuda.is_available() else configs.cpu_configs(os.cpu_count())
    cfg = configs.by_name(cfgset)[a.config]
    for k, v in cfg.env().items():
        assert os.environ.get(k) == v
    cfg.apply()
    dev = cfg.device
    x, _ = data.cifar_test() if a.model == "resnet" else data.ag_test()
    idx = subset_indices(len(x), a.n)
    xs = x[idx].to(dev)
    m = models.load(a.model, dev)
    out = os.path.join("results", a.label, ".depth_tmp", a.model, cfg.name)
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    # one pass per batch of cfg.batch, stitched per layer in input order
    per_layer = {}  # key -> list of tensors
    order = []
    with torch.no_grad():
        for i in range(0, a.n, cfg.batch):
            store = []
            hs = hook_all(m, store)
            logits = m(xs[i:i + cfg.batch])
            for h in hs:
                h.remove()
            store.append(("logits", 0, logits.detach().cpu()))
            for name, c, t in store:
                key = f"{name}#{c}"
                if key not in per_layer:
                    per_layer[key] = []; order.append(key)
                per_layer[key].append(t)
    for j, key in enumerate(order):
        t = torch.cat(per_layer[key], 0)
        torch.save(t, os.path.join(out, f"{j:03d}_{key.replace('/', '_')}.pt"))
    print(f"{cfg.name}: {len(order)} layers, {a.n} inputs -> {out}", flush=True)


if __name__ == "__main__":
    main()
