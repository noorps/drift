"""run one configuration over the whole test set in a fresh process and save
the logits (float32, every bit) and the wall-clock time.

    python -m drift.sweep --label L --model resnet --config b32 [--repeat 2]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from . import configs, data, models, env


def run_pass(m, x, batch, dev):
    out = np.empty((len(x), 10 if x.ndim == 4 else 4), dtype=np.float32)
    with torch.no_grad():
        xd = x.to(dev)
        # one untimed batch so autotuning / lazy init is not in the timing
        t0 = time.perf_counter()
        m(xd[:batch])
        if dev == "cuda":
            torch.cuda.synchronize()
        warm = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in range(0, len(x), batch):
            out[i:i + batch] = m(xd[i:i + batch]).float().cpu().numpy()
        if dev == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    return out, wall, warm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="resnet", choices=["resnet", "text"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--suffix", default="")          # e.g. _proc for the separate-process repeat
    ap.add_argument("--repeat", type=int, default=1)  # >1: run again in the same process, saved as <name>_again
    a = ap.parse_args()
    cfgset = configs.gpu_configs(os.cpu_count()) if torch.cuda.is_available() else configs.cpu_configs(os.cpu_count())
    cfg = configs.by_name(cfgset)[a.config]
    for k, v in cfg.env().items():
        assert os.environ.get(k) == v, f"{k} must be set in the environment before torch starts"
    cfg.apply()
    dev = cfg.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no cuda")
    outdir = os.path.join("results", a.label, "logits", a.model)
    os.makedirs(outdir, exist_ok=True)
    x, y = data.cifar_test() if a.model == "resnet" else data.ag_test()
    m = models.load(a.model, dev)
    rec = {"config": cfg.to_dict(), "model": a.model, "n": len(x), "runs": []}
    for r in range(a.repeat):
        name = cfg.name + a.suffix + ("_again" if r > 0 else "")
        try:
            logits, wall, warm = run_pass(m, x, cfg.batch, dev)
        except Exception as e:  # e.g. deterministic mode with no deterministic kernel
            rec["runs"].append({"name": name, "error": repr(e)})
            print("FAILED", name, repr(e), flush=True)
            continue
        np.save(os.path.join(outdir, name + ".npy"), logits)
        acc = float((logits.argmax(1) == y.numpy()).mean())
        rec["runs"].append({"name": name, "wall_s": wall, "warm_s": warm, "accuracy": acc,
                            "threads": torch.get_num_threads()})
        print(f"{name:<18} {wall:8.2f}s  acc {acc:.4f}", flush=True)
    rec["env"] = env.capture()
    with open(os.path.join(outdir, cfg.name + a.suffix + ".json"), "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
