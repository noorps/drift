"""the driver: one fresh process per configuration (so env-level knobs take
effect and nothing leaks between runs), then the activation dumps, then the
analysis. results land in results/<label>/.

    python -m drift.run --label L [--models resnet,text] [--depth-n 64]
"""
import argparse, json, os, shutil, subprocess, sys, time
import torch
from . import configs, env


def spawn(args, extra_env):
    e = dict(os.environ); e.update(extra_env); e["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m"] + args, env=e)
    return r.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--models", default="resnet,text")
    ap.add_argument("--depth-n", type=int, default=64)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-depth", action="store_true")
    a = ap.parse_args()
    gpu = torch.cuda.is_available()
    cfgset = configs.gpu_configs(os.cpu_count()) if gpu else configs.cpu_configs(os.cpu_count())
    outdir = os.path.join("results", a.label)
    os.makedirs(outdir, exist_ok=True)
    log = {"label": a.label, "env": env.capture(), "configs": [c.to_dict() for c in cfgset], "steps": []}
    print(json.dumps(log["env"], indent=1), flush=True)
    for model in a.models.split(","):
        if not a.skip_sweep:
            for c in cfgset:
                # the reference (and batch 128 on a gpu) is run twice in-process and once more in a new process
                rep = 2 if c.name in ("ref", "b128") else 1
                rc, dt = spawn(["drift.sweep", "--label", a.label, "--model", model, "--config", c.name, "--repeat", str(rep)], c.env())
                log["steps"].append({"step": "sweep", "model": model, "config": c.name, "rc": rc, "s": dt})
                if c.name in ("ref", "b128"):
                    rc, dt = spawn(["drift.sweep", "--label", a.label, "--model", model, "--config", c.name, "--suffix", "_proc"], c.env())
                    log["steps"].append({"step": "sweep", "model": model, "config": c.name + "_proc", "rc": rc, "s": dt})
        if not a.skip_depth:
            names = sorted({n for p in configs.depth_pairs(cfgset) for n in p})
            byname = configs.by_name(cfgset)
            for n in names:
                rc, dt = spawn(["drift.depth", "--label", a.label, "--model", model, "--config", n, "--n", str(a.depth_n)], byname[n].env())
                log["steps"].append({"step": "depth", "model": model, "config": n, "rc": rc, "s": dt})
    rc, dt = spawn(["drift.analyze", a.label], {})
    log["steps"].append({"step": "analyze", "rc": rc, "s": dt})
    shutil.rmtree(os.path.join(outdir, ".depth_tmp"), ignore_errors=True)
    log["total_s"] = sum(s["s"] for s in log["steps"])
    with open(os.path.join(outdir, "run.json"), "w") as f:
        json.dump(log, f, indent=1)
    print("failed steps:", [s for s in log["steps"] if s["rc"] != 0], flush=True)


if __name__ == "__main__":
    main()
