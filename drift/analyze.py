"""turn saved logits and activation dumps into the numbers: per pair of
configurations, how far apart the logits are, how many predictions flipped,
where the flips sit in the margin distribution, and divergence per layer."""
import glob, json, os
import numpy as np
import torch
from . import configs, data

PCT_EDGES = [0.01, 0.05, 0.25]   # margin buckets: lowest 1%, 1-5%, 5-25%, top 75%
BUCKET_NAMES = ["lowest 1%", "1-5%", "5-25%", "top 75%"]


def margin_of(logits):
    s = np.sort(logits, axis=1)
    return s[:, -1] - s[:, -2]


def compare(a, b):
    """a = base logits, b = other logits, both [n,c] float32"""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    per_img_max = d.max(1)
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)
    same_bits = (a.view(np.uint32) == b.view(np.uint32)).all(1)
    pa, pb = a.argmax(1), b.argmax(1)
    return {
        "abs_max": float(d.max()),
        "abs_median": float(np.median(d)),
        "abs_median_per_image_max": float(np.median(per_img_max)),
        "rel_max": float(rel.max()),
        "rel_median": float(np.median(rel)),
        "identical_images": int(same_bits.sum()),
        "n": int(len(a)),
        "flips": int((pa != pb).sum()),
    }


def margin_buckets(margin):
    edges = np.quantile(margin, PCT_EDGES)
    bucket = np.searchsorted(edges, margin, side="right")   # 0..3
    return bucket, edges


def flip_analysis(a, b, y, names):
    pa, pb = a.argmax(1), b.argmax(1)
    margin = margin_of(a)
    bucket, edges = margin_buckets(margin)
    flipped = np.nonzero(pa != pb)[0]
    rows = []
    for k in range(4):
        n = int((bucket == k).sum()); f = int(((bucket == k) & (pa != pb)).sum())
        rows.append({"bucket": BUCKET_NAMES[k], "n": n, "flips": f, "rate": f / n if n else 0.0})
    order = np.argsort(margin)
    rank = np.empty(len(margin), dtype=np.int64); rank[order] = np.arange(len(margin))
    items = []
    for i in flipped:
        items.append({"index": int(i), "label": names[int(y[i])], "base_pred": names[int(pa[i])],
                      "other_pred": names[int(pb[i])], "margin": float(margin[i]),
                      "margin_pct": float(rank[i] / len(margin)), "bucket": BUCKET_NAMES[int(bucket[i])],
                      "base_correct": bool(pa[i] == y[i]), "other_correct": bool(pb[i] == y[i])})
    items.sort(key=lambda r: r["margin"])
    # how close did it come: per image, the disagreement as a fraction of the margin
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).max(1)
    ratio = diff / np.maximum(margin, 1e-30)
    second = np.argsort(a, axis=1)[:, -2]
    top = np.argsort(-ratio)[:24]
    closest = [{"index": int(i), "label": names[int(y[i])], "pred": names[int(pa[i])], "runner_up": names[int(second[i])],
                "margin": float(margin[i]), "diff": float(diff[i]), "ratio": float(ratio[i]),
                "margin_pct": float(rank[i] / len(margin))} for i in top]
    return {"edges": [float(e) for e in edges], "buckets": rows, "flipped": items,
            "flips_in_lowest_5pct": int(sum(1 for r in items if r["margin_pct"] < 0.05)),
            "min_margin": float(margin.min()), "closest": closest,
            "nearest_miss_ratio": float(ratio.max()),
            "n_within_10pct": int((ratio > 0.1).sum()), "n_within_1pct": int((ratio > 0.01).sum())}


def load_logits(label, model, name):
    p = os.path.join("results", label, "logits", model, name + ".npy")
    return np.load(p) if os.path.exists(p) else None


def load_timing(label, model):
    out = {}
    for p in glob.glob(os.path.join("results", label, "logits", model, "*.json")):
        rec = json.load(open(p))
        for r in rec["runs"]:
            out[r["name"]] = dict(r, config=rec["config"])
    return out


def analyze_model(label, model, cfgset):
    y = (data.cifar_test() if model == "resnet" else data.ag_test())[1].numpy()
    names = data.CIFAR_CLASSES if model == "resnet" else data.AG_CLASSES
    timing = load_timing(label, model)
    pairs = configs.pairs_for(cfgset)
    pairs = [("ref", "ref_again", "same config again, same process"),
             ("ref", "ref_proc", "same config again, new process")] + pairs
    if "b128_again" in timing:
        pairs += [("b128", "b128_again", "batch 128 again, same process"),
                  ("b128", "b128_proc", "batch 128 again, new process")]
    cache = {}

    def L(n):
        if n not in cache:
            cache[n] = load_logits(label, model, n)
        return cache[n]
    out = {"model": model, "n": int(len(y)), "classes": names, "configs": {}, "pairs": [], "not_run": []}
    for c in cfgset:
        out["configs"][c.name] = c.to_dict()
    for n, t in timing.items():
        out["configs"].setdefault(n, t["config"])
        if "error" in t:
            out["not_run"].append({"name": n, "error": t["error"]})
    for base, other, knob in pairs:
        a, b = L(base), L(other)
        if a is None or b is None:
            out["not_run"].append({"pair": [base, other], "knob": knob,
                                   "reason": "no logits for " + (base if a is None else other)})
            continue
        rec = {"base": base, "other": other, "knob": knob, "stats": compare(a, b),
               "flip": flip_analysis(a, b, y, names),
               "wall_s": {base: timing[base]["wall_s"], other: timing[other]["wall_s"]},
               "accuracy": {base: timing[base]["accuracy"], other: timing[other]["accuracy"]}}
        out["pairs"].append(rec)
    ref = L("ref")
    if ref is not None:
        m = margin_of(ref)
        out["reference"] = {"margin": [round(float(v), 6) for v in m],
                            "pred": ref.argmax(1).tolist(), "label": y.tolist(),
                            "accuracy": float((ref.argmax(1) == y).mean())}
        anyflip = np.zeros(len(y), dtype=bool)
        for p in out["pairs"]:
            if p["base"] == "ref":
                for r in p["flip"]["flipped"]:
                    anyflip[r["index"]] = True
        bucket, _ = margin_buckets(m)
        out["any_flip_vs_ref"] = {"n": int(anyflip.sum()),
                                  "in_lowest_5pct": int((anyflip & (bucket <= 1)).sum()),
                                  "per_bucket": [int((anyflip & (bucket == k)).sum()) for k in range(4)],
                                  "indices": np.nonzero(anyflip)[0].tolist()}
        out["margin_edges"] = [float(e) for e in np.quantile(m, PCT_EDGES)]
    return out


def depth_stats(label, model, base, other):
    root = os.path.join("results", label, ".depth_tmp", model)
    da, db = os.path.join(root, base), os.path.join(root, other)
    if not (os.path.isdir(da) and os.path.isdir(db)):
        return None
    fa = sorted(os.listdir(da)); fb = sorted(os.listdir(db))
    if fa != fb:
        return {"base": base, "other": other, "error": "layer lists differ"}
    layers = []
    for f in fa:
        a = torch.load(os.path.join(da, f)).double(); b = torch.load(os.path.join(db, f)).double()
        n = a.shape[0]
        d = (a - b).abs().reshape(n, -1)
        an = a.reshape(n, -1).norm(dim=1); dn = (a - b).reshape(n, -1).norm(dim=1)
        rel = dn / an.clamp(min=1e-30)
        layers.append({"layer": f[4:-3], "numel": int(a[0].numel()),
                       "abs_max": float(d.max()), "abs_mean": float(d.mean()),
                       "abs_max_per_input_median": float(d.max(1).values.median()),
                       "rel_l2_median": float(rel.median()), "rel_l2_mean": float(rel.mean()),
                       "rel_l2_max": float(rel.max()),
                       "identical_inputs": int((d.max(1).values == 0).sum()),
                       "n_inputs": int(n)})
    return {"base": base, "other": other, "layers": layers}


def main(label):
    cfgset = configs.gpu_configs(os.cpu_count()) if torch.cuda.is_available() else configs.cpu_configs(os.cpu_count())
    outdir = os.path.join("results", label)
    for model in ("resnet", "text"):
        if not os.path.isdir(os.path.join(outdir, "logits", model)):
            continue
        res = analyze_model(label, model, cfgset)
        with open(os.path.join(outdir, f"compare_{model}.json"), "w") as f:
            json.dump(res, f, indent=0)
        dpath = os.path.join(outdir, f"depth_{model}.json")
        if not os.path.isdir(os.path.join(outdir, ".depth_tmp", model)) and os.path.exists(dpath):
            print(model, "pairs", len(res["pairs"]), "(depth json kept, no dumps present)", flush=True)
            continue
        depth = {"model": model, "pairs": [], "not_run": []}
        for base, other in configs.depth_pairs(cfgset):
            d = depth_stats(label, model, base, other)
            if d:
                depth["pairs"].append(d)
            else:
                depth["not_run"].append([base, other])
        with open(dpath, "w") as f:
            json.dump(depth, f, indent=1)
        print(model, "pairs", len(res["pairs"]), "depth pairs", len(depth["pairs"]), flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
