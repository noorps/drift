"""web/data.js and web/img/ from a results folder. the page reads nothing else.

    python -m drift.export_web <label> [web]
"""
import json, os, shutil, sys


def slim_pair(p):
    return {"base": p["base"], "other": p["other"], "knob": p["knob"], "stats": p["stats"],
            "buckets": p["flip"]["buckets"], "edges": p["flip"]["edges"],
            "flips_in_lowest_5pct": p["flip"]["flips_in_lowest_5pct"],
            "flipped": [{k: r[k] for k in ("index", "label", "base_pred", "other_pred", "margin", "margin_pct", "bucket")}
                        for r in p["flip"]["flipped"]],
            "wall_s": p["wall_s"], "accuracy": p["accuracy"],
            "min_margin": p["flip"].get("min_margin"), "closest": p["flip"].get("closest", []),
            "nearest_miss_ratio": p["flip"].get("nearest_miss_ratio"),
            "n_within_10pct": p["flip"].get("n_within_10pct"), "n_within_1pct": p["flip"].get("n_within_1pct")}


def main(label, web="web"):
    d = os.path.join("results", label)
    run = json.load(open(os.path.join(d, "run.json")))
    out = {"label": label, "env": run["env"], "models": {}}
    for model in ("resnet", "text"):
        cp = os.path.join(d, f"compare_{model}.json")
        if not os.path.exists(cp):
            continue
        cmp = json.load(open(cp))
        m = {"n": cmp["n"], "classes": cmp["classes"], "configs": cmp["configs"],
             "pairs": [slim_pair(p) for p in cmp["pairs"]], "not_run": cmp.get("not_run", []),
             "accuracy": cmp.get("reference", {}).get("accuracy"),
             "any_flip": {k: v for k, v in cmp.get("any_flip_vs_ref", {}).items() if k != "indices"},
             "margin_edges": cmp.get("margin_edges"), "depth": []}
        dp = os.path.join(d, f"depth_{model}.json")
        if os.path.exists(dp):
            for p in json.load(open(dp))["pairs"]:
                if "layers" in p:
                    m["depth"].append({"base": p["base"], "other": p["other"],
                                       "layers": [l["layer"] for l in p["layers"]],
                                       "rel": [l["rel_l2_median"] for l in p["layers"]],
                                       "abs": [l["abs_max"] for l in p["layers"]],
                                       "n_inputs": p["layers"][0]["n_inputs"]})
        out["models"][model] = m
    os.makedirs(os.path.join(web, "img"), exist_ok=True)
    src = os.path.join(d, "flips")
    n = 0
    if os.path.isdir(src):
        for fn in os.listdir(src):
            if fn[:-4].isdigit():
                shutil.copy(os.path.join(src, fn), os.path.join(web, "img", fn)); n += 1
    with open(os.path.join(web, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.DRIFT = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"web/data.js written ({os.path.getsize(os.path.join(web, 'data.js')) // 1024} kb), {n} flip images")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
