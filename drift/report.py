"""summary.md for a results folder, and the README tables filled from the
same json. every number in the README passes through here; none is typed."""
import json, os, re, sys


def f(x, nd=2):
    if x == 0:
        return "0"
    return f"{x:.{nd}e}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{nd + 1}g}"


def env_table(run):
    e = run["env"]
    rows = [("gpu", e.get("gpu", "none")), ("driver", e.get("driver", "")), ("cuda", e.get("cuda", "")),
            ("cudnn", e.get("cudnn", "")), ("cpu", f"{e['cpu']} ({e['cpu_count']} logical)"),
            ("torch", e["torch"]), ("torchvision", e["torchvision"]), ("python", e["python"]), ("os", e["platform"])]
    return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows if v != "")


def knob_table(cmp):
    w = cmp["configs"]
    out = ["| knob | compared against | max abs | median abs | max rel | median rel | bitwise identical | flips | time (base, other) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for p in cmp["pairs"]:
        s = p["stats"]
        out.append(f"| {p['knob']} | {w[p['base']]['words']} | {f(s['abs_max'])} | {f(s['abs_median'])} | {f(s['rel_max'])} | "
                   f"{f(s['rel_median'])} | {s['identical_images']:,} / {s['n']:,} | {s['flips']} | "
                   f"{p['wall_s'][p['base']]:.1f} s, {p['wall_s'][p['other']]:.1f} s |")
    for nr in cmp.get("not_run", []):
        if "pair" in nr:
            out.append(f"| {nr['knob']} | not run: {nr['reason']} | | | | | | | |")
        else:
            out.append(f"| {nr['name']} | failed: {nr['error'][:80]} | | | | | | | |")
    return "\n".join(out)


def margin_table(cmp):
    pairs = [p for p in cmp["pairs"] if p["stats"]["flips"] > 0]
    if not pairs:
        return "no pair flipped a prediction."
    b = pairs[0]["flip"]["buckets"]; e = pairs[0]["flip"]["edges"]
    head = "| knob | flips | " + " | ".join(f"{x['bucket']} (n={x['n']:,})" for x in b) + " | flips in lowest 5% |"
    out = [head, "|---|---|" + "---|" * (len(b) + 1)]
    for p in pairs:
        cells = " | ".join(f"{x['flips']} ({100 * x['rate']:.2f}%)" for x in p["flip"]["buckets"])
        out.append(f"| {p['knob']} (vs {cmp['configs'][p['base']]['words']}) | {p['stats']['flips']} | {cells} | {p['flip']['flips_in_lowest_5pct']} |")
    out.append(f"\nbucket edges on the reference margin: {e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}.")
    return "\n".join(out)


def depth_table(depth, cmp):
    out = ["| pair | layers | first layer rel | last layer rel | ratio | first layer abs max | last layer abs max | ratio |",
           "|---|---|---|---|---|---|---|---|"]
    for p in depth["pairs"]:
        if "layers" not in p:
            continue
        L = [l for l in p["layers"] if l["rel_l2_median"] > 0]
        if not L:
            out.append(f"| {cmp['configs'][p['base']]['words']} vs {cmp['configs'][p['other']]['words']} | {len(p['layers'])} | bitwise identical at every layer | | | | | |")
            continue
        a, z = L[0], L[-1]
        out.append(f"| {cmp['configs'][p['base']]['words']} vs {cmp['configs'][p['other']]['words']} | {len(p['layers'])} | "
                   f"{f(a['rel_l2_median'])} | {f(z['rel_l2_median'])} | {z['rel_l2_median'] / a['rel_l2_median']:.1f}x | "
                   f"{f(a['abs_max'])} | {f(z['abs_max'])} | {z['abs_max'] / a['abs_max']:.1f}x |")
    for nr in depth.get("not_run", []):
        out.append(f"| {nr[0]} vs {nr[1]} | not run | | | | | | |")
    return "\n".join(out)


def headline(cmp):
    a = cmp.get("any_flip_vs_ref")
    if not a:
        return "no reference run."
    return (f"{a['n']} of {cmp['n']:,} came out with a different answer in at least one configuration, "
            f"and {a['in_lowest_5pct']} of those were in the lowest-margin 5%.")


def tables(label):
    d = os.path.join("results", label)
    run = json.load(open(os.path.join(d, "run.json")))
    t = {"env": env_table(run), "label": label}
    for model in ("resnet", "text"):
        cp = os.path.join(d, f"compare_{model}.json")
        if not os.path.exists(cp):
            continue
        cmp = json.load(open(cp))
        t[f"knobs_{model}"] = knob_table(cmp)
        t[f"margin_{model}"] = margin_table(cmp)
        t[f"headline_{model}"] = headline(cmp)
        acc = cmp.get("reference", {}).get("accuracy")
        t[f"acc_{model}"] = f"{100 * acc:.2f}%" if acc is not None else "n/a"
        dp = os.path.join(d, f"depth_{model}.json")
        if os.path.exists(dp):
            t[f"depth_{model}"] = depth_table(json.load(open(dp)), cmp)
    return t


def summary(label):
    t = tables(label)
    s = [f"# drift results: {label}\n", "## environment\n", t["env"], ""]
    for model, ttl in (("resnet", "resnet18 on cifar-10 (10,000 test images)"), ("text", "small transformer on ag news (7,600 test texts)")):
        if f"knobs_{model}" not in t:
            continue
        s += [f"## {ttl}\n", f"reference accuracy {t[f'acc_{model}']}. {t[f'headline_{model}']}\n",
              "### what each knob does to the logits\n", t[f"knobs_{model}"], "",
              "### flips by margin bucket\n", t[f"margin_{model}"], ""]
        if f"depth_{model}" in t:
            s += ["### divergence through depth\n", t[f"depth_{model}"], ""]
    open(os.path.join("results", label, "summary.md"), "w", encoding="utf-8").write("\n".join(s))
    return t


def fill_readme(label, path="README.md"):
    t = tables(label)
    s = open(path, encoding="utf-8").read()
    for k, v in t.items():
        s = re.sub(rf"(<!-- TABLE: {k} -->\n).*?(<!-- /TABLE -->)", lambda m: m.group(1) + v + "\n" + m.group(2), s, flags=re.S)
    open(path, "w", encoding="utf-8").write(s)


if __name__ == "__main__":
    summary(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--readme":
        fill_readme(sys.argv[1])
    print("summary.md written")
