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


def serve_tables(rec):
    """the inference backends tables, from serve.json written by serve_stage2.py."""
    ok = [(k, int(bs), c) for k, b in rec["backends"].items() for bs, c in b["batches"].items() if "error" not in c]
    speed = ["| backend | batch | p50 ms | p99 ms | img/s | build s | peak mb | accuracy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, b in rec["backends"].items():
        for bs, c in b["batches"].items():
            if "error" in c:
                speed.append(f"| {k} | {bs} | failed: {c['error'][:60]} | | | | | |")
                continue
            l = c["latency_ms"]
            speed.append(f"| {k} | {bs} | {l['p50']:.3f} | {l['p99']:.3f} | {c['throughput_img_s']:,.0f} | "
                         f"{c['build_s']:.1f} | {c['peak_alloc_mb']:.0f} | {100 * c['accuracy']:.2f}% |")

    cmp = rec["comparisons"]
    num = ["| backend | batch | max abs | max rel | bitwise identical | flips | headroom | accuracy change |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cmp:
        s = c["stats"]
        num.append(f"| {c['backend']} | {c['batch']} | {f(s['abs_max'])} | {f(s['rel_max'])} | "
                   f"{s['identical_images']:,} / {s['n']:,} | {c['flips']} | {f(c['headroom'])} | "
                   f"{100 * c['accuracy_delta']:+.2f} pts |")

    tol = ["| backend | atol | rtol |", "|---|---:|---:|"]
    for k in dict.fromkeys(c["backend"] for c in cmp):
        cs = [c for c in cmp if c["backend"] == k]
        tol.append(f"| {k} | {f(max(c['tolerance_needed']['abs'] for c in cs))} | "
                   f"{f(max(c['tolerance_needed']['rel'] for c in cs))} |")

    notes = [f"- {n}" for n in rec.get("notes", [])]
    chk = rec.get("stage1_baseline_check", {})
    if "reproduces_stage1" in chk:
        notes.append(f"- eager fp32 at batch 1 reproduced stage 1's reference logits bitwise on "
                     f"{chk['bitwise_identical_images']:,} of {chk['n']:,} images"
                     + ("" if chk["reproduces_stage1"] else f" (max abs difference {f(chk['abs_max'])})"))
    elif chk:
        notes.append(f"- stage 1 baseline check skipped: {chk.get('skipped', '')}")

    # one generated sentence per finding, so the prose around the tables carries no typed numbers
    head = []
    by = {}
    for k, bs, c in ok:
        by.setdefault(bs, []).append((k, c))
    lo, hi = min(by), max(by)
    if len(by[lo]) > 1:
        fast = min(by[lo], key=lambda t: t[1]["latency_ms"]["p50"])
        base = dict(by[lo]).get("eager_fp32")
        if base:
            head.append(f"at batch {lo} the fastest backend was {fast[0]}, {fast[1]['latency_ms']['p50']:.3f} ms p50 "
                        f"against eager fp32's {base['latency_ms']['p50']:.3f} ms "
                        f"({base['latency_ms']['p50'] / fast[1]['latency_ms']['p50']:.1f}x).")
    if hi != lo and len(by[hi]) > 1:
        r = sorted(by[hi], key=lambda t: -t[1]["throughput_img_s"])
        head.append(f"at batch {hi} the fastest was {r[0][0]} at {r[0][1]['throughput_img_s']:,.0f} img/s "
                    f"and the slowest {r[-1][0]} at {r[-1][1]['throughput_img_s']:,.0f} img/s.")
    if cmp:
        worst = max(cmp, key=lambda c: c["flips"])
        fp32 = [c for c in cmp if c["backend"].endswith("fp32")]
        if worst["flips"]:
            head.append(f"the most predictions any backend changed was {worst['flips']} of {rec['n']:,} "
                        f"({worst['backend']} at batch {worst['batch']}).")
        else:
            head.append(f"no backend changed a single one of the {rec['n']:,} predictions.")
        if fp32:
            m = max(fp32, key=lambda c: c["headroom"])
            head.append(f"the largest fp32 disagreement reached {f(m['headroom'])} of the margin it would need to "
                        f"cross, and fp32 flipped {sum(c['flips'] for c in fp32)} predictions in total.")
    return {"serve_speed": "\n".join(speed), "serve_numerics": "\n".join(num), "serve_tolerance": "\n".join(tol),
            "serve_notes": "\n".join(notes), "serve_headline": " ".join(head)}


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
    sp = os.path.join(d, "serve.json")
    if os.path.exists(sp):
        t.update(serve_tables(json.load(open(sp))))
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
