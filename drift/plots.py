"""the three charts, drawn from the json beside them. palette chosen to pass
a colour-vision check on the site's #FAF8F1 background."""
import json, os
import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt

BG, PANEL, TEXT, MUTED, RULE = "#FAF8F1", "#F2F0E4", "#2B2E28", "#6E7566", "#E7E2D4"
SERIES = ["#2F7F4F", "#3B6FD1", "#D07A2A", "#8A4FC8", "#C8463A", "#0E9AAE"]

plt.rcParams.update({
    "font.family": ["DM Sans", "Segoe UI", "Helvetica", "Arial", "sans-serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": RULE, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9,
})


def words(cmp, name):
    return cmp["configs"][name]["words"]


def plot_depth(depth, cmp, out, title_model):
    pairs = [p for p in depth["pairs"] if "layers" in p]
    if not pairs:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    base = pairs[0]["base"]
    for k, (ax, key, ttl) in enumerate(zip(axes, ("abs_max", "rel_l2_median"),
                                           ("absolute: max |a - b| over the layer's activations",
                                            "relative: median over inputs of ||a - b|| / ||a||"))):
        for i, p in enumerate(pairs):
            ys = np.array([l[key] for l in p["layers"]], dtype=float)
            xs = np.arange(len(ys))
            ys = np.where(ys > 0, ys, np.nan)   # exactly zero is off a log axis; shown as a gap
            ax.plot(xs, ys, color=SERIES[i % 6], lw=1.8, label=words(cmp, p["other"]))
            if np.isfinite(ys).any() and k == 1:
                j = np.nanargmax(np.where(np.isfinite(ys), xs, -1))
                ax.annotate(words(cmp, p["other"]), (xs[j], ys[j]), xytext=(4, 0), textcoords="offset points",
                            fontsize=8, color=SERIES[i % 6], va="center")
        ax.set_yscale("log")
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("layer, in forward order")
        layers = pairs[0]["layers"]
        ticks = [i for i, l in enumerate(layers) if l["layer"].count(".") == 1 and "#" in l["layer"] and l["layer"].endswith("#0") and l["layer"].split("#")[0].count(".") == 1 and not any(s in l["layer"] for s in ("conv", "bn", "relu", "downsample", "ln", "attn", "ff", "act", "qkv", "out"))]
        ticks = [0] + ticks + [len(layers) - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([layers[i]["layer"].split("#")[0] for i in ticks], rotation=30, ha="right", fontsize=8)
        if np.isfinite(ys).all() or True:
            pass
    axes[0].legend(loc="lower right", title=f"{words(cmp, base)}  vs:", title_fontsize=9)
    axes[1].set_xlim(0, len(pairs[0]["layers"]) * 1.25)
    fig.suptitle(f"{title_model}: how far two runs drift apart, layer by layer ({len(layers)} hook points, {pairs[0]['layers'][0]['n_inputs']} inputs)",
                 x=0.01, ha="left", fontsize=12, fontweight="medium")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_flips(cmp, out, title_model, max_pairs=6):
    pairs = [p for p in cmp["pairs"] if p["base"] == "ref" and p["stats"]["flips"] > 0]
    pairs.sort(key=lambda p: -p["stats"]["flips"])
    pairs = pairs[:max_pairs]
    if not pairs:
        return False
    buckets = [b["bucket"] for b in pairs[0]["flip"]["buckets"]]
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    w = 0.8 / len(pairs)
    for i, p in enumerate(pairs):
        rates = [100 * b["rate"] for b in p["flip"]["buckets"]]
        xs = np.arange(len(buckets)) + (i - (len(pairs) - 1) / 2) * w
        bars = ax.bar(xs, rates, width=w * 0.92, color=SERIES[i % 6], label=f"{p['knob']} ({p['stats']['flips']} flips)")
        for b, bk in zip(bars, p["flip"]["buckets"]):
            if bk["flips"]:
                ax.annotate(str(bk["flips"]), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=7, color=MUTED)
    n = cmp["n"]
    edges = pairs[0]["flip"]["edges"]
    labels = [f"{buckets[0]}\nmargin < {edges[0]:.2f}", f"{buckets[1]}\n{edges[0]:.2f} to {edges[1]:.2f}",
              f"{buckets[2]}\n{edges[1]:.2f} to {edges[2]:.2f}", f"{buckets[3]}\nmargin > {edges[2]:.2f}"]
    ax.set_xticks(range(len(buckets))); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("images whose top-1 class changed, % of bucket")
    ax.set_xlabel(f"bucket by the reference run's margin between its top two logits (n = {n:,} per model)")
    ax.legend(loc="upper right", title=f"vs {words(cmp, 'ref')}", title_fontsize=9)
    ax.set_title(f"{title_model}: where the flipped predictions sit", loc="left", fontsize=12, fontweight="medium")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_cost(cmp, out, title_model):
    pairs = [p for p in cmp["pairs"]]
    if not pairs:
        return False
    rows = []
    for p in pairs:
        rows.append((p["knob"], p["stats"]["rel_max"], p["wall_s"][p["other"]] / p["wall_s"][p["base"]],
                     p["stats"]["identical_images"] == p["stats"]["n"], p["base"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(rows) + 1.6), constrained_layout=True, sharey=True)
    ys = np.arange(len(rows))[::-1]
    col = [SERIES[0] if r[4] == "ref" else SERIES[1] if r[4] == "cpu" else SERIES[2] for r in rows]
    ax = axes[0]
    vals = np.array([r[1] for r in rows])
    floor = max(vals[vals > 0].min() / 10 if (vals > 0).any() else 1e-9, 1e-9)
    ax.barh(ys, np.where(vals > 0, vals, floor), color=col, height=0.62)
    ax.set_xscale("log"); ax.set_xlim(left=floor)
    for y, r, v in zip(ys, rows, vals):
        ax.annotate("bitwise identical" if r[3] else f"{v:.1e}", (max(v, floor), y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_title("largest relative change in any image's logits", loc="left")
    ax.set_xlabel("max over images of ||a - b|| / ||a||")
    ax = axes[1]
    ratio = np.array([r[2] for r in rows])
    ax.barh(ys, ratio, color=col, height=0.62)
    ax.axvline(1, color=MUTED, lw=0.8, ls="--")
    for y, r in zip(ys, ratio):
        ax.annotate(f"{r:.2f}x", (r, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.set_title("wall-clock, relative to the run it is compared against", loc="left")
    ax.set_xlabel("time for the full test set / time of the base run")
    ax.set_xlim(0, max(ratio.max() * 1.25, 1.3))
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=SERIES[0], label=f"base: {words(cmp, 'ref')}"),
                            Patch(color=SERIES[1], label=f"base: {words(cmp, 'cpu')}" if "cpu" in cmp["configs"] else "base: cpu"),
                            Patch(color=SERIES[2], label=f"base: {words(cmp, 'b128')}" if "b128" in cmp["configs"] else "base: batch 128")],
                   loc="lower left", fontsize=8)
    fig.suptitle(f"{title_model}: what each knob costs", x=0.01, ha="left", fontsize=12, fontweight="medium")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main(label):
    outdir = os.path.join("results", label)
    made = []
    for model, ttl in (("resnet", "resnet18 on cifar-10"), ("text", "small transformer on ag news")):
        cp = os.path.join(outdir, f"compare_{model}.json")
        if not os.path.exists(cp):
            continue
        cmp = json.load(open(cp))
        sfx = "" if model == "resnet" else "_text"
        dp = os.path.join(outdir, f"depth_{model}.json")
        if os.path.exists(dp) and plot_depth(json.load(open(dp)), cmp, os.path.join(outdir, f"depth{sfx}.png"), ttl):
            made.append(f"depth{sfx}.png")
        if plot_flips(cmp, os.path.join(outdir, f"flips_by_margin{sfx}.png"), ttl):
            made.append(f"flips_by_margin{sfx}.png")
        if plot_cost(cmp, os.path.join(outdir, f"knob_cost{sfx}.png"), ttl):
            made.append(f"knob_cost{sfx}.png")
    print("charts:", made)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
