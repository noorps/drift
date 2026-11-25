"""drift stage 2: what production inference backends do to speed and to numerics.

stage 1 measured how far two runs of the same model drift apart and established a
per-image margin distribution. this stage runs the same trained resnet18 through
eager, torch.compile and tensorrt at fp32 and fp16, and asks whether any of them
move the logits far enough to cross that margin.

    python serve_stage2.py --label t4 [--batches 1,8,32,128] [--iters 100] [--trt]

run from the drift repo root. reuses drift.data, drift.models and drift.analyze
so the numerics here are computed by the same code stage 1 used.
"""
import argparse, json, os, time
import numpy as np
import torch

from drift import analyze, data, env, models

BACKENDS_NO_INSTALL = ["eager_fp32", "eager_fp16", "compile_fp32", "compile_fp16"]
BACKENDS_TRT = ["trt_fp32", "trt_fp16"]


# ---------------------------------------------------------------- backends

def build(kind, eager, example, dev):
    """returns (callable, build_seconds, note). the callable takes a batch and
    returns float32 logits on the device. build cost is a serving tradeoff, so
    it is measured rather than hidden."""
    t0 = time.perf_counter()
    note = ""

    if kind == "eager_fp32":
        fn = lambda x: eager(x)

    elif kind == "eager_fp16":
        def fn(x):
            with torch.autocast("cuda", dtype=torch.float16):
                return eager(x)

    elif kind == "compile_fp32":
        c = torch.compile(eager)
        c(example)                                  # trigger the compile inside the timer
        torch.cuda.synchronize()
        fn = lambda x: c(x)

    elif kind == "compile_fp16":
        c = torch.compile(eager)
        with torch.autocast("cuda", dtype=torch.float16):
            c(example)
        torch.cuda.synchronize()

        def fn(x):
            with torch.autocast("cuda", dtype=torch.float16):
                return c(x)

    elif kind in ("trt_fp32", "trt_fp16"):
        half = kind.endswith("fp16")
        try:
            import torch_tensorrt
            prec = {torch.float16} if half else {torch.float32}
            # dynamic batch: the test set does not divide evenly by every batch size,
            # so a fixed-shape engine throws on the trailing partial batch.
            s = tuple(example.shape)
            inp = torch_tensorrt.Input(min_shape=(1,) + s[1:], opt_shape=s, max_shape=s,
                                       dtype=torch.float32)
            m = torch_tensorrt.compile(eager, inputs=[inp], enabled_precisions=prec,
                                       truncate_long_and_double=True)
            fn = lambda x: m(x)
            note = "torch-tensorrt"
        except Exception as e:
            fn, note = _trt_via_onnx(eager, example, half)
            note = f"onnx fallback ({type(e).__name__}: {e})"[:300] if fn else note
            if fn is None:
                raise
    else:
        raise ValueError(kind)

    torch.cuda.synchronize()
    return fn, time.perf_counter() - t0, note


def _trt_via_onnx(eager, example, half):
    """fallback when torch-tensorrt will not install against this torch. exports
    to onnx and builds the engine with the tensorrt python api directly."""
    try:
        import tensorrt as trt
    except Exception:
        return None, ""
    path = "/tmp/drift_resnet.onnx"
    torch.onnx.export(eager, example, path, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "b"}, "logits": {0: "b"}}, opset_version=17)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("onnx parse failed: " +
                               "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    if half:
        cfg.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    s = tuple(example.shape)
    profile.set_shape("x", (1,) + s[1:], s, (max(s[0], 128),) + s[1:])
    cfg.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, cfg)
    if engine is None:
        raise RuntimeError("tensorrt engine build returned none")
    rt = trt.Runtime(logger)
    eng = rt.deserialize_cuda_engine(engine)
    ctx = eng.create_execution_context()

    def fn(x):
        x = x.contiguous()
        ctx.set_input_shape("x", tuple(x.shape))
        out = torch.empty((x.shape[0], 10), dtype=torch.float32, device=x.device)
        ctx.set_tensor_address("x", x.data_ptr())
        ctx.set_tensor_address("logits", out.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        return out

    return fn, "onnx + tensorrt python api"


# ---------------------------------------------------------------- measurement

def latency(fn, x, iters, warmup):
    """per-iteration milliseconds, timed with cuda events rather than wall clock
    wrapped around async launches."""
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            fn(x)
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
    t = np.array(ts)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "min": float(t.min()), "mean": float(t.mean()),
            "iters": iters, "warmup": warmup}


def full_pass(fn, x, batch):
    """every test image through the backend, logits kept at float32."""
    out = np.empty((len(x), 10), dtype=np.float32)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(x), batch):
            out[i:i + batch] = fn(x[i:i + batch]).float().cpu().numpy()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    return out, wall


def clocks_locked():
    """try to pin the gpu clocks. colab usually refuses, and that variance
    belongs in the json rather than in an excuse later."""
    rc = os.system("nvidia-smi -lgc 1590 >/dev/null 2>&1")
    return rc == 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--trt", action="store_true", help="also try tensorrt")
    ap.add_argument("--backends", default="", help="comma separated subset; eager_fp32 is "
                                                   "always included because it is the baseline")
    ap.add_argument("--save-logits", action="store_true")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this stage needs a gpu")

    batches = [int(b) for b in a.batches.split(",")]
    outdir = os.path.join("results", a.label)
    os.makedirs(outdir, exist_ok=True)

    # same knobs stage 1's reference used, set explicitly so nothing is defaulted
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(0)

    x, y = data.cifar_test()
    yn = y.numpy()
    xd = x.to("cuda")
    eager = models.load("resnet", "cuda")

    cap = torch.cuda.get_device_capability()
    rec = {
        "label": a.label, "model": "resnet", "n": int(len(x)), "batches": batches,
        "env": env.capture(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": f"{cap[0]}.{cap[1]}",
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "tf32_available": cap[0] >= 8,
            "bf16_tensor_cores": cap[0] >= 8,
            "int8_tensor_cores": cap >= (7, 5),
            "clocks_locked": clocks_locked(),
        },
        "cuda": {"runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "backends": {}, "comparisons": [], "notes": [],
    }
    if not rec["gpu"]["tf32_available"]:
        rec["notes"].append("no tf32 units on this device, so there is no tf32 row to measure")
    if not rec["gpu"]["bf16_tensor_cores"]:
        rec["notes"].append("no bf16 tensor core path on this device, so fp16 is the only "
                            "reduced precision worth timing; a bf16 row would measure emulation")
    if not rec["gpu"]["clocks_locked"]:
        rec["notes"].append("gpu clocks could not be locked, so thermal and neighbour variance "
                            "is inside the latency numbers")

    wanted = list(BACKENDS_NO_INSTALL) + (BACKENDS_TRT if a.trt else [])
    if a.backends:
        pick = [b.strip() for b in a.backends.split(",")]
        wanted = ["eager_fp32"] + [b for b in wanted if b in pick and b != "eager_fp32"]

    # ---- baseline self-check against stage 1, if stage 1 ran under this label
    ref1 = analyze.load_logits(a.label, "resnet", "ref")
    baselines = {}

    for kind in wanted:
        rec["backends"][kind] = {"batches": {}}
        for batch in batches:
            ex = xd[:batch]
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                fn, build_s, note = build(kind, eager, ex, "cuda")
                lat = latency(fn, ex, a.iters, a.warmup)
                logits, wall = full_pass(fn, xd, batch)
                cell = {
                    "latency_ms": lat,
                    "throughput_img_s": batch / (lat["p50"] / 1000.0),
                    "build_s": build_s,
                    "full_pass_s": wall,
                    "accuracy": float((logits.argmax(1) == yn).mean()),
                    "peak_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
                    "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
                    "note": note,
                }
                rec["backends"][kind]["batches"][str(batch)] = cell
                print(f"{kind:<14} b{batch:<4} p50 {lat['p50']:7.3f}ms  p99 {lat['p99']:7.3f}ms  "
                      f"{cell['throughput_img_s']:8.0f} img/s  acc {cell['accuracy']:.4f}"
                      f"{'  ' + note if note else ''}", flush=True)

                if kind == "eager_fp32":
                    baselines[batch] = logits
                    if a.save_logits or batch == 1:
                        d = os.path.join(outdir, "logits", "resnet")
                        os.makedirs(d, exist_ok=True)
                        np.save(os.path.join(d, f"serve_{kind}_b{batch}.npy"), logits)
                else:
                    base = baselines.get(batch)
                    if base is not None:
                        stats = analyze.compare(base, logits)
                        flip = analyze.flip_analysis(base, logits, yn, data.CIFAR_CLASSES)
                        rec["comparisons"].append({
                            "backend": kind, "batch": batch,
                            "base": f"eager_fp32 b{batch}",
                            "stats": stats,
                            "min_margin": flip["min_margin"],
                            "headroom": flip["nearest_miss_ratio"],
                            "flips": stats["flips"],
                            "flips_in_lowest_5pct": flip["flips_in_lowest_5pct"],
                            "n_within_10pct_of_margin": flip["n_within_10pct"],
                            "n_within_1pct_of_margin": flip["n_within_1pct"],
                            "buckets": flip["buckets"],
                            "accuracy_delta": (float((logits.argmax(1) == yn).mean())
                                               - float((base.argmax(1) == yn).mean())),
                            "tolerance_needed": {"abs": stats["abs_max"], "rel": stats["rel_max"]},
                        })
                    if a.save_logits:
                        d = os.path.join(outdir, "logits", "resnet")
                        os.makedirs(d, exist_ok=True)
                        np.save(os.path.join(d, f"serve_{kind}_b{batch}.npy"), logits)
            except Exception as e:
                rec["backends"][kind]["batches"][str(batch)] = {"error": f"{type(e).__name__}: {e}"}
                print(f"{kind:<14} b{batch:<4} FAILED {type(e).__name__}: {e}", flush=True)

    # if the fp16 engine produced bit-identical numbers to the fp32 one, the fp16
    # kernels were almost certainly never selected. say so rather than reporting
    # an fp16 result that is really an fp32 one.
    byname = {(c["backend"], c["batch"]): c for c in rec["comparisons"]}
    for batch in batches:
        c32, c16 = byname.get(("trt_fp32", batch)), byname.get(("trt_fp16", batch))
        if c32 and c16 and c32["stats"]["abs_max"] == c16["stats"]["abs_max"]:
            rec["notes"].append(
                "trt_fp16 and trt_fp32 agree to the last bit at batch %d, so the fp16 "
                "kernels were probably not selected; treat the fp16 row as unconfirmed" % batch)

    if ref1 is not None and 1 in baselines:
        same = int((baselines[1].view(np.uint32) == ref1.view(np.uint32)).all(1).sum())
        rec["stage1_baseline_check"] = {
            "compared_against": "results/%s/logits/resnet/ref.npy" % a.label,
            "bitwise_identical_images": same, "n": int(len(ref1)),
            "reproduces_stage1": same == len(ref1),
            "abs_max": float(np.abs(baselines[1].astype(np.float64) - ref1.astype(np.float64)).max()),
        }
        print("\nstage 1 baseline check: %d/%d images bitwise identical" % (same, len(ref1)), flush=True)
    else:
        rec["stage1_baseline_check"] = {"skipped": "no stage 1 ref logits under this label"}

    with open(os.path.join(outdir, "serve.json"), "w") as f:
        json.dump(rec, f, indent=1)

    summary(rec, outdir)
    print("\nwrote", os.path.join(outdir, "serve.json"), flush=True)


def summary(rec, outdir):
    """lowercase prose, every number from rec. nothing typed by hand."""
    L = []
    g = rec["gpu"]
    L.append("# inference backends\n")
    L.append("one trained resnet18, %d cifar-10 test images, on a %s (compute %s). "
             "each backend is diffed against eager fp32 at the same batch size, so the "
             "comparison isolates the backend rather than the batch.\n"
             % (rec["n"], g["name"], g["capability"]))

    L.append("\n## speed\n")
    L.append("| backend | batch | p50 ms | p99 ms | img/s | build s | peak mb | accuracy |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k, b in rec["backends"].items():
        for bs, c in b["batches"].items():
            if "error" in c:
                L.append("| %s | %s | | | | | | %s |" % (k, bs, c["error"][:40]))
                continue
            l = c["latency_ms"]
            L.append("| %s | %s | %.3f | %.3f | %.0f | %.1f | %.0f | %.4f |"
                     % (k, bs, l["p50"], l["p99"], c["throughput_img_s"], c["build_s"],
                        c["peak_alloc_mb"], c["accuracy"]))

    if rec["comparisons"]:
        L.append("\n## numerics against eager fp32\n")
        L.append("headroom is the largest disagreement this backend produced as a fraction of "
                 "the image's own top-two margin. below 1 means no prediction could change.\n")
        L.append("| backend | batch | abs max | rel max | identical | flips | headroom | acc delta |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for c in rec["comparisons"]:
            s = c["stats"]
            L.append("| %s | %d | %.3g | %.3g | %d | %d | %.3g | %+.4f |"
                     % (c["backend"], c["batch"], s["abs_max"], s["rel_max"],
                        s["identical_images"], c["flips"], c["headroom"], c["accuracy_delta"]))

        L.append("\n## tolerance you would need\n")
        L.append("to compare a backend against eager fp32 in a test without false failures.\n")
        L.append("| backend | batch | atol | rtol |")
        L.append("| --- | ---: | ---: | ---: |")
        for c in rec["comparisons"]:
            t = c["tolerance_needed"]
            L.append("| %s | %d | %.3g | %.3g |" % (c["backend"], c["batch"], t["abs"], t["rel"]))

    if rec["notes"]:
        L.append("\n## about this machine\n")
        for n in rec["notes"]:
            L.append("- %s" % n)

    c = rec.get("stage1_baseline_check", {})
    if "reproduces_stage1" in c:
        L.append("\n## baseline check\n")
        L.append("- eager fp32 at batch 1 reproduced stage 1's reference logits on %d of %d "
                 "images bitwise%s" % (c["bitwise_identical_images"], c["n"],
                                       "" if c["reproduces_stage1"] else
                                       " (max abs difference %.3g)" % c["abs_max"]))

    p = os.path.join(outdir, "serve_summary.md")
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return p


if __name__ == "__main__":
    main()
