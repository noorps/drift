"""Configurations: one named set of knobs each. Every knob is applied in a fresh
process so that env-level settings (CUBLAS_WORKSPACE_CONFIG) take effect."""
from dataclasses import dataclass, asdict
import os


@dataclass(frozen=True)
class Config:
    name: str
    device: str = "cuda"
    batch: int = 1
    tf32: bool = False               # both matmul.allow_tf32 and cudnn.allow_tf32
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False
    deterministic: bool = False      # torch.use_deterministic_algorithms(True)
    threads: int = 0                 # 0 = leave torch's default

    def env(self):
        e = {}
        if self.deterministic:
            e["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        return e

    def apply(self):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.backends.cudnn.allow_tf32 = self.tf32
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.use_deterministic_algorithms(self.deterministic)
        if self.threads:
            torch.set_num_threads(self.threads)

    def words(self):
        """plain-language label"""
        parts = ["gpu" if self.device == "cuda" else "cpu", f"batch {self.batch}"]
        if self.device == "cuda":
            parts.append("tf32 on" if self.tf32 else "tf32 off")
            if self.cudnn_benchmark:
                parts.append("cudnn autotune")
            if self.cudnn_deterministic:
                parts.append("cudnn deterministic")
        if self.deterministic:
            parts.append("deterministic mode")
        if self.threads:
            parts.append(f"{self.threads} thread" + ("s" if self.threads > 1 else ""))
        return ", ".join(parts)

    def to_dict(self):
        d = asdict(self)
        d["words"] = self.words()
        return d


def gpu_configs(ncpu):
    """the configurations swept on a machine with a gpu. the reference is the
    first one: gpu, batch 1, everything off."""
    cs = [
        Config("ref"),
        Config("b8", batch=8),
        Config("b32", batch=32),
        Config("b128", batch=128),
        Config("cpu", device="cpu"),
        Config("tf32", tf32=True),
        Config("benchmark", cudnn_benchmark=True),
        Config("cudnn_det", cudnn_deterministic=True),
        Config("det", deterministic=True),
        Config("cpu_t1", device="cpu", threads=1),
    ]
    if ncpu >= 4:
        cs.append(Config("cpu_t4", device="cpu", threads=4))
    cs += [
        Config("b128_tf32", batch=128, tf32=True),
        Config("b128_benchmark", batch=128, cudnn_benchmark=True),
        Config("b128_cudnn_det", batch=128, cudnn_deterministic=True),
        Config("b128_det", batch=128, deterministic=True),
    ]
    return cs


def cpu_configs(ncpu):
    cs = [
        Config("ref", device="cpu"),
        Config("b8", device="cpu", batch=8),
        Config("b32", device="cpu", batch=32),
        Config("b128", device="cpu", batch=128),
        Config("t1", device="cpu", threads=1),
    ]
    if ncpu >= 4:
        cs.append(Config("t4", device="cpu", threads=4))
    cs.append(Config("det", device="cpu", deterministic=True))
    return cs


def pairs_for(configs):
    """(base, other, knob) triples: each knob measured against one base.
    gpu knobs against ref; cpu thread counts against the cpu run; the batch-128
    set against batch 128. run-to-run pairs are added by the sweep driver."""
    names = {c.name for c in configs}
    out = []
    knob = {
        "b8": "batch size 8", "b32": "batch size 32", "b128": "batch size 128",
        "cpu": "cpu instead of gpu", "tf32": "tf32 on", "benchmark": "cudnn benchmark",
        "cudnn_det": "cudnn deterministic", "det": "deterministic algorithms",
        "t1": "1 thread", "t4": "4 threads",
    }
    for n, k in knob.items():
        if n in names:
            out.append(("ref", n, k))
    for n, k in (("cpu_t1", "1 thread (cpu)"), ("cpu_t4", "4 threads (cpu)")):
        if n in names:
            out.append(("cpu", n, k))
    for n, k in (("b128_tf32", "tf32 on, batch 128"), ("b128_benchmark", "cudnn benchmark, batch 128"),
                 ("b128_cudnn_det", "cudnn deterministic, batch 128"), ("b128_det", "deterministic algorithms, batch 128")):
        if n in names:
            out.append(("b128", n, k))
    return out


def depth_pairs(configs):
    names = {c.name for c in configs}
    want = [("ref", "b32"), ("ref", "b128"), ("ref", "cpu"), ("ref", "tf32"),
            ("ref", "benchmark"), ("ref", "det"), ("ref", "t1"), ("ref", "t4")]
    return [p for p in want if p[1] in names]


def by_name(configs):
    return {c.name: c for c in configs}
