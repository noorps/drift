"""what machine and software a run happened on"""
import os, platform, subprocess, sys


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def cpu_name():
    if platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    if platform.system() == "Windows":
        out = _sh(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if out:
            return out.splitlines()[0].strip()
    return platform.processor() or "unknown"


def capture():
    import torch, torchvision
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_threads_default": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import hashlib
        from . import data
        info["cifar10_test_sha256"] = hashlib.sha256(data.cifar_test_images().tobytes()).hexdigest()
    except Exception as e:
        info["cifar10_test_sha256"] = "unavailable: " + repr(e)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": p.name,
            "gpu_sm": f"{p.major}.{p.minor}",
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "driver": _sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        })
    return info
