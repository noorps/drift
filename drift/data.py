"""datasets as plain tensors, cached on disk. cifar-10 test set (10,000 images)
and ag news test set (7,600 headlines + leads)."""
import hashlib, os, re, urllib.request
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)   # the normalisation the committed weights were trained with
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

AG_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/{}.csv"
AG_CLASSES = ["world", "sports", "business", "sci/tech"]
AG_MAXLEN = 64
AG_VOCAB = 20000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CIFAR_PARQUET = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/{}-00000-of-00001.parquet"
CIFAR_PARQUET_SHA = {"test": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
                     "train": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883"}


def _cifar_raw(split):
    """uint8 [n,32,32,3] and labels, from a pinned parquet copy of the original
    cifar-10 (same order as the original python pickles; the first ten test
    labels are 3 8 8 0 6 6 1 6 3 1). falls back to torchvision's download of
    the original archive, which is the same bytes but a slow server."""
    os.makedirs(ROOT, exist_ok=True)
    cache = os.path.join(ROOT, f"cifar10_{split}_uint8.npy")
    if os.path.exists(cache):
        d = np.load(os.path.join(ROOT, f"cifar10_{split}_labels.npy"))
        return np.load(cache), d
    p = os.path.join(ROOT, f"cifar10_{split}.parquet")
    try:
        if not os.path.exists(p) or sha256(p) != CIFAR_PARQUET_SHA[split]:
            urllib.request.urlretrieve(CIFAR_PARQUET.format(split), p)
        assert sha256(p) == CIFAR_PARQUET_SHA[split], "parquet checksum mismatch"
        import io, pyarrow.parquet as pq
        from PIL import Image
        t = pq.read_table(p).to_pydict()
        arr = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
        labels = np.array(t["label"], dtype=np.int64)
    except Exception as e:
        print("parquet copy unavailable (", repr(e), "), using torchvision's download")
        import torchvision
        ds = torchvision.datasets.CIFAR10(ROOT, train=(split == "train"), download=True)
        arr, labels = ds.data, np.array(ds.targets, dtype=np.int64)
    np.save(cache, arr); np.save(os.path.join(ROOT, f"cifar10_{split}_labels.npy"), labels)
    return arr, labels


def _normalise(arr):
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    x = (x - torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)) / torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    return x.contiguous()


def cifar_test():
    """returns (x [10000,3,32,32] float32 normalised, y [10000] int64)"""
    cache = os.path.join(ROOT, "cifar10_test.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["x"], d["y"]
    arr, labels = _cifar_raw("test")
    x, y = _normalise(arr), torch.from_numpy(labels)
    torch.save({"x": x, "y": y}, cache)
    return x, y


def cifar_test_images():
    """raw uint8 [10000,32,32,3] for saving flipped images"""
    return _cifar_raw("test")[0]


def cifar_train():
    arr, labels = _cifar_raw("train")
    return _normalise(arr), torch.from_numpy(labels)


_tok = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def tokenize(s):
    return _tok.findall(s.lower())


def _ag_fetch(split):
    os.makedirs(ROOT, exist_ok=True)
    p = os.path.join(ROOT, f"ag_news_{split}.csv")
    if not os.path.exists(p):
        urllib.request.urlretrieve(AG_URL.format(split), p)
    return p


def _ag_rows(path):
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            rows.append((int(r[0]) - 1, r[1] + " . " + r[2]))
    return rows


def ag_vocab():
    """vocabulary from the training set: the AG_VOCAB most frequent tokens.
    index 0 = pad, 1 = unknown."""
    cache = os.path.join(ROOT, "ag_vocab.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    from collections import Counter
    c = Counter()
    for _, text in _ag_rows(_ag_fetch("train")):
        c.update(tokenize(text))
    words = [w for w, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[: AG_VOCAB - 2]]
    vocab = {w: i + 2 for i, w in enumerate(words)}
    torch.save(vocab, cache)
    return vocab


def _encode(rows, vocab):
    x = torch.zeros(len(rows), AG_MAXLEN, dtype=torch.int64)
    y = torch.zeros(len(rows), dtype=torch.int64)
    for i, (label, text) in enumerate(rows):
        ids = [vocab.get(t, 1) for t in tokenize(text)][:AG_MAXLEN]
        x[i, : len(ids)] = torch.tensor(ids)
        y[i] = label
    return x, y


def ag_test():
    cache = os.path.join(ROOT, "ag_test.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["x"], d["y"]
    x, y = _encode(_ag_rows(_ag_fetch("test")), ag_vocab())
    torch.save({"x": x, "y": y}, cache)
    return x, y


def ag_test_text():
    return [t for _, t in _ag_rows(_ag_fetch("test"))]


def ag_train():
    return _encode(_ag_rows(_ag_fetch("train")), ag_vocab())
