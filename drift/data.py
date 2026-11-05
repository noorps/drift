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


def cifar_test():
    """returns (x [10000,3,32,32] float32 normalised, y [10000] int64)"""
    os.makedirs(ROOT, exist_ok=True)
    cache = os.path.join(ROOT, "cifar10_test.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["x"], d["y"]
    import torchvision
    ds = torchvision.datasets.CIFAR10(ROOT, train=False, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0
    x = (x - torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)) / torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    x = x.contiguous()
    y = torch.tensor(ds.targets, dtype=torch.int64)
    torch.save({"x": x, "y": y}, cache)
    return x, y


def cifar_test_images():
    """raw uint8 [10000,32,32,3] for saving flipped images"""
    cache = os.path.join(ROOT, "cifar10_test_uint8.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import torchvision
    ds = torchvision.datasets.CIFAR10(ROOT, train=False, download=True)
    np.save(cache, ds.data)
    return ds.data


def cifar_train():
    cache = os.path.join(ROOT, "cifar10_train.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["x"], d["y"]
    import torchvision
    ds = torchvision.datasets.CIFAR10(ROOT, train=True, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0
    x = (x - torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)) / torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    return x.contiguous(), torch.tensor(ds.targets, dtype=torch.int64)


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
