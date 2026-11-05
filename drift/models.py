"""the two models. resnet18 in its cifar form (3x3 stem, no maxpool) and a
small transformer encoder written out by hand so every matmul is visible."""
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


def resnet18_cifar():
    import torchvision
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pad):              # x [b,t,d], pad [b,t] True where padded
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, d // self.h).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))       # [b,h,t,dh]
        s = q @ k.transpose(-1, -2) / math.sqrt(d // self.h)   # [b,h,t,t]
        s = s.masked_fill(pad[:, None, None, :], float("-inf"))
        a = F.softmax(s, dim=-1)
        y = (a @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(ff, d)

    def forward(self, x, pad):
        x = x + self.attn(self.ln1(x), pad)
        x = x + self.ff2(self.act(self.ff1(self.ln2(x))))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab=20000, maxlen=64, d=128, heads=4, ff=256, layers=2, classes=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.fc = nn.Linear(d, classes)

    def forward(self, ids):                 # ids [b,t]
        pad = ids == 0
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        for blk in self.blocks:
            x = blk(x, pad)
        x = self.ln(x)
        keep = (~pad).float()[..., None]
        x = (x * keep).sum(1) / keep.sum(1).clamp(min=1)   # mean over real tokens
        return self.fc(x)


def load(kind, device):
    if kind == "resnet":
        m = resnet18_cifar()
        p = os.path.join(WEIGHTS, "resnet18_cifar10.pt")
    else:
        m = TinyTransformer()
        p = os.path.join(WEIGHTS, "transformer_agnews.pt")
    m.load_state_dict(torch.load(p, map_location="cpu"))
    return m.to(device).eval()
