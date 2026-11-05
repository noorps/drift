"""save every image whose prediction flipped in any pair, as a png grid per
pair and as individual 32x32 pngs (shared across pairs) for the web page."""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
from . import data


def main(label, web_dir=None):
    d = os.path.join("results", label)
    cp = os.path.join(d, "compare_resnet.json")
    if not os.path.exists(cp):
        return
    cmp = json.load(open(cp))
    imgs = data.cifar_test_images()
    out = os.path.join(d, "flips"); os.makedirs(out, exist_ok=True)
    idxs = set(cmp.get("any_flip_vs_ref", {}).get("indices", []))
    for p in cmp["pairs"]:
        for r in p["flip"]["flipped"]:
            idxs.add(r["index"])
    for i in sorted(idxs):
        Image.fromarray(imgs[i]).save(os.path.join(out, f"{i}.png"))
    for p in cmp["pairs"]:
        fl = p["flip"]["flipped"]
        if not fl:
            continue
        cols = 10; rows = (len(fl) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 100, rows * 118), "#FAF8F1")
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(fl):
            x, y = (k % cols) * 100, (k // cols) * 118
            sheet.paste(Image.fromarray(imgs[r["index"]]).resize((96, 96), Image.NEAREST), (x + 2, y + 2))
            dr.text((x + 2, y + 99), f"{r['base_pred'][:5]}>{r['other_pred'][:5]} m{r['margin']:.2f}", fill="#2B2E28")
        sheet.save(os.path.join(out, f"{p['base']}_vs_{p['other']}.png"))
    print(f"{len(idxs)} flipped images saved to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
