"""move a results folder through a text channel: pack everything except the
regenerable parts (logits, flip images, charts) into base64 chunks, and unpack.

    python -m drift.pack pack results/<label>     -> results/<label>.b64/chunk_000.txt ...
    python -m drift.pack unpack <dir of chunks> results/
"""
import base64, io, os, sys, tarfile

CHUNK = 48000
SKIP = ("logits", "flips", ".depth_tmp")


def pack(src):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for fn in files:
                if fn.endswith(".png"):
                    continue
                p = os.path.join(root, fn)
                tf.add(p, arcname=os.path.relpath(p, os.path.dirname(src)))
    b = base64.b64encode(buf.getvalue()).decode()
    out = src.rstrip("/\\") + ".b64"
    os.makedirs(out, exist_ok=True)
    n = 0
    for i in range(0, len(b), CHUNK):
        with open(os.path.join(out, f"chunk_{n:03d}.txt"), "w") as f:
            f.write(b[i:i + CHUNK])
        n += 1
    print(f"{len(buf.getvalue())} bytes gz, {len(b)} chars b64, {n} chunks in {out}")
    return out, n


def unpack(chunkdir, dest):
    files = sorted(f for f in os.listdir(chunkdir) if f.startswith("chunk_"))
    b = "".join(open(os.path.join(chunkdir, f)).read().strip() for f in files)
    raw = base64.b64decode(b)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(dest)
    print("unpacked", len(raw), "bytes into", dest)


if __name__ == "__main__":
    if sys.argv[1] == "pack":
        pack(sys.argv[2])
    else:
        unpack(sys.argv[2], sys.argv[3])
