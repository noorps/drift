"""open web/index.html (or a url) headless, click through the controls, and
print what the page says, plus any console errors. usage: python tools/check_web.py [path-or-url] [outdir]"""
import json, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

target = sys.argv[1] if len(sys.argv) > 1 else "web/index.html"
url = target if target.startswith("http") else Path(target).resolve().as_uri()
out = Path(sys.argv[2] if len(sys.argv) > 2 else "web_check"); out.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1100, "height": 900})
    errs = []
    pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    pg.goto(url); pg.wait_for_timeout(1200)

    def state(tag):
        d = pg.evaluate("""() => ({headline: document.getElementById('headline').textContent,
            status: document.getElementById('status').textContent, verdict: document.getElementById('verdict').textContent,
            why: document.getElementById('why').textContent.slice(0, 160), cap: document.getElementById('depth-cap').textContent.slice(0, 160),
            cells: document.querySelectorAll('.cell').length, path: (document.querySelector('.chart .line').getAttribute('d') || '').length,
            w: document.scrollingElement.scrollWidth, iw: innerWidth})""")
        print(tag, json.dumps(d, ensure_ascii=False))
    state("initial")
    pg.screenshot(path=str(out / "01_initial.png"), full_page=True)
    def chip(side, text):
        pg.locator(f"#run-{side} .chip", has_text=text).first.click(); pg.wait_for_timeout(700)
    chip("b", "batch size 128"); state("b=batch 128")
    chip("b", "cpu"); state("b=cpu")
    chip("b", "batch size 1"); state("b=cpu batch 1")
    chip("b", "gpu"); chip("b", "cudnn autotune"); state("b=autotune")
    chip("b", "deterministic mode"); state("b=det")
    chip("b", "tf32 on"); state("b=tf32")
    chip("b", "run it again, new process"); state("b=again proc")
    chip("b", "nothing else"); chip("b", "batch size 32"); chip("a", "batch size 8"); state("a=b8 b=b32 (unmeasured pair)")
    chip("a", "batch size 1"); state("back to ref vs b32")
    pg.screenshot(path=str(out / "02_b32.png"), full_page=True)
    pg.set_viewport_size({"width": 375, "height": 800}); pg.wait_for_timeout(400); state("375px")
    pg.screenshot(path=str(out / "03_mobile.png"), full_page=True)
    print("console:", errs)
    b.close()
