/* drift explainer. everything on the page comes from data.js (results/<label>/*.json). */
(function () {
  "use strict";
  const D = window.DRIFT, M = D.models.resnet;
  const $ = (id) => document.getElementById(id);
  const fmt = (n) => n.toLocaleString("en-US");
  const sci = (x) => x === 0 ? "0" : x < 1e-3 ? x.toExponential(1) : x.toPrecision(2);
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- the configuration space, in plain words -----------------------------
  const canon = Object.entries(M.configs).filter(([n]) => !/_again$|_proc$/.test(n)).map(([n, c]) => ({ name: n, ...c }));
  const EXTRAS = [
    ["none", "nothing else", (c) => !c.tf32 && !c.cudnn_benchmark && !c.cudnn_deterministic && !c.deterministic && !c.threads],
    ["tf32", "tf32 on", (c) => c.tf32],
    ["benchmark", "cudnn autotune", (c) => c.cudnn_benchmark],
    ["cudnn_det", "cudnn deterministic", (c) => c.cudnn_deterministic],
    ["det", "deterministic mode", (c) => c.deterministic],
    ["t1", "1 cpu thread", (c) => c.threads === 1],
    ["t4", "4 cpu threads", (c) => c.threads === 4],
    ["again", "run it again, same process", null],
    ["proc", "run it again, new process", null],
  ];
  const BATCHES = [...new Set(canon.map((c) => c.batch))].sort((a, b) => a - b);
  const DEVICES = [["cuda", "gpu"], ["cpu", "cpu"]].filter(([d]) => canon.some((c) => c.device === d));
  const measuredNames = new Set(Object.keys(M.configs));
  const pairFor = (a, b) => M.pairs.find((p) => (p.base === a && p.other === b) || (p.base === b && p.other === a));

  function nameFor(sel) {
    const ex = EXTRAS.find((e) => e[0] === sel.extra);
    const baseTest = ex[2] || EXTRAS[0][2];
    const c = canon.find((c) => c.device === sel.device && c.batch === sel.batch && baseTest(c));
    if (!c) return null;
    if (sel.extra === "again" || sel.extra === "proc") {
      const n = c.name + "_" + sel.extra;
      return measuredNames.has(n) ? n : null;
    }
    return c.name;
  }
  const wordsOf = (name) => {
    const w = M.configs[name].words;
    return /_again$/.test(name) ? w + ", run again" : /_proc$/.test(name) ? w + ", run again in a new process" : w;
  };

  const sel = { a: { device: "cuda", batch: 1, extra: "none" }, b: { device: "cuda", batch: 32, extra: "none" } };
  if (!DEVICES.some(([d]) => d === "cuda")) { sel.a.device = sel.b.device = "cpu"; }
  if (!BATCHES.includes(32)) sel.b.batch = BATCHES[BATCHES.length - 1];

  function chipRow(side, key, label, options, current) {
    const row = document.createElement("div"); row.className = "pick";
    const k = document.createElement("span"); k.className = "k"; k.textContent = label; row.appendChild(k);
    for (const [val, text] of options) {
      const b = document.createElement("button"); b.className = "chip"; b.type = "button"; b.textContent = text;
      b.setAttribute("aria-pressed", String(val === current));
      const probe = { ...sel[side], [key]: val };
      if (nameFor(probe) === null) { b.classList.add("unmeasured"); b.title = "no run with these settings together"; }
      b.addEventListener("click", () => { sel[side][key] = val; render(); });
      row.appendChild(b);
    }
    return row;
  }

  function renderControls() {
    for (const side of ["a", "b"]) {
      const box = $("run-" + side);
      box.querySelectorAll(".pick").forEach((e) => e.remove());
      box.appendChild(chipRow(side, "device", "where it runs", DEVICES, sel[side].device));
      box.appendChild(chipRow(side, "batch", "photos per batch", BATCHES.map((b) => [b, "batch size " + b]), sel[side].batch));
      box.appendChild(chipRow(side, "extra", "one more setting", EXTRAS.map((e) => [e[0], e[1]]), sel[side].extra));
    }
  }

  // ---- the divergence chart --------------------------------------------------
  const W = 640, H = 300, PL = 54, PR = 16, PT = 14, PB = 46;
  const allRel = M.depth.flatMap((d) => d.rel).filter((v) => v > 0);
  const yMin = allRel.length ? Math.min(...allRel) / 3 : 1e-8, yMax = allRel.length ? Math.max(...allRel) * 3 : 1;
  const L = M.depth.length ? M.depth[0].layers.length : 1;
  const sx = (i) => PL + (i / Math.max(L - 1, 1)) * (W - PL - PR);
  const sy = (v) => { const t = (Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)); return PT + (1 - t) * (H - PT - PB); };
  const pathOf = (ys) => ys.map((v, i) => (i ? "L" : "M") + sx(i).toFixed(1) + " " + sy(v).toFixed(1)).join(" ");
  let cur = null, anim = null;

  function buildChart() {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "relative difference between the two runs at every layer of the network");
    const g = (tag, attrs, parent = svg) => { const e = document.createElementNS(ns, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
    for (let p = Math.ceil(Math.log10(yMin)); p <= Math.floor(Math.log10(yMax)); p++) {
      const y = sy(10 ** p);
      g("line", { x1: PL, x2: W - PR, y1: y, y2: y, class: "grid-line" });
      const t = g("text", { x: PL - 6, y: y + 4, "text-anchor": "end", class: "axis" }); t.textContent = "1e" + p;
    }
    if (M.depth.length) {
      const names = M.depth[0].layers;
      const marks = names.map((n, i) => [n, i]).filter(([n]) => /^layer\d\.\d#0$/.test(n) || n === "logits#0" || n === "conv1#0");
      for (const [n, i] of marks) {
        const t = g("text", { x: sx(i), y: H - PB + 16, "text-anchor": "middle", class: "axis" }); t.textContent = n.replace("#0", "");
      }
      const t = g("text", { x: (PL + W - PR) / 2, y: H - 8, "text-anchor": "middle", class: "axis" }); t.textContent = "layer, in the order the network runs them (" + L + " hook points)";
      for (const d of M.depth) g("path", { d: pathOf(d.rel), class: "ghost" });
    }
    const line = g("path", { d: "", class: "line" });
    const hl = g("line", { y1: PT, y2: H - PB, class: "hover-line" });
    const tipg = g("g", { opacity: 0 });
    const tipbg = g("rect", { rx: 3, height: 20, class: "tipbg" }, tipg);
    const tip = g("text", { class: "tip", y: 14 }, tipg);
    const empty = g("text", { x: (PL + W - PR) / 2, y: H / 2, "text-anchor": "middle", class: "empty" });
    svg.addEventListener("mousemove", (ev) => {
      if (!cur) return;
      const r = svg.getBoundingClientRect(); const x = (ev.clientX - r.left) * W / r.width;
      const i = Math.max(0, Math.min(L - 1, Math.round((x - PL) / (W - PL - PR) * (L - 1))));
      hl.setAttribute("x1", sx(i)); hl.setAttribute("x2", sx(i)); hl.setAttribute("opacity", 1);
      const v = cur.rel[i];
      tip.textContent = cur.layers[i].replace("#0", "") + ": " + (v === 0 ? "identical" : sci(v));
      const tw = tip.getComputedTextLength() + 12; const tx = Math.min(sx(i) + 8, W - PR - tw);
      tipg.setAttribute("transform", `translate(${tx},${Math.max(PT, sy(v) - 26)})`); tipbg.setAttribute("width", tw); tipbg.setAttribute("x", 0); tip.setAttribute("x", 6);
      tipg.setAttribute("opacity", 1);
    });
    svg.addEventListener("mouseleave", () => { hl.setAttribute("opacity", 0); tipg.setAttribute("opacity", 0); });
    $("depth-chart").appendChild(svg);
    return { line, empty };
  }
  const chart = buildChart();

  function showDepth(d) {
    if (!d) { cur = null; chart.line.setAttribute("d", ""); return; }
    const target = d.rel.map((v) => Math.max(v, yMin));
    const from = cur ? cur.rel.map((v) => Math.max(v, yMin)) : target.map(() => yMin);
    cur = d;
    if (anim) cancelAnimationFrame(anim);
    if (reduced) { chart.line.setAttribute("d", pathOf(target)); return; }
    const t0 = performance.now(), dur = 550;
    const step = (now) => {
      const u = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - u, 3);
      chart.line.setAttribute("d", pathOf(target.map((v, i) => Math.exp(Math.log(from[i]) + (Math.log(v) - Math.log(from[i])) * e))));
      if (u < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  // ---- the words and the grid ------------------------------------------------
  let showAll = false;
  function render() {
    renderControls();
    const a = nameFor(sel.a), b = nameFor(sel.b);
    const status = $("status"); status.className = "status";
    const cap = $("depth-cap"), grid = $("grid");
    grid.innerHTML = ""; $("grid-cap").textContent = ""; $("why").textContent = "";
    if (!a || !b) {
      status.className = "status warn";
      status.textContent = "i did not run a configuration with those settings together (dashed chips). the ones i ran are listed under the details.";
      $("verdict").textContent = ""; showDepth(null); chart.empty.textContent = "no run with these settings"; cap.textContent = ""; return;
    }
    if (a === b) {
      status.textContent = "both sides are the same run. pick 'run it again' on one side to see run-to-run drift.";
      $("verdict").textContent = ""; showDepth(null); chart.empty.textContent = "same run on both sides"; cap.textContent = ""; return;
    }
    const p = pairFor(a, b);
    status.textContent = "run a: " + wordsOf(a) + "   ·   run b: " + wordsOf(b);
    if (!p) {
      status.className = "status warn";
      status.textContent += "   ·   i measured both runs, but not against each other (every knob was compared to one reference run).";
      $("verdict").textContent = ""; showDepth(null); chart.empty.textContent = "pair not measured"; cap.textContent = ""; return;
    }
    const d = M.depth.find((x) => (x.base === a && x.other === b) || (x.base === b && x.other === a));
    chart.empty.textContent = d ? "" : "i hooked the layers for " + M.depth.length + " pairs; not this one";
    showDepth(d);
    if (d) {
      const first = d.rel.find((v) => v > 0), last = d.rel[d.rel.length - 1];
      cap.textContent = first === undefined
        ? "every layer matched to the last bit between these two runs. the grey lines are the other pairs i measured."
        : "how far apart the two runs are at each layer, relative to the size of the numbers there (median over " + d.n_inputs + " photos). starts at " + sci(first) + ", ends at " + sci(last) + " at the final scores. grey lines are the other pairs.";
    } else cap.textContent = "";
    const s = p.stats, n = s.n, f = s.flips, low = p.flips_in_lowest_5pct;
    const v = $("verdict");
    if (f === 0) {
      v.innerHTML = `<span class="big">0</span> of ${fmt(n)} photos got a different answer.`;
      $("why").textContent = s.identical_images === n ? "the scores were identical to the last bit for every photo." :
        `the scores matched to the last bit for ${fmt(s.identical_images)} of them; the rest differed by at most ${sci(s.abs_max)} and never enough to change the answer.`;
    } else {
      v.innerHTML = `of ${fmt(n)} photos, <span class="big">${f}</span> got a different answer, and ${low} of those were in the closest-call 5%.`;
      const bk = p.buckets;
      $("why").textContent = `by how sure the model was: ${bk.map((x) => x.flips + " in the " + x.bucket).join(", ")}. ` +
        (s.identical_images ? `${fmt(s.identical_images)} photos matched to the last bit. ` : "") +
        `largest change in any photo's scores: ${sci(s.abs_max)}.`;
    }
    const items = p.flipped, base = p.base === a ? "a" : "b", other = base === "a" ? "b" : "a";
    const shown = showAll ? items : items.slice(0, 24);
    for (const r of shown) {
      const c = document.createElement("div"); c.className = "cell";
      c.innerHTML = `<img src="img/${r.index}.png" width="32" height="32" alt="cifar-10 test image ${r.index}, labelled ${r.label}">` +
        `${base}: <b>${r.base_pred}</b><br>${other}: <b>${r.other_pred}</b><br><span class="m">margin ${r.margin.toFixed(2)} · is a ${r.label}</span>`;
      grid.appendChild(c);
    }
    $("grid-wrap").style.display = items.length ? "" : "none";
    if (items.length) {
      $("grid-cap").textContent = `the ${items.length} photos that got a different answer, closest calls first. margin = how far ahead the winning class was in run ${base}; the whole set's median is ${M.margin_edges ? sci(M.margin_edges[2]) + " at the 25% mark" : ""}.`;
      if (items.length > 24 && !showAll) {
        const m = document.createElement("button"); m.className = "more"; m.type = "button"; m.textContent = "show all " + items.length;
        m.addEventListener("click", () => { showAll = true; render(); }); $("grid-cap").appendChild(document.createElement("br")); $("grid-cap").appendChild(m);
      }
    }
  }

  function renderDetails() {
    const e = D.env, box = $("details");
    const ref = M.configs.ref ? M.configs.ref.words : "reference";
    let h = `<p>the model is resnet18 trained once on cifar-10 (test accuracy ${(100 * M.accuracy).toFixed(2)}%). the reference run is ${ref}. every knob was changed one at a time against that run; "run again" rows are the identical configuration a second time. "max rel" is the largest relative change in any photo's ten scores. times are for all ${fmt(M.n)} photos.</p><div class="tw"><table><tr><th>knob</th><th>against</th><th>max rel</th><th>identical</th><th>flips</th><th>time</th></tr>`;
    for (const p of M.pairs) h += `<tr><td>${p.knob}</td><td>${M.configs[p.base].words}</td><td>${p.stats.identical_images === p.stats.n ? "bitwise same" : sci(p.stats.rel_max)}</td><td>${fmt(p.stats.identical_images)}</td><td>${p.stats.flips}</td><td>${p.wall_s[p.base].toFixed(1)} s → ${p.wall_s[p.other].toFixed(1)} s</td></tr>`;
    h += "</table></div>";
    if (M.not_run.length) h += `<p>not run: ${M.not_run.map((x) => x.knob || x.name).join(", ")}.</p>`;
    const T = D.models.text;
    if (T) {
      h += `<p>the same sweep on a small transformer (text classification on ag news, ${fmt(T.n)} headlines, accuracy ${(100 * T.accuracy).toFixed(2)}%): ${T.any_flip.n} of ${fmt(T.n)} changed answer in at least one configuration, ${T.any_flip.in_lowest_5pct} of them in the closest-call 5%.</p>`;
    }
    h += `<p>${e.gpu ? e.gpu + ", driver " + e.driver + ", cuda " + e.cuda + ", cudnn " + e.cudnn + ". " : "no gpu. "}${e.cpu}. torch ${e.torch}, torchvision ${e.torchvision}, python ${e.python}. the readme in the repo has the full tables, the per-layer chart as an image, and what to do about it.</p>`;
    box.innerHTML = h;
  }

  const af = M.any_flip || { n: 0, in_lowest_5pct: 0 };
  $("headline").textContent = `${af.n} of ${fmt(M.n)} photos came out with a different answer, and ${af.in_lowest_5pct} of those were ones the model was already unsure about.`;
  $("label").textContent = D.label;
  renderDetails();
  render();
})();
