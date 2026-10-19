// Builds slides/MTH9821_HW3_Team9_slides.pptx from slides/deck_data.json (run slides/deck_data.py first).
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(path.join(__dirname, "apply_theme.js"));

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "MTH9821_HW3_Team9_slides.pptx");
const f2 = (x) => x.toFixed(2);
const f3 = (x) => x.toFixed(3);
const f4 = (x) => x.toFixed(4);

const THEME = {
  name: "American Put", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: {
    dk1: "1E2328", lt1: "FFFFFF", dk2: "23313D", lt2: "EEF2F4",
    accent1: "D9622B", accent2: "2F7D8C", accent3: "8FA3AE", accent4: "E9B44C",
    accent5: "5B6770", accent6: "B23A48", hlink: "2F7D8C", folHlink: "5B6770",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "American options: policy evaluation and fitted Q-learning";
pres.author = "Team 9";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "Title Dark",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.4, w: 8.8, h: 1.5,
        fontSize: 34, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.05, w: 8.8, h: 1.6,
        fontSize: 16, color: C.accent3, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75,
        fontSize: 26, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "MTH 9821 · Team 9", options: { x: 0.5, y: 5.2, w: 4, h: 0.3, fontSize: 10,
        color: C.accent5, isTextBox: true } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5 },
});

// Rich text with _x / _{xy} subscripts and ^x / ^{xy} superscripts.
function runs(str, base0 = {}) {
  const { breakLine, ...base } = base0;
  const out = [];
  const re = /([_^])(\{[^}]*\}|[^\s,.;:)])/g;
  let last = 0, m;
  while ((m = re.exec(str))) {
    if (m.index > last) out.push({ text: str.slice(last, m.index), options: { ...base } });
    const t = m[2].startsWith("{") ? m[2].slice(1, -1) : m[2];
    out.push({ text: t, options: { ...base, [m[1] === "_" ? "subscript" : "superscript"]: true } });
    last = re.lastIndex;
  }
  if (last < str.length) out.push({ text: str.slice(last), options: { ...base } });
  if (breakLine) out[out.length - 1].options.breakLine = true;
  return out;
}
// Labeled paragraphs without bullet glyphs (keeps sub/superscript runs in one paragraph).
function bullets(items, extra = {}) {
  const out = [];
  items.forEach((it, i) => {
    const [label, body] = Array.isArray(it) ? it : [null, it];
    const r = runs(body, extra);
    if (label) r.unshift({ text: label + "  ", options: { ...extra, bold: true, color: "D9622B" } });
    if (i < items.length - 1) r[r.length - 1].options.breakLine = true;
    out.push(...r);
  });
  return out;
}
const card = (s, x, y, w, h, name) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name });
const tx = (o) => ({ fontSize: 14, color: C.text1, paraSpaceAfter: 7, valign: "top", isTextBox: true, ...o });
const head = (t) => runs(t, { bold: true, fontSize: 15, color: C.text2, breakLine: true });

const R = (seed, m, d) => D.results.find((r) => r.seed === seed && r.method === m && r.degree === d);
const P = (seed, d) => D.paired.find((r) => r.seed === seed && r.degree === d);
const SEEDS = [17, 29, 43];
const ck = D.checks, tp = ck.two_period, ref = ck.reference.bermudan_fine;

(async () => {
  // ---------------------------------------------------------------- 1. Title
  let s = pres.addSlide({ masterName: "Title Dark" });
  s.addText("American options: policy evaluation and fitted Q-learning", { placeholder: "title" });
  s.addText("Bermudan put, 50 exercise dates: Longstaff–Schwartz vs fitted Q-learning\nTeam 9 · Jaskaran Kalra, William McDonnell",
    { placeholder: "body" });
  s.addNotes("Bermudan put with S0 = K = 100, r = 5%, sigma = 20%, T = 1, exercise at 51 grid dates. We learn an exercise policy two ways, freeze it, and price it on an independent set of paths. Question: do the methods really differ once the policies are evaluated independently?");

  // ---------------------------------------------------------------- 2. Q1
  s = pres.addSlide({ masterName: "Content" });
  s.addText("Q1: the value function", { placeholder: "title" });
  s.addText(bullets([
    ["(1)", "V^a_n(s,x) = E[Σ_{m≥n} β^{m−n} x_m a_m g(S_m) | S_n = s, x_n = x]"],
    ["(1)", "Recursion: V^a_n = x a_n g(s) + β E[V^a_{n+1}(S_{n+1}, (x − a_n)^+)]"],
    ["Why", "x = 0 stays 0, so an exercised option has value 0: it pays once"],
    ["(2)", "C^*_n(s) = β E[V^*_{n+1}(S_{n+1},1) | S_n = s],  V^*_n = max(g, C^*_n)"],
    ["Why", "S is Markov: past prices add nothing about S_{n+1}, so using history cannot beat V^*"],
  ], { fontSize: 13 }), tx({ x: 0.5, y: 1.2, w: 5.3, h: 3.8, objectName: "q1 text" }));
  card(s, 6.0, 1.2, 3.5, 3.8, "q1 card");
  s.addText(head("(3) Two-period check").concat(runs(
    `q = ${f3(tp.q)}`, { fontSize: 13, breakLine: true })).concat(runs(
    `Up (120): C_1 = ${f2(tp.C1_up)} > g = 0 → continue`, { fontSize: 13, breakLine: true })).concat(runs(
    `Down (80): C_1 = ${f2(tp.C1_down)} < g = 20 → exercise`, { fontSize: 13, breakLine: true })).concat(runs(
    `V_0 = ${f4(tp.V0)}`, { fontSize: 15, bold: true, color: C.accent1, breakLine: true })).concat(runs(
    "Why alive differs: down–up was already exercised at 80; up–down is still alive and pays 4 at the end",
    { fontSize: 12, color: C.text1 })),
    tx({ x: 6.2, y: 1.35, w: 3.1, h: 3.5, paraSpaceAfter: 8, objectName: "q1 card text" }));
  s.addNotes("(1) The policy value is a deterministic function of the state; along one path the discounted cash flow is random, and its conditional mean is V. (2) Backward induction: at each date any policy either exercises (gets g) or continues (gets at most C*), so it gets at most max(g, C*); the greedy policy attains it. (3) Both middle paths end at 96, but the alive flag depends on the path.");

  // ---------------------------------------------------------------- 3. Q2
  const shared = "Both methods: basis 1, z, …, z^d with z = s/K − 1, d = 2, 4; least squares by lstsq; clip to [0, K]; exercise iff g > 0 and g ≥ Ĉ_n";
  s = pres.addSlide({ masterName: "Content" });
  s.addText("Q2: Longstaff–Schwartz", { placeholder: "title" });
  s.addText(bullets([
    ["(1)", "Backward over dates; fit on in-the-money paths only; then set τ̂ = n where rule (7) says exercise"],
    ["(2)", "Date 0: no regression, use the average cash flow; fitted value = max(average, g(S_0))"],
    ["(3) Why cash flow", "one realized draw; its average given S_n is the continuation value, so no fitted value is fed back"],
    ["(3) Why future", "allowed in training labels, but not in evaluation: that would be look-ahead"],
    ["(3) Why date n", "the response is compared with g(S_n), paid at date n"],
  ], { fontSize: 13 }), tx({ x: 0.5, y: 1.2, w: 5.3, h: 3.9, paraSpaceAfter: 8, objectName: "q2 text" }));
  card(s, 6.0, 1.2, 3.5, 3.8, "q2 card");
  s.addText(head("Response (8)").concat(runs("Y_n = β^{τ̂−n} g(S_{τ̂})", { fontSize: 17, color: C.accent1, breakLine: true }))
    .concat(runs("τ̂ = first exercise date of the already fitted future policy (starts at N)", { fontSize: 12, breakLine: true }))
    .concat(runs(shared, { fontSize: 11, color: C.accent5 })),
    tx({ x: 6.2, y: 1.35, w: 3.1, h: 3.5, paraSpaceAfter: 10, objectName: "q2 card text" }));
  s.addNotes("All responses at a date are formed before any exercise index is updated. No small-sample fallback was needed in any of the twelve fits.");

  // ---------------------------------------------------------------- 4. Q3
  s = pres.addSlide({ masterName: "Content" });
  s.addText("Q3: fitted Q-learning", { placeholder: "title" });
  s.addText(bullets([
    ["(1)", "Backward over dates; fit on all paths; freeze V̂_{n+1} before fitting date n"],
    ["(2)", "Date 0: clipped average of targets vs g(S_0); actions from rule (7); evaluated on fresh paths"],
    ["(3) From (9)", "Q^*(continue) = β E[V^*_{n+1}]: replace V^* by V̂ and the expectation by regression"],
    ["(3) vs (8)", "one step realized, the rest estimated (V̂); in (8) the whole cash flow is realized"],
    ["(3) Why all paths", "an out-of-the-money stock can still fall below K, so V̂_n = Ĉ_n > 0 there and feeds earlier targets"],
  ], { fontSize: 13 }), tx({ x: 0.5, y: 1.2, w: 5.3, h: 3.9, paraSpaceAfter: 8, objectName: "q3 text" }));
  card(s, 6.0, 1.2, 3.5, 3.8, "q3 card");
  s.addText(head("Targets (10)–(12)").concat(runs("V̂_N = g", { fontSize: 15, color: C.accent1, breakLine: true }))
    .concat(runs("Y_n = β V̂_{n+1}(S_{n+1})", { fontSize: 15, color: C.accent1, breakLine: true }))
    .concat(runs("V̂_n = max(g, Ĉ_n)", { fontSize: 15, color: C.accent1, breakLine: true }))
    .concat(runs("Errors in V̂ and the max carry back through all dates", { fontSize: 12 })),
    tx({ x: 6.2, y: 1.35, w: 3.1, h: 3.5, paraSpaceAfter: 10, objectName: "q3 card text" }));
  s.addNotes("Fitted Q has lower-variance targets than LSM, but its regression error and the upward bias of the max in (12) are passed back through all 49 dates. Fitting only in-the-money states would leave out-of-the-money values to polynomial extrapolation.");

  // ---------------------------------------------------------------- 4. Q4(1)
  s = pres.addSlide({ masterName: "Content" });
  s.addText("Q4 (1): checks and the twelve policies", { placeholder: "title" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.text2, fill: { color: C.background2 } } });
  const rows = [[hdr("policy"), hdr("fitted V₀"), hdr("seed 17"), hdr("seed 29"), hdr("seed 43")]];
  for (const [m, d] of [["LSM", 2], ["FQ", 2], ["LSM", 4], ["FQ", 4]]) {
    const fv = SEEDS.map((sd) => R(sd, m, d).fitted_value);
    rows.push([`${m}, d = ${d}`, `${f2(Math.min(...fv))}–${f2(Math.max(...fv))}`,
      ...SEEDS.map((sd) => f3(R(sd, m, d).policy_value))]);
  }
  s.addTable(rows, { x: 0.5, y: 1.2, w: 5.4, colW: [1.2, 1.2, 1.0, 1.0, 1.0], fontSize: 12, color: C.text1,
    border: { type: "solid", pt: 0.5, color: "D5DBDF" }, rowH: 0.32, objectName: "results table" });
  s.addText(runs(`Independent policy values (100,000 evaluation paths, SE ≈ 0.02–0.03). Benchmark ${f4(ref)}.`,
    { fontSize: 11, color: C.accent5 }), tx({ x: 0.5, y: 2.95, w: 5.4, h: 0.5, objectName: "table note" }));
  s.addText(bullets([
    ["Why", "FQ fitted values are far above its policy values (≈ 35–80 SE): regression errors and the max carry back through all dates; not Monte Carlo noise"],
  ], { fontSize: 12 }), tx({ x: 0.5, y: 3.5, w: 5.4, h: 1.5, objectName: "why text" }));
  card(s, 6.1, 1.2, 3.4, 3.8, "checks card");
  s.addText(head("Checks").concat(bullets([
    ["Two-period:", `evaluator gives ${f4(tp.evaluator_value)} = V_0`],
    ["European:", `${f3(ck.european.mean)} vs Black–Scholes ${f3(ck.european.black_scholes)} (z = ${f2(ck.european.z)})`],
    ["Reference:", `our CRR tree reproduces ${f4(ck.crr_2500_supplied)} (2500 steps)`],
    ["Below bound:", "all 12 policy values are below the benchmark"],
  ], { fontSize: 12 })), tx({ x: 6.25, y: 1.35, w: 3.1, h: 3.5, paraSpaceAfter: 7, objectName: "checks text" }));
  s.addNotes("Training: 20,000 paths per seed, reused by both methods and degrees. One evaluation set, seed 20261008, never used for tuning. Training takes about 0.15 s per policy and evaluation about 0.3–0.45 s. LSM fitted values differ from policy values by at most 0.08, within about 1.6 training standard errors.");

  // ---------------------------------------------------------------- 5. Q4(2)-(3)
  s = pres.addSlide({ masterName: "Content" });
  s.addText("Q4 (2)–(3): paired differences and decisions", { placeholder: "title" });
  const pr = (d) => SEEDS.map((sd) => f2(P(sd, d).mean_diff_FQ_minus_LSM)).join(", ");
  s.addText(bullets([
    ["(2)", `FQ − LSM, d = 2: ${pr(2)}`],
    ["(2)", `FQ − LSM, d = 4: ${pr(4)}`],
    ["(2)", "All six 95% CIs are below 0: LSM is better"],
    ["Why paired", "same evaluation paths cancel common noise (SE 0.009–0.015 vs ≈ 0.034 unpaired)"],
    ["(2)–(3)", "degree 4 narrows the gap and raises both methods' policy values, much more for FQ"],
    ["Why small", "near the crossing g ≈ C, a wrong decision loses only |g − C| ≈ 0"],
  ], { fontSize: 13 }), tx({ x: 0.5, y: 1.2, w: 4.6, h: 3.9, paraSpaceAfter: 6, objectName: "paired text" }));
  s.addImage({ path: path.join(__dirname, "n25_decisions.png"), x: 5.2, y: 1.15, w: 4.4, h: 4.4 / 1.588,
    objectName: "n25 figure" });
  s.addText("Seed 17, n = 25. Exercise for s ≤ 85.4 (LSM d = 2), 84.5 (LSM d = 4), 82.6 (FQ d = 4); FQ d = 2 never exercises",
    tx({ x: 5.2, y: 4.0 + 0.05, w: 4.4, h: 0.6, fontSize: 10, color: C.accent5, objectName: "fig caption" }));
  s.addNotes("Differences are paired on the same 100,000 evaluation paths, so the SE is much smaller than comparing two separate prices. The three seeds are reported separately, not pooled. In the plot, FQ degree 2 places continuation above the payoff where exercise should happen, so it exercises too late. LSM d = 2 and d = 4 boundaries differ by about 1 in the stock price but the price differs by only about 0.02.");

  // ---------------------------------------------------------------- 6. Q4(4)
  s = pres.addSlide({ masterName: "Content" });
  s.addText("Q4 (4): what the evidence supports", { placeholder: "title" });
  s.addText(bullets([
    ["Bound", "Given the training data, a frozen policy is admissible, so E[H | data] ≤ V^*_0 ≈ 6.079 (Part 1)"],
    ["Test means", "can still exceed V^*_0 by noise (about one SE ≈ 0.02)"],
    ["Fitted values", "are not payoffs of a policy: LSM is in-sample, FQ has max and basis bias, so neither must be below V^*_0"],
    ["SE measures", "only evaluation Monte Carlo noise for a fixed policy"],
    ["SE omits", "training-seed variation, basis/degree error, and reference error"],
  ], { fontSize: 14 }), tx({ x: 0.5, y: 1.2, w: 9.0, h: 2.7, paraSpaceAfter: 8, objectName: "q4d text" }));
  card(s, 0.5, 4.0, 9.0, 1.05, "takeaway card");
  s.addText(runs(`Result: LSM d = 4 is within 0.009 of the benchmark for every seed; fitted Q is lower in all six paired comparisons, and its fitted values overstate what its policies earn.`,
    { fontSize: 13, color: C.text2 }), tx({ x: 0.7, y: 4.08, w: 8.6, h: 0.9, valign: "middle", objectName: "takeaway text" }));
  s.addNotes("The benchmark is the finest refined tree value; its last refinement changed it by 0.00003, which is an empirical check, not a certified bound.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
