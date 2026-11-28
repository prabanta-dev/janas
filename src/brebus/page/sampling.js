/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* sampling.js - Brebus's page (app.html): how a model chooses its words,
   set in two places. A model's own, in My models (models.js), which Brebus
   gives janas-server when it loads it (--sampling), for every request that
   does not say, Brebus's chats and other programs alike; and a chat's, in
   its Instructions and answer (more.js), over the model's. A field left
   empty is the one below it: a chat's the model's settings, a model's its
   makers' recommendation, as its file says (general.sampling.*), else
   Janas's. Reset empties them all; settings kept under a name (the same
   list as the chat's saved instructions) fill them again. */

/* by the server's names (llama.cpp's); def: Janas's where the model has
   no say; more: under More sampling */
const SAMPLING = [
  {k: "temperature", def: "0.7", label: "Temperature", min: 0, max: 2, step: 0.05, note: "0: always the likeliest word; higher: freer"},
  {k: "top_k", smax: 200, def: "20", label: "Top-k", min: 0, max: 1000, step: 1, note: "only the k likeliest words (0: no limit)"},
  {k: "top_p", def: "0.8", label: "Top-p", min: 0, max: 1, step: 0.01, note: "how many of the likely words it chooses among"},
  {k: "min_p", def: "0", label: "Min-p", min: 0, max: 1, step: 0.01, note: "no word less likely than this share of the likeliest"},
  {k: "repeat_penalty", def: "1", label: "Repetition penalty", min: 0, max: 3, step: 0.05, note: "above 1: the words already used less likely (1: off)"},
  {k: "presence_penalty", def: "0", label: "Presence penalty", min: -2, max: 2, step: 0.1, note: "a word already in the answer, less likely"},
  {k: "frequency_penalty", def: "0", label: "Frequency penalty", min: -2, max: 2, step: 0.1, note: "a word, less likely each time it comes back"},
  {k: "repeat_last_n", smax: 4096, def: "0", label: "Penalties' window", min: 0, max: 100000, step: 1, more: true, note: "the last tokens the penalties look at (0: the whole answer)"},
  {k: "dry_multiplier", def: "0", label: "DRY", min: 0, max: 5, step: 0.05, more: true, note: "against repeating whole phrases (0: off; 0.8 is usual)"},
  {k: "dry_base", def: "1.75", label: "DRY base", min: 1, max: 4, step: 0.05, more: true, note: "how fast DRY grows with the phrase's length"},
  {k: "dry_allowed_length", smax: 20, def: "2", label: "DRY allowed length", min: 0, max: 100, step: 1, more: true, note: "the tokens a phrase may repeat freely"},
  {k: "dry_penalty_last_n", smax: 4096, def: "0", label: "DRY window", min: 0, max: 100000, step: 1, more: true, note: "the last tokens DRY looks at (0: the whole answer)"},
  {k: "typical_p", def: "1", label: "Typical-p", min: 0, max: 1, step: 0.01, more: true, note: "the words as surprising as the model usually is (1: off)"},
  {k: "top_n_sigma", def: "0", label: "Top-n-sigma", min: 0, max: 10, step: 0.1, more: true, note: "the words within n deviations of the best (0: off)"},
  {k: "xtc_probability", def: "0", label: "XTC probability", min: 0, max: 1, step: 0.05, more: true, note: "how often the likeliest words are set aside, for variety (0: off)"},
  {k: "xtc_threshold", def: "0.1", label: "XTC threshold", min: 0, max: 0.5, step: 0.01, more: true, note: "the words more likely than this, set aside"},
  {k: "mirostat", def: "0", label: "Mirostat", min: 0, max: 2, step: 1, more: true, note: "1 or 2: a steady surprise in place of top-k and top-p (0: off)"},
  {k: "mirostat_tau", def: "5", label: "Mirostat tau", min: 0, max: 20, step: 0.1, more: true, note: "the surprise aimed at"},
  {k: "mirostat_eta", def: "0.1", label: "Mirostat eta", min: 0, max: 1, step: 0.01, more: true, note: "how fast it learns"},
  {k: "dynatemp_range", def: "0", label: "Dynamic temperature", min: 0, max: 2, step: 0.05, more: true, note: "the temperature up to this much hotter or colder, by how unsure the model is (0: off)"},
  {k: "dynatemp_exponent", def: "1", label: "Its exponent", min: 0, max: 10, step: 0.1, more: true, note: "how it follows the model's doubt"},
  {k: "seed", noslide: true, def: "—", label: "Seed", min: 0, max: 2 ** 53, step: 1, more: true, note: "the same seed, the same answer to the same question"},
];
const SAMPLING_ORDER = "top_n_sigma;top_k;typ_p;temperature;top_p;min_p;xtc";

/* the fields, filled with v (a chat's or a model's), each empty one
   showing in grey what it falls back to (under: the values below it);
   opts.reset: the Reset button's words; opts.presets: the saved settings
   to fill them from, and a name to keep these under */
function samplingFields(v, under, opts = {}) {
  v = v || {}; under = under || {};
  /* a slider for the rough value, the field for the exact one, each
     following the other; the slider's mark, the value in grey (what an
     empty field means) */
  const field = (x) => {
    const g = under[x.k] ?? x.def, gv = parseFloat(g), own = v[x.k] ?? "";
    const id = "smpdl" + (++samplingIds);
    const slide = x.noslide ? "" :
      `<input type="range" class="smprange${own === "" ? " idle" : ""}" data-k="${x.k}" min="${x.min}" max="${x.smax ?? x.max}" step="${x.step}" value="${esc(own !== "" ? own : isFinite(gv) ? gv : x.min)}"${isFinite(gv) ? ` list="${id}"` : ""} aria-label="${esc(T(x.label))}">` +
      (isFinite(gv) ? `<datalist id="${id}"><option value="${esc(gv)}"></option></datalist>` : "");
    return `<label>${T(x.label)} <span class="smpctl">${slide}<input class="smp" data-k="${x.k}" type="number" min="${x.min}" max="${x.max}" step="${x.step}" value="${esc(own)}" placeholder="${esc(g)}"></span>
          <span class="note">${T(x.note)}</span></label>`;
  };
  const brk = (a) => (a || []).map((b) => JSON.stringify(b)).join(", ");
  const presets = Object.keys(settings._presets || {});
  return `<div class="smpbox">
      <span class="note">${T("Empty: what is shown in grey, the model's settings or what its makers recommend.")}</span>
      <span class="row smprow">${SAMPLING.filter((x) => !x.more).map(field).join("")}</span>
      <details><summary>${T("More sampling")}</summary>
        <span class="row smprow">${SAMPLING.filter((x) => x.more).map(field).join("")}</span>
        <label>${T("Order of the samplers")} <input class="smporder" value="${esc(v.samplers || "")}" placeholder="${esc(under.samplers || SAMPLING_ORDER)}" spellcheck="false">
          <span class="note">${T("llama.cpp's names separated by ';': penalties, dry, top_n_sigma, top_k, typ_p, top_p, min_p, xtc, temperature")}</span></label>
        <label>${T("DRY's breakers")} <input class="smpbrk" value="${esc(brk(v.dry_sequence_breakers))}" placeholder='"\\n", ":", "\\"", "*"' spellcheck="false">
          <span class="note">${T("where a repeated phrase is allowed to start again, as JSON strings separated by commas")}</span></label>
      </details>
      <span class="row smpacts">
        ${opts.reset ? `<button type="button" class="act quiet smpreset">${T(opts.reset)}</button>` : ""}
        ${opts.presets ? `<label${presets.length ? "" : " hidden"}>${T("Use saved settings")} <select class="smppre">${samplingPresetOptions()}</select></label>` : ""}
        ${opts.presets ? `<label>${T("Keep these settings as")} <input class="smpname" placeholder="${T("a name")}"> <button type="button" class="act quiet smpsave">${T("Keep")}</button></label>` : ""}
      </span></div>`;
}

let samplingIds = 0;
/* a slider and its field, kept together: the slider sets the field; the
   field moves the slider; an empty field leaves the slider at the grey
   value, dimmed */
function samplingSync(box, k, from) {
  const f = box.querySelector(`.smp[data-k="${k}"]`), r = box.querySelector(`.smprange[data-k="${k}"]`);
  if (!f || !r) return;
  if (from === r) { f.value = r.value; r.classList.remove("idle"); return; }
  const v = parseFloat(f.value), g = parseFloat(f.placeholder);
  r.classList.toggle("idle", f.value.trim() === "" || !isFinite(v));
  r.value = isFinite(v) ? String(v) : isFinite(g) ? String(g) : r.min;
}
document.addEventListener("input", (e) => {
  const t = e.target, box = t.closest && t.closest(".smpbox");
  if (!box || !t.dataset || !t.dataset.k) return;
  if (t.classList.contains("smprange") || t.classList.contains("smp")) samplingSync(box, t.dataset.k, t);
});
/* the slider let go: the form told, as a field's change tells it */
document.addEventListener("change", (e) => {
  const t = e.target;
  if (t.classList && t.classList.contains("smprange")) {
    const f = t.closest(".smpbox").querySelector(`.smp[data-k="${t.dataset.k}"]`);
    if (f) f.dispatchEvent(new Event("change", {bubbles: true}));
  }
});

/* the saved settings, as a select's options */
function samplingPresetOptions() {
  return `<option value="">-</option>` +
    Object.keys(settings._presets || {}).map((n) => `<option>${esc(n)}</option>`).join("");
}

/* the fields filled with v (all emptied for none) */
function samplingFill(box, v) {
  v = v || {};
  for (const x of SAMPLING) {
    box.querySelector(`.smp[data-k="${x.k}"]`).value = v[x.k] ?? "";
    samplingSync(box, x.k, null);
  }
  box.querySelector(".smporder").value = v.samplers || "";
  box.querySelector(".smpbrk").value = (v.dry_sequence_breakers || []).map((b) => JSON.stringify(b)).join(", ");
}

/* what the fields set: {temperature: 0.6, ..., samplers, dry_sequence_breakers},
   only those filled; null when the breakers are not JSON strings (said) */
function samplingRead(box) {
  const out = {};
  for (const x of SAMPLING) {
    const s = box.querySelector(`.smp[data-k="${x.k}"]`).value.trim();
    if (s !== "" && isFinite(+s)) out[x.k] = +s;
  }
  const order = box.querySelector(".smporder").value.trim();
  if (order) out.samplers = order;
  const brk = box.querySelector(".smpbrk").value.trim();
  if (brk) {
    let a = null;
    try { a = JSON.parse("[" + brk + "]"); } catch (e) {}
    if (!Array.isArray(a) || a.some((b) => typeof b !== "string")) {
      say(T("DRY's breakers are JSON strings separated by commas, such as \"\\n\", \":\""));
      return null;
    }
    out.dry_sequence_breakers = a;
  }
  return out;
}

/* only the sampling of a saved setting (which may hold a chat's
   instructions too) */
function samplingOf(p) {
  const out = {};
  for (const x of SAMPLING) if (p && p[x.k] !== undefined) out[x.k] = p[x.k];
  if (p && p.samplers) out.samplers = p.samplers;
  if (p && p.dry_sequence_breakers) out.dry_sequence_breakers = p.dry_sequence_breakers;
  return out;
}

/* Reset, a saved setting, Keep: wherever the fields are; the form they
   are in (a model's settings) told of the change */
function samplingChanged(box) {
  const f = box.closest("form");
  if (f) f.dispatchEvent(new Event("change", {bubbles: false}));
}
document.addEventListener("click", async (e) => {
  const box = e.target.closest && e.target.closest(".smpbox");
  if (!box) return;
  if (e.target.classList.contains("smpreset")) {
    samplingFill(box, {});
    samplingChanged(box);
  } else if (e.target.classList.contains("smpsave")) {
    const name = box.querySelector(".smpname").value.trim();
    if (!name) { say(T("Give these settings a name first.")); return; }
    const v = samplingRead(box);
    if (!v) return;
    settings._presets = {...(settings._presets || {}), [name]: v};
    await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
    box.querySelector(".smpname").value = "";
    /* every list of them on the page, this one at once */
    for (const sel of document.querySelectorAll(".smppre")) {
      sel.innerHTML = samplingPresetOptions();
      sel.closest("label").hidden = false;
    }
    say(T("Kept as {name}.", {name}));
  }
});
document.addEventListener("change", (e) => {
  if (!e.target.classList || !e.target.classList.contains("smppre")) return;
  const box = e.target.closest(".smpbox"), p = (settings._presets || {})[e.target.value];
  e.stopPropagation();
  if (!box || !p) return;
  samplingFill(box, samplingOf(p));
  e.target.value = "";
  samplingChanged(box);
}, true);

/* what a model samples with where a chat says nothing: the server's word
   when it runs it (/api/show, its --sampling over the recommendation),
   else the model's settings over its file's recommendation */
async function modelSampling(name) {
  const info = (typeof modelInfo !== "undefined" && modelInfo[name]) || {};
  const out = {...(info.recommended || {}), ...samplingOf((settings[name] || {}).sampling)};
  if (server.running && server.model === name) {
    try {
      const r = await fetch("/api/show", {method: "POST", body: JSON.stringify({model: name})});
      if (r.ok)
        for (const l of String((await r.json()).parameters || "").split("\n")) {
          const m = /^(\w+)\s+(.+)$/.exec(l.trim());
          if (m) out[m[1]] = m[2];
        }
    } catch (e) {}
  }
  return out;
}
