/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* models.js - Brebus's page (app.html): the models' settings, Hugging Face, the downloads, the server. build.sh puts it into
   the page where app.html includes it. */
/* each model's settings, kept by Brebus, used when it is loaded */
let settings = {};
const CTX_OWN = 16384; /* janas-server's context when none is asked */
/* the contexts to choose from: 4,096 tokens and each double, up to the
   most the model was trained for, that one too (40,960...), 1,048,576 at
   most (Brebus's own bound); 262,144 when the model does not say */
function ctxSizes(max) {
  const top = max >= 512 ? Math.min(max, 1 << 20) : 262144, out = [];
  for (let n = 4096; n < top; n *= 2) out.push(n);
  out.push(top);
  return out;
}
/* "32,768 tokens (about 25,000 words)": a word of English is about 4/3 of
   a token, two figures of it */
function ctxText(n) {
  const w = n * 0.75, p = Math.pow(10, Math.max(0, Math.floor(Math.log10(w)) - 1));
  return `${n.toLocaleString()} tokens (about ${(Math.round(w / p) * p).toLocaleString()} words)`;
}
function setForm(x) {
  const v = settings[x.name] || {};
  const sizes = ctxSizes(x.context);
  const other = v.context && !sizes.includes(v.context);
  const most = sizes[sizes.length - 1];
  return `<form class="setf plain" data-m="${esc(x.name)}" onsubmit="return false">
    <label>How much of the chat it keeps in mind
      <select name="context"><option value="">${ctxText(Math.min(CTX_OWN, most))}: Janas's own</option>
        ${sizes.map((n) => `<option value="${n}"${v.context === n ? " selected" : ""}>${ctxText(n)}${
          n === x.context ? ": the most this model takes" : ""}</option>`).join("")}
        ${other ? `<option value="${v.context}" selected>${ctxText(v.context)}</option>` : ""}</select>
      <span class="note">More is remembered with more, but it takes more memory and the first answer of a long chat comes later. When it fills up, the earlier messages are summarized to make room (the ring under the field shows how full it is).</span></label>
    <label${x.thinks === false || x.thinks === 0 ? " hidden" : ""}>Think before answering
      <select name="think"><option value="">no</option><option value="on"${v.think === true ? " selected" : ""}>yes</option></select>
      <span class="note">Some models can reason step by step before they answer: better on hard questions, slower on every one. The reasoning is shown, folded, above the answer. This is how a new chat starts; the brain under the chat's field turns it on or off for that chat, at any message. Other programs choose with each request.</span></label>
    <details><summary>Advanced</summary>
      ${[["prompt_on", "Where it reads your message", "the message and the documents given with it, before it answers"],
         ["drafts_on", "Where it checks its guesses", "words guessed ahead, checked a few at a time, to answer faster"],
         ["reply_on", "Where it writes the reply", "one word at a time"]].map(([n, t, about]) => `
      <label>${t}
        <select name="${n}"><option value="">chosen by Janas, measuring this computer</option>
          ${[["cpu", "the processor"], ["gpu", "the graphics card (GPU)"], ["npu", "the NPU"], ["npu+gpu", "the NPU and the GPU together"]]
            .map(([d, l]) => `<option value="${d}"${v[n] === d ? " selected" : ""}>${l}</option>`).join("")}</select>
        <span class="note">${about}</span></label>`).join("")}
      <p class="note">Janas tries what this computer has and keeps the fastest; Info shows what it chose. Choose here only to try a device, or to leave one free. The processor always takes part.</p>
      <label>How it reaches the graphics card
        <select name="gpu_api"><option value="">chosen by Janas: Vulkan, or the card's maker's own where there is no Vulkan</option>
          ${[["vulkan", "Vulkan (every maker)"], ["cuda", "CUDA (NVIDIA)"], ["hip", "HIP (AMD, with ROCm or the HIP SDK)"]]
            .map(([d, l]) => `<option value="${d}"${v.gpu_api === d ? " selected" : ""}>${l}</option>`).join("")}</select>
        <span class="note">Vulkan does the most on the card; CUDA and HIP are its makers' own ways, for the parts Janas has written for them.</span></label>
      <label>Chats answered at the same time <input name="parallel" type="number" min="1" max="8" placeholder="from the memory" value="${v.parallel || ""}">
        <span class="note">For other programs using the model through the server too: each one takes memory of its own.</span></label>
    </details>
    <span class="note">These take effect the next time the model is loaded.</span></form>`;
}
async function saveSettings(f) {
  const v = {};
  const c = parseInt(f.context.value, 10), p = parseInt(f.parallel.value, 10);
  if (c >= 512) v.context = c;
  if (p >= 1 && p <= 8) v.parallel = p;
  if (f.think.value) v.think = f.think.value === "on";
  for (const n of ["prompt_on", "drafts_on", "reply_on", "gpu_api"]) if (f[n].value) v[n] = f[n].value;
  settings[f.dataset.m] = v;
  await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
}

/* a file of Hugging Face: hf:owner/repo/file, or its page's address
   (huggingface.co/owner/repo/blob/main/file, or resolve/) */
function hfSpec(t) {
  t = t.trim();
  const m = t.match(/^https?:\/\/(?:www\.)?huggingface\.co\/([\w.-]+)\/([\w.-]+)\/(?:blob|resolve)\/[\w.-]+\/([^?#]+)/);
  if (m) return `hf:${m[1]}/${m[2]}/${decodeURIComponent(m[3])}`;
  return t.startsWith("hf:") ? t : "";
}
/* one field: an address gets that file, words search */
/* the search's results gone, the catalog in view again */
function clearHf() {
  $("#hfres").innerHTML = "";
  $("#hfsearch").q.value = "";
  const [sb, gb] = $("#hfgo").children; sb.hidden = false; gb.hidden = true;
}
const isAddress = (v) => /^(https?:\/\/|hf:)/i.test(v.trim());
$("#hfsearch").q.addEventListener("input", () => {
  if (!$("#hfsearch").q.value.trim()) $("#hfres").innerHTML = "";
  const get = isAddress($("#hfsearch").q.value);
  const [s, g] = $("#hfgo").children;
  s.hidden = get; g.hidden = !get;
});
function hfGet(v) {
  const spec = hfSpec(v);
  if (!/^hf:[\w.-]+\/[\w.-]+\/[\w.\/-]+\.(gguf|jns)$/.test(spec)) {
    say("A GGUF or JNS file of Hugging Face: its address, or hf:owner/repo/file.gguf");
    return;
  }
  getModel(spec);
}

/* Hugging Face searched by words, through Brebus */
const num = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "k" : String(n);
$("#hfsearch").onsubmit = async () => {
  const q = $("#hfsearch").q.value.trim();
  if (!q) return false;
  if (isAddress(q)) { hfGet(q); return false; }
  $("#hfres").innerHTML = '<div class="note">Searching…</div>';
  const j = await ask("hf/search?q=" + encodeURIComponent(q));
  if (j.error) { $("#hfres").innerHTML = `<div class="note">${esc(j.error)}</div>`; return false; }
  $("#hfres").innerHTML = `<div class="hfhead"><b>On Hugging Face</b> <span class="note">${j.models.length} found</span>
    <button class="act quiet" id="hfclear" title="Clear these results: back to Janas's own models">Clear</button></div>` + j.models.map((m) =>
    `<button class="hfm" data-r="${esc(m.id)}" title="Show its files: their sizes, whether they run here, Get"><span>${esc(m.id)}</span><small>${num(m.downloads)} downloads · ${num(m.likes)} likes</small></button>
     <div class="hffiles" data-r="${esc(m.id)}" hidden></div>`).join("") || '<div class="note">Nothing found with GGUF files.</div>';
  for (const b of document.querySelectorAll("#hfres .hfm")) b.onclick = () => hfFiles(b.dataset.r);
  $("#hfclear").onclick = clearHf;
  return false;
};
async function hfFiles(repo) {
  const box = document.querySelector(`.hffiles[data-r="${CSS.escape(repo)}"]`);
  if (!box.hidden) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = '<div class="note hff">Reading…</div>';
  const j = await ask("hf/files?repo=" + encodeURIComponent(repo));
  if (j.error) { box.innerHTML = `<div class="note hff">${esc(j.error)}</div>`; return; }
  const mem = modelInfo.__mem || 0;
  box.innerHTML = j.files.map((f) => {
    const why = f.split ? "split in parts: not taken yet" : f.mmproj ? "a vision encoder, not a model" : "";
    const st = fileState(f.file);
    const gguf = /\.gguf$/i.test(f.file);
    return `<div class="hff" data-f="${esc(f.file)}"><span class="fname" title="${esc(f.file)}">${esc(f.file)}</span>
      <small class="fsize" title="${gguf ? "Converting it here takes about twice its size free on the disk" : ""}">${gib(f.bytes)}${gguf ? ` · ${gib(f.bytes * 2)} to convert` : ""}</small>
      <span class="hfst" title="${esc(st.tip)}"><span class="badge ${st.cls}">${esc(st.label)}</span>
      ${!why && gguf && !st.final ? `<button class="act quiet chk" title="Read the file's first megabyte from Hugging Face: does this engine run its kind of model?">Check</button>` : ""}</span>
      ${why ? `<small class="fwhy">${why}</small>` : (st.final ? `<button class="act quiet get" disabled title="It is in My models already">Get</button>`
                : `<button class="act quiet get" data-s="hf:${esc(repo)}/${esc(f.file)}" data-b="${f.bytes || 0}" title="Download it, convert it for Janas and check it">Get</button>`)}</div>`;
  }).join("") || '<div class="note hff">No GGUF or JNS file.</div>';
  for (const b of box.querySelectorAll("button.get")) b.onclick = () => getModel(b.dataset.s, +b.dataset.b);
  /* a file looked at before its download: its header read for its kind */
  for (const b of box.querySelectorAll("button.chk"))
    b.onclick = async () => {
      const row = b.closest(".hff"), file = row.dataset.f;
      b.disabled = true; b.textContent = "Checking…";
      const c = await ask(`hf/check?repo=${encodeURIComponent(repo)}&file=${encodeURIComponent(file)}`).catch(() => ({state: "unknown"}));
      const st = {
        likely: {label: "Likely to run", cls: "fits", tip: `Its kind of model (${c.arch}) is one this engine runs. Not yet proven: the conversion and the first load will tell.`},
        unsupported: {label: "Not supported yet", cls: "big", tip: `Its kind of model (${c.arch}) is not one this engine runs yet: the conversion would refuse it.`},
        closed: {label: "Not available", cls: "big", tip: (c.why || "Hugging Face would not give it") + ". A gated model must be accepted on its page first, with a token."},
        unknown: {label: "Not known", cls: "", tip: "Its header could not be read: only a conversion would tell."},
      }[c.state] || {label: "Not known", cls: "", tip: ""};
      row.querySelector(".hfst").innerHTML = `<span class="badge ${st.cls}" title="${esc(st.tip)}">${esc(st.label)}</span>`;
      row.querySelector(".hfst").title = st.tip;
    };
}

/* how far a file of the Hub is from running here: found; converted
   (its JNS among the models); loaded now */
function fileState(file) {
  /* by their letters and digits alone: Qwen3-0.6B-Q4_K_M is the
     catalog's qwen3-0.6b-q4km */
  const key = (s) => s.replace(/^.*\//, "").replace(/\.(gguf|jns)$/i, "").toLowerCase().replace(/[^a-z0-9]/g, "");
  const name = Object.keys(modelInfo).find((n) => key(n) === key(file)) || "";
  if (name && isLoaded(name))
    return {label: "Loaded here", cls: "fits", final: true, tip: "Converted, checked and running now."};
  if (name)
    return {label: "Converted here", cls: "fits", final: true, tip: "Converted and checked: it is in My models."};
  return {label: "Found", cls: "", tip: "Listed on Hugging Face: whether it runs here is not known yet. Check reads its header; Get downloads it, converts it and checks it."};
}

/* downloads, by janas-get: several at once, followed every second while
   any goes on */
let getting = false, getTimer = 0;
async function getModel(name, bytes) {
  /* got again: its ended boxes may show again */
  for (const k of [...dlClosed]) if (k.startsWith(name + "\u0000")) { dlClosed.delete(k); dlFading.delete(k); }
  /* its buttons off at once, before the first status comes back */
  for (const b of document.querySelectorAll("button.get, button.vget"))
    if ((b.dataset.m || b.dataset.s) === name) { b.disabled = true; b.textContent = "Getting…"; }
  const r = await fetch("/brebus/get/start", {method: "POST", body: JSON.stringify({model: name, bytes: bytes || 0})});
  const j = await r.json();
  if (!r.ok) { say(j.error || "it did not start"); getStatus(); return; }
  getStatus();
}
/* the ended downloads' boxes closed (by their X, or by themselves after a
   while), by model and outcome; a new Get of that model shows it again */
const dlClosed = new Set(), dlFading = new Map();
let dlLast = "";
async function getStatus() {
  const all = (await ask("get")).downloads || [];
  const box = $("#dl");
  const key = (g) => g.model + "\u0000" + g.code;
  const shown = all.filter((g) => g.running || !dlClosed.has(key(g)));
  box.hidden = !shown.length;
  const html = shown.map((g) => {
    if (g.running) {
      const bar = g.total_gb ? `${g.done_gb.toFixed(2)} of ${g.total_gb.toFixed(2)} GB (${g.percent}%), ${g.mb_s} MB/s` : "";
      const step = g.step.startsWith(g.model + ": ") ? g.step.slice(g.model.length + 2) : g.step;
      /* the bar the whole width; under it the figures on the left, Stop on
         the right, its edge the bar's: the figures change, it stays */
      return `<div class="dlone"><div><b>${esc(g.model)}</b>: ${esc(step || "starting")}</div>
        <progress max="100" value="${g.percent}"></progress>
        <div class="dlrow"><span class="mono">${esc(bar)}</span><button class="act quiet dlstop" data-m="${esc(g.model)}" title="Stop the download: Get again goes on from where it stopped">Stop</button></div></div>`;
    }
    const k = key(g);
    return `<div class="dlone dlend${dlFading.get(k) === "fade" ? " fading" : ""}" data-k="${esc(k)}"><span>${g.code === 0 ? `<b>${esc(g.model)}</b> is ready: load it in My models.`
      : g.code === -2 ? `<b>${esc(g.model)}</b>: stopped. Get it again to go on from where it was.`
      : `<b>${esc(g.model)}</b> did not download: ${esc(g.said || "janas-get ended with " + g.code)}`}</span><button class="act quiet dlx" title="Close" aria-label="Close">×</button></div>`;
  }).join("");
  /* drawn again only when something in it changed: a box redrawn under the
     pointer loses its tooltip (and its hover) every second */
  if (html !== dlLast) {
    dlLast = html;
    box.innerHTML = html;
    for (const b of box.querySelectorAll(".dlstop"))
      b.onclick = async () => { await fetch("/brebus/get/stop", {method: "POST", body: JSON.stringify({model: b.dataset.m})}); getStatus(); };
    for (const b of box.querySelectorAll(".dlx"))
      b.onclick = () => { dlClosed.add(b.closest(".dlend").dataset.k); getStatus(); };
  }
  /* an ended one closes by itself: ten seconds, then a second's fading */
  for (const g of shown)
    if (!g.running && !dlFading.has(key(g))) {
      const k = key(g);
      dlFading.set(k, "wait");
      setTimeout(() => {
        dlFading.set(k, "fade");
        const el = [...box.querySelectorAll(".dlend")].find((e) => e.dataset.k === k);
        if (el) el.classList.add("fading");
        setTimeout(() => { dlClosed.add(k); getStatus().catch(() => {}); }, 1000);
      }, 10000);
    }
  /* a Get whose download goes on: off, saying so (a second click would
     start it twice); on again when it ends */
  const busyNames = new Set(all.filter((g) => g.running).map((g) => g.model));
  for (const b of document.querySelectorAll("button.get, button.vget")) {
    const name = b.dataset.m || b.dataset.s;
    if (!name) continue; /* in My models already: off for good */
    const on = busyNames.has(name);
    put(b, "disabled", on);
    put(b, "textContent", on ? "Getting…" : "Get");
  }
  const going = all.some((g) => g.running);
  clearTimeout(getTimer);
  if (going) { getting = true; getTimer = setTimeout(() => getStatus().catch(() => {}), 1000); return; }
  if (getting) { getting = false; await mine(); await status(); }
}
let server = {running: false, loaded: []};
/* a model among those loaded (several may be, each with its server) */
const isLoaded = (name) => (server.loaded || []).some((l) => l.model === name);
let ctxOf = {}; /* the context each model was loaded with, by its name */
async function status() {
  server = await ask("server");
  /* the context the model runs with, as Ollama's /api/ps tells it */
  if (server.running && !ctxOf[server.model])
    try {
      const ps = await (await fetch("/api/ps")).json();
      const m = (ps.models || [])[0];
      if (m && m.context_length) ctxOf[server.model] = m.context_length;
    } catch (e) {}
  server.context = server.running ? ctxOf[server.model] || 0 : 0;
  if (typeof ctxRing === "function") ctxRing();
  if (typeof readReasons === "function") readReasons();
  /* the model under the chat: the one running, or the one that will be */
  const next = nextModel(), known = Object.keys(modelInfo).length > 0;
  /* written only when it changes: rewriting what the pointer is over, even
     with the same words, hides its tooltip until the pointer leaves */
  put($("#modelbtn").firstChild, "textContent", server.running ? server.model
    : next || (known ? "Choose a model" : "…"));
  $("#modelbtn").classList.toggle("off", !server.running);
  if (!modelLoading)
    put($("#loaded"), "textContent", server.running ? "" : next ? "loads with your message" : "");
  $("#modelset").hidden = !(server.running || next);
  const base = (u) => u.replace(/\/v1$/, "");
  const srvHtml = server.running
    ? `<dt>Model</dt><dd>${esc(server.model)}</dd>
       <dt>Address for programs</dt><dd><code>${esc(server.url)}</code> <span class="note">for those that speak OpenAI's language (most); <code>${esc(base(server.url))}</code> for those that speak Ollama's or Anthropic's</span></dd>
       ${server.lan_url ? `<dt>On the local network</dt><dd><code>${esc(server.lan_url)}</code>, <code>${esc(base(server.lan_url))}</code></dd>` : ""}
       ${server.https ? `<dt>Certificate</dt><dd>Brebus's own, its fingerprint (SHA-256): <code class="fp">${esc(server.fingerprint)}</code></dd>` : ""}
       <dt>Key</dt><dd>${server.api_key ? "<code>" + esc(server.api_key) + "</code>" : "none"}</dd>
       <dt>Engine monitor</dt><dd><a href="/monitor" target="_blank" rel="noopener">open it</a>: the reply being written, the engine's memory and devices</dd>
       ${(server.loaded || []).length > 1 ? `<dt>Loaded</dt><dd>${server.loaded.map((l) => `<div class="ldone"><span><b>${esc(l.model)}</b> <code>${esc(l.url)}</code>${l.lan_url ? ` <code>${esc(l.lan_url)}</code>` : ""}</span> <button class="act quiet unl" data-m="${esc(l.model)}" title="Unload it, freeing its memory">Unload</button></div>`).join("")}<span class="note">The chat uses the one it names; each has its own address for programs.</span></dd>` : ""}`
    : `<dt>State</dt><dd>off${server.error ? " - " + esc(server.error) : ""}</dd>
       <dt>To start it</dt><dd>choose a model under the chat's field, or load one in My models</dd>`;
  if (put($("#srv"), "innerHTML", srvHtml))
    for (const b of $("#srv").querySelectorAll("button.unl"))
      b.onclick = () => unload(b.dataset.m);
  for (const b of document.querySelectorAll("button.load"))
  {
    const cur = server.running && b.dataset.m === server.model, on = isLoaded(b.dataset.m);
    put(b, "textContent", cur ? "In use" : on ? "Use" : "Load");
    put(b, "disabled", cur); /* the one used now: nothing to load */
    put(b, "title", cur ? "The model the chat uses now" : on ? "Loaded already: use it now, at once"
      : "Load it: the chat and other programs use it");
  }
}
/* the server's own settings, under "_server" with the models' ones */
function srvForm() {
  const f = $("#srvset"), v = settings._server || {};
  f.lan.checked = !!v.lan;
  f.https.checked = !!v.https;
  f.api_key.value = v.api_key || "";
  f.origins.value = (v.origins || []).join("\n");
  f.ttl.value = v.ttl || "";
  f.max_loaded.value = v.max_loaded || "";
}
async function srvSave() {
  const f = $("#srvset");
  const origins = f.origins.value.split(/\s+/).filter((x) => /^https?:\/\/[^\s]+$/.test(x));
  const key = f.api_key.value.trim();
  if (key && !/^[\x21-\x7e]{8,199}$/.test(key)) { say("A key is 8 or more letters, digits or signs, no spaces."); return; }
  if (f.lan.checked && !key && !await yes("Without a key anyone on this network can use the model.", "No key", "Go on")) return;
  const most = Math.min(4, Math.max(0, Math.round(+f.max_loaded.value || 0)));
  settings._server = {lan: f.lan.checked, https: f.https.checked, api_key: key, origins, ttl: +f.ttl.value || 0, max_loaded: most};
  await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
  if (server.running && await yes("Saved. Start the model's server again now, to use them?", "Settings", "Start again")) {
    const name = server.model;
    await fetch("/brebus/server/stop", {method: "POST", body: "{}"}); /* all: they answer alike */
    const b = document.querySelector(`button.load[data-m="${CSS.escape(name)}"]`);
    if (b) await loadModel(name, b);
  }
  $("#srvnote").textContent = server.running ? "" : "Saved: used at the next Load.";
}
$("#srvsave").onclick = srvSave;
$("#genkey").onclick = () => {
  const a = new Uint8Array(24); crypto.getRandomValues(a);
  $("#srvset").api_key.value = "jns-" + Array.from(a, (x) => x.toString(16).padStart(2, "0")).join("");
};

function loadModel(name, btn) {
  btn.disabled = true; btn.textContent = "Loading…";
  return whileLoading(async () => {
    if (!isLoaded(name)) $("#loaded").textContent = "Loading " + name + "…";
    sendButton();
    const r = await fetch("/brebus/server/start", {method: "POST",
      body: JSON.stringify({model: name, ...(settings[name] || {})})});
    const j = await r.json().catch(() => ({}));
    btn.disabled = false;
    if (!r.ok) say(j.error || "it did not start");
    else remember(name);
    await status();
  });
}
async function unload(name) {
  /* the one named, else all */
  await fetch("/brebus/server/stop", {method: "POST", body: name ? JSON.stringify({model: name}) : "{}"});
  await status();
}

