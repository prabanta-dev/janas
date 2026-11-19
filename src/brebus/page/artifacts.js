/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* artifacts.js - Brebus's page (app.html): the artifacts beside the chat. build.sh puts it into
   the page where app.html includes it. */
/* ---- artifacts: documents the model makes, beside the chat ---- */
const ART_RULE = "When you write a self-contained document the user may want to see or keep - a web page, an interactive app, a chart, a diagram, an SVG drawing, a long Markdown document, a program - put it whole between <artifact type=\"html\" title=\"A short title\"> and </artifact>. Types: html (a whole page); react (one React component exported as default, needing no props, styled with Tailwind classes; it may import react, recharts, lucide-react, d3, lodash, mathjs, papaparse, xlsx, chart.js, three, tone, and shadcn/ui from @/components/ui/<name>); mermaid (a diagram); svg; markdown (math between $$ and $$); code (add lang=\"python\" or the language). There is no network: an html page may load from cdn.jsdelivr.net or cdnjs only chart.js, d3, three (r128), plotly, mathjs, lodash, papaparse, xlsx, tone, katex, mermaid, react, react-dom, @babel/standalone and tailwind; images only as data: URLs. In html and react, await window.claude.complete(prompt) asks the model and returns its answer as a plain string; await window.storage.set(key, value), get(key) (gives {key, value} or null), delete(key) and list(prefix) keep data between visits; await window.fs.readFile(name, {encoding: \"utf8\"}) reads a file the user attached. To change one you made, write it again whole with the same title. Outside the artifact, say in a sentence what it is. Short answers and short code stay in the text.";
const NO_RUN_RULE = "You cannot run code in this chat: nothing you write is executed. Never state what a program prints or computes as if it had run; give the code and say the user can run it, or work a result out step by step and say it is worked out by hand.";
const attrsOf = (t) => {
  const a = {};
  for (const m of t.matchAll(/(\w+)\s*=\s*"([^"]*)"/g)) a[m[1]] = m[2];
  return a;
};
/* the parts of a reply: text, and artifacts (closed, or open while it
   is being written) - <artifact ...>...</artifact>, or a whole HTML or
   SVG document in a fenced block, from a model that does not write tags */
/* a title the artifact gives itself: a page's or a drawing's <title>, a
   document's first heading, a program's first comment */
function titleOf(type, content) {
  const t = (content.match(/<title>([^<]{1,80})<\/title>/i) || [])[1] ||
    (type === "markdown" && (content.match(/^#{1,3}\s+(.{1,80})$/m) || [])[1]) ||
    (type === "code" && (content.match(/^\s*(?:#(?!!)|\/\/|--|;)\s*([A-Za-z\u00c0-\u024f].{2,60}?)\s*$/m) || [])[1]);
  return t ? t.replace(/[*_`#]/g, "").trim() : "";
}
function artParts(text) {
  const parts = [];
  const re = /<artifact\b([^>]*)>([\s\S]*?)(<\/artifact>|$)|```(html|svg)[^\n]*\n(\s*(?:<!doctype html|<html|<svg)[\s\S]*?)(```|$)/gi;
  let at = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > at) parts.push({text: text.slice(at, m.index)});
    if (m[1] !== undefined || m[2] !== undefined) {
      const a = attrsOf(m[1] || "");
      const type = (a.type || "html").toLowerCase();
      /* a model may fence what it puts in the tag too */
      const content = m[2].trim().replace(/^```[\w-]*\n([\s\S]*?)\n?```$/, "$1");
      parts.push({art: {title: a.title || titleOf(type, content) || "Untitled", type,
                        lang: a.lang || "", content, open: !m[3], untitled: !a.title, raw: m[0]}});
    } else {
      const svg = m[4].toLowerCase() === "svg";
      const t = (m[5].match(/<title>([^<]{1,80})<\/title>/i) || [])[1];
      parts.push({art: {title: t || "Untitled", type: svg ? "svg" : "html",
                        lang: "", content: m[5].trim(), open: !m[6], untitled: !t, raw: m[0]}});
    }
    at = re.lastIndex;
  }
  if (at < text.length) parts.push({text: text.slice(at)});
  return parts;
}
/* every artifact of the conversation, by its title: its versions */
function artVersions(title) {
  const out = [];
  for (const m of cur ? cur.messages : [])
    if (m.role === "assistant" && m.content)
      for (const p of artParts(m.content))
        if (p.art && !p.art.open && p.art.title === title) {
          const saved = (m.saved || []).find((x) => x.title === title && x.content === p.art.content);
          out.push({...p.art, path: saved ? saved.path : ""});
        }
  return out;
}
let artShown = null;
function artShow(title, version) {
  const vs = artVersions(title);
  if (!vs.length) return;
  const k = version === undefined ? vs.length - 1 : version;
  const a = vs[k];
  artShown = {title, k}; fileShown = null;
  $("#artdown").hidden = $("#artcopy").hidden = false;
  if ($("#artpane").hidden && innerWidth < 1500 && !$("#side").classList.contains("mini")) {
    navMini(true, false); navAuto = true;
  }
  $("#artpane").hidden = false;
  $("#arttitle").textContent = title;
  $("#artver").innerHTML = vs.map((v, i) => `<option value="${i}"${i === k ? " selected" : ""}>version ${i + 1}</option>`).join("");
  $("#artver").hidden = vs.length < 2;
  const body = $("#artbody");
  body.innerHTML = "";
  $("#arterr").hidden = true;
  artFrame = null;
  const live = ["html", "svg", "react", "mermaid"].includes(a.type);
  $("#artview").hidden = a.type === "code";
  for (const b of $("#artview").children) b.classList.toggle("on", b.dataset.v === artMode);
  if (live && artMode === "preview") {
    /* in Brebus's frame (frame.html): no origin of its own, so no access to
       Brebus or to this page; no network, the libraries from Brebus */
    const f = document.createElement("iframe");
    f.setAttribute("sandbox", "allow-scripts allow-forms allow-modals allow-downloads");
    f.src = "/brebus/frame";
    artFrame = {win: null, title, art: a, el: f};
    body.append(f);
    artFrame.win = f.contentWindow;
  } else if (a.type === "markdown" && artMode === "preview") {
    const d = document.createElement("div"); d.className = "text"; render(d, a.content); body.append(d);
    decorate(d);
  } else {
    const p = document.createElement("pre"); p.textContent = a.content; body.append(p);
    p.dataset.lang = a.type === "code" ? a.lang : a.type === "react" ? "jsx" : a.type === "svg" ? "xml" : a.type === "mermaid" ? "" : a.type;
    highlight(p);
  }
  $("#artpath").textContent = a.path ? a.path : "";
  $("#artopen").hidden = !a.path;
}
$("#artver").onchange = () => artShow(artShown.title, +$("#artver").value);
/* an artifact renamed: in every reply's tag, in what was saved, its folder */
async function artRename() {
  if (!artShown) return;
  const from = artShown.title;
  const to = ((await input("A name for this artifact", from)) || "").trim().replace(/"/g, "'");
  if (!to || to === from) return;
  if (artTitles().has(to)) { say(`There is an artifact named ${to} in this chat already.`); return; }
  const r = await fetch("/brebus/artifacts/rename", {method: "POST", body: JSON.stringify({from, to})});
  if (!r.ok) { say((await r.json()).error || "not renamed"); return; }
  for (const m of cur.messages) {
    if (m.role !== "assistant" || !m.content) continue;
    for (const p of artParts(m.content))
      if (p.art && p.art.title === from) {
        const tag = p.art.raw.startsWith("<artifact")
          ? p.art.raw.replace(/^<artifact\b[^>]*>/, (h) => /title\s*=\s*"[^"]*"/.test(h)
              ? h.replace(/title\s*=\s*"[^"]*"/, `title="${to}"`) : h.replace(/>$/, ` title="${to}">`))
          : `<artifact type="${p.art.type}" title="${to}">\n${p.art.content}\n</artifact>`;
        m.content = m.content.replace(p.art.raw, () => tag);
      }
    for (const x of m.saved || []) if (x.title === from) {
      x.title = to;
      x.path = x.path.replace(/([\\/]artifact[\\/])[^\\/]+(?=[\\/][^\\/]+$)/, (_, a) => a + to.replace(/[^\w \-\u0080-\uffff]/g, ""));
    }
  }
  await saveConv(cur);
  drawLog();
  artShow(to);
}
$("#arttitle").title = "Click to rename";
$("#arttitle").onclick = () => artRename();
let artMode = "preview", artFrame = null;
for (const b of $("#artview").children)
  b.onclick = () => { artMode = b.dataset.v;
    if (artShown) artShow(artShown.title, artShown.k);
    else if (fileShown) artShowFile(fileShown.path, fileShown.title); };
$("#artfull").onclick = () => $("#artpane").classList.toggle("full");
$("#artdown").onclick = () => {
  const a = artVersions(artShown.title)[artShown.k];
  const ext = {html: "html", svg: "svg", markdown: "md", react: "jsx", mermaid: "mmd"}[a.type] ||
    ({python: "py", javascript: "js", typescript: "ts", bash: "sh", rust: "rs"}[a.lang] || a.lang || "txt");
  const u = URL.createObjectURL(new Blob([a.content], {type: "text/plain"}));
  const l = document.createElement("a");
  l.href = u; l.download = a.title.replace(/[\\/:*?"<>|]/g, "_") + "." + ext;
  document.body.append(l); l.click(); l.remove();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
};
$("#arterrx").onclick = () => { $("#arterr").hidden = true; };
$("#artfix").onclick = () => {
  const t = $("#arterr pre").textContent;
  $("#arterr").hidden = true;
  $("#artpane").classList.remove("full");
  const ask = $("#ask");
  ask.value = `The artifact "${artShown.title}" gives this error:\n\n${t}\n\nFix it, and write it again whole with the same title.`;
  ask.dispatchEvent(new Event("input"));
  $("#send").click();
};
/* what an artifact's frame asks: only the frame shown, only these */
window.addEventListener("message", async (e) => {
  const f = artFrame;
  if (!f || e.source !== f.el.contentWindow || !e.data) return;
  const d = e.data;
  if (d.janasReady) {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    f.el.contentWindow.postMessage({janasArtifact: {type: f.art.type, content: f.art.content, title: f.art.title, dark}}, "*");
  } else if (d.janasError) {
    /* each error once, the first ones kept */
    const pre = $("#arterr pre");
    const was = $("#arterr").hidden ? "" : pre.textContent;
    if (!was.includes(d.janasError) && was.length < 6000)
      pre.textContent = was ? was + "\n\n" + d.janasError : d.janasError;
    $("#arterr").hidden = false;
  } else if (d.janasCall) {
    const c = d.janasCall;
    const answer = (ok, value, error) => f.el.contentWindow &&
      f.el.contentWindow.postMessage({janasReply: {id: c.id, ok, value, error}}, "*");
    try { answer(true, await artCall(f, c)); }
    catch (err) { answer(false, undefined, String(err.message || err)); }
  }
});
let artAsking = Promise.resolve();
async function artCall(f, c) {
  if (c.op === "complete") {
    /* one question at a time, as Claude's: the model has one reply to write */
    const run = async () => {
      if (!await ensureModel()) throw new Error("no model is loaded");
      const r = await fetch("/v1/chat/completions", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({model: server.model, stream: false, max_tokens: 4096,
                              messages: [{role: "user", content: String(c.prompt).slice(0, 200000)}]})});
      const j = await r.json();
      if (!r.ok) throw new Error((j.error && j.error.message) || "the model answered " + r.status);
      return (j.choices[0].message.content || "").replace(/<think>[\s\S]*?<\/think>\s*/g, "");
    };
    const p = artAsking.then(run, run);
    artAsking = p.catch(() => {});
    return p;
  }
  if (c.op === "storage") {
    if (!ws.path) throw new Error("no workspace: storage needs one");
    const r = await fetch("/brebus/artifact-store", {method: "POST",
      body: JSON.stringify({title: f.title, act: c.act, key: c.key, value: c.value, prefix: c.prefix})});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "storage failed");
    return j;
  }
  if (c.op === "file") {
    /* the files attached to this conversation, by name or path */
    const all = (cur ? cur.messages : []).flatMap((m) => m.attached || []);
    const a = all.find((x) => x.name === c.name || x.path === c.name || x.path.endsWith("/" + c.name));
    if (!a) throw new Error(`no file "${c.name}" is attached to this conversation; attached: ${all.map((x) => x.name).join(", ") || "none"}`);
    const r = await fetch("/brebus/file?path=" + encodeURIComponent(a.path));
    if (!r.ok) throw new Error("cannot read " + a.name);
    return new Uint8Array(await r.arrayBuffer());
  }
  throw new Error("not a call an artifact can make");
}
/* code in colours and formulas, with the libraries Brebus serves (loaded
   when first needed) */
const libLoaded = new Map();
function loadLib(js, css) {
  const key = js + (css || "");
  if (!libLoaded.has(key))
    libLoaded.set(key, new Promise((ok, no) => {
      if (css) for (const c of [].concat(css)) {
        const l = document.createElement("link");
        l.rel = "stylesheet"; l.href = "/brebus/lib/" + c; document.head.append(l);
      }
      const js1 = [].concat(js);
      const next = (i) => {
        if (i >= js1.length) return ok();
        const s = document.createElement("script");
        s.src = "/brebus/lib/" + js1[i]; s.onload = () => next(i + 1); s.onerror = () => no(new Error(js1[i]));
        document.head.append(s);
      };
      next(0);
    }));
  return libLoaded.get(key);
}
async function highlight(pre) {
  try {
    await loadLib("@highlightjs/cdn-assets@11.10.0/highlight.min.js",
      matchMedia("(prefers-color-scheme: dark)").matches ? "@highlightjs/cdn-assets@11.10.0/styles/github-dark.min.css"
                                                         : "@highlightjs/cdn-assets@11.10.0/styles/github.min.css");
    const lang = (pre.dataset.lang || "").toLowerCase();
    const r = lang && hljs.getLanguage(lang) ? hljs.highlight(pre.textContent, {language: lang})
                                             : hljs.highlightAuto(pre.textContent);
    pre.innerHTML = r.value;
    pre.classList.add("hljs");
  } catch (e) { /* offline the first time: the code as it is */ }
}
/* a ```mermaid block of a reply drawn where it is, by Brebus's copy of
   Mermaid (strict: no script, no HTML in its labels) and shown as a picture,
   which runs nothing; a diagram drawn once for its text. One not yet whole,
   or that Mermaid cannot draw, stays as its text */
const diagrams = new Map();
let diagramN = 0, diagramQ = Promise.resolve();
/* one at a time: Mermaid draws one diagram at once */
function diagram(pre) { diagramQ = diagramQ.then(() => drawDiagram(pre)); }
async function drawDiagram(pre) {
  if (!pre.isConnected) return; /* the reply drawn again meanwhile */
  const src = pre.textContent;
  try {
    let svg = diagrams.get(src);
    if (!svg) {
      await loadLib("mermaid@11.4.1/dist/mermaid.min.js");
      const dark = matchMedia("(prefers-color-scheme: dark)").matches;
      mermaid.initialize({startOnLoad: false, securityLevel: "strict", suppressErrorRendering: true,
        theme: dark ? "dark" : "default", htmlLabels: false, flowchart: {htmlLabels: false}});
      const id = "brebus-diagram-" + ++diagramN;
      try { svg = (await mermaid.render(id, src)).svg; }
      finally { document.getElementById(id)?.remove(); document.getElementById("d" + id)?.remove(); }
      diagrams.set(src, svg);
    }
    if (!pre.isConnected) return;
    const fig = document.createElement("figure");
    fig.className = "diagram";
    const img = new Image();
    img.alt = "A diagram"; img.title = "A diagram, drawn from the text the model wrote";
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
    /* at the size Mermaid drew it, narrower only where the chat is */
    const vb = svg.match(/viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ([\d.]+)"/);
    if (vb) img.style.width = Math.round(+vb[1]) + "px";
    fig.append(img);
    pre.replaceWith(fig);
  } catch (e) { /* its text, as it is */ }
}
async function decorate(el) {
  for (const p of el.querySelectorAll("pre"))
    if (/^mermaid$/i.test(p.dataset.lang || "")) diagram(p); else if (p.dataset.lang) highlight(p);
  el.normalize(); /* the renderer writes text in runs: a formula in one piece */
  /* $...$ too, each one that is plainly a formula - a backslash, ^, _,
     braces or = inside, no space at its edges - written as \(...\) in its
     text first: "$5 to $10" beside a formula stays money */
  const INLINE = /(^|[^\\$\w])\$(?=\S)([^$\n]*?[\\^_{}=][^$\n]*?\S)\$(?![\w$])/g;
  let inline = false;
  const walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {acceptNode: (n) =>
    n.parentElement && n.parentElement.closest("pre, code") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT});
  for (let n; (n = walk.nextNode());) {
    const v = n.nodeValue.replace(INLINE, (_, a, f) => { inline = true; return a + "\\(" + f + "\\)"; });
    if (v !== n.nodeValue) n.nodeValue = v;
  }
  if (!inline && !/\$\$|\\\(|\\\[/.test(el.textContent)) return;
  try {
    await loadLib(["katex@0.16.11/dist/katex.min.js", "katex@0.16.11/dist/contrib/auto-render.min.js"],
                  "katex@0.16.11/dist/katex.min.css");
    renderMathInElement(el, {throwOnError: false, ignoredTags: ["pre", "code"],
      delimiters: [{left: "$$", right: "$$", display: true}, {left: "\\[", right: "\\]", display: true},
                   {left: "\\(", right: "\\)", display: false}]});
  } catch (e) {}
}
$("#artclose").onclick = () => {
  $("#artpane").hidden = true; artShown = null; fileShown = null;
  $("#artpane").classList.remove("full");
  if (navAuto) { navMini(false, false); navAuto = false; }
};
try { const w = +localStorage.getItem("brebus.artw"); if (w > 0) $("#artpane").style.width = w + "px"; } catch (e) {}
$("#artgrip").addEventListener("pointerdown", (e) => {
  e.preventDefault();
  document.body.classList.add("drag");
  const box = $("#chat").getBoundingClientRect();
  const move = (ev) => {
    const w = Math.max(320, Math.min(box.right - ev.clientX, box.width - 360));
    $("#artpane").style.width = w + "px";
  };
  const up = () => {
    document.body.classList.remove("drag");
    removeEventListener("pointermove", move); removeEventListener("pointerup", up);
    try { localStorage.setItem("brebus.artw", parseInt($("#artpane").style.width)); } catch (e) {}
  };
  addEventListener("pointermove", move); addEventListener("pointerup", up);
});
$("#artcopy").onclick = () => {
  const a = artVersions(artShown.title)[artShown.k];
  navigator.clipboard?.writeText(a.content).catch(() => {});
};
$("#artopen").onclick = () => {
  if (fileShown) return fetch("/brebus/open", {method: "POST", body: JSON.stringify({path: fileShown.path})});
  const a = artVersions(artShown.title)[artShown.k];
  /* by the system's own program, as a file: never with Brebus's origin */
  if (a.path) fetch("/brebus/open", {method: "POST", body: JSON.stringify({path: a.path})});
};
/* the artifacts of a reply saved in the workspace, the last one shown */
/* the titles of the conversation's artifacts, all of them */
function artTitles() {
  const t = new Set();
  for (const m of cur ? cur.messages : [])
    if (m.role === "assistant" && m.content)
      for (const p of artParts(m.content)) if (p.art) t.add(p.art.title);
  return t;
}
/* an artifact with no name of its own gets the next Untitled-NNN (three
   digits: they sort as they come), written into the reply's tag, so that
   the model and the page both know it by it */
function artName(reply) {
  for (const p of artParts(reply.content || "")) {
    if (!p.art || p.art.open || p.art.title !== "Untitled") continue;
    let k = 1;
    const have = artTitles();
    while (have.has("Untitled-" + String(k).padStart(3, "0"))) k++;
    const name = "Untitled-" + String(k).padStart(3, "0");
    const tag = `<artifact type="${p.art.type}"${p.art.lang ? ` lang="${p.art.lang}"` : ""} title="${name}">\n${p.art.content}\n</artifact>`;
    reply.content = reply.content.replace(p.art.raw, () => tag);
  }
}
async function artSave(reply) {
  artName(reply);
  let last = null;
  for (const p of artParts(reply.content || ""))
    if (p.art && !p.art.open) {
      const r = await fetch("/brebus/artifacts", {method: "POST", body: JSON.stringify(p.art)});
      const j = await r.json();
      if (r.ok) (reply.saved = reply.saved || []).push({title: p.art.title, content: p.art.content, path: j.path});
      last = p.art.title;
    }
  if (last) artShow(last);
}

function apiMessages() {
  const out = [];
  const sys = [cur.system, $("#useart").checked ? ART_RULE : "",
               /* without the python tool nothing runs: a model that said what
                  its code printed made it up (17 Nov, five CRCs invented) */
               $("#usecode").checked ? "" : NO_RUN_RULE,
               $("#usetools").checked && tools.catalog ? tools.catalog : ""].filter(Boolean).join("\n\n");
  if (sys) out.push({role: "system", content: sys});
  for (const m of cur.messages) {
    if (m.role === "user") {
      /* pictures as pictures, for a model that sees them */
      const pics = modelInfo[server.model]?.vision
        ? (m.attached || []).filter((a) => a.media === "image" && pictures.has(a.path)) : [];
      out.push({role: "user", content: pics.length
        ? [{type: "text", text: userContent(m)},
           ...pics.map((a) => ({type: "image_url", image_url: {url: pictures.get(a.path)}}))]
        : userContent(m)});
    }
    else if (m.role === "assistant") {
      const a = {role: "assistant", content: m.content || ""};
      if (m.tool_calls) a.tool_calls = m.tool_calls;
      out.push(a);
    } else if (m.role === "tool")
      out.push({role: "tool", tool_call_id: m.tool_call_id, content: m.content});
  }
  return out;
}

