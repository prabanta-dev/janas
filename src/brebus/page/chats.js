/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* chats.js - Brebus's page (app.html): the conversations, their messages drawn, the tools, the attachments. build.sh puts it into
   the page where app.html includes it. */
/* conversations, a file each in Brebus's data folder */
let convs = [], cur = null;
const newId = () => Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
async function saveConv(c) {
  c.updated = Date.now();
  await fetch("/brebus/chats/" + c.id, {method: "POST", body: JSON.stringify(c)});
  refreshTree(); /* its title, its place among the newest */
}
async function readConvs() {
  /* those the first Brebus kept in the browser, moved once */
  let old = [];
  try { old = JSON.parse(localStorage.getItem("brebus.convs") || "[]"); } catch (e) {}
  if (old.length) {
    for (const c of old) {
      c.id = newId(); c.updated = Date.now();
      await saveConv(c);
    }
    try { localStorage.removeItem("brebus.convs"); } catch (e) {}
  }
  convs = (await ask("chats")).chats;
}
async function openConv(id) {
  followLog = true; /* a chat opened at its end */
  const r = await fetch("/brebus/chats/" + id);
  cur = r.ok ? await r.json() : null;
  /* the chat used last, opened again at the next start */
  if (cur) try { localStorage.setItem("brebus.last", JSON.stringify({ws: ws.path, id: cur.id})); } catch (e) {}
  listConvs(); drawLog();
}
/* the chats of every recent workspace, searched as the user types */
let searchTimer = null;
async function searchChats() {
  const q = $("#csearch").value.trim();
  const box = $("#hits");
  box.innerHTML = "";
  $("#tree").classList.toggle("finding", !!q);
  if (!q) return;
  const j = await ask("chats/search?q=" + encodeURIComponent(q));
  if (!j.chats.length) {
    const n = document.createElement("div"); n.className = "hitnote note"; n.textContent = "Nothing found.";
    box.appendChild(n); return;
  }
  for (const h of j.chats) {
    const d = document.createElement("div");
    d.className = "hit";
    d.innerHTML = `${esc(h.title || "New chat")}<small>📁 ${esc(h.workspace_name)}${h.snippet ? " · " + esc(h.snippet) : ""}</small>`;
    d.title = h.workspace;
    d.onclick = async () => {
      $("#csearch").value = ""; searchChats();
      await goChat(h.workspace, h.id);
    };
    box.appendChild(d);
  }
  if (j.more) {
    const n = document.createElement("div"); n.className = "hitnote note";
    n.textContent = `and ${j.more} more: say more words`; box.appendChild(n);
  }
}
$("#csearch").addEventListener("input", () => {
  clearTimeout(searchTimer); searchTimer = setTimeout(() => searchChats().catch(() => {}), 250);
});
/* the side bar's chats and the chat's own bar, drawn again */
function listConvs() {
  markTree();
  chatBar();
}
function newConv() {
  followLog = true;
  cur = {id: newId(), title: "", model: "", system: "", messages: [], updated: Date.now()};
  convs.unshift({id: cur.id, title: "", model: "", updated: cur.updated, messages: 0});
  show("chat");
  listConvs(); drawLog();
}
/* a new chat of no project: in the free chats' place */
async function newFree() {
  if (!ws.free) { await wsOpen("", "free", "new"); show("chat"); return; }
  newConv();
}
async function renameConv() {
  const t = await input("A title for this chat", cur.title || "");
  if (t === null) return;
  cur.title = t.trim();
  await saveConv(cur);
  const e = convs.find((c) => c.id === cur.id); if (e) e.title = cur.title;
  listConvs(); refreshTree();
}
async function deleteConv() {
  if (!await yes(`"${cur.title || "New chat"}" and its messages are deleted from the workspace.`, "Delete this chat?", "Delete")) return;
  await fetch("/brebus/chats/" + cur.id, {method: "DELETE"});
  convs = convs.filter((c) => c.id !== cur.id);
  cur = null;
  const live = convs.filter((c) => !c.archived);
  if (live.length) await openConv(live[0].id); else newConv();
  refreshTree();
}
async function exportConv(c) {
  if (!c || c instanceof Event) c = cur;
  const how = await dialog({title: "Export this chat", text: "As a Markdown file in the workspace's exports/, or as a PDF through the system's print dialog (Print to file).",
    buttons: [{label: "Markdown", value: "md", primary: true}, {label: "PDF", value: "pdf"}, {label: "Cancel", value: "no"}]});
  if (how.button === "no") return;
  if (how.button === "pdf") {
    if (c !== cur) { say("Open the chat first: the PDF is printed from the page that shows it."); return; }
    document.title = cur.title || "Chat"; window.print(); document.title = "Brebus"; return;
  }
  const md = [`# ${c.title || "Chat"}`, ""];
  for (const m of c.messages)
    md.push(`## ${m.role === "user" ? "You" : "Assistant"}`, "", m.content || "", "");
  /* saved by Brebus in the downloads folder: a window has no downloads */
  const r = await fetch("/brebus/export", {method: "POST",
    body: JSON.stringify({name: c.title || "chat", text: md.join("\n")})});
  const j = await r.json();
  say(r.ok ? j.path : j.error || "not saved", r.ok ? "Exported" : "Not exported");
}

/* the reasoning and the tools' boxes the user opened: a message is drawn
   anew at each piece of a reply, and a box would close again at once */
const opened = new WeakSet();
function keepOpen(t, m) {
  if (opened.has(m)) t.open = true;
  t.addEventListener("toggle", () => { if (t.open) opened.add(m); else opened.delete(m); });
}

/* the pictures a reply names from the workspace - ![a circle](circle.svg),
   a file its Python code wrote - shown in it, read through Brebus (an img
   runs no script, an SVG's neither); addresses of the web are not loaded */
const PICTURE = /!\[([^\]\n]*)\]\(\s*<?([^()\s<>]+\.(?:png|jpe?g|gif|webp|svg|bmp))>?\s*\)/gi;
function wsFile(name) {
  if (!ws.path) return null;
  name = name.replace(/^sandbox:/, "");
  const m = name.match(/^\/mnt\/data\/(.*)$/); /* the cage's own name for the workspace */
  if (m) name = m[1];
  else if (/^[a-z][a-z0-9+.-]*:\/\//i.test(name)) return null; /* the web's */
  else if (/^([A-Za-z]:)?[\\/]/.test(name)) return name; /* a whole path: Brebus decides */
  const sep = ws.path.includes("\\") ? "\\" : "/";
  return ws.path.replace(/[\\/]+$/, "") + sep + name.replace(/^\.[\\/]/, "").replace(/[\\/]/g, sep);
}
/* a picture of the workspace, beside the chat in the artifacts' pane: as
   the picture it is, its SVG's text under Code, the file kept where it is */
let fileShown = null;
async function artShowFile(path, title) {
  artShown = null; artFrame = null; fileShown = {path, title};
  /* the file is in the workspace already: nothing to download or copy */
  $("#artdown").hidden = $("#artcopy").hidden = true;
  if ($("#artpane").hidden && innerWidth < 1500 && !$("#side").classList.contains("mini")) {
    navMini(true, false); navAuto = true;
  }
  $("#artpane").hidden = false;
  $("#arttitle").textContent = title;
  $("#artver").hidden = true;
  $("#arterr").hidden = true;
  $("#artview").hidden = !/\.svg$/i.test(path);
  for (const b of $("#artview").children) b.classList.toggle("on", b.dataset.v === artMode);
  $("#artpath").textContent = path;
  $("#artopen").hidden = false;
  const body = $("#artbody");
  body.innerHTML = "";
  try {
    const blob = await fileBlob(path);
    if (artMode !== "preview" && /\.svg$/i.test(path)) {
      const pre = document.createElement("pre"); pre.textContent = await blob.text();
      pre.dataset.lang = "xml"; body.append(pre); highlight(pre);
    } else {
      /* an img runs no script, an SVG's neither; as a data: address, the
         only kind the page's policy lets in */
      const img = document.createElement("img");
      img.className = "artpic"; img.alt = title;
      img.src = await dataURL(blob.type === "application/octet-stream" && /\.svg$/i.test(path)
                              ? new Blob([await blob.text()], {type: "image/svg+xml"}) : blob);
      body.append(img);
    }
  } catch (e) {
    const p = document.createElement("p"); p.className = "note";
    p.textContent = `${title}: not found in the workspace (${path})`; body.append(p);
  }
}
/* the pictures a reply names, by their paths */
function picRefs(text) {
  const out = [];
  for (const m of (text || "").matchAll(PICTURE)) {
    const path = wsFile(m[2]);
    if (path) out.push({path, title: m[1] || m[2].split(/[\\/]/).pop()});
  }
  return out;
}
function renderText(el, text) {
  let at = 0;
  for (const m of text.matchAll(PICTURE)) {
    const path = wsFile(m[2]);
    if (!path) continue;
    if (m.index > at) { const t = document.createElement("div"); render(t, text.slice(at, m.index)); el.append(t); }
    const title = m[1] || m[2].split(/[\\/]/).pop();
    const c = document.createElement("button");
    c.className = "artcard";
    c.innerHTML = `<span>🖼</span><span><b>${esc(title)}</b><br><small>${esc(m[2])} · open it beside the chat</small></span>`;
    c.title = "Open it beside the chat";
    c.onclick = () => artShowFile(path, title);
    el.append(c);
    at = m.index + m[0].length;
  }
  if (at === 0) return render(el, text);
  if (at < text.length) { const t = document.createElement("div"); render(t, text.slice(at)); el.append(t); }
}

function bubble(m) {
  const d = document.createElement("div");
  d.className = "msg " + m.role;
  if (m.think) {
    const t = document.createElement("details");
    t.className = "think";
    t.innerHTML = "<summary>Reasoning</summary><div class=\"thinktext\"></div>";
    t.lastChild.textContent = m.think;
    keepOpen(t, m);
    d.append(t);
  }
  if (m.role === "tool") {
    const t = document.createElement("details");
    keepOpen(t, m);
    t.innerHTML = `<summary>${esc(m.name)}${m.denied ? " (not allowed)" : ""}</summary>`;
    const pre = document.createElement("pre");
    pre.textContent = (m.args ? "arguments: " + m.args + "\n\n" : "") + (m.content || "");
    t.append(pre); d.append(t);
    return d;
  }
  const body = document.createElement("div");
  if (m.role === "assistant") {
    const parts = m.content ? artParts(m.content) : [];
    for (const p of parts) {
      if (p.text !== undefined) {
        if (!p.text.trim()) continue;
        const t = document.createElement("div"); t.className = "text"; renderText(t, p.text); body.append(t);
        if (!busy || m !== cur.messages.at(-1)) decorate(t);
      } else {
        const c = document.createElement("button");
        c.className = "artcard";
        c.innerHTML = `<span>📄</span><span><b>${esc(p.art.title)}</b><br><small>${esc(p.art.type)}${p.art.open ? " · being written…" : " · open it beside the chat"}</small></span>`;
        c.title = p.art.open ? "Being written" : "Open it beside the chat";
        if (!p.art.open) c.onclick = () => artShow(p.art.title);
        body.append(c);
      }
    }
    if (!parts.length && m.tool_calls) body.textContent = "…";
  } else body.textContent = m.content || "";
  d.append(body);
  if (m.attached && m.attached.length) {
    const a = document.createElement("div");
    a.className = "att";
    a.textContent = "📎 " + m.attached.map((x) => x.name).join(", ");
    d.append(a);
  }
  if (m.found) {
    const f = document.createElement("details");
    f.className = "att";
    f.innerHTML = "<summary>Passages found in the documents</summary>";
    const pre = document.createElement("pre");
    pre.style.whiteSpace = "pre-wrap"; pre.textContent = m.found;
    f.append(pre); d.append(f);
  }
  if (m.speed) {
    const s = document.createElement("div");
    s.className = "speed"; s.textContent = m.speed; d.append(s);
  }
  return d;
}
/* a reply being written, drawn again in place: its reasoning's box kept,
   only its text changed - a box replaced at every piece could not be
   clicked open (pressed on one, released on the next) */
function redraw(el, m) {
  const fresh = bubble(m);
  const old = el.querySelector(":scope > details.think");
  const now = fresh.querySelector(":scope > details.think");
  if (!old || !now) return el.replaceWith(fresh);
  old.querySelector(".thinktext").textContent = m.think;
  now.remove();
  while (old.nextSibling) old.nextSibling.remove();
  while (old.previousSibling) old.previousSibling.remove();
  el.className = fresh.className;
  el.append(...fresh.childNodes);
}
function drawLog() {
  if (typeof ctxRing === "function") ctxRing();
  const log = $("#log");
  log.innerHTML = "";
  const ms = cur ? cur.messages : [];
  ms.forEach((m, k) => {
    const d = bubble(m);
    /* each message's own actions, small, shown under the pointer: copy;
       a question changed and asked again; a reply read aloud, written
       again (the last one), its versions paged through */
    const act = (ic, title, run) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "msgico"; b.title = title;
      b.setAttribute("aria-label", title);
      b.innerHTML = icon(ic);
      b.onclick = run;
      return b;
    };
    const copy = (b) => {
      const done = () => { b.innerHTML = icon("check"); setTimeout(() => { b.innerHTML = icon("copy"); }, 1200); };
      /* the clipboard's API, or the old way where it is refused */
      const old = () => {
        const t = Object.assign(document.createElement("textarea"), {value: m.content || ""});
        t.style.position = "fixed"; t.style.opacity = "0"; document.body.append(t); t.select();
        const ok = document.execCommand("copy"); t.remove();
        if (ok) done();
      };
      if (navigator.clipboard) navigator.clipboard.writeText(m.content || "").then(done).catch(old);
      else old();
    };
    if (m.role === "user" && !busy) {
      const bar = document.createElement("div");
      bar.className = "msgacts";
      const c = act("copy", "Copy this message", () => copy(c));
      bar.append(c);
      if (!cur.archived) bar.append(act("pencil", "Change this message and ask again from it", () => editMessage(k)));
      d.append(bar);
    }
    if (m.role === "assistant" && m.content && !busy) {
      const bar = document.createElement("div");
      bar.className = "msgacts";
      const c = act("copy", "Copy this reply", () => copy(c));
      bar.append(c);
      if ("speechSynthesis" in window)
        bar.append(act("speakOn", "Read this reply aloud (again: stop)", () => {
          if (speechSynthesis.speaking) { speechSynthesis.cancel(); return; }
          speakReply(m, true);
        }));
      d.append(bar);
    }
    if (m.role === "assistant" && k === ms.length - 1 && !busy && !cur.archived) {
      const bar = d.querySelector(".msgacts") || d.appendChild(Object.assign(document.createElement("div"), {className: "msgacts"}));
      if (m.variants && m.variants.length > 1) {
        const prev = document.createElement("button"), next = document.createElement("button");
        prev.className = next.className = "msgact";
        prev.textContent = "‹"; next.textContent = "›";
        prev.title = "The earlier version of this reply"; next.title = "The later version of this reply";
        prev.disabled = m.vi === 0; next.disabled = m.vi === m.variants.length - 1;
        prev.onclick = () => showVariant(m, m.vi - 1);
        next.onclick = () => showVariant(m, m.vi + 1);
        const n = document.createElement("span");
        n.textContent = `${m.vi + 1}/${m.variants.length}`;
        bar.append(prev, n, next);
      }
      bar.append(act("restore", "Write this reply again; the earlier ones are kept", regenerate));
    }
    log.append(d);
  });
  if (followLog) log.scrollTop = log.scrollHeight;
}
/* while a reply is written, the chat follows it to its end only as long as
   the user stays there: scrolled up to read, it is left where it is */
let followLog = true;
$("#log").addEventListener("scroll", () => {
  const l = $("#log");
  followLog = l.scrollHeight - l.scrollTop - l.clientHeight < 48;
});
/* one button inside the field: an arrow to send, a square to stop while
   the reply comes (Esc stops it too) */
const ICON_SEND = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
const ICON_STOP = '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/></svg>';
let busy = null; /* the AbortController of the reply coming */
let rec = null; /* the voice being recorded (side.js, dictate) */
function sendButton() {
  const b = $("#send");
  /* the field empty: a microphone, to dictate */
  const mic = !busy && !$("#ask").value.trim();
  b.innerHTML = busy || rec ? ICON_STOP : mic ? icon("mic") : ICON_SEND;
  b.classList.toggle("rec", !!rec);
  b.title = busy ? "Stop (Esc)" : rec ? "Stop recording: what you said is written in the field"
    : mic ? "Dictate: speak, then click again" : "Send (Enter)";
  b.setAttribute("aria-label", busy ? "Stop" : mic ? "Dictate" : "Send");
  b.disabled = false;
}
function growAsk() {
  const t = $("#ask");
  t.style.height = "auto";
  t.style.height = t.scrollHeight + 2 + "px";
  sendButton();
}
function stopReply() { if (busy) busy.abort(); }
/* the tools of the workspace (janas-docs, janas-files), and those that
   only read */
let tools = {tools: [], read_only: []};
async function readTools() {
  try { tools = await ask("tools"); } catch (e) { tools = {tools: [], read_only: []}; }
  const has = (n) => tools.tools.some((t) => t.function && t.function.name === n);
  $("#usedocs").disabled = !has("docs_search");
  $("#usetools").disabled = !tools.tools.length;
  /* Python: where the cage has it; on Windows also where it can be
     installed on the user's word (pyinstall.c) */
  try { pyState = await ask("python"); } catch (e) { pyState = {needed: false}; }
  const canInstall = pyState.needed && !pyState.installed;
  $("#usecode").disabled = !has("python") && !canInstall;
  $("#usecode").parentElement.hidden = !has("python") && !canInstall;
  if (canInstall) $("#usecode").checked = false;
  /* the embedding model of janas-docs: words only without one */
  const em = tools.embedding_models || [];
  $("#wsemb").hidden = !ws.path;
  $("#embnote").textContent = em.length
    ? "The documents are searched by their words, and by their meaning with an embedding model, which finds what is said in other words. It indexes them again when changed."
    : "The documents are searched by their words alone: get qwen3-embedding-0.6b in Discover to search them by their meaning too.";
  $("#embsel").hidden = !em.length;
  $("#embsel").innerHTML = em.map((n) => `<option value="${esc(n)}"${n === tools.embedding ? " selected" : ""}>${esc(n)}</option>`).join("") +
    `<option value="none"${tools.embedding ? "" : " selected"}>words only</option>`;
}
const pref = (k, d) => { try { const v = localStorage.getItem(k); return v === null ? d : v === "1"; } catch (e) { return d; } };
$("#usedocs").checked = pref("brebus.usedocs", true);
$("#usetools").checked = pref("brebus.usetools", true);
$("#usecode").checked = pref("brebus.usecode", false);
$("#useart").checked = pref("brebus.useart", true);
for (const id of ["usedocs", "usetools", "usecode", "useart"])
  $("#" + id).onchange = () => { try { localStorage.setItem("brebus." + id, $("#" + id).checked ? "1" : "0"); } catch (e) {} };
/* Python on Windows: downloaded only once the user says so */
let pyState = {needed: false};
const mb = (n) => (n / 1e6).toFixed(0) + " MB";
$("#usecode").addEventListener("change", async (e) => {
  if (!e.target.checked || !pyState.needed || pyState.installed) return;
  e.target.checked = false;
  const d = await dialog({title: "Install Python for Run code?",
    text: `To run the model's code Brebus needs Python, which Janas does not carry. It downloads Python ${pyState.version} from python.org and numpy and matplotlib from PyPI (${mb(pyState.total)} in all), checks each file against the fingerprint written in Brebus, and installs them in your AppData folder, for Brebus alone. Nothing is downloaded unless you say so; you can remove it from Machine.`,
    buttons: [{label: "Install", value: "ok", primary: true}, {label: "Cancel", value: "no"}]});
  if (d.button !== "ok") return;
  await installPython();
  if (pyState.installed) { e.target.checked = true; try { localStorage.setItem("brebus.usecode", "1"); } catch (x) {} }
});
async function installPython() {
  await fetch("/brebus/python/install", {method: "POST", body: "{}"});
  for (;;) {
    pyState = await ask("python");
    $("#loaded").textContent = pyState.running
      ? `Installing Python: ${pyState.step} - ${mb(pyState.done)} of ${mb(pyState.total)}`
      : pyState.error ? "Python not installed: " + pyState.error : "Python installed.";
    if (!pyState.running) break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  if (pyState.error) say("Python was not installed: " + pyState.error);
  await readTools();
  status();
}

async function callTool(name, args) {
  const r = await fetch("/brebus/tools/call", {method: "POST",
    body: JSON.stringify({name, arguments: args})});
  const j = await r.json();
  return r.ok ? j : {result: j.error || "the tool did not answer", is_error: true};
}

/* attachments waiting for the next message */
let pending = [];
function drawChips() {
  $("#chips").innerHTML = pending.map((a, i) =>
    `<span>📎 ${esc(a.name)} <button data-i="${i}" title="Remove">×</button></span>`).join("");
  for (const b of document.querySelectorAll("#chips button"))
    b.onclick = () => { pending.splice(+b.dataset.i, 1); drawChips(); };
}
const kb = (n) => n < 1e6 ? (n / 1024).toFixed(0) + " KB" : (n / 1048576).toFixed(1) + " MB";
async function attach() {
  if (!ws.path) { wsShow(true); return; }
  let src;
  if (ws.window) {
    const d = await dialog({title: "Attach", text: "A file of this computer, by its full path, or a web address (downloaded into the workspace's sources/).",
      input: "", buttons: [{label: "OK", value: "ok", primary: true}, {label: "Browse…", value: "browse"}, {label: "Cancel", value: "no"}]});
    if (d.button === "no") return;
    src = d.button === "browse" ? await pick(true, "Attach a file") : d.value;
  } else
    src = await input("Attach", "", "A file of this computer, by its full path, or a web address (downloaded into the workspace's sources/).");
  src = (src || "").trim();
  if (!src) return;
  let mode = "copy";
  if (!/^https?:\/\//.test(src)) {
    const i = await (await fetch("/brebus/attach", {method: "POST",
      body: JSON.stringify({source: src, mode: "info"})})).json();
    if (i.error) { say(i.error); return; }
    if (!i.inside) {
      /* copied into the workspace, or read where it is: copying offered up
         to 100 MB, the choice remembered for files of that size if asked */
      const big = i.bytes > 100 * 1048576, key = "brebus.attach." + (big ? "big" : "small");
      let kept = null;
      try { kept = localStorage.getItem(key); } catch (e) {}
      if (kept === "copy" || kept === "place") mode = kept;
      else {
        const copy = {label: "Copy into the workspace", value: "copy", primary: !big};
        const place = {label: "Read it where it is", value: "place", primary: big};
        const d = await dialog({title: "Attach " + src.split("/").pop(),
          text: `${kb(i.bytes)}. A copy keeps the workspace whole and is read again as it is now; read where it is, it is added to the folders read only and follows its changes.`,
          buttons: big ? [place, copy, {label: "Cancel", value: "no"}] : [copy, place, {label: "Cancel", value: "no"}],
          remember: `Do the same for files ${big ? "over" : "up to"} 100 MB`});
        if (d.button === "no") return;
        mode = d.button;
        if (d.remember) try { localStorage.setItem(key, mode); } catch (e) {}
      }
    }
  }
  $("#attach").textContent = "…";
  const r = await fetch("/brebus/attach", {method: "POST", body: JSON.stringify({source: src, mode})});
  const j = await r.json();
  $("#attach").textContent = "+";
  if (!r.ok) { say(j.error || "not attached"); return; }
  pending.push(j); drawChips();
  if (mode === "place") readFolders();
}
$("#attach").onclick = () => attach().catch((e) => say(e.message));

const WHOLE = 24000; /* characters of a document put whole into a message */
/* what the model reads of a user's message: its text, the documents
   attached, the passages found */
function userContent(m) {
  let c = m.content;
  const sees = modelInfo[server.model]?.vision;
  for (const a of m.attached || []) {
    if (a.media === "image" && !sees)
      c += `\n\n[A picture was attached, ${a.name}; this model does not see pictures.]`;
    else if (a.media === "image")
      continue; /* sent as the picture itself (apiMessages) */
    else if (a.media === "audio" && a.transcript !== undefined)
      c += `\n\n<transcript of="${a.name}">\n${a.transcript}\n</transcript>`;
    else if (a.media === "audio")
      c += `\n\n[A recording was attached, ${a.name}, which could not be transcribed: ${a.unread || "no speech model"}.]`;
    else if (a.text !== undefined && a.text.length <= WHOLE && !a.cut)
      c += `\n\n<document name="${a.name}" path="${a.path}">\n${a.text}\n</document>`;
    else if (a.text !== undefined)
      c += `\n\n[Attached: ${a.path}, ${kb(a.bytes)}, too long to be put here whole: search it with docs_search, or read it with docs_read.]`;
    else
      c += `\n\n[Attached: ${a.path}, which could not be read: ${a.unread}.]`;
  }
  if (m.found)
    c += "\n\n[Passages found in the user's documents for this question, each with its document. Use them if they answer, and name the documents you use; say so if they do not answer.]\n" + m.found;
  return c;
}
