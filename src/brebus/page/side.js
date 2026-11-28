/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* side.js - Brebus's page (app.html): the side bar's chats and projects,
   the chat's menu, the model's menu, archived chats. build.sh puts it into
   the page where app.html includes it. */

/* the page's icons, drawn for Brebus: strokes on a 24 x 24 grid */
const GEAR = '<circle cx="12" cy="12" r="5.6"/><circle cx="12" cy="12" r="2.1"/>' +
  [0, 45, 90, 135, 180, 225, 270, 315].map((a) =>
    `<path d="M12 6.4V3.4" transform="rotate(${a} 12 12)" style="stroke-width:3.4;stroke-linecap:butt"/>`).join("");
const ICON = {
  chat: '<path d="M4 5h16v11H10l-5 4v-4H4z"/>',
  folder: '<path d="M3.5 6.5a1.5 1.5 0 0 1 1.5-1.5h4.5l2 2.5H19a1.5 1.5 0 0 1 1.5 1.5v8.5A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5z"/>',
  archive: '<rect x="3.5" y="4.5" width="17" height="4" rx="1"/><path d="M5 8.5V18a1.5 1.5 0 0 0 1.5 1.5h11A1.5 1.5 0 0 0 19 18V8.5M10 12.5h4"/>',
  pencil: '<path d="M4.5 19.5l1-4.5L15.5 5l3.5 3.5-10 10z"/><path d="M13.5 7l3.5 3.5"/>',
  sliders: '<path d="M5 7h9M18 7h1M5 17h3M12 17h7"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
  move: '<path d="M3.5 6.5a1.5 1.5 0 0 1 1.5-1.5h4.5l2 2.5H19a1.5 1.5 0 0 1 1.5 1.5v8.5A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5z"/><path d="M9 13h6M13 11l2 2-2 2"/>',
  export: '<path d="M12 4v11M8 8l4-4 4 4"/><path d="M5 13v5.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V13"/>',
  window: '<rect x="3.5" y="5" width="17" height="14" rx="1.5"/><path d="M3.5 9h17M13 13h4v3"/>',
  restore: '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3L4.5 9"/><path d="M4.5 4.5V9H9"/>',
  rewind: '<path d="M11 6.5l-6 5.5 6 5.5"/><path d="M19 6.5l-6 5.5 6 5.5"/>',
  trash: '<path d="M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 12.5h9l1-12.5"/>',
  model: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
  unload: '<path d="M12 4v9M8 9l4 4 4-4"/><path d="M5 19.5h14"/>',
  compass: '<circle cx="12" cy="12" r="9"/><path d="M15.5 8.5l-2 5-5 2 2-5z"/>',
  gear: GEAR,
  mic: '<rect x="9" y="3.5" width="6" height="11" rx="3"/><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v2.5"/>',
  speakOff: '<path d="M4.5 9.5h3l4.5-4v13l-4.5-4h-3z"/><path d="M15.5 9.5l5 5M20.5 9.5l-5 5"/>',
  /* a brain: two halves, their folds */
  brain: '<path d="M12 5.5a3 3 0 0 0-5.6-1.3A3 3 0 0 0 4 8.5a3 3 0 0 0 .6 4.9A3.2 3.2 0 0 0 7 18.5a3 3 0 0 0 5 1z"/><path d="M12 5.5a3 3 0 0 1 5.6-1.3A3 3 0 0 1 20 8.5a3 3 0 0 1-.6 4.9 3.2 3.2 0 0 1-2.4 5.1 3 3 0 0 1-5 1z"/><path d="M8 9.5c1.2 0 2 .8 2 2M16 9.5c-1.2 0-2 .8-2 2M7.5 15c1 0 1.8-.4 2.3-1M16.5 15c-1 0-1.8-.4-2.3-1"/>',
  speakOn: '<path d="M4.5 9.5h3l4.5-4v13l-4.5-4h-3z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/>',
  /* two arrows meeting at a line: summarized */
  fold: '<path d="M12 3v6M9 6l3 3 3-3M12 21v-6M9 18l3-3 3 3M4 12h16"/>',
  /* a loudspeaker at full volume: three waves */
  speakLoud: '<path d="M3.5 9.5h3l4.5-4v13l-4.5-4h-3z"/><path d="M14 10a2.8 2.8 0 0 1 0 4M16.5 7.5a6.3 6.3 0 0 1 0 9M19 5a9.8 9.8 0 0 1 0 14"/>',
  file: '<path d="M6.5 3.5h7l4 4v13h-11z"/><path d="M13.5 3.5v4h4"/>',
  art: '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 9h16M9 9v11"/>',
  branch: '<circle cx="6" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="7" r="2"/><path d="M6 7v10"/><path d="M18 9c0 5-12 3-12 8"/>',
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="1.5"/><path d="M15.5 8.5V5.5A1.5 1.5 0 0 0 14 4H5.5A1.5 1.5 0 0 0 4 5.5V14a1.5 1.5 0 0 0 1.5 1.5h3"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  more: '<circle cx="6" cy="12" r="1.2"/><circle cx="12" cy="12" r="1.2"/><circle cx="18" cy="12" r="1.2"/>',
  down: '<path d="M6 9l6 6 6-6"/>',
  pulse: '<path d="M3 12h4l2.5-6 4 12 2.5-6H21"/>',
  zin: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5M8 10.5h5M10.5 8v5"/>',
  zout: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5M8 10.5h5"/>',
  zfit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/><rect x="8" y="8" width="8" height="8" rx="1"/>',
};
const icon = (n) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICON[n] || ""}</svg>`;

/* where the pointer last went down: a menu's place when its button has
   been drawn anew meanwhile */
const lastPointer = {x: 0, y: 0};
addEventListener("pointerdown", (e) => { lastPointer.x = e.clientX; lastPointer.y = e.clientY; }, true);

/* a small menu by an element: items {label, run, note, on, danger}, or
   "-" for a line between; closed by a click elsewhere or Escape */
function popMenu(anchor, items, up) {
  document.querySelectorAll(".pop").forEach((x) => x.remove());
  const m = document.createElement("div");
  m.className = "pop";
  m.setAttribute("role", "menu");
  for (const it of items) {
    if (it === "-") { m.append(Object.assign(document.createElement("hr"))); continue; }
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "menuitem");
    b.className = (it.on ? "on " : "") + (it.danger ? "danger" : "");
    b.innerHTML = `${it.icon ? icon(it.icon) : ""}<span class="t"><span>${esc(it.label)}</span>${it.note ? `<small>${esc(it.note)}</small>` : ""}</span>`;
    b.title = it.note ? `${it.label}: ${it.note}` : it.label;
    b.onclick = (e) => { e.stopPropagation(); m.remove(); it.run(); };
    m.append(b);
  }
  document.body.append(m);
  /* by the anchor, or where the pointer was when it has been drawn anew */
  let r = anchor.isConnected ? anchor.getBoundingClientRect() : null;
  if (!r || (!r.width && !r.height)) r = {left: lastPointer.x, right: lastPointer.x, top: lastPointer.y, bottom: lastPointer.y};
  const w = m.offsetWidth, h = m.offsetHeight;
  m.style.left = Math.max(8, Math.min(r.left, innerWidth - w - 8)) + "px";
  m.style.top = (up ? Math.max(8, r.top - h - 4) : Math.min(r.bottom + 4, innerHeight - h - 8)) + "px";
  const close = (e) => {
    if (e.type === "keydown" && e.key !== "Escape") return;
    if (e.type === "pointerdown" && m.contains(e.target)) return;
    m.remove();
    removeEventListener("pointerdown", close, true); removeEventListener("keydown", close, true);
  };
  setTimeout(() => { addEventListener("pointerdown", close, true); addEventListener("keydown", close, true); });
  m.querySelector("button")?.focus();
}

/* ---- the side bar: the free chats, the projects with theirs, the
   archived ones; drawn from /brebus/projects ---- */
let tree = {free: "", free_chats: {chats: []}, projects: []};
let shut = {};
try { shut = JSON.parse(localStorage.getItem("brebus.shut") || "{}"); } catch (e) {}
const keepShut = () => { try { localStorage.setItem("brebus.shut", JSON.stringify(shut)); } catch (e) {} };

async function refreshTree() {
  try { tree = await ask("projects"); } catch (e) { return; }
  drawTree();
}
/* one chat's row: its title, the chat open marked */
function chatRow(c, where) {
  const r = document.createElement("div");
  r.className = "row";
  r.dataset.id = c.id;
  r.dataset.ws = where;
  const d = document.createElement("button");
  d.type = "button";
  d.className = "c";
  d.textContent = chatTitle(c.title);
  d.title = d.textContent + (c.model ? " · " + c.model : "");
  d.onclick = () => goChat(where, c.id);
  const m = document.createElement("button");
  m.type = "button";
  m.className = "rmore";
  m.title = T("This chat: rename, move, export, archive, delete");
  m.setAttribute("aria-label", T("Menu of this chat"));
  m.innerHTML = icon("more");
  m.onclick = (e) => {
    e.stopPropagation();
    if (cur && cur.id === c.id && ws.path === where) chatMenu(m);
    else rowMenu(c, where, m);
  };
  r.append(d, m);
  return r;
}
function drawTree() {
  const free = (tree.free_chats.chats || []);
  const fl = $("#freelist");
  fl.innerHTML = "";
  for (const c of free.filter((c) => !c.archived)) fl.append(chatRow(c, tree.free));
  if (!fl.children.length) fl.innerHTML = `<div class="empty note">${T("None yet.")}</div>`;
  const pl = $("#projlist");
  pl.innerHTML = "";
  const arch = free.filter((c) => c.archived).map((c) => [c, tree.free]);
  for (const p of tree.projects) {
    const g = document.createElement("div");
    g.className = "proj" + (shut["p:" + p.path] ? " shut" : "") + (p.path === ws.path ? " here" : "");
    const h = document.createElement("div");
    h.className = "phead";
    h.title = p.path;
    h.innerHTML = `<span class="arrow"></span>${icon("folder")}<span class="pname">${esc(p.name)}</span>
      <button class="gadd" type="button" title="${T("A new chat in {p}", {p: esc(p.name)})}" aria-label="${T("New chat in this project")}">+</button>
      <button class="gadd" type="button" title="${T("{p}: its documents, services and folders", {p: esc(p.name)})}" aria-label="${T("Project settings")}">${icon("more")}</button>`;
    h.onclick = (e) => {
      if (e.target.closest(".gadd")) return;
      shut["p:" + p.path] = !shut["p:" + p.path]; keepShut(); drawTree();
    };
    const [add, more] = h.querySelectorAll(".gadd");
    add.onclick = () => goChat(p.path, "new");
    more.onclick = async () => {
      if (p.path !== ws.path) await wsOpen(p.path, false);
      psShow("gen");
    };
    const list = document.createElement("div");
    list.className = "plist";
    const chats = p.chats.chats || [];
    for (const c of chats.filter((c) => !c.archived)) list.append(chatRow(c, p.path));
    if (!list.children.length) list.innerHTML = `<div class="empty note">${T("No chats yet.")}</div>`;
    arch.push(...chats.filter((c) => c.archived).map((c) => [c, p.path]));
    g.append(h, list);
    pl.append(g);
  }
  if (!tree.projects.length) pl.innerHTML = `<div class="empty note">${T("A project keeps a folder of yours with its documents, files and chats: + makes one.")}</div>`;
  const al = $("#archlist");
  al.innerHTML = "";
  for (const [c, where] of arch) al.append(chatRow(c, where));
  $("#tree .grp[data-g=archived]").hidden = !arch.length;
  for (const g of document.querySelectorAll("#tree .grp"))
    g.classList.toggle("shut", g.dataset.g === "archived" ? shut.archived !== false : !!shut[g.dataset.g]);
  markTree();
}
/* the chat open, marked where it is */
function markTree() {
  for (const d of document.querySelectorAll("#tree .row"))
    d.classList.toggle("on", !!cur && d.dataset.id === cur.id && d.dataset.ws === ws.path);
  for (const d of document.querySelectorAll("#tree .proj"))
    d.classList.toggle("here", d.querySelector(".phead").title === ws.path);
}
for (const h of document.querySelectorAll("#tree .grp > .ghead"))
  h.onclick = (e) => {
    if (e.target.closest(".gadd")) return;
    const g = h.parentElement.dataset.g;
    shut[g] = !h.parentElement.classList.contains("shut");
    keepShut(); drawTree();
  };
$("#addchat").onclick = () => newFree();
/* a new project: a folder chosen, or made in Documents/Brebus */
$("#addproj").onclick = () => wsShow(true);

/* a chat of a project (or of the free chats) opened; "new": a new one */
async function goChat(where, id) {
  show("chat");
  if (where !== ws.path) { await wsOpen(where, where === tree.free ? "free" : false, id); return; }
  if (id === "new") newConv(); else await openConv(id);
}

/* ---- the chat's own bar: its title, a menu of what can be done ---- */
function chatBar() {
  const t = cur ? chatTitle(cur.title) : "";
  $("#ctitle").firstChild.textContent = t;
  $("#ctitle").hidden = !cur;
  $("#cwhere").textContent = ws.free ? "" : ws.name ? T("in {name}", {name: ws.name}) : "";
  $("#cwhere").title = ws.free ? "" : ws.path;
  const arch = !!(cur && cur.archived);
  $("#archbar").hidden = !arch;
  $("#form").hidden = $("#under").hidden = arch;
}
/* the chat open's menu, by the element at */
function chatMenu(at) {
  if (!cur) return;
  const arch = !!cur.archived;
  popMenu(at, [
    {label: T("Rename"), icon: "pencil", run: renameConv},
    {label: T("Move to…"), icon: "move", note: ws.free ? T("a project") : T("another project, or the free chats"), run: () => moveConv(at)},
    {label: T("Duplicate"), icon: "copy", note: T("a copy to take another way"), run: () => branchConv(cur)},
    {label: T("Compact now"), icon: "fold", note: T("the earlier messages summarized, to make room"), run: compactNow},
    {label: T("Export"), icon: "export", note: T("Markdown, text or PDF"), run: exportConv},
    {label: T("In a new window"), icon: "window", run: () => window.open("/#chat/" + cur.id, "_blank")},
    "-",
    {label: arch ? T("Restore") : T("Archive"), icon: arch ? "restore" : "archive", note: arch ? T("back among the chats") : T("kept to read, no longer continued"), run: () => archiveConv(!arch)},
    {label: T("Delete"), icon: "trash", danger: true, run: deleteConv},
  ]);
}
$("#ctitle").onclick = () => chatMenu($("#ctitle"));
/* the menu of a chat of the side bar that is not the one open: what is
   done to it, done where it is, without opening it */
function rowMenu(c, where, at) {
  const url = (id) => `/brebus/chats/${id}?ws=${encodeURIComponent(where)}`;
  const load = async () => { const r = await fetch(url(c.id)); return r.ok ? r.json() : null; };
  const save = async (x) => { x.updated = Date.now();
    await fetch(url(c.id), {method: "POST", body: JSON.stringify(x)}); refreshTree(); };
  popMenu(at, [
    {label: T("Open"), icon: "chat", run: () => goChat(where, c.id)},
    {label: T("Rename"), icon: "pencil", run: async () => {
      const x = await load(); if (!x) return;
      const t = await input(T("A title for this chat"), x.title || "");
      if (t === null) return;
      x.title = t.trim(); await save(x);
    }},
    {label: T("Move to…"), icon: "move", note: T("another project, or the free chats"), run: () => {
      const places = [...(where === tree.free ? [] : [{label: T("The free chats"), path: ""}]),
        ...tree.projects.filter((p) => p.path !== where).map((p) => ({label: p.name, path: p.path}))];
      if (!places.length) { say(T("There is no other project yet: + beside Projects makes one.")); return; }
      popMenu(at, places.map((p) => ({label: p.label, icon: p.path ? "folder" : "chat", note: p.path, run: async () => {
        const r = await fetch("/brebus/chats/move", {method: "POST", body: JSON.stringify({id: c.id, from: where, to: p.path})});
        if (!r.ok) say((await r.json()).error || T("not moved"));
        refreshTree();
      }})));
    }},
    {label: T("Duplicate"), icon: "copy", note: T("a copy to take another way"), run: async () => { const x = await load(); if (x) branchConv(x, undefined, where); }},
    {label: T("Export"), icon: "export", note: T("Markdown or text"), run: async () => { const x = await load(); if (x) exportConv(x); }},
    "-",
    {label: c.archived ? T("Restore") : T("Archive"), icon: c.archived ? "restore" : "archive",
     note: c.archived ? T("back among the chats") : T("kept to read, no longer continued"),
     run: async () => { const x = await load(); if (!x) return; x.archived = !c.archived; await save(x); }},
    {label: T("Delete"), icon: "trash", danger: true, run: async () => {
      if (!await yes(T('"{t}" and its messages are deleted.', {t: chatTitle(c.title)}), T("Delete this chat?"), T("Delete"))) return;
      await fetch(url(c.id), {method: "DELETE"}); refreshTree();
    }},
  ]);
}
async function archiveConv(on) {
  cur.archived = on;
  await saveConv(cur);
  const e = convs.find((c) => c.id === cur.id); if (e) e.archived = on;
  chatBar(); drawLog(); refreshTree();
}
$("#unarch").onclick = () => archiveConv(false);
/* the chat moved to another project, or to the free chats */
async function moveConv(at) {
  const places = [...(ws.free ? [] : [{label: T("The free chats"), path: ""}]),
    ...tree.projects.filter((p) => p.path !== ws.path).map((p) => ({label: p.name, path: p.path}))];
  if (!places.length) { say(T("There is no other project yet: + beside Projects makes one.")); return; }
  popMenu(at || $("#ctitle"), places.map((p) => ({label: p.label, icon: p.path ? "folder" : "chat", note: p.path, run: async () => {
    /* saved already: saving it again here would hold the file open
       while it is moved */
    const r = await fetch("/brebus/chats/move", {method: "POST", body: JSON.stringify({id: cur.id, to: p.path})});
    const j = await r.json();
    if (!r.ok) { say(j.error || T("not moved")); return; }
    await goChat(p.path || tree.free, cur.id);
  }})));
}

/* ---- the model under the chat: chosen from those installed ---- */
$("#modelbtn").onclick = () => {
  const names = Object.keys(modelInfo).filter((n) => !n.startsWith("__")).sort();
  const items = names.map((n) => ({label: n, icon: "model", on: server.running && server.model === n,
    note: `${gib(modelInfo[n].bytes)}${modelInfo[n].vision ? " · " + T("sees pictures") : ""}${isLoaded(n) && server.model !== n ? " · " + T("loaded") : ""}`,
    run: () => useModel(n)}));
  /* no Unload here: the chat needs a model (My models and Server have it) */
  items.push("-", {label: T("Other models…"), icon: "compass", note: T("Discover"), run: () => show("discover")});
  popMenu($("#modelbtn"), items, true);
};
function useModel(name) {
  if (server.running && server.model === name) return;
  return whileLoading(async () => {
    if (!isLoaded(name)) {
      $("#loaded").textContent = T("Loading {m}…", {m: name});
      delete ctxOf[name]; /* loaded anew: its context asked again */
    }
    sendButton(); /* the field says so, and waits */
    const r = await fetch("/brebus/server/start", {method: "POST",
      body: JSON.stringify({model: name, ...(settings[name] || {})})});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) say(j.error || T("{m} did not start", {m: name}));
    else remember(name);
    await status();
  });
}
/* the model's settings, in a dialog of their own */
$("#modelset").onclick = async () => {
  const name = server.running ? server.model : nextModel();
  if (!name || !modelInfo[name]) return;
  let changed = false;
  await dialog({title: T("Settings of {m}", {m: name}), html: `<p class="note">${T("How this model runs: kept for it, used each time it is loaded.")}</p>${setForm(modelInfo[name])}`,
    ready: (box) => { const f = box.querySelector("form"); f.onchange = () => { changed = true; saveSettings(f); }; },
    buttons: [{label: T("Close"), value: "ok", primary: true}]});
  if (changed && server.running && server.model === name &&
      await yes(T("Load it again now, to use them?"), T("Settings saved"), T("Load again"))) {
    await fetch("/brebus/server/stop", {method: "POST", body: JSON.stringify({model: name})});
    await useModel(name);
  }
};

/* Ctrl+K: the search of the chats; Ctrl+Shift+O: a new chat */
$("#findbtn").onclick = () => findChats();
/* the search of every chat, with filters: words in titles, in the user's
   messages or in all; a project or the free chats; a time; a model;
   archived ones too */
async function findChats() {
  const projects = tree.projects.map((p) => `<option value="${esc(p.path)}">${esc(p.name)}</option>`).join("");
  const models = [...new Set([...tree.free_chats.chats, ...tree.projects.flatMap((p) => p.chats.chats || [])]
    .map((c) => c.model).filter(Boolean))].sort().map((m) => `<option>${esc(m)}</option>`).join("");
  let timer = 0, hits = [];
  await dialog({title: T("Find a chat"), wide: true, html: `<div class="srvset find">
      <input class="fq" type="search" placeholder="${T("Words to find (empty: every chat the filters keep)")}" spellcheck="false">
      <div class="row">
        <label>${T("In")} <select class="fin"><option value="all">${T("titles and messages")}</option><option value="mine">${T("my messages only")}</option><option value="titles">${T("titles only")}</option></select></label>
        <label>${T("Where")} <select class="fwhere"><option value="">${T("everywhere")}</option><option value="free">${T("the free chats")}</option>${projects}</select></label>
      </div><div class="row">
        <label>${T("When")} <select class="fsince"><option value="0">${T("any time")}</option><option value="1">${T("today")}</option><option value="7">${T("the last week")}</option><option value="30">${T("the last month")}</option><option value="365">${T("the last year")}</option></select></label>
        <label>${T("Model")} <select class="fmodel"><option value="">${T("any")}</option>${models}</select></label>
        <label class="tog farchl"><input type="checkbox" class="farch" checked> ${T("archived too")}</label>
      </div>
      <div class="fcount note"></div>
      <div class="fres"></div></div>`,
    ready: (box) => {
      const run = async () => {
        const v = (c) => box.querySelector(c);
        const days = +v(".fsince").value;
        const p = new URLSearchParams({q: v(".fq").value.trim(), in: v(".fin").value, where: v(".fwhere").value,
          model: v(".fmodel").value, archived: v(".farch").checked ? "1" : "0",
          since: days ? String(Date.now() - days * 86400000) : "0"});
        const j = await ask("chats/search?" + p);
        const res = v(".fres");
        hits = j.chats;
        res.innerHTML = j.chats.length ? "" : `<p class="note empty">${T("Nothing found.")}</p>`;
        j.chats.forEach((h, i) => {
          const b = document.createElement("button");
          b.type = "button"; b.className = "hit"; b.dataset.k = i;
          b.innerHTML = `<span class="l1"><b>${esc(chatTitle(h.title))}</b><span class="when">${dateOnly(h.updated)}</span></span>` +
            `<span class="l2">${h.free ? T("Chats") : T("Project {p}", {p: esc(h.workspace_name)})}${h.model ? " · " + esc(h.model) : ""}${h.archived ? " · " + T("archived") : ""}</span>` +
            (h.snippet ? `<span class="l3">${esc(h.snippet)}</span>` : "");
          res.append(b);
        });
        box.querySelector(".fcount").textContent = !j.chats.length ? ""
          : j.more ? T("{n}+ chats found", {n: num(j.chats.length)}) : Tn(j.chats.length, "{n} chat found", "{n} chats found");
      };
      /* one listener for the list: a result opens its chat */
      box.querySelector(".fres").addEventListener("click", (e) => {
        const b = e.target.closest(".hit");
        if (!b) return;
        const h = hits[+b.dataset.k];
        box.querySelector("button[data-i]")?.click();
        if (h) goChat(h.workspace, h.id);
      });
      box.querySelector(".fq").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 250); });
      for (const s of box.querySelectorAll("select, input[type=checkbox]")) s.onchange = run;
      run();
      setTimeout(() => box.querySelector(".fq").focus());
    },
    buttons: [{label: T("Close"), value: "ok", primary: true}]});
}
document.addEventListener("keydown", (e) => {
  if (!(e.ctrlKey || e.metaKey) || e.altKey || dialogOpen()) return;
  const k = e.key.toLowerCase();
  if (k === "k" && !e.shiftKey) { e.preventDefault(); $("#findbtn").click(); }
  if (k === "o" && e.shiftKey) { e.preventDefault(); newFree(); }
});

/* icons where the page is drawn: the groups', the gear's */
{
  const at = {chats: "chat", projects: "folder", archived: "archive"};
  for (const g of document.querySelectorAll("#tree .grp"))
    g.querySelector(".ghead .arrow").insertAdjacentHTML("afterend", icon(at[g.dataset.g]));
  $("#modelset").innerHTML = icon("gear");
  document.querySelector("nav button[data-s=prefs] svg").innerHTML = GEAR;
}
/* the side bar's width, dragged by its right edge and kept */
{
  const side = $("#side");
  try { const w = +localStorage.getItem("brebus.sidew"); if (w >= 200) document.documentElement.style.setProperty("--sidew", w + "px"); } catch (e) {}
  const g = document.createElement("div");
  g.id = "sidegrip"; g.title = T("Drag to widen");
  side.append(g);
  g.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    document.body.classList.add("drag");
    const move = (ev) => document.documentElement.style.setProperty("--sidew",
      Math.max(200, Math.min(ev.clientX, innerWidth * 0.45)) + "px");
    const up = () => {
      document.body.classList.remove("drag");
      removeEventListener("pointermove", move); removeEventListener("pointerup", up);
      try { localStorage.setItem("brebus.sidew", parseInt(getComputedStyle(side).width)); } catch (e) {}
    };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  });
}

/* ---- the chat's own menu, at the right of its bar ---- */
$("#chatmore").innerHTML = icon("more");
$("#chatmore").onclick = () => {
  if (!cur) return;
  const items = [];
  const arts = [...artTitles()];
  /* the pictures the replies named from the project, as artifacts too */
  const pics = cur.messages.filter((m) => m.role === "assistant").flatMap((m) => picRefs(m.content));
  const seen = new Set();
  for (const p of pics) if (!seen.has(p.path)) {
    seen.add(p.path);
    arts.push({pic: p});
  }
  if (arts.length) {
    items.push(...arts.map((t) => t.pic
      ? {label: t.pic.title, icon: "art", note: T("picture in the project"), run: () => artShowFile(t.pic.path, t.pic.title)}
      : {label: t, icon: "art", note: T("artifact"), run: () => artShow(t)}));
  } else items.push({label: T("No artifacts yet"), icon: "art", note: T("pages, drawings, programs the model makes"), run: () => {}});
  items.push("-");
  const files = cur.messages.flatMap((m) => m.attached || []);
  if (files.length)
    items.push(...files.map((f) => ({label: f.name, icon: "file", note: f.path,
      run: () => fetch("/brebus/open", {method: "POST", body: JSON.stringify({path: f.path})})})));
  else items.push({label: T("No files attached"), icon: "file", note: T("+ under the field attaches one"), run: () => {}});
  items.push("-", {label: T("Instructions and answer"), icon: "sliders", note: T("the model's role, how free it is, the answer's form"), run: chatSettings});
  popMenu($("#chatmore"), items);
};

/* ---- the replies read aloud, by the system's voices for now (Janas's own
   synthesis will take their place) ---- */
let speakOn = false;
try { speakOn = localStorage.getItem("brebus.speak") === "1"; } catch (e) {}
function speakButton() {
  const b = $("#speak");
  b.innerHTML = icon(speakOn ? "speakOn" : "speakOff");
  b.setAttribute("aria-pressed", speakOn ? "true" : "false");
  b.title = speakOn ? T("The replies are read aloud: click to stop") : T("Read the replies aloud");
  b.hidden = !("speechSynthesis" in window);
}
$("#speak").onclick = () => {
  speakOn = !speakOn;
  try { localStorage.setItem("brebus.speak", speakOn ? "1" : ""); } catch (e) {}
  if (!speakOn) speechSynthesis.cancel();
  speakButton();
};
speakButton();

/* ---- reasoning before the answer, for this chat: beside the voice, for
   a model that can be told (its capabilities, /api/show); a chat starts
   as the model's setting has it (off unless yes), and changes it at any
   message, at once: it goes with each request ---- */
const reasonsOf = {};
function thinkNow() {
  if (cur && cur.think !== undefined) return cur.think;
  const s = settings[server.model];
  return !!(s && s.think === true);
}
function thinkButton() {
  const b = $("#thinkbtn"), can = server.running && reasonsOf[server.model];
  put(b, "hidden", !can);
  if (!can) return;
  const on = thinkNow();
  put(b, "innerHTML", icon("brain"));
  b.setAttribute("aria-pressed", on ? "true" : "false");
  put(b, "title", on ? T("The model reasons before it answers, in this chat: slower, better on hard questions. Click to stop")
                     : T("Let the model reason before it answers, in this chat: slower, better on hard questions"));
}
async function readReasons() {
  const name = server.running && server.model;
  if (name && !(name in reasonsOf))
    try {
      const r = await fetch("/api/show", {method: "POST", headers: {"Content-Type": "application/json"},
                                          body: JSON.stringify({model: name})});
      const j = await r.json();
      if (r.ok) reasonsOf[name] = (j.capabilities || []).includes("thinking");
    } catch (e) {}
  thinkButton();
}
$("#thinkbtn").onclick = async () => {
  await booted; /* the start's own chat, not one it replaces */
  if (!cur) newConv();
  cur.think = !thinkNow();
  thinkButton();
  if (cur.messages.length) await saveConv(cur); /* a new chat: with its first message */
};
/* a reply's words, without its marks, its code or its artifacts */
function speakReply(m, asked) {
  if (!(speakOn || asked) || !m || !m.content || !("speechSynthesis" in window)) return;
  const text = artParts(m.content).filter((p) => p.text !== undefined).map((p) => p.text).join(" ")
    .replace(/```[\s\S]*?```/g, " ").replace(/`[^`]*`/g, " ").replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/[#*_>|~-]{1,}/g, " ").replace(/\s+/g, " ").trim();
  if (!text) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = document.documentElement.lang || navigator.language || "en";
  let want = "";
  try { want = localStorage.getItem("brebus.voice") || ""; } catch (e) {}
  const v = want && speechSynthesis.getVoices().find((x) => x.name === want);
  if (v) { u.voice = v; u.lang = v.lang; }
  speechSynthesis.speak(u);
}

/* ---- how much of the model's memory the chat fills: its last reply's
   tokens, of the context the model was loaded with ---- */
function ctxRing() {
  const used = ctxUsed(), size = ctxSize();
  const f = size ? Math.min(1, used / size) : 0;
  const r = $("#ctxring");
  r.querySelector(".fg").style.strokeDashoffset = String(53.4 * (1 - f));
  r.classList.toggle("warm", f >= 0.7 && f < 0.9);
  r.classList.toggle("full", f >= 0.9);
  /* the details in a popup of their own (below); the label for readers */
  r.ariaLabel = !used ? T("The model's memory of this chat: nothing used yet")
    : (size ? T("The model's memory of this chat: {used} of {size} tokens ({p}%)", {used: num(used), size: num(size), p: num(Math.round(f * 100))})
            : T("The model's memory of this chat: {used} of {size} tokens", {used: num(used), size: "?"})) +
      (f >= COMPACT_AT ? T(". Nearly full: the earlier messages will be summarized before the next one is sent.") : "");
}

/* ---- dictation: the field empty, the send button is a microphone; the
   voice is written by the server's Whisper ---- */
async function dictate() {
  if (rec) { rec.stop(); return; }
  /* no speech model yet: offered, with what it is and where it comes from */
  if (!speechModels.length) {
    const d = await dialog({title: T("Dictation needs a speech model"),
      text: T("Your voice is written by OpenAI's Whisper, run on this computer: nothing you say leaves it. whisper-small (0.49 GB) is downloaded from Hugging Face (prabanta-dev/whisper-small-JNS) and checked against its published fingerprint. Settings, Voice, has a larger, more accurate one."),
      buttons: [{label: T("Get it"), value: "get", primary: true}, {label: T("Not now"), value: "no"}]});
    if (d.button === "get") { show("discover"); getModel("whisper-small"); }
    return;
  }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({audio: true}); }
  catch (e) { say(T("The microphone is not available: {e}", {e: e.message}), T("Dictation")); return; }
  /* the sound itself, through Web Audio, sent as a WAV file: not
     MediaRecorder, whose WebM or Ogg needs the system's codecs (WebKitGTK
     records through GStreamer, and without its plugins the recording comes
     out empty, issue #81), where a WAV is read by Janas itself */
  let ctx;
  try { ctx = new (window.AudioContext || window.webkitAudioContext)(); }
  catch (e) { stream.getTracks().forEach((t) => t.stop()); say(T("The microphone is not available: {e}", {e: e.message}), T("Dictation")); return; }
  if (ctx.state === "suspended") await ctx.resume().catch(() => {});
  const src = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(4096, 1, 1);
  const parts = [];
  let samples = 0;
  const most = ctx.sampleRate * 600; /* ten minutes at most */
  node.onaudioprocess = (e) => {
    if (samples >= most) return;
    const x = e.inputBuffer.getChannelData(0);
    parts.push(new Float32Array(x));
    samples += x.length;
  };
  src.connect(node);
  node.connect(ctx.destination); /* a processor runs only when connected */
  rec = {stop: async () => {
    node.onaudioprocess = null;
    try { src.disconnect(); node.disconnect(); } catch (e) {}
    stream.getTracks().forEach((t) => t.stop());
    const rate = ctx.sampleRate;
    ctx.close().catch(() => {});
    rec = null; sendButton();
    /* silence all through is not nothing to say: it is said */
    let peak = 0;
    for (const p of parts) for (let i = 0; i < p.length; i += 16) peak = Math.max(peak, Math.abs(p[i]));
    if (!samples || peak < 1e-4) {
      say(T("No sound came from the microphone: check that the right one is chosen in the system's sound settings, and that it is not muted."), T("Dictation"));
      return;
    }
    const blob = wavBlob(parts, samples, rate);
    $("#loaded").textContent = T("writing what you said…");
    try {
      if (!await ensureModel()) return;
      const fd = new FormData();
      fd.append("file", blob, "voice.wav");
      fd.append("model", "whisper");
      const r = await fetch("/v1/audio/transcriptions", {method: "POST", body: fd});
      const j = await r.json().catch(() => ({}));
      if (r.ok && typeof j.text === "string") {
        const t = $("#ask");
        t.value = (t.value ? t.value + " " : "") + j.text.trim();
        growAsk(); t.focus();
        if (!j.text.trim()) say(T("Whisper heard no words in it."), T("Dictation"));
      } else {
        const why = (j.error && (j.error.message || j.error)) || "";
        say(/speech model/i.test(why)
          ? T("Dictation needs a speech model (Whisper): get one in Settings, Voice. It is used from the next time the chat's model is loaded.")
          : why ? T("The voice could not be written: {why}", {why}) : T("The voice could not be written."), T("Dictation"));
      }
    } finally { status(); }
  }};
  sendButton();
}

/* mono samples as a 16-bit PCM WAV file */
function wavBlob(parts, n, rate) {
  const b = new ArrayBuffer(44 + 2 * n), v = new DataView(b);
  const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + 2 * n, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, 2 * n, true);
  let o = 44;
  for (const p of parts) for (let i = 0; i < p.length && o < 44 + 2 * n; i++, o += 2) {
    const x = Math.max(-1, Math.min(1, p[i]));
    v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return new Blob([b], {type: "audio/wav"});
}

/* the carets that say "a menu here"; the engine's monitor by the speaker */
for (const c of document.querySelectorAll(".caret")) c.innerHTML = icon("down");
$("#monbtn").innerHTML = icon("pulse");
$("#monbtn").onclick = () => window.open("/monitor", "_blank", "noopener");

/* ---- the context's details, under the pointer: shown while the pointer
   is on the ring or on them ---- */
{
  let pop = null, leave = 0;
  const close = () => { if (pop) { pop.remove(); pop = null; } };
  const later = () => { clearTimeout(leave); leave = setTimeout(close, 250); };
  const ring = $("#ctxring");
  ring.removeAttribute("title");
  ring.addEventListener("mouseenter", () => {
    clearTimeout(leave);
    if (pop) return;
    const used = ctxUsed(), size = ctxSize();
    const name = server.running ? server.model : nextModel();
    const f = size ? Math.min(1, used / size) : 0;
    const words = (n) => T("about {n} words", {n: num(Math.round(n * 0.75))});
    pop = document.createElement("div");
    pop.className = "ctxpop";
    pop.innerHTML = `<b>${T("The model's memory of this chat")}</b>
      <div class="bar${f >= 0.9 ? " full" : f >= 0.7 ? " warm" : ""}"><i style="width:${(f * 100).toFixed(1)}%"></i></div>
      <dl><dt>${T("Used")}</dt><dd>${size ? T("{n} tokens ({p}%)", {n: num(used), p: num(Math.round(f * 100))}) : T("{n} tokens", {n: num(used)})}</dd>
        <dt>${T("Room")}</dt><dd>${size ? T("{n} tokens", {n: num(size)}) : T("not known yet")}</dd>
        ${size ? `<dt>${T("Left")}</dt><dd>${T("{n} tokens, {w}", {n: num(Math.max(0, size - used)), w: words(Math.max(0, size - used))})}</dd>` : ""}
        <dt>${T("Model")}</dt><dd>${esc(name || T("none"))}</dd></dl>
      <p>${T("Everything the model reads counts: your messages, its replies, the documents found, the tools' results.")} ${
        f >= COMPACT_AT ? T("It is nearly full: before your next message the earlier ones are summarized, and the model reads the summary in their place.")
        : f >= 0.7 ? T("At 80% the earlier messages are summarized, to make room.")
        : T("When it fills up, the earlier messages are summarized to make room (Compact now, in the chat's menu, does it at once).")} ${T("The room is set in the model's settings (the gear).")}</p>`;
    document.body.append(pop);
    const r = ring.getBoundingClientRect();
    pop.style.left = Math.max(8, Math.min(r.left + r.width / 2 - 140, innerWidth - 288)) + "px";
    pop.style.top = Math.max(8, r.top - pop.offsetHeight - 8) + "px";
    pop.addEventListener("mouseenter", () => clearTimeout(leave));
    pop.addEventListener("mouseleave", later);
  });
  ring.addEventListener("mouseleave", later);
}

/* ---- Voice, in Settings (each model's line in English, put into the
   page's language where shown): dictation's model (Whisper, downloaded from
   the catalog, checked), the voices that read the replies aloud ---- */
const SPEECH = [["whisper-small", "small and quick: for dictation on most computers"],
                ["whisper-large-v3-turbo", "more accurate, for long recordings and difficult sound: wants more memory"]];
function drawVoice() {
  const box = $("#voicecard");
  if (!box) return;
  const have = SPEECH.filter(([n]) => speechModels.includes(n));
  const size = (n) => { const c = catalog && catalog.models.find((m) => m.name === n); return c ? Number(c.download_gb).toLocaleString(LOCALE, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + " GB" : ""; };
  const voices = "speechSynthesis" in window ? speechSynthesis.getVoices() : [];
  const lang = (navigator.language || "en").slice(0, 2);
  const mine = voices.filter((v) => v.lang.toLowerCase().startsWith(lang));
  let chosen = "";
  try { chosen = localStorage.getItem("brebus.voice") || ""; } catch (e) {}
  box.innerHTML = `
    <div class="vrow"><b>${T("Dictation")}</b>
      <span class="note">${T("The microphone under the field writes what you say, through a speech model (OpenAI's Whisper) that runs on this computer: nothing leaves it.")}</span>
      ${SPEECH.map(([n, what]) => `<div class="vline"><span><b>${esc(n)}</b> <small>${esc(size(n))} · ${esc(T(what))}</small></span>
        ${speechModels.includes(n)
          ? `<span class="badge fits">${T("installed")}</span><button class="act quiet vrm" data-m="${esc(n)}" title="${T("Delete it from this computer")}">${T("Remove")}</button>`
          : `<button class="act quiet vget" data-m="${esc(n)}" title="${T("Download it from Hugging Face, checked against its published fingerprint")}">${T("Get")}</button>`}</div>`).join("")}
      ${have.length && server.running ? `<span class="note">${T("A model just got is used from the next time the chat's model loads.")}</span>` : ""}
    </div>
    <div class="vrow"><b>${T("Reading aloud")}</b>
      <span class="note">${voices.length ? T("The speaker under the field reads the replies with this computer's own voices.")
        : T("The speaker under the field reads the replies with this computer's own voices - none found here: on Linux, install speech-dispatcher with a voice (espeak-ng).")} ${T("Janas's own voices will come with its speech synthesis.")}</span>
      ${voices.length ? `<label>${T("Voice")} <select id="voicesel"><option value="">${T("this computer's default for {lang}", {lang: esc(lang)})}</option>
        ${(mine.length ? mine : voices).map((v) => `<option value="${esc(v.name)}"${v.name === chosen ? " selected" : ""}>${esc(v.name)} (${esc(v.lang)})</option>`).join("")}</select></label>` : ""}
    </div>`;
  for (const b of box.querySelectorAll(".vget")) b.onclick = () => getModel(b.dataset.m);
  for (const b of box.querySelectorAll(".vrm"))
    b.onclick = async () => {
      if (!await yes(T("{m} is deleted from this computer; dictation will offer it again.", {m: b.dataset.m}), T("Remove this speech model?"), T("Remove"))) return;
      const r = await fetch("/brebus/models/remove", {method: "POST", body: JSON.stringify({name: b.dataset.m})});
      if (!r.ok) say((await r.json()).error || T("not removed: unload the chat's model first, which uses it"));
      await mine();
    };
  const sel = box.querySelector("#voicesel");
  if (sel) sel.onchange = () => { try { localStorage.setItem("brebus.voice", sel.value); } catch (e) {} };
}
if ("speechSynthesis" in window) speechSynthesis.onvoiceschanged = () => drawVoice();

