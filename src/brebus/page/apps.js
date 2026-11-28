/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* apps.js - Brebus's page (app.html): the views of MCP Apps (the extension
   io.modelcontextprotocol/ui, SEP-1865): a tool whose server gives it a
   view shows its result in it, in a frame of its own (/brebus/view: its
   own policy, a sandbox, an origin of its own), which this page talks to
   by JSON-RPC over postMessage. The view gets the tool's input and result,
   the page's theme and language; it may open a link (the browser, http and
   https alone) and call the tools of its own server that only read (Brebus
   checks it too). build.sh puts it into the page where app.html includes
   it. */
const appViews = new Map(); /* frame -> its message */
let viewSaved = 0; /* when a view last saved a file */

/* Janas's services by a name for people and a line of what they do, in the
   page's language (the lines in C are the model's, in English); a server
   of the user's by the name its file gives it */
function svcNames() {
  return {
    docs: [T("Your documents"), T("Finds the passages of your documents that answer a question.")],
    files: [T("Your files"), T("Reads your files and folders; writes or edits one only when you allow it.")],
    flights: [T("Flights"), T("Where a flight is, the flights of a route, the planes over a sea or near a place.")],
    git: [T("Git repositories"), T("The changes, commits and branches of this computer's repositories; a commit, pull or push only when you allow it.")],
    github: [T("GitHub"), T("A project's issues, pull requests, latest release and recent commits.")],
    maps: [T("Maps and routes"), T("Routes by car, bike, on foot or by public transport, what is near, where an address is.")],
    prices: [T("Prices and finance"), T("Currencies, crypto-assets, shares, inflation, central banks' rates, electricity, fuel.")],
    quakes: [T("Earthquakes"), T("The latest earthquakes near a place, the strongest in the world, one in detail.")],
    system: [T("This computer"), T("Its memory, processor, disks, battery and temperatures, the programs running, its errors and updates.")],
    weather: [T("Weather"), T("The weather now, the forecast, the sea and its waves, weather warnings.")],
    wiki: [T("Wikipedia"), T("Wikipedia's pages, and what there is to see around a place.")],
  };
}
/* the services whose tools stay on this computer */
const SVC_LOCAL = ["docs", "files", "git", "system"];
const svcLabel = (n) => (svcNames()[n] || [n])[0];
const svcAbout = (n, d) => (svcNames()[n] || [null, d || ""])[1];
/* a tool's service, by its name: a server of the user's by its prefix (the
   longest: "a__x" is server a's, "a___x" server a_'s), Janas's before "__",
   the workspace's own by theirs */
function toolService(name) {
  const s = (typeof tools !== "undefined" && tools.servers || []).filter((x) => name.startsWith(x.prefix))
    .sort((x, y) => y.prefix.length - x.prefix.length)[0];
  if (s) return s.prefix.slice(0, -2);
  const i = name.indexOf("__");
  if (i > 0) return name.slice(0, i);
  const w = /^(docs|files)_/.exec(name);
  return w ? w[1] : "";
}
/* a tool's call as the chat names it: the service, then the tool's own name
   in words */
function toolLabel(m) {
  let a = {};
  try { a = JSON.parse(m.args || "{}"); } catch (e) {}
  if (m.name === "janas__open_service") return T("Opened: {s}", {s: svcLabel(a.service || "")});
  if (m.name === "janas__skill") return T("Skill: {s}", {s: a.name || a.skill || ""});
  if (m.name === "python") return T("Python code");
  const svc = toolService(m.name);
  return svc ? svcLabel(svc) + " · " + toolShort(m.name) : toolShort(m.name);
}
/* a tool's own name, without its service's, in words */
function toolShort(name) {
  const i = name.indexOf("__");
  let own = i > 0 ? name.slice(i + 2) : name.replace(/^(docs|files)_/, "");
  if (i > 0 && own.startsWith(name.slice(0, i) + "_")) own = own.slice(i + 1); /* quakes__quakes_near: near */
  return own.replace(/_/g, " ");
}
/* a call's arguments as a list, a value each, not as JSON */
function argsList(a) {
  if (!a || typeof a !== "object" || Array.isArray(a)) return `<pre class="dargs">${esc(JSON.stringify(a, null, 2))}</pre>`;
  const rows = Object.entries(a).map(([k, v]) =>
    `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${esc(typeof v === "string" ? v : JSON.stringify(v, null, 2))}</dd>`).join("");
  return rows ? `<dl class="cardl dlargs">${rows}</dl>` : "";
}

/* the question of a call that changes something, in words: the service by
   its name and what it does, the arguments as a list, the code as it is */
function approvalText(name, args) {
  let a = null;
  try { a = JSON.parse(args); } catch (e) {}
  const svc = toolService(name);
  let title = T("The model wants to use {s}", {s: svc ? svcLabel(svc) : name}), html;
  if (name === "python") {
    title = T("The model wants to run Python code");
    html = `<p>${T("Run in a sandbox: the workspace (read and write) and its folders read only, no network.")}</p>` +
           `<pre class="dargs">${esc(a && typeof a.code === "string" ? a.code : args)}</pre>`;
  } else if (name === "janas__open_service" && a && typeof a.service === "string") {
    const s = a.service, user = !svcNames()[s];
    title = T("The model asks to open {s}", {s: svcLabel(s)});
    html = `<p>${esc(user ? T("A server of yours, from mcp.json.") : svcAbout(s))}</p>` +
           `<p class="note">${user ? T("Its tools may reach the network.") : SVC_LOCAL.includes(s) ? T("Its tools work on this computer.") : T("Its tools look things up on the network.")}</p>`;
  } else {
    const t = tools.tools.find((x) => x.function && x.function.name === name);
    const d = t && t.function.description ? t.function.description.split(/(?<=[.!?])\s/)[0] : "";
    html = `<p>${T("It changes something: check what it will do.")}</p>` +
           (d ? `<p class="note">${esc(toolShort(name))}: ${esc(d)}</p>` : "") +
           (a ? argsList(a) : `<pre class="dargs">${esc(args)}</pre>`);
  }
  return {title, html};
}

/* a view in the pane beside the chat, as an artifact: drawn once, so that
   what the user did in it stays while the chat is drawn again */
let viewShown = null;
function artShowView(m) {
  artShown = null; artFrame = null; fileShown = null; viewShown = m;
  $("#artdown").hidden = $("#artcopy").hidden = true;
  if ($("#artpane").hidden && innerWidth < 1500 && !$("#side").classList.contains("mini")) {
    navMini(true, false); navAuto = true;
  }
  $("#artpane").hidden = false;
  $("#arttitle").textContent = svcLabel(toolService(m.name)) || toolLabel(m);
  /* the view has its own controls: no versions, source, zoom or file */
  $("#artver").hidden = $("#arterr").hidden = $("#artview").hidden = $("#artopen").hidden = $("#artzoom").hidden = true;
  $("#artpath").textContent = "";
  const body = $("#artbody");
  body.innerHTML = "";
  viewMount(body, m);
  viewsPrune();
}
function viewMount(box, m) {
  const v = m.view;
  if (!v || !v.server || !v.uri || !m.raw) return;
  const f = document.createElement("iframe");
  f.className = "mcpview";
  f.setAttribute("sandbox", "allow-scripts");
  f.setAttribute("referrerpolicy", "no-referrer");
  f.title = T("{name}'s view", {name: m.name});
  f.src = "/brebus/view?server=" + encodeURIComponent(v.server) + "&uri=" + encodeURIComponent(v.uri);
  box.append(f);
  appViews.set(f, m);
}
/* frames gone from the page (the chat drawn again) are forgotten */
function viewsPrune() {
  for (const f of appViews.keys()) if (!f.isConnected) appViews.delete(f);
}
/* the page's colours as MCP Apps names them, its theme and its language */
function viewContext() {
  const s = getComputedStyle(document.documentElement), v = (n) => s.getPropertyValue(n).trim();
  return {
    theme: document.documentElement.classList.contains("dark") ? "dark" : "light",
    locale: typeof LOCALE !== "undefined" && LOCALE ? LOCALE : navigator.language,
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    /* in the pane beside the chat: the whole of it */
    platform: "desktop", displayMode: "fullscreen", availableDisplayModes: ["fullscreen"],
    styles: {variables: {
      "--color-background-primary": v("--panel"), "--color-background-secondary": v("--bg"),
      "--color-text-primary": v("--fg"), "--color-text-secondary": v("--dim"),
      "--color-border-primary": v("--line"), "--color-ring-primary": v("--link"),
      "--font-sans": v("--sans"), "--font-mono": v("--mono")}},
  };
}
function viewsTheme() {
  viewsPrune();
  const c = viewContext();
  for (const f of appViews.keys())
    f.contentWindow && f.contentWindow.postMessage({jsonrpc: "2.0", method: "ui/notifications/host-context-changed", params: c}, "*");
}
window.addEventListener("message", async (e) => {
  let f = null;
  for (const k of appViews.keys()) if (k.contentWindow === e.source) { f = k; break; }
  if (!f) return;
  const m = appViews.get(f), q = e.data;
  if (!q || q.jsonrpc !== "2.0" || typeof q.method !== "string") return;
  const post = (x) => f.contentWindow && f.contentWindow.postMessage(Object.assign({jsonrpc: "2.0"}, x), "*");
  const ok = (r) => q.id !== undefined && post({id: q.id, result: r});
  const no = (code, msg) => q.id !== undefined && post({id: q.id, error: {code, message: msg}});
  switch (q.method) {
  case "ui/initialize":
    /* the tool it shows, by its server's name for it */
    ok({protocolVersion: "2026-01-26", hostInfo: {name: "Brebus", version: "1"},
        hostCapabilities: {openLinks: {}, serverTools: {}, "x-brebus": {download: {}}},
        hostContext: Object.assign(viewContext(), {toolInfo: {tool: {name: m.name.slice(m.view.prefix.length)}}})});
    break;
  case "ui/notifications/initialized": {
    let args = {};
    try { args = typeof m.args === "string" ? JSON.parse(m.args || "{}") : (m.args || {}); } catch (x) {}
    post({method: "ui/notifications/tool-input", params: {arguments: args}});
    post({method: "ui/notifications/tool-result", params: m.raw});
    break;
  }
  case "ui/notifications/size-changed":
    break; /* the pane's height is the user's: the view scrolls in it */
  case "ui/open-link": {
    const u = String((q.params || {}).url || "");
    if (!/^https?:\/\//i.test(u)) { no(-32000, "Invalid URL"); break; }
    window.open(u, "_blank", "noopener");
    ok({});
    break;
  }
  case "tools/call": {
    const p = q.params || {};
    if (typeof p.name !== "string") { no(-32602, "Invalid params"); break; }
    const r = await fetch("/brebus/tools/call", {method: "POST", body: JSON.stringify(
      {name: m.view.prefix + p.name, arguments: p.arguments || {}, from_view: m.view.server})});
    const j = await r.json().catch(() => ({}));
    ok(j.raw || {content: [{type: "text", text: String(j.result || j.error || "")}], isError: !!j.is_error || !r.ok});
    break;
  }
  case "x-brebus/download": {
    /* a file the view made (a table exported, the map saved), saved as
       the artifacts' are: its kinds alone, 20 MB at most, one a second */
    const p = q.params || {};
    const ext = {"text/csv": "csv", "text/tab-separated-values": "tsv", "application/json": "json", "text/markdown": "md",
                 "image/png": "png", "image/svg+xml": "svg"}[p.mimeType];
    if (!ext || typeof p.data !== "string" || p.data.length > 28e6) { no(-32602, "Invalid params"); break; }
    if (Date.now() - viewSaved < 1000) { no(-32000, "Too many files at once"); break; }
    viewSaved = Date.now();
    let body = p.data;
    if (p.base64) {
      try { const b = atob(p.data), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); body = u; }
      catch (e) { no(-32602, "Invalid params"); break; }
    }
    const name = (String(p.name || "data").replace(/[\\/:*?"<>|\x00-\x1f]+/g, "_").replace(/\.[a-z0-9]{1,5}$/i, "").slice(0, 100) || "data") + "." + ext;
    const u = URL.createObjectURL(new Blob([body], {type: p.mimeType}));
    const l = document.createElement("a");
    l.href = u; l.download = name;
    document.body.append(l); l.click(); l.remove();
    setTimeout(() => URL.revokeObjectURL(u), 2000);
    ok({name});
    break;
  }
  case "ui/request-display-mode": ok({mode: "fullscreen"}); break;
  case "ui/update-model-context": ok({}); break;
  case "ping": ok({}); break;
  case "notifications/message": break;
  case "ui/message": no(-32000, "Message sending denied"); break;
  default: no(-32601, "Method not found");
  }
});
