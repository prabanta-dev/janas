/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* app.js - a view's side of MCP Apps (the extension io.modelcontextprotocol/ui,
   SEP-1865, 2026-01-26): JSON-RPC over postMessage with the client that
   shows it (its host), which hands it the tool's input and result, its
   theme, its language; the view tells it its size, and asks it to open a
   link or to call a tool of its server. Written for Janas's services, no
   SDK: what they need of the protocol and nothing else. */
const App = (() => {
  let next = 0;
  const pending = new Map(), on = {};
  const ctx = {theme: "light", locale: navigator.language || "en"};
  let caps = {}; /* what the host said it can do (hostCapabilities) */
  let input = {}; /* the tool's arguments, as the host gave them */
  const send = (m) => window.parent.postMessage(m, "*");
  function request(method, params) {
    const id = ++next;
    send({jsonrpc: "2.0", id, method, params: params || {}});
    return new Promise((ok, no) => pending.set(id, {ok, no}));
  }
  const notify = (method, params) => send({jsonrpc: "2.0", method, params: params || {}});
  /* the host's theme: its colours as CSS variables (the view keeps its own
     where it gives none), light or dark */
  function context(c) {
    if (!c) return;
    Object.assign(ctx, c);
    const root = document.documentElement;
    if (c.theme) root.dataset.theme = c.theme;
    /* the whole of the host's room: the view fills it, its list scrolls */
    if (c.displayMode) root.classList.toggle("fill", c.displayMode === "fullscreen");
    const vars = c.styles && c.styles.variables;
    if (vars) for (const k of Object.keys(vars)) if (/^--[a-z0-9-]+$/.test(k) && typeof vars[k] === "string") root.style.setProperty(k, vars[k]);
    if (on.context) on.context(ctx);
  }
  window.addEventListener("message", (e) => {
    if (e.source !== window.parent) return;
    const m = e.data;
    if (!m || m.jsonrpc !== "2.0") return;
    if (m.id !== undefined && !m.method) { /* an answer */
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      if (m.error) p.no(new Error(m.error.message || "error")); else p.ok(m.result);
      return;
    }
    switch (m.method) {
    case "ui/resource-teardown": send({jsonrpc: "2.0", id: m.id, result: {}}); break;
    case "ui/notifications/host-context-changed": context(m.params); break;
    case "ui/notifications/tool-input": input = (m.params || {}).arguments || {}; if (on.input) on.input(input); break;
    case "ui/notifications/tool-result": if (on.result) on.result(m.params || {}); break;
    case "ui/notifications/tool-cancelled": if (on.cancelled) on.cancelled((m.params || {}).reason || ""); break;
    default: if (m.id !== undefined) send({jsonrpc: "2.0", id: m.id, error: {code: -32601, message: "Method not found"}});
    }
  });
  /* its size, whenever it changes: the host gives it that much room */
  let last = "";
  const size = () => {
    const w = Math.ceil(document.documentElement.scrollWidth), h = Math.ceil(document.documentElement.scrollHeight);
    if (w + "x" + h === last) return;
    last = w + "x" + h;
    notify("ui/notifications/size-changed", {width: w, height: h});
  };
  return {
    ctx,
    /* on("result" | "input" | "cancelled" | "context", f) */
    on: (what, f) => { on[what] = f; },
    start(name) {
      request("ui/initialize", {protocolVersion: "2026-01-26", clientInfo: {name, version: "1"},
        appCapabilities: {availableDisplayModes: ["inline"]}})
        .then((r) => { caps = (r && r.hostCapabilities) || {}; context(r && r.hostContext); notify("ui/notifications/initialized"); })
        .catch(() => notify("ui/notifications/initialized"));
      if (window.ResizeObserver) new ResizeObserver(size).observe(document.documentElement);
      window.addEventListener("load", size);
    },
    size,
    openLink: (url) => request("ui/open-link", {url}),
    /* a file handed to the host to save, where it says it can (Brebus:
       x-brebus's download): text, or bytes as base64 */
    canDownload: () => !!(caps["x-brebus"] && caps["x-brebus"].download),
    download: (name, mimeType, data, base64) => request("x-brebus/download", {name, mimeType, data, base64: !!base64}),
    callTool: (name, args) => request("tools/call", {name, arguments: args || {}}),
    /* the tool this view shows (the host's word, toolInfo) and its input:
       to ask it again; whether the host lets the view call its server */
    toolName: () => (ctx.toolInfo && ctx.toolInfo.tool && ctx.toolInfo.tool.name) || "",
    input: () => input,
    canCall: () => !!caps.serverTools,
    /* a card of details given its × at the top right, which calls close */
    closer(card, close) {
      const w = {en: "Close", it: "Chiudi", fr: "Fermer", es: "Cerrar", de: "Schließen", pt: "Fechar"};
      const l = (ctx.locale || "en").toLowerCase(), word = w[l] || w[l.split("-")[0]] || w.en;
      const b = document.createElement("button");
      b.type = "button"; b.className = "vclose"; b.textContent = "×"; b.title = word; b.setAttribute("aria-label", word);
      b.onclick = close;
      card.classList.add("hasclose");
      card.prepend(b);
    },
    /* the view's words in the host's language: t({en: ..., it: ...}) */
    t(words) {
      const l = (ctx.locale || "en").toLowerCase();
      return words[l] || words[l.split("-")[0]] || words.en;
    },
  };
})();
