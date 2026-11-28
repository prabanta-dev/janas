/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* ext.js - Brebus's page (app.html): Extensions, what the model can use
   beyond what it knows, in three tabs. Installed: Janas's services and the
   user's MCP servers (mcp.json, edited here: an address or a program, its
   headers or arguments and environment), each on or off, signed in or not;
   desktop extensions (.mcpb) installed from a file. Connectors: Janas's
   list of services on the web (from GitHub, ext.c; the copy this page was
   built with otherwise) and a search in MCP's official registry, its
   servers added to mcp.json with what they ask for. Skills: those there
   are, on or off; Janas's list; a repository of GitHub; a folder.
   build.sh puts it into the page where app.html includes it. */
let extTabNow = "inst";
function extTab(t) {
  extTabNow = t;
  for (const b of document.querySelectorAll("#ext .exttabs button")) {
    b.classList.toggle("on", b.dataset.t === t);
    b.setAttribute("aria-selected", b.dataset.t === t ? "true" : "false");
  }
  for (const s of document.querySelectorAll("#ext .exttab")) s.hidden = s.dataset.t !== t;
  if (t === "conn") extConnectors();
  if (t === "skill") extSkills();
}
for (const b of document.querySelectorAll("#ext .exttabs button")) b.onclick = () => extTab(b.dataset.t);
function showExt(t) { show("ext"); extTab(t || extTabNow); drawExt(); }
$('nav button[data-s="ext"]').addEventListener("click", () => { extTab(extTabNow); drawExt(); });
$("#prefsext").onclick = (e) => { e.preventDefault(); showExt("inst"); };

/* Janas's list of connectors or skills: from GitHub, or this page's own */
const extLists = {};
async function extList(kind) {
  if (extLists[kind]) return extLists[kind];
  let j = null;
  try { const r = await fetch("/brebus/ext/catalog?kind=" + kind); if (r.ok) j = await r.json(); } catch (e) {}
  extLists[kind] = (j && Array.isArray(j.items) ? j : EXT_FALLBACK[kind]).items;
  return extLists[kind];
}
const extMatch = (q, ...fields) => !q || q.toLowerCase().split(/\s+/).filter(Boolean)
  .every((w) => fields.some((f) => String(f || "").toLowerCase().includes(w)));

/* ---- Installed ---- */
async function drawExt() {
  let j;
  try { j = await ask("services"); } catch (e) { return; }
  const mine = j.services.filter((x) => x.user), janas = j.services.filter((x) => !x.user);
  const row = (x) => {
    const state = !x.http ? "" : x.signed_in ? T("signed in") : x.sign_in ? T("it asks you to sign in") : "";
    const auth = !x.http ? "" : x.signed_in
      ? `<button type="button" class="act quiet exauth" data-n="${esc(x.name)}" data-out="1">${T("Sign out")}</button>`
      : `<button type="button" class="act exauth" data-n="${esc(x.name)}">${T("Sign in")}</button>`;
    const title = x.user ? x.name : svcLabel(x.name);
    const about = x.user ? (x.where || "") : svcAbout(x.name, x.about);
    return `<div class="svc"><label><input type="checkbox" class="exon" data-n="${esc(x.name)}"${x.on ? " checked" : ""}> <b>${esc(title)}</b>
      <small${x.user ? ' class="mono"' : ""}>${esc(about)}${state ? " · " + esc(state) : ""}</small></label>
      <span class="exbtns">${auth}${x.user ? `<button type="button" class="act quiet exedit" data-n="${esc(x.name)}">${T("Edit")}</button>
      <button type="button" class="act quiet exrm" data-n="${esc(x.name)}">${T("Remove")}</button>` : ""}</span></div>`;
  };
  $("#extjanas").innerHTML = janas.map(row).join("") || `<div class="note">${T("none found beside Brebus")}</div>`;
  $("#extmine").innerHTML = mine.map(row).join("") || `<div class="note">${T("None yet: add one below, or find one among the Connectors.")}</div>`;
  $("#extcfg").innerHTML = T("They are kept in <code>{config}</code>, in the format of Cursor and LM Studio: what you change here is written there.", {config: esc(j.config)});
  for (const c of document.querySelectorAll("#ext .exon"))
    c.onchange = async () => {
      const off = [...document.querySelectorAll("#ext .exon")].filter((x) => !x.checked).map((x) => x.dataset.n);
      settings._services = {off};
      await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
      await fetch("/brebus/tools/restart", {method: "POST", body: "{}"});
      await readTools(); readServices();
    };
  for (const b of document.querySelectorAll("#ext .exauth"))
    b.onclick = async () => { await (b.dataset.out ? signOut(b.dataset.n) : signIn(b.dataset.n)); drawExt(); };
  for (const b of document.querySelectorAll("#ext .exedit")) b.onclick = () => extEdit(b.dataset.n);
  for (const b of document.querySelectorAll("#ext .exrm"))
    b.onclick = async () => {
      if (!await yes(T("{n} is taken out of mcp.json (a desktop extension's folder too); the model can no longer open it.", {n: b.dataset.n}), T("Remove this server?"), T("Remove"))) return;
      const r = await fetch("/brebus/mcp/remove", {method: "POST", body: JSON.stringify({name: b.dataset.n})});
      if (!r.ok) say((await r.json()).error || T("not removed"));
      extChanged();
    };
  drawExtensions();
}
async function extChanged() {
  await fetch("/brebus/tools/restart", {method: "POST", body: "{}"}).catch(() => {});
  await readTools(); readServices(); drawExt();
  if (extTabNow === "conn") extConnectors();
}
/* "Name: value" a line, "NAME=value" a line, a value a line */
const extPairs = (t, sep) => Object.fromEntries(t.split("\n").map((l) => l.trim()).filter(Boolean)
  .map((l) => { const i = l.indexOf(sep); return i < 0 ? [l, ""] : [l.slice(0, i).trim(), l.slice(i + 1).trim()]; }));
const extLines = (o, sep) => Object.entries(o || {}).map(([k, v]) => k + sep + v).join("\n");
/* a server of the user's, new or to change: an address or a program */
async function extEdit(name) {
  let e = {}, ext = false;
  if (name) {
    const r = await fetch("/brebus/mcp/get", {method: "POST", body: JSON.stringify({name})});
    const j = await r.json();
    if (!r.ok) { say(j.error || T("not found")); return; }
    e = j.entry; ext = j.extension;
  }
  const prog = !!e.command;
  const html = `<div class="srvset plainset exform">
    ${ext ? `<p class="note">${T("A desktop extension's: its program is in the folder Brebus installed it in.")}</p>` : ""}
    <label>${T("Name")} <input class="exname" value="${esc(name || "")}" spellcheck="false" placeholder="my-server">
      <span class="note">${T("Letters, digits, - _ and . alone: the model sees its tools under this name.")}</span></label>
    <div class="seg exkind" role="radiogroup"><button type="button" data-k="url" class="${prog ? "" : "on"}">${T("An address (HTTP)")}</button><button type="button" data-k="cmd" class="${prog ? "on" : ""}">${T("A program on this computer")}</button></div>
    <div class="exk" data-k="url"${prog ? " hidden" : ""}>
      <label>${T("Address")} <input class="exurl" value="${esc(e.url || "")}" spellcheck="false" placeholder="https://example.com/mcp"></label>
      <label>${T("Headers (optional), one a line")} <textarea class="exhdr" rows="3" spellcheck="false" placeholder="Authorization: Bearer …">${esc(extLines(e.headers, ": "))}</textarea>
        <span class="note">${T("A server that signs you in by OAuth needs none: Sign in, once added.")}</span></label></div>
    <div class="exk" data-k="cmd"${prog ? "" : " hidden"}>
      <label>${T("Command")} <input class="excmd" value="${esc(e.command || "")}" spellcheck="false" placeholder="npx"></label>
      <label>${T("Arguments, one a line")} <textarea class="exargs" rows="3" spellcheck="false" placeholder="-y&#10;@scope/server">${esc((e.args || []).join("\n"))}</textarea></label>
      <label>${T("Environment (optional), NAME=value a line")} <textarea class="exenv" rows="3" spellcheck="false" placeholder="API_KEY=…">${esc(extLines(e.env, "="))}</textarea></label>
      <p class="note">${T("The program runs on this computer whenever the model opens it, as you: add only one you trust.")}</p></div></div>`;
  const d = await dialog({title: name ? T("Change {n}", {n: name}) : T("Add an MCP server"), wide: true, html,
    ready: (box) => {
      for (const b of box.querySelectorAll(".exkind button"))
        b.onclick = () => {
          for (const x of box.querySelectorAll(".exkind button")) x.classList.toggle("on", x === b);
          for (const k of box.querySelectorAll(".exk")) k.hidden = k.dataset.k !== b.dataset.k;
        };
    },
    read: (box) => {
      const cmd = box.querySelector('.exkind [data-k="cmd"]').classList.contains("on");
      const out = {};
      if (cmd) {
        out.command = box.querySelector(".excmd").value.trim();
        const args = box.querySelector(".exargs").value.split("\n").map((s) => s.trim()).filter(Boolean);
        if (args.length) out.args = args;
        const env = extPairs(box.querySelector(".exenv").value, "=");
        if (Object.keys(env).length) out.env = env;
      } else {
        out.url = box.querySelector(".exurl").value.trim();
        const h = extPairs(box.querySelector(".exhdr").value, ":");
        if (Object.keys(h).length) out.headers = h;
      }
      if (e.disabled) out.disabled = e.disabled;
      return {name: box.querySelector(".exname").value.trim(), entry: out};
    },
    buttons: [{label: T("Save"), value: "ok", primary: true}, {label: T("Cancel"), value: "no"}]});
  if (d.button !== "ok") return;
  const r = await fetch("/brebus/mcp/set", {method: "POST", body: JSON.stringify({name: d.form.name, old: name || "", entry: d.form.entry})});
  const j = await r.json();
  if (!r.ok) { say(j.error || T("not saved"), T("Not saved")); return; }
  await extChanged();
  say(T("{n} is saved: the model can open it from the next message.", {n: j.name}), T("Saved"));
}
$("#extadd").onclick = () => extEdit("");

/* desktop extensions (.mcpb), as Claude Desktop installs them */
async function drawExtensions() {
  let mine;
  try { mine = await ask("connectors"); } catch (e) { return; }
  $("#extmcpbnote").innerHTML = T("MCP Bundles (.mcpb), as Claude Desktop installs them: a program that reaches an application of this computer. Brebus asks for the settings it needs and installs it in <code>{dir}</code>; it is then listed among your servers.", {dir: esc(mine.dir)});
  $("#cnbrowse").hidden = !(ws && ws.window);
}
$("#cnbrowse").onclick = async () => {
  const p = await pick(true, T("A desktop extension (.mcpb)"));
  if (p) $("#cnadd").path.value = p;
};
$("#cnadd").onsubmit = async () => {
  const path = $("#cnadd").path.value.trim();
  if (!path) return false;
  const r = await fetch("/brebus/connectors/inspect", {method: "POST", body: JSON.stringify({path})});
  const m = await r.json();
  if (!r.ok) { say(m.error || T("not an extension"), T("Not installed")); return false; }
  if (!m.runs) { say(`${m.display_name}: ${m.why}.`, T("Not installed")); return false; }
  const fields = m.user_config.map((f) =>
    `<label>${f.required ? esc(f.title) : T("{f} (optional)", {f: esc(f.title)})}
      <input class="cnf" data-k="${esc(f.key)}" type="${f.sensitive ? "password" : f.type === "number" ? "number" : "text"}" value="${f.default === null || f.default === undefined ? "" : esc(String(f.default))}" spellcheck="false">
      ${f.description ? `<span class="note">${esc(f.description)}</span>` : ""}</label>`).join("");
  const d = await dialog({title: T("Install {n}?", {n: m.display_name}),
    html: `<div class="srvset plainset"><p class="note">${esc(m.description || "")}${m.version ? " (v" + esc(m.version) + ")" : ""}. ${T("Its program runs on this computer when the model opens it, with your settings below; it is not in a cage.")}</p>${fields}</div>`,
    read: (b) => Object.fromEntries([...b.querySelectorAll(".cnf")].map((i) => [i.dataset.k, i.type === "number" && i.value !== "" ? +i.value : i.value])),
    buttons: [{label: T("Install"), value: "ok", primary: true}, {label: T("Cancel"), value: "no"}]});
  if (d.button !== "ok") return false;
  const ri = await fetch("/brebus/connectors/install", {method: "POST", body: JSON.stringify({path, config: d.form})});
  const k = await ri.json();
  say(ri.ok ? T("{n} is installed: the model can open it, as the other services.", {n: m.display_name}) : k.error, ri.ok ? T("Installed") : T("Not installed"));
  $("#cnadd").path.value = "";
  extChanged();
  return false;
};

/* ---- Connectors ---- */
/* never a server of Russian or Chinese makers, where it shows: the
   registry's name (its domain reversed), its address, its repository */
const EXT_DENY = /(^|[^a-z])(ru|su|cn|xn--p1ai)\.|\.(ru|su|cn|xn--p1ai)([/:]|$)|yandex|baidu|alibaba|aliyun|alipay|taobao|tencent|wechat|weixin|bytedance|volcengine|doubao|huawei|zhipu|bigmodel|moonshot|kimi|deepseek|minimax|dashscope|qwen|kuaishou|xiaomi|sber|gigachat|kaspersky|mail\.ru|vk\.com|ozon|wildberries|tinkoff|1c-/i;
const extDenied = (...s) => s.some((x) => x && EXT_DENY.test(String(x)));
let extReg = {q: null, items: [], next: ""};
async function extConnectors() {
  const q = $("#connq").value.trim();
  const [list, sv] = await Promise.all([extList("connectors"), ask("services").catch(() => ({services: []}))]);
  const mine = sv.services.filter((x) => x.user);
  const has = (c) => mine.find((x) => x.name === c.id || (x.where && x.where === c.url));
  const rec = list.filter((c) => extMatch(q, c.title, c.about, c.url));
  $("#connrec").innerHTML = rec.map((c) => {
    const m = has(c);
    return `<div class="svc"><label><b>${esc(c.title)}</b> <small>${esc(c.about)}${c.sign_in === "none" ? " · " + T("no sign-in") : " · " + T("sign in once added")}</small></label>
      <button type="button" class="act ${m ? "quiet " : ""}cnrec" data-id="${esc(c.id)}" data-n="${m ? esc(m.name) : ""}">${m ? T("Remove") : T("Add")}</button></div>`;
  }).join("") || `<div class="note">${T("Nothing in Janas's list matches.")}</div>`;
  for (const b of document.querySelectorAll("#connrec .cnrec"))
    b.onclick = async () => {
      const c = list.find((x) => x.id === b.dataset.id);
      if (b.dataset.n) await fetch("/brebus/mcp/remove", {method: "POST", body: JSON.stringify({name: b.dataset.n})});
      else {
        const r = await fetch("/brebus/mcp/set", {method: "POST", body: JSON.stringify({name: c.id, entry: {url: c.url}})});
        if (!r.ok) { say((await r.json()).error || T("not added")); return; }
        if (c.sign_in !== "none") say(T("{n} is added: Sign in to it in My extensions.", {n: c.title}), T("Added"));
      }
      extChanged();
    };
  if (q !== extReg.q) await extRegistry(q, false);
  else drawRegistry(mine);
}
async function extRegistry(q, more) {
  const box = $("#connreg");
  if (!more) { extReg = {q, items: [], next: ""}; box.innerHTML = `<div class="note">${T("Asking the registry…")}</div>`; }
  let j;
  try {
    const r = await fetch("/brebus/ext/registry?q=" + encodeURIComponent(q) + (more && extReg.next ? "&cursor=" + encodeURIComponent(extReg.next) : ""));
    j = await r.json();
    if (!r.ok) throw new Error(j.error || r.status);
  } catch (e) { box.innerHTML = `<div class="note">${T("The registry did not answer: {e}", {e: esc(e.message)})}</div>`; return; }
  const seen = new Set(extReg.items.map((x) => x.name));
  for (const e of j.servers || []) {
    const s = e.server || {}, meta = (e._meta || {})["io.modelcontextprotocol.registry/official"] || {};
    if (!s.name || seen.has(s.name) || meta.status === "deleted" || meta.isLatest === false) continue;
    if (extDenied(s.name, s.websiteUrl, s.repository && s.repository.url, ...(s.remotes || []).map((r) => r.url))) continue;
    seen.add(s.name); extReg.items.push(s);
  }
  extReg.next = (j.metadata && j.metadata.nextCursor) || "";
  const sv = await ask("services").catch(() => ({services: []}));
  drawRegistry(sv.services.filter((x) => x.user));
}
/* how a server of the registry runs here: an address, or npx or uvx */
function regWay(s) {
  const r = (s.remotes || []).find((x) => x.type === "streamable-http" || x.type === "sse");
  if (r) return {kind: "url", remote: r};
  const p = (s.packages || []).find((x) => x.registryType === "npm" || x.registryType === "pypi");
  if (p && (!p.transport || p.transport.type === "stdio")) return {kind: "pkg", pkg: p};
  return null;
}
function drawRegistry(mine) {
  const box = $("#connreg");
  box.innerHTML = extReg.items.map((s, i) => {
    const w = regWay(s), name = regName(s);
    const added = mine.find((x) => x.name === name);
    const how = !w ? T("needs a way Brebus does not run (a container…)") : w.kind === "url" ? T("on the web") : w.pkg.registryType === "npm" ? T("a program, through npx (Node.js)") : T("a program, through uvx (Python)");
    return `<div class="svc"><label><b>${esc(s.title || s.name)}</b> <small class="desc" title="${esc(s.description || "")}">${esc(s.description || "")}</small><small>${esc(how)}${s.version ? " · v" + esc(s.version) : ""}</small>
      <small class="mono">${esc(s.name)}</small></label>
      <span class="exbtns">${s.repository && s.repository.url ? `<button type="button" class="act quiet regsrc" data-u="${esc(s.repository.url)}">${T("Source")}</button>` : ""}
      ${w ? `<button type="button" class="act ${added ? "quiet " : ""}regadd" data-i="${i}"${added ? ` data-n="${esc(added.name)}"` : ""}>${added ? T("Remove") : T("Add")}</button>` : ""}</span></div>`;
  }).join("") || `<div class="note">${T("Nothing found.")}</div>`;
  if (extReg.next) box.insertAdjacentHTML("beforeend", `<p><button type="button" class="act quiet" id="regmore">${T("More")}</button></p>`);
  const mb = $("#regmore");
  if (mb) mb.onclick = () => extRegistry(extReg.q, true);
  for (const b of box.querySelectorAll(".regsrc")) b.onclick = () => window.open(b.dataset.u, "_blank", "noopener");
  for (const b of box.querySelectorAll(".regadd"))
    b.onclick = async () => {
      if (b.dataset.n) { await fetch("/brebus/mcp/remove", {method: "POST", body: JSON.stringify({name: b.dataset.n})}); extChanged(); return; }
      regAdd(extReg.items[+b.dataset.i]);
    };
}
/* a name for mcp.json: the registry's last part, made fit */
const regName = (s) => (s.name.split("/").pop() || s.name).toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^[.-]+/, "").slice(0, 60) || "server";
async function regAdd(s) {
  const w = regWay(s);
  /* what it asks for: its headers, or its environment's variables */
  const asks = w.kind === "url" ? (w.remote.headers || []) : (w.pkg.environmentVariables || []);
  const fields = asks.map((h, k) =>
    `<label>${esc(h.name)}${h.isRequired ? "" : " " + T("(optional)")}
      <input class="rgf" data-k="${k}" type="${h.isSecret ? "password" : "text"}" spellcheck="false" value="${esc(h.value && !/\{[^}]+\}/.test(h.value) ? h.value : h.default || "")}" placeholder="${esc(h.value && /\{[^}]+\}/.test(h.value) ? h.value : "")}">
      ${h.description ? `<span class="note">${esc(h.description)}</span>` : ""}</label>`).join("");
  const cmd = w.kind === "pkg" ? (w.pkg.registryType === "npm" ? `npx -y ${w.pkg.identifier}${w.pkg.version ? "@" + w.pkg.version : ""}` : `uvx ${w.pkg.identifier}${w.pkg.version ? "==" + w.pkg.version : ""}`) : "";
  const d = await dialog({title: T("Add {n}?", {n: s.title || s.name}), wide: true,
    html: `<div class="srvset plainset"><p class="note">${esc(s.description || "")}</p>
      <p class="note">${w.kind === "url" ? T("Its address: {u}", {u: `<code>${esc(w.remote.url)}</code>`}) : T("It runs on this computer as {c}, whenever the model opens it: add it only if you trust its makers. It needs {r} installed.", {c: `<code>${esc(cmd)}</code>`, r: w.pkg.registryType === "npm" ? "Node.js" : "uv"})}</p>
      <p class="note">${T("From MCP's registry: not checked by Janas.")}</p>${fields}</div>`,
    read: (box) => [...box.querySelectorAll(".rgf")].map((i) => i.value),
    buttons: [{label: T("Add"), value: "ok", primary: true}, {label: T("Cancel"), value: "no"}]});
  if (d.button !== "ok") return;
  const vals = d.form || [];
  /* a header's value is its template with {the parts} the user's value */
  const entry = {};
  if (w.kind === "url") {
    entry.url = w.remote.url;
    const h = {};
    asks.forEach((a, k) => {
      const v = vals[k];
      if (!v && !a.isRequired) return;
      /* "Bearer {key}": the key alone, or the whole value as typed */
      const tpl = a.value && /\{[^}]+\}/.test(a.value) ? a.value : "";
      const head = tpl ? tpl.slice(0, tpl.indexOf("{")) : "";
      h[a.name] = tpl && !(head && v.startsWith(head)) ? tpl.replace(/\{[^}]+\}/, v) : v;
    });
    if (Object.keys(h).length) entry.headers = h;
  } else {
    entry.command = w.pkg.registryType === "npm" ? "npx" : "uvx";
    entry.args = w.pkg.registryType === "npm" ? ["-y", w.pkg.identifier + (w.pkg.version ? "@" + w.pkg.version : "")] : [w.pkg.identifier + (w.pkg.version ? "==" + w.pkg.version : "")];
    const env = {};
    asks.forEach((a, k) => { if (vals[k] || a.isRequired) env[a.name] = vals[k] || ""; });
    if (Object.keys(env).length) entry.env = env;
  }
  /* a name not taken */
  const sv = await ask("services").catch(() => ({services: []}));
  let name = regName(s), k = 2;
  while (sv.services.some((x) => x.name === name)) name = regName(s).slice(0, 56) + "-" + k++;
  const r = await fetch("/brebus/mcp/set", {method: "POST", body: JSON.stringify({name, entry})});
  const j = await r.json();
  if (!r.ok) { say(j.error || T("not added"), T("Not added")); return; }
  await extChanged();
  say(T("{n} is added among your servers, in My extensions.", {n: name}), T("Added"));
}
let connTimer = null;
$("#connq").oninput = () => { clearTimeout(connTimer); connTimer = setTimeout(extConnectors, 500); };

/* ---- Skills ---- */
async function extSkills() {
  const q = $("#skq").value.trim();
  let j;
  try { j = await ask("skills"); } catch (e) { return; }
  $("#skdir").innerHTML = T("Instructions for particular tasks, which the model reads when a task needs them: skills in the format of Claude's, a folder with a SKILL.md. Those for every chat are in <code>{dir}</code>, a project's own in its <code>.brebus/skills</code>.", {dir: esc(j.dir)});
  $("#sklist").innerHTML = j.skills.map((s) =>
    `<div class="svc"><label><input type="checkbox" data-n="${esc(s.name)}"${s.on ? " checked" : ""}> <b>${esc(s.name)}</b> <small class="desc" title="${esc(s.description || "")}">${esc((s.description || T("no description")) + (s.project ? " · " + T("this project's own") : ""))}</small></label>` +
    (s.project ? "" : `<button type="button" class="act quiet skrm" data-n="${esc(s.name)}" title="${T("Its folder is deleted from Brebus's skills")}">${T("Remove")}</button>`) + "</div>").join("") ||
    `<div class="note">${T("None yet: install one below.")}</div>`;
  for (const c of document.querySelectorAll("#sklist input"))
    c.onchange = async () => {
      const off = [...document.querySelectorAll("#sklist input")].filter((x) => !x.checked).map((x) => x.dataset.n);
      settings._skills = {off};
      await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
      readTools();
    };
  for (const b of document.querySelectorAll("#sklist .skrm"))
    b.onclick = async () => {
      if (!await yes(T("The skill {n} and its folder are deleted from Brebus's skills.", {n: b.dataset.n}), T("Remove this skill?"), T("Remove"))) return;
      await fetch("/brebus/skills/remove", {method: "POST", body: JSON.stringify({name: b.dataset.n})});
      extSkills(); readTools();
    };
  const have = new Set(j.skills.map((s) => s.name));
  const list = (await extList("skills")).filter((s) => extMatch(q, s.title, s.name, s.description, s.source));
  $("#skrec").innerHTML = list.map((s) =>
    `<div class="svc"><label><b>${esc(s.title || s.name)}</b> <small class="desc" title="${esc(s.description)}">${esc(s.description)}</small>
      <small>${esc(s.source)} · ${esc(s.license)}${s.scripts && s.scripts.length ? " · " + T("with scripts ({k}), run in Brebus's cage", {k: esc(s.scripts.join(", "))}) : ""}</small></label>
      ${have.has(s.name) ? `<span class="badge fits">${T("installed")}</span>` : `<button type="button" class="act skget" data-repo="${esc(s.repo)}" data-ref="${esc(s.ref || "")}" data-path="${esc(s.path)}">${T("Install")}</button>`}</div>`).join("") ||
    `<div class="note">${T("Nothing in Janas's list matches.")}</div>`;
  for (const b of document.querySelectorAll("#skrec .skget")) b.onclick = () => skFetch(b);
}
async function skFetch(b) {
  b.disabled = true; b.textContent = T("Installing…");
  const r = await fetch("/brebus/skills/fetch", {method: "POST", body: JSON.stringify({repo: b.dataset.repo, ref: b.dataset.ref, path: b.dataset.path})});
  const k = await r.json();
  say(r.ok ? T("The skill {n} is added: the model sees it from the next message.", {n: k.name}) : k.error, r.ok ? T("Added") : T("Not added"));
  extSkills(); readTools();
  if (!r.ok) { b.disabled = false; b.textContent = T("Install"); }
}
let skTimer = null;
$("#skq").oninput = () => { clearTimeout(skTimer); skTimer = setTimeout(extSkills, 300); };
$("#skgh").onsubmit = async () => {
  const url = $("#skgh").url.value.trim();
  if (!url) return false;
  const box = $("#skghres");
  box.innerHTML = `<div class="note">${T("Reading the repository…")}</div>`;
  const r = await fetch("/brebus/skills/scan", {method: "POST", body: JSON.stringify({url})});
  const j = await r.json();
  if (!r.ok) { box.innerHTML = `<div class="note">${esc(j.error || T("not read"))}</div>`; return false; }
  box.innerHTML = `<p class="note">${T("{r}: licence {l}. Not checked by Janas: read a skill's scripts before you use it.", {r: esc(j.repo), l: esc(j.license || T("not stated"))})}</p>` +
    (j.skills.map((s) => `<div class="svc"><label><b>${esc(s.name)}</b> <small class="desc" title="${esc(s.description)}">${esc(s.description)}</small>
      <small class="mono">${esc(s.path || "/")} · ${Tn(s.files, "{n} file", "{n} files")}${s.license_file ? " · " + T("its own licence") : ""}</small></label>
      <button type="button" class="act skget" data-repo="${esc(j.repo)}" data-ref="${esc(j.ref)}" data-path="${esc(s.path)}">${T("Install")}</button></div>`).join("") ||
      `<div class="note">${T("No folder with a SKILL.md there.")}</div>`);
  for (const b of box.querySelectorAll(".skget")) b.onclick = () => skFetch(b);
  return false;
};
$("#skbrowse").onclick = async () => {
  const p = await pick(false, T("A skill's folder (with its SKILL.md)"));
  if (p) $("#skadd").path.value = p;
};
$("#skadd").onsubmit = async () => {
  const path = $("#skadd").path.value.trim();
  if (!path) return false;
  const r = await fetch("/brebus/skills/add", {method: "POST", body: JSON.stringify({path})});
  const k = await r.json();
  say(r.ok ? T("The skill {n} is added: the model sees it from the next message.", {n: k.name}) : k.error, r.ok ? T("Added") : T("Not added"));
  if (r.ok) $("#skadd").path.value = "";
  extSkills(); readTools();
  return false;
};
