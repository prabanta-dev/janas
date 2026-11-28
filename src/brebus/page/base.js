/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* base.js - Brebus's page (app.html): its dialogs, the side bar, the models listed. build.sh puts it into
   the page where app.html includes it. */
"use strict";
const $ = (s) => document.querySelector(s);
/* the theme the page is drawn in: theme.js and the page's first script
   keep the class (auto, light or dark) */
const isDark = () => document.documentElement.classList.contains("dark");

/* Brebus's own dialogs, in place of the browser's alert, confirm and
   prompt: a title, a text (or HTML), a field when asked, buttons, a box
   to remember the choice. A promise of {button, value, remember}; Esc is
   the last button, Enter the primary one. window.__dialogAuto, when a
   test sets it, answers in the user's place. */
function dialog(o) {
  if (window.__dialogAuto) return Promise.resolve(window.__dialogAuto(o));
  return new Promise((done) => {
    const box = document.createElement("div");
    box.className = "dlg";
    const btns = o.buttons || [{label: T("OK"), value: "ok", primary: true}];
    const button = (b, i) => `<button class="act${b.primary ? "" : " quiet"}" data-i="${i}" title="${esc(b.title || (b.primary ? T("{label} (Enter)", {label: b.label}) : i === btns.length - 1 ? T("{label} (Esc)", {label: b.label}) : b.label))}">${esc(b.label)}</button>`;
    /* a button "beside" goes next to the field (Browse…), the others at the
       foot */
    const beside = btns.map((b, i) => b.beside && o.input !== undefined ? button(b, i) : "").join("");
    const field = o.input === undefined ? "" : o.multiline
      ? `<textarea class="din" rows="4" spellcheck="false">${esc(o.input)}</textarea>`
      : `<input class="din" spellcheck="false" value="${esc(o.input)}" placeholder="${esc(o.placeholder || "")}">`;
    box.innerHTML = `<div class="box${o.wide ? " wide" : ""}" role="dialog" aria-modal="true">
      ${o.title ? `<h2>${esc(o.title)}</h2>` : ""}
      <div class="dtext">${o.html || esc(o.text || "").replace(/\n/g, "<br>")}</div>
      ${beside ? `<div class="dinrow">${field}${beside}</div>` : field}
      ${o.remember ? `<label class="note drem"><input type="checkbox"> ${esc(o.remember)}</label>` : ""}
      <p class="right">${btns.map((b, i) => b.beside && o.input !== undefined ? "" : button(b, i)).join(" ")}</p>
    </div>`;
    document.body.append(box);
    /* the buttons at the foot all as wide as the widest */
    const foot = [...box.querySelectorAll("p.right button")];
    const w = Math.max(0, ...foot.map((b) => b.getBoundingClientRect().width));
    for (const b of foot) b.style.minWidth = Math.ceil(w) + "px";
    const fieldEl = box.querySelector(".din");
    const end = (b) => {
      const res = {button: b.value, value: fieldEl ? fieldEl.value : undefined,
                   remember: !!box.querySelector(".drem input")?.checked,
                   form: o.read ? o.read(box) : undefined};
      box.remove(); document.removeEventListener("keydown", key, true);
      done(res);
    };
    const key = (e) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); end(btns[btns.length - 1]); }
      /* Enter: the main button from a field; on a button, that button (the
         browser's own click) - never the main one, as Cancel or Don't allow
         would then do what they refuse */
      else if (e.key === "Enter" && !e.shiftKey && !(fieldEl && fieldEl.tagName === "TEXTAREA") &&
               !(e.target instanceof Element && e.target.closest("button, a, select, summary"))) {
        e.preventDefault(); end(btns.find((b) => b.primary) || btns[0]);
      }
    };
    document.addEventListener("keydown", key, true);
    for (const b of box.querySelectorAll("button[data-i]")) b.onclick = () => end(btns[+b.dataset.i]);
    if (o.ready) o.ready(box);
    (fieldEl || box.querySelector("button.act:not(.quiet)") || box.querySelector("button"))?.focus();
  });
}
/* Prabanta's five dots, lit in turn (app.css): Brebus working */
const PINTA = '<svg class="pinta" viewBox="0 0 24 24" width="26" height="26" aria-hidden="true"><circle cx="12.00" cy="4.00" r="2.3"/><circle cx="19.61" cy="9.53" r="2.3"/><circle cx="16.70" cy="18.47" r="2.3"/><circle cx="7.30" cy="18.47" r="2.3"/><circle cx="4.39" cy="9.53" r="2.3"/></svg>';
window.markdownAsk = (text) => yes(text, T("A link the model wrote"), T("Open it"));
/* Brebus's window closing in order (main.c): what it is doing, over the
   page, which can no longer be used; the reply waited for stopped here */
window.brebusClosing = (step) => {
  let box = document.getElementById("closing");
  if (!box) {
    if (typeof stopReply === "function") stopReply();
    box = document.createElement("div");
    box.id = "closing";
    box.className = "dlg";
    box.innerHTML = `<div class="box" role="alertdialog" aria-live="polite"><h2>${T("Closing Brebus")}</h2>
      <p class="dtext"><span class="working">${PINTA}</span> <span id="closingstep"></span></p></div>`;
    document.body.append(box);
  }
  /* the step as main.c says it (close_say), in the page's language */
  document.getElementById("closingstep").textContent = T(step);
};
const say = (text, title) => dialog({title, text, buttons: [{label: T("OK"), value: "ok", primary: true}]});
const yes = async (text, title, ok = T("OK")) => (await dialog({title, text,
  buttons: [{label: ok, value: "yes", primary: true}, {label: T("Cancel"), value: "no"}]})).button === "yes";
const input = async (title, value, text, multiline) => {
  const r = await dialog({title, text, input: value || "", multiline,
    buttons: [{label: T("OK"), value: "ok", primary: true}, {label: T("Cancel"), value: "no"}]});
  return r.button === "ok" ? r.value : null;
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
const gib = (b) => (b / 2 ** 30).toFixed(1) + " GiB";
async function ask(what) {
  const r = await fetch("/brebus/" + what);
  if (!r.ok) throw new Error(what + ": " + r.status);
  return r.json();
}
function show(id) {
  for (const s of document.querySelectorAll("section"))
    s.classList.toggle("on", s.id === id);
  for (const b of document.querySelectorAll("nav button[data-s]"))
    b.setAttribute("aria-current", b.dataset.s === id ? "page" : "false");
  try { localStorage.setItem("brebus.section", id); } catch (e) {}
}
for (const b of document.querySelectorAll("nav button[data-s]"))
  b.addEventListener("click", () => { show(b.dataset.s); if (b.dataset.s === "machine") machineNow(); });
/* Info's account, when Brebus is served with a login (404 in its own
   window, where there is none) */
async function account() {
  let me;
  try { me = await ask("auth/me"); } catch (e) { return; }
  if (!me || !me.user) return;
  $("#acct").hidden = false;
  $("#navout").hidden = false;
  $("#acctme").innerHTML = `<dt>${T("Logged in as")}</dt><dd>${me.admin ? T("{user} (administrator)", {user: esc(me.user)}) : esc(me.user)}</dd>`;
  try {
    const l = (await ask("auth/logins")).logins || [];
    $("#logins").innerHTML = l.slice(0, 10).map((x) => `<li class="mono">${esc(x)}</li>`).join("");
  } catch (e) {}
}
$("#navout").onclick = () => $("#logout").onclick();
$("#logout").onclick = async () => {
  await fetch("/brebus/auth/logout", {method: "POST", body: "{}"});
  location.replace("/login");
};
$("#endall").onclick = async () => {
  const r = await fetch("/brebus/auth/end", {method: "POST", body: "{}"});
  say(r.ok ? T("Your other sessions are ended.") : T("Not done."));
  account();
};
$("#newpw").onclick = async () => {
  const d = await dialog({title: T("Change the password"),
    html: `<label>${T("The password now")} <input type="password" class="pw0" autocomplete="current-password"></label>
           <label>${T("The new one, 12 characters or more")} <input type="password" class="pw1" autocomplete="new-password"></label>
           <label>${T("The new one again")} <input type="password" class="pw2" autocomplete="new-password"></label>
           <p class="note">${T("Your sessions in other browsers end.")}</p>`,
    read: (box) => ({old: box.querySelector(".pw0").value, a: box.querySelector(".pw1").value, b: box.querySelector(".pw2").value}),
    buttons: [{label: T("Change it"), value: "ok", primary: true}, {label: T("Cancel"), value: "no"}]});
  if (d.button !== "ok") return;
  if (d.form.a !== d.form.b) { say(T("The two new passwords differ.")); return; }
  const r = await fetch("/brebus/auth/password", {method: "POST", body: JSON.stringify({old: d.form.old, new: d.form.a})});
  const j = await r.json().catch(() => ({}));
  say(r.ok ? T("Password changed.") : j.error || T("Not changed."));
  account();
};
/* Info's live lines: the memory free now, the model loaded */
async function machineNow() {
  account();
  try {
    const m = await ask("machine");
    await status();
    /* what the engine chose for the model loaded, measuring this computer */
    let chose = "";
    if (server.running) try { chose = (await ask("server/choices")).choices; } catch (e) {}
    $("#mach").innerHTML = [
      [T("Processor"), m.cpu || T("unknown")],
      [T("Instructions"), m.features || T("none known")],
      [T("Memory"), T("{total}, {free} free now", {total: gib(m.memory_total), free: gib(m.memory_available)})],
      [T("Power"), m.mains === 1 ? T("mains") : m.mains === 0 ? T("battery") : T("unknown")],
      [T("Model loaded"), server.running ? server.model : T("none")],
      ...(chose ? [[T("How it runs"), chose]] : []),
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  } catch (e) {}
}
/* a model started or stopped from elsewhere (another program, the
   server's API): seen within a few seconds, while the page is in view */
setInterval(() => { if (!document.hidden && !busy && !modelLoading) status().catch(() => {}); }, 8000);
/* a dialog in front: the page's own keys (Ctrl+B, Ctrl+K, Ctrl+Shift+O)
   wait for it */
const dialogOpen = () => !!document.querySelector('[role="dialog"][aria-modal="true"]') ||
                         !$("#wsdlg").hidden || !$("#psdlg").hidden;
/* the side bar: whole, or a strip of icons (its button, Ctrl+B), as left */
let navAuto = false; /* hidden by an artifact opened, shown again when it closes */
function navMini(on, remember) {
  $("#side").classList.toggle("mini", on);
  $("#navtog").title = on ? T("Show the side bar (Ctrl+B)") : T("Hide the side bar (Ctrl+B)");
  $("#navtog").setAttribute("aria-label", $("#navtog").title);
  if (remember) { navAuto = false; try { localStorage.setItem("brebus.mini", on ? "1" : ""); } catch (e) {} }
}
try { navMini(localStorage.getItem("brebus.mini") === "1", false); } catch (e) {}
$("#navtog").onclick = () => navMini(!$("#side").classList.contains("mini"), true);
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "b" && !dialogOpen()) {
    e.preventDefault();
    navMini(!$("#side").classList.contains("mini"), true);
  }
});
/* Brebus opens on its chat, whatever page was shown last: a new one
   (more.js), in the project given by --workspace if any; an address with
   #models and the like still opens that page */
let first = "chat";
const hashChat = (location.hash.match(/^#chat\/([\w-]+)$/) || [])[1];
if (location.hash.length > 1) first = location.hash.slice(1).split("/")[0];
show(document.getElementById(first) ? first : "discover");

let machineMem = 0;
async function load() {
  const m = await ask("machine");
  machineMem = m.memory_total;
  $("#aboutver").textContent = m.version;
  $("#mem").textContent = T("This computer: {mem}.", {mem: gib(m.memory_total)});
  try { pyState = await ask("python"); } catch (e) {}
  $("#mach").innerHTML = [
    [T("Processor"), m.cpu || T("unknown")],
    [T("Instructions"), m.features || T("none known")],
    [T("Memory"), T("{total}, {free} available", {total: gib(m.memory_total), free: gib(m.memory_available)})],
    [T("Power"), m.mains === 1 ? T("mains") : m.mains === 0 ? T("battery") : T("unknown")],
  ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  /* Settings: the start with the computer, Python where it is downloaded */
  $("#autost").checked = !!m.autostart;
  $("#pyset").hidden = !pyState.needed;
  $("#pyline").innerHTML = pyState.installed
    ? `<span class="btnline"><span>${T("Python {v} is installed for Run code, downloaded from python.org with numpy and matplotlib.", {v: esc(pyState.version)})}</span> <button class="act quiet" id="pyrm" title="${esc(T("Delete the downloaded Python; Run code offers it again"))}">${T("Remove it")}</button></span>`
    : T("Not installed: turning on Run code offers to download it (about 45 MB, from python.org and PyPI), and asks first.");
  if ($("#pyrm")) $("#pyrm").onclick = async () => {
    const r = await fetch("/brebus/python/remove", {method: "POST", body: "{}"});
    pyState = await r.json();
    await readTools(); load();
  };
  $("#autost").onchange = async (e) => {
    const r = await fetch("/brebus/autostart", {method: "POST", body: JSON.stringify({on: e.target.checked})});
    const j = await r.json();
    e.target.checked = j.autostart;
  };
  await mine();
  await status();
  await getStatus();
}
let catalog = null, modelInfo = {}, speechModels = [];
/* el[prop] = v only when it differs (true when written): what is under the
   pointer keeps its tooltip and its hover. innerHTML is compared with the
   last string written, as the browser rewrites it its own way */
function put(el, prop, v) {
  if (!el) return false;
  if (prop === "innerHTML") {
    if (el.dataset.put === v) return false;
    el.innerHTML = v;
    el.dataset.put = v;
    return true;
  }
  if (el[prop] === v) return false;
  el[prop] = v;
  return true;
}

async function mine() {
  const c = catalog = catalog || await ask("catalog");
  const my = await ask("models");
  /* the models as they are now: one removed is gone from every list */
  modelInfo = {};
  for (const x of my.models) modelInfo[x.name] = x;
  speechModels = my.speech || [];
  if (typeof drawVoice === "function") drawVoice();
    try { settings = await ask("settings"); } catch (e) { settings = {}; }
  srvForm();
  if (typeof drawExt === "function") drawExt();
  const have = new Set(my.models.map((x) => x.path.split("/").pop()));
  $("#cat").innerHTML = c.models.filter((x) => !x.test_only).map((x) =>
    `<tr><td><b>${esc(x.name)}</b><br><span class="mono" style="color:var(--dim)">${esc(x.file)}</span></td>
     <td>${esc(x.what)}</td><td class="num">${x.download_gb.toFixed(1)} GB</td>
     <td class="num">${x.ram_gb} GB</td>
     <td class="tag">${x.fits ? '<span class="badge fits">' + T("fits") + '</span>' : '<span class="badge big">' + T("needs more") + '</span>'}</td>
     <td class="tag">${have.has(x.file) ? '<span class="badge">' + T("installed") + '</span>'
       : `<button class="act quiet get" data-m="${esc(x.name)}" title="${esc(T("Download it into the models' folder, checked against its published fingerprint"))}">${T("Get")}</button>`}</td></tr>`
  ).join("") || '<tr><td colspan="6" class="empty">' + T("None.") + '</td></tr>';
  for (const b of document.querySelectorAll("button.get"))
    b.onclick = () => getModel(b.dataset.m);
  $("#dir").value = my.dir;
  $("#mine").innerHTML = my.models.map((x) =>
    `<tr><td><b>${esc(x.name)}</b></td>
     <td>${esc(x.arch)}${x.recurrent ? " · " + T("recurrent") : ""}${x.fim ? " · " + T("fills in the middle") : ""}${x.vision ? " · " + T("sees pictures") : ""}</td>
     <td class="num">${gib(x.bytes)}</td><td class="num">${x.context ? num(x.context) : "—"}</td>
     <td class="row-acts"><button class="act load" data-m="${esc(x.name)}" title="${esc(T("Load it: the chat and other programs use it"))}">${T("Load")}</button>
       <button class="act quiet opts" data-m="${esc(x.name)}" title="${esc(T("How it runs: how much it remembers, whether it thinks first"))}" aria-pressed="false">${T("Settings")}</button>
       <button class="act quiet rep" data-m="${esc(x.name)}" title="${esc(T("Tell Janas's makers that this model does not work fully with Janas"))}">${T("Report")}</button>
       <button class="act quiet rmm" data-m="${esc(x.name)}" title="${esc(T("Delete it from this computer"))}">${T("Remove")}</button></td></tr>
     <tr class="set" data-m="${esc(x.name)}" hidden><td colspan="5">${setForm(x)}</td></tr>`
  ).join("") || '<tr><td colspan="5" class="empty">' + T("None yet: get one in Discover, or with <code>janas-get</code>.") + '</td></tr>';
  for (const b of document.querySelectorAll("button.load"))
    b.onclick = () => loadModel(b.dataset.m, b);
  for (const b of document.querySelectorAll("button.rep"))
    b.onclick = () => reportModel(b.dataset.m);
  for (const b of document.querySelectorAll("button.rmm"))
    b.onclick = async () => {
      const x = modelInfo[b.dataset.m];
      if (server.running && server.model === b.dataset.m) {
        say(T("{name} is the model in use: choose another one under the chat's field first.", {name: b.dataset.m}));
        return;
      }
      const v = {name: b.dataset.m, size: gib(x.bytes)};
      const q = isLoaded(b.dataset.m)
        ? T("{name} ({size}) is deleted from this computer; it can be got again in Discover. It is loaded now: it is unloaded first.", v)
        : T("{name} ({size}) is deleted from this computer; it can be got again in Discover.", v);
      if (!await yes(q, T("Remove this model?"), T("Remove"))) return;
      const r = await fetch("/brebus/models/remove", {method: "POST", body: JSON.stringify({name: b.dataset.m})});
      const j = await r.json();
      if (!r.ok) say(j.error || T("not removed"));
      await mine(); await status();
    };
  for (const b of document.querySelectorAll("button.opts"))
    b.onclick = () => {
      const t = document.querySelector(`tr.set[data-m="${CSS.escape(b.dataset.m)}"]`);
      t.hidden = !t.hidden;
      /* open: the button says so, and how to close them */
      b.setAttribute("aria-pressed", t.hidden ? "false" : "true");
      b.textContent = t.hidden ? T("Settings") : T("Close");
      b.title = t.hidden ? T("How it runs: how much it remembers, whether it thinks first") : T("Close this model's settings");
    };
  for (const f of document.querySelectorAll("tr.set form"))
    f.onchange = () => saveSettings(f);
}

/* the account, when Brebus is served: its Log out in the side bar at once */
account().catch(() => {});

/* a model that does not work fully with Janas, reported: the report
   written here, shown whole, and opened on GitHub's page for a new issue,
   where the user reads it again and sends it, or not. Brebus sends
   nothing itself */
async function reportModel(name) {
  const x = modelInfo[name] || {};
  const m = await ask("machine").catch(() => ({}));
  /* what Janas found it cannot do, when the model is the one loaded */
  let limits = null;
  if (server.running && server.model === name) {
    const r = await fetch("/v1/models").then((r) => r.json()).catch(() => ({}));
    const d = (r.data || []).find((d) => d.limits);
    if (d) limits = d.limits;
  }
  const facts = [
    "Model: " + name,
    "File: " + (x.path || "").split(/[\\/]/).pop(),
    "Architecture: " + (x.arch || "?") + (x.recurrent ? " (recurrent)" : ""),
    "Size: " + gib(x.bytes || 0) + ", context " + (x.context || "?"),
    "Brebus: " + (m.version || "?") + ", " + navigator.platform,
    "CPU: " + (m.cpu || "?"),
    limits === null ? "What Janas found: (the model was not loaded)"
      : limits.length ? "What Janas found:\n" + limits.map((l) => "- " + l).join("\n")
      : "What Janas found: nothing"].join("\n");
  const d = await dialog({title: T("Report a problem with {name}", {name}),
    html: `<div class="srvset plainset">
      <label>${T("What does not work")}
        <textarea class="what" rows="5" placeholder="${T("For example: it answers in the wrong language; its tool calls show as text; it never stops.")}"></textarea></label>
      <label>${T("Sent with it")}<textarea class="facts mono" rows="8" readonly>${esc(facts)}</textarea>
        <span class="note">${T("Nothing is sent from here: GitHub's page for a new issue opens with this written in it, and you send it there, after reading it again, with your GitHub account.")}</span></label></div>`,
    read: (box) => ({what: box.querySelector(".what").value.trim()}),
    buttons: [{label: T("Open it on GitHub"), value: "ok", primary: true}, {label: T("Cancel"), value: "no"}]});
  if (d.button !== "ok") return;
  const body = (d.form.what || "(describe what does not work)") + "\n\n```\n" + facts + "\n```\n";
  const url = "https://github.com/prabanta-dev/janas/issues/new?title=" +
    encodeURIComponent("[model] " + name + ": ") + "&body=" + encodeURIComponent(body);
  window.open(url, "_blank", "noopener");
}
