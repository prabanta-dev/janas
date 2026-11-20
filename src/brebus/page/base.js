/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* base.js - Brebus's page (app.html): its dialogs, the side bar, the models listed. build.sh puts it into
   the page where app.html includes it. */
"use strict";
const $ = (s) => document.querySelector(s);

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
    const btns = o.buttons || [{label: "OK", value: "ok", primary: true}];
    box.innerHTML = `<div class="box${o.wide ? " wide" : ""}" role="dialog" aria-modal="true">
      ${o.title ? `<h2>${esc(o.title)}</h2>` : ""}
      <div class="dtext">${o.html || esc(o.text || "").replace(/\n/g, "<br>")}</div>
      ${o.input !== undefined ? (o.multiline
        ? `<textarea class="din" rows="4" spellcheck="false">${esc(o.input)}</textarea>`
        : `<input class="din" spellcheck="false" value="${esc(o.input)}" placeholder="${esc(o.placeholder || "")}">`) : ""}
      ${o.remember ? `<label class="note drem"><input type="checkbox"> ${esc(o.remember)}</label>` : ""}
      <p class="right">${btns.map((b, i) => `<button class="act${b.primary ? "" : " quiet"}" data-i="${i}" title="${esc(b.title || (b.primary ? b.label + " (Enter)" : i === btns.length - 1 ? b.label + " (Esc)" : b.label))}">${esc(b.label)}</button>`).join(" ")}</p>
    </div>`;
    document.body.append(box);
    const field = box.querySelector(".din");
    const end = (b) => {
      const res = {button: b.value, value: field ? field.value : undefined,
                   remember: !!box.querySelector(".drem input")?.checked,
                   form: o.read ? o.read(box) : undefined};
      box.remove(); document.removeEventListener("keydown", key, true);
      done(res);
    };
    const key = (e) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); end(btns[btns.length - 1]); }
      else if (e.key === "Enter" && !e.shiftKey && !(field && field.tagName === "TEXTAREA")) {
        e.preventDefault(); end(btns.find((b) => b.primary) || btns[0]);
      }
    };
    document.addEventListener("keydown", key, true);
    for (const b of box.querySelectorAll("button[data-i]")) b.onclick = () => end(btns[+b.dataset.i]);
    if (o.ready) o.ready(box);
    (field || box.querySelector("button.act:not(.quiet)") || box.querySelector("button"))?.focus();
  });
}
window.markdownAsk = (text) => yes(text, "A link the model wrote", "Open it");
const say = (text, title) => dialog({title, text, buttons: [{label: "OK", value: "ok", primary: true}]});
const yes = async (text, title, ok = "OK") => (await dialog({title, text,
  buttons: [{label: ok, value: "yes", primary: true}, {label: "Cancel", value: "no"}]})).button === "yes";
const input = async (title, value, text, multiline) => {
  const r = await dialog({title, text, input: value || "", multiline,
    buttons: [{label: "OK", value: "ok", primary: true}, {label: "Cancel", value: "no"}]});
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
  $("#acctme").innerHTML = `<dt>Logged in as</dt><dd>${esc(me.user)}${me.admin ? " (administrator)" : ""}</dd>`;
  try {
    const l = (await ask("auth/logins")).logins || [];
    $("#logins").innerHTML = l.slice(0, 10).map((x) => `<li class="mono">${esc(x)}</li>`).join("");
  } catch (e) {}
}
$("#logout").onclick = async () => {
  await fetch("/brebus/auth/logout", {method: "POST", body: "{}"});
  location.replace("/login");
};
$("#endall").onclick = async () => {
  const r = await fetch("/brebus/auth/end", {method: "POST", body: "{}"});
  say(r.ok ? "Your other sessions are ended." : "Not done.");
  account();
};
$("#newpw").onclick = async () => {
  const d = await dialog({title: "Change the password",
    html: `<label>The password now <input type="password" class="pw0" autocomplete="current-password"></label>
           <label>The new one, 12 characters or more <input type="password" class="pw1" autocomplete="new-password"></label>
           <label>The new one again <input type="password" class="pw2" autocomplete="new-password"></label>
           <p class="note">Your sessions in other browsers end.</p>`,
    read: (box) => ({old: box.querySelector(".pw0").value, a: box.querySelector(".pw1").value, b: box.querySelector(".pw2").value}),
    buttons: [{label: "Change it", value: "ok", primary: true}, {label: "Cancel", value: "no"}]});
  if (d.button !== "ok") return;
  if (d.form.a !== d.form.b) { say("The two new passwords differ."); return; }
  const r = await fetch("/brebus/auth/password", {method: "POST", body: JSON.stringify({old: d.form.old, new: d.form.a})});
  const j = await r.json().catch(() => ({}));
  say(r.ok ? "Password changed." : j.error || "Not changed.");
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
      ["Processor", m.cpu || "unknown"],
      ["Instructions", m.features || "none known"],
      ["Memory", gib(m.memory_total) + ", " + gib(m.memory_available) + " free now"],
      ["Power", m.mains === 1 ? "mains" : m.mains === 0 ? "battery" : "unknown"],
      ["Model loaded", server.running ? server.model : "none"],
      ...(chose ? [["How it runs", chose]] : []),
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  } catch (e) {}
}
/* a model started or stopped from elsewhere (another program, the
   server's API): seen within a few seconds, while the page is in view */
setInterval(() => { if (!document.hidden && !busy && !modelLoading) status().catch(() => {}); }, 8000);
/* the side bar: whole, or a strip of icons (its button, Ctrl+B), as left */
let navAuto = false; /* hidden by an artifact opened, shown again when it closes */
function navMini(on, remember) {
  $("#side").classList.toggle("mini", on);
  $("#navtog").title = (on ? "Show" : "Hide") + " the side bar (Ctrl+B)";
  $("#navtog").setAttribute("aria-label", $("#navtog").title);
  if (remember) { navAuto = false; try { localStorage.setItem("brebus.mini", on ? "1" : ""); } catch (e) {} }
}
try { navMini(localStorage.getItem("brebus.mini") === "1", false); } catch (e) {}
$("#navtog").onclick = () => navMini(!$("#side").classList.contains("mini"), true);
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "b") {
    e.preventDefault();
    navMini(!$("#side").classList.contains("mini"), true);
  }
});
/* Brebus opens on its chat, whatever page was shown last: the last chat
   (more.js), or a new one; an address with #models and the like still
   opens that page */
let first = "chat";
const hashChat = (location.hash.match(/^#chat\/([\w-]+)$/) || [])[1];
if (location.hash.length > 1) first = location.hash.slice(1).split("/")[0];
show(document.getElementById(first) ? first : "discover");

let machineMem = 0;
async function load() {
  const m = await ask("machine");
  machineMem = m.memory_total;
  $("#aboutver").textContent = m.version;
  $("#mem").textContent = "This computer: " + gib(m.memory_total) + ".";
  try { pyState = await ask("python"); } catch (e) {}
  $("#mach").innerHTML = [
    ["Processor", m.cpu || "unknown"],
    ["Instructions", m.features || "none known"],
    ["Memory", gib(m.memory_total) + ", " + gib(m.memory_available) + " available"],
    ["Power", m.mains === 1 ? "mains" : m.mains === 0 ? "battery" : "unknown"],
  ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  /* Settings: the start with the computer, Python where it is downloaded */
  $("#autost").checked = !!m.autostart;
  $("#pyset").hidden = !pyState.needed;
  $("#pyline").innerHTML = pyState.installed
    ? `Python ${esc(pyState.version)} is installed for Run code, downloaded from python.org with numpy and matplotlib. <button class="act quiet" id="pyrm" title="Delete the downloaded Python; Run code offers it again">Remove it</button>`
    : "Not installed: turning on Run code offers to download it (about 45 MB, from python.org and PyPI), and asks first.";
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
async function mine() {
  const c = catalog = catalog || await ask("catalog");
  const my = await ask("models");
  for (const x of my.models) modelInfo[x.name] = x;
  speechModels = my.speech || [];
  if (typeof drawVoice === "function") drawVoice();
    try { settings = await ask("settings"); } catch (e) { settings = {}; }
  srvForm();
  if (typeof drawSkills === "function") drawSkills();
  if (typeof drawConnectors === "function") drawConnectors();
  const have = new Set(my.models.map((x) => x.path.split("/").pop()));
  $("#cat").innerHTML = c.models.filter((x) => !x.test_only).map((x) =>
    `<tr><td><b>${esc(x.name)}</b><br><span class="mono" style="color:var(--dim)">${esc(x.file)}</span></td>
     <td>${esc(x.what)}</td><td class="num">${x.download_gb.toFixed(1)} GB</td>
     <td class="num">${x.ram_gb} GB</td>
     <td class="tag">${x.fits ? '<span class="badge fits">fits</span>' : '<span class="badge big">needs more</span>'}</td>
     <td class="tag">${have.has(x.file) ? '<span class="badge">installed</span>'
       : `<button class="act quiet get" data-m="${esc(x.name)}" title="Download it into the models' folder, checked against its published fingerprint">Get</button>`}</td></tr>`
  ).join("") || '<tr><td colspan="6" class="empty">None.</td></tr>';
  for (const b of document.querySelectorAll("button.get"))
    b.onclick = () => getModel(b.dataset.m);
  $("#dir").value = my.dir;
  $("#mine").innerHTML = my.models.map((x) =>
    `<tr><td><b>${esc(x.name)}</b></td>
     <td>${esc(x.arch)}${x.recurrent ? " · recurrent" : ""}${x.fim ? " · fills in the middle" : ""}${x.vision ? " · sees pictures" : ""}</td>
     <td class="num">${gib(x.bytes)}</td><td class="num">${x.context ? x.context.toLocaleString() : "—"}</td>
     <td class="row-acts"><button class="act load" data-m="${esc(x.name)}" title="Load it: the chat and other programs use it">Load</button>
       <button class="act quiet opts" data-m="${esc(x.name)}" title="How it runs: how much it remembers, whether it thinks first">Settings</button>
       <button class="act quiet rmm" data-m="${esc(x.name)}" title="Delete it from this computer">Remove</button></td></tr>
     <tr class="set" data-m="${esc(x.name)}" hidden><td colspan="5">${setForm(x)}</td></tr>`
  ).join("") || '<tr><td colspan="5" class="empty">None yet: get one in Discover, or with <code>janas-get</code>.</td></tr>';
  for (const b of document.querySelectorAll("button.load"))
    b.onclick = () => loadModel(b.dataset.m, b);
  for (const b of document.querySelectorAll("button.rmm"))
    b.onclick = async () => {
      const x = modelInfo[b.dataset.m];
      if (!await yes(`${b.dataset.m} (${gib(x.bytes)}) is deleted from this computer; it can be got again in Discover.`, "Remove this model?", "Remove")) return;
      const r = await fetch("/brebus/models/remove", {method: "POST", body: JSON.stringify({name: b.dataset.m})});
      const j = await r.json();
      if (!r.ok) say(j.error || "not removed");
      await mine(); await status();
    };
  for (const b of document.querySelectorAll("button.opts"))
    b.onclick = () => { const t = document.querySelector(`tr.set[data-m="${CSS.escape(b.dataset.m)}"]`); t.hidden = !t.hidden; };
  for (const f of document.querySelectorAll("tr.set form"))
    f.onchange = () => saveSettings(f);
}

