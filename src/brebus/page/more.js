/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* more.js - Brebus's page (app.html): edit a message, services, folders, settings, the workspace, the start. build.sh puts it into
   the page where app.html includes it. */
/* ---- a message edited; a reply's versions (written again by a Brebus
   before, whose button is gone: their arrows still page through them) ---- */
const variantOf = (m) => ({content: m.content, think: m.think, speed: m.speed, saved: m.saved,
                           tool_calls: m.tool_calls});
/* one of the user's messages changed: the conversation goes on from it */
async function editMessage(k) {
  if (busy) return;
  const m = cur.messages[k];
  const t = await input(T("Edit the message"), m.content, T("The conversation after it is replaced by the new answer."), true);
  if (t === null || !t.trim()) return;
  if (!await ensureModel()) return;
  m.content = t.trim();
  delete m.found;
  cur.messages = cur.messages.slice(0, k + 1);
  drawLog();
  await runTurn(m);
}
function showVariant(m, vi) {
  const v = m.variants[vi];
  Object.assign(m, {content: v.content, think: v.think, speed: v.speed, saved: v.saved, tool_calls: v.tool_calls});
  m.vi = vi;
  drawLog(); saveConv(cur);
}

/* Janas's services and the user's MCP servers: on or off */
async function readServices() {
  const j = await ask("services");
  $("#wssvc").hidden = !ws.path;
  $("#svcnote").innerHTML = T("The model opens one when a question needs it. Your own MCP servers go in <code>{config}</code>, in the format of Cursor and LM Studio.", {config: esc(j.config)});
  /* a server of yours over HTTP that asks for a sign-in (OAuth) has its
     button: the authorization server's page opens in the browser */
  const auth = (x) => !x.http ? "" : x.signed_in
    ? `<button type="button" class="act quiet svcauth" data-n="${esc(x.name)}" data-out="1" title="${T("Its token is forgotten here")}">${T("Sign out")}</button>`
    : `<button type="button" class="act svcauth" data-n="${esc(x.name)}" title="${T("Its sign-in page opens in the browser")}">${T("Sign in")}</button>`;
  $("#svclist").innerHTML = j.services.map((x) =>
    `<div class="svc"><label><input type="checkbox" data-n="${esc(x.name)}"${x.on ? " checked" : ""}> <b>${esc(x.user ? x.name : svcLabel(x.name))}</b> <small>${esc((x.user ? x.where || T("A server of yours, from mcp.json.") : svcAbout(x.name, x.about)) + (!x.http ? "" : x.signed_in ? " · " + T("signed in") : x.sign_in ? " · " + T("it asks you to sign in") : ""))}</small></label>${auth(x)}</div>`).join("") ||
    `<div class="note">${T("none found beside Brebus")}</div>`;
  /* how long an approval of a chat lasts, and taking the chat's back */
  const age = $("#apprage");
  age.value = String(settings._approvals ? settings._approvals.age : 86400);
  age.onchange = async () => {
    settings._approvals = {age: +age.value};
    await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
  };
  $("#apprclear").disabled = !cur || !cur.approvals || !Object.keys(cur.approvals).length;
  $("#apprclear").onclick = async () => {
    if (!cur) return;
    delete cur.approvals; await saveConv(cur);
    $("#apprclear").disabled = true; say(T("The tools of this chat ask again."));
  };
  for (const b of document.querySelectorAll("#svclist .svcauth"))
    b.onclick = () => b.dataset.out ? signOut(b.dataset.n) : signIn(b.dataset.n);
  for (const c of document.querySelectorAll("#svclist input"))
    c.onchange = async () => {
      const off = [...document.querySelectorAll("#svclist input")].filter((x) => !x.checked).map((x) => x.dataset.n);
      settings._services = {off};
      await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
      await fetch("/brebus/tools/restart", {method: "POST", body: "{}"});
      await readTools();
    };
}

async function signIn(name) {
  const r = await fetch("/brebus/oauth/begin", {method: "POST",
    body: JSON.stringify({name, redirect: location.origin + "/brebus/oauth/callback"})});
  const j = await r.json();
  if (!r.ok) { say(j.error || T("the sign-in did not start"), T("Not signed in")); return; }
  window.open(j.url, "_blank"); /* Brebus's window gives it to the browser */
  say(T("Sign in to {name} in the page just opened; this list shows it when it is done.", {name}), T("Sign in"));
  /* the list follows the sign-in, for five minutes at most */
  for (let i = 0; i < 150; i++) {
    await new Promise((ok) => setTimeout(ok, 2000));
    const x = (await ask("services")).services.find((v) => v.name === name);
    if (x && x.signed_in) { await readServices(); say(T("Signed in to {name}.", {name}), T("Signed in")); return; }
  }
}
async function signOut(name) {
  if (!await yes(T("Brebus forgets the token of {name}; the next time it will ask you to sign in again.", {name}), T("Sign out?"), T("Sign out"))) return;
  await fetch("/brebus/oauth/forget", {method: "POST", body: JSON.stringify({name})});
  await readServices();
}

/* the folders read only beside the workspace */
async function readFolders() {
  const f = await ask("folders");
  $("#wsro").hidden = !ws.path;
  $("#rolist").innerHTML = f.read_only.length
    ? f.read_only.map((p, i) => `<div><span>${esc(p)}</span><button data-i="${i}" title="${T("The model no longer reads this folder (nothing in it is deleted)")}">${T("remove")}</button></div>`).join("")
    : `<div class="note">${T("no folder")}</div>`;
  for (const b of document.querySelectorAll("#rolist button"))
    b.onclick = async () => { f.read_only.splice(+b.dataset.i, 1); await setFolders(f.read_only); };
  return f;
}
async function setFolders(list) {
  const r = await fetch("/brebus/folders", {method: "POST", body: JSON.stringify({read_only: list})});
  const j = await r.json();
  if (j.refused) say(T("Some were not added: a folder or file there is, by its full path."));
  await readFolders(); await readTools();
}
$("#embsel").onchange = async () => {
  settings._docs = {embedding: $("#embsel").value};
  await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
  await fetch("/brebus/tools/restart", {method: "POST", body: "{}"});
  await readTools();
};
$("#wsbrowse").onclick = async () => {
  const p = await pick(false, T("The folder Brebus works in"));
  if (p) wsOpen(p, false);
};
$("#robrowse").onclick = async () => {
  const p = await pick(false, T("A folder to read only, beside the workspace"));
  if (!p) return;
  const f = await ask("folders");
  await setFolders([...f.read_only, p]);
};
$("#roform").onsubmit = async () => {
  const p = $("#roform").path.value.trim();
  if (!p) return false;
  const f = await ask("folders");
  await setFolders([...f.read_only, p]);
  $("#roform").path.value = "";
  return false;
};

$("#newc").onclick = () => newFree();
$("#form").onsubmit = sendAsk;
$("#ask").addEventListener("input", growAsk);
/* Enter sends, Shift+Enter a new line; an IME's Enter composes */
$("#ask").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!busy) sendAsk(e);
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && busy) { e.preventDefault(); stopReply(); }
});
sendButton();
/* a conversation's settings: its system message, its sampling and the
   answer's form (JSON, by a schema), saved as a preset to use again */
async function chatSettings() {
  const p = cur.params || {};
  const presets = settings._presets || {};
  /* the model's settings in grey: what an empty field is */
  const mname = server.running ? server.model : (typeof nextModel === "function" ? nextModel() : "");
  const own = await modelSampling(mname);
  const style = p.temperature === undefined ? "" : p.temperature <= 0.3 ? "precise" : p.temperature >= 0.9 ? "creative" : "custom";
  const d = await dialog({title: T("Instructions and answer"),
    html: `<div class="srvset plainset">
      <label>${T("Instructions for the model")}
        <textarea class="sysm" rows="4" placeholder="${T("For example: You are a patient teacher; answer in Italian, briefly, with an example.")}">${esc(cur.system || "")}</textarea>
        <span class="note">${T("What the model should be or do in this chat: its role, the language, how long its answers are. It reads them before every message.")}</span></label>
      <label>${T("Its answers")}
        <select class="style"><option value=""${style === "" ? " selected" : ""}>${T("as the model answers by itself")}</option>
          <option value="precise"${style === "precise" ? " selected" : ""}>${T("precise: the same question, nearly the same answer")}</option>
          <option value="creative"${style === "creative" ? " selected" : ""}>${T("creative: more varied, for ideas and stories")}</option>
          ${style === "custom" ? `<option value="custom" selected>${T("as set under Advanced")}</option>` : ""}</select>
        <span class="note">${T("For facts, sums and code, precise; for writing and ideas, creative.")}</span></label>
      ${Object.keys(presets).length ? `<label>${T("Use saved instructions")} <select class="pre"><option value="">-</option>${Object.keys(presets).map((n) => `<option>${esc(n)}</option>`).join("")}</select></label>` : '<select class="pre" hidden></select>'}
      <details><summary>${T("Advanced")}</summary>
        ${samplingFields(p, own, {reset: "Reset to the model's settings"})}
        <label>${T("Answer as")} <select class="fmt"><option value="">${T("text")}</option><option value="json"${p.format === "json" ? " selected" : ""}>${T("a JSON object")}</option><option value="schema"${p.format === "schema" ? " selected" : ""}>${T("JSON by this schema")}</option></select>
          <span class="note">${T("for programs that read the answers: data in a fixed shape")}</span></label>
        <textarea class="schema" rows="4" placeholder='{"type": "object", "properties": {...}}'>${esc(p.schema || "")}</textarea>
        <label>${T("Template variables")} <textarea class="kwargs mono" rows="2" spellcheck="false" placeholder='{"reasoning_effort": "high"}'>${esc(p.kwargs ? JSON.stringify(p.kwargs) : "")}</textarea>
          <span class="note">${T("Given to the model's chat template in this chat, as a JSON object, over the model's own.")}</span></label>
      </details>
      <label>${T("Keep these instructions to use again")} <input class="pname" placeholder="${T("a name (optional)")}"></label></div>`,
    ready: (box) => {
      const smp = (k) => box.querySelector(`.smp[data-k="${k}"]`);
      /* the style chooses the temperature */
      box.querySelector(".style").onchange = (e) => {
        const t = {precise: "0.2", creative: "1.0", "": ""}[e.target.value];
        if (t !== undefined) { smp("temperature").value = t; smp("top_p").value = ""; samplingSync(box, "temperature", null); samplingSync(box, "top_p", null); }
      };
      box.querySelector(".pre").onchange = (e) => {
        const v = presets[e.target.value]; if (!v) return;
        box.querySelector(".sysm").value = v.system || "";
        samplingFill(box, samplingOf(v));
        box.querySelector(".fmt").value = v.format || "";
        box.querySelector(".schema").value = v.schema || "";
        box.querySelector(".kwargs").value = v.kwargs ? JSON.stringify(v.kwargs) : "";
      };
    },
    read: (box) => {
      const f = {system: box.querySelector(".sysm").value.trim(),
        format: box.querySelector(".fmt").value, schema: box.querySelector(".schema").value.trim(),
        name: box.querySelector(".pname").value.trim(), sampling: samplingRead(box),
        kwargs: box.querySelector(".kwargs").value.trim()};
      return f;
    },
    buttons: [{label: T("Save"), value: "ok", primary: true}, {label: T("Cancel"), value: "no"}]});
  if (d.button !== "ok") return;
  const f = d.form;
  if (f.format === "schema") {
    try { JSON.parse(f.schema); } catch (e) { say(T("The schema is not JSON: {why}", {why: e.message})); return; }
  }
  if (!f.sampling) return; /* its breakers are not JSON: said */
  let kw;
  if (f.kwargs) {
    try { kw = JSON.parse(f.kwargs); } catch (e) {}
    if (!kw || typeof kw !== "object" || Array.isArray(kw)) { say(T("The template variables are not a JSON object.")); return; }
  }
  cur.system = f.system;
  cur.params = {format: f.format, schema: f.schema, ...f.sampling};
  if (kw) cur.params.kwargs = kw;
  if (f.name) {
    settings._presets = {...presets, [f.name]: {system: f.system, ...cur.params}};
    await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
  }
  await saveConv(cur);
}
$("#dirchg").onclick = async () => {
  let p = ws.window ? await pick(false, T("The folder of the models")) : null;
  if (p === null) p = await input(T("The folder of the models"), $("#dir").value);
  if (!p) return;
  const r = await fetch("/brebus/models-dir", {method: "POST", body: JSON.stringify({path: p})});
  if (!r.ok) { say(T("Not a folder there is: {path}", {path: p})); return; }
  catalog = null; await mine();
};
/* the workspace: chosen before the conversations, which live in it */
let ws = {path: ""};
/* the system's dialog, in Brebus's window; from a browser, the folders of
   the machine Brebus runs on, in a dialog of the page: the path, ""
   cancelled */
async function pick(file, title) {
  if (ws.window) {
    const r = await fetch("/brebus/pick", {method: "POST", body: JSON.stringify({file, title})});
    if (r.ok) return (await r.json()).path;
  }
  return webPick(file, title);
}
async function webPick(file, title) {
  let at = "", chosen = "";
  const draw = async (box) => {
    const j = await (await fetch("/brebus/dirs?files=" + (file ? 1 : 0) + "&path=" + encodeURIComponent(at))).json();
    const list = box.querySelector(".wplist");
    if (j.error) { list.textContent = j.error; return; }
    at = j.path; chosen = file ? "" : j.path;
    box.querySelector(".wpat").textContent = j.path;
    list.innerHTML = (j.up ? `<button class="act quiet wpgo" data-p="${esc(j.up)}">⬆ ${T("up")}</button>` : "") +
      j.dirs.map((d) => `<button class="act quiet wpgo" data-p="${esc(j.path + "/" + d)}">📁 ${esc(d)}</button>`).join("") +
      (j.files || []).map((f) => `<button class="act quiet wpfile" data-p="${esc(j.path + "/" + f)}">${esc(f)}</button>`).join("") +
      (j.dirs.length || (j.files || []).length ? "" : `<span class="note">${file ? T("nothing here") : T("nothing here: this folder may be chosen")}</span>`);
    for (const b of list.querySelectorAll(".wpgo")) b.onclick = () => { at = b.dataset.p; draw(box); };
    for (const b of list.querySelectorAll(".wpfile")) b.onclick = () => {
      chosen = b.dataset.p;
      for (const x of list.querySelectorAll(".wpfile")) x.classList.toggle("on", x === b);
    };
  };
  const d = await dialog({title, wide: true,
    html: `<p class="note">${T("On the computer Brebus runs on: {path}", {path: '<code class="wpat"></code>'})}</p><div class="wplist"></div>`,
    ready: (box) => draw(box), read: () => chosen,
    buttons: [{label: file ? T("Choose the file") : T("Choose this folder"), value: "ok", primary: true}, {label: T("Cancel"), value: "no"}]});
  return d.button === "ok" ? d.form || "" : "";
}
/* fresh: for a new project (+ beside Projects), without the settings of
   the project open now */
function wsShow(open) {
  /* Browse: the window's dialog, or the page's own from a browser (the
     dialog's own: Skills' and Connectors' take the window's only) */
  for (const b of document.querySelectorAll("#wsdlg button.browse")) b.hidden = false;
  /* paths as this system writes them */
  if (ws.windows)
    for (const i of document.querySelectorAll('input[name="path"]'))
      i.placeholder = i.placeholder.replace(/^\/home\/([^/]+)\//, "C:\\Users\\$1\\").replaceAll("/", "\\");
  $("#wsrecent").innerHTML = ws.recent.length
    ? `<b>${T("Recent")}</b>` + ws.recent.map((p) => `<button data-p="${esc(p)}" title="${esc(p)}">${esc(p)}</button>`).join("")
    : "";
  for (const b of document.querySelectorAll("#wsrecent button")) b.onclick = () => wsOpen(b.dataset.p, false);
  $("#wsnewin").textContent = T("in {folder}, named by today's date", {folder: ws.new_in});
  $("#wsclose").hidden = !ws.path;
  $("#wsdlg").hidden = !open;
}
/* a project opened (create "new": a new folder; "free": the free chats'
   own place), then its chat id, or its newest, or a new one */
async function wsOpen(path, create, id) {
  const r = await fetch(create === "new" ? "/brebus/workspace/new"
                        : create === "free" ? "/brebus/workspace/free" : "/brebus/workspace/open",
    {method: "POST", body: JSON.stringify({path, create: !!create && create !== "free"})});
  const j = await r.json();
  if (!r.ok) { say(j.error || T("it did not open")); return false; }
  ws = j; wsShow(false);
  await readTools(); await readFolders(); await readServices();
  cur = null; await readConvs();
  const live = convs.filter((c) => !c.archived);
  if (id && convs.some((c) => c.id === id)) await openConv(id);
  else if (id === "new" || !live.length) newConv();
  else await openConv(live[0].id);
  await refreshTree();
  return true;
}
$("#wsclose").onclick = () => wsShow(false);

/* ---- a project's settings (its ⋯): named by its folder, in tabs ---- */
function psTab(t) {
  for (const b of document.querySelectorAll("#psdlg .pstabs button")) {
    b.classList.toggle("on", b.dataset.t === t);
    b.setAttribute("aria-selected", b.dataset.t === t ? "true" : "false");
  }
  for (const s of document.querySelectorAll("#psdlg .pstab")) s.hidden = s.dataset.t !== t;
}
async function psShow(tab) {
  if (!ws.path || ws.free) return;
  $("#pstitle").textContent = ws.path.split(/[\\/]/).filter(Boolean).pop() || ws.path;
  $("#pspath").textContent = ws.path;
  /* the side bar's list, the one just drawn */
  const p = (typeof tree !== "undefined" && tree && tree.projects || []).find((x) => x.path === ws.path);
  const n = ((p && p.chats && p.chats.chats) || convs).filter((c) => !c.archived).length;
  $("#pschats").textContent = Tn(n, "{n} chat", "{n} chats");
  psTab(tab || "gen");
  $("#psdlg").hidden = false;
  readServices(); readFolders(); /* changed elsewhere meanwhile, perhaps */
}
const psClose = () => { $("#psdlg").hidden = true; };
for (const b of document.querySelectorAll("#psdlg .pstabs button")) b.onclick = () => psTab(b.dataset.t);
$("#psx").onclick = psClose;
$("#psext").onclick = () => { psClose(); showExt("inst"); };
/* Esc, or a click outside the box, closes either dialog (the opening one
   only when a project is open behind it) */
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#psdlg").hidden && !document.querySelector('.dlg [role="dialog"][aria-modal="true"]')) { e.preventDefault(); e.stopPropagation(); psClose(); }
  else if (!$("#wsdlg").hidden && ws.path) { e.preventDefault(); e.stopPropagation(); wsShow(false); }
}, true);
for (const [id, close] of [["#psdlg", psClose], ["#wsdlg", () => ws.path && wsShow(false)]]) {
  let down = null; /* a press that starts inside and ends outside is not a click outside */
  $(id).addEventListener("pointerdown", (e) => { down = e.target; });
  $(id).addEventListener("click", (e) => { if (e.target === $(id) && down === $(id)) close(); });
}
$("#psdel").onclick = async () => {
  const name = $("#pstitle").textContent, path = ws.path;
  const d = await dialog({title: T("Remove {p} from Brebus?", {p: name}),
    html: `<p>${T("Brebus stops listing it. Your files in {f} are not touched.", {f: `<code>${esc(path)}</code>`})}</p>` +
          `<label class="drem"><input type="checkbox" id="pspurge"> ${T("Also delete its chats and settings (.brebus) and its artifacts (artifact): this cannot be undone")}</label>`,
    read: (box) => box.querySelector("#pspurge").checked,
    buttons: [{label: T("Remove"), value: "yes", primary: true}, {label: T("Cancel"), value: "no"}]});
  if (d.button !== "yes") return;
  const r = await fetch("/brebus/workspace/forget", {method: "POST", body: JSON.stringify({path, purge: !!d.form})});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { say(j.error || T("not removed")); return; }
  psClose();
  /* it was open: the free chats are, now */
  await wsOpen("", "free", "new");
  say(d.form ? T("{p} is removed, with its chats and artifacts.", {p: name}) : T("{p} is removed from the list: open its folder again to have it back.", {p: name}), T("Removed"));
};
$("#wsnew").onclick = () => wsOpen("", "new");
$("#wsform").onsubmit = () => {
  const p = $("#wsform").path.value.trim();
  if (p) yes(T("Make {path} if it is not there?", {path: p}), T("Open a folder"), T("Make it")).then((m) => wsOpen(p, m));
  return false;
};
const booted = (async () => {
  ws = await ask("workspace");
  /* the page loaded again in another language (langSet): back where the
     user was, the chat and the section */
  let back = null;
  try { back = JSON.parse(sessionStorage.getItem("brebus.resume") || "null"); sessionStorage.removeItem("brebus.resume"); } catch (e) {}
  if (back && await wsOpen(back.free ? "" : back.path, back.free ? "free" : false, back.chat || "new")) {
    if (back.section && $("#" + back.section)) show(back.section);
    return;
  }
  /* a chat named in the address (In a new window): in the workspace open */
  if (hashChat && ws.path) {
    wsShow(false);
    await readTools(); await readFolders(); await readServices();
    await readConvs();
    if (convs.some((c) => c.id === hashChat)) { await openConv(hashChat); await refreshTree(); return; }
  }
  /* a project given to brebus (--workspace): a new chat in it */
  if (ws.given && ws.path && !ws.free) { await wsOpen(ws.path, false, "new"); return; }
  /* else a new chat, of no project, as Claude opens: written only with its
     first message; the others a click away in the side bar */
  await wsOpen("", "free", "new");
})().catch(() => {});
status().catch(() => {});

const loaded = load().catch((e) => { document.body.insertAdjacentHTML("beforeend",
  `<p style="position:fixed;bottom:8px;right:16px;color:var(--warn)">${esc(e.message)}</p>`); });
/* the model loaded as Brebus opens - the chat's own, or the one used last -
   ready before the first message, not with it */
Promise.all([booted, loaded]).then(() => { if (!server.running && nextModel()) ensureModel(); })
  .catch(() => {});
