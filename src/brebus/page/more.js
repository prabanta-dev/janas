/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* more.js - Brebus's page (app.html): regenerate and edit, services, folders, settings, the workspace, the start. build.sh puts it into
   the page where app.html includes it. */
/* ---- regenerate, and edit a message ---- */
const variantOf = (m) => ({content: m.content, think: m.think, speed: m.speed, saved: m.saved,
                           tool_calls: m.tool_calls});
/* the answer to the last question written again, the earlier ones kept */
async function regenerate() {
  if (busy || !await ensureModel()) return;
  let i = cur.messages.length - 1;
  while (i >= 0 && cur.messages[i].role !== "user") i--;
  if (i < 0) return;
  const last = cur.messages[cur.messages.length - 1];
  const keep = last && last.role === "assistant"
    ? (last.variants ? last.variants : [variantOf(last)]) : [];
  cur.messages = cur.messages.slice(0, i + 1);
  drawLog();
  await runTurn(cur.messages[i], keep);
}
/* one of the user's messages changed: the conversation goes on from it */
async function editMessage(k) {
  if (busy) return;
  const m = cur.messages[k];
  const t = await input("Edit the message", m.content, "The conversation after it is replaced by the new answer.", true);
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
  $("#svcnote").innerHTML = `The model opens one when a question needs it. Your own MCP servers go in <code>${esc(j.config)}</code>, in the format of Cursor and LM Studio.`;
  $("#svclist").innerHTML = j.services.map((x) =>
    `<label><input type="checkbox" data-n="${esc(x.name)}"${x.on ? " checked" : ""}> <b>${esc(x.name)}</b> <small>${esc(x.about)}</small></label>`).join("") ||
    '<div class="note">none found beside Brebus</div>';
  for (const c of document.querySelectorAll("#svclist input"))
    c.onchange = async () => {
      const off = [...document.querySelectorAll("#svclist input")].filter((x) => !x.checked).map((x) => x.dataset.n);
      settings._services = {off};
      await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
      await fetch("/brebus/tools/restart", {method: "POST", body: "{}"});
      await readTools();
    };
}

/* the folders read only beside the workspace */
async function readFolders() {
  const f = await ask("folders");
  $("#wsro").hidden = !ws.path;
  $("#rolist").innerHTML = f.read_only.length
    ? f.read_only.map((p, i) => `<div><span>${esc(p)}</span><button data-i="${i}" title="The model no longer reads this folder (nothing in it is deleted)">remove</button></div>`).join("")
    : '<div class="note">none</div>';
  for (const b of document.querySelectorAll("#rolist button"))
    b.onclick = async () => { f.read_only.splice(+b.dataset.i, 1); await setFolders(f.read_only); };
  return f;
}
async function setFolders(list) {
  const r = await fetch("/brebus/folders", {method: "POST", body: JSON.stringify({read_only: list})});
  const j = await r.json();
  if (j.refused) say("Some were not added: a folder or file there is, by its full path.");
  await readFolders(); await readTools();
}
$("#embsel").onchange = async () => {
  settings._docs = {embedding: $("#embsel").value};
  await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
  await fetch("/brebus/tools/restart", {method: "POST", body: "{}"});
  await readTools();
};
$("#wsbrowse").onclick = async () => {
  const p = await pick(false, "The folder Brebus works in");
  if (p) wsOpen(p, false);
};
$("#robrowse").onclick = async () => {
  const p = await pick(false, "A folder to read only, beside the workspace");
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
/* a conversation's settings: its system message, temperature, top-p and
   the answer's form (JSON, by a schema), saved as a preset to use again */
async function chatSettings() {
  const p = cur.params || {};
  const presets = settings._presets || {};
  const style = p.temperature === undefined ? "" : p.temperature <= 0.3 ? "precise" : p.temperature >= 0.9 ? "creative" : "custom";
  const d = await dialog({title: "Instructions and answer",
    html: `<div class="srvset plainset">
      <label>Instructions for the model
        <textarea class="sysm" rows="4" placeholder="For example: You are a patient teacher; answer in Italian, briefly, with an example.">${esc(cur.system || "")}</textarea>
        <span class="note">What the model should be or do in this chat: its role, the language, how long its answers are. It reads them before every message.</span></label>
      <label>Its answers
        <select class="style"><option value=""${style === "" ? " selected" : ""}>as the model answers by itself</option>
          <option value="precise"${style === "precise" ? " selected" : ""}>precise: the same question, nearly the same answer</option>
          <option value="creative"${style === "creative" ? " selected" : ""}>creative: more varied, for ideas and stories</option>
          ${style === "custom" ? '<option value="custom" selected>as set under Advanced</option>' : ""}</select>
        <span class="note">For facts, sums and code, precise; for writing and ideas, creative.</span></label>
      ${Object.keys(presets).length ? `<label>Use saved instructions <select class="pre"><option value="">-</option>${Object.keys(presets).map((n) => `<option>${esc(n)}</option>`).join("")}</select></label>` : '<select class="pre" hidden></select>'}
      <details><summary>Advanced</summary>
        <span class="row"><label>Temperature <input class="temp" type="number" min="0" max="2" step="0.1" value="${p.temperature ?? ""}" placeholder="the model's">
          <span class="note">0: always the likeliest word; higher: freer</span></label>
          <label>Top-p <input class="topp" type="number" min="0" max="1" step="0.05" value="${p.top_p ?? ""}" placeholder="the model's">
          <span class="note">how many of the likely words it chooses among</span></label></span>
        <label>Answer as <select class="fmt"><option value="">text</option><option value="json"${p.format === "json" ? " selected" : ""}>a JSON object</option><option value="schema"${p.format === "schema" ? " selected" : ""}>JSON by this schema</option></select>
          <span class="note">for programs that read the answers: data in a fixed shape</span></label>
        <textarea class="schema" rows="4" placeholder='{"type": "object", "properties": {...}}'>${esc(p.schema || "")}</textarea>
      </details>
      <label>Keep these instructions to use again <input class="pname" placeholder="a name (optional)"></label></div>`,
    ready: (box) => {
      /* the style chooses the temperature */
      box.querySelector(".style").onchange = (e) => {
        const t = {precise: "0.2", creative: "1.0", "": ""}[e.target.value];
        if (t !== undefined) { box.querySelector(".temp").value = t; box.querySelector(".topp").value = ""; }
      };
      box.querySelector(".pre").onchange = (e) => {
        const v = presets[e.target.value]; if (!v) return;
        box.querySelector(".sysm").value = v.system || "";
        box.querySelector(".temp").value = v.temperature ?? "";
        box.querySelector(".topp").value = v.top_p ?? "";
        box.querySelector(".fmt").value = v.format || "";
        box.querySelector(".schema").value = v.schema || "";
      };
    },
    read: (box) => ({system: box.querySelector(".sysm").value.trim(),
      temperature: box.querySelector(".temp").value, top_p: box.querySelector(".topp").value,
      format: box.querySelector(".fmt").value, schema: box.querySelector(".schema").value.trim(),
      name: box.querySelector(".pname").value.trim()}),
    buttons: [{label: "Save", value: "ok", primary: true}, {label: "Cancel", value: "no"}]});
  if (d.button !== "ok") return;
  const f = d.form;
  if (f.format === "schema") {
    try { JSON.parse(f.schema); } catch (e) { say("The schema is not JSON: " + e.message); return; }
  }
  cur.system = f.system;
  cur.params = {temperature: f.temperature === "" ? undefined : +f.temperature,
                top_p: f.top_p === "" ? undefined : +f.top_p, format: f.format, schema: f.schema};
  if (f.name) {
    settings._presets = {...presets, [f.name]: {system: f.system, ...cur.params}};
    await fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)});
  }
  await saveConv(cur);
}
$("#dirchg").onclick = async () => {
  let p = ws.window ? await pick(false, "The folder of the models") : null;
  if (p === null) p = await input("The folder of the models", $("#dir").value);
  if (!p) return;
  const r = await fetch("/brebus/models-dir", {method: "POST", body: JSON.stringify({path: p})});
  if (!r.ok) { say("Not a folder there is: " + p); return; }
  catalog = null; await mine();
};
/* the workspace: chosen before the conversations, which live in it */
let ws = {path: ""};
/* the system's dialog, in Brebus's window: the path, "" cancelled, null
   where there is none (the browser) */
async function pick(file, title) {
  const r = await fetch("/brebus/pick", {method: "POST", body: JSON.stringify({file, title})});
  if (!r.ok) return null;
  return (await r.json()).path;
}
function wsShow(open) {
  for (const b of document.querySelectorAll("button.browse")) b.hidden = !ws.window;
  /* paths as this system writes them */
  if (ws.windows)
    for (const i of document.querySelectorAll('input[name="path"]'))
      i.placeholder = i.placeholder.replace("/home/you/", "C:\\Users\\you\\").replaceAll("/", "\\");
  $("#wsrecent").innerHTML = ws.recent.length
    ? "<b>Recent</b>" + ws.recent.map((p) => `<button data-p="${esc(p)}" title="${esc(p)}">${esc(p)}</button>`).join("")
    : "";
  for (const b of document.querySelectorAll("#wsrecent button")) b.onclick = () => wsOpen(b.dataset.p, false);
  $("#wsnewin").textContent = "in " + ws.new_in + ", named by today's date";
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
  if (!r.ok) { say(j.error || "it did not open"); return false; }
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
$("#wsnew").onclick = () => wsOpen("", "new");
$("#wsform").onsubmit = () => {
  const p = $("#wsform").path.value.trim();
  if (p) yes(`Make ${p} if it is not there?`, "Open a folder", "Make it").then((m) => wsOpen(p, m));
  return false;
};
const booted = (async () => {
  ws = await ask("workspace");
  /* the chat used last, where it is (a project, the free chats), if it is
     still there; else the free chats' newest, or a new one */
  let last = null;
  try { last = JSON.parse(localStorage.getItem("brebus.last") || "null"); } catch (e) {}
  if (last && last.ws && last.id && !hashChat) {
    const t = await ask("projects").catch(() => null);
    const there = t && [{path: t.free, chats: t.free_chats}, ...t.projects]
      .find((p) => p.path === last.ws && (p.chats.chats || []).some((c) => c.id === last.id && !c.archived));
    if (there) { await wsOpen(last.ws, last.ws === t.free ? "free" : false, last.id); return; }
  }
  if (!ws.path) { await wsOpen("", "free"); return; }
  wsShow(false);
  await readTools(); await readFolders(); await readServices();
  await readConvs();
  const live = convs.filter((c) => !c.archived);
  if (hashChat && convs.some((c) => c.id === hashChat)) await openConv(hashChat);
  else if (live.length) await openConv(live[0].id); else newConv();
  await refreshTree();
})().catch(() => {});
status().catch(() => {});

const loaded = load().catch((e) => { document.body.insertAdjacentHTML("beforeend",
  `<p style="position:fixed;bottom:8px;right:16px;color:var(--warn)">${esc(e.message)}</p>`); });
/* the model loaded as Brebus opens - the chat's own, or the one used last -
   ready before the first message, not with it */
Promise.all([booted, loaded]).then(() => { if (!server.running && nextModel()) ensureModel(); })
  .catch(() => {});
