/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* turn.js - Brebus's page (app.html): a turn of the model: its reply streamed, its tools' calls. build.sh puts it into
   the page where app.html includes it. */
/* one reply of the model, streamed into reply: its tool calls, if any */
async function streamTurn(reply, withTools) {
  const t0 = performance.now(); let t1 = 0, tokens = 0, pieces = 0;
  const calls = [];
  const body = {model: server.model, stream: true, stream_options: {include_usage: true},
                messages: apiMessages().slice(0, -1)};
  const p = cur.params || {};
  if (p.temperature !== undefined) body.temperature = p.temperature;
  if (p.top_p !== undefined) body.top_p = p.top_p;
  if (p.format === "json") body.response_format = {type: "json_object"};
  else if (p.format === "schema" && p.schema)
    try { body.response_format = {type: "json_schema", json_schema: {name: "answer", schema: JSON.parse(p.schema)}}; } catch (e) {}
  if (withTools)
    body.tools = tools.tools.filter((t) => t.function.name === "python"
      ? $("#usecode").checked : $("#usetools").checked);
  const r = await fetch("/v1/chat/completions", {method: "POST", signal: busy.signal,
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!r.ok) { reply.content = "(" + r.status + ") " + await r.text(); return []; }
  const rd = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const {done, value} = await rd.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const ev2 = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of ev2.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          try {
            const j = JSON.parse(data);
            if (j.usage) {
              tokens = j.usage.completion_tokens || 0;
              /* the memory the chat fills now */
              reply.ctx = (j.usage.prompt_tokens || 0) + tokens;
            }
            const d = j.choices?.[0]?.delta || {};
            if (d.reasoning_content) reply.think += d.reasoning_content;
            if (d.content) reply.content += d.content;
            for (const tc of d.tool_calls || []) {
              const k = tc.index ?? calls.length;
              calls[k] = calls[k] || {id: "", type: "function", function: {name: "", arguments: ""}};
              if (tc.id) calls[k].id = tc.id;
              if (tc.function?.name) calls[k].function.name += tc.function.name;
              if (tc.function?.arguments) calls[k].function.arguments += tc.function.arguments;
            }
            if (d.content || d.reasoning_content) { pieces++; if (!t1) t1 = performance.now(); }
          } catch (e) {}
        }
        const last = $("#log").lastElementChild;
        if (last) redraw(last, reply);
        if (followLog) $("#log").scrollTop = $("#log").scrollHeight;
      }
    }
  } catch (err) {
    reply.speed = err.name === "AbortError" ? `stopped after ${pieces} pieces` : "failed: " + err.message;
    throw err;
  }
  const t2 = performance.now();
  if (tokens > 1 && t1)
    reply.speed = `${tokens} tokens, ${((tokens - 1) / ((t2 - t1) / 1000)).toFixed(1)} tok/s, first after ${((t1 - t0) / 1000).toFixed(2)} s`;
  return calls.filter(Boolean);
}

/* the pictures of the conversation, as data URLs, kept in memory only (the
   conversation's file keeps their paths) */
const pictures = new Map();
async function fileBlob(path) {
  const r = await fetch("/brebus/file?path=" + encodeURIComponent(path));
  if (!r.ok) throw new Error("not found: " + path);
  return r.blob();
}
const dataURL = (blob) => new Promise((ok, no) => {
  const f = new FileReader(); f.onload = () => ok(f.result); f.onerror = no; f.readAsDataURL(blob);
});
async function prepareMedia() {
  for (const m of cur.messages)
    for (const a of (m.role === "user" && m.attached) || []) {
      if (a.media === "image" && !pictures.has(a.path))
        try { pictures.set(a.path, await dataURL(await fileBlob(a.path))); } catch (e) {}
      if (a.media === "audio" && a.transcript === undefined && !a.unread) {
        /* transcribed by the model's server (its Whisper) */
        try {
          const fd = new FormData();
          fd.append("file", await fileBlob(a.path), a.name);
          fd.append("model", "whisper");
          const r = await fetch("/v1/audio/transcriptions", {method: "POST", body: fd});
          const j = await r.json();
          if (r.ok && typeof j.text === "string") a.transcript = j.text.trim();
          else a.unread = j.error?.message || j.error || "the server did not transcribe it";
        } catch (e) { a.unread = e.message; }
      }
    }
}

/* a turn of the model for the user's message um, the last of the
   conversation: its documents searched, the model's replies and its tools'
   calls, until it answers; the answer it gives last keeps the earlier ones
   (keep: those of a regeneration) to be paged through */
/* no model running - unloaded after a quiet while, or Brebus started
   anew: the chat's own model loaded by itself, or else the one used last;
   My models only when there is none */
function nextModel() {
  return [cur && cur.model, server.last, settings._last]
    .find((n) => n && modelInfo[n]) || bestFit();
}
/* none remembered (the last one removed, a first start): the largest
   installed model that leaves this computer half its memory */
function bestFit() {
  const room = (machineMem || 0) / 2;
  return Object.keys(modelInfo).filter((n) => !n.startsWith("__") && modelInfo[n].bytes <= room)
    .sort((a, b) => modelInfo[b].bytes - modelInfo[a].bytes)[0] || "";
}
let modelLoading = null; /* one start at a time: Brebus's own, a message's */
function ensureModel() {
  if (!modelLoading) modelLoading = startModel().finally(() => { modelLoading = null; });
  return modelLoading;
}
async function startModel() {
  /* asked anew: the server may have been stopped from elsewhere */
  await status();
  if (server.running) return true;
  const name = nextModel();
  if (!name) { say("Get a model first, in My models."); return false; }
  $("#loaded").textContent = "Loading " + name + "…";
  ctxOf = {};
  const r = await fetch("/brebus/server/start", {method: "POST",
    body: JSON.stringify({model: name, ...(settings[name] || {})})});
  if (!r.ok) { const j = await r.json().catch(() => ({})); say(j.error || name + " did not start"); }
  else remember(name);
  await status();
  return r.ok && server.running;
}
/* the model used last, kept for the next start of Brebus */
function remember(name) {
  if (settings._last === name) return;
  settings._last = name;
  fetch("/brebus/settings", {method: "POST", body: JSON.stringify(settings)}).catch(() => {});
}
async function runTurn(um, keep) {
  busy = new AbortController(); sendButton();
  followLog = true; /* a message sent: its reply followed */
  try {
    await prepareMedia();
    /* the workspace's documents searched for the question */
    if ($("#usedocs").checked && !$("#usedocs").disabled) {
      const f = await callTool("docs_search", JSON.stringify({query: um.content, max: 6}));
      if (!f.is_error && f.result && !/^0 passages|^No /.test(f.result))
        um.found = f.result.slice(0, 11000);
      drawLog();
    }
    /* an answer asked in JSON is a direct one: no tools then */
    const json = cur.params && cur.params.format;
    const withTools = !json && ($("#usetools").checked || $("#usecode").checked) && tools.tools.length > 0;
    for (let round = 0; round < 7; round++) {
      /* the tools' rounds spent: the last one answers with what it has */
      const lastRound = round === 6;
      const reply = {role: "assistant", content: "", think: ""};
      cur.messages.push(reply); drawLog();
      const calls = await streamTurn(reply, withTools && !lastRound);
      if ($("#useart").checked || /<artifact\b/.test(reply.content)) await artSave(reply);
      /* a picture the reply names from the workspace, shown beside it */
      const pics = picRefs(reply.content);
      if (pics.length) artShowFile(pics.at(-1).path, pics.at(-1).title);
      if (!calls.length) break;
      reply.tool_calls = calls;
      for (const c of calls) {
        const name = c.function.name, args = c.function.arguments || "{}";
        const only_reads = tools.read_only.includes(name);
        let res;
        let pretty = args, what = "It changes something in the workspace. Its arguments:";
        try {
          const a = JSON.parse(args);
          pretty = name === "python" && typeof a.code === "string" ? a.code : JSON.stringify(a, null, 2);
          if (name === "python") what = "Python code, run in a sandbox: the workspace (read and write) and its folders read only, no network:";
          if (name === "janas__open_service") what = "It starts one of Janas's services, which gives the model its tools (it may reach the network):";
        } catch (e) {}
        /* a tool of the user's servers may be allowed always in this chat,
           for the server it was given to; Janas's own that change something
           are asked every time */
        const srv = (tools.servers || []).find((s) => name.startsWith(s.prefix));
        const ap = srv && cur.approvals && cur.approvals[name];
        const good = ap && ap.ident === srv.ident && ap.version === srv.version;
        let ok = only_reads || good;
        if (!ok) {
          const btns = [{label: "Allow", value: "yes", primary: true}];
          if (srv) btns.push({label: "Allow always in this chat", value: "always"});
          btns.push({label: "Don't allow", value: "no"});
          const b = (await dialog({title: "The model wants to run " + name,
                html: `${what}<pre class="dargs">${esc(pretty)}</pre>`, buttons: btns})).button;
          ok = b === "yes" || b === "always";
          if (b === "always") {
            cur.approvals = cur.approvals || {};
            cur.approvals[name] = {ident: srv.ident, version: srv.version, used: Date.now()};
          }
        } else if (good) {
          ap.used = Date.now(); /* its clock starts again */
        }
        if (!ok)
          res = {result: "The user did not allow this call.", is_error: true, denied: true};
        else
          res = await callTool(name, args);
        cur.messages.push({role: "tool", tool_call_id: c.id, name, args,
                           content: res.result, denied: !!res.denied});
        if (res.opened) await readTools(); /* the service's tools, from now */
        drawLog();
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      const last = cur.messages[cur.messages.length - 1];
      if (last && last.role === "assistant" && !last.speed) last.speed = "failed: " + err.message;
    }
  } finally {
    const last = cur.messages[cur.messages.length - 1];
    if (keep && keep.length && last && last.role === "assistant") {
      last.variants = [...keep, variantOf(last)];
      last.vi = last.variants.length - 1;
    }
    busy = null; sendButton();
    const said = cur.messages[cur.messages.length - 1];
    if (said && said.role === "assistant") speakReply(said);
    drawLog(); await saveConv(cur);
    $("#ask").focus();
  }
}

async function sendAsk(ev) {
  ev.preventDefault();
  if (busy) { stopReply(); return; }
  const text = $("#ask").value.trim();
  if (!text) { if (ev.type === "submit") dictate(); return; }
  if (!ws.path) { wsShow(true); return; }
  if (!await ensureModel()) return;
  if (!cur) newConv();
  const um = {role: "user", content: text};
  if (pending.length) { um.attached = pending; pending = []; drawChips(); }
  cur.messages.push(um);
  if (!cur.title) cur.title = text.slice(0, 40);
  cur.model = server.model;
  const e = convs.find((c) => c.id === cur.id); if (e) e.title = cur.title;
  $("#ask").value = ""; growAsk();
  listConvs(); drawLog();
  await runTurn(um);
}

