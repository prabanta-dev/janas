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
  /* reasoning: the chat's choice (the button under the field), else the
     model's setting, else off; a model that cannot be told ignores it */
  const p = cur.params || {};
  /* the chat's own template variables under it */
  body.chat_template_kwargs = {...(p.kwargs || {}), enable_thinking: thinkNow()};
  /* the sampling the chat set (more.js: SAMPLING); the model's for the rest */
  for (const x of SAMPLING) if (p[x.k] !== undefined) body[x.k] = p[x.k];
  if (p.samplers) body.samplers = p.samplers;
  if (p.dry_sequence_breakers) body.dry_sequence_breakers = p.dry_sequence_breakers;
  if (p.format === "json") body.response_format = {type: "json_object"};
  else if (p.format === "schema" && p.schema)
    try { body.response_format = {type: "json_schema", json_schema: {name: "answer", schema: JSON.parse(p.schema)}}; } catch (e) {}
  if (withTools)
    body.tools = tools.tools.filter((t) => t.function.name === "python"
      ? $("#usecode").checked : $("#usetools").checked);
  const r = await fetch("/v1/chat/completions", {method: "POST", signal: busy.signal,
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!r.ok) { reply.content = "(" + r.status + ") " + await r.text(); reply.failed = true; return []; }
  const rd = r.body.getReader(), dec = new TextDecoder();
  let buf = "", sawDone = false, srvErr = "";
  try {
    for (;;) {
      const {done, value} = await rd.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const ev2 = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of ev2.split("\n")) {
          /* the server's word while it reads a long prompt: how far */
          const rd = /^: reading (\d+)\/(\d+)/.exec(line);
          if (rd) { reply.reading = {done: +rd[1], total: +rd[2]}; continue; }
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") { sawDone = true; continue; }
          try {
            const j = JSON.parse(data);
            /* an error the server sent in the stream itself */
            if (j.error) srvErr = typeof j.error === "string" ? j.error : j.error.message || JSON.stringify(j.error);
            if (j.usage) {
              tokens = j.usage.completion_tokens || 0;
              /* the memory the chat fills now */
              reply.ctx = (j.usage.prompt_tokens || 0) + tokens;
            }
            const d = j.choices?.[0]?.delta || {};
            if (d.reasoning_content) { if (!reply.thinkAt) reply.thinkAt = Date.now(); reply.think += d.reasoning_content; }
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
        /* the last message's box, in its turn (box and its line under) */
        const turn = $("#log").lastElementChild;
        const last = turn && (turn.querySelector(":scope > .msg") || turn);
        if (last) redraw(last, reply);
        if (followLog) $("#log").scrollTop = $("#log").scrollHeight;
      }
    }
  } catch (err) {
    /* kept with the reply, in the language of the page that wrote it */
    reply.speed = err.name === "AbortError" ? Tn(pieces, "stopped after {n} piece", "stopped after {n} pieces")
      : T("failed: {error}", {error: err.message});
    throw err;
  }
  delete reply.reading; /* the prompt's reading, over */
  /* nothing came, or an error did: said, not left as an empty answer (and
     kept out of what the model is sent next time) */
  if (srvErr || (!sawDone && !reply.content && !reply.think && !calls.length)) {
    reply.content = srvErr ? T("(error) {error}", {error: srvErr})
      : T("(no answer) the model's server ended the reply without a word: it may have stopped (see My models, Server)");
    reply.failed = true;
    await status().catch(() => {});
    return [];
  }
  const t2 = performance.now();
  if (tokens > 1 && t1)
    reply.speed = T("{tokens} tokens, {rate} tok/s, first after {first} s", {tokens: num(tokens),
      rate: num(+((tokens - 1) / ((t2 - t1) / 1000)).toFixed(1)), first: num(+((t1 - t0) / 1000).toFixed(2))});
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
let modelLoading = null; /* one start at a time: Brebus's own, a message's,
                             a model chosen under the field or in My models */
/* a model's start, run: the field and its button off until it is over,
   as at Brebus's start; one after the other */
function whileLoading(run) {
  const p = (modelLoading || Promise.resolve()).catch(() => {}).then(run).finally(() => {
    if (modelLoading === p) modelLoading = null;
    /* the words under the chat and the field change together */
    if (server.running) $("#loaded").textContent = "";
    sendButton();
  });
  modelLoading = p;
  sendButton();
  return p;
}
/* a model running for a message: true when there is one */
function ensureModel() {
  return modelLoading ? modelLoading.then(() => server.running) : whileLoading(startModel);
}
async function startModel() {
  /* asked anew: the server may have been stopped from elsewhere */
  await status();
  if (server.running) return true;
  const name = nextModel();
  if (!name) { say(T("Get a model first, in My models.")); return false; }
  $("#loaded").textContent = T("Loading {name}…", {name});
  sendButton(); /* the field says so, and waits */
  ctxOf = {};
  const r = await fetch("/brebus/server/start", {method: "POST",
    body: JSON.stringify({model: name, ...(settings[name] || {})})});
  if (!r.ok) { const j = await r.json().catch(() => ({})); say(j.error || T("{name} did not start", {name})); }
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
      const reply = {role: "assistant", content: "", think: "", t: Date.now()};
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
        const {title, html} = approvalText(name, args);
        /* a tool of the user's servers may be allowed always in this chat,
           for the server it was given to; Janas's own that change something
           are asked every time */
        /* the longest prefix: "a__x" is server a's, "a___x" server a_'s */
        const srv = (tools.servers || []).filter((s) => name.startsWith(s.prefix)).sort((x, y) => y.prefix.length - x.prefix.length)[0];
        const ap = srv && cur.approvals && cur.approvals[name];
        const good = ap && ap.ident === srv.ident && ap.version === srv.version;
        let ok = only_reads || good;
        if (!ok) {
          const btns = [{label: T("Allow"), value: "yes", primary: true}];
          if (srv) btns.push({label: T("Allow always in this chat"), value: "always"});
          btns.push({label: T("Don't allow"), value: "no"});
          const b = (await dialog({title, html, buttons: btns})).button;
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
        const tm = {role: "tool", tool_call_id: c.id, name, args,
                    content: res.result, denied: !!res.denied};
        /* a view of MCP Apps: where it comes from, and the result for it */
        if (res.view && res.raw) { tm.view = res.view; tm.raw = res.raw; }
        cur.messages.push(tm);
        if (tm.view) artShowView(tm); /* beside the chat, as an artifact */
        if (res.opened) await readTools(); /* the service's tools, from now */
        drawLog();
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      const last = cur.messages[cur.messages.length - 1];
      if (last && last.role === "assistant" && !last.speed) last.speed = T("failed: {error}", {error: err.message});
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
  /* the start's own chat opened first: a message sent before would go to a
     chat the start then replaces, its reply written into the new one */
  await booted;
  const text = $("#ask").value.trim();
  /* a title being asked is let finish (a few words): a request cut short
     while the model reads it left janas-server's next reply wrong */
  if (text && titleRun) await titleRun;
  if (!text) { if (ev.type === "submit") dictate(); return; }
  if (!ws.path) { wsShow(true); return; }
  if (!await ensureModel()) return;
  const fresh = !cur;
  if (fresh) newConv();
  const um = {role: "user", content: text, t: Date.now()};
  if (pending.length) { um.attached = pending; pending = []; drawChips(); }
  cur.messages.push(um);
  if (!cur.title) { cur.title = provisionalTitle(text); cur.autoTitle = true; }
  cur.model = server.model;
  let e = convs.find((c) => c.id === cur.id);
  if (!e) convs.unshift(e = {id: cur.id, title: "", model: cur.model, updated: Date.now(), messages: 0});
  e.title = cur.title;
  $("#ask").value = ""; growAsk();
  /* the question shown at once and followed: a chat that fills the
     model's memory is summarized first, under it, in sight (its line at
     the chat's end), the question itself kept whole after the summary */
  followLog = true;
  listConvs(); drawLog();
  if (!fresh) await compactIfFull();
  await runTurn(um);
  if (cur && cur.autoTitle && cur.messages.filter((m) => m.role === "user").length === 1) modelTitle(cur);
}
/* a chat's title until the model gives one: the first sentence of its
   first message, without marks, at most eight words */
function provisionalTitle(text) {
  let t = text.replace(/```[\s\S]*?(```|$)/g, " ").replace(/[`*_#>\[\]]/g, "").replace(/\s+/g, " ").trim();
  const s = t.match(/^.{8,}?[.?!;:](?=\s|$)/);
  if (s) t = s[0].replace(/[.;:]$/, "");
  const w = t.split(" ");
  if (w.length > 8) t = w.slice(0, 8).join(" ") + "…";
  if (t.length > 60) t = t.slice(0, 60).replace(/\s+\S*$/, "") + "…";
  return t || "New chat";
}
/* after the first reply, while the model is free: a title of a few words
   from it, unless the user named the chat; a message sent meanwhile waits
   for it (sendAsk) */
const TITLE_ASK = "Give this conversation a title of 2 to 6 words, in the language the user writes in. " +
  "Answer with the title alone: no quotes, no full stop.";
let titleRun = null;
function modelTitle(c) {
  if (busy || !server.running || titleRun) return;
  titleRun = askTitle(c).finally(() => { titleRun = null; });
}
async function askTitle(c) {
  const q = c.messages.find((m) => m.role === "user");
  const a = c.messages.find((m) => m.role === "assistant" && m.content && !m.failed);
  if (!q || !a) return;
  const clip = (t) => t.replace(/<artifact\b[^>]*title="([^"]*)"[\s\S]*?(<\/artifact>|$)/gi, '[artifact "$1"]').slice(0, 1500);
  try {
    const r = await fetch("/v1/chat/completions", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model: server.model, max_tokens: 24, temperature: 0.2,
        chat_template_kwargs: {enable_thinking: false},
        messages: [{role: "system", content: TITLE_ASK},
                   {role: "user", content: `User: ${clip(userContent(q))}\n\nAssistant: ${clip(a.content)}`}]})});
    const j = await r.json().catch(() => ({}));
    const t = (j.choices?.[0]?.message?.content || "").replace(/<think>[\s\S]*?<\/think>/g, "")
      .trim().split("\n")[0].replace(/^["'«“*#\s]+|["'»”*.\s]+$/g, "").trim();
    if (!r.ok || !t || t.length > 80 || !c.autoTitle) return;
    c.title = t; delete c.autoTitle;
    const e = convs.find((x) => x.id === c.id); if (e) e.title = t;
    await saveConv(c);
    listConvs();
  } catch (e) {
  }
}

