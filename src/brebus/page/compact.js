/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* compact.js - Brebus's page (app.html): a chat that fills the model's memory summarized. build.sh puts it into the
   page where app.html includes it.
   The earlier messages are summarized by the model itself and the summary
   takes their place in what it is sent (apiMessages): cur.summary =
   {upto, text, t, ctx}, the messages before upto summarized. They stay in
   the chat, under a line that opens to the summary. By itself when the
   chat is 80% of the model's memory, before the next message is sent; at
   will with Compact now (the chat's menu) or Summarize up to here (a
   reply's). The system message and the project's instructions are never
   summarized: they are sent whole. */

const COMPACT_AT = 0.8; /* of the context: summarized before the next */
const KEEP_TURNS = 2;   /* the last questions kept whole, with their replies */

/* the tokens the chat fills: its last reply's count, or the summary's
   estimate when it is newer */
function ctxUsed() {
  if (!cur) return 0;
  let at = -1, used = 0;
  cur.messages.forEach((m, k) => { if (m.ctx) { at = k; used = m.ctx; } });
  const s = cur.summary;
  if (s && s.ctx && s.upto <= cur.messages.length && s.after >= at) used = s.ctx;
  return used;
}
function ctxSize() {
  const name = server.running ? server.model : nextModel();
  return (settings[name] && +settings[name].context) || server.context ||
         (modelInfo[name] && modelInfo[name].context) || 0;
}
/* the summary in force; a chat gone back before its point (a rewind, a
   message changed, a branch from earlier) loses it: what follows will be
   other messages */
function summaryNow() {
  const s = cur && cur.summary;
  if (s && !(s.upto > 0 && s.upto <= cur.messages.length)) delete cur.summary;
  return cur && cur.summary || null;
}
/* where a summary can end: at a question of the user's, so that what
   follows starts with one; the point keeping the last keep questions
   whole, past the summary there is (0: none) */
function compactPoint(keep) {
  const ms = cur.messages, from = summaryNow() ? cur.summary.upto : 0;
  for (let k = ms.length - 1, left = keep; k > from; k--)
    if (ms[k].role === "user" && --left === 0) return k;
  return 0;
}
/* the conversation up to upto as plain text for the model to summarize:
   the summary before it, then each message; artifacts by their titles,
   tools' results cut short */
function transcript(upto) {
  const out = [], s = summaryNow();
  if (s) out.push("[Summary of what came before]\n" + s.text);
  for (const m of cur.messages.slice(s ? s.upto : 0, upto)) {
    if (m.failed) continue;
    let t = m.role === "user" ? userContent(m) : m.content || "";
    t = t.replace(/<artifact\b[^>]*title="([^"]*)"[^>]*>[\s\S]*?(<\/artifact>|$)/gi, '[artifact "$1"]');
    if (m.role === "tool") t = `${m.name || "tool"} answered: ${t.slice(0, 2000)}${t.length > 2000 ? " […]" : ""}`;
    else if (m.tool_calls) t += m.tool_calls.map((c) => `\n[called ${c.function.name}]`).join("");
    if (t.trim()) out.push(`${m.role === "user" ? "User" : m.role === "tool" ? "Tool" : "Assistant"}: ${t}`);
  }
  /* the model must have room to read it and to write the summary */
  const room = Math.max(4000, (ctxSize() - 1500) * 3);
  const all = out.join("\n\n");
  return all.length > room ? "[…]\n" + all.slice(all.length - room) : all;
}
const SUMMARY_ASK = "Summarize the conversation below so that it can go on from the summary alone, " +
  "in the language the user writes in. Keep: what the user asked and wants, decisions taken, " +
  "facts, numbers, names, files, code and artifacts named, what was done and what is still to do, " +
  "the user's preferences. Leave out greetings and repetitions. Plain text, at most 500 words.";

/* the messages before upto summarized: true when done */
async function compactChat(upto, why) {
  if (!cur || busy || upto <= 0 || !server.running) return false;
  /* this chat's, even if another one is opened meanwhile */
  const c = cur;
  const ctl = new AbortController();
  busy = ctl; sendButton();
  c.compacting = why; drawLog();
  try {
    const r = await fetch("/v1/chat/completions", {method: "POST", signal: ctl.signal,
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model: server.model, max_tokens: 1200, temperature: 0.2,
        chat_template_kwargs: {enable_thinking: false},
        messages: [{role: "system", content: SUMMARY_ASK},
                   {role: "user", content: transcript(upto)}]})});
    const j = await r.json().catch(() => ({}));
    let text = (j.choices?.[0]?.message?.content || "").replace(/<think>[\s\S]*?<\/think>\s*/g, "").trim();
    if (!r.ok || !text) {
      await say(`The earlier messages could not be summarized: ${j.error?.message || j.error || "the model wrote nothing"}. The chat goes on as it was.`, "Not summarized");
      return false;
    }
    c.summary = {upto, text, t: Date.now(), after: c.messages.length - 1};
    /* the chat open again meanwhile is read anew: the summary goes to it */
    const open = cur && cur.id === c.id ? cur : null;
    if (open) {
      open.summary = c.summary;
      /* what the chat fills now, until the next reply counts it */
      c.summary.ctx = Math.round(JSON.stringify(apiMessages()).length / 3.5);
    }
    await saveConv(open || c);
    return true;
  } catch (e) {
    return false;
  } finally {
    delete c.compacting;
    busy = null; sendButton(); drawLog();
  }
}
/* before a message is sent: the chat summarized when it fills COMPACT_AT
   of the model's memory */
async function compactIfFull() {
  const size = ctxSize();
  if (!size || ctxUsed() / size < COMPACT_AT) return;
  const k = compactPoint(KEEP_TURNS);
  if (k) await compactChat(k, "The chat fills the model's memory: summarizing the earlier messages…");
}
/* Compact now, from the chat's menu */
async function compactNow() {
  const k = compactPoint(1);
  if (!k) { await say("There is nothing earlier to summarize yet: the last question and its reply are always kept whole.", "Compact"); return; }
  await compactChat(k, "Summarizing the earlier messages…");
}
/* Summarize up to here, under a reply: everything up to it and its tools'
   rounds, to the next question */
async function summarizeTo(k) {
  const ms = cur.messages;
  let upto = k + 1;
  while (upto < ms.length && ms[upto].role !== "user") upto++;
  if (summaryNow() && upto <= cur.summary.upto) return;
  await compactChat(upto, "Summarizing up to here…");
}
/* the line in the chat where the summary ends, opened to read it */
function summaryLine() {
  const s = summaryNow();
  if (!s) return null;
  const d = document.createElement("details");
  d.className = "sumline";
  d.innerHTML = `<summary>Summarized up to here: the model reads the summary in their place</summary><div class="sumtext text"></div>`;
  renderText(d.querySelector(".sumtext"), s.text); /* Markdown, as the model writes it */
  return d;
}
function compactingLine() {
  if (!cur || !cur.compacting) return null;
  const d = document.createElement("div");
  d.className = "sumline busy";
  d.innerHTML = `<span class="working">${PINTA}</span> <span></span>`;
  d.lastElementChild.textContent = cur.compacting;
  return d;
}
