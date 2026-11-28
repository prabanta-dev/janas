/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* update.js - Brebus's page (app.html): its updates (update.c). Settings'
   card says which version this is, the latest, what Brebus is doing about
   it, and how they are installed; a new version installed is said once,
   at the top of the page, with a link to what changed. */

/* what Brebus is doing, said whole (the downloading's {pct} filled) */
const UPD_PHASE = {
  checking: "Now asking GitHub for the latest version.",
  downloading: "Now downloading it ({pct}%).",
  installing: "Now installing it: Brebus starts again in a moment.",
};
let updLast = null;

function updWhen(t) {
  return t ? dateTime(t * 1000) : T("never");
}

function updShow(u) {
  updLast = u;
  const sel = $("#updmode");
  if (document.activeElement !== sel) sel.value = u.mode;
  const fixed = u.kind === "system" || u.kind === "none";
  sel.disabled = fixed;
  const lv = {v: esc(u.version), latest: esc(u.latest || ""), when: esc(updWhen(u.last_check))};
  let line = !u.latest ? T("This is Brebus {v} (looked {when}).", lv)
    : u.newer ? T("This is Brebus {v}; {latest} is out (looked {when}).", lv)
    : T("This is Brebus {v}, the latest (looked {when}).", lv);
  if (fixed && u.why) line += ` ${esc((w => w[0].toUpperCase() + w.slice(1))(Ts(u.why)))}.`;
  const phase = UPD_PHASE[u.phase];
  if (phase)
    line += " " + T(phase, {pct: num(Math.round(u.pct))});
  else if (u.staged)
    line += " " + (u.mode === "auto" ? T("{v} is downloaded and checked: it is installed as Brebus closes.", {v: esc(u.staged)})
                                     : T("{v} is downloaded and checked: install it when you like.", {v: esc(u.staged)}));
  if (u.error) line += ` <b>${esc(Ts(u.error))}</b>`;
  $("#updline").innerHTML = line;
  $("#updinst").hidden = !u.staged || u.phase === "installing";
  /* the side bar's button, as an app says it: ready, restart */
  const btn = $("#updbtn"), lbl = btn.querySelector(".lbl");
  if (u.phase === "installing") {
    btn.hidden = false; btn.disabled = true; lbl.textContent = T("Restarting to update...");
  } else if (u.staged) {
    btn.hidden = false; btn.disabled = false; lbl.textContent = T("Update ready: restart");
    btn.title = T("Brebus {v} is downloaded and checked: restart Brebus to finish the update (or it is installed when Brebus closes)", {v: u.staged});
  } else if (u.phase === "downloading") {
    btn.hidden = false; btn.disabled = true; lbl.textContent = T("Downloading an update ({pct}%)", {pct: num(Math.round(u.pct))});
  } else btn.hidden = true;
  $("#updcheck").disabled = u.mode === "off" || u.phase === "checking" || u.phase === "downloading";
  /* installed at the last start: said once */
  if (u.done) {
    const n = $("#updnote");
    n.innerHTML = (u.from ? T("Brebus was updated to {v} from {from}.", {v: esc(u.done), from: esc(u.from)})
                          : T("Brebus was updated to {v}.", {v: esc(u.done)})) + " " +
      `<a href="https://github.com/prabanta-dev/brebus/releases/tag/${encodeURIComponent(u.done)}" target="_blank" rel="noopener">${T("What changed")}</a> ` +
      `<button type="button" class="act quiet" id="updseen">${T("OK")}</button>`;
    n.hidden = false;
    $("#updseen").onclick = () => { n.hidden = true; updPost({seen: true}); };
  }
}

async function updPost(what) {
  try {
    const r = await fetch("/brebus/update", {method: "POST", body: JSON.stringify(what)});
    const j = await r.json();
    if (j.version) updShow(j);
  } catch (e) {}
}

async function updRead() {
  try { updShow(await ask("update")); } catch (e) {}
}

$("#updmode").onchange = (e) => updPost({mode: e.target.value});
$("#updcheck").onclick = () => updPost({check: true});
$("#updinst").onclick = () => updPost({install: true});
$("#updbtn").onclick = () => updPost({install: true});
updRead();
/* while something happens, more often */
setInterval(() => {
  if (!updLast || UPD_PHASE[updLast.phase] || $("#prefs").classList.contains("on")) updRead();
}, 5000);
