/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* live.js - a view kept up to date by itself: a button that updates it
   now, a switch, Auto update every, and how often; each time the view's own tool asked again with its
   arguments (App.callTool, where the host lets the view call its server's
   tools that only read), the answer handed to the view to draw without
   losing its zoom or its choice. How often is the view's to say, within
   what its source allows. Off by default; nothing asked while the view is
   out of sight. Needs app.js before it. */
const Live = (() => {
  const W = {
    live: {en: "Auto update every", it: "Auto aggiorna ogni", fr: "Mise à jour auto toutes les", es: "Autoactualizar cada", de: "Auto-Aktualisierung alle", pt: "Atualizar auto a cada"},
    now: {en: "Update now", it: "Aggiorna ora", fr: "Mettre à jour maintenant", es: "Actualizar ahora", de: "Jetzt aktualisieren", pt: "Atualizar agora"},
    sec: {en: "{n} s", it: "{n} s", fr: "{n} s", es: "{n} s", de: "{n} s", pt: "{n} s"},
    min: {en: "{n} min", it: "{n} min", fr: "{n} min", es: "{n} min", de: "{n} Min.", pt: "{n} min"},
    at: {en: "updated {h}", it: "aggiornato {h}", fr: "mis à jour {h}", es: "actualizado {h}", de: "aktualisiert {h}", pt: "atualizado {h}"},
    got: {en: "loaded {h}", it: "caricato {h}", fr: "chargé {h}", es: "cargado {h}", de: "geladen {h}", pt: "carregado {h}"},
    fail: {en: "not updated: {e}", it: "non aggiornato: {e}", fr: "pas mis à jour : {e}", es: "no actualizado: {e}", de: "nicht aktualisiert: {e}", pt: "não atualizado: {e}"},
  };
  const t = (k, v) => App.t(W[k]).replace(/\{(\w+)\}/g, (_, n) => v && v[n] !== undefined ? v[n] : "");
  const label = (s) => s < 60 ? t("sec", {n: s}) : t("min", {n: Math.round(s / 60)});
  /* when, with its day: 28/11/2026, 00:21:05 */
  const stamp = () => new Date().toLocaleString(App.ctx.locale, {day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit"});
  /* the switch, put in where: o = {choices: [seconds...], dflt, and
     either ask: async () => true when asked and drawn (throws on a
     failure), or args?: () => the arguments (else the tool's input) and
     onresult(r); onswitch?(on) } */
  function control(where, o) {
    /* when the data came, always: under the controls, or alone where
       the host lets the view ask nothing */
    const box = document.createElement("span");
    box.className = "live";
    if (!App.canCall() || !App.toolName()) {
      box.innerHTML = `<span class="note lwhen"></span>`;
      box.firstChild.textContent = t("got", {h: stamp()});
      where.prepend(box);
      return {on: () => false};
    }
    box.innerHTML = `<span class="lrow"><button type="button" class="xbtn lnow"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg></button>` +
      `<label><input type="checkbox"> <span></span></label> <select></select></span><span class="note lwhen"></span>`;
    const on = box.querySelector("input"), sel = box.querySelector("select"), note = box.querySelector(".note"), now = box.querySelector(".lnow");
    box.querySelector("label span").textContent = t("live");
    now.title = t("now"); now.setAttribute("aria-label", t("now"));
    for (const s of o.choices) {
      const op = document.createElement("option");
      op.value = s; op.textContent = label(s);
      sel.appendChild(op);
    }
    sel.value = String(o.dflt || o.choices[0]);
    let timer = null, busy = false;
    const schedule = () => { clearTimeout(timer); timer = on.checked ? setTimeout(tick, +sel.value * 1000) : null; };
    async function tick(asked) {
      if (busy || (!on.checked && !asked)) return;
      /* out of sight: asked when it comes back */
      if (document.hidden && !asked) { schedule(); return; }
      busy = true; now.classList.add("busy");
      try {
        /* the view asks by itself (o.ask: true when done, throws on a
           failure), or its tool asked again (o.args, else its input) */
        let done = false;
        if (o.ask) done = await o.ask();
        else {
          const r = await App.callTool(App.toolName(), o.args ? o.args() : App.input());
          if (!r || !r.structuredContent || r.isError) throw new Error((r && r.content && r.content[0] && r.content[0].text || "").slice(0, 80));
          o.onresult(r);
          done = true;
        }
        if (done) note.textContent = t("at", {h: stamp()});
      } catch (e) { note.textContent = t("fail", {e: e.message}); }
      busy = false; now.classList.remove("busy");
      schedule();
    }
    /* asked now, by its button: the next time by itself counted from here */
    now.onclick = () => tick(true);
    on.onchange = () => { box.classList.toggle("on", on.checked); if (o.onswitch) o.onswitch(on.checked); if (on.checked) tick(); else schedule(); };
    sel.onchange = schedule;
    document.addEventListener("visibilitychange", () => { if (!document.hidden && on.checked && !busy) tick(); });
    note.textContent = t("got", {h: stamp()});
    where.prepend(box);
    return {on: () => on.checked};
  }
  return {control};
})();
