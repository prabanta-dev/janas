/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* float.js - a map's view over the whole of its room: when the host gives
   the view all of it (displayMode fullscreen: html.fill), the map fills it
   and the rest floats over it, in panels one moves by dragging them: the
   title, the legend, the commands, the list and the credit in one, which
   folds to its title; the card of details opens in it, between the
   commands and the list, which gives it room (the panel unfolded if it
   was folded, and folded again when the card closes).
   Elsewhere (inline) the page stays as written. Needs base.css and app.js. */
const Floats = (() => {
  const W = {
    fold: {en: "Fold", it: "Riduci", fr: "Réduire", es: "Contraer", de: "Einklappen", pt: "Recolher"},
    unfold: {en: "Unfold", it: "Espandi", fr: "Déplier", es: "Desplegar", de: "Ausklappen", pt: "Expandir"},
  };
  const t = (k) => App.t(W[k]);
  const root = document.documentElement;
  let on = false, side = null, head = null, body = null, fold = null, order = [], card = null, wasFolded = null;
  /* a panel moved by dragging handle (not its buttons, links, lists),
     kept inside the room */
  function drag(panel, handle) {
    handle.addEventListener("pointerdown", (e) => {
      if (!on || e.button !== 0 || e.target.closest("button, a, input, select, label, li, .xpanel, .vclose")) return;
      const r = panel.getBoundingClientRect(), x0 = e.clientX, y0 = e.clientY;
      let moved = false;
      const move = (m) => {
        const dx = m.clientX - x0, dy = m.clientY - y0;
        if (!moved && Math.hypot(dx, dy) < 4) return;
        if (!moved) { moved = true; panel.classList.add("moving"); }
        m.preventDefault();
        panel.style.left = Math.max(0, Math.min(window.innerWidth - 60, r.left + dx)) + "px";
        panel.style.top = Math.max(0, Math.min(window.innerHeight - 32, r.top + dy)) + "px";
        panel.style.right = "auto"; panel.style.bottom = "auto";
      };
      const up = () => {
        panel.classList.remove("moving");
        window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up);
      };
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up);
    });
  }
  function words() { fold.title = t(side.classList.contains("folded") ? "unfold" : "fold"); fold.setAttribute("aria-label", fold.title); }
  /* into panels, or back as written */
  function apply() {
    const want = root.classList.contains("fill");
    if (want === on) return;
    on = want;
    root.classList.toggle("float", on);
    const h1 = document.querySelector("body > h1, .fhead > h1");
    if (on) {
      order = [...document.body.children];
      if (h1) head.prepend(h1);
      for (const el of order)
        if (el !== h1 && el.tagName !== "SCRIPT" && !el.classList.contains("mapbox") && !el.classList.contains("mapgrip") && el !== card)
          body.appendChild(el);
      if (card) {
        card.classList.add("fcard");
        /* under the commands, over the list */
        const tools = body.querySelector(":scope > .vtools"), list = body.querySelector(":scope > ol");
        if (tools) tools.after(card); else if (list) list.before(card); else body.prepend(card);
      }
      document.body.appendChild(side);
      cardShown();
      words();
    } else {
      for (const el of order) document.body.appendChild(el);
      side.remove();
      if (card) card.classList.remove("fcard");
    }
    window.dispatchEvent(new Event("resize"));
  }
  /* the card opened: the panel unfolded to show it; closed: folded again
     if it was */
  function cardShown() {
    if (!on || !card) return;
    if (!card.hidden && wasFolded === null) { wasFolded = side.classList.contains("folded"); side.classList.remove("folded"); words(); }
    else if (card.hidden && wasFolded !== null) { side.classList.toggle("folded", wasFolded); wasFolded = null; words(); }
  }
  function init() {
    if (side) return;
    side = document.createElement("div");
    side.className = "fpanel fside";
    side.innerHTML = `<div class="fhead"><button type="button" class="ffold"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 15l6-6 6 6"/></svg></button></div><div class="fbody"></div>`;
    head = side.firstChild; body = side.lastChild; fold = head.querySelector(".ffold");
    fold.onclick = () => { side.classList.toggle("folded"); words(); };
    drag(side, head);
    card = document.getElementById("card");
    if (card) new MutationObserver(cardShown).observe(card, {attributes: true, attributeFilter: ["hidden"]});
    new MutationObserver(apply).observe(root, {attributes: true, attributeFilter: ["class"]});
    apply();
  }
  return {init};
})();
