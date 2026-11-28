/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* language.js - Brebus's page (app.html): Settings' choice of its language
   (i18n.js): as the system, or one of those it speaks, each by its own
   name. Kept in this browser; the page is loaded again in it. */

(() => {
  const sel = $("#langsel");
  const sys = LANGS.find((x) => x[0] === LANG);
  const chosen = langChosen();
  sel.innerHTML = `<option value="">${esc(T("As the system"))}${chosen ? "" : ` (${esc(sys[1])})`}</option>` +
    LANGS.map(([c, name]) => `<option value="${c}" lang="${c}">${esc(name)}</option>`).join("");
  sel.value = LANGS.some((x) => x[0] === chosen) ? chosen : "";
  sel.onchange = () => langSet(sel.value);
})();
