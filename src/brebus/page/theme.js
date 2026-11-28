/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* theme.js - Brebus's page (app.html): its theme, a menu at the foot of
   the side bar beside the interface's zoom. Auto follows the system, as
   it changes; light and dark are the user's, kept in this browser. The
   page's first script draws it before anything else (app.html); here it is
   chosen and kept, and the artifact shown drawn again in it. */

/* the labels in English, put into the page's language where shown */
const THEMES = [["auto", "Auto", "As the system"], ["light", "Light", ""], ["dark", "Dark", ""]];
let theme = "auto";
try { theme = localStorage.getItem("brebus.theme") || "auto"; } catch (e) {}
if (!THEMES.some((t) => t[0] === theme)) theme = "auto";
const sysDark = matchMedia("(prefers-color-scheme: dark)");

function themeApply() {
  const dark = theme === "dark" || (theme === "light" ? false : sysDark.matches);
  const was = isDark();
  document.documentElement.classList.toggle("dark", dark);
  const t = THEMES.find((x) => x[0] === theme);
  const b = $("#themebtn");
  const name = t[1];
  b.querySelector("b").textContent = T(name);
  b.setAttribute("aria-label", T("Theme: {t}", {t: T(name)}));
  /* an artifact drawn in the other theme (its diagrams, its code's
     colours) is drawn again */
  if (was !== dark && typeof artShown !== "undefined" && artShown)
    artShow(artShown.title, artShown.k);
  /* the views of MCP Apps told */
  if (was !== dark && typeof viewsTheme === "function") viewsTheme();
}
function themeSet(t) {
  theme = t;
  try { if (t === "auto") localStorage.removeItem("brebus.theme"); else localStorage.setItem("brebus.theme", t); } catch (e) {}
  themeApply();
}
$("#themebtn").onclick = (e) => popMenu(e.currentTarget, THEMES.map(([v, label, note]) =>
  ({label: T(label), note: note && T(note), on: v === theme, run: () => themeSet(v)})), true);
sysDark.addEventListener("change", () => { if (theme === "auto") themeApply(); });
themeApply();
