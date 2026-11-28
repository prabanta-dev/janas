/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* zoom.js - Brebus's page (app.html): the interface's zoom and each
   artifact's own. build.sh puts it into the page where app.html includes it.
   The interface: in Brebus's window (its address says ?window) the page
   keeps the zoom and asks it of the window (/brebus/zoom), which draws the
   whole page larger as a browser's zoom does: Ctrl+wheel, Ctrl+plus,
   Ctrl+minus, Ctrl+0 back to 100%, the side bar's foot (Interface) and
   Settings › Appearance. In a browser the browser's own zoom does it.
   An artifact: its own zoom, in its bar or Ctrl+wheel over it, kept for
   it; a page or a drawing keeps its layout and scrolls, a text flows to
   the pane's width; or fitted to the pane's width.
   Both are menus: their item the zoom now, opening on Larger, Smaller
   (25% a step, from 25% to 400%), the artifact's Fit to the window, and
   the usual levels. */

const ZMIN = 0.25, ZMAX = 4, ZSTEP = 0.25;
const ZLEVELS = [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2];
const zclamp = (z) => Math.min(ZMAX, Math.max(ZMIN, +z || 1));
/* the zoom one step from z: dir 1 larger, -1 smaller, 0 back to 100%;
   a zoom between steps (fitted, or kept from before) goes to the next */
function zoomStep(z, dir) {
  if (!dir) return 1;
  const k = z / ZSTEP;
  return zclamp((dir > 0 ? Math.floor(k + 1e-6) + 1 : Math.ceil(k - 1e-6) - 1) * ZSTEP);
}
const pct = (z) => Math.round(z * 100) + "%";
/* the menu of a zoom: now, set(z), fit (null: no fitting) and fitted */
function zoomMenu(anchor, now, set, fit, fitted, up) {
  const items = [
    {label: T("Larger"), note: T("Ctrl+plus"), icon: "zin", run: () => set(zoomStep(now, 1))},
    {label: T("Smaller"), note: T("Ctrl+minus"), icon: "zout", run: () => set(zoomStep(now, -1))}];
  if (fit) items.push({label: T("Fit to the window"), on: fitted, icon: "zfit", run: fit});
  items.push("-");
  for (const z of ZLEVELS)
    items.push({label: pct(z), on: !fitted && Math.abs(z - now) < 0.001, run: () => set(z)});
  popMenu(anchor, items, up);
}
/* a wheel turned with Ctrl, in steps: a notch is one, a touchpad's pinch
   gathers its small turns into one */
function wheelStep(st, e) {
  const now = Date.now();
  if (now - st.t > 400) st.acc = 0;
  st.t = now;
  st.acc += e.deltaY * (e.deltaMode === 1 ? 40 : e.deltaMode === 2 ? 400 : 1);
  if (Math.abs(st.acc) < 40) return 0;
  const dir = st.acc < 0 ? 1 : -1;
  st.acc = 0;
  return dir;
}
/* Ctrl with plus, minus or 0: the step, or null */
function keyStep(e) {
  if (!(e.ctrlKey || e.metaKey) || e.altKey) return null;
  return e.key === "+" || e.key === "=" ? 1 : e.key === "-" || e.key === "_" ? -1 : e.key === "0" ? 0 : null;
}

/* ---- the interface's ---- */
let inWindow = /[?&]window\b/.test(location.search);
try {
  if (inWindow) sessionStorage.setItem("brebus.window", "1");
  inWindow = inWindow || sessionStorage.getItem("brebus.window") === "1";
} catch (e) {}
let uiZoom = 1;
try { uiZoom = +localStorage.getItem("brebus.zoom") || 1; } catch (e) {}
uiZoom = zclamp(uiZoom);

async function setZoom(z) {
  if (!inWindow) return;
  const r = await fetch("/brebus/zoom", {method: "POST", body: JSON.stringify({zoom: z})}).catch(() => null);
  if (!r || !r.ok) return; /* no window after all: the browser's zoom */
  uiZoom = z;
  try { if (z === 1) localStorage.removeItem("brebus.zoom"); else localStorage.setItem("brebus.zoom", String(z)); } catch (e) {}
  drawZoom();
}
function drawZoom() {
  /* in the side bar, always: its size at a glance, steps for who does not
     use the keys */
  const u = $("#uizoom");
  u.hidden = !inWindow;
  u.querySelector("b").textContent = pct(uiZoom);
  u.classList.toggle("on", uiZoom !== 1);
  u.setAttribute("aria-label", T("Interface zoom {z}", {z: pct(uiZoom)}));
  const card = $("#zoomcard");
  if (inWindow) {
    card.innerHTML = `<div class="btnline"><span>${T("Size of the interface")}</span>
      <button type="button" class="zmenu" id="uizoom2" aria-haspopup="menu" title="${T("Larger, smaller, or a level (Ctrl+plus, Ctrl+minus, Ctrl+0)")}"><b>${pct(uiZoom)}</b><span class="caret">${icon("down")}</span></button></div>
      <span class="note">${T("Ctrl+wheel, Ctrl+plus and Ctrl+minus change it anywhere in Brebus; Ctrl+0 brings it back to 100%. Brebus remembers it.")}</span>
      <span class="note">${T("An artifact has a zoom of its own, in its bar: Ctrl+wheel over it zooms it alone.")}</span>`;
    $("#uizoom2").onclick = (e) => zoomMenu(e.currentTarget, uiZoom, setZoom, null, false, false);
  } else
    card.innerHTML = `<span class="note">${T("In a browser the size of the interface is the browser's own zoom: Ctrl+plus, Ctrl+minus, Ctrl+0 for 100%.")}</span>
      <span class="note">${T("An artifact has a zoom of its own, in its bar: Ctrl+wheel over it zooms it alone.")}</span>`;
}
$("#uizoom").onclick = (e) => zoomMenu(e.currentTarget, uiZoom, setZoom, null, false, true);
drawZoom();
if (inWindow && uiZoom !== 1) setZoom(uiZoom);

/* ---- an artifact's own, kept for it (by its title, a picture by its
   path): the last ones used, at most 200 ---- */
const artZooms = (() => { try { return JSON.parse(localStorage.getItem("brebus.artzoom") || "{}"); } catch (e) { return {}; } })();
const artZoomKey = () => artShown ? "a:" + artShown.title : fileShown ? "f:" + fileShown.path : "";
/* the artifact fitted to the pane: its zoom as the frame, or the picture,
   measured it (artZoomApply) */
let artFitted = 1;
const artFitOn = () => artZooms[artZoomKey()] === "fit";
function artZoomNow() {
  if (artFitOn()) return artFitted;
  const z = +artZooms[artZoomKey()];
  return z > 0 ? zclamp(z) : 1;
}
/* the artifact shown drawn at its zoom: a frame told, a text or a
   picture zoomed in the pane */
function artZoomApply() {
  const fit = artFitOn(), el = $("#artbody").firstElementChild;
  if (fit && !(artFrame && artFrame.el.contentWindow) && el) {
    /* a picture, or a text, to the pane's width: never larger than 400% */
    el.style.zoom = "";
    const w = el.tagName === "IMG" ? el.naturalWidth : el.scrollWidth;
    artFitted = w ? zclamp($("#artbody").clientWidth / w) : 1;
  }
  const z = artZoomNow();
  $("#artzpct").textContent = fit ? T("Fit {z}", {z: pct(z)}) : pct(z);
  if (artFrame && artFrame.el.contentWindow)
    artFrame.el.contentWindow.postMessage(fit ? {janasZoomFit: true} : {janasZoom: z}, "*");
  else if (el)
    el.style.zoom = z === 1 ? "" : z;
}
/* the zoom the frame chose when fitting itself: the bar shows it */
function artZoomFitted(z) {
  if (!artFitOn() || !(z > 0)) return;
  artFitted = z;
  $("#artzpct").textContent = T("Fit {z}", {z: pct(z)});
}
function artZoomKeep(key, v) {
  delete artZooms[key];
  if (v !== 1) artZooms[key] = v;
  const keys = Object.keys(artZooms);
  for (const k of keys.slice(0, Math.max(0, keys.length - 200))) delete artZooms[k];
  try { localStorage.setItem("brebus.artzoom", JSON.stringify(artZooms)); } catch (e) {}
  artZoomApply();
}
/* a step (dir 1, -1, 0), or a zoom of its own (to) */
function artZoomSet(dir, to) {
  const key = artZoomKey();
  if (!key) return;
  artZoomKeep(key, to !== undefined ? zclamp(to) : zoomStep(artZoomNow(), dir));
}
$("#artzoom").onclick = (e) => {
  if (!artZoomKey()) return;
  zoomMenu(e.currentTarget, artZoomNow(), (z) => artZoomSet(0, z),
           () => artZoomKeep(artZoomKey(), "fit"), artFitOn(), false);
};

/* Ctrl+wheel: over the artifact, its zoom (a frame tells its own, as
   janasZoomStep); elsewhere in the window, the interface's */
const wheelState = {acc: 0, t: 0};
addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  const art = e.target.closest && e.target.closest("#artbody");
  if (!art && !inWindow) return; /* the browser's zoom */
  e.preventDefault();
  const dir = wheelStep(wheelState, e);
  if (!dir) return;
  if (art) artZoomSet(dir);
  else setZoom(zoomStep(uiZoom, dir));
}, {passive: false});
addEventListener("keydown", (e) => {
  const dir = keyStep(e);
  if (dir === null || !inWindow) return;
  e.preventDefault();
  setZoom(zoomStep(uiZoom, dir));
});
