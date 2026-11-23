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
   the pane's width. */

const ZOOMS = [0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3];
/* the zoom one step from z: dir 1 larger, -1 smaller, 0 back to 100% */
function zoomStep(z, dir) {
  if (!dir) return 1;
  return dir > 0 ? ZOOMS.find((x) => x > z + 0.001) || z
                 : [...ZOOMS].reverse().find((x) => x < z - 0.001) || z;
}
const pct = (z) => Math.round(z * 100) + "%";
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
if (!ZOOMS.includes(uiZoom)) uiZoom = 1;

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
  const card = $("#zoomcard");
  if (inWindow) {
    card.innerHTML = `<div class="btnline"><span>Size of the interface</span>
      <span class="seg zoomseg"><button data-z="-1" title="Smaller (Ctrl+minus)">−</button><button data-z="0" title="Back to 100% (Ctrl+0)"><b>${pct(uiZoom)}</b></button><button data-z="1" title="Larger (Ctrl+plus)">+</button></span></div>
      <span class="note">Ctrl+wheel, Ctrl+plus and Ctrl+minus change it anywhere in Brebus; Ctrl+0 brings it back to 100%. Brebus remembers it.</span>
      <span class="note">An artifact has a zoom of its own, in its bar: Ctrl+wheel over it zooms it alone.</span>`;
    for (const b of card.querySelectorAll("[data-z]")) b.onclick = () => setZoom(zoomStep(uiZoom, +b.dataset.z));
  } else
    card.innerHTML = `<span class="note">In a browser the size of the interface is the browser's own zoom: Ctrl+plus, Ctrl+minus, Ctrl+0 for 100%.</span>
      <span class="note">An artifact has a zoom of its own, in its bar: Ctrl+wheel over it zooms it alone.</span>`;
}
for (const b of $("#uizoom").querySelectorAll("[data-z]"))
  b.onclick = () => setZoom(zoomStep(uiZoom, +b.dataset.z));
drawZoom();
if (inWindow && uiZoom !== 1) setZoom(uiZoom);

/* ---- an artifact's own, kept for it (by its title, a picture by its
   path): the last ones used, at most 200 ---- */
const artZooms = (() => { try { return JSON.parse(localStorage.getItem("brebus.artzoom") || "{}"); } catch (e) { return {}; } })();
const artZoomKey = () => artShown ? "a:" + artShown.title : fileShown ? "f:" + fileShown.path : "";
function artZoomNow() {
  const z = +artZooms[artZoomKey()];
  return ZOOMS.includes(z) ? z : 1;
}
/* the artifact shown drawn at its zoom: a frame told, a text or a
   picture zoomed in the pane */
function artZoomApply() {
  const z = artZoomNow();
  $("#artzpct").textContent = pct(z);
  if (artFrame && artFrame.el.contentWindow)
    artFrame.el.contentWindow.postMessage({janasZoom: z}, "*");
  else if ($("#artbody").firstElementChild)
    $("#artbody").firstElementChild.style.zoom = z === 1 ? "" : z;
}
function artZoomSet(dir) {
  const key = artZoomKey();
  if (!key) return;
  const z = zoomStep(artZoomNow(), dir);
  delete artZooms[key];
  if (z !== 1) artZooms[key] = z;
  const keys = Object.keys(artZooms);
  for (const k of keys.slice(0, Math.max(0, keys.length - 200))) delete artZooms[k];
  try { localStorage.setItem("brebus.artzoom", JSON.stringify(artZooms)); } catch (e) {}
  artZoomApply();
}
for (const b of $("#artzoom").children) b.onclick = () => artZoomSet(+b.dataset.z);

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
