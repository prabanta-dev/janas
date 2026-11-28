/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* map.js - the views' map: the land of world.js (Natural Earth) drawn on a
   canvas in Web Mercator, nothing fetched from anywhere; or, when the user
   asks and the host is Brebus, a map of OpenStreetMap's data over it (the
   layers' menu: OpenStreetMap, OpenTopoMap), whose tiles Brebus
   fetches and keeps (/brebus/tile/<layer>/z/x/y.png, its own address: the view
   still reaches nothing else), with their attribution. Dragged to move;
   the wheel or the buttons to zoom; the lens (or Shift and a drag) zooms
   to a box drawn; a grip under it makes it taller or shorter. Points on
   it drawn by the view, the nearest one under a click handed back. Needs
   world.js before it, app.js for its words and links. */
const MAP_W = {
  layers: {en: "Maps", it: "Mappe", fr: "Cartes", es: "Mapas", de: "Karten", pt: "Mapas"},
  plain: {en: "Plain, drawn here (nothing fetched)", it: "Semplice, disegnata qui (non scarica nulla)", fr: "Simple, dessinée ici (rien n'est téléchargé)", es: "Sencillo, dibujado aquí (no descarga nada)", de: "Einfach, hier gezeichnet (nichts geladen)", pt: "Simples, desenhado aqui (não descarrega nada)"},
  osm: {en: "OpenStreetMap", it: "OpenStreetMap", fr: "OpenStreetMap", es: "OpenStreetMap", de: "OpenStreetMap", pt: "OpenStreetMap"},
  topo: {en: "OpenTopoMap: relief and paths", it: "OpenTopoMap: rilievi e sentieri", fr: "OpenTopoMap : relief et sentiers", es: "OpenTopoMap: relieve y senderos", de: "OpenTopoMap: Relief und Wege", pt: "OpenTopoMap: relevo e trilhos"},
  save: {en: "Save the map", it: "Salva la mappa", fr: "Enregistrer la carte", es: "Guardar el mapa", de: "Die Karte speichern", pt: "Guardar o mapa"},
  png: {en: "Picture (PNG), as it looks", it: "Immagine (PNG), com'è ora", fr: "Image (PNG), telle qu'elle est", es: "Imagen (PNG), tal como se ve", de: "Bild (PNG), wie sie aussieht", pt: "Imagem (PNG), tal como está"},
  svg: {en: "Drawing (SVG), in vectors", it: "Disegno (SVG), vettoriale", fr: "Dessin (SVG), vectoriel", es: "Dibujo (SVG), vectorial", de: "Zeichnung (SVG), als Vektoren", pt: "Desenho (SVG), vetorial"},
  fetched: {en: "fetched by Brebus, from OpenStreetMap's data", it: "le scarica Brebus, dai dati di OpenStreetMap", fr: "téléchargées par Brebus, des données d'OpenStreetMap", es: "las descarga Brebus, de los datos de OpenStreetMap", de: "von Brebus geladen, aus den Daten von OpenStreetMap", pt: "descarregadas pelo Brebus, dos dados do OpenStreetMap"},
  circle: {en: "Draw a circle: press at its centre and drag to its edge", it: "Disegna un cerchio: premi al centro e trascina fino al bordo", fr: "Tracer un cercle : appuyez en son centre et glissez jusqu'au bord", es: "Dibujar un círculo: pulsa en su centro y arrastra hasta el borde", de: "Einen Kreis ziehen: in seiner Mitte drücken und bis zum Rand ziehen", pt: "Desenhar um círculo: carrega no centro e arrasta até à borda"},
  nocircle: {en: "Remove the circle", it: "Togli il cerchio", fr: "Retirer le cercle", es: "Quitar el círculo", de: "Kreis entfernen", pt: "Remover o círculo"},
  lens: {en: "Zoom to a box: draw it on the map (or Shift and drag)", it: "Ingrandisci un riquadro: disegnalo sulla mappa (o Maiusc e trascina)", fr: "Zoomer sur un cadre : tracez-le sur la carte (ou Maj et glisser)", es: "Ampliar un recuadro: dibújalo en el mapa (o Mayús y arrastra)", de: "Auf einen Rahmen zoomen: ziehe ihn auf der Karte (oder Umschalt und ziehen)", pt: "Ampliar uma área: desenha-a no mapa (ou Shift e arrasta)"},
  grip: {en: "Drag to make the map taller or shorter", it: "Trascina per allungare o accorciare la mappa", fr: "Glisser pour agrandir ou réduire la carte", es: "Arrastra para alargar o acortar el mapa", de: "Ziehen, um die Karte höher oder niedriger zu machen", pt: "Arrasta para aumentar ou diminuir a altura do mapa"},
};
const mapT = (k) => typeof App !== "undefined" ? App.t(MAP_W[k]) : MAP_W[k].en;
/* a view's credit line as true for the map shown: "drawn here, nothing
   fetched" said of the plain map alone (the views' credits end so, in
   their six languages) */
const MAP_PLAIN = [", drawn here, nothing fetched", ", disegnata qui, senza scaricare nulla", ", dessinée ici, rien n'est téléchargé",
  ", dibujado aquí, sin descargar nada", ", hier gezeichnet, nichts geladen", ", desenhado aqui, sem descarregar nada"];
function mapCredit(text, layer) {
  if (!layer) return text;
  for (const p of MAP_PLAIN) text = text.replace(p, "");
  return text;
}
/* a canvas's drawing turned into SVG: what the map and the views draw with
   (paths, arcs, rectangles, dashes, a rotation, the tiles as pictures),
   each shape an element with its own colours */
function SvgCtx(W, H) {
  const out = [];
  let m = [1, 0, 0, 1, 0, 0], d = "", stack = [];
  const st = {fillStyle: "#000", strokeStyle: "#000", lineWidth: 1, dash: []};
  const pt = (x, y) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
  const f = (v) => Math.round(v * 100) / 100;
  const attr = (s) => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
  const scale = () => Math.sqrt(Math.abs(m[0] * m[3] - m[1] * m[2])) || 1;
  const ctx = {
    get fillStyle() { return st.fillStyle; }, set fillStyle(v) { st.fillStyle = v; },
    get strokeStyle() { return st.strokeStyle; }, set strokeStyle(v) { st.strokeStyle = v; },
    get lineWidth() { return st.lineWidth; }, set lineWidth(v) { st.lineWidth = v; },
    setLineDash(a) { st.dash = a || []; },
    setTransform(a, b, c, dd, e, ff) { m = [a, b, c, dd, e, ff]; },
    translate(x, y) { m = [m[0], m[1], m[2], m[3], m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]; },
    rotate(r) { const c = Math.cos(r), s = Math.sin(r); m = [m[0] * c + m[2] * s, m[1] * c + m[3] * s, -m[0] * s + m[2] * c, -m[1] * s + m[3] * c, m[4], m[5]]; },
    scale(x, y) { m = [m[0] * x, m[1] * x, m[2] * y, m[3] * y, m[4], m[5]]; },
    save() { stack.push([m, {...st, dash: st.dash.slice()}]); },
    restore() { const s = stack.pop(); if (s) { m = s[0]; Object.assign(st, s[1]); } },
    beginPath() { d = ""; },
    moveTo(x, y) { const p = pt(x, y); d += "M" + f(p[0]) + " " + f(p[1]); },
    lineTo(x, y) { const p = pt(x, y); d += "L" + f(p[0]) + " " + f(p[1]); },
    closePath() { d += "Z"; },
    arc(x, y, r, a0, a1) {
      const n = Math.max(12, Math.min(64, Math.ceil(r)));
      for (let i = 0; i <= n; i++) {
        const a = a0 + (a1 - a0) * i / n, p = pt(x + r * Math.cos(a), y + r * Math.sin(a));
        d += (i || !d ? (i ? "L" : "M") : "L") + f(p[0]) + " " + f(p[1]);
      }
    },
    rect(x, y, w, h) { ctx.moveTo(x, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + h); ctx.lineTo(x, y + h); ctx.closePath(); },
    fill() { if (d) out.push(`<path d="${d}" fill="${attr(st.fillStyle)}" fill-rule="nonzero"/>`); },
    stroke() {
      if (d) out.push(`<path d="${d}" fill="none" stroke="${attr(st.strokeStyle)}" stroke-width="${f(st.lineWidth * scale())}" stroke-linejoin="round"` +
        (st.dash.length ? ` stroke-dasharray="${st.dash.map((v) => f(v * scale())).join(" ")}"` : "") + "/>");
    },
    fillRect(x, y, w, h) { const k = d; d = ""; ctx.rect(x, y, w, h); ctx.fill(); d = k; },
    strokeRect(x, y, w, h) { const k = d; d = ""; ctx.rect(x, y, w, h); ctx.stroke(); d = k; },
    clearRect() {},
    drawImage(im, x, y, w, h) {
      /* a tile, as a picture inside the drawing */
      try {
        const c = document.createElement("canvas"); c.width = im.naturalWidth; c.height = im.naturalHeight;
        c.getContext("2d").drawImage(im, 0, 0);
        const p = pt(x, y), s = scale();
        out.push(`<image x="${f(p[0])}" y="${f(p[1])}" width="${f(w * s)}" height="${f(h * s)}" href="${c.toDataURL("image/png")}"/>`);
      } catch (e) {}
    },
    /* a line of text at the bottom right, on white: a map's attribution */
    note(text) {
      const w = Math.min(W - 8, text.length * 5.6 + 10);
      out.push(`<rect x="${f(W - w - 4)}" y="${H - 18}" width="${f(w)}" height="14" rx="3" fill="#ffffffcc"/>` +
        `<text x="${W - 9}" y="${H - 7.5}" font-family="sans-serif" font-size="10" text-anchor="end" fill="#333">${attr(text)}</text>`);
    },
    svg() {
      return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">` +
        `<defs><clipPath id="v"><rect width="${W}" height="${H}"/></clipPath></defs><g clip-path="url(#v)">\n${out.join("\n")}\n</g></svg>\n`;
    },
  };
  return ctx;
}

/* how far and which way from one point to another: [km, "NNE"] */
function geoFar(la0, lo0, la1, lo1) {
  const k = Math.PI / 180, dl = (lo1 - lo0) * k;
  const a = Math.sin((la1 - la0) * k / 2) ** 2 + Math.cos(la0 * k) * Math.cos(la1 * k) * Math.sin(dl / 2) ** 2;
  const b = Math.atan2(Math.sin(dl) * Math.cos(la1 * k), Math.cos(la0 * k) * Math.sin(la1 * k) - Math.sin(la0 * k) * Math.cos(la1 * k) * Math.cos(dl));
  const C = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  return [2 * 6371 * Math.asin(Math.min(1, Math.sqrt(a))), C[Math.round(((b / k + 360) % 360) / 22.5) % 16]];
}
/* whether a point is inside bounds [south, west, north, east] */
function geoIn(b, lat, lon) {
  return lat >= b[0] && lat <= b[2] && (b[1] <= b[3] ? lon >= b[1] && lon <= b[3] : lon >= b[1] || lon <= b[3]);
}
/* a circle of interest drawn on a map: a light tint inside, its line
   blue dashed over a white halo, its centre a cross the same way - seen
   on every map, plain, streets or relief, and a little seen through, the
   map's names and roads under it */
const RING_C = "#1d4ed8";
function ringLine(g, dash) {
  g.setLineDash([]); g.strokeStyle = "#ffffff80"; g.lineWidth = 5; g.stroke();
  g.setLineDash(dash); g.strokeStyle = RING_C + "b3"; g.lineWidth = 2.5; g.stroke(); g.setLineDash([]);
}
function drawCircle(g, at, c) {
  const [cx, cy] = at(c.lon, c.lat), [ex] = at(c.lon + c.km / (111.32 * Math.cos(c.lat * Math.PI / 180)), c.lat);
  g.beginPath(); g.arc(cx, cy, Math.abs(ex - cx), 0, 2 * Math.PI);
  g.fillStyle = "#1d4ed80f"; g.fill();
  ringLine(g, [8, 5]);
  g.beginPath(); g.moveTo(cx - 7, cy); g.lineTo(cx + 7, cy); g.moveTo(cx, cy - 7); g.lineTo(cx, cy + 7);
  ringLine(g, []);
}
function MapView(box, opts) {
  opts = opts || {};
  const cv = document.createElement("canvas");
  cv.className = "map";
  box.appendChild(cv);
  let g = cv.getContext("2d"); /* the canvas's, or an SVG's while one is made */
  /* the land, once: rings of [x, y] in Mercator units (0..1 across the world) */
  const rings = [];
  for (const r of WORLD.split(";")) {
    const v = r.split(",").map(Number), pts = [];
    let x = v[0], y = v[1];
    pts.push(merc(x / 100, y / 100));
    for (let i = 2; i + 1 < v.length; i += 2) { x += v[i]; y += v[i + 1]; pts.push(merc(x / 100, y / 100)); }
    /* its box in the world, so that what is out of sight is not drawn */
    let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
    for (const p of pts) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
    pts.box = [x0, y0, x1, y1];
    rings.push(pts);
  }
  function merc(lon, lat) {
    const la = Math.max(-85, Math.min(85, lat)) * Math.PI / 180;
    return [(lon + 180) / 360, (1 - Math.log(Math.tan(la) + 1 / Math.cos(la)) / Math.PI) / 2];
  }
  /* the view: the world's point at the canvas's centre, and its scale */
  let cx = 0.5, cy = 0.5, scale = 1, W = 0, H = 0, dpr = 1, userH = 0;
  /* OpenStreetMap's tiles: only inside Brebus, whose address they are */
  const tilesHere = /\/brebus\/view$/.test(location.pathname);
  /* the map shown: plain, or a layer of tiles (tiles.c's LAYERS), each with
     its deepest zoom and the attribution its terms ask for */
  const LAYERS = {
    osm: {maxz: 19, attr: "© OpenStreetMap contributors", link: "https://www.openstreetmap.org/copyright"},
    topo: {maxz: 17, attr: "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)", link: "https://opentopomap.org/about"},
  };
  let layer = "", sel = null, lensOn = false, ring = null, ringOn = false;
  const tiles = new Map(); /* "z/x/y" -> an Image, loaded or not */
  let repaint = 0;
  const later = () => { if (!repaint) repaint = requestAnimationFrame(() => { repaint = 0; paint(); }); };
  let points = [], draw = null;
  const css = (n, d) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || d;
  function toScreen(p) {
    return [(p[0] - cx) * scale * W + W / 2, (p[1] - cy) * scale * W + H / 2];
  }
  function size() {
    dpr = window.devicePixelRatio || 1;
    W = box.clientWidth;
    /* filling the host's room: half of it, unless the grip said otherwise,
       and never so tall that the list has no room */
    const fill = document.documentElement.classList.contains("fill");
    /* over the whole room, the rest floating over it (float.js) */
    if (document.documentElement.classList.contains("float")) H = box.clientHeight || window.innerHeight;
    else {
      H = userH || opts.height || (fill ? Math.max(160, Math.round(window.innerHeight * 0.5)) : Math.max(220, Math.round(W * 0.55)));
      if (fill) H = Math.max(120, Math.min(H, window.innerHeight - 160));
    }
    cv.style.width = W + "px"; cv.style.height = H + "px";
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    paint();
  }
  function paint() {
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = css("--map-sea", "#dfeaf3");
    g.fillRect(0, 0, W, H);
    g.fillStyle = css("--map-land", "#f4f1ea");
    g.strokeStyle = css("--map-coast", "#b9b2a4");
    g.lineWidth = 0.7;
    /* the world drawn three times across, so that the Pacific joins */
    for (const off of [-1, 0, 1]) {
      g.beginPath();
      for (const r of rings) {
        const a = toScreen([r.box[0] + off, r.box[1]]), b = toScreen([r.box[2] + off, r.box[3]]);
        if (b[0] < 0 || a[0] > W || b[1] < 0 || a[1] > H) continue;
        let first = true;
        for (const p of r) {
          const s = toScreen([p[0] + off, p[1]]);
          if (first) { g.moveTo(s[0], s[1]); first = false; } else g.lineTo(s[0], s[1]);
        }
        g.closePath();
      }
      g.fill(); g.stroke();
    }
    if (layer) drawTiles();
    /* the view's points on each copy of the world in sight */
    if (draw) for (const off of [-1, 0, 1])
      draw(g, (lon, lat) => { const p = merc(lon, lat); return toScreen([p[0] + off, p[1]]); });
    /* the box being drawn with the lens */
    if (sel) {
      g.setLineDash([5, 4]); g.strokeStyle = "#000c"; g.lineWidth = 1.5;
      g.fillStyle = "#3a86ff22";
      const x = Math.min(sel[0], sel[2]), y = Math.min(sel[1], sel[3]);
      g.fillRect(x, y, Math.abs(sel[2] - sel[0]), Math.abs(sel[3] - sel[1]));
      g.strokeRect(x, y, Math.abs(sel[2] - sel[0]), Math.abs(sel[3] - sel[1]));
      g.setLineDash([]);
    }
    /* the circle being drawn: its centre, its edge, its radius in km */
    if (ring) {
      const r = Math.hypot(ring[2] - ring[0], ring[3] - ring[1]);
      g.beginPath(); g.arc(ring[0], ring[1], r, 0, 2 * Math.PI);
      g.fillStyle = "#1d4ed81c"; g.fill();
      ringLine(g, [8, 5]);
      g.beginPath(); g.moveTo(ring[0], ring[1]); g.lineTo(ring[2], ring[3]); ringLine(g, []);
      if (r > 8) {
        const km = ringKm(ring), txt = (km < 10 ? km.toFixed(1) : Math.round(km)) + " km";
        g.font = "12px sans-serif"; const w = g.measureText(txt).width;
        g.fillStyle = "#fffd"; g.fillRect(ring[2] + 8, ring[3] - 16, w + 8, 16);
        g.fillStyle = "#222"; g.fillText(txt, ring[2] + 12, ring[3] - 4);
      }
    }
  }
  /* a point of the canvas in degrees: [lon, lat] */
  function toLonLat(x, y) {
    const mx = cx + (x - W / 2) / (scale * W), my = cy + (y - H / 2) / (scale * W);
    return [((mx * 360 - 180) % 360 + 540) % 360 - 180, Math.atan(Math.sinh(Math.PI * (1 - 2 * my))) * 180 / Math.PI];
  }
  /* the radius of a circle drawn [x0, y0, x1, y1], in km along the ground */
  function ringKm(r) {
    const [lo0, la0] = toLonLat(r[0], r[1]), [lo1, la1] = toLonLat(r[2], r[3]), k = Math.PI / 180;
    const a = Math.sin((la1 - la0) * k / 2) ** 2 + Math.cos(la0 * k) * Math.cos(la1 * k) * Math.sin((lo1 - lo0) * k / 2) ** 2;
    return 2 * 6371 * Math.asin(Math.min(1, Math.sqrt(a)));
  }
  /* the tiles in sight at the zoom that fits, each asked once; the plain
     land shows where one is still coming */
  function drawTiles() {
    const S = scale * W, z = Math.max(0, Math.min(LAYERS[layer].maxz, Math.round(Math.log2(S / 256)))), n = 2 ** z, ts = S / n;
    const x0 = Math.floor((cx - W / 2 / S) * n), x1 = Math.floor((cx + W / 2 / S) * n);
    const y0 = Math.max(0, Math.floor((cy - H / 2 / S) * n)), y1 = Math.min(n - 1, Math.floor((cy + H / 2 / S) * n));
    if ((x1 - x0 + 1) * (y1 - y0 + 1) > 120) return;
    for (let ix = x0; ix <= x1; ix++)
      for (let iy = y0; iy <= y1; iy++) {
        const wx = ((ix % n) + n) % n, key = layer + "/" + z + "/" + wx + "/" + iy;
        let im = tiles.get(key);
        if (!im) {
          im = new Image();
          im.crossOrigin = "anonymous"; /* Brebus says any may read it: the map can be saved */
          im.onload = later;
          im.onerror = () => { im.bad = true; };
          im.src = "/brebus/tile/" + key + ".png";
          tiles.set(key, im);
          /* the oldest let go */
          if (tiles.size > 400) tiles.delete(tiles.keys().next().value);
        }
        if (im.complete && !im.bad && im.naturalWidth) {
          const s = toScreen([ix / n, iy / n]);
          g.drawImage(im, Math.floor(s[0]), Math.floor(s[1]), Math.ceil(ts) + 1, Math.ceil(ts) + 1);
        }
      }
  }
  /* fits points of [lon, lat] (or the world when none) */
  function fit(lls, pad) {
    if (!lls.length) { cx = 0.5; cy = 0.55; scale = 1; paint(); return; }
    /* points across the antimeridian (Fiji and Tonga, Kamchatka and
       Alaska) taken together, the western ones a world further east */
    const lo = Math.min(...lls.map((p) => p[0])), hi = Math.max(...lls.map((p) => p[0]));
    if (hi - lo > 180) {
      const span = (q) => Math.max(...q) - Math.min(...q);
      const shifted = lls.map((p) => [p[0] < 0 ? p[0] + 360 : p[0], p[1]]);
      if (span(shifted.map((p) => p[0])) < hi - lo) lls = shifted;
    }
    let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
    for (const [lon, lat] of lls) {
      const p = merc(lon, lat);
      x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
    }
    const w = Math.max(x1 - x0, 1e-6), h = Math.max(y1 - y0, 1e-6);
    if (x0 > 1) { x0 -= 1; x1 -= 1; }
    /* what the view's floating panel covers (float.js) is left out: the
       points fill the rest of the map */
    const [L, T] = cover(), k = 1 + (pad || 0.4);
    scale = Math.min((W - L) / W / (w * k), (H - T) / W / (h * k), 50000);
    scale = Math.max(scale, 1);
    cx = (x0 + x1) / 2 - L / 2 / (scale * W); cy = (y0 + y1) / 2 - T / 2 / (scale * W);
    paint();
  }
  /* the room the floating panel takes: [left, top] in pixels, when it is
     open over the map, beside it (a wide room) or over its top (a narrow one) */
  function cover() {
    const p = document.documentElement.classList.contains("float") && document.querySelector(".fside");
    if (!p || !W) return [0, 0];
    const r = p.getBoundingClientRect(), b = box.getBoundingClientRect();
    if (r.width > W * 0.6) return [0, Math.max(0, Math.min(H * 0.7, r.bottom - b.top))];
    return [Math.max(0, Math.min(W * 0.7, r.right - b.left)), 0];
  }
  /* moving and zooming */
  let drag = null, moved = false, stopT = 0;
  /* moved or zoomed: said to the view once it stops */
  const stopped = () => { clearTimeout(stopT); if (opts.onmove) stopT = setTimeout(opts.onmove, 600); };
  const at = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  /* a point pressed a while, or the right button: said to the view
     (opts.onpress(lon, lat), "what is here") */
  let pressT = 0;
  const press = (e) => { if (!opts.onpress) return; const p = at(e), [lon, lat] = toLonLat(p[0], p[1]); opts.onpress(lon, lat); };
  cv.addEventListener("contextmenu", (e) => { if (opts.onpress) { e.preventDefault(); clearTimeout(pressT); press(e); } });
  cv.addEventListener("pointerdown", (e) => {
    clearTimeout(pressT);
    if (opts.onpress && e.button === 0 && !ringOn && !lensOn && !e.shiftKey) {
      const x0 = e.clientX, y0 = e.clientY;
      pressT = setTimeout(() => { if (drag && !moved && Math.abs(lastX - x0) + Math.abs(lastY - y0) < 6) { drag = null; press({clientX: x0, clientY: y0}); } }, 600);
    }
    cv.setPointerCapture(e.pointerId);
    if (ringOn) { const p = at(e); ring = [p[0], p[1], p[0], p[1]]; return; }
    if (lensOn || e.shiftKey) { const p = at(e); sel = [p[0], p[1], p[0], p[1]]; return; }
    drag = [e.clientX, e.clientY, cx, cy]; moved = false;
  });
  let lastX = 0, lastY = 0;
  cv.addEventListener("pointermove", (e) => {
    lastX = e.clientX; lastY = e.clientY;
    if (ring) { const p = at(e); ring[2] = p[0]; ring[3] = p[1]; paint(); return; }
    if (sel) { const p = at(e); sel[2] = p[0]; sel[3] = p[1]; paint(); return; }
    if (!drag) return;
    const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
    if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
    cx = drag[2] - dx / (scale * W); cy = drag[3] - dy / (scale * W);
    paint();
  });
  cv.addEventListener("pointerup", (e) => {
    clearTimeout(pressT);
    if (ring) {
      /* the circle drawn, said to the view: its centre and its radius */
      const r = ring; ring = null; circling(false);
      if (Math.hypot(r[2] - r[0], r[3] - r[1]) > 8) { const [lon, lat] = toLonLat(r[0], r[1]); opts.oncircle({lat, lon, km: ringKm(r)}); }
      paint();
      return;
    }
    if (sel) {
      /* the box drawn fills the map */
      const s = sel; sel = null; lens(false);
      const w = Math.abs(s[2] - s[0]), h = Math.abs(s[3] - s[1]);
      if (w > 8 && h > 8) {
        const mx = (s[0] + s[2]) / 2, my = (s[1] + s[3]) / 2;
        cx += (mx - W / 2) / (scale * W); cy += (my - H / 2) / (scale * W);
        scale = Math.max(1, Math.min(50000, scale * Math.min(W / w, H / h)));
        stopped();
      }
      paint();
      return;
    }
    const was = drag; drag = null;
    if (was && moved) stopped();
    if (!was || moved || !opts.onpick) return;
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 18 * 18;
    for (const p of points) {
      const s = toScreen(merc(p.lon, p.lat)), d = (s[0] - x) ** 2 + (s[1] - y) ** 2;
      if (d < bd) { bd = d; best = p; }
    }
    opts.onpick(best);
  });
  function zoom(f, x, y) {
    const k = Math.max(1, Math.min(50000, scale * f)) / scale;
    if (x !== undefined) { /* about the pointer */
      const wx = cx + (x - W / 2) / (scale * W), wy = cy + (y - H / 2) / (scale * W);
      cx = wx - (wx - cx) / k; cy = wy - (wy - cy) / k;
    }
    scale *= k;
    paint();
    stopped();
  }
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = cv.getBoundingClientRect();
    zoom(e.deltaY < 0 ? 1.25 : 0.8, e.clientX - r.left, e.clientY - r.top);
  }, {passive: false});
  const bar = document.createElement("div");
  bar.className = "mapbar";
  for (const [t, f] of [["+", 1.5], ["−", 1 / 1.5]]) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = t; b.onclick = () => zoom(f);
    bar.appendChild(b);
  }
  const lensB = document.createElement("button");
  lensB.type = "button"; lensB.title = mapT("lens"); lensB.setAttribute("aria-label", lensB.title);
  lensB.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5"/></svg>';
  function lens(on) { lensOn = on; lensB.classList.toggle("on", on); cv.classList.toggle("lens", on); }
  lensB.onclick = () => { circling(false); lens(!lensOn); };
  bar.appendChild(lensB);
  /* a circle of the view's (opts.oncircle({lat, lon, km}), null when
     removed): one button draws it, another removes it */
  let ringB = null, unringB = null;
  function circling(on) { ringOn = on; if (ringB) ringB.classList.toggle("on", on); cv.classList.toggle("lens", on || lensOn); }
  if (opts.oncircle) {
    ringB = document.createElement("button");
    ringB.type = "button";
    ringB.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8" stroke-dasharray="3 2.4"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><path d="M12 12h8"/></svg>';
    ringB.onclick = () => { lens(false); circling(!ringOn); };
    unringB = document.createElement("button");
    unringB.type = "button"; unringB.hidden = true;
    unringB.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8" stroke-dasharray="3 2.4"/><path d="M8.5 8.5l7 7M15.5 8.5l-7 7"/></svg>';
    unringB.onclick = () => { unringB.hidden = true; opts.oncircle(null); };
    bar.appendChild(ringB); bar.appendChild(unringB);
  }
  /* the maps, inside Brebus: a menu of them, as a map's layers */
  const attr = document.createElement("button");
  attr.type = "button"; attr.className = "mapattr"; attr.hidden = true;
  attr.onclick = () => layer && typeof App !== "undefined" && App.openLink(LAYERS[layer].link);
  const menu = document.createElement("div");
  menu.className = "maplayers"; menu.hidden = true; menu.setAttribute("role", "menu");
  let layersB = null;
  function choose(l) {
    layer = l; menu.hidden = true;
    attr.hidden = !l; attr.textContent = l ? LAYERS[l].attr : "";
    if (layersB) layersB.classList.toggle("on", !!l);
    paint();
    if (opts.onlayer) opts.onlayer(l);
  }
  function drawMenu() {
    menu.innerHTML = "";
    for (const l of ["", "osm", "topo"]) {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "menuitemradio"); b.setAttribute("aria-checked", l === layer ? "true" : "false");
      b.className = l === layer ? "on" : "";
      b.innerHTML = "<b></b><small></small>";
      b.firstChild.textContent = mapT(l || "plain");
      b.lastChild.textContent = l ? mapT("fetched") : "";
      b.onclick = () => choose(l);
      menu.appendChild(b);
    }
  }
  if (tilesHere) {
    layersB = document.createElement("button");
    layersB.type = "button"; layersB.className = "layers"; layersB.setAttribute("aria-haspopup", "menu");
    layersB.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>';
    layersB.onclick = (e) => { e.stopPropagation(); drawMenu(); menu.hidden = !menu.hidden; };
    bar.appendChild(layersB);
    box.appendChild(menu);
    document.addEventListener("pointerdown", (e) => { if (!menu.hidden && !menu.contains(e.target) && e.target !== layersB && !layersB.contains(e.target)) menu.hidden = true; });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") menu.hidden = true; });
    /* a view of streets starts on a map of streets (opts.layer) */
    if (opts.layer && LAYERS[opts.layer]) choose(opts.layer);
  }
  /* the map saved, inside Brebus: a picture as it looks, or a drawing */
  if (tilesHere) {
    const saveB = document.createElement("button");
    saveB.type = "button"; saveB.className = "savemap"; saveB.setAttribute("aria-haspopup", "menu");
    saveB.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11M7.5 10.5L12 15l4.5-4.5M5 19h14"/></svg>';
    const smenu = document.createElement("div");
    smenu.className = "maplayers"; smenu.hidden = true; smenu.setAttribute("role", "menu");
    for (const k of ["png", "svg"]) {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "menuitem"); b.dataset.k = k;
      b.onclick = () => { smenu.hidden = true; saveMap(k); };
      smenu.appendChild(b);
    }
    saveB.onclick = (e) => {
      e.stopPropagation(); menu.hidden = true;
      for (const b of smenu.children) b.textContent = mapT(b.dataset.k);
      smenu.hidden = !smenu.hidden;
    };
    document.addEventListener("pointerdown", (e) => { if (!smenu.hidden && !smenu.contains(e.target) && !saveB.contains(e.target)) smenu.hidden = true; });
    bar.appendChild(saveB); box.appendChild(smenu);
    /* shown only where the host can save a file */
    const canSave = () => typeof App !== "undefined" && App.canDownload();
    saveB.hidden = true;
    bar.addEventListener("pointerenter", () => { saveB.hidden = !canSave(); saveB.title = mapT("save"); saveB.setAttribute("aria-label", saveB.title); });
    setTimeout(() => { saveB.hidden = !canSave(); }, 1500);
  }
  function mapName() {
    const h = document.getElementById("title");
    return ((h && h.textContent) || "map").replace(/[\\/:*?"<>|\x00-\x1f]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 80) || "map";
  }
  async function saveMap(kind) {
    const note = layer ? LAYERS[layer].attr : "";
    if (kind === "svg") {
      const k = g, kd = dpr, S = SvgCtx(W, H);
      g = S; dpr = 1;
      try { paint(); } finally { g = k; dpr = kd; paint(); }
      if (note) S.note(note);
      return App.download(mapName() + ".svg", "image/svg+xml", S.svg());
    }
    /* the canvas as it is, its attribution written on it */
    const c = document.createElement("canvas");
    c.width = cv.width; c.height = cv.height;
    const x = c.getContext("2d");
    x.drawImage(cv, 0, 0);
    if (note) {
      x.setTransform(dpr, 0, 0, dpr, 0, 0);
      x.font = "10px sans-serif"; x.textAlign = "right"; x.textBaseline = "bottom";
      const w = x.measureText(note).width + 10;
      x.fillStyle = "#ffffffcc"; x.fillRect(W - w - 4, H - 18, w, 14);
      x.fillStyle = "#333"; x.fillText(note, W - 9, H - 6);
    }
    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return App.download(mapName() + ".png", "image/png", btoa(bin), true);
  }
  box.appendChild(bar);
  box.appendChild(attr);
  /* the grip under it: the map's height, the user's */
  const grip = document.createElement("div");
  grip.className = "mapgrip"; grip.title = mapT("grip");
  grip.addEventListener("pointerdown", (e) => {
    e.preventDefault(); grip.setPointerCapture(e.pointerId);
    const y0 = e.clientY, h0 = H;
    const move = (m) => { userH = Math.max(160, Math.min(1600, Math.round(h0 + m.clientY - y0))); size(); if (opts.onresize) opts.onresize(); };
    const up = () => { grip.removeEventListener("pointermove", move); grip.removeEventListener("pointerup", up); if (typeof App !== "undefined") App.size(); };
    grip.addEventListener("pointermove", move); grip.addEventListener("pointerup", up);
  });
  box.after(grip);
  /* the words in the host's language, which comes after the map is made */
  const words = () => {
    lensB.title = mapT("lens"); lensB.setAttribute("aria-label", lensB.title);
    if (ringB) { ringB.title = mapT("circle"); ringB.setAttribute("aria-label", ringB.title); unringB.title = mapT("nocircle"); unringB.setAttribute("aria-label", unringB.title); }
    grip.title = mapT("grip");
    if (layersB) { layersB.title = mapT("layers"); layersB.setAttribute("aria-label", layersB.title); }
  };
  bar.addEventListener("pointerenter", words);
  grip.addEventListener("pointerenter", words);
  if (window.ResizeObserver) new ResizeObserver(size).observe(box);
  window.addEventListener("resize", size);
  size();
  return {
    /* the points the view draws (each with lon, lat), and how */
    set(pts, f) { points = pts; draw = f; paint(); },
    /* whether the view has a circle: its Remove shown */
    circled(on) { if (unringB) unringB.hidden = !on; },
    fit, paint,
    layer: () => layer,
    /* what is in sight: [south, west, north, east], in degrees */
    bounds() {
      const lon = (x) => ((x * 360 - 180) % 360 + 540) % 360 - 180, lat = (y) => Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180 / Math.PI;
      const S = scale * W;
      return [lat(Math.min(1, cy + H / 2 / S)), lon(cx - W / 2 / S), lat(Math.max(0, cy - H / 2 / S)), lon(cx + W / 2 / S)];
    },
    center(lon, lat, s) { const p = merc(lon, lat); cx = p[0]; cy = p[1]; if (s) scale = s; paint(); },
    /* a point at a zoom of the tiles' (16: streets) */
    view(lon, lat, z) { const p = merc(lon, lat); cx = p[0]; cy = p[1]; scale = Math.max(1, Math.min(50000, 256 * 2 ** z / Math.max(W, 1))); paint(); stopped(); },
    /* the zoom of the tiles' now */
    zoom: () => Math.log2(scale * W / 256),
  };
}
