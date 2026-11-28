/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* export.js - a view's table saved as a file: a button that opens a
   panel, each column in or out, the form (CSV with the BOM spreadsheets
   want, TSV, JSON, Markdown), Save; the file handed to the host to save
   (App.download), where it says it can. Any view with rows of data uses
   it: Exporter.button(where, () => ({name, columns: [{key, label}], rows}))
   - each row an object, a value a string or a number, as the user should
   read it. Needs app.js before it. */
const Exporter = (() => {
  const W = {
    export: {en: "Export", it: "Esporta", fr: "Exporter", es: "Exportar", de: "Exportieren", pt: "Exportar"},
    title: {en: "Export the data", it: "Esporta i dati", fr: "Exporter les données", es: "Exportar los datos", de: "Die Daten exportieren", pt: "Exportar os dados"},
    cols: {en: "Columns", it: "Colonne", fr: "Colonnes", es: "Columnas", de: "Spalten", pt: "Colunas"},
    all: {en: "all", it: "tutte", fr: "toutes", es: "todas", de: "alle", pt: "todas"},
    none: {en: "none", it: "nessuna", fr: "aucune", es: "ninguna", de: "keine", pt: "nenhuma"},
    form: {en: "Format", it: "Formato", fr: "Format", es: "Formato", de: "Format", pt: "Formato"},
    rows: {en: "{n} rows", it: "{n} righe", fr: "{n} lignes", es: "{n} filas", de: "{n} Zeilen", pt: "{n} linhas"},
    save: {en: "Save", it: "Salva", fr: "Enregistrer", es: "Guardar", de: "Speichern", pt: "Guardar"},
    cancel: {en: "Cancel", it: "Annulla", fr: "Annuler", es: "Cancelar", de: "Abbrechen", pt: "Cancelar"},
    nocol: {en: "Choose a column at least.", it: "Scegli almeno una colonna.", fr: "Choisissez au moins une colonne.", es: "Elige al menos una columna.", de: "Wähle mindestens eine Spalte.", pt: "Escolhe pelo menos uma coluna."},
    saved: {en: "Saved: {f}", it: "Salvato: {f}", fr: "Enregistré : {f}", es: "Guardado: {f}", de: "Gespeichert: {f}", pt: "Guardado: {f}"},
    lat: {en: "Latitude", it: "Latitudine", fr: "Latitude", es: "Latitud", de: "Breite", pt: "Latitude"},
    lon: {en: "Longitude", it: "Longitudine", fr: "Longitude", es: "Longitud", de: "Länge", pt: "Longitude"},
    url: {en: "Link", it: "Collegamento", fr: "Lien", es: "Enlace", de: "Link", pt: "Ligação"},
    km: {en: "Distance (km)", it: "Distanza (km)", fr: "Distance (km)", es: "Distancia (km)", de: "Entfernung (km)", pt: "Distância (km)"},
    dir: {en: "Direction", it: "Direzione", fr: "Direction", es: "Dirección", de: "Richtung", pt: "Direção"},
    time: {en: "Time", it: "Ora", fr: "Heure", es: "Hora", de: "Zeit", pt: "Hora"},
    date: {en: "Date", it: "Data", fr: "Date", es: "Fecha", de: "Datum", pt: "Data"},
    name: {en: "Name", it: "Nome", fr: "Nom", es: "Nombre", de: "Name", pt: "Nome"},
    desc: {en: "Description", it: "Descrizione", fr: "Description", es: "Descripción", de: "Beschreibung", pt: "Descrição"},
    flight: {en: "Flight", it: "Volo", fr: "Vol", es: "Vuelo", de: "Flug", pt: "Voo"},
    reg: {en: "Registration", it: "Marche", fr: "Immatriculation", es: "Matrícula", de: "Kennzeichen", pt: "Matrícula"},
    from: {en: "From", it: "Da", fr: "De", es: "Desde", de: "Von", pt: "De"},
    to: {en: "To", it: "A", fr: "À", es: "A", de: "Nach", pt: "Para"},
    wind: {en: "Wind (km/h)", it: "Vento (km/h)", fr: "Vent (km/h)", es: "Viento (km/h)", de: "Wind (km/h)", pt: "Vento (km/h)"},
    gust: {en: "Gusts (km/h)", it: "Raffiche (km/h)", fr: "Rafales (km/h)", es: "Rachas (km/h)", de: "Böen (km/h)", pt: "Rajadas (km/h)"},
    wdir: {en: "Wind from (°)", it: "Vento da (°)", fr: "Vent de (°)", es: "Viento de (°)", de: "Wind aus (°)", pt: "Vento de (°)"},
    failed: {en: "Not saved: {e}", it: "Non salvato: {e}", fr: "Non enregistré : {e}", es: "No guardado: {e}", de: "Nicht gespeichert: {e}", pt: "Não guardado: {e}"},
  };
  const t = (k, v) => App.t(W[k]).replace(/\{(\w+)\}/g, (_, n) => v && v[n] !== undefined ? v[n] : "");
  const FORMS = [
    {id: "csv", label: "CSV", mime: "text/csv", ext: "csv"},
    {id: "csvx", label: "CSV (Excel, ;)", mime: "text/csv", ext: "csv"},
    {id: "tsv", label: "TSV", mime: "text/tab-separated-values", ext: "tsv"},
    {id: "json", label: "JSON", mime: "application/json", ext: "json"},
    {id: "md", label: "Markdown", mime: "text/markdown", ext: "md"},
  ];
  const text = (v) => v === undefined || v === null ? "" : String(v);
  /* the file's text: the rows, by the columns chosen */
  function make(form, cols, rows) {
    if (form === "json")
      return JSON.stringify(rows.map((r) => Object.fromEntries(cols.map((c) => [c.label, r[c.key] === undefined ? null : r[c.key]]))), null, 2) + "\n";
    if (form === "md") {
      const esc = (v) => text(v).replace(/\|/g, "\\|").replace(/\n/g, " ");
      return "| " + cols.map((c) => esc(c.label)).join(" | ") + " |\n|" + cols.map(() => "---").join("|") + "|\n" +
        rows.map((r) => "| " + cols.map((c) => esc(r[c.key])).join(" | ") + " |").join("\n") + "\n";
    }
    /* CSV for Excel: ';' between the values and numbers written as the
       language writes them (a decimal comma where it has one), as Excel
       reads them in that language */
    const sep = form === "tsv" ? "\t" : form === "csvx" ? ";" : ",";
    const dec = (0.5).toLocaleString(App.ctx.locale).charAt(1);
    const q = (v) => {
      const s = form === "csvx" && typeof v === "number" ? String(v).replace(".", dec) : text(v);
      if (form === "tsv") return s.replace(/[\t\n\r]/g, " ");
      return (form === "csvx" ? /[";\n\r]/ : /[",\n\r]/).test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    /* the BOM: Excel reads the accents right */
    return (form !== "tsv" ? "\ufeff" : "") + [cols.map((c) => q(c.label)).join(sep), ...rows.map((r) => cols.map((c) => q(r[c.key])).join(sep))].join("\r\n") + "\r\n";
  }
  /* a name for a file, from the view's words */
  const fileName = (n) => (text(n).replace(/[\\/:*?"<>|\x00-\x1f]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 80) || "data");
  let chosen = null, form = "csv"; /* kept while the view lives */
  function panel(anchor, get) {
    const old = document.querySelector(".xpanel");
    if (old) { old.remove(); return; }
    const tbl = get();
    if (!chosen) chosen = {};
    const p = document.createElement("div");
    p.className = "xpanel"; p.setAttribute("role", "dialog");
    p.innerHTML = `<b class="xtitle"></b><div class="xrow"><span class="xlbl"></span> <button type="button" class="link xall"></button> · <button type="button" class="link xnone"></button></div>
      <div class="xcols"></div><div class="xrow"><span class="xlbl xf"></span> <select class="xform"></select> <span class="note xn"></span></div>
      <div class="xmsg note"></div><div class="xbtns"><button type="button" class="xsave"></button><button type="button" class="xcancel"></button></div>`;
    p.querySelector(".xtitle").textContent = t("title");
    p.querySelector(".xlbl").textContent = t("cols");
    p.querySelector(".xall").textContent = t("all");
    p.querySelector(".xnone").textContent = t("none");
    p.querySelector(".xf").textContent = t("form");
    p.querySelector(".xn").textContent = t("rows", {n: tbl.rows.length});
    p.querySelector(".xsave").textContent = t("save");
    p.querySelector(".xcancel").textContent = t("cancel");
    const box = p.querySelector(".xcols");
    for (const c of tbl.columns) {
      const l = document.createElement("label");
      const i = document.createElement("input");
      i.type = "checkbox"; i.checked = chosen[c.key] !== false; i.dataset.k = c.key;
      i.onchange = () => { chosen[c.key] = i.checked; };
      l.append(i, " " + c.label);
      box.appendChild(l);
    }
    const sel = p.querySelector(".xform");
    for (const f of FORMS) { const o = document.createElement("option"); o.value = f.id; o.textContent = f.label; sel.appendChild(o); }
    sel.value = form; sel.onchange = () => { form = sel.value; };
    const all = (on) => { for (const i of box.querySelectorAll("input")) { i.checked = on; chosen[i.dataset.k] = on; } };
    p.querySelector(".xall").onclick = () => all(true);
    p.querySelector(".xnone").onclick = () => all(false);
    p.querySelector(".xcancel").onclick = () => p.remove();
    p.querySelector(".xsave").onclick = async () => {
      const cols = tbl.columns.filter((c) => chosen[c.key] !== false);
      const msg = p.querySelector(".xmsg");
      if (!cols.length) { msg.textContent = t("nocol"); return; }
      const f = FORMS.find((x) => x.id === form), name = fileName(tbl.name) + "." + f.ext;
      try {
        await App.download(name, f.mime, make(f.id, cols, get().rows));
        p.remove();
        Exporter.note(anchor, t("saved", {f: name}));
      } catch (e) { msg.textContent = t("failed", {e: e.message}); }
    };
    /* under the row of buttons, as wide as the view */
    (anchor.parentNode.classList.contains("vtools") ? anchor.parentNode : anchor).after(p);
    App.size();
  }
  return {
    make,
    /* the columns many views have, in the host's language: lat, lon, url, km, dir */
    label: (k) => t(k),
    /* the Export button, put after where (nothing where the host cannot
       save a file) */
    button(where, get) {
      if (!App.canDownload()) return null;
      const b = document.createElement("button");
      b.type = "button"; b.className = "xbtn";
      b.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11M7.5 10.5L12 15l4.5-4.5M5 19h14"/></svg><span></span>';
      b.lastChild.textContent = t("export");
      b.onclick = () => panel(b, get);
      where.appendChild(b);
      return b;
    },
    /* a line said under the button for a few seconds */
    note(anchor, words) {
      const n = document.createElement("span");
      n.className = "note xdone"; n.textContent = words;
      anchor.after(n);
      setTimeout(() => n.remove(), 4000);
    },
  };
})();
