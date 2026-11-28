/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* i18n.js - Brebus's page (app.html, login.html): the languages it speaks.
   The page is written in English, the only language inside brebus; each
   other is a file beside it (src/brebus/lang/<code>.json, README.md
   there), data alone, which brebus reads, checks and hands to the page as
   a script of its own making (/brebus/lang.js, lang.c): LANGS, those there
   are, LANG and LOCALE, the language spoken, and I18N[LANG], for each
   English text, its translation. What a file lacks stays in English.

   The language: the one chosen in Settings (kept in this browser), or, as
   the system, the first of the system's that Brebus speaks: in Brebus's
   window those of the computer (the window's address says them, ?lang=,
   from brebus: lang.c), in a browser the browser's. English otherwise.

   T("text") is the text in the page's language; T("Brebus {v} is out",
   {v: x}) fills {v} (the value as given: escaped by the caller where it
   goes into HTML). Tn(n, "{n} chat", "{n} chats") the form for n (the
   file gives [one, other] for the English plural). i18nTree(el) puts the
   page's own HTML (app.html) into the language: its texts, titles,
   placeholders and labels; a text with markup inside (<code>, <b>, <a>...)
   is looked up whole. The messages of the model (prompts) stay in English:
   they are for the model. */

let LANGS = [["en", "English"]];
let LANG = "en", LOCALE = "en";
const I18N = {};

(() => {
  let pick = "";
  try { pick = localStorage.getItem("brebus.lang") || ""; } catch (e) {}
  /* the window's address says the system's languages; kept for the
     window's next pages (the address loses it) */
  let sys = new URLSearchParams(location.search).get("lang") || "";
  try {
    if (sys) sessionStorage.setItem("brebus.syslang", sys);
    else sys = sessionStorage.getItem("brebus.syslang") || "";
  } catch (e) {}
  const want = [pick, ...sys.split(","), ...(navigator.languages || [navigator.language])]
    .filter(Boolean).join(",");
  /* read before the scripts after this one: the page is still being read,
     so the script written here runs next */
  document.write(`<script src="/brebus/lang.js?want=${encodeURIComponent(want)}"><\/script>`);
})();

/* what is missing from the language's file, for the tests: the English
   texts asked and not found */
const i18nMissing = new Set();
const i18nNorm = (s) => String(s).replace(/\s+/g, " ").trim();
const i18nFill = (r, v) => v ? r.replace(/\{(\w+)\}/g, (m, k) => (k in v ? String(v[k]) : m)) : r;

function i18nLook(s) {
  if (LANG === "en") return null;
  const r = I18N[LANG] && I18N[LANG][s];
  if (r === undefined) i18nMissing.add(s);
  return r === undefined ? null : r;
}
function T(s, v) {
  const r = i18nLook(s);
  return i18nFill(typeof r === "string" ? r : s, v);
}
function Tn(n, one, other, v) {
  const r = i18nLook(other);
  v = Object.assign({n: num(n)}, v);
  if (Array.isArray(r))
    return i18nFill(new Intl.PluralRules(LOCALE).select(n) === "one" ? r[0] : r[r.length - 1], v);
  return i18nFill(n === 1 ? one : other, v);
}
/* a text that brebus sent (an error, why an update cannot be): the
   language's file has it whole, or as a pattern whose {parts} match
   anything ("a build between releases ({v}): not updated"); otherwise as
   it came. Its parts are put back as they came. */
let i18nPats = null;
function Ts(s) {
  if (LANG === "en" || typeof s !== "string" || !s) return s;
  const d = I18N[LANG] || {};
  if (typeof d[s] === "string") return d[s];
  if (!i18nPats) {
    const re = (k) => new RegExp("^" + k.split(/\{\w+\}/).map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("([\\s\\S]*?)") + "$");
    i18nPats = Object.keys(d).filter((k) => /\{\w+\}/.test(k) && typeof d[k] === "string")
      .map((k) => ({re: re(k), names: [...k.matchAll(/\{(\w+)\}/g)].map((m) => m[1]), to: d[k]}));
  }
  for (const p of i18nPats) {
    const m = p.re.exec(s);
    if (m) { const v = {}; p.names.forEach((n, i) => { v[n] = m[i + 1]; }); return i18nFill(p.to, v); }
  }
  i18nMissing.add(s);
  return s;
}
/* numbers and dates as the language writes them */
const num = (n) => Number(n).toLocaleString(LOCALE);
const dateTime = (t) => new Date(t).toLocaleString(LOCALE);
const dateOnly = (t) => new Date(t).toLocaleDateString(LOCALE);

const I18N_INLINE = new Set(["A", "B", "BR", "CODE", "EM", "I", "KBD", "STRONG"]);
const I18N_ATTRS = ["title", "placeholder", "aria-label"];
function i18nTree(root) {
  if (LANG === "en") return;
  const word = /[A-Za-z]{2}/;
  const walk = (el) => {
    for (const a of I18N_ATTRS) {
      if (!el.hasAttribute(a)) continue;
      const k = i18nNorm(el.getAttribute(a));
      const r = word.test(k) ? i18nLook(k) : null;
      if (typeof r === "string") el.setAttribute(a, r);
    }
    if (el.tagName === "SCRIPT" || el.tagName === "STYLE" || el.tagName === "CODE" ||
        el.tagName === "PRE" || el.namespaceURI === "http://www.w3.org/2000/svg" ||
        el.hasAttribute("data-noi18n")) return;
    const kids = [...el.childNodes];
    const els = kids.filter((n) => n.nodeType === 1);
    /* a text with markup inside: whole, when the file has it so */
    if (els.length && els.every((e) => I18N_INLINE.has(e.tagName)) &&
        kids.some((n) => n.nodeType === 3 && word.test(n.data))) {
      const k = i18nNorm(el.innerHTML);
      const r = I18N[LANG] && I18N[LANG][k];
      if (typeof r === "string") { el.innerHTML = r; return; }
    }
    for (const n of kids) {
      if (n.nodeType === 1) walk(n);
      else if (n.nodeType === 3 && word.test(n.data)) {
        const r = i18nLook(i18nNorm(n.data));
        if (typeof r === "string") {
          const m = /^(\s*)[\s\S]*?(\s*)$/.exec(n.data);
          n.data = m[1] + r + m[2];
        }
      }
    }
  };
  walk(root);
}

/* the language chosen: "" as the system */
function langChosen() {
  try { return localStorage.getItem("brebus.lang") || ""; } catch (e) { return ""; }
}
function langSet(l) {
  try { if (l) localStorage.setItem("brebus.lang", l); else localStorage.removeItem("brebus.lang"); } catch (e) {}
  /* where the user was - the project, the chat, the section - opened
     again once the page is loaded in the new language (more.js, booted) */
  try {
    sessionStorage.setItem("brebus.resume", JSON.stringify({path: ws && ws.path || "", free: !!(ws && ws.free),
      chat: cur && cur.messages && cur.messages.length ? cur.id : "", section: localStorage.getItem("brebus.section") || ""}));
  } catch (e) {}
  location.reload();
}
