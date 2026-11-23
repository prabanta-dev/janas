/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * md_parse.js - Markdown read as CommonMark 0.31.2 reads it, with GitHub's
 * extensions (GFM: tables, ~strikethrough~, task lists, links written out
 * as www.…, http://… and addresses): mdParse(text, opts) gives the
 * document's tree, markdown.js shows it on a page. Both specifications'
 * examples are the tests (tests/markdown/). The way follows the
 * specification's own appendix: the blocks line by line first, then the
 * inlines of each paragraph, heading and cell with the link references of
 * the whole text known. opts.math reads formulas too - $…$, $$…$$, \(…\),
 * \[…\] - as one piece each, for KaTeX, so that nothing in them is a mark;
 * opts.strict is CommonMark alone, none of GitHub's extensions.
 *
 * A node: t (its type), parent, first, last, prev, next and the fields of
 * its type - doc, para, heading (level), hr, code (info, lit, fenced),
 * html (lit), quote, list (ordered, start, tight, bullet, delim), item
 * (task: null, false or true), table (align) > tr (head) > td (align); the
 * inlines text (lit), softbreak, hardbreak, codespan (lit), htmlspan (lit), em,
 * strong, del, link and image (href, title), math (lit, src: as written,
 * display).
 */

function mdNode(t) {
  return {t, parent: null, first: null, last: null, prev: null, next: null};
}
function mdAppend(p, c) {
  c.parent = p; c.next = null; c.prev = p.last;
  if (p.last) p.last.next = c; else p.first = c;
  p.last = c;
}
function mdUnlink(n) {
  if (n.prev) n.prev.next = n.next; else if (n.parent) n.parent.first = n.next;
  if (n.next) n.next.prev = n.prev; else if (n.parent) n.parent.last = n.prev;
  n.parent = n.prev = n.next = null;
}
function mdInsertAfter(a, n) {
  n.parent = a.parent; n.prev = a; n.next = a.next;
  if (a.next) a.next.prev = n; else if (a.parent) a.parent.last = n;
  a.next = n;
}
/* every node under root, depth first, without recursion (a text may nest
   thousands deep): enter(n) before its children - false skips them - and
   exit(n) after them; enter may change n's later siblings, not n */
function mdWalk(root, enter, exit) {
  let n = root.first;
  while (n) {
    if (enter(n) !== false && n.first) { n = n.first; continue; }
    if (exit) exit(n);
    while (!n.next) {
      n = n.parent;
      if (!n || n === root) return;
      if (exit) exit(n);
    }
    n = n.next;
  }
}

/* ---- characters, escapes, entities, addresses ---- */
const MD_ESC = "[!-/:-@[-`{-~]";
const MD_ENT = "&(?:#[xX][0-9a-fA-F]{1,6}|#[0-9]{1,7}|[A-Za-z][A-Za-z0-9]{1,31});";
const mdReUnescape = new RegExp("\\\\" + MD_ESC + "|" + MD_ENT, "g");
const mdReEscapable = new RegExp("^" + MD_ESC);
const mdReEntity = new RegExp("^" + MD_ENT);
const mdReWsChar = /^[ \t\n\x0b\x0c\x0d]/;
const mdReUniWs = /^[\t\n\f\r \p{Zs}]$/u;
const mdReUniPunct = /^[\p{P}\p{S}]$/u;

function mdDecodeEntity(s) {
  if (s[1] === "#") {
    const n = s[2] === "x" || s[2] === "X" ? parseInt(s.slice(3, -1), 16)
                                          : parseInt(s.slice(2, -1), 10);
    return n === 0 || n > 0x10ffff || (n >= 0xd800 && n <= 0xdfff)
      ? "\ufffd" : String.fromCodePoint(n);
  }
  const v = mdEntity(s.slice(1, -1));
  return v === undefined ? s : v;
}
function mdUnescape(s) {
  return s.indexOf("\\") < 0 && s.indexOf("&") < 0 ? s
    : s.replace(mdReUnescape, (m) => m[0] === "\\" ? m[1] : mdDecodeEntity(m));
}
/* an address as a link carries it: what is not plainly ASCII written as
   %XX of its UTF-8, the %XX already there kept */
function mdEncodeURI(s) {
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "%" && /^[0-9a-fA-F]{2}$/.test(s.slice(i + 1, i + 3))) out += c;
    else if (/[A-Za-z0-9;/?:@&=+$,\-_.!~*'()#]/.test(c)) out += c;
    else {
      const cp = s.codePointAt(i);
      if (cp >= 0xd800 && cp <= 0xdfff) { out += "%EF%BF%BD"; continue; }
      if (cp > 0xffff) i++;
      out += encodeURIComponent(String.fromCodePoint(cp));
    }
  }
  return out;
}
/* a link's label as the definitions know it: case and spaces aside */
function mdNormLabel(s) {
  return s.slice(1, -1).trim().replace(/[ \t\r\n]+/g, " ").toLowerCase().toUpperCase();
}

/* ---- raw HTML ---- */
const MD_TAGNAME = "[A-Za-z][A-Za-z0-9-]*";
const MD_ATTR = "(?:[ \\t\\n]+[a-zA-Z_:][a-zA-Z0-9:._-]*" +
  "(?:[ \\t\\n]*=[ \\t\\n]*(?:[^\"'=<>`\\x00-\\x20]+|'[^']*'|\"[^\"]*\"))?)";
const MD_OPENTAG = "<" + MD_TAGNAME + MD_ATTR + "*[ \\t\\n]*/?>";
const MD_CLOSETAG = "</" + MD_TAGNAME + "[ \\t\\n]*>";
const mdReHtmlTag = new RegExp("^(?:" + MD_OPENTAG + "|" + MD_CLOSETAG +
  "|<!-->|<!--->|<!--[\\s\\S]*?-->|<[?][\\s\\S]*?[?]>|<![A-Za-z][^>]*>" +
  "|<!\\[CDATA\\[[\\s\\S]*?\\]\\]>)");
const MD_NOT_RAW = "(?!(?:pre|script|style|textarea)(?![A-Za-z0-9-]))";
const mdHtmlOpen = [null,
  /^<(?:script|pre|textarea|style)(?:[ \t]|>|$)/i,
  /^<!--/, /^<[?]/, /^<![A-Za-z]/, /^<!\[CDATA\[/,
  /^<[/]?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[123456]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?:[ \t]|[/]?[>]|$)/i,
  new RegExp("^(?:<" + MD_NOT_RAW + MD_TAGNAME + MD_ATTR + "*[ \\t]*/?>|</" +
             MD_NOT_RAW + MD_TAGNAME + "[ \\t]*>)[ \\t]*$", "i")];
const mdHtmlClose = [null, /<\/(?:script|pre|textarea|style)>/i, /-->/, /\?>/, />/, /\]\]>/];

/* ---- the blocks ---- */
const mdReAtx = /^#{1,6}(?:[ \t]+|$)/;
const mdReFence = /^`{3,}(?!.*`)|^~{3,}/;
const mdReCloseFence = /^(?:`{3,}|~{3,})(?=[ \t]*$)/;
const mdReHr = /^(?:\*[ \t]*){3,}$|^(?:_[ \t]*){3,}$|^(?:-[ \t]*){3,}$/;
const mdReSetext = /^(?:=+|-+)[ \t]*$/;
const mdReTableDelim = /^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/;
const mdReMaybeSpecial = /^[#`~*+_=<>0-9|:-]/;

/* a row of a table: its cells, the outer bars aside, \| a bar of the text */
function mdTableCells(line) {
  const s = line.replace(/^[ \t]+|[ \t]+$/g, "");
  const cells = [];
  let cur = "", i = s[0] === "|" ? 1 : 0, sep = false;
  for (; i < s.length; i++) {
    const c = s[i];
    sep = false;
    if (c === "\\" && i + 1 < s.length) {
      cur += s[i + 1] === "|" ? "|" : c + s[i + 1];
      i++;
    } else if (c === "|") {
      cells.push(cur); cur = ""; sep = true;
    } else cur += c;
  }
  if (!sep || cur) cells.push(cur);
  return cells.map((c) => c.replace(/^[ \t]+|[ \t]+$/g, ""));
}

class MdParser {
  constructor(opts) {
    this.opts = opts || {};
    this.refmap = new Map();
  }
  parse(input) {
    this.doc = mdNode("doc");
    this.doc.open = true; this.doc.sl = 1; this.doc.content = "";
    this.tip = this.doc;
    this.oldtip = this.doc;
    this.lineNumber = 0;
    this.lastMatched = this.doc;
    this.allClosed = true;
    input = input.replace(/\0/g, "\ufffd");
    const lines = input.split(/\r\n|\n|\r/);
    let n = lines.length;
    if (input.length && (input.endsWith("\n") || input.endsWith("\r"))) n--;
    for (let i = 0; i < n; i++) this.line(lines[i]);
    while (this.tip) this.finalize(this.tip, n);
    if (!this.opts.strict) this.tasks();
    mdWalk(this.doc, (c) => {
      if (c.t !== "para" && c.t !== "heading" && c.t !== "td") return true;
      new MdInline(this).parse(c);
      return false;
    });
    if (!this.opts.strict) mdAutolinks(this.doc);
    return this.doc;
  }

  /* where the line is */
  peek(i) { return this.ln[i]; }
  findNextNonspace() {
    const ln = this.ln;
    let i = this.offset, cols = this.column, c;
    while ((c = ln[i]) !== undefined) {
      if (c === " ") { i++; cols++; }
      else if (c === "\t") { i++; cols += 4 - (cols % 4); }
      else break;
    }
    this.blank = c === undefined;
    this.nextNonspace = i;
    this.nextNonspaceColumn = cols;
    this.indent = cols - this.column;
    this.indented = this.indent >= 4;
  }
  advanceNextNonspace() {
    this.offset = this.nextNonspace;
    this.column = this.nextNonspaceColumn;
    this.partialTab = false;
  }
  advanceOffset(count, columns) {
    const ln = this.ln;
    let c;
    while (count > 0 && (c = ln[this.offset]) !== undefined) {
      if (c === "\t") {
        const toTab = 4 - (this.column % 4);
        if (columns) {
          this.partialTab = toTab > count;
          const adv = toTab > count ? count : toTab;
          this.column += adv;
          this.offset += this.partialTab ? 0 : 1;
          count -= adv;
        } else {
          this.partialTab = false;
          this.column += toTab; this.offset++; count--;
        }
      } else {
        this.partialTab = false;
        this.offset++; this.column++; count--;
      }
    }
  }

  /* the tree */
  addLine() {
    if (this.partialTab) {
      this.offset++;
      this.tip.content += " ".repeat(4 - (this.column % 4));
    }
    this.tip.content += this.ln.slice(this.offset) + "\n";
  }
  addChild(t) {
    while (!this.canContain(this.tip.t, t)) this.finalize(this.tip, this.lineNumber - 1);
    const n = mdNode(t);
    n.open = true; n.sl = this.lineNumber; n.content = "";
    mdAppend(this.tip, n);
    this.tip = n;
    return n;
  }
  canContain(p, c) {
    if (p === "doc" || p === "quote" || p === "item") return c !== "item";
    if (p === "list") return c === "item";
    return false;
  }
  accepts(t) { return t === "para" || t === "code" || t === "html" || t === "table"; }
  closeUnmatched() {
    if (!this.allClosed) {
      while (this.oldtip !== this.lastMatched) {
        const p = this.oldtip.parent;
        this.finalize(this.oldtip, this.lineNumber - 1);
        this.oldtip = p;
      }
      this.allClosed = true;
    }
  }

  /* does an open block go on with this line: 0 yes, 1 no, 2 yes and the
     line is all its own (a closing fence) */
  cont(b) {
    switch (b.t) {
    case "doc": case "list": return 0;
    case "quote":
      if (!this.indented && this.peek(this.nextNonspace) === ">") {
        this.advanceNextNonspace();
        this.advanceOffset(1, false);
        const c = this.peek(this.offset);
        if (c === " " || c === "\t") this.advanceOffset(1, true);
        return 0;
      }
      return 1;
    case "item":
      if (this.blank) {
        if (!b.first) return 1;
        this.advanceNextNonspace();
      } else if (this.indent >= b.data.markerOffset + b.data.padding) {
        this.advanceOffset(b.data.markerOffset + b.data.padding, true);
      } else return 1;
      return 0;
    case "heading": case "hr": return 1;
    case "code":
      if (b.fenced) {
        const m = this.indent <= 3 && this.peek(this.nextNonspace) === b.fenceChar &&
                  this.ln.slice(this.nextNonspace).match(mdReCloseFence);
        if (m && m[0].length >= b.fenceLength) {
          this.finalize(b, this.lineNumber);
          return 2;
        }
        let i = b.fenceOffset, c;
        while (i > 0 && ((c = this.peek(this.offset)) === " " || c === "\t")) {
          this.advanceOffset(1, true); i--;
        }
        return 0;
      }
      if (this.indent >= 4) this.advanceOffset(4, true);
      else if (this.blank) this.advanceNextNonspace();
      else return 1;
      return 0;
    case "html":
      return this.blank && (b.htmlType === 6 || b.htmlType === 7) ? 1 : 0;
    case "para": case "table":
      return this.blank ? 1 : 0;
    }
    return 1;
  }

  /* the blocks a line may open: 0 none, 1 a container, 2 a leaf */
  starts(container) {
    const ln = this.ln, nn = this.nextNonspace, ind = this.indented;
    const c0 = ln[nn];
    let m;
    /* a quote */
    if (!ind && c0 === ">") {
      this.advanceNextNonspace();
      this.advanceOffset(1, false);
      const c = this.peek(this.offset);
      if (c === " " || c === "\t") this.advanceOffset(1, true);
      this.closeUnmatched();
      this.addChild("quote");
      return 1;
    }
    /* # a heading */
    if (!ind && c0 === "#" && (m = ln.slice(nn).match(mdReAtx))) {
      this.advanceNextNonspace();
      this.advanceOffset(m[0].length, false);
      this.closeUnmatched();
      const h = this.addChild("heading");
      h.level = m[0].trim().length;
      h.content = ln.slice(this.offset).replace(/^[ \t]*#+[ \t]*$/, "")
                    .replace(/[ \t]+#+[ \t]*$/, "");
      this.advanceOffset(ln.length - this.offset, false);
      return 2;
    }
    /* ``` a fenced block */
    if (!ind && (c0 === "`" || c0 === "~") && (m = ln.slice(nn).match(mdReFence))) {
      this.closeUnmatched();
      const b = this.addChild("code");
      b.fenced = true;
      b.fenceLength = m[0].length;
      b.fenceChar = m[0][0];
      b.fenceOffset = this.indent;
      this.advanceNextNonspace();
      this.advanceOffset(m[0].length, false);
      return 2;
    }
    /* <tag> raw HTML */
    if (!ind && c0 === "<") {
      const s = ln.slice(nn);
      for (let k = 1; k <= 7; k++) {
        if (mdHtmlOpen[k].test(s) &&
            (k < 7 || (container.t !== "para" &&
                       !(!this.allClosed && !this.blank && this.tip.t === "para")))) {
          this.closeUnmatched();
          const b = this.addChild("html");
          b.htmlType = k;
          return 2;
        }
      }
    }
    /* | a | table |, under its first row */
    if (!ind && !this.opts.strict && container.t === "para" && ln.indexOf("|") >= 0 &&
        mdReTableDelim.test(ln)) {
      this.refs(container);
      const lines = container.content.replace(/\n$/, "").split("\n");
      const head = lines[lines.length - 1];
      const align = mdTableCells(ln).map((c) =>
        c.endsWith(":") ? (c.startsWith(":") ? "center" : "right")
                        : c.startsWith(":") ? "left" : "");
      const cells = head !== undefined && container.content ? mdTableCells(head) : null;
      if (cells && cells.length === align.length) {
        this.closeUnmatched();
        const t = mdNode("table");
        t.open = true; t.sl = this.lineNumber - 1; t.content = "";
        t.align = align; t.head = cells;
        mdInsertAfter(container, t);
        if (lines.length > 1) {
          container.content = lines.slice(0, -1).join("\n") + "\n";
          container.open = false;
          container.el = this.lineNumber - 2;
        } else mdUnlink(container);
        this.tip = t;
        this.advanceOffset(ln.length - this.offset, false);
        return 2;
      }
    }
    /* a heading underlined */
    if (!ind && container.t === "para" && (m = ln.slice(nn).match(mdReSetext))) {
      this.closeUnmatched();
      this.refs(container);
      if (container.content.length > 0) {
        const h = mdNode("heading");
        h.sl = container.sl; h.open = true;
        h.level = m[0][0] === "=" ? 1 : 2;
        h.content = container.content;
        mdInsertAfter(container, h);
        mdUnlink(container);
        this.tip = h;
        this.advanceOffset(ln.length - this.offset, false);
        return 2;
      }
      return 0;
    }
    /* --- a rule */
    if (!ind && mdReHr.test(ln.slice(nn))) {
      this.closeUnmatched();
      this.addChild("hr");
      this.advanceOffset(ln.length - this.offset, false);
      return 2;
    }
    /* - a list's item */
    let d;
    if ((!ind || container.t === "list") && (d = this.listMarker(container))) {
      this.closeUnmatched();
      if (this.tip.t !== "list" || !this.listsMatch(container.data, d)) {
        const l = this.addChild("list");
        l.data = d;
      }
      const it = this.addChild("item");
      it.data = d;
      return 1;
    }
    /* code indented */
    if (ind && this.tip.t !== "para" && !this.blank) {
      this.advanceOffset(4, true);
      this.closeUnmatched();
      this.addChild("code");
      return 2;
    }
    return 0;
  }
  listMarker(container) {
    if (this.indent >= 4) return null;
    const rest = this.ln.slice(this.nextNonspace);
    let m, d;
    if ((m = rest.match(/^[*+-]/))) {
      d = {type: "bullet", bullet: m[0], delim: null, start: null};
    } else if ((m = rest.match(/^(\d{1,9})([.)])/)) &&
               (container.t !== "para" || m[1] === "1")) {
      d = {type: "ordered", bullet: null, delim: m[2], start: parseInt(m[1], 10)};
    } else return null;
    const next = this.ln[this.nextNonspace + m[0].length];
    if (!(next === undefined || next === " " || next === "\t")) return null;
    if (container.t === "para" && !/[^ \t]/.test(this.ln.slice(this.nextNonspace + m[0].length)))
      return null;
    d.markerOffset = this.indent;
    this.advanceNextNonspace();
    this.advanceOffset(m[0].length, true);
    const startCol = this.column, startOff = this.offset;
    let c;
    do {
      this.advanceOffset(1, true);
      c = this.peek(this.offset);
    } while (this.column - startCol < 5 && (c === " " || c === "\t"));
    const blankItem = this.peek(this.offset) === undefined;
    const after = this.column - startCol;
    if (after >= 5 || after < 1 || blankItem) {
      d.padding = m[0].length + 1;
      this.column = startCol;
      this.offset = startOff;
      c = this.peek(this.offset);
      if (c === " " || c === "\t") this.advanceOffset(1, true);
    } else d.padding = m[0].length + after;
    return d;
  }
  listsMatch(a, b) {
    return a.type === b.type && a.delim === b.delim && a.bullet === b.bullet;
  }

  line(ln) {
    this.ln = ln;
    this.offset = 0; this.column = 0; this.blank = false; this.partialTab = false;
    this.lineNumber++;
    let container = this.doc;
    this.oldtip = this.tip;
    let last;
    while ((last = container.last) && last.open) {
      container = last;
      this.findNextNonspace();
      const r = this.cont(container);
      if (r === 2) return;
      if (r === 1) { container = container.parent; break; }
    }
    this.allClosed = container === this.oldtip;
    this.lastMatched = container;
    let leaf = container.t !== "para" && container.t !== "table" && this.accepts(container.t);
    while (!leaf) {
      this.findNextNonspace();
      if (!this.indented && !mdReMaybeSpecial.test(ln.slice(this.nextNonspace))) {
        this.advanceNextNonspace();
        break;
      }
      const r = this.starts(container);
      if (r === 1) { container = this.tip; continue; }
      if (r === 2) { container = this.tip; leaf = true; break; }
      this.advanceNextNonspace();
      break;
    }
    if (!this.allClosed && !this.blank && this.tip.t === "para") {
      this.addLine(); /* a paragraph's lazy line */
      return;
    }
    this.closeUnmatched();
    const t = container.t;
    if (this.accepts(t)) {
      this.addLine();
      if (t === "html" && container.htmlType <= 5 &&
          mdHtmlClose[container.htmlType].test(ln.slice(this.offset)))
        this.finalize(container, this.lineNumber);
    } else if (this.offset < ln.length && !this.blank) {
      this.addChild("para");
      this.advanceNextNonspace();
      this.addLine();
    }
  }

  /* the link definitions at a paragraph's start, taken out of it */
  refs(p) {
    let pos;
    while (p.content[0] === "[" && (pos = new MdInline(this).reference(p.content)))
      p.content = p.content.slice(pos);
  }
  finalize(b, line) {
    const above = b.parent;
    b.open = false;
    b.el = line;
    switch (b.t) {
    case "para":
      this.refs(b);
      if (!/[^ \t\n]/.test(b.content)) mdUnlink(b);
      break;
    case "code":
      if (b.fenced) {
        const nl = b.content.indexOf("\n");
        b.info = mdUnescape(b.content.slice(0, nl).trim());
        b.lit = b.content.slice(nl + 1);
      } else {
        const lines = b.content.split("\n");
        while (lines.length && /^[ \t]*$/.test(lines[lines.length - 1])) lines.pop();
        b.lit = lines.join("\n") + "\n";
      }
      break;
    case "html":
      b.lit = b.content.replace(/(\n *)+$/, "");
      break;
    case "item":
      b.el = b.last ? b.last.el : b.sl;
      break;
    case "list": {
      b.ordered = b.data.type === "ordered";
      b.start = b.data.start;
      b.bullet = b.data.bullet;
      b.delim = b.data.delim;
      b.tight = true;
      const gap = (n) => n.next && n.el !== n.next.sl - 1;
      outer:
      for (let it = b.first; it; it = it.next) {
        if (gap(it)) { b.tight = false; break; }
        for (let s = it.first; s; s = s.next)
          if (gap(s)) { b.tight = false; break outer; }
      }
      b.el = b.last ? b.last.el : b.sl;
      break;
    }
    case "table": {
      const n = b.align.length;
      const row = (cells, head) => {
        const tr = mdNode("tr");
        tr.head = head;
        for (let i = 0; i < n; i++) {
          const td = mdNode("td");
          td.align = b.align[i];
          td.content = cells[i] || "";
          mdAppend(tr, td);
        }
        mdAppend(b, tr);
      };
      row(b.head, true);
      /* the first line is what the delimiter row left: nothing */
      const lines = b.content.split("\n").slice(1, -1);
      for (const l of lines) row(mdTableCells(l), false);
      break;
    }
    }
    this.tip = above;
  }

  /* [ ] and [x] at the start of an item: a task */
  tasks() {
    const items = [];
    mdWalk(this.doc, (c) => { if (c.t === "item") items.push(c); });
    for (const c of items) {
      c.task = null;
      const p = c.first;
      let m;
      if (p && p.t === "para" && (m = p.content.match(/^\[([ \txX])\](?:[ \t]+|(?=\n)|$)/))) {
        c.task = m[1] === "x" || m[1] === "X";
        p.content = p.content.slice(m[0].length);
        if (!/[^ \t\n]/.test(p.content)) mdUnlink(p);
      }
    }
  }
}

/* ---- the inlines ---- */
const mdReMain = /^[^\n`[\]\\!<&*_~$]+/;
const mdReTicks = /`+/g;
const mdReLinkTitle = /^(?:"(?:\\[\s\S]|[^\\"\0])*"|'(?:\\[\s\S]|[^\\'\0])*'|\((?:\\[\s\S]|[^\\()\0])*\))/;
const mdReLinkLabel = /^\[(?:[^\\[\]]|\\[\s\S]){0,1000}\]/;
const mdReDestBraces = /^<(?:[^<>\n\\\0]|\\[\s\S])*>/;
const mdReEmailAuto = /^<([a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*)>/;
const mdReAuto = /^<[A-Za-z][A-Za-z0-9.+-]{1,31}:[^<>\x00-\x20]*>/;
const mdReSpnl = /^[ \t]*(?:\n[ \t]*)?/;
const mdReMathDollar = /^\$(?=\S)([^$\n]*?[\\^_{}=][^$\n]*?\S)\$(?![\w$])/;

/* the code point that ends before i, and the one at i */
function mdCharBefore(s, i) {
  if (i <= 0) return "\n";
  const c = s.charCodeAt(i - 1);
  if (c >= 0xdc00 && c <= 0xdfff && i >= 2) {
    const h = s.charCodeAt(i - 2);
    if (h >= 0xd800 && h <= 0xdbff) return s.slice(i - 2, i);
  }
  return s[i - 1];
}
function mdCharAt(s, i) {
  if (i >= s.length) return "\n";
  return String.fromCodePoint(s.codePointAt(i));
}

class MdInline {
  constructor(parser) {
    this.refmap = parser.refmap;
    this.opts = parser.opts;
  }
  text(block, s) {
    const n = mdNode("text");
    n.lit = s;
    mdAppend(block, n);
    return n;
  }
  match(re) {
    const m = re.exec(this.subject.slice(this.pos));
    if (!m) return null;
    this.pos += m.index + m[0].length;
    return m[0];
  }
  spnl() { this.match(mdReSpnl); return true; }

  parse(block) {
    this.subject = block.content.replace(/^[ \t\n]+|[ \t\n]+$/g, "");
    this.pos = 0;
    this.delimiters = null;
    this.brackets = null;
    while (this.one(block));
    block.content = null;
    this.emphasis(null);
    mdMergeText(block);
  }
  one(block) {
    const c = this.subject[this.pos];
    if (c === undefined) return false;
    let res = false;
    switch (c) {
    case "\n": res = this.newline(block); break;
    case "\\": res = this.backslash(block); break;
    case "`": res = this.backticks(block); break;
    case "*": case "_": res = this.delim(c, block); break;
    case "~": res = !this.opts.strict && this.delim(c, block); break;
    case "[": res = this.openBracket(block); break;
    case "!": res = this.bang(block); break;
    case "]": res = this.closeBracket(block); break;
    case "<": res = this.autolink(block) || this.htmlTag(block); break;
    case "&": res = this.entity(block); break;
    case "$": res = this.math(block); break;
    default: {
      const m = this.match(mdReMain);
      if (m) { this.text(block, m); res = true; }
    }
    }
    if (!res) { this.pos++; this.text(block, c); }
    return true;
  }

  newline(block) {
    this.pos++;
    const l = block.last;
    if (l && l.t === "text" && l.lit.endsWith(" ")) {
      const hard = l.lit.endsWith("  ");
      l.lit = l.lit.replace(/ +$/, "");
      mdAppend(block, mdNode(hard ? "hardbreak" : "softbreak"));
    } else mdAppend(block, mdNode("softbreak"));
    this.match(/^[ \t]*/);
    return true;
  }
  backslash(block) {
    const s = this.subject;
    const n = s[this.pos + 1];
    if (this.opts.math && (n === "(" || n === "[")) {
      const close = n === "(" ? "\\)" : "\\]";
      const end = s.indexOf(close, this.pos + 2);
      if (end > this.pos + 2 && /\S/.test(s.slice(this.pos + 2, end)) &&
          (n === "[" || s.slice(this.pos + 2, end).indexOf("\n") < 0)) {
        const m = mdNode("math");
        m.lit = s.slice(this.pos + 2, end);
        m.src = s.slice(this.pos, end + 2);
        m.display = n === "[";
        mdAppend(block, m);
        this.pos = end + 2;
        return true;
      }
    }
    this.pos++;
    if (n === "\n") {
      this.pos++;
      mdAppend(block, mdNode("hardbreak"));
      this.match(/^[ \t]*/);
    } else if (n !== undefined && mdReEscapable.test(n)) {
      this.pos++;
      this.text(block, n);
    } else this.text(block, "\\");
    return true;
  }
  backticks(block) {
    const ticks = this.match(/^`+/);
    const after = this.pos;
    mdReTicks.lastIndex = after;
    let m;
    while ((m = mdReTicks.exec(this.subject))) {
      if (m[0].length === ticks.length) {
        let c = this.subject.slice(after, m.index).replace(/\n/g, " ");
        if (c.length >= 2 && /[^ ]/.test(c) && c[0] === " " && c[c.length - 1] === " ")
          c = c.slice(1, -1);
        const n = mdNode("codespan");
        n.lit = c;
        mdAppend(block, n);
        this.pos = m.index + m[0].length;
        return true;
      }
    }
    this.pos = after;
    this.text(block, ticks);
    return true;
  }
  math(block) {
    if (!this.opts.math) return false;
    const s = this.subject.slice(this.pos);
    let m;
    if (s.startsWith("$$")) {
      const end = s.indexOf("$$", 2);
      if (end > 2 && /\S/.test(s.slice(2, end))) {
        const n = mdNode("math");
        n.lit = s.slice(2, end);
        n.src = s.slice(0, end + 2);
        n.display = true;
        mdAppend(block, n);
        this.pos += end + 2;
        return true;
      }
      return false;
    }
    const before = mdCharBefore(this.subject, this.pos);
    if (/[\w$\\]/.test(before) || !(m = s.match(mdReMathDollar))) return false;
    const n = mdNode("math");
    n.lit = m[1];
    n.src = m[0];
    n.display = false;
    mdAppend(block, n);
    this.pos += m[0].length;
    return true;
  }
  /* a run of * _ or ~: whether it opens, closes, both or neither */
  scanDelims(cc) {
    const s = this.subject, start = this.pos;
    let n = 0;
    while (s[start + n] === cc) n++;
    const before = mdCharBefore(s, start), after = mdCharAt(s, start + n);
    const aWs = mdReUniWs.test(after), bWs = mdReUniWs.test(before);
    const aP = mdReUniPunct.test(after), bP = mdReUniPunct.test(before);
    const left = !aWs && (!aP || bWs || bP);
    const right = !bWs && (!bP || aWs || aP);
    let open, close;
    if (cc === "_") {
      open = left && (!right || bP);
      close = right && (!left || aP);
    } else {
      open = left;
      close = right;
    }
    return {n, open, close};
  }
  delim(cc, block) {
    const r = this.scanDelims(cc);
    const start = this.pos;
    this.pos += r.n;
    const node = this.text(block, this.subject.slice(start, this.pos));
    if ((r.open || r.close) && (cc !== "~" || r.n <= 2)) {
      this.delimiters = {cc, numdelims: r.n, origdelims: r.n, node,
                         previous: this.delimiters, next: null,
                         canOpen: r.open, canClose: r.close};
      if (this.delimiters.previous) this.delimiters.previous.next = this.delimiters;
    }
    return true;
  }
  removeDelimiter(d) {
    if (d.previous) d.previous.next = d.next;
    if (d.next) d.next.previous = d.previous;
    else this.delimiters = d.previous;
  }
  emphasis(bottom) {
    const openersBottom = new Map();
    let closer = this.delimiters;
    while (closer && closer.previous !== bottom) closer = closer.previous;
    while (closer) {
      if (!closer.canClose) { closer = closer.next; continue; }
      const cc = closer.cc;
      const key = cc === "~" ? "~" + closer.numdelims
                             : cc + (closer.canOpen ? "1" : "0") + (closer.origdelims % 3);
      let opener = closer.previous, found = false;
      const floor = openersBottom.get(key);
      while (opener && opener !== bottom && opener !== floor) {
        if (opener.cc === cc && opener.canOpen) {
          if (cc === "~") {
            if (opener.numdelims === closer.numdelims) { found = true; break; }
          } else {
            const odd = (closer.canOpen || opener.canClose) && closer.origdelims % 3 !== 0 &&
                        (opener.origdelims + closer.origdelims) % 3 === 0;
            if (!odd) { found = true; break; }
          }
        }
        opener = opener.previous;
      }
      const old = closer;
      if (found) {
        const use = cc === "~" ? closer.numdelims
                               : closer.numdelims >= 2 && opener.numdelims >= 2 ? 2 : 1;
        const oi = opener.node, ci = closer.node;
        opener.numdelims -= use;
        closer.numdelims -= use;
        oi.lit = oi.lit.slice(0, oi.lit.length - use);
        ci.lit = ci.lit.slice(0, ci.lit.length - use);
        const e = mdNode(cc === "~" ? "del" : use === 1 ? "em" : "strong");
        let tmp = oi.next;
        while (tmp && tmp !== ci) {
          const nx = tmp.next;
          mdUnlink(tmp);
          mdAppend(e, tmp);
          tmp = nx;
        }
        mdInsertAfter(oi, e);
        if (opener.next !== closer) { opener.next = closer; closer.previous = opener; }
        if (opener.numdelims === 0) { mdUnlink(oi); this.removeDelimiter(opener); }
        if (closer.numdelims === 0) {
          mdUnlink(ci);
          const nx = closer.next;
          this.removeDelimiter(closer);
          closer = nx;
        }
      } else {
        closer = closer.next;
        openersBottom.set(key, old.previous);
        if (!old.canOpen) this.removeDelimiter(old);
      }
    }
    while (this.delimiters && this.delimiters !== bottom) this.removeDelimiter(this.delimiters);
  }

  addBracket(node, index, image) {
    if (this.brackets) this.brackets.bracketAfter = true;
    this.brackets = {node, previous: this.brackets, previousDelimiter: this.delimiters,
                     index, image, active: true, bracketAfter: false};
  }
  openBracket(block) {
    const start = this.pos;
    this.pos++;
    this.addBracket(this.text(block, "["), start, false);
    return true;
  }
  bang(block) {
    const start = this.pos;
    this.pos++;
    if (this.subject[this.pos] === "[") {
      this.pos++;
      this.addBracket(this.text(block, "!["), start + 1, true);
    } else this.text(block, "!");
    return true;
  }
  closeBracket(block) {
    this.pos++;
    const start = this.pos;
    const opener = this.brackets;
    if (!opener) { this.text(block, "]"); return true; }
    if (!opener.active) {
      this.text(block, "]");
      this.brackets = opener.previous;
      return true;
    }
    const image = opener.image;
    const save = this.pos;
    let dest = null, title = null, matched = false;
    if (this.subject[this.pos] === "(") {
      this.pos++;
      this.spnl();
      if ((dest = this.destination()) !== null) {
        this.spnl();
        if (mdReWsChar.test(this.subject[this.pos - 1] || "")) title = this.title();
        this.spnl();
        if (this.subject[this.pos] === ")") { this.pos++; matched = true; }
      }
      if (!matched) this.pos = save;
    }
    if (!matched) {
      const before = this.pos;
      const n = this.label();
      let ref = null;
      if (n > 2) ref = this.subject.slice(before, before + n);
      else if (!opener.bracketAfter) ref = this.subject.slice(opener.index, start);
      if (n === 0) this.pos = save;
      if (ref) {
        const link = this.refmap.get(mdNormLabel(ref));
        if (link) { dest = link.dest; title = link.title; matched = true; }
      }
    }
    if (matched) {
      const node = mdNode(image ? "image" : "link");
      node.href = dest;
      node.title = title || "";
      let tmp = opener.node.next;
      while (tmp) {
        const nx = tmp.next;
        mdUnlink(tmp);
        mdAppend(node, tmp);
        tmp = nx;
      }
      mdAppend(block, node);
      this.emphasis(opener.previousDelimiter);
      this.brackets = opener.previous;
      mdUnlink(opener.node);
      if (!image)
        for (let o = this.brackets; o; o = o.previous) if (!o.image) o.active = false;
      return true;
    }
    this.brackets = opener.previous;
    this.pos = start;
    this.text(block, "]");
    return true;
  }
  destination() {
    const m = this.match(mdReDestBraces);
    if (m) return mdEncodeURI(mdUnescape(m.slice(1, -1)));
    const s = this.subject;
    if (s[this.pos] === "<") return null;
    const save = this.pos;
    let open = 0, c;
    while ((c = s[this.pos]) !== undefined) {
      if (c === "\\" && mdReEscapable.test(s[this.pos + 1] || "")) this.pos += 2;
      else if (c === "(") {
        /* 32 deep at most, as the specification allows: deeper, each ( of
           a long text would read it to its end */
        if (++open > 32) { this.pos = save; return null; }
        this.pos++;
      }
      else if (c === ")") { if (open < 1) break; this.pos++; open--; }
      else if (c.charCodeAt(0) <= 0x20 || c === "\x7f") break;
      else this.pos++;
    }
    if (this.pos === save && c !== ")") return null;
    if (open !== 0) { this.pos = save; return null; }
    return mdEncodeURI(mdUnescape(s.slice(save, this.pos)));
  }
  title() {
    const m = this.match(mdReLinkTitle);
    return m === null ? null : mdUnescape(m.slice(1, -1));
  }
  label() {
    const m = this.match(mdReLinkLabel);
    if (m === null || m.length > 1001) {
      if (m) this.pos -= m.length;
      return 0;
    }
    return m.length;
  }
  /* [label]: destination "title" at the start of s: its length, 0 if none */
  reference(s) {
    this.subject = s;
    this.pos = 0;
    const n = this.label();
    if (n === 0) return 0;
    const raw = s.slice(0, n);
    if (s[this.pos] !== ":") return 0;
    this.pos++;
    this.spnl();
    const dest = this.destination();
    if (dest === null) return 0;
    const beforeTitle = this.pos;
    this.spnl();
    let title = null;
    if (this.pos !== beforeTitle) title = this.title();
    if (title === null) this.pos = beforeTitle;
    let atEnd = true;
    if (this.match(/^[ \t]*(?:\n|$)/) === null) {
      if (title === null) atEnd = false;
      else {
        title = null;
        this.pos = beforeTitle;
        atEnd = this.match(/^[ \t]*(?:\n|$)/) !== null;
      }
    }
    if (!atEnd) return 0;
    const norm = mdNormLabel(raw);
    if (norm === "") return 0;
    if (!this.refmap.has(norm)) this.refmap.set(norm, {dest, title: title || ""});
    return this.pos;
  }
  autolink(block) {
    let m, href;
    if ((m = this.match(mdReEmailAuto))) href = "mailto:" + m.slice(1, -1);
    else if ((m = this.match(mdReAuto))) href = m.slice(1, -1);
    else return false;
    const n = mdNode("link");
    n.href = mdEncodeURI(href);
    n.title = "";
    n.auto = true;
    this.text(n, m.slice(1, -1));
    mdAppend(block, n);
    return true;
  }
  htmlTag(block) {
    const m = this.match(mdReHtmlTag);
    if (m === null) return false;
    const n = mdNode("htmlspan");
    n.lit = m;
    mdAppend(block, n);
    return true;
  }
  entity(block) {
    const m = this.match(mdReEntity);
    if (m === null) return false;
    this.text(block, mdDecodeEntity(m));
    return true;
  }
}

/* texts side by side, one text */
function mdMergeText(root) {
  mdWalk(root, (c) => {
    while (c.t === "text" && c.next && c.next.t === "text") {
      c.lit += c.next.lit;
      mdUnlink(c.next);
    }
  });
}

/* ---- GFM: links written out, www.… http://… ftp://… and addresses ---- */
const mdReHostChar = /^[^\s\p{P}!-/:-@[-`{-~]$/u;
function mdIsAlnum(c) { return c !== undefined && /[A-Za-z0-9]/.test(c); }
/* the domain at s[i]: its length, 0 when there is none */
function mdDomain(s, i, short) {
  let j = i, dots = 0, us1 = 0, us2 = 0, first = true;
  while (j < s.length) {
    const cp = s.codePointAt(j), c = String.fromCodePoint(cp);
    if (c === "_") us2++;
    else if (c === ".") { us1 = us2; us2 = 0; dots++; }
    else if (first || (c !== "-" && !mdReHostChar.test(c))) {
      if (!mdReHostChar.test(c)) break;
    }
    first = false;
    j += c.length;
  }
  if (j === i || us1 > 0 || us2 > 0) return 0;
  return short || dots ? j - i : 0;
}
/* what a link written out leaves outside it: the punctuation after it, a
   parenthesis it does not open, &entity; */
function mdAutolinkEnd(s, start, end) {
  const lt = s.indexOf("<", start);
  if (lt >= 0 && lt < end) end = lt;
  let o = -1, cl = 0; /* its parentheses, counted once */
  while (end > start) {
    const c = s[end - 1];
    if ("?!.,:*_~'\"".indexOf(c) >= 0) end--;
    else if (c === ";") {
      let k = end - 2;
      while (k > start && /[A-Za-z]/.test(s[k])) k--;
      if (k < end - 2 && s[k] === "&") end = k; else end--;
    } else if (c === ")") {
      if (o < 0) {
        o = cl = 0;
        for (let k = start; k < end; k++) { if (s[k] === "(") o++; else if (s[k] === ")") cl++; }
      }
      if (cl <= o) break;
      end--;
      cl--;
    } else break;
  }
  return end;
}
/* the links in a text: [from, to, href] */
function mdFindAutolinks(s, prev) {
  const out = [];
  let from = 0, i = 0;
  const pathEnd = (k) => { while (k < s.length && !/\s/.test(s[k]) && s[k] !== "<") k++; return k; };
  while (i < s.length) {
    const c = s[i];
    const before = i > 0 ? s[i - 1] : prev;
    let got = null;
    if (c === "w" && s.startsWith("www.", i) &&
        (before === undefined || /\s/.test(before) || "*_~(".indexOf(before) >= 0)) {
      const d = mdDomain(s, i, false);
      if (d) {
        const e = mdAutolinkEnd(s, i, pathEnd(i + d));
        if (e > i) got = [i, e, "http://" + s.slice(i, e)];
      }
    } else if (c === ":" && s.startsWith("//", i + 1)) {
      let a = i;
      while (a > from && /[A-Za-z]/.test(s[a - 1])) a--;
      if (/^(?:https?|ftp)$/i.test(s.slice(a, i))) {
        const d = mdDomain(s, i + 3, true);
        if (d) {
          const e = mdAutolinkEnd(s, a, pathEnd(i + 3 + d));
          if (e > i + 3) got = [a, e, s.slice(a, e)];
        }
      }
    } else if (c === "@") {
      let a = i;
      while (a > from && (mdIsAlnum(s[a - 1]) || ".+-_".indexOf(s[a - 1]) >= 0)) a--;
      let e = i + 1, dots = 0;
      while (e < s.length) {
        const d = s[e];
        if (mdIsAlnum(d) || d === "-" || d === "_") e++;
        else if (d === "." && mdIsAlnum(s[e + 1])) { dots++; e++; }
        else break;
      }
      if (a < i && e > i + 1 && dots > 0 && /[A-Za-z]/.test(s[e - 1])) {
        let scheme = "mailto:";
        const pre = (p) => a - p.length >= from && s.slice(a - p.length, a) === p &&
                           !mdIsAlnum(s[a - p.length - 1]);
        if (pre("mailto:")) { a -= 7; scheme = ""; }
        else if (pre("xmpp:")) {
          a -= 5;
          scheme = "";
          if (s[e] === "/") {
            let k = e + 1;
            while (k < s.length && (mdIsAlnum(s[k]) || "@.-_".indexOf(s[k]) >= 0)) k++;
            while (k > e + 1 && s[k - 1] === ".") k--;
            if (k > e + 1) e = k;
          }
        }
        got = [a, e, scheme + s.slice(a, e)];
      }
    }
    if (got) {
      out.push(got);
      from = i = got[1];
    } else i++;
  }
  return out;
}
function mdAutolinks(root) {
  const texts = [];
  mdWalk(root, (c) => {
    if (c.t === "link" || c.t === "image") return false;
    if (c.t === "text") texts.push(c);
    return true;
  });
  const inline = (t) => t === "em" || t === "strong" || t === "del";
  for (const c of texts) {
    const p = c.prev;
    /* what comes before: the line's start, a mark of emphasis, or text */
    const prev = !p ? (inline(c.parent.t) ? "*" : undefined)
               : p.t === "softbreak" || p.t === "hardbreak" ? "\n"
               : inline(p.t) ? "*" : "x";
    const found = mdFindAutolinks(c.lit, prev);
    if (!found.length) continue;
    const s = c.lit;
    let at = 0, after = c;
    const put = (node) => { mdInsertAfter(after, node); after = node; };
    const text = (lit) => { const t = mdNode("text"); t.lit = lit; return t; };
    for (const [a, e, href] of found) {
      if (a > at) put(text(s.slice(at, a)));
      const l = mdNode("link");
      l.href = mdEncodeURI(href);
      l.title = "";
      l.auto = true;
      mdAppend(l, text(s.slice(a, e)));
      put(l);
      at = e;
    }
    if (at < s.length) put(text(s.slice(at)));
    mdUnlink(c);
  }
}

function mdParse(text, opts) {
  return new MdParser(opts).parse(text);
}
