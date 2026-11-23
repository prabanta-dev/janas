/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * check.js - src/web/md_parse.js against the examples of the
 * specifications: CommonMark 0.31.2's 652 (commonmark-0.31.2.json, the
 * spec's own spec.json) and GFM's extensions (gfm-extensions.json: the
 * examples of cmark-gfm's spec.txt and extensions.txt marked as
 * extensions, footnotes aside), and the formulas of opts.math (math.json,
 * ours); then markdown.js, the page's drawing of the tree, on a document
 * of its own (the elements it makes, nothing a text could turn into a
 * script). Run by test_markdown.c: node check.js <repo>; prints one line,
 * and each example that fails with what was expected and what came out.
 * Both specifications are under CC BY-SA 4.0 (tests/markdown/README.md).
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = process.argv[2] || ".";
/* a document of the few calls markdown.js makes */
class El {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.childNodes = [];
    this.dataset = {};
    this.style = {};
    this.className = "";
    this.classList = {add: (c) => { this.className = (this.className + " " + c).trim(); }};
  }
  appendChild(c) { this.childNodes.push(c); return c; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  set textContent(s) { this.childNodes = s ? [{text: s}] : []; }
  addEventListener(type, f) { this.click = type === "click" && f; }
}
const document = {createElement: (t) => new El(t), createTextNode: (s) => ({text: s})};
const ctx = vm.createContext({document, window: {}});
for (const f of ["src/web/md_entities.js", "src/web/md_parse.js", "tests/markdown/html.js",
                 "src/web/markdown.js"])
  vm.runInContext(fs.readFileSync(path.join(root, f), "utf8"), ctx, {filename: f});
const render = vm.runInContext("(t, o) => mdHtml(mdParse(t, o), o)", ctx);

/* cmark-gfm writes a task's box as <input type="checkbox" disabled="" />,
   its spec as <input disabled="" type="checkbox">: one of them */
const norm = (h) => h
  .replace(/<input type="checkbox" checked="" disabled="" \/>/g, '<input checked="" disabled="" type="checkbox">')
  .replace(/<input type="checkbox" disabled="" \/>/g, '<input disabled="" type="checkbox">');

let fails = 0, total = 0;
const show = (set, e, got) => {
  fails++;
  if (fails > 40) return;
  console.log(`FAIL ${set} #${e.example} (${e.section})\n--- markdown\n${JSON.stringify(e.markdown)}` +
              `\n--- expected\n${e.html}--- got\n${got}`);
};
const sets = [
  ["commonmark", "commonmark-0.31.2.json", {strict: true}],
  ["gfm", "gfm-extensions.json", {tagfilter: true}],
  ["math", "math.json", {math: true}],
];
const counts = [];
for (const [set, file, opts] of sets) {
  const list = JSON.parse(fs.readFileSync(path.join(root, "tests/markdown", file), "utf8"));
  let ok = 0;
  for (const e of list) {
    total++;
    let got;
    try {
      got = render(e.markdown, opts);
    } catch (err) {
      show(set, e, "THROWS " + err.stack + "\n");
      continue;
    }
    if (e.html === "<IGNORE>\n" || norm(got) === norm(e.html)) ok++;
    else show(set, e, got);
  }
  counts.push(`${set} ${ok}/${list.length}`);
}
/* the page's elements, written out: tag.class[lang]{align}(attributes) */
const ser = (n) => n.text !== undefined ? n.text
  : "<" + n.tagName.toLowerCase() + (n.className ? "." + n.className.replace(/ /g, ".") : "") +
    (n.dataset.lang ? "[" + n.dataset.lang + "]" : "") +
    (n.style.textAlign ? "{" + n.style.textAlign + "}" : "") +
    (n.href ? "(" + n.href + ")" : "") + (n.checked ? "(checked)" : "") +
    (n.start ? "(" + n.start + ")" : "") + ">" + n.childNodes.map(ser).join("") +
    "</" + n.tagName.toLowerCase() + ">";
const draw = vm.runInContext("(el, t, o) => render(el, t, o)", ctx);
const page = [
  ["a\nb  \nc", "<p>a<br></br>b<br></br>c</p>"],
  ["- [x] done\n- [ ] to do\n- plain", "<ul><li.task><input(checked)></input>done</li>" +
   "<li.task><input></input>to do</li><li>plain</li></ul>"],
  ["3. three\n4. four", "<ol(3)><li>three</li><li>four</li></ol>"],
  ["<script>alert(1)</script>", "<div.rawhtml><script>alert(1)</script></div>"],
  ["x<sup>2</sup> <b onclick=\"f()\">y</b> a<br>b", "<p>x<sup>2</sup> <b onclick=\"f()\">y</b> a<br></br>b</p>"],
  ["[a](javascript:alert(1)) [b](https://e.org) <mailto:x@y.z>",
   "<p><span.link.off>a</span> <a.link(https://e.org)>b</a> <span.link.off>mailto:x@y.z</span></p>"],
  ["![a *cat*](https://e.org/c.png) ![b](c.png)",
   "<p><a.link.pic(https://e.org/c.png)>\u{1F5BC} a cat</a> <span.link.off.pic>\u{1F5BC} b</span></p>"],
  ["$x^2$, \\(y_1\\) and $5 to $10", "<p><span.math>$x^2$</span>, <span.math>\\(y_1\\)</span> and $5 to $10</p>"],
  ["```py\nx = 1\n```", "<pre[py]>x = 1</pre>"],
  ["| a | b |\n|:-:|--:|\n| *c* | d |", "<div.table><table><thead><tr><th{center}>a</th><th{right}>b</th></tr></thead>" +
   "<tbody><tr><td{center}><em>c</em></td><td{right}>d</td></tr></tbody></table></div>"],
  ["- a\n\n- b", "<ul.loose><li><p>a</p></li><li><p>b</p></li></ul>"],
  ["# T\n> q ~~s~~\n\n---", "<h1>T</h1><blockquote><p>q <del>s</del></p></blockquote><hr></hr>"],
];
let pageOk = 0;
for (const [md, want] of page) {
  const el = new El("div");
  draw(el, md, {});
  const got = el.childNodes.map(ser).join("");
  total++;
  if (got === want) pageOk++;
  else show("page", {example: pageOk + 1, section: "markdown.js", markdown: md, html: want + "\n"}, got + "\n");
}
counts.push(`page ${pageOk}/${page.length}`);

/* nothing a text can hold makes it slow: the inputs that are quadratic in
   a careless parser, each a few hundred kB, in well under a second */
const t0 = Date.now();
const heavy = ["*a ".repeat(30000), "[".repeat(30000) + "a" + "]".repeat(30000),
  "`".repeat(5000) + "a", "> ".repeat(3000) + "a", "- ".repeat(3000) + "a",
  "a **b ".repeat(20000), "<a " .repeat(20000), "[a](".repeat(20000),
  "*_".repeat(30000), "www.a.com".repeat(20000) + ")".repeat(20000),
  "|a|\n|-|\n" + "|b|\n".repeat(20000), "&#x" + "1".repeat(50000),
  "` `` ".repeat(10000), "<".repeat(50000), "![".repeat(30000), "_a ".repeat(30000),
  "a\n".repeat(50000), "\\".repeat(50000), "<sup>".repeat(20000) + "</sup>".repeat(20000)];
for (const h of heavy) { render(h, {math: true}); draw(new El("div"), h, {}); }
const ms = Date.now() - t0;
const slow = ms > 8000;
console.log(`test_markdown: ${fails || slow ? "FAIL" : "ok"} (${counts.join(", ")}; ` +
            `${heavy.length} heavy texts in ${ms} ms)`);
process.exit(fails || slow ? 1 : 0);
