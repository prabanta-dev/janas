/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * html.js - md_parse.js's tree written as the HTML the specifications'
 * examples expect (commonmark.js's and cmark-gfm's way, character for
 * character), for the tests alone: the pages draw the tree themselves
 * (src/web/markdown.js), raw HTML as text. opts.tagfilter: GFM's
 * disallowed raw HTML.
 */
function mdHtml(doc, opts) {
  opts = opts || {};
  /* in pieces, the last one's end at hand: endsWith on a long string
     would flatten it each time */
  const out = [];
  let noTags = 0, nl = true;
  const put = (s) => { if (s) { out.push(s); nl = s.endsWith("\n"); } };
  const cr = () => { if (!nl) put("\n"); };
  const esc = (s) => s.replace(/[&<>"]/g, (c) =>
    c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&quot;");
  const raw = (s) => opts.tagfilter
    ? s.replace(/<(\/?)(title|textarea|style|xmp|iframe|noembed|noframes|script|plaintext)(?=[\s/>]|$)/gi,
                "&lt;$1$2")
    : s;
  const tight = (n) => n.parent.t === "item" && n.parent.parent && n.parent.parent.tight;
  const title = (n) => n.title ? ' title="' + esc(n.title) + '"' : "";
  mdWalk(doc, (n) => {
    switch (n.t) {
    case "para": if (!tight(n)) { cr(); put("<p>"); } break;
    case "heading": cr(); put("<h" + n.level + ">"); break;
    case "hr": cr(); put("<hr />"); cr(); break;
    case "code": {
      const lang = (n.info || "").split(/[ \t\n]+/)[0];
      cr();
      put("<pre><code" + (lang ? ' class="language-' + esc(lang) + '"' : "") + ">" +
          esc(n.lit) + "</code></pre>");
      cr();
      break;
    }
    case "html": cr(); put(raw(n.lit)); cr(); break;
    case "quote": cr(); put("<blockquote>"); cr(); break;
    case "list":
      cr();
      put(n.ordered ? "<ol" + (n.start !== 1 ? ' start="' + n.start + '"' : "") + ">" : "<ul>");
      cr();
      break;
    case "item":
      put("<li>");
      if (n.task !== null && n.task !== undefined)
        put((n.task ? '<input checked="" disabled="" type="checkbox">'
                    : '<input disabled="" type="checkbox">') + " ");
      break;
    case "table": cr(); put("<table>"); cr(); break;
    case "tr":
      if (n.head) { put("<thead>"); cr(); }
      else if (n.prev && n.prev.head) { put("<tbody>"); cr(); }
      put("<tr>"); cr();
      break;
    case "td":
      put("<" + (n.parent.head ? "th" : "td") + (n.align ? ' align="' + n.align + '"' : "") + ">");
      break;
    case "text": put(esc(n.lit)); break;
    case "codespan": put(noTags ? esc(n.lit) : "<code>" + esc(n.lit) + "</code>"); break;
    case "htmlspan": if (!noTags) put(raw(n.lit)); break;
    case "softbreak": put("\n"); break;
    case "hardbreak": put(noTags ? "\n" : "<br />\n"); break;
    case "math":
      put('<span class="math">' + (n.display ? "\\[" : "\\(") + esc(n.lit) +
          (n.display ? "\\]" : "\\)") + "</span>");
      break;
    case "em": case "strong": case "del": if (!noTags) put("<" + n.t + ">"); break;
    case "link": if (!noTags) put('<a href="' + esc(n.href) + '"' + title(n) + ">"); break;
    case "image":
      if (!noTags) put('<img src="' + esc(n.href) + '" alt="');
      noTags++;
      break;
    }
  }, (n) => {
    switch (n.t) {
    case "para": if (!tight(n)) { put("</p>"); cr(); } break;
    case "heading": put("</h" + n.level + ">"); cr(); break;
    case "quote": cr(); put("</blockquote>"); cr(); break;
    case "list": cr(); put(n.ordered ? "</ol>" : "</ul>"); cr(); break;
    case "item": put("</li>"); cr(); break;
    case "table":
      if (n.last && !n.last.head) { put("</tbody>"); cr(); }
      put("</table>"); cr();
      break;
    case "tr":
      put("</tr>"); cr();
      if (n.head) { put("</thead>"); cr(); }
      break;
    case "td": put("</" + (n.parent.head ? "th" : "td") + ">"); cr(); break;
    case "em": case "strong": case "del": if (!noTags) put("</" + n.t + ">"); break;
    case "link": if (!noTags) put("</a>"); break;
    case "image":
      noTags--;
      if (!noTags) put('"' + title(n) + " />");
      break;
    }
  });
  return out.join("");
}
