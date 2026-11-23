# Markdown tests

The examples `test_markdown` checks `src/web/md_parse.js` against, through `check.js` (run by node: `node tests/markdown/check.js` from the repository's root).

- `commonmark-0.31.2.json`: the 652 examples of the CommonMark Spec 0.31.2, its `spec.json` as published at <https://spec.commonmark.org/0.31.2/spec.json>, unchanged. Read with `{strict: true}`: CommonMark alone.
- `gfm-extensions.json`: the examples of GitHub Flavored Markdown's extensions (tables, task lists, strikethrough, autolinks, disallowed raw HTML), taken from cmark-gfm's `test/spec.txt` (the examples marked as extensions) and `test/extensions.txt` (footnotes aside: GFM's spec does not have them), <https://github.com/github/cmark-gfm>. A task's box is written there two ways; `check.js` takes both.
- `math.json`: ours, the formulas read with `{math: true}`.

The CommonMark Spec is copyright John MacFarlane, and the GitHub Flavored Markdown Spec is GitHub's, both under the Creative Commons Attribution-ShareAlike 4.0 International license (<https://creativecommons.org/licenses/by-sa/4.0/>); their examples are kept here under it.

`html.js` writes the parser's tree as the HTML the examples expect, for these tests alone: the pages draw the tree with `src/web/markdown.js`, raw HTML shown as text.
