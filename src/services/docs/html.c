/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * html.c - the text a reader sees of a web page, and of an EPUB book (zip
 * archive of XHTML chapters).
 *
 * A page: no tags, and nothing of what is inside script, style, template,
 * noscript, svg, head (but its title, as the page's heading); the blocks
 * (p, div, li, tr, h1-h6, blockquote...) on lines of their own, list items
 * with "- ", a table's cells by tabs, headings with Markdown's "#"; spaces
 * and line ends folded as a browser does, but inside pre; entities turned
 * back. A page not in UTF-8 is read as Latin-1 (or Windows-1252).
 *
 * A book: META-INF/container.xml names its package (.opf), whose spine
 * gives the chapters' order and whose manifest their files, under the
 * package's folder; each chapter read as a page.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "docs.h"

#define MAX_CHAPTERS 4000

struct hs {
    struct janas_buf *out;
    int skip;    /* inside script, style...: its depth */
    int pre;     /* inside pre */
    int space;   /* a space is owed before the next word */
    int nl;      /* newlines just written */
    int cell;    /* inside a table's cell: its blocks on its line */
    size_t item; /* where a list item's text starts, after its "- " */
};

static void ht_put(struct hs *s, const char *p, size_t n)
{
    if (!n)
        return;
    if (s->space && s->out->n && !s->nl &&
        s->out->p[s->out->n - 1] != '\t') /* not at a cell's start */
        janas_buf_put(s->out, " ", 1);
    s->space = 0;
    janas_buf_put(s->out, p, n);
    s->nl = 0;
}

static void ht_newline(struct hs *s, int max)
{
    /* a cell's paragraphs, and a list item's first block, on their line */
    if (s->cell || s->out->n == s->item) {
        s->space = s->out->n != s->item;
        return;
    }
    s->space = 0;
    while (s->out->n && s->out->p[s->out->n - 1] == ' ')
        s->out->n--;
    if (s->out->n && s->nl < max) {
        janas_buf_put(s->out, "\n", 1);
        s->nl++;
    }
}

/* text: its entities back, its spaces folded but in pre */
static void ht_text(struct hs *s, const char *p, size_t n)
{
    if (s->skip || !n)
        return;
    struct janas_buf t = {0};
    dz_unescape(p, n, &t);
    if (t.oom) {
        janas_buf_free(&t);
        return;
    }
    if (s->pre) {
        ht_put(s, t.p, t.n);
        if (t.n && t.p[t.n - 1] == '\n')
            s->nl = 1;
        janas_buf_free(&t);
        return;
    }
    size_t i = 0;
    while (i < t.n) {
        if (t.p[i] == ' ' || t.p[i] == '\t' || t.p[i] == '\n' ||
            t.p[i] == '\r' || t.p[i] == '\f') {
            s->space = 1;
            i++;
            continue;
        }
        size_t w = i;
        while (w < t.n && t.p[w] != ' ' && t.p[w] != '\t' && t.p[w] != '\n' &&
               t.p[w] != '\r' && t.p[w] != '\f')
            w++;
        ht_put(s, t.p + i, w - i);
        i = w;
    }
    janas_buf_free(&t);
}

static int ht_named(const char *nm, size_t n, const char *const *list)
{
    for (; *list; list++)
        if (strlen(*list) == n && !strncasecmp(nm, *list, n))
            return 1;
    return 0;
}

static const char *const SKIP[] = {"script",   "style",  "template", "noscript",
                                   "svg",      "math",   "head",     "iframe",
                                   "textarea", "select", NULL};
static const char *const BLOCK[] = {
    "p",        "div",        "section", "article",    "header",  "footer",
    "nav",      "aside",      "main",    "ul",         "ol",      "dl",
    "dt",       "dd",         "table",   "tbody",      "thead",   "tfoot",
    "tr",       "blockquote", "figure",  "figcaption", "address", "form",
    "fieldset", "hr",         "caption", "details",    "summary", "body",
    NULL};

int dz_html_text(const char *h, size_t n, struct janas_buf *out)
{
    /* a page that is not UTF-8: read as Latin-1, made UTF-8 */
    struct janas_buf conv = {0};
    if (!dz_is_text(h, n)) {
        for (size_t i = 0; i < n; i++) {
            unsigned char c = (unsigned char)h[i];
            if (c < 0x80)
                janas_buf_put(&conv, (const char *)&c, 1);
            else {
                char b[2] = {(char)(0xc0 | c >> 6), (char)(0x80 | (c & 0x3f))};
                janas_buf_put(&conv, b, 2);
            }
        }
        if (conv.oom) {
            janas_buf_free(&conv);
            return -1;
        }
        h = conv.p;
        n = conv.n;
    }
    struct hs s = {.out = out, .item = (size_t)-1};
    /* the new lines already at out's end (a book's chapter after another) */
    for (size_t k = out->n; k > 0 && out->p[k - 1] == '\n' && s.nl < 2; k--)
        s.nl++;
    size_t base = out->n;
    /* the title, from the head that is otherwise skipped */
    const char *ti = NULL;
    for (size_t i = 0; i + 7 <= n && !ti; i++)
        if (h[i] == '<' && !strncasecmp(h + i + 1, "title", 5) &&
            (h[i + 6] == '>' || h[i + 6] == ' '))
            ti = h + i;
    if (ti) {
        const char *a = memchr(ti, '>', (size_t)(h + n - ti));
        const char *z = a ? a : NULL;
        while (z && z < h + n && *z != '<')
            z++;
        if (a && z && z > a + 1) {
            ht_put(&s, "# ", 2);
            ht_text(&s, a + 1, (size_t)(z - a - 1));
            ht_newline(&s, 2);
            ht_newline(&s, 2);
        }
    }
    size_t i = 0, run = 0;
    while (i < n) {
        if (h[i] != '<') {
            i++;
            continue;
        }
        ht_text(&s, h + run, i - run);
        /* a comment, a declaration, CDATA */
        if (i + 4 <= n && !memcmp(h + i, "<!--", 4)) {
            const char *e = NULL;
            for (size_t k = i + 4; k + 3 <= n && !e; k++)
                if (!memcmp(h + k, "-->", 3))
                    e = h + k;
            i = e ? (size_t)(e - h) + 3 : n;
            run = i;
            continue;
        }
        size_t k = i + 1;
        int closing = k < n && h[k] == '/';
        k += closing;
        size_t nm = k;
        while (k < n &&
               (isalnum((unsigned char)h[k]) || h[k] == ':' || h[k] == '-'))
            k++;
        size_t nn = k - nm;
        char quote = 0;
        while (k < n && (quote || h[k] != '>')) {
            if (quote && h[k] == quote)
                quote = 0;
            else if (!quote && (h[k] == '"' || h[k] == '\''))
                quote = h[k];
            k++;
        }
        int empty = k > 0 && k < n && h[k - 1] == '/';
        i = run = k < n ? k + 1 : n;
        if (!nn)
            continue; /* <!DOCTYPE>, <?xml?> */
        const char *tag = h + nm;
        static const char *const RAW[] = {"script", "style", "textarea", NULL};
        if (!closing && !empty && ht_named(tag, nn, RAW)) {
            /* raw text: no tags inside, up to its own end tag */
            size_t k2 = i;
            while (k2 + 2 + nn <= n && !(h[k2] == '<' && h[k2 + 1] == '/' &&
                                         !strncasecmp(h + k2 + 2, tag, nn)))
                k2++;
            const char *e2 =
                k2 + 2 + nn <= n ? memchr(h + k2, '>', n - k2) : NULL;
            i = run = e2 ? (size_t)(e2 - h) + 1 : n;
            continue;
        }
        if (ht_named(tag, nn, SKIP)) {
            if (!closing && !empty)
                s.skip++;
            else if (closing && s.skip)
                s.skip--;
            continue;
        }
        if (s.skip)
            continue;
        if (nn == 3 && !strncasecmp(tag, "pre", 3)) {
            ht_newline(&s, 2);
            s.pre = !closing;
        } else if (nn == 2 && tolower((unsigned char)tag[0]) == 'h' &&
                   tag[1] >= '1' && tag[1] <= '6') {
            ht_newline(&s, 2);
            if (!closing) {
                ht_newline(&s, 2);
                ht_put(&s, "######", (size_t)(tag[1] - '0'));
                ht_put(&s, " ", 1);
                s.nl = 0;
            }
        } else if (nn == 2 && !strncasecmp(tag, "li", 2)) {
            ht_newline(&s, 1);
            if (!closing) {
                ht_put(&s, "- ", 2);
                s.item = out->n;
            }
        } else if (nn == 2 && !strncasecmp(tag, "br", 2)) {
            ht_newline(&s, 2);
        } else if (nn == 2 &&
                   (!strncasecmp(tag, "td", 2) || !strncasecmp(tag, "th", 2))) {
            if (!closing && !empty)
                s.cell++;
            else if (closing && s.cell) {
                s.cell--;
                while (out->n > base && out->p[out->n - 1] == ' ')
                    out->n--;
                janas_buf_put(out, "\t", 1);
            }
            s.space = 0;
        } else if (ht_named(tag, nn, BLOCK)) {
            ht_newline(&s, 1);
        }
    }
    ht_text(&s, h + run, n - run);
    /* no spaces nor tabs at a line's end, no empty lines at the end */
    size_t w = base;
    for (size_t r = base; r < out->n; r++) {
        if (out->p[r] == '\n')
            while (w > base && (out->p[w - 1] == ' ' || out->p[w - 1] == '\t'))
                w--;
        out->p[w++] = out->p[r];
    }
    out->n = w;
    while (out->n > base &&
           (out->p[out->n - 1] == '\n' || out->p[out->n - 1] == ' ' ||
            out->p[out->n - 1] == '\t'))
        out->n--;
    janas_buf_free(&conv);
    return out->oom ? -1 : 0;
}

/* attribute name of the tag [t, e) into v (as office.c's) */
static int ht_attr(const char *t, const char *e, const char *name, char *v,
                   size_t cap)
{
    size_t nl = strlen(name);
    for (const char *q = t + 1; q + nl + 2 < e; q++) {
        if ((q[-1] != ' ' && q[-1] != '\t' && q[-1] != '\n' && q[-1] != '\r') ||
            memcmp(q, name, nl) || q[nl] != '=' ||
            (q[nl + 1] != '"' && q[nl + 1] != '\''))
            continue;
        char quote = q[nl + 1];
        const char *a = q + nl + 2, *b = a;
        while (b < e && *b != quote)
            b++;
        size_t n = (size_t)(b - a) < cap - 1 ? (size_t)(b - a) : cap - 1;
        memcpy(v, a, n);
        v[n] = 0;
        return 1;
    }
    return 0;
}

/* each tag named name (lower case, with a prefix such as opf: allowed) */
static const char *ht_next_tag(const char *p, const char *end, const char *name,
                               const char **tag_end)
{
    size_t nl = strlen(name);
    for (; p < end; p++) {
        if (*p != '<' || p[1] == '/')
            continue;
        const char *q = p + 1, *colon = NULL;
        const char *z = q;
        while (z < end && (isalnum((unsigned char)*z) || *z == ':' ||
                           *z == '-' || *z == '_')) {
            if (*z == ':')
                colon = z;
            z++;
        }
        const char *local = colon ? colon + 1 : q;
        if ((size_t)(z - local) == nl && !strncasecmp(local, name, nl)) {
            const char *e = memchr(z, '>', (size_t)(end - z));
            *tag_end = e ? e + 1 : end;
            return p;
        }
    }
    return NULL;
}

/* %-escapes of an href undone, its #fragment dropped */
static void ht_unhref(const char *in, char *out, size_t cap)
{
    size_t k = 0;
    for (; *in && *in != '#' && k + 1 < cap; in++) {
        if (in[0] == '%' && isxdigit((unsigned char)in[1]) &&
            isxdigit((unsigned char)in[2])) {
            char hx[3] = {in[1], in[2], 0};
            out[k++] = (char)strtol(hx, NULL, 16);
            in += 2;
        } else
            out[k++] = *in;
    }
    out[k] = 0;
}

int dz_epub_text(const uint8_t *p, size_t n, struct janas_buf *out, char *why,
                 size_t wn)
{
    char *c = NULL, *opf = NULL;
    size_t cn, on;
    if (dz_zip_get(p, n, "META-INF/container.xml", (uint8_t **)&c, &cn, why,
                   wn) != 0) {
        if (!why[0])
            snprintf(why, wn, "no META-INF/container.xml: not an EPUB");
        return -1;
    }
    const char *te, *root = ht_next_tag(c, c + cn, "rootfile", &te);
    char path[512] = "", raw[512];
    if (root && ht_attr(root, te, "full-path", raw, sizeof raw))
        ht_unhref(raw, path, sizeof path);
    free(c);
    if (!path[0] ||
        dz_zip_get(p, n, path, (uint8_t **)&opf, &on, why, wn) != 0) {
        free(opf);
        if (!why[0])
            snprintf(why, wn, "its package (.opf) is not there");
        return -1;
    }
    /* the package's folder: the chapters' hrefs are under it */
    char dir[512];
    snprintf(dir, sizeof dir, "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash)
        slash[1] = 0;
    else
        dir[0] = 0;
    const char *end = opf + on, *q = opf;
    int chapters = 0;
    for (const char *ir;
         chapters < MAX_CHAPTERS && (ir = ht_next_tag(q, end, "itemref", &te));
         chapters++) {
        q = te;
        char idref[256];
        if (!ht_attr(ir, te, "idref", idref, sizeof idref))
            continue;
        /* its item in the manifest */
        const char *ie, *mq = opf, *href_tag = NULL;
        char id[256];
        for (const char *it; (it = ht_next_tag(mq, end, "item", &ie));) {
            mq = ie;
            if (ht_attr(it, ie, "id", id, sizeof id) && !strcmp(id, idref)) {
                href_tag = it;
                break;
            }
        }
        char href[512], file[1100];
        if (!href_tag || !ht_attr(href_tag, ie, "href", raw, sizeof raw))
            continue;
        ht_unhref(raw, href, sizeof href);
        snprintf(file, sizeof file, "%s%s", dir, href);
        char *x = NULL;
        size_t xn;
        if (dz_zip_get(p, n, file, (uint8_t **)&x, &xn, why, wn) != 0) {
            free(x);
            why[0] = 0;
            continue;
        }
        while (out->n && out->p[out->n - 1] == '\n')
            out->n--;
        if (out->n)
            janas_buf_puts(out, "\n\n");
        dz_html_text(x, xn, out);
        free(x);
    }
    free(opf);
    why[0] = 0;
    if (out->oom) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    return 0;
}
