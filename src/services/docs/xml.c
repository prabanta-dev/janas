/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * xml.c - the text of a word processor's document: word/document.xml of a
 * docx, content.xml of an odt. Not an XML parser: the tags that end a
 * paragraph, break a line or stand for a tab or spaces are turned into
 * them, the headings get the "#" of Markdown (their level from the style,
 * Heading1 or text:outline-level), the entities become their characters,
 * and only the text a reader sees is kept - a docx's w:t runs (not its
 * deleted text nor its fields' codes), an odt's office:body without its
 * comments. The document is input from outside: every read is bounded by
 * its length, a tag left open at the end only ends the text.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "docs.h"

struct xs {
    const char *x;
    size_t n, at;
    struct janas_buf *out;
    int kind;
    int in_t;    /* docx: inside a w:t */
    int in_body; /* odt: inside office:body */
    int skip;    /* odt: inside a comment (office:annotation), its depth */
    int nl;      /* newlines just written, at most two kept */
};

static void emit(struct xs *s, const char *p, size_t n)
{
    if (!n)
        return;
    janas_buf_put(s->out, p, n);
    s->nl = 0;
}

static void newline(struct xs *s)
{
    if (s->out->n && s->nl < 2) {
        janas_buf_put(s->out, "\n", 1);
        s->nl++;
    }
}

static int keeping(const struct xs *s)
{
    return s->kind == DZ_DOCX ? s->in_t : s->in_body && !s->skip;
}

/* a code point as UTF-8 */
static void utf8(struct xs *s, unsigned long c)
{
    char b[4];
    size_t n;
    if (c == 0 || c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff))
        c = 0xfffd;
    if (c < 0x80) {
        b[0] = (char)c;
        n = 1;
    } else if (c < 0x800) {
        b[0] = (char)(0xc0 | c >> 6);
        b[1] = (char)(0x80 | (c & 0x3f));
        n = 2;
    } else if (c < 0x10000) {
        b[0] = (char)(0xe0 | c >> 12);
        b[1] = (char)(0x80 | (c >> 6 & 0x3f));
        b[2] = (char)(0x80 | (c & 0x3f));
        n = 3;
    } else {
        b[0] = (char)(0xf0 | c >> 18);
        b[1] = (char)(0x80 | (c >> 12 & 0x3f));
        b[2] = (char)(0x80 | (c >> 6 & 0x3f));
        b[3] = (char)(0x80 | (c & 0x3f));
        n = 4;
    }
    emit(s, b, n);
}

/* text from s->at to the next '<', its entities turned back */
static void text(struct xs *s)
{
    const char *x = s->x;
    size_t i = s->at, run = i;
    while (i < s->n && x[i] != '<') {
        if (x[i] != '&') {
            i++;
            continue;
        }
        if (keeping(s))
            emit(s, x + run, i - run);
        size_t e = i + 1;
        while (e < s->n && e - i < 12 && x[e] != ';' && x[e] != '<')
            e++;
        if (e >= s->n || x[e] != ';') { /* a bare '&': kept as it is */
            if (keeping(s))
                emit(s, "&", 1);
            run = ++i;
            continue;
        }
        const char *en = x + i + 1;
        size_t len = e - i - 1;
        if (keeping(s)) {
            if (len == 3 && !memcmp(en, "amp", 3))
                emit(s, "&", 1);
            else if (len == 2 && !memcmp(en, "lt", 2))
                emit(s, "<", 1);
            else if (len == 2 && !memcmp(en, "gt", 2))
                emit(s, ">", 1);
            else if (len == 4 && !memcmp(en, "quot", 4))
                emit(s, "\"", 1);
            else if (len == 4 && !memcmp(en, "apos", 4))
                emit(s, "'", 1);
            else if (len >= 2 && en[0] == '#') {
                int hex = en[1] == 'x' || en[1] == 'X';
                unsigned long c = 0;
                int ok = len > (size_t)(1 + hex);
                for (size_t k = 1 + hex; k < len && ok; k++) {
                    char d = en[k];
                    int v = d >= '0' && d <= '9'          ? d - '0'
                            : hex && d >= 'a' && d <= 'f' ? d - 'a' + 10
                            : hex && d >= 'A' && d <= 'F' ? d - 'A' + 10
                                                          : -1;
                    ok = v >= 0 && c <= 0x10ffff;
                    c = c * (hex ? 16 : 10) + (unsigned long)(v < 0 ? 0 : v);
                }
                utf8(s, ok ? c : 0xfffd);
            }
        }
        run = i = e + 1;
    }
    if (keeping(s))
        emit(s, x + run, i - run);
    s->at = i;
}

/* the value of attribute name in the tag's text [p, e), into v */
static int attr(const char *p, const char *e, const char *name, char *v,
                size_t cap)
{
    size_t nl = strlen(name);
    for (const char *q = p; q + nl + 2 < e; q++) {
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

static void heading(struct xs *s, int level)
{
    if (level < 1 || level > 6)
        return;
    newline(s);
    emit(s, "######", (size_t)level);
    emit(s, " ", 1);
}

static int is(const char *name, size_t n, const char *w)
{
    return n == strlen(w) && !memcmp(name, w, n);
}

/* a tag at s->at ('<'): what it means for the text */
static void tag(struct xs *s)
{
    const char *x = s->x;
    size_t i = s->at + 1;
    if (i < s->n && (x[i] == '?' || x[i] == '!')) {
        const char *end = x[i] == '?' ? "?>" : "-->";
        if (s->n - i >= 8 && !memcmp(x + i, "![CDATA[", 8))
            end = "]]>";
        else if (x[i] == '!' && (s->n - i < 3 || memcmp(x + i, "!--", 3)))
            end = ">"; /* a declaration */
        size_t el = strlen(end), k = i;
        while (k + el <= s->n && memcmp(x + k, end, el))
            k++;
        if (!strcmp(end, "]]>") && keeping(s))
            emit(s, x + i + 8, k > i + 8 ? k - i - 8 : 0);
        s->at = k + el <= s->n ? k + el : s->n;
        return;
    }
    int closing = i < s->n && x[i] == '/';
    i += closing;
    size_t name = i;
    while (i < s->n && x[i] != ' ' && x[i] != '\t' && x[i] != '\n' &&
           x[i] != '\r' && x[i] != '/' && x[i] != '>')
        i++;
    size_t nn = i - name;
    char quote = 0;
    while (i < s->n && (quote || x[i] != '>')) {
        if (quote && x[i] == quote)
            quote = 0;
        else if (!quote && (x[i] == '"' || x[i] == '\''))
            quote = x[i];
        i++;
    }
    if (i >= s->n) { /* left open: the text ends here */
        s->at = s->n;
        return;
    }
    int empty = !closing && i > name && x[i - 1] == '/';
    const char *nm = x + name, *ae = x + i;
    s->at = i + 1;
    char v[64];
    if (s->kind == DZ_DOCX) {
        if (is(nm, nn, "w:t"))
            s->in_t = !closing && !empty;
        else if (is(nm, nn, "w:p") && (closing || empty))
            newline(s);
        else if (is(nm, nn, "w:tab") && !closing)
            emit(s, "\t", 1);
        else if ((is(nm, nn, "w:br") || is(nm, nn, "w:cr")) && !closing)
            newline(s);
        else if (is(nm, nn, "w:pStyle") && !closing &&
                 attr(nm, ae, "w:val", v, sizeof v)) {
            /* Heading1..6, or a localized style named the same way */
            size_t k = strlen(v);
            if (k > 1 && v[k - 1] >= '1' && v[k - 1] <= '6' &&
                (strstr(v, "eading") || strstr(v, "itolo") ||
                 strstr(v, "berschrift") || strstr(v, "itre")))
                heading(s, v[k - 1] - '0');
        }
        return;
    }
    if (is(nm, nn, "office:body")) {
        s->in_body = !closing && !empty;
    } else if (is(nm, nn, "office:annotation")) {
        if (!closing && !empty)
            s->skip++;
        else if (closing && s->skip)
            s->skip--;
    } else if (is(nm, nn, "text:p") || is(nm, nn, "text:h")) {
        if ((closing || empty) && keeping(s))
            newline(s);
        else if (is(nm, nn, "text:h") &&
                 attr(nm, ae, "text:outline-level", v, sizeof v))
            heading(s, atoi(v));
    } else if (is(nm, nn, "text:tab") && !closing && keeping(s)) {
        emit(s, "\t", 1);
    } else if (is(nm, nn, "text:line-break") && !closing && keeping(s)) {
        newline(s);
    } else if (is(nm, nn, "text:s") && !closing && keeping(s)) {
        int c = attr(nm, ae, "text:c", v, sizeof v) ? atoi(v) : 1;
        for (int k = 0; k < (c < 1 ? 1 : c > 64 ? 64 : c); k++)
            emit(s, " ", 1);
    }
}

int dz_xml_text(const char *x, size_t n, int kind, struct janas_buf *out)
{
    struct xs s = {.x = x, .n = n, .out = out, .kind = kind};
    while (s.at < s.n) {
        if (x[s.at] == '<')
            tag(&s);
        else
            text(&s);
    }
    while (out->n && out->p[out->n - 1] == '\n')
        out->n--;
    return out->oom ? -1 : 0;
}
