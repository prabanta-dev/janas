/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * office.c - the text of Microsoft Office's workbooks and presentations,
 * zip archives of XML files as a docx is:
 *
 *  - an xlsx: its sheets in the workbook's order (xl/workbook.xml, their
 *    files found through xl/_rels/workbook.xml.rels), each headed by its
 *    name, a row a line, the cells by tabs - the shared strings
 *    (xl/sharedStrings.xml) put back where a cell names one, a number as
 *    it is written, a formula's last value;
 *  - a pptx: its slides in the presentation's order (ppt/presentation.xml
 *    and its rels), each headed "## Slide N", a paragraph (a:p) a line.
 *
 * Not an XML parser: the few elements that hold the text are looked for
 * in a file the archive gives NUL-terminated, every read bounded by it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "docs.h"

#define MAX_PARTS 512
#define MAX_COLS 256

static void of_utf8(struct janas_buf *o, unsigned long c)
{
    char b[4];
    if (c == 0 || c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff))
        c = 0xfffd;
    if (c < 0x80) {
        b[0] = (char)c;
        janas_buf_put(o, b, 1);
    } else if (c < 0x800) {
        b[0] = (char)(0xc0 | c >> 6);
        b[1] = (char)(0x80 | (c & 0x3f));
        janas_buf_put(o, b, 2);
    } else if (c < 0x10000) {
        b[0] = (char)(0xe0 | c >> 12);
        b[1] = (char)(0x80 | (c >> 6 & 0x3f));
        b[2] = (char)(0x80 | (c & 0x3f));
        janas_buf_put(o, b, 3);
    } else {
        b[0] = (char)(0xf0 | c >> 18);
        b[1] = (char)(0x80 | (c >> 12 & 0x3f));
        b[2] = (char)(0x80 | (c >> 6 & 0x3f));
        b[3] = (char)(0x80 | (c & 0x3f));
        janas_buf_put(o, b, 4);
    }
}

void dz_unescape(const char *s, size_t n, struct janas_buf *out)
{
    static const struct {
        const char *name;
        unsigned long c;
    } named[] = {{"amp", '&'},      {"lt", '<'},       {"gt", '>'},
                 {"quot", '"'},     {"apos", '\''},    {"nbsp", ' '},
                 {"ndash", 0x2013}, {"mdash", 0x2014}, {"hellip", 0x2026},
                 {"laquo", 0xab},   {"raquo", 0xbb},   {"lsquo", 0x2018},
                 {"rsquo", 0x2019}, {"ldquo", 0x201c}, {"rdquo", 0x201d},
                 {"copy", 0xa9},    {"reg", 0xae},     {"trade", 0x2122},
                 {"euro", 0x20ac},  {"times", 0xd7},   {"middot", 0xb7},
                 {"bull", 0x2022},  {"deg", 0xb0},     {"agrave", 0xe0},
                 {"egrave", 0xe8},  {"eacute", 0xe9},  {"igrave", 0xec},
                 {"ograve", 0xf2},  {"ugrave", 0xf9},  {"Agrave", 0xc0},
                 {"Egrave", 0xc8},  {"Eacute", 0xc9},  {"ccedil", 0xe7},
                 {"uuml", 0xfc},    {"ouml", 0xf6},    {"auml", 0xe4},
                 {"szlig", 0xdf},   {"ntilde", 0xf1},  {"shy", 0xad}};
    size_t run = 0, i = 0;
    while (i < n) {
        if (s[i] != '&') {
            i++;
            continue;
        }
        janas_buf_put(out, s + run, i - run);
        size_t e = i + 1;
        while (e < n && e - i < 12 && s[e] != ';' && s[e] != '&' && s[e] != '<')
            e++;
        if (e >= n || s[e] != ';') { /* a bare '&': as it is */
            janas_buf_put(out, "&", 1);
            run = ++i;
            continue;
        }
        const char *en = s + i + 1;
        size_t len = e - i - 1;
        int done = 0;
        if (len >= 2 && en[0] == '#') {
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
            of_utf8(out, ok ? c : 0xfffd);
            done = 1;
        }
        for (size_t k = 0; !done && k < sizeof named / sizeof *named; k++)
            if (strlen(named[k].name) == len &&
                !memcmp(named[k].name, en, len)) {
                of_utf8(out, named[k].c);
                done = 1;
            }
        if (!done) /* unknown: as it is */
            janas_buf_put(out, s + i, e + 1 - i);
        run = i = e + 1;
    }
    janas_buf_put(out, s + run, n - run);
}

/* the next tag named name (with its attributes) at or after p, in a text
   ending at end: where it starts, or NULL */
static const char *of_find_tag(const char *p, const char *end, const char *name)
{
    size_t nl = strlen(name);
    while (p && p < end) {
        p = memchr(p, '<', (size_t)(end - p));
        if (!p || (size_t)(end - p) < nl + 2)
            return NULL;
        if (!memcmp(p + 1, name, nl) &&
            (p[nl + 1] == '>' || p[nl + 1] == ' ' || p[nl + 1] == '/' ||
             p[nl + 1] == '\t' || p[nl + 1] == '\n' || p[nl + 1] == '\r'))
            return p;
        p++;
    }
    return NULL;
}

/* the end of the tag starting at t ('>' included), or end */
static const char *of_tag_end(const char *t, const char *end)
{
    const char *e = memchr(t, '>', (size_t)(end - t));
    return e ? e + 1 : end;
}

/* attribute name of the tag [t, e) into v */
static int of_attr(const char *t, const char *e, const char *name, char *v,
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

/* the text of every <name>...</name> (a:t, t) between p and end, appended */
static void of_texts(const char *p, const char *end, const char *name,
                     struct janas_buf *out)
{
    char close[32];
    snprintf(close, sizeof close, "</%s>", name);
    for (const char *t; (t = of_find_tag(p, end, name));) {
        const char *e = of_tag_end(t, end);
        if (e[-2] == '/') { /* <t/> */
            p = e;
            continue;
        }
        const char *c = e;
        size_t cl = strlen(close);
        while (c + cl <= end && memcmp(c, close, cl))
            c++;
        if (c + cl > end)
            return;
        dz_unescape(e, (size_t)(c - e), out);
        p = c + cl;
    }
}

/* a part of the archive, NUL-terminated: 0, 1 when there is none, -1 */
static int of_part(const uint8_t *z, size_t n, const char *name, char **x,
                   size_t *xn, char *why, size_t wn)
{
    return dz_zip_get(z, n, name, (uint8_t **)x, xn, why, wn);
}

/* the targets of the relationships of a rels file, by their ids */
struct rel {
    char id[64], target[256];
};

static int of_rels(const char *x, size_t n, struct rel *r, int max)
{
    const char *p = x, *end = x + n;
    int k = 0;
    for (const char *t; k < max && (t = of_find_tag(p, end, "Relationship"));) {
        const char *e = of_tag_end(t, end);
        if (of_attr(t, e, "Id", r[k].id, sizeof r[k].id) &&
            of_attr(t, e, "Target", r[k].target, sizeof r[k].target))
            k++;
        p = e;
    }
    return k;
}

/* a target as a name in the archive: under dir unless it starts with / */
static void of_target(const char *dir, const char *target, char *out,
                      size_t cap)
{
    if (target[0] == '/')
        snprintf(out, cap, "%s", target + 1);
    else
        snprintf(out, cap, "%s%s", dir, target);
}

/* ---- xlsx ---- */

/* "BC12": its column, from 0 */
static int of_column(const char *ref)
{
    int c = 0, k = 0;
    while (*ref >= 'A' && *ref <= 'Z' && k++ < 4)
        c = c * 26 + (*ref++ - 'A' + 1);
    return c - 1;
}

int dz_xlsx_text(const uint8_t *p, size_t n, struct janas_buf *out, char *why,
                 size_t wn)
{
    char *wb = NULL, *rl = NULL, *ss = NULL;
    size_t wbn, rln, ssn;
    if (of_part(p, n, "xl/workbook.xml", &wb, &wbn, why, wn) != 0) {
        if (!why[0])
            snprintf(why, wn, "no xl/workbook.xml: not an Excel workbook");
        return -1;
    }
    /* each call its own (the index and a tool read at once) */
    struct rel *r = calloc(MAX_PARTS, sizeof *r);
    if (!r) {
        free(rl);
        snprintf(why, wn, "out of memory");
        return -1;
    }
    int nr =
        of_part(p, n, "xl/_rels/workbook.xml.rels", &rl, &rln, why, wn) == 0
            ? of_rels(rl, rln, r, MAX_PARTS)
            : 0;
    /* the shared strings, each one's text in a buffer, by their index */
    struct janas_buf strs = {0};
    size_t *at = NULL, nat = 0;
    if (of_part(p, n, "xl/sharedStrings.xml", &ss, &ssn, why, wn) == 0) {
        const char *q = ss, *end = ss + ssn;
        for (const char *t; (t = of_find_tag(q, end, "si"));) {
            const char *c = t;
            const char *stop = NULL;
            for (const char *k = of_tag_end(t, end); k + 5 <= end; k++)
                if (!memcmp(k, "</si>", 5)) {
                    stop = k;
                    break;
                }
            if (!stop)
                break;
            size_t *na = realloc(at, (nat + 1) * sizeof *at);
            if (!na)
                break;
            at = na;
            at[nat++] = strs.n;
            of_texts(c, stop, "t", &strs);
            janas_buf_put(&strs, "", 1);
            q = stop + 5;
        }
    }
    why[0] = 0;
    const char *q = wb, *end = wb + wbn;
    int sheets = 0;
    for (const char *t; sheets < MAX_PARTS && out->n < DZ_TEXT_MAX &&
                        (t = of_find_tag(q, end, "sheet"));
         sheets++) {
        const char *e = of_tag_end(t, end);
        char name[256] = "", rid[64] = "", file[512];
        of_attr(t, e, "name", name, sizeof name);
        of_attr(t, e, "r:id", rid, sizeof rid);
        q = e;
        const char *target = NULL;
        for (int i = 0; i < nr && !target; i++)
            if (!strcmp(r[i].id, rid))
                target = r[i].target;
        if (target)
            of_target("xl/", target, file, sizeof file);
        else
            snprintf(file, sizeof file, "xl/worksheets/sheet%d.xml",
                     sheets + 1);
        char *x = NULL;
        size_t xn;
        if (of_part(p, n, file, &x, &xn, why, wn) != 0) {
            free(x);
            why[0] = 0;
            continue;
        }
        if (out->n)
            janas_buf_puts(out, "\n\n");
        janas_buf_puts(out, "## ");
        dz_unescape(name, strlen(name), out);
        janas_buf_puts(out, "\n");
        const char *rp = x, *rend = x + xn;
        for (const char *row;
             out->n < DZ_TEXT_MAX && (row = of_find_tag(rp, rend, "row"));) {
            const char *re = of_tag_end(row, rend);
            if (re[-2] == '/') { /* an empty row */
                rp = re;
                continue;
            }
            const char *stop = NULL;
            for (const char *k = re; k + 6 <= rend; k++)
                if (!memcmp(k, "</row>", 6)) {
                    stop = k;
                    break;
                }
            if (!stop)
                break;
            int last = -1; /* the column of the last cell written */
            for (const char *c;
                 out->n < DZ_TEXT_MAX && (c = of_find_tag(re, stop, "c"));) {
                const char *ce = of_tag_end(c, stop);
                char ref[16] = "", type[16] = "";
                of_attr(c, ce, "r", ref, sizeof ref);
                of_attr(c, ce, "t", type, sizeof type);
                re = ce;
                if (ce[-2] == '/')
                    continue; /* an empty cell */
                const char *cs = NULL;
                for (const char *k = ce; k + 4 <= stop; k++)
                    if (!memcmp(k, "</c>", 4)) {
                        cs = k;
                        break;
                    }
                if (!cs)
                    break;
                re = cs + 4;
                struct janas_buf v = {0};
                if (!strcmp(type, "inlineStr"))
                    of_texts(ce, cs, "t", &v);
                else
                    of_texts(ce, cs, "v", &v);
                janas_buf_put(&v, "", 1);
                if (!v.oom && v.n > 1) {
                    const char *val = v.p;
                    if (!strcmp(type, "s")) {
                        size_t idx = (size_t)strtoul(v.p, NULL, 10);
                        val = idx < nat ? strs.p + at[idx] : "";
                    } else if (!strcmp(type, "b"))
                        val = v.p[0] == '1' ? "TRUE" : "FALSE";
                    /* a tab for each column from the last one written: the
                       empty cells between keep their places */
                    int c = ref[0] ? of_column(ref) : last + 1;
                    if (c <= last)
                        c = last + 1;
                    if (c > last + MAX_COLS)
                        c = last + MAX_COLS;
                    for (int k = last < 0 ? 0 : last; k < c; k++)
                        janas_buf_put(out, "\t", 1);
                    janas_buf_puts(out, val);
                    last = c;
                }
                janas_buf_free(&v);
            }
            if (last >= 0)
                janas_buf_put(out, "\n", 1);
            rp = stop + 6;
        }
        free(x);
    }
    free(wb);
    free(rl);
    free(r);
    free(ss);
    free(at);
    janas_buf_free(&strs);
    while (out->n && out->p[out->n - 1] == '\n')
        out->n--;
    if (out->oom) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    return 0;
}

/* ---- pptx ---- */

int dz_pptx_text(const uint8_t *p, size_t n, struct janas_buf *out, char *why,
                 size_t wn)
{
    char *pr = NULL, *rl = NULL;
    size_t prn, rln;
    if (of_part(p, n, "ppt/presentation.xml", &pr, &prn, why, wn) != 0) {
        if (!why[0])
            snprintf(why, wn,
                     "no ppt/presentation.xml: not a PowerPoint presentation");
        return -1;
    }
    /* each call its own (the index and a tool read at once) */
    struct rel *r = calloc(MAX_PARTS, sizeof *r);
    if (!r) {
        free(rl);
        snprintf(why, wn, "out of memory");
        return -1;
    }
    int nr = of_part(p, n, "ppt/_rels/presentation.xml.rels", &rl, &rln, why,
                     wn) == 0
                 ? of_rels(rl, rln, r, MAX_PARTS)
                 : 0;
    why[0] = 0;
    const char *q = pr, *end = pr + prn;
    int slide = 0;
    for (const char *t; slide < MAX_PARTS && out->n < DZ_TEXT_MAX &&
                        (t = of_find_tag(q, end, "p:sldId"));) {
        const char *e = of_tag_end(t, end);
        char rid[64] = "", file[512];
        of_attr(t, e, "r:id", rid, sizeof rid);
        q = e;
        const char *target = NULL;
        for (int i = 0; i < nr && !target; i++)
            if (!strcmp(r[i].id, rid))
                target = r[i].target;
        if (!target)
            continue;
        of_target("ppt/", target, file, sizeof file);
        char *x = NULL;
        size_t xn;
        if (of_part(p, n, file, &x, &xn, why, wn) != 0) {
            free(x);
            why[0] = 0;
            continue;
        }
        while (out->n && out->p[out->n - 1] == '\n')
            out->n--;
        janas_buf_printf(out, "%s## Slide %d\n", out->n ? "\n\n" : "", ++slide);
        const char *ap = x, *aend = x + xn;
        for (const char *para;
             out->n < DZ_TEXT_MAX && (para = of_find_tag(ap, aend, "a:p"));) {
            const char *pe = of_tag_end(para, aend);
            if (pe[-2] == '/') {
                ap = pe;
                continue;
            }
            const char *stop = NULL;
            for (const char *k = pe; k + 6 <= aend; k++)
                if (!memcmp(k, "</a:p>", 6)) {
                    stop = k;
                    break;
                }
            if (!stop)
                break;
            size_t before = out->n;
            of_texts(pe, stop, "a:t", out);
            if (out->n > before)
                janas_buf_put(out, "\n", 1);
            ap = stop + 6;
        }
        free(x);
    }
    free(pr);
    free(rl);
    free(r);
    while (out->n && out->p[out->n - 1] == '\n')
        out->n--;
    if (out->oom) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    return 0;
}
