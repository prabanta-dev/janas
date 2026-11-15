/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * rtf.c - the text of an RTF document: its words, \par and \line as new
 * lines, \tab and the end of a table's cell as tabs, \'hh as a character
 * of Windows-1252 (the code page of nearly every RTF there is), \uN as
 * the code point N with the \ucN characters that follow it for older
 * readers skipped. Left out, with all they hold: the font, colour and
 * style tables, pictures, objects, the document's information, and every
 * destination marked \* that is not known. The document is input from
 * outside: groups deeper than 256 are not followed, every read is bounded.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "docs.h"

#define DEPTH 256

/* Windows-1252's 0x80-0x9f; the rest is Latin-1 */
static const unsigned short CP1252[32] = {
    0x20ac, 0xfffd, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021,
    0x02c6, 0x2030, 0x0160, 0x2039, 0x0152, 0xfffd, 0x017d, 0xfffd,
    0xfffd, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014,
    0x02dc, 0x2122, 0x0161, 0x203a, 0x0153, 0xfffd, 0x017e, 0x0178};

static void rt_utf8(struct janas_buf *o, unsigned long c)
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

static unsigned long rt_cp1252(unsigned char c)
{
    return c >= 0x80 && c < 0xa0 ? CP1252[c - 0x80] : c;
}

/* the destinations whose text is not the document's */
static int rt_dropped(const char *w)
{
    static const char *const D[] = {"fonttbl",
                                    "colortbl",
                                    "stylesheet",
                                    "info",
                                    "pict",
                                    "object",
                                    "header",
                                    "footer",
                                    "headerl",
                                    "headerr",
                                    "headerf",
                                    "footerl",
                                    "footerr",
                                    "footerf",
                                    "fldinst",
                                    "themedata",
                                    "colorschememapping",
                                    "datastore",
                                    "latentstyles",
                                    "listtable",
                                    "listoverridetable",
                                    "rsidtbl",
                                    "xmlnstbl",
                                    "generator",
                                    "filetbl",
                                    "revtbl",
                                    "bkmkstart",
                                    "bkmkend",
                                    NULL};
    for (int i = 0; D[i]; i++)
        if (!strcmp(w, D[i]))
            return 1;
    return 0;
}

int dz_rtf_text(const char *r, size_t n, struct janas_buf *out, char *why,
                size_t wn)
{
    if (n < 5 || memcmp(r, "{\\rtf", 5)) {
        snprintf(why, wn, "not an RTF document");
        return -1;
    }
    /* each group's state: skipped, and how many characters a \u has after
       it for older readers */
    struct {
        unsigned char skip, uc;
    } st[DEPTH];
    int d = 0;
    st[0].skip = 0;
    st[0].uc = 1;
    int star = 0;    /* a \* was just read: the next word's group is dropped
                        unless known */
    int pending = 0; /* characters of a \u's fallback still to skip */
    size_t base = out->n;
    for (size_t i = 0; i < n;) {
        char c = r[i];
        if (c == '{') {
            if (d + 1 < DEPTH) {
                st[d + 1] = st[d];
                d++;
            } else {
                st[d].skip = 1; /* too deep: what is inside is not read */
            }
            star = 0;
            i++;
            continue;
        }
        if (c == '}') {
            if (d > 0)
                d--;
            star = 0;
            pending = 0;
            i++;
            continue;
        }
        if (c == '\r' || c == '\n') {
            i++;
            continue;
        }
        if (c != '\\') {
            if (pending) {
                pending--;
            } else if (!st[d].skip) {
                rt_utf8(out, rt_cp1252((unsigned char)c));
            }
            i++;
            continue;
        }
        /* a control */
        i++;
        if (i >= n)
            break;
        c = r[i];
        if (c == '\'') { /* \'hh */
            unsigned v = 0;
            int ok = i + 2 < n && isxdigit((unsigned char)r[i + 1]) &&
                     isxdigit((unsigned char)r[i + 2]);
            if (ok) {
                char hx[3] = {r[i + 1], r[i + 2], 0};
                v = (unsigned)strtoul(hx, NULL, 16);
                i += 3;
            } else
                i++;
            if (pending)
                pending--;
            else if (ok && !st[d].skip)
                rt_utf8(out, rt_cp1252((unsigned char)v));
            continue;
        }
        if (c == '*') {
            star = 1;
            i++;
            continue;
        }
        if (!isalpha((unsigned char)c)) { /* a control symbol */
            i++;
            if (st[d].skip)
                continue;
            if (c == '\\' || c == '{' || c == '}')
                rt_utf8(out, (unsigned char)c);
            else if (c == '~')
                rt_utf8(out, 0xa0);
            else if (c == '_')
                rt_utf8(out, 0x2011);
            else if (c == '\n' || c == '\r')
                janas_buf_put(out, "\n", 1);
            continue;
        }
        char w[32];
        size_t k = 0;
        while (i < n && isalpha((unsigned char)r[i]) && k + 1 < sizeof w)
            w[k++] = r[i++];
        while (i < n && isalpha((unsigned char)r[i]))
            i++;
        w[k] = 0;
        long num = 0;
        int has = 0, neg = 0;
        if (i < n && r[i] == '-')
            neg = 1, i++;
        while (i < n && isdigit((unsigned char)r[i])) {
            if (num < 10000000)
                num = num * 10 + (r[i] - '0');
            has = 1;
            i++;
        }
        if (neg)
            num = -num;
        if (i < n && r[i] == ' ')
            i++; /* the space that ends a control word */
        if (star || rt_dropped(w)) {
            st[d].skip = 1;
            star = 0;
            continue;
        }
        if (!strcmp(w, "uc") && has) {
            st[d].uc = (unsigned char)(num < 0 ? 0 : num > 8 ? 8 : num);
            continue;
        }
        if (st[d].skip)
            continue;
        if (!strcmp(w, "u") && has) {
            if (num < 0)
                num += 65536;
            rt_utf8(out, (unsigned long)num);
            pending = st[d].uc;
        } else if (!strcmp(w, "par") || !strcmp(w, "line") ||
                   !strcmp(w, "row") || !strcmp(w, "sect") ||
                   !strcmp(w, "page")) {
            janas_buf_put(out, "\n", 1);
        } else if (!strcmp(w, "tab") || !strcmp(w, "cell")) {
            janas_buf_put(out, "\t", 1);
        } else if (!strcmp(w, "emdash")) {
            rt_utf8(out, 0x2014);
        } else if (!strcmp(w, "endash")) {
            rt_utf8(out, 0x2013);
        } else if (!strcmp(w, "bullet")) {
            rt_utf8(out, 0x2022);
        } else if (!strcmp(w, "lquote")) {
            rt_utf8(out, 0x2018);
        } else if (!strcmp(w, "rquote")) {
            rt_utf8(out, 0x2019);
        } else if (!strcmp(w, "ldblquote")) {
            rt_utf8(out, 0x201c);
        } else if (!strcmp(w, "rdblquote")) {
            rt_utf8(out, 0x201d);
        }
    }
    /* at most one empty line in a row, none at the end */
    size_t wpos = base;
    int nl = 0;
    for (size_t k = base; k < out->n; k++) {
        if (out->p[k] == '\n') {
            while (wpos > base &&
                   (out->p[wpos - 1] == ' ' || out->p[wpos - 1] == '\t'))
                wpos--;
            if (++nl > 2)
                continue;
        } else
            nl = 0;
        out->p[wpos++] = out->p[k];
    }
    out->n = wpos;
    while (out->n > base && out->p[out->n - 1] == '\n')
        out->n--;
    if (out->oom) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    return 0;
}
