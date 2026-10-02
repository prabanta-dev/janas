/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * read.c - Wikipedia's answers read (see wiki.h): the pages of a search or
 * of the places around a point, a page with its plain text, and the text
 * cut by its headings. The API's plain text (TextExtracts, explaintext)
 * marks a heading as a line "== Name ==", more signs a level deeper; the
 * tables and the boxes are left out of it.
 */
#define _GNU_SOURCE /* strcasestr */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "wiki.h"

static void copy(char *to, size_t cap, const char *s)
{
    snprintf(to, cap, "%s", s ? s : "");
}

/* The API's error, when it gave one ({"error": {"code", "info"}}). */
static int api_error(const struct janas_json *root, char *err, size_t err_len)
{
    const struct janas_json *e = janas_json_get(root, "error");
    if (!e)
        return 0;
    const char *info = janas_json_str(janas_json_get(e, "info"));
    const char *code = janas_json_str(janas_json_get(e, "code"));
    snprintf(err, err_len, "Wikipedia said: %s",
             info   ? info
             : code ? code
                    : "an error");
    return 1;
}

static void point_of(const struct janas_json *page, double *lat, double *lon)
{
    const struct janas_json *co = janas_json_get(page, "coordinates");
    const struct janas_json *c0 =
        co && co->type == JANAS_JSON_ARRAY ? co->child : NULL;
    *lat = janas_json_num(janas_json_get(c0, "lat"), NAN);
    *lon = janas_json_num(janas_json_get(c0, "lon"), NAN);
}

int wiki_read_hits(const char *text, size_t n, struct wiki_hit *hits, int max,
                   char *err, size_t err_len)
{
    char why[128];
    struct janas_json_doc *d = janas_json_parse(text, n, why, sizeof why);
    if (!d) {
        snprintf(err, err_len, "Wikipedia: an answer it could not read (%s)",
                 why);
        return -1;
    }
    const struct janas_json *root = janas_json_root(d);
    if (api_error(root, err, err_len)) {
        janas_json_free(d);
        return -1;
    }
    const struct janas_json *pages =
        janas_json_get(janas_json_get(root, "query"), "pages");
    int k = 0;
    for (const struct janas_json *p =
             pages && pages->type == JANAS_JSON_ARRAY ? pages->child : NULL;
         p && k < max; p = p->next) {
        struct wiki_hit *h = &hits[k];
        memset(h, 0, sizeof *h);
        copy(h->title, sizeof h->title,
             janas_json_str(janas_json_get(p, "title")));
        if (!h->title[0])
            continue;
        copy(h->desc, sizeof h->desc,
             janas_json_str(janas_json_get(p, "description")));
        const char *ex = janas_json_str(janas_json_get(p, "extract"));
        if (ex) {
            size_t len = strlen(ex);
            size_t cut = wiki_cut(ex, len, sizeof h->text - 1);
            memcpy(h->text, ex, cut);
            h->text[cut] = 0;
        }
        point_of(p, &h->lat, &h->lon);
        h->index = (int)janas_json_num(janas_json_get(p, "index"), k);
        /* in the search's order: the API gives them by page id */
        int i = k++;
        while (i > 0 && hits[i - 1].index > hits[i].index) {
            struct wiki_hit t = hits[i - 1];
            hits[i - 1] = hits[i];
            hits[i] = t;
            i--;
        }
    }
    janas_json_free(d);
    return k;
}

/* The text with one blank line at most between paragraphs, and none made
   of spaces: the plain text leaves two or three around each heading, and
   every byte is the model's context. Its length. */
static size_t squeeze(const char *s, char *to)
{
    size_t n = 0;
    int newlines = 2; /* none at the start */
    for (; *s; s++) {
        if (*s == '\n') {
            while (n && to[n - 1] == ' ')
                n--;
            if (++newlines > 2)
                continue;
        } else if (*s == ' ' && newlines) {
            continue; /* a line's leading spaces */
        } else {
            newlines = 0;
        }
        to[n++] = *s;
    }
    while (n && to[n - 1] == '\n')
        n--;
    to[n] = 0;
    return n;
}

int wiki_read_page(const char *text, size_t n, struct wiki_page *p, char *err,
                   size_t err_len)
{
    memset(p, 0, sizeof *p);
    p->lat = p->lon = NAN;
    char why[128];
    struct janas_json_doc *d = janas_json_parse(text, n, why, sizeof why);
    if (!d) {
        snprintf(err, err_len, "Wikipedia: an answer it could not read (%s)",
                 why);
        return -1;
    }
    const struct janas_json *root = janas_json_root(d);
    if (api_error(root, err, err_len)) {
        janas_json_free(d);
        return -1;
    }
    const struct janas_json *q = janas_json_get(root, "query");
    const struct janas_json *pages = janas_json_get(q, "pages");
    const struct janas_json *pg =
        pages && pages->type == JANAS_JSON_ARRAY ? pages->child : NULL;
    if (!pg) {
        snprintf(err, err_len, "Wikipedia: an answer without the page");
        janas_json_free(d);
        return -1;
    }
    copy(p->title, sizeof p->title,
         janas_json_str(janas_json_get(pg, "title")));
    const struct janas_json *rd = janas_json_get(q, "redirects");
    if (rd && rd->type == JANAS_JSON_ARRAY && rd->child)
        copy(p->from, sizeof p->from,
             janas_json_str(janas_json_get(rd->child, "from")));
    const struct janas_json *inv = janas_json_get(pg, "invalid");
    if (inv) {
        const char *r = janas_json_str(janas_json_get(pg, "invalidreason"));
        snprintf(err, err_len, "Wikipedia: not a page's title (%s)",
                 r ? r : "invalid");
        janas_json_free(d);
        return -1;
    }
    p->missing = janas_json_get(pg, "missing") != NULL;
    p->disamb = janas_json_get(janas_json_get(pg, "pageprops"),
                               "disambiguation") != NULL;
    copy(p->desc, sizeof p->desc,
         janas_json_str(janas_json_get(pg, "description")));
    const char *url = janas_json_str(janas_json_get(pg, "canonicalurl"));
    copy(p->url, sizeof p->url,
         url ? url : janas_json_str(janas_json_get(pg, "fullurl")));
    point_of(pg, &p->lat, &p->lon);
    const char *ex = janas_json_str(janas_json_get(pg, "extract"));
    int ok = 0;
    if (ex) {
        p->text = malloc(strlen(ex) + 1);
        if (p->text)
            p->len = squeeze(ex, p->text);
        ok = p->text != NULL;
    } else {
        ok = 1;
    }
    janas_json_free(d);
    if (!ok)
        snprintf(err, err_len, "out of memory");
    return ok ? 0 : -1;
}

void wiki_page_free(struct wiki_page *p)
{
    free(p->text);
    p->text = NULL;
    p->len = 0;
}

/* ---- the text by its headings ---- */

struct heading {
    size_t at, end; /* the line: its start, and past its newline */
    int level;      /* "== A ==" 2, "=== B ===" 3 */
    const char *name;
    size_t name_n;
};

/* The next heading from from, which is at a line's start. */
static int next_heading(const char *t, size_t len, size_t from,
                        struct heading *h)
{
    for (size_t i = from; i < len;) {
        const char *nl = memchr(t + i, '\n', len - i);
        size_t e = nl ? (size_t)(nl - t) : len;
        size_t lv = 0;
        while (i + lv < e && t[i + lv] == '=')
            lv++;
        if (lv >= 2 && e > i + 2 * lv && t[e - 1] == '=') {
            size_t a = i + lv, b = e;
            while (b > a && t[b - 1] == '=')
                b--;
            while (a < b && t[a] == ' ')
                a++;
            while (b > a && t[b - 1] == ' ')
                b--;
            *h = (struct heading){.at = i,
                                  .end = nl ? e + 1 : e,
                                  .level = (int)lv,
                                  .name = t + a,
                                  .name_n = b - a};
            return 1;
        }
        i = nl ? e + 1 : len;
    }
    return 0;
}

static size_t trim_end(const char *t, size_t n)
{
    while (n && (t[n - 1] == '\n' || t[n - 1] == ' '))
        n--;
    return n;
}

size_t wiki_intro_len(const char *text, size_t len)
{
    struct heading h;
    return trim_end(text, next_heading(text, len, 0, &h) ? h.at : len);
}

int wiki_section(const char *text, size_t len, const char *name, size_t *start,
                 size_t *n)
{
    while (*name == ' ')
        name++;
    size_t want = strlen(name);
    while (want && name[want - 1] == ' ')
        want--;
    if (!want)
        return -1;
    struct heading h, best = {0};
    int best_rank = 0;
    for (size_t at = 0; next_heading(text, len, at, &h); at = h.end) {
        char hn[256];
        snprintf(hn, sizeof hn, "%.*s", (int)h.name_n, h.name);
        char nm[256];
        snprintf(nm, sizeof nm, "%.*s", (int)want, name);
        int rank = strcasecmp(hn, nm) == 0                ? 3
                   : strncasecmp(hn, nm, strlen(nm)) == 0 ? 2
                   : strcasestr(hn, nm)                   ? 1
                                                          : 0;
        if (rank > best_rank) {
            best = h;
            best_rank = rank;
            if (rank == 3)
                break;
        }
    }
    if (!best_rank)
        return -1;
    size_t end = len;
    for (size_t at = best.end; next_heading(text, len, at, &h); at = h.end)
        if (h.level <= best.level) {
            end = h.at;
            break;
        }
    *start = best.at;
    *n = trim_end(text + best.at, end - best.at);
    return 0;
}

void wiki_section_list(const char *text, size_t len, struct janas_buf *out)
{
    struct heading h;
    int top = 0, open = 0, subs = 0;
    for (size_t at = 0; next_heading(text, len, at, &h); at = h.end) {
        if (!top || h.level < top)
            top = h.level; /* the shallowest seen: 2 as a rule */
        if (h.level == top) {
            if (open)
                janas_buf_puts(out, subs ? "); " : "; ");
            janas_buf_put(out, h.name, h.name_n);
            open = 1;
            subs = 0;
        } else if (h.level == top + 1 && open) {
            janas_buf_puts(out, subs ? ", " : " (");
            janas_buf_put(out, h.name, h.name_n);
            subs++;
        }
    }
    if (open && subs)
        janas_buf_puts(out, ")");
}

size_t wiki_drop_heads(const char *t, size_t n)
{
    for (;;) {
        n = trim_end(t, n);
        size_t a = n;
        while (a && t[a - 1] != '\n')
            a--;
        struct heading h;
        if (a == n || !next_heading(t, n, a, &h) || h.at != a)
            return n;
        n = a;
    }
}

size_t wiki_cut(const char *s, size_t n, size_t max)
{
    if (n <= max)
        return n;
    size_t lo = max / 2;
    for (size_t i = max; i > lo; i--)
        if (s[i - 1] == '\n')
            return trim_end(s, i);
    for (size_t i = max; i > lo; i--)
        if (s[i - 1] == '.' && s[i] == ' ')
            return i;
    for (size_t i = max; i > lo; i--)
        if (s[i] == ' ')
            return i;
    size_t i = max;
    while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80)
        i--; /* not inside a character */
    return i;
}
