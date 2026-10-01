/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * wiki.h - janas-wiki, Wikipedia as an MCP service: what its parts share.
 * fetch.c asks Wikipedia's API (answers kept a while), read.c reads its
 * answers and cuts a page into its sections, tools.c offers the tools.
 */
#ifndef JANAS_WIKI_H
#define JANAS_WIKI_H

#include <stddef.h>

#include "llm/json.h"

/* ---- fetch.c ---- */

/* GET of url, the body into out, with the User-Agent Wikimedia asks for;
   answers of status 200 are kept keep_s seconds and given again meanwhile.
   The HTTP status, or -1 with the reason in err. */
int wiki_fetch(const char *url, int keep_s, struct janas_buf *out, char *err,
               size_t err_len);
void wiki_fetch_free(void);

/* ---- read.c ---- */

#define WIKI_HITS 30

/* A page as a search or a list gives it */
struct wiki_hit {
    char title[256];
    char desc[256];  /* Wikidata's short description; "" when none */
    char text[1200]; /* its first sentences; "" when not asked */
    double lat, lon; /* its primary coordinates; NAN when none */
    double km;       /* from the point asked about (wiki_nearby) */
    int index;       /* its rank in the search, or in the list */
};

/* The pages of an answer of action=query with a generator (search,
   geosearch), in their order. How many, or -1 with the reason in err (an
   error of the API, an answer not read). */
int wiki_read_hits(const char *text, size_t n, struct wiki_hit *hits, int max,
                   char *err, size_t err_len);

struct wiki_page {
    char title[256];
    char from[256]; /* the title asked for, when it was a redirect */
    char desc[256];
    char url[512];
    double lat, lon;
    int missing; /* there is no page of that title */
    int disamb;  /* a disambiguation page */
    char *text;  /* the plain text, "== Heading ==" lines; malloc'd */
    size_t len;
};

/* A page from an answer of action=query&titles=... (extracts,
   coordinates, description, info, pageprops). 0, or -1 with the reason in
   err; free with wiki_page_free. */
int wiki_read_page(const char *text, size_t n, struct wiki_page *p, char *err,
                   size_t err_len);
void wiki_page_free(struct wiki_page *p);

/* The page's text before its first heading. */
size_t wiki_intro_len(const char *text, size_t len);

/* The section whose heading is name (its name, case aside; else the first
   heading that begins with it, else holds it), with its subsections: its
   start (at its heading) and length. 0, or -1 when there is none. */
int wiki_section(const char *text, size_t len, const char *name, size_t *start,
                 size_t *n);

/* The page's sections as "A (A1, A2); B; ...": level 2 headings, those of
   level 3 in brackets. */
void wiki_section_list(const char *text, size_t len, struct janas_buf *out);

/* n less the headings (and blank lines) the text ends with: a cut text
   does not end on a heading whose text is left out. */
size_t wiki_drop_heads(const char *text, size_t n);

/* At most max bytes of s, cut at the end of a paragraph or else of a
   sentence or a word, and never inside a UTF-8 character. */
size_t wiki_cut(const char *s, size_t n, size_t max);

/* ---- tools.c ---- */

void wiki_tools_list(void *ctx, struct janas_buf *b);
int wiki_tools_call(void *ctx, const struct janas_json *params,
                    struct janas_buf *b);

#endif
