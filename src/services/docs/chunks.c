/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * chunks.c - a document's text cut into pieces, and the words of a text
 * for the search by words (BM25).
 */
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "docs.h"

static int space(char c)
{
    return c == ' ' || c == '\n' || c == '\t' || c == '\r';
}

long dz_cut(const char *t, size_t n, size_t size, size_t overlap,
            struct dz_piece **out)
{
    *out = NULL;
    size_t cap = 0, k = 0, at = 0;
    while (at < n && space(t[at]))
        at++;
    while (at < n) {
        size_t end = at + size < n ? at + size : n;
        if (end < n) { /* back to a paragraph or a space, not past half */
            size_t e = end;
            while (e > at + size / 2 && !(t[e] == '\n' && t[e - 1] == '\n'))
                e--;
            if (e <= at + size / 2)
                for (e = end; e > at + size / 2 && !space(t[e]);)
                    e--;
            if (e > at + size / 2)
                end = e;
            while (end < n && ((unsigned char)t[end] & 0xc0) == 0x80)
                end++;
        }
        if (k == cap) {
            cap = cap ? 2 * cap : 16;
            struct dz_piece *p = realloc(*out, cap * sizeof(*p));
            if (!p) {
                free(*out);
                *out = NULL;
                return -1;
            }
            *out = p;
        }
        size_t e = end;
        while (e > at && space(t[e - 1]))
            e--;
        (*out)[k++] = (struct dz_piece){.at = at, .n = e - at};
        if (end >= n)
            break;
        size_t next = end > at + overlap ? end - overlap : end;
        while (next < end && !space(t[next]))
            next++; /* begin at a word */
        while (next < end && space(t[next]))
            next++;
        if (next >= end)
            for (next = end; next < n && space(t[next]);)
                next++;
        at = next;
    }
    return (long)k;
}

int dz_word(const char **p, const char *end, char *w, size_t cap)
{
    const unsigned char *s = (const unsigned char *)*p,
                        *e = (const unsigned char *)end;
    while (s < e && !(isalnum(*s) || *s >= 0x80))
        s++;
    if (s >= e) {
        *p = (const char *)s;
        return 0;
    }
    size_t n = 0;
    while (s < e && (isalnum(*s) || *s >= 0x80)) {
        if (n + 1 < cap)
            w[n++] = (char)tolower(*s);
        s++;
    }
    w[n] = 0;
    *p = (const char *)s;
    return 1;
}

/* Words too common to tell pieces apart, English and Italian. */
int dz_stopword(const char *w)
{
    static const char *const S[] = {
        "the",    "and",   "for",   "with",   "how",    "what",   "who",
        "which",  "that",  "this",  "are",    "was",    "were",   "from",
        "into",   "about", "does",  "can",    "you",    "your",   "its",
        "his",    "her",   "their", "not",    "but",    "have",   "has",
        "had",    "will",  "would", "should", "could",  "there",  "where",
        "when",   "why",   "all",   "any",    "some",   "one",    "two",
        "also",   "per",   "che",   "chi",    "con",    "come",   "dove",
        "quando", "del",   "della", "dei",    "delle",  "degli",  "nel",
        "nella",  "una",   "uno",   "sono",   "non",    "gli",    "anche",
        "dal",    "dalla", "alla",  "allo",   "agli",   "alle",   "sul",
        "sulla",  "cosa",  "quale", "quali",  "questo", "questa", "quello",
        "quella"};
    for (size_t i = 0; i < sizeof(S) / sizeof(S[0]); i++)
        if (!strcmp(w, S[i]))
            return 1;
    return 0;
}
