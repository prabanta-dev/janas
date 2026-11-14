/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * search.c - the pieces that answer a query: by its words (BM25, with the
 * rarity of each word in the whole index) and, when the index has vectors,
 * by its meaning (the cosine of the query's vector with each piece's). The
 * two rankings are joined by their ranks (reciprocal rank fusion), not by
 * their scores, which do not measure the same thing: a piece first by one
 * and tenth by the other comes before one fifth by both. Words find the
 * names, codes and numbers a vector blurs; the meaning finds what is said
 * with other words.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "docs.h"
#include "services/common/private.h"

#define MAX_TERMS 32
#define RRF_K 60.0

struct cand {
    size_t file, chunk;
    double words, meaning;
    double rw, rm; /* ranks, from 1; 0 when not ranked */
};

static int by_words(const void *a, const void *b)
{
    const struct cand *x = a, *y = b;
    return (x->words < y->words) - (x->words > y->words);
}

static int by_meaning(const void *a, const void *b)
{
    const struct cand *x = a, *y = b;
    return (x->meaning < y->meaning) - (x->meaning > y->meaning);
}

static double fused(const struct cand *c)
{
    return (c->rw ? 1 / (RRF_K + c->rw) : 0) +
           (c->rm ? 1 / (RRF_K + c->rm) : 0);
}

static int by_fused(const void *a, const void *b)
{
    double x = fused(a), y = fused(b);
    return (x < y) - (x > y);
}

long dz_search(struct dz_index *x, const char *query, const float *qvec,
               int qdim, const char *folder, int max, struct dz_hit **hits)
{
    *hits = NULL;
    /* the query's words, each once */
    char terms[MAX_TERMS][64];
    int nt = 0;
    const char *q = query, *qe = query + strlen(query);
    char w[64];
    while (nt < MAX_TERMS && dz_word(&q, qe, w, sizeof w)) {
        int digits = w[0] >= '0' && w[0] <= '9';
        if ((strlen(w) < 3 && !digits) || dz_stopword(w))
            continue;
        int seen = 0;
        for (int i = 0; i < nt && !seen; i++)
            seen = !strcmp(terms[i], w);
        if (!seen)
            snprintf(terms[nt++], sizeof terms[0], "%s", w);
    }
    int use_vec = qvec && qdim && qdim == x->dim;
    size_t total = 0;
    for (size_t i = 0; i < x->n; i++)
        if (!folder || janas_path_under(x->f[i].path, folder))
            total += x->f[i].n_chunk;
    if (!total)
        return 0;
    struct cand *c = calloc(total, sizeof(*c));
    float *tf = nt ? calloc(total * (size_t)nt, sizeof(float)) : NULL;
    float *len = calloc(total, sizeof(float));
    if (!c || (nt && !tf) || !len) {
        free(c);
        free(tf);
        free(len);
        return -1;
    }
    double df[MAX_TERMS] = {0}, sum_len = 0;
    size_t k = 0;
    for (size_t i = 0; i < x->n; i++) {
        if (folder && !janas_path_under(x->f[i].path, folder))
            continue;
        for (uint32_t j = 0; j < x->f[i].n_chunk; j++, k++) {
            const struct dz_chunk *ch = &x->f[i].c[j];
            c[k].file = i;
            c[k].chunk = j;
            const char *p = ch->text, *e = ch->text + ch->n;
            float *t = tf + k * (size_t)nt;
            while (dz_word(&p, e, w, sizeof w)) {
                len[k]++;
                for (int m = 0; m < nt; m++)
                    if (!strcmp(w, terms[m]))
                        t[m]++;
            }
            for (int m = 0; m < nt; m++)
                df[m] += t[m] > 0;
            sum_len += len[k];
            if (use_vec && ch->vec) {
                double d = 0;
                for (int m = 0; m < qdim; m++)
                    d += (double)qvec[m] * ch->vec[m];
                c[k].meaning = d;
            } else {
                c[k].meaning = -2; /* below any cosine: not ranked */
            }
        }
    }
    double avg = sum_len / (double)total, k1 = 1.2, b = 0.75;
    for (k = 0; k < total; k++) {
        double s = 0;
        for (int m = 0; m < nt; m++) {
            double t = tf[k * (size_t)nt + m];
            if (!t)
                continue;
            double idf = log(1 + ((double)total - df[m] + 0.5) / (df[m] + 0.5));
            s += idf * t * (k1 + 1) /
                 (t + k1 * (1 - b + b * len[k] / (avg > 0 ? avg : 1)));
        }
        c[k].words = s;
    }
    free(tf);
    free(len);
    qsort(c, total, sizeof(*c), by_words);
    for (k = 0; k < total && c[k].words > 0; k++)
        c[k].rw = (double)(k + 1);
    if (use_vec) {
        qsort(c, total, sizeof(*c), by_meaning);
        for (k = 0; k < total && c[k].meaning > -2; k++)
            c[k].rm = (double)(k + 1);
    }
    qsort(c, total, sizeof(*c), by_fused);
    size_t n = 0;
    while (n < total && n < (size_t)max && (c[n].rw || c[n].rm))
        n++;
    struct dz_hit *h = n ? calloc(n, sizeof(*h)) : NULL;
    if (n && !h) {
        free(c);
        return -1;
    }
    for (k = 0; k < n; k++)
        h[k] = (struct dz_hit){.file = c[k].file,
                               .chunk = c[k].chunk,
                               .words = c[k].words,
                               .meaning = c[k].rm ? c[k].meaning : 0,
                               .score = fused(&c[k])};
    free(c);
    *hits = h;
    return (long)n;
}
