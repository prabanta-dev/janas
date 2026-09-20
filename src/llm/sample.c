/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sample.c - choosing a token from a row of logits (see sample.h).
 */
#include "sample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct cand {
    float logit;
    int32_t id;
};

struct janas_sampler {
    uint32_t nv;
    struct janas_sample_params p;
    uint64_t rng;
    struct cand *cands; /* nv of them */
    float *work;        /* the row with bias and penalties, when there are */
    /* logit_bias */
    int32_t *bias_id;
    float *bias;
    uint32_t n_bias;
    /* the penalties: how often each token was picked since the mark, and
       which ones, to clear them */
    uint32_t *count;
    int32_t *seen;
    uint32_t n_seen;
    struct janas_sample_filter filter;
    int has_filter;
    uint32_t forced;
};

struct janas_sampler *janas_sampler_create(uint32_t nv)
{
    struct janas_sampler *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->nv = nv;
    s->cands = malloc(nv * sizeof(struct cand));
    s->count = calloc(nv, sizeof(uint32_t));
    s->seen = malloc(nv * sizeof(int32_t));
    if (!s->cands || !s->count || !s->seen) {
        janas_sampler_destroy(s);
        return NULL;
    }
    s->p.logprobs = -1;
    return s;
}

void janas_sampler_destroy(struct janas_sampler *s)
{
    if (!s)
        return;
    free(s->cands);
    free(s->work);
    free(s->bias_id);
    free(s->bias);
    free(s->count);
    free(s->seen);
    free(s);
}

void janas_sampler_set(struct janas_sampler *s,
                       const struct janas_sample_params *p, uint64_t seed)
{
    s->p = *p;
    if (s->p.logprobs > JANAS_SAMPLE_MAX_TOP)
        s->p.logprobs = JANAS_SAMPLE_MAX_TOP;
    s->rng = seed;
}

int janas_sampler_bias(struct janas_sampler *s, const int32_t *ids,
                       const float *bias, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        if (ids[i] < 0 || (uint32_t)ids[i] >= s->nv)
            return -1;
    int32_t *bi = n ? malloc(n * sizeof(int32_t)) : NULL;
    float *bv = n ? malloc(n * sizeof(float)) : NULL;
    if (n && (!bi || !bv)) {
        free(bi);
        free(bv);
        return -1;
    }
    if (n) {
        memcpy(bi, ids, n * sizeof(int32_t));
        memcpy(bv, bias, n * sizeof(float));
    }
    free(s->bias_id);
    free(s->bias);
    s->bias_id = bi;
    s->bias = bv;
    s->n_bias = n;
    return 0;
}

void janas_sampler_filter(struct janas_sampler *s,
                          const struct janas_sample_filter *f)
{
    s->has_filter = f != NULL;
    if (f)
        s->filter = *f;
}

void janas_sampler_mark(struct janas_sampler *s)
{
    for (uint32_t i = 0; i < s->n_seen; i++)
        s->count[s->seen[i]] = 0;
    s->n_seen = 0;
}

uint32_t janas_sampler_forced(const struct janas_sampler *s)
{
    return s->forced;
}

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static int32_t argmax(const float *v, uint32_t n)
{
    uint32_t b = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[b])
            b = i;
    return (int32_t)b;
}

static int cand_cmp(const void *pa, const void *pb)
{
    const struct cand *a = pa, *b = pb;
    if (a->logit != b->logit)
        return a->logit > b->logit ? -1 : 1;
    return a->id < b->id ? -1 : 1;
}

/*
 * The k largest logits, largest first. A heap of k holds the best so far, so
 * the vocabulary is read once and almost every token fails a single compare
 * against the smallest of them. Collecting every logit within thirty times
 * the temperature of the maximum and sorting those cost 3.6 ms a token on a
 * vocabulary of 151936 - it kept 34000 candidates to use twenty, on one
 * thread while the other nineteen waited.
 */
static uint32_t top_k_cands(struct cand *h, uint32_t k, const float *lg,
                            uint32_t nv, float floor)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < nv; i++) {
        if (lg[i] < floor) /* the same ones the old code left out */
            continue;
        if (n == k) {
            if (lg[i] <= h[0].logit)
                continue;
            h[0] = (struct cand){lg[i], (int32_t)i};
            uint32_t p = 0; /* sift down in the heap of smallest at the root */
            for (;;) {
                uint32_t l = 2 * p + 1, r = l + 1, m = p;
                if (l < k && h[l].logit < h[m].logit)
                    m = l;
                if (r < k && h[r].logit < h[m].logit)
                    m = r;
                if (m == p)
                    break;
                struct cand t = h[p];
                h[p] = h[m];
                h[m] = t;
                p = m;
            }
            continue;
        }
        h[n] = (struct cand){lg[i], (int32_t)i};
        uint32_t c = n++;
        while (c && h[(c - 1) / 2].logit > h[c].logit) { /* sift up */
            struct cand t = h[c];
            h[c] = h[(c - 1) / 2];
            h[(c - 1) / 2] = t;
            c = (c - 1) / 2;
        }
    }
    qsort(h, n, sizeof(struct cand), cand_cmp);
    return n;
}

/* Everything at or above floor, sorted; with the filter on, only what it
   allows. */
static uint32_t all_cands(struct janas_sampler *s, const float *lg, float floor,
                          int filtered)
{
    uint32_t nc = 0;
    for (uint32_t i = 0; i < s->nv; i++)
        if (lg[i] >= floor &&
            (!filtered || s->filter.allows(s->filter.ctx, (int32_t)i)))
            s->cands[nc++] = (struct cand){lg[i], (int32_t)i};
    qsort(s->cands, nc, sizeof(struct cand), cand_cmp);
    return nc;
}

/*
 * The k best candidates the filter allows. A grammar allows most of what the
 * model finds likely anyway, so the best few are tried first; the whole
 * vocabulary is read only when they do not give k.
 */
static uint32_t filtered_cands(struct janas_sampler *s, const float *lg,
                               uint32_t k, float floor)
{
    if (k >= s->nv)
        return all_cands(s, lg, floor, 1);
    uint32_t m = k < 16 ? 64 : 4 * k;
    if (m > s->nv)
        m = s->nv;
    uint32_t nc = top_k_cands(s->cands, m, lg, s->nv, floor), got = 0;
    for (uint32_t i = 0; i < nc && got < k; i++)
        if (s->filter.allows(s->filter.ctx, s->cands[i].id))
            s->cands[got++] = s->cands[i];
    if (got == k || nc < m)
        return got; /* enough, or nothing else above the floor */
    nc = all_cands(s, lg, floor, 1);
    return nc < k ? nc : k;
}

/* The row with the bias and the penalties, or the row itself without. */
static const float *adjusted(struct janas_sampler *s, const float *lg)
{
    int pen = (s->p.presence != 0 || s->p.frequency != 0) && s->n_seen;
    if (!s->n_bias && !pen)
        return lg;
    if (!s->work && !(s->work = malloc(s->nv * sizeof(float))))
        return lg;
    memcpy(s->work, lg, s->nv * sizeof(float));
    for (uint32_t i = 0; i < s->n_bias; i++)
        s->work[s->bias_id[i]] += s->bias[i];
    for (uint32_t i = 0; pen && i < s->n_seen; i++) {
        int32_t t = s->seen[i];
        s->work[t] -= s->p.frequency * (float)s->count[t] + s->p.presence;
    }
    return s->work;
}

/* The log-probability of t and the best alternatives, from the raw row. */
static void learn(struct janas_sampler *s, const float *lg, int32_t t,
                  struct janas_token_lp *lp)
{
    float mx = lg[argmax(lg, s->nv)];
    double sum = 0;
    for (uint32_t i = 0; i < s->nv; i++)
        sum += exp((double)(lg[i] - mx));
    float lse = mx + (float)log(sum);
    lp->logprob = lg[t] - lse;
    lp->n_top = 0;
    if (s->p.logprobs > 0) {
        uint32_t n = top_k_cands(s->cands, (uint32_t)s->p.logprobs, lg, s->nv,
                                 -INFINITY);
        for (uint32_t i = 0; i < n; i++) {
            lp->top_id[i] = s->cands[i].id;
            lp->top_lp[i] = s->cands[i].logit - lse;
        }
        lp->n_top = (int32_t)n;
    }
}

static int32_t draw(struct janas_sampler *s, const float *lg, int filtered)
{
    const struct janas_sample_params *o = &s->p;
    if (o->temperature <= 0) {
        if (!filtered)
            return argmax(lg, s->nv);
        uint32_t n = filtered_cands(s, lg, 1, -INFINITY);
        return n ? s->cands[0].id : -1;
    }
    float mx = lg[argmax(lg, s->nv)];
    float floor = mx - 30.0f * o->temperature;
    uint32_t nc;
    uint32_t k =
        o->top_k > 0 && (uint32_t)o->top_k < s->nv ? (uint32_t)o->top_k : s->nv;
    if (filtered)
        nc = filtered_cands(s, lg, k, floor);
    else if (k < s->nv)
        nc = top_k_cands(s->cands, k, lg, s->nv, floor);
    else /* no limit on the count: everything that could be drawn at all */
        nc = all_cands(s, lg, floor, 0);
    if (nc == 0 && filtered) /* nothing allowed above the floor: below it */
        nc = filtered_cands(s, lg, k, -INFINITY);
    if (nc == 0)
        return -1;
    /* the probabilities, relative to the best candidate left (with no filter
       that is the maximum itself, as it always was) */
    float top = s->cands[0].logit;
    double sum = 0;
    for (uint32_t i = 0; i < nc; i++) {
        /* the probability, unnormalized, in place of the logit */
        s->cands[i].logit = expf((s->cands[i].logit - top) / o->temperature);
        sum += s->cands[i].logit;
    }
    if (o->top_p > 0 && o->top_p < 1) {
        double c = 0;
        uint32_t kk = 0;
        while (kk < nc && c < o->top_p * sum)
            c += s->cands[kk++].logit;
        nc = kk;
        sum = c;
    }
    if (o->min_p > 0) {
        uint32_t kk = 1;
        while (kk < nc && s->cands[kk].logit >= o->min_p * s->cands[0].logit)
            kk++;
        for (uint32_t i = kk; i < nc; i++)
            sum -= s->cands[i].logit;
        nc = kk;
    }
    if (nc <= 1)
        s->forced++; /* the draw had no choice to make */
    double u = (double)(splitmix64(&s->rng) >> 11) * 0x1.0p-53 * sum;
    for (uint32_t i = 0; i + 1 < nc; i++) {
        u -= s->cands[i].logit;
        if (u < 0)
            return s->cands[i].id;
    }
    return s->cands[nc - 1].id;
}

int32_t janas_sampler_pick(struct janas_sampler *s, const float *logits,
                           struct janas_token_lp *lp)
{
    int filtered = s->has_filter && s->filter.active(s->filter.ctx);
    int32_t t = draw(s, adjusted(s, logits), filtered);
    if (t < 0)
        return -1;
    if (lp && s->p.logprobs >= 0)
        learn(s, logits, t, lp);
    if (s->count[t]++ == 0)
        s->seen[s->n_seen++] = t;
    if (s->has_filter)
        s->filter.accept(s->filter.ctx, t);
    return t;
}
