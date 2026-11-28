/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sample.c - choosing a token from a row of logits (see sample.h).
 */
#include "sample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "jmath.h"

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
    /* the tokens picked since the mark, in order: the penalties' window,
       DRY's runs, and what a rewind takes back */
    int32_t *hist;
    uint32_t n_hist, cap_hist;
    uint32_t *wcount; /* a window's counts, while they are applied */
    const uint8_t *breaker;
    float *prob; /* the candidates' probabilities, unnormalized */
    struct cand *alt;
    /* mirostat's mu, and its value before each of the last draws */
    float mu;
    float mu_ring[64];
    uint32_t n_draws;
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
    free(s->hist);
    free(s->wcount);
    free(s->prob);
    free(s->alt);
    free(s);
}

void janas_sampler_set(struct janas_sampler *s,
                       const struct janas_sample_params *p, uint64_t seed)
{
    s->p = *p;
    if (s->p.logprobs > JANAS_SAMPLE_MAX_TOP)
        s->p.logprobs = JANAS_SAMPLE_MAX_TOP;
    if (s->p.more.n_stage < 0 || s->p.more.n_stage > JANAS_SAMPLE_MAX_STAGES)
        s->p.more.n_stage = 0;
    s->rng = seed;
    s->mu = 2 * s->p.more.mirostat_tau;
    s->n_draws = 0;
}

void janas_sampler_breakers(struct janas_sampler *s, const uint8_t *is)
{
    s->breaker = is;
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
    s->n_hist = 0;
    s->mu = 2 * s->p.more.mirostat_tau;
    s->n_draws = 0;
}

#define SPLITMIX_STEP 0x9e3779b97f4a7c15ull /* splitmix64 below */

/* A draw takes one number, and only above temperature 0 (draw() below):
   the sequence is a counter, so going back is a subtraction. */
void janas_sampler_rewind(struct janas_sampler *s, uint32_t n)
{
    if (s->p.temperature > 0)
        s->rng -= (uint64_t)n * SPLITMIX_STEP;
    /* the tokens drawn go out of the counts, mirostat's mu back */
    for (uint32_t i = 0; i < n && s->n_hist; i++) {
        int32_t t = s->hist[--s->n_hist];
        if (s->count[t] && --s->count[t] == 0)
            for (uint32_t k = 0; k < s->n_seen; k++)
                if (s->seen[k] == t) {
                    s->seen[k] = s->seen[--s->n_seen];
                    break;
                }
    }
    uint32_t ring = sizeof s->mu_ring / sizeof *s->mu_ring;
    if (n && n <= s->n_draws && n <= ring) {
        s->n_draws -= n;
        s->mu = s->mu_ring[s->n_draws % ring];
    }
}

uint32_t janas_sampler_forced(const struct janas_sampler *s)
{
    return s->forced;
}

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += SPLITMIX_STEP);
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

/* DRY: each token that would go on repeating a run of the history longer
   than allowed loses multiplier x base^(length - allowed). A run is the
   longest one ending at the history's last token that also ends just
   before an earlier place, the token there the one it would repeat; it
   does not cross a breaker. */
#define DRY_WINDOW 8192 /* tokens looked back at most */
#define DRY_RUN 256     /* the longest run measured */

static void dry(struct janas_sampler *s, float *row)
{
    const struct janas_sample_more *m = &s->p.more;
    uint32_t n = s->n_hist;
    uint32_t w = m->dry_last_n > 0 ? (uint32_t)m->dry_last_n : DRY_WINDOW;
    if (w > DRY_WINDOW)
        w = DRY_WINDOW;
    uint32_t start = n > w ? n - w : 0;
    const int32_t *h = s->hist;
    const uint8_t *brk = s->breaker;
    if (n - start < 2 || (brk && brk[h[n - 1]]))
        return;
    int32_t allowed = m->dry_allowed > 0 ? m->dry_allowed : 0;
    /* each token's largest penalty in prob[], wcount[] marking it taken */
    for (uint32_t j = start + 1; j < n; j++) {
        uint32_t len = 0;
        while (len < DRY_RUN && j >= start + 1 + len) {
            int32_t a = h[j - 1 - len], b = h[n - 1 - len];
            if (a != b || (brk && brk[a]))
                break;
            len++;
        }
        if (!len || (int32_t)len < allowed)
            continue;
        float pen = m->dry_multiplier *
                    powf(m->dry_base, (float)((int32_t)len - allowed));
        int32_t t = h[j];
        if (!s->wcount[t] || pen > s->prob[t]) {
            s->prob[t] = pen;
            s->wcount[t] = 1;
        }
    }
    for (uint32_t j = start + 1; j < n; j++) {
        int32_t t = h[j];
        if (s->wcount[t]) {
            row[t] -= s->prob[t];
            s->wcount[t] = 0;
        }
    }
}

static int need_scratch(struct janas_sampler *s)
{
    if (!s->wcount && !(s->wcount = calloc(s->nv, sizeof(uint32_t))))
        return -1;
    if (!s->prob && !(s->prob = malloc(s->nv * sizeof(float))))
        return -1;
    if (!s->alt && !(s->alt = malloc(s->nv * sizeof(struct cand))))
        return -1;
    return 0;
}

/* The row with the bias and the penalties, or the row itself without. */
static const float *adjusted(struct janas_sampler *s, const float *lg)
{
    const struct janas_sample_more *m = &s->p.more;
    int rep = m->repeat != 0 && m->repeat != 1;
    int pen = (s->p.presence != 0 || s->p.frequency != 0 || rep) && s->n_seen;
    int dr = m->dry_multiplier > 0 && s->n_hist >= 2;
    if (!s->n_bias && !pen && !dr)
        return lg;
    if (!s->work && !(s->work = malloc(s->nv * sizeof(float))))
        return lg;
    if ((dr || (pen && m->last_n > 0)) && need_scratch(s) != 0)
        return lg;
    memcpy(s->work, lg, s->nv * sizeof(float));
    for (uint32_t i = 0; i < s->n_bias; i++)
        s->work[s->bias_id[i]] += s->bias[i];
    if (pen && m->last_n > 0) {
        /* the last last_n tokens alone */
        uint32_t from = s->n_hist > (uint32_t)m->last_n
                            ? s->n_hist - (uint32_t)m->last_n
                            : 0;
        for (uint32_t i = from; i < s->n_hist; i++)
            s->wcount[s->hist[i]]++;
        for (uint32_t i = from; i < s->n_hist; i++) {
            int32_t t = s->hist[i];
            uint32_t c = s->wcount[t];
            if (!c)
                continue; /* done already */
            float *v = &s->work[t];
            if (rep)
                *v = *v > 0 ? *v / m->repeat : *v * m->repeat;
            *v -= s->p.frequency * (float)c + s->p.presence;
            s->wcount[t] = 0;
        }
    } else
        for (uint32_t i = 0; pen && i < s->n_seen; i++) {
            int32_t t = s->seen[i];
            float *v = &s->work[t];
            if (rep)
                *v = *v > 0 ? *v / m->repeat : *v * m->repeat;
            *v -= s->p.frequency * (float)s->count[t] + s->p.presence;
        }
    if (dr)
        dry(s, s->work);
    return s->work;
}

/* The log-probability of t and the best alternatives, from the raw row. */
static void learn(struct janas_sampler *s, const float *lg, int32_t t,
                  struct janas_token_lp *lp)
{
    float mx = lg[argmax(lg, s->nv)];
    double sum = 0;
    for (uint32_t i = 0; i < s->nv; i++)
        sum += janas_exp((double)(lg[i] - mx));
    float lse = mx + (float)janas_log(sum);
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

/* The plain draw, as it always was. */
static int32_t draw_plain(struct janas_sampler *s, const float *lg,
                          int filtered)
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
        s->cands[i].logit =
            janas_expf((s->cands[i].logit - top) / o->temperature);
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

/* Janas's own order of llama.cpp's stages, for those asked without one */
static const uint8_t OWN_ORDER[] = {
    JANAS_STAGE_TOP_N_SIGMA, JANAS_STAGE_TOP_K, JANAS_STAGE_TYPICAL,
    JANAS_STAGE_TEMPERATURE, JANAS_STAGE_TOP_P, JANAS_STAGE_MIN_P,
    JANAS_STAGE_XTC};

/* whether a stage does anything with these settings */
static int stage_on(const struct janas_sample_params *o, int st)
{
    const struct janas_sample_more *m = &o->more;
    switch (st) {
    case JANAS_STAGE_TOP_N_SIGMA:
        return m->top_n_sigma > 0;
    case JANAS_STAGE_TOP_K:
        return o->top_k > 0;
    case JANAS_STAGE_TYPICAL:
        return m->typical_p > 0 && m->typical_p < 1;
    case JANAS_STAGE_TOP_P:
        return o->top_p > 0 && o->top_p < 1;
    case JANAS_STAGE_MIN_P:
        return o->min_p > 0;
    case JANAS_STAGE_XTC:
        return m->xtc_probability > 0 && m->xtc_threshold > 0 &&
               m->xtc_threshold <= 0.5f;
    case JANAS_STAGE_TEMPERATURE:
        return 1;
    default: /* penalties and dry: on the row, before */
        return 0;
    }
}

/* whether anything beyond the plain draw is asked */
static int chain_asked(const struct janas_sample_params *o)
{
    const struct janas_sample_more *m = &o->more;
    return m->n_stage > 0 || m->top_n_sigma > 0 ||
           (m->typical_p > 0 && m->typical_p < 1) ||
           stage_on(o, JANAS_STAGE_XTC) || m->mirostat || m->dynatemp_range > 0;
}

/* the candidates' probabilities at 1/inv, unnormalized, and their sum */
static double probs(struct janas_sampler *s, uint32_t nc, float inv)
{
    double sum = 0;
    float top = s->cands[0].logit;
    for (uint32_t i = 0; i < nc; i++)
        sum += s->prob[i] = janas_expf((s->cands[i].logit - top) * inv);
    return sum;
}

/* a number in [0, 1) for this draw besides its own, without taking one
   from the sequence (XTC's coin) */
static double aside(const struct janas_sampler *s)
{
    uint64_t x = s->rng ^ 0x5851f42d4c957f2dull;
    return (double)(splitmix64(&x) >> 11) * 0x1.0p-53;
}

static int32_t pick_from(struct janas_sampler *s, uint32_t nc, float inv)
{
    if (nc <= 1)
        s->forced++;
    double sum = probs(s, nc, inv);
    double u = (double)(splitmix64(&s->rng) >> 11) * 0x1.0p-53 * sum;
    for (uint32_t i = 0; i + 1 < nc; i++) {
        u -= s->prob[i];
        if (u < 0)
            return s->cands[i].id;
    }
    return s->cands[nc - 1].id;
}

/* Mirostat, 1 or 2 (llama.cpp's): the candidates at the temperature, cut
   to the surprise mu allows, mu moved towards tau by what was drawn. */
static int32_t mirostat(struct janas_sampler *s, uint32_t nc, float inv)
{
    const struct janas_sample_more *m = &s->p.more;
    double sum = probs(s, nc, inv);
    uint32_t keep = nc;
    if (m->mirostat == 1) {
        /* Zipf's exponent from the top 100, then k from it */
        double sti_b = 0, sti2 = 0;
        for (uint32_t i = 0; i + 1 < nc && i < 99; i++) {
            double t = log((double)(i + 2) / (double)(i + 1));
            double b = log(s->prob[i] / s->prob[i + 1]);
            sti_b += t * b;
            sti2 += t * t;
        }
        double sh = sti2 > 0 ? sti_b / sti2 : 1, eps = sh - 1;
        double k =
            eps != 0
                ? pow(eps * pow(2.0, s->mu) / (1 - pow((double)s->nv, -eps)),
                      1 / sh)
                : (double)nc;
        keep = k < 1 ? 1 : k < nc ? (uint32_t)k : nc;
    } else {
        keep = 0;
        while (keep < nc && -log2(s->prob[keep] / sum) <= s->mu)
            keep++;
        if (!keep)
            keep = 1;
    }
    int32_t t = pick_from(s, keep, inv);
    double all = 0, pt = 0;
    for (uint32_t i = 0; i < keep; i++) {
        all += s->prob[i];
        if (s->cands[i].id == t)
            pt = s->prob[i];
    }
    if (pt > 0)
        s->mu -= m->mirostat_eta * (float)(-log2(pt / all) - m->mirostat_tau);
    return t;
}

static int cand_cmp_up(const void *pa, const void *pb)
{
    const struct cand *a = pa, *b = pb;
    if (a->logit != b->logit)
        return a->logit < b->logit ? -1 : 1;
    return a->id < b->id ? -1 : 1;
}

/* The draw through llama.cpp's stages, in the order asked. */
static int32_t draw_chain(struct janas_sampler *s, const float *lg,
                          int filtered)
{
    const struct janas_sample_params *o = &s->p;
    const struct janas_sample_more *m = &o->more;
    if (need_scratch(s) != 0)
        return draw_plain(s, lg, filtered);
    const uint8_t *st = m->n_stage ? m->stage : OWN_ORDER;
    int ns =
        m->n_stage ? m->n_stage : (int)(sizeof OWN_ORDER / sizeof *OWN_ORDER);
    float hot =
        o->temperature + (m->dynatemp_range > 0 ? m->dynatemp_range : 0);
    if (hot < 1)
        hot = 1;
    float mx = lg[argmax(lg, s->nv)];
    float floor = mx - 30.0f * hot;
    int at = 0;
    /* the stages that cut before the candidates are gathered: top-n-sigma
       over the whole row, top-k by the heap */
    uint32_t k = s->nv;
    for (; at < ns; at++) {
        int x = st[at];
        if (!stage_on(o, x) || x == JANAS_STAGE_PENALTIES ||
            x == JANAS_STAGE_DRY)
            continue;
        if (x == JANAS_STAGE_TOP_N_SIGMA && !m->mirostat) {
            double sum = 0, sq = 0;
            uint32_t cnt = 0;
            for (uint32_t i = 0; i < s->nv; i++)
                if (isfinite(lg[i])) {
                    sum += lg[i];
                    sq += (double)lg[i] * lg[i];
                    cnt++;
                }
            double mean = cnt ? sum / cnt : 0;
            double sd = cnt ? sqrt(fmax(sq / cnt - mean * mean, 0)) : 0;
            float f = mx - m->top_n_sigma * (float)sd;
            if (f > floor)
                floor = f;
            continue;
        }
        if (x == JANAS_STAGE_TOP_K && !m->mirostat)
            k = (uint32_t)o->top_k < s->nv ? (uint32_t)o->top_k : s->nv, at++;
        break;
    }
    uint32_t nc = filtered    ? filtered_cands(s, lg, k, floor)
                  : k < s->nv ? top_k_cands(s->cands, k, lg, s->nv, floor)
                              : all_cands(s, lg, floor, 0);
    if (nc == 0 && filtered)
        nc = filtered_cands(s, lg, k, -INFINITY);
    if (nc == 0)
        return -1;
    float inv = 1; /* 1 / the temperature, once it is applied */
    if (m->mirostat)
        return mirostat(s, nc, 1 / o->temperature);
    for (; at < ns && nc > 1; at++) {
        int x = st[at];
        if (!stage_on(o, x))
            continue;
        switch (x) {
        case JANAS_STAGE_TOP_K:
            if ((uint32_t)o->top_k < nc)
                nc = (uint32_t)o->top_k;
            break;
        case JANAS_STAGE_TOP_N_SIGMA: {
            double sum = 0, sq = 0;
            for (uint32_t i = 0; i < nc; i++) {
                sum += s->cands[i].logit;
                sq += (double)s->cands[i].logit * s->cands[i].logit;
            }
            double mean = sum / nc, sd = sqrt(fmax(sq / nc - mean * mean, 0));
            float f = s->cands[0].logit - m->top_n_sigma * (float)sd;
            uint32_t kk = 1;
            while (kk < nc && s->cands[kk].logit >= f)
                kk++;
            nc = kk;
            break;
        }
        case JANAS_STAGE_TOP_P: {
            double sum = probs(s, nc, inv), c = 0;
            uint32_t kk = 0;
            while (kk < nc && c < o->top_p * sum)
                c += s->prob[kk++];
            nc = kk ? kk : 1;
            break;
        }
        case JANAS_STAGE_MIN_P: {
            float f = s->cands[0].logit + logf(o->min_p) / inv;
            uint32_t kk = 1;
            while (kk < nc && s->cands[kk].logit >= f)
                kk++;
            nc = kk;
            break;
        }
        case JANAS_STAGE_TYPICAL: {
            /* the tokens whose surprise is nearest the entropy, up to the
               mass asked */
            double sum = probs(s, nc, inv), h = 0;
            for (uint32_t i = 0; i < nc; i++) {
                double p = s->prob[i] / sum;
                if (p > 0)
                    h -= p * log(p);
            }
            for (uint32_t i = 0; i < nc; i++) {
                double p = s->prob[i] / sum;
                s->alt[i] = (struct cand){
                    (float)fabs((p > 0 ? -log(p) : 1e30) - h), (int32_t)i};
            }
            qsort(s->alt, nc, sizeof *s->alt, cand_cmp_up);
            double c = 0;
            uint32_t kk = 0;
            while (kk < nc && c < m->typical_p * sum)
                c += s->prob[s->alt[kk++].id];
            if (!kk)
                kk = 1;
            /* the ones kept, back in the candidates' order */
            for (uint32_t i = 0; i < kk; i++)
                s->alt[i] = s->cands[s->alt[i].id];
            qsort(s->alt, kk, sizeof *s->alt, cand_cmp);
            memcpy(s->cands, s->alt, kk * sizeof *s->alt);
            nc = kk;
            break;
        }
        case JANAS_STAGE_XTC:
            /* now and then, every choice above the threshold but the
               least likely of them left out */
            if (aside(s) < m->xtc_probability) {
                double sum = probs(s, nc, inv);
                uint32_t last = 0;
                while (last + 1 < nc &&
                       s->prob[last + 1] / sum >= m->xtc_threshold)
                    last++;
                if (last >= 1 && s->prob[0] / sum >= m->xtc_threshold) {
                    memmove(s->cands, s->cands + last,
                            (nc - last) * sizeof *s->cands);
                    nc -= last;
                }
            }
            break;
        case JANAS_STAGE_TEMPERATURE: {
            float t = o->temperature;
            if (m->dynatemp_range > 0) {
                /* hotter where the model is unsure: the entropy of the
                   candidates, over its largest, to the exponent */
                double sum = probs(s, nc, 1), h = 0;
                for (uint32_t i = 0; i < nc; i++) {
                    double p = s->prob[i] / sum;
                    if (p > 0)
                        h -= p * log(p);
                }
                double e = h / log((double)nc);
                float lo = t - m->dynatemp_range, hi = t + m->dynatemp_range;
                if (lo < 0)
                    lo = 0;
                t = lo + (hi - lo) * (float)pow(e, m->dynatemp_exponent > 0
                                                       ? m->dynatemp_exponent
                                                       : 1);
            }
            if (t <= 0) {
                nc = 1;
                break;
            }
            inv = 1 / t;
            break;
        }
        default:
            break;
        }
    }
    return pick_from(s, nc, inv);
}

static int32_t draw(struct janas_sampler *s, const float *lg, int filtered)
{
    if (s->p.temperature <= 0 || !chain_asked(&s->p))
        return draw_plain(s, lg, filtered);
    return draw_chain(s, lg, filtered);
}

int janas_sample_stages(const char *names, uint8_t *stage, int max)
{
    static const struct {
        const char *name;
        char letter;
        uint8_t st;
    } N[] = {{"penalties", 'e', JANAS_STAGE_PENALTIES},
             {"dry", 'd', JANAS_STAGE_DRY},
             {"top_n_sigma", 's', JANAS_STAGE_TOP_N_SIGMA},
             {"top_k", 'k', JANAS_STAGE_TOP_K},
             {"typ_p", 'y', JANAS_STAGE_TYPICAL},
             {"typical_p", 0, JANAS_STAGE_TYPICAL},
             {"top_p", 'p', JANAS_STAGE_TOP_P},
             {"min_p", 'm', JANAS_STAGE_MIN_P},
             {"xtc", 'x', JANAS_STAGE_XTC},
             {"temperature", 't', JANAS_STAGE_TEMPERATURE},
             {"temp", 0, JANAS_STAGE_TEMPERATURE}};
    int n = 0;
    const char *p = names ? names : "";
    size_t sep = strcspn(p, ";, ");
    /* the letters alone, as llama.cpp's --sampling-seq */
    if (p[sep] == 0 && sep && !strchr(p, '_')) {
        int letters = 1;
        for (size_t i = 0; i < sep && letters; i++) {
            int f = 0;
            for (size_t j = 0; j < sizeof N / sizeof *N; j++)
                f |= N[j].letter == p[i];
            letters = f;
        }
        int word = 0;
        for (size_t j = 0; j < sizeof N / sizeof *N; j++)
            word |= strlen(N[j].name) == sep && !strncmp(N[j].name, p, sep);
        if (letters && !word) {
            for (size_t i = 0; i < sep; i++) {
                if (n == max)
                    return -1;
                for (size_t j = 0; j < sizeof N / sizeof *N; j++)
                    if (N[j].letter == p[i]) {
                        stage[n++] = N[j].st;
                        break;
                    }
            }
            return n;
        }
    }
    while (*p) {
        size_t l = strcspn(p, ";, ");
        if (l) {
            size_t j = 0;
            while (j < sizeof N / sizeof *N &&
                   !(strlen(N[j].name) == l && !strncmp(N[j].name, p, l)))
                j++;
            if (j == sizeof N / sizeof *N || n == max)
                return -1;
            stage[n++] = N[j].st;
        }
        p += l;
        if (*p)
            p++;
    }
    return n;
}

int32_t janas_sampler_pick(struct janas_sampler *s, const float *logits,
                           struct janas_token_lp *lp)
{
    int filtered = s->has_filter && s->filter.active(s->filter.ctx);
    float mu = s->mu;
    int32_t t = draw(s, adjusted(s, logits), filtered);
    if (t < 0) {
        s->mu = mu;
        return -1;
    }
    /* mu before this draw, for a rewind */
    s->mu_ring[s->n_draws++ % (sizeof s->mu_ring / sizeof *s->mu_ring)] = mu;
    if (lp && s->p.logprobs >= 0)
        learn(s, logits, t, lp);
    if (s->count[t]++ == 0)
        s->seen[s->n_seen++] = t;
    if (s->n_hist == s->cap_hist) {
        uint32_t cap = s->cap_hist ? 2 * s->cap_hist : 256;
        int32_t *h = realloc(s->hist, cap * sizeof *h);
        if (h) {
            s->hist = h;
            s->cap_hist = cap;
        }
    }
    if (s->n_hist < s->cap_hist)
        s->hist[s->n_hist++] = t;
    if (s->has_filter)
        s->filter.accept(s->filter.ctx, t);
    return t;
}
