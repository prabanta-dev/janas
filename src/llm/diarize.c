/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * diarize.c - who speaks when: the stretches of speech cut into windows of
 * 1.5 s every 0.75 s (a shorter stretch one window), each embedded
 * (speaker.c) and normalized, grouped by WeSpeaker's spectral clustering:
 * the cosine affinities pruned to each window's nearest, the Laplacian's
 * smallest eigenvalues' largest gap for the number of speakers, k-means on
 * its eigenvectors; past 2000 windows (25 minutes of speech) by chunks,
 * as NeMo's long-form clustering. Known speakers' embeddings take the groups
 * nearest them, one each, where their cosine is above 0.5. Groups numbered in
 * the order they first speak.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "speaker.h"

#define DIM 256
#define WIN_FRAMES 150
#define STEP_FRAMES 75

void janas_diar_free(struct janas_diar *d)
{
    free(d->win);
    free(d->speaker);
    free(d->known);
    memset(d, 0, sizeof *d);
}

static void normalize(float *v)
{
    double s = 0;
    for (int i = 0; i < DIM; i++)
        s += (double)v[i] * v[i];
    float k = s > 0 ? (float)(1 / sqrt(s)) : 0;
    for (int i = 0; i < DIM; i++)
        v[i] *= k;
}

static float cosine(const float *a, const float *b)
{
    double s = 0, na = 0, nb = 0;
    for (int i = 0; i < DIM; i++) {
        s += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return na > 0 && nb > 0 ? (float)(s / sqrt(na * nb)) : 0;
}

/* the eigenvalues (ascending, into d) and eigenvectors (columns of a, n x
   n row-major) of a symmetric matrix a: Householder's reduction to a
   tridiagonal and the implicit QL iterations (EISPACK's tred2 and tql2) */
static int eigh(double *a, int n, double *d)
{
    double *e = calloc((size_t)n, sizeof *e);
    if (!e)
        return -1;
#define A(i, j) a[(size_t)(i) * n + (j)]
    for (int i = n - 1; i > 0; i--) {
        int l = i - 1;
        double h = 0, scale = 0;
        if (l > 0) {
            for (int k = 0; k <= l; k++)
                scale += fabs(A(i, k));
            if (scale == 0) {
                e[i] = A(i, l);
            } else {
                for (int k = 0; k <= l; k++) {
                    A(i, k) /= scale;
                    h += A(i, k) * A(i, k);
                }
                double f = A(i, l);
                double g = f >= 0 ? -sqrt(h) : sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                A(i, l) = f - g;
                f = 0;
                for (int j = 0; j <= l; j++) {
                    A(j, i) = A(i, j) / h;
                    g = 0;
                    for (int k = 0; k <= j; k++)
                        g += A(j, k) * A(i, k);
                    for (int k = j + 1; k <= l; k++)
                        g += A(k, j) * A(i, k);
                    e[j] = g / h;
                    f += e[j] * A(i, j);
                }
                double hh = f / (h + h);
                for (int j = 0; j <= l; j++) {
                    f = A(i, j);
                    e[j] = g = e[j] - hh * f;
                    for (int k = 0; k <= j; k++)
                        A(j, k) -= f * e[k] + g * A(i, k);
                }
            }
        } else {
            e[i] = A(i, l);
        }
        d[i] = h;
    }
    d[0] = 0;
    e[0] = 0;
    for (int i = 0; i < n; i++) {
        int l = i - 1;
        if (d[i] != 0) {
            for (int j = 0; j <= l; j++) {
                double g = 0;
                for (int k = 0; k <= l; k++)
                    g += A(i, k) * A(k, j);
                for (int k = 0; k <= l; k++)
                    A(k, j) -= g * A(k, i);
            }
        }
        d[i] = A(i, i);
        A(i, i) = 1;
        for (int j = 0; j <= l; j++)
            A(j, i) = A(i, j) = 0;
    }
    /* tql2 */
    for (int i = 1; i < n; i++)
        e[i - 1] = e[i];
    e[n - 1] = 0;
    for (int l = 0; l < n; l++) {
        int iter = 0, m;
        do {
            for (m = l; m < n - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) <= 1e-15 * dd)
                    break;
            }
            if (m != l) {
                if (iter++ == 60) {
                    free(e);
                    return -1;
                }
                double g = (d[l + 1] - d[l]) / (2 * e[l]);
                double r = hypot(g, 1);
                g = d[m] - d[l] + e[l] / (g + (g >= 0 ? fabs(r) : -fabs(r)));
                double sn = 1, c = 1, pp = 0;
                int i;
                for (i = m - 1; i >= l; i--) {
                    double f = sn * e[i], b = c * e[i];
                    e[i + 1] = r = hypot(f, g);
                    if (r == 0) {
                        d[i + 1] -= pp;
                        e[m] = 0;
                        break;
                    }
                    sn = f / r;
                    c = g / r;
                    g = d[i + 1] - pp;
                    r = (d[i] - g) * sn + 2 * c * b;
                    pp = sn * r;
                    d[i + 1] = g + pp;
                    g = c * r - b;
                    for (int k = 0; k < n; k++) {
                        f = A(k, i + 1);
                        A(k, i + 1) = sn * A(k, i) + c * f;
                        A(k, i) = c * A(k, i) - sn * f;
                    }
                }
                if (r == 0 && i >= l)
                    continue;
                d[l] -= pp;
                e[l] = g;
                e[m] = 0;
            }
        } while (m != l);
    }
    /* ascending, the vectors with them */
    for (int i = 0; i < n - 1; i++) {
        int k = i;
        for (int j = i + 1; j < n; j++)
            if (d[j] < d[k])
                k = j;
        if (k != i) {
            double t = d[i];
            d[i] = d[k];
            d[k] = t;
            for (int r = 0; r < n; r++) {
                t = A(r, i);
                A(r, i) = A(r, k);
                A(r, k) = t;
            }
        }
    }
#undef A
    free(e);
    return 0;
}

static uint64_t rng_next(uint64_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s;
}

/* k-means of n points of dim dimensions into k groups: k-means++ starts,
   n_init of them, the one of least inertia kept */
static void kmeans(const double *x, int n, int dim, int k, int n_init,
                   uint32_t *label)
{
    double *c = malloc(sizeof(double) * (size_t)k * dim);
    double *dist2 = malloc(sizeof(double) * (size_t)n);
    uint32_t *lab = malloc(sizeof(uint32_t) * (size_t)n);
    if (!c || !dist2 || !lab) {
        for (int i = 0; i < n; i++)
            label[i] = 0;
        free(c);
        free(dist2);
        free(lab);
        return;
    }
    uint64_t seed = 0x9e3779b97f4a7c15ull;
    double best = INFINITY;
    for (int init = 0; init < n_init; init++) {
        /* k-means++: each point's distance to the nearest centre so far,
           brought down by each new one */
        int first = (int)(rng_next(&seed) % (uint64_t)n);
        memcpy(c, x + (size_t)first * dim, sizeof(double) * (size_t)dim);
        for (int i = 0; i < n; i++)
            dist2[i] = INFINITY;
        for (int j = 1; j < k; j++) {
            double tot = 0;
            const double *cq = c + (size_t)(j - 1) * dim;
            for (int i = 0; i < n; i++) {
                double s2 = 0;
                for (int t = 0; t < dim; t++) {
                    double dd = x[(size_t)i * dim + t] - cq[t];
                    s2 += dd * dd;
                }
                if (s2 < dist2[i])
                    dist2[i] = s2;
                tot += dist2[i];
            }
            double r =
                (double)(rng_next(&seed) >> 11) / 9007199254740992.0 * tot;
            int pick = n - 1;
            for (int i = 0; i < n; i++) {
                r -= dist2[i];
                if (r <= 0) {
                    pick = i;
                    break;
                }
            }
            memcpy(c + (size_t)j * dim, x + (size_t)pick * dim,
                   sizeof(double) * (size_t)dim);
        }
        /* Lloyd's iterations */
        double inertia = 0;
        for (int it = 0; it < 300; it++) {
            int moved = 0;
            inertia = 0;
            for (int i = 0; i < n; i++) {
                double m = INFINITY;
                uint32_t bq = 0;
                for (int q = 0; q < k; q++) {
                    double s2 = 0;
                    for (int t = 0; t < dim; t++) {
                        double dd =
                            x[(size_t)i * dim + t] - c[(size_t)q * dim + t];
                        s2 += dd * dd;
                    }
                    if (s2 < m) {
                        m = s2;
                        bq = (uint32_t)q;
                    }
                }
                if (it == 0 || lab[i] != bq)
                    moved = 1;
                lab[i] = bq;
                inertia += m;
            }
            if (!moved)
                break;
            for (int q = 0; q < k; q++) {
                int cnt = 0;
                double *cq = c + (size_t)q * dim;
                memset(cq, 0, sizeof(double) * (size_t)dim);
                for (int i = 0; i < n; i++)
                    if (lab[i] == (uint32_t)q) {
                        cnt++;
                        for (int t = 0; t < dim; t++)
                            cq[t] += x[(size_t)i * dim + t];
                    }
                for (int t = 0; t < dim && cnt; t++)
                    cq[t] /= cnt;
            }
        }
        if (inertia < best) {
            best = inertia;
            memcpy(label, lab, sizeof(uint32_t) * (size_t)n);
        }
    }
    free(c);
    free(dist2);
    free(lab);
}

/* WeSpeaker's spectral clustering of n normalized embeddings: 0, or -1
   when out of memory */
static int spectral(const float *emb, uint32_t nw, uint32_t max_spk,
                    uint32_t *label)
{
    int n = (int)nw;
    if (n <= 2) {
        for (int i = 0; i < n; i++)
            label[i] = 0;
        return 0;
    }
    double *m = malloc(sizeof(double) * (size_t)n * n);
    double *row = malloc(sizeof(double) * (size_t)n);
    double *d = malloc(sizeof(double) * (size_t)n);
    int *idx = malloc(sizeof(int) * (size_t)n);
    if (!m || !row || !d || !idx) {
        free(m);
        free(row);
        free(d);
        free(idx);
        return -1;
    }
    int keep = n < 1000 ? (n - 10 > 2 ? 10 : n - 2) : n - (int)(0.99 * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int t = 0; t < DIM; t++)
                s +=
                    (double)emb[(size_t)i * DIM + t] * emb[(size_t)j * DIM + t];
            row[j] = 0.5 * (1 + s);
            idx[j] = j;
        }
        /* the keep largest to 1, the rest to 0 (a stable order: ties by
           index, as numpy's argsort) */
        for (int a = 0; a < keep; a++) {
            int b = a;
            for (int c = a + 1; c < n; c++)
                if (row[idx[c]] > row[idx[b]] ||
                    (row[idx[c]] == row[idx[b]] && idx[c] > idx[b]))
                    b = c;
            int t = idx[a];
            idx[a] = idx[b];
            idx[b] = t;
        }
        for (int j = 0; j < n; j++)
            m[(size_t)i * n + j] = 0;
        for (int a = 0; a < keep; a++)
            m[(size_t)i * n + idx[a]] = 1;
    }
    for (int i = 0; i < n; i++) /* symmetric, no self-loops: the Laplacian */
        for (int j = i; j < n; j++) {
            double v = 0.5 * (m[(size_t)i * n + j] + m[(size_t)j * n + i]);
            m[(size_t)i * n + j] = m[(size_t)j * n + i] = i == j ? 0 : -v;
        }
    for (int i = 0; i < n; i++) {
        double deg = 0;
        for (int j = 0; j < n; j++)
            deg -= m[(size_t)i * n + j];
        m[(size_t)i * n + i] = deg;
    }
    if (eigh(m, n, d) != 0) {
        free(m);
        free(row);
        free(d);
        free(idx);
        return -1;
    }
    int lim = (int)max_spk < n - 1 ? (int)max_spk : n - 1, k = 1;
    double gap = -1;
    for (int i = 0; i < lim; i++)
        if (d[i + 1] - d[i] > gap) {
            gap = d[i + 1] - d[i];
            k = i + 1;
        }
    double *x = malloc(sizeof(double) * (size_t)n * k);
    if (!x) {
        free(m);
        free(row);
        free(d);
        free(idx);
        return -1;
    }
    for (int i = 0; i < n; i++)
        for (int q = 0; q < k; q++)
            x[(size_t)i * k + q] = m[(size_t)i * n + q];
    kmeans(x, n, k, k, 10, label);
    free(x);
    free(m);
    free(row);
    free(d);
    free(idx);
    return 0;
}

/* past CHUNK windows, as NeMo's long-form clustering: each run of about
   CHUNK windows reduced by k-means to REPS centroids, the centroids grouped
   (reduced the same way again while they are more than CHUNK), each window
   taking its centroid's group. Memory and time grow with the length, not
   with its square; CHUNK windows or fewer are grouped as before */
#define CHUNK 2000
#define REPS 64
int janas_diar_cluster(const float *emb, uint32_t n, uint32_t max_spk,
                       uint32_t chunk, uint32_t *label)
{
    if (!chunk)
        chunk = CHUNK;
    if (n <= chunk)
        return spectral(emb, n, max_spk, label);
    uint32_t nc = (n + chunk - 1) / chunk, nr = nc * REPS, r = 0;
    float *rep = calloc((size_t)nr * DIM, sizeof(float));
    uint32_t *of = malloc(sizeof(uint32_t) * n);
    uint32_t *rl = malloc(sizeof(uint32_t) * nr);
    double *x = malloc(sizeof(double) * ((size_t)n / nc + 1) * DIM);
    int rc = -1;
    if (!rep || !of || !rl || !x)
        goto end;
    for (uint32_t ch = 0; ch < nc; ch++) {
        uint32_t a = (uint32_t)((uint64_t)n * ch / nc);
        uint32_t b = (uint32_t)((uint64_t)n * (ch + 1) / nc);
        for (size_t i = 0; i < (size_t)(b - a) * DIM; i++)
            x[i] = emb[(size_t)a * DIM + i];
        kmeans(x, (int)(b - a), DIM, REPS, 1, of + a);
        /* the centroids that took windows, numbered on from r */
        uint32_t map[REPS];
        for (int q = 0; q < REPS; q++)
            map[q] = UINT32_MAX;
        for (uint32_t i = a; i < b; i++) {
            if (map[of[i]] == UINT32_MAX)
                map[of[i]] = r++;
            of[i] = map[of[i]];
            float *p = rep + (size_t)of[i] * DIM;
            for (int t = 0; t < DIM; t++)
                p[t] += emb[(size_t)i * DIM + t];
        }
    }
    for (uint32_t i = 0; i < r; i++)
        normalize(rep + (size_t)i * DIM);
    if (janas_diar_cluster(rep, r, max_spk, chunk, rl) != 0)
        goto end;
    for (uint32_t i = 0; i < n; i++)
        label[i] = rl[of[i]];
    /* a centroid may hold windows of two speakers: each window to the
       group whose centre (of all its windows) is nearest */
    uint32_t ng = 0;
    for (uint32_t i = 0; i < n; i++)
        ng = label[i] + 1 > ng ? label[i] + 1 : ng;
    float *gc = calloc((size_t)ng * DIM, sizeof(float));
    if (!gc)
        goto end;
    for (uint32_t i = 0; i < n; i++)
        for (int t = 0; t < DIM; t++)
            gc[(size_t)label[i] * DIM + t] += emb[(size_t)i * DIM + t];
    for (uint32_t g = 0; g < ng; g++)
        normalize(gc + (size_t)g * DIM);
    for (uint32_t i = 0; i < n; i++) {
        double best = -2;
        for (uint32_t g = 0; g < ng; g++) {
            double c = 0;
            for (int t = 0; t < DIM; t++)
                c += (double)emb[(size_t)i * DIM + t] * gc[(size_t)g * DIM + t];
            if (c > best) {
                best = c;
                label[i] = g;
            }
        }
    }
    free(gc);
    rc = 0;
end:
    free(rep);
    free(of);
    free(rl);
    free(x);
    return rc;
}

/* the whole filter banks' frames f0 on */
static const float *whole_frames(void *user, uint32_t f0, uint32_t f1)
{
    (void)f1;
    return (const float *)user + (size_t)f0 * 80;
}

int janas_diarize(struct janas_speaker *s, const float *pcm, size_t n,
                  const size_t *regions, size_t n_regions, const float *ref,
                  uint32_t n_ref, uint32_t max_speakers, struct janas_diar *out)
{
    memset(out, 0, sizeof *out);
    uint32_t frames = 0;
    float *fb = janas_speaker_fbank(pcm, n, &frames);
    if (!fb)
        return -1;
    int rc = janas_diarize_read(s, frames, whole_frames, fb, regions, n_regions,
                                ref, n_ref, max_speakers, out);
    free(fb);
    return rc;
}

int janas_diarize_read(struct janas_speaker *s, uint32_t frames,
                       const float *(*get)(void *user, uint32_t f0,
                                           uint32_t f1),
                       void *user, const size_t *regions, size_t n_regions,
                       const float *ref, uint32_t n_ref, uint32_t max_speakers,
                       struct janas_diar *out)
{
    memset(out, 0, sizeof *out);
    /* the windows, frames of the filter banks (10 ms each) */
    size_t cap = 64;
    uint32_t nw = 0;
    uint32_t(*w)[2] = malloc(sizeof *w * cap);
    float *emb = NULL;
    uint32_t *of = NULL;
    int rc = -1;
    if (!w)
        goto end;
    for (size_t r = 0; r < n_regions; r++) {
        uint32_t a = (uint32_t)(regions[2 * r] / 160);
        uint32_t b = (uint32_t)(regions[2 * r + 1] / 160);
        if (b > frames)
            b = frames;
        if (b <= a || b - a < 8)
            continue;
        for (uint32_t f = a;; f += STEP_FRAMES) {
            uint32_t e = f + WIN_FRAMES < b ? f + WIN_FRAMES : b;
            if (e - f < 8)
                break;
            if (nw == cap) {
                cap *= 2;
                uint32_t(*t)[2] = realloc(w, sizeof *w * cap);
                if (!t)
                    goto end;
                w = t;
            }
            w[nw][0] = f;
            w[nw][1] = e;
            nw++;
            if (e == b)
                break;
        }
    }
    out->win = malloc(sizeof(size_t) * 2 * (nw ? nw : 1));
    out->speaker = malloc(sizeof(int) * (nw ? nw : 1));
    emb = malloc(sizeof(float) * DIM * (nw ? nw : 1));
    of = malloc(sizeof(uint32_t) * (nw ? nw : 1));
    if (!out->win || !out->speaker || !emb || !of)
        goto end;
    for (uint32_t i = 0; i < nw; i++) {
        out->win[2 * i] = (size_t)w[i][0] * 160;
        out->win[2 * i + 1] = (size_t)w[i][1] * 160 + 240;
        const float *fb = get(user, w[i][0], w[i][1]);
        if (!fb || janas_speaker_embed(s, fb, 0, w[i][1] - w[i][0],
                                       emb + (size_t)i * DIM) != 0)
            goto end;
        normalize(emb + (size_t)i * DIM);
    }
    out->n_win = nw;
    /* spectral clustering, as WeSpeaker's: the affinity of the cosine
       similarities kept to each window's nearest (10, or the top 1% past
       1000 windows) and made symmetric, its Laplacian, as many groups as
       the largest gap among its smallest eigenvalues says, k-means on
       their eigenvectors; long recordings by chunks */
    if (janas_diar_cluster(emb, nw, max_speakers ? max_speakers : 20, 0, of) !=
        0)
        goto end;
    /* numbered in order of first appearance */
    int *label = malloc(sizeof(int) * (nw ? nw : 1));
    if (!label)
        goto end;
    for (uint32_t i = 0; i < nw; i++)
        label[i] = -1;
    uint32_t ns = 0;
    for (uint32_t i = 0; i < nw; i++) {
        if (label[of[i]] < 0)
            label[of[i]] = (int)ns++;
        out->speaker[i] = label[of[i]];
    }
    out->n_speakers = ns;
    /* the known speakers: each to the group nearest it */
    out->known = malloc(sizeof(int) * (ns ? ns : 1));
    float *gc = calloc((size_t)(ns ? ns : 1) * DIM, sizeof(float));
    if (!out->known || !gc) {
        free(label);
        free(gc);
        goto end;
    }
    for (uint32_t g = 0; g < ns; g++)
        out->known[g] = -1;
    for (uint32_t i = 0; i < nw; i++)
        for (int k = 0; k < DIM; k++)
            gc[(size_t)out->speaker[i] * DIM + k] += emb[(size_t)i * DIM + k];
    if (n_ref && ns) {
        char *ref_used = calloc(n_ref, 1);
        for (;;) {
            float best = 0.5f;
            int bg = -1, br = -1;
            for (uint32_t g = 0; g < ns; g++) {
                if (out->known[g] >= 0)
                    continue;
                for (uint32_t r = 0; r < n_ref; r++) {
                    if (ref_used && ref_used[r])
                        continue;
                    float c =
                        cosine(gc + (size_t)g * DIM, ref + (size_t)r * DIM);
                    if (c > best) {
                        best = c;
                        bg = (int)g;
                        br = (int)r;
                    }
                }
            }
            if (bg < 0 || !ref_used)
                break;
            out->known[bg] = br;
            ref_used[br] = 1;
        }
        free(ref_used);
    }
    free(gc);
    free(label);
    rc = 0;
end:
    if (rc != 0)
        janas_diar_free(out);
    free(w);
    free(emb);
    free(of);
    return rc;
}
