/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gate.c - the entrance gate: its file read and checked (an untrusted
 * input, as a model's), and its decision on a residual (gate.h).
 */
#include "gate.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GATE_MAGIC "JANASGT1"
#define GATE_NAME 32

struct janas_gate {
    uint32_t d, layer, n_out, n_dom;
    char (*name)[GATE_NAME]; /* n_out + n_dom names */
    float *w;                /* (n_out + n_dom) rows of d + 1 */
};

static const char *const OUTCOMES[JANAS_GATE_OUTCOMES] = {
    "go on", "Specialist", "I don't know", "correct", "did you mean"};

const char *janas_gate_outcome_name(int outcome)
{
    return outcome >= 0 && outcome < JANAS_GATE_OUTCOMES ? OUTCOMES[outcome]
                                                         : "?";
}

void janas_gate_free(struct janas_gate *g)
{
    if (!g)
        return;
    free(g->name);
    free(g->w);
    free(g);
}

struct janas_gate *janas_gate_parse(const void *buf, size_t len,
                                    uint32_t d_model, uint32_t n_layer,
                                    char *err, size_t err_len)
{
    const unsigned char *p = buf;
    uint32_t h[5];
    if (!buf || len < 8 + sizeof(h) || memcmp(p, GATE_MAGIC, 8)) {
        snprintf(err, err_len, "not a gate file");
        return NULL;
    }
    memcpy(h, p + 8, sizeof(h));
    uint32_t ver = h[0], d = h[1], layer = h[2], n_out = h[3], n_dom = h[4];
    if (ver != 1) {
        snprintf(err, err_len, "gate file version %u, not 1", ver);
        return NULL;
    }
    if (d != d_model || layer >= n_layer) {
        snprintf(err, err_len,
                 "the gate is for width %u and layer %u, the model has "
                 "width %u and %u layers",
                 d, layer, d_model, n_layer);
        return NULL;
    }
    if (n_out < JANAS_GATE_UNKNOWN + 1 || n_out > JANAS_GATE_OUTCOMES ||
        n_dom > 256) {
        snprintf(err, err_len, "a gate of %u outcomes and %u domains", n_out,
                 n_dom);
        return NULL;
    }
    uint64_t rows = (uint64_t)n_out + n_dom;
    uint64_t row = GATE_NAME + ((uint64_t)d + 1) * sizeof(float);
    if (len - 8 - sizeof(h) != rows * row) {
        snprintf(err, err_len, "the gate file is %zu bytes, not %llu", len,
                 (unsigned long long)(8 + sizeof(h) + rows * row));
        return NULL;
    }
    struct janas_gate *g = calloc(1, sizeof(*g));
    if (!g || !(g->name = calloc(rows, GATE_NAME)) ||
        !(g->w = malloc(rows * ((size_t)d + 1) * sizeof(float)))) {
        janas_gate_free(g);
        snprintf(err, err_len, "out of memory for the gate");
        return NULL;
    }
    g->d = d, g->layer = layer, g->n_out = n_out, g->n_dom = n_dom;
    const unsigned char *q = p + 8 + sizeof(h);
    for (uint64_t r = 0; r < rows; r++, q += row) {
        memcpy(g->name[r], q, GATE_NAME);
        g->name[r][GATE_NAME - 1] = 0;
        float *w = g->w + r * ((size_t)d + 1);
        memcpy(w, q + GATE_NAME, ((size_t)d + 1) * sizeof(float));
        for (uint32_t i = 0; i <= d; i++)
            if (!isfinite(w[i])) {
                janas_gate_free(g);
                snprintf(err, err_len, "a gate weight not finite");
                return NULL;
            }
    }
    return g;
}

struct janas_gate *janas_gate_load(const char *path, uint32_t d_model,
                                   uint32_t n_layer, char *err, size_t err_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, err_len, "%s: cannot open", path);
        return NULL;
    }
    struct janas_gate *g = NULL;
    long len = -1;
    unsigned char *buf = NULL;
    if (fseek(f, 0, SEEK_END) == 0 && (len = ftell(f)) > 0 &&
        len < (64L << 20) && fseek(f, 0, SEEK_SET) == 0 &&
        (buf = malloc((size_t)len)) &&
        fread(buf, 1, (size_t)len, f) == (size_t)len)
        g = janas_gate_parse(buf, (size_t)len, d_model, n_layer, err, err_len);
    else
        snprintf(err, err_len, "%s: cannot read", path);
    free(buf);
    fclose(f);
    return g;
}

uint32_t janas_gate_layer(const struct janas_gate *g)
{
    return g->layer;
}

uint32_t janas_gate_domains(const struct janas_gate *g)
{
    return g->n_dom;
}

const char *janas_gate_domain_name(const struct janas_gate *g, uint32_t i)
{
    return i < g->n_dom ? g->name[g->n_out + i] : "";
}

static double score(const struct janas_gate *g, uint32_t r, const float *x)
{
    const float *w = g->w + (size_t)r * (g->d + 1);
    double s = w[g->d];
    for (uint32_t i = 0; i < g->d; i++)
        s += (double)w[i] * x[i];
    return s;
}

int janas_gate_decide(const struct janas_gate *g, const float *x,
                      uint32_t *domain, double *scores)
{
    int best = 0;
    double sb = -INFINITY;
    for (uint32_t r = 0; r < JANAS_GATE_OUTCOMES; r++) {
        double s = r < g->n_out ? score(g, r, x) : -INFINITY;
        if (scores)
            scores[r] = s;
        if (s > sb)
            sb = s, best = (int)r;
    }
    if (domain) {
        uint32_t bd = 0;
        double sd = -INFINITY;
        for (uint32_t i = 0; i < g->n_dom; i++) {
            double s = score(g, g->n_out + i, x);
            if (s > sd)
                sd = s, bd = i;
        }
        *domain = bd;
    }
    return best;
}
