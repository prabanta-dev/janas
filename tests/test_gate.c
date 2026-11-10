/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_gate.c - the entrance gate's file (src/llm/gate.c): a valid gate
 * decides as its readers say; a gate for another model is refused; and,
 * since the file is untrusted, random byte flips and truncations never
 * crash the parser, every accepted gate keeping its invariants (run it
 * under the asan and ubsan variants too).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/gate.h"

static int failures;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fputc('\n', stderr);                                               \
        }                                                                      \
    } while (0)

enum { D = 8, LAYER = 2, N_LAYER = 4, N_OUT = 3, N_DOM = 2 };

/* A gate whose reader r scores x[r] (outcomes) and x[3 + i] (domains). */
static size_t build(unsigned char *buf)
{
    size_t p = 0;
    uint32_t h[5] = {1, D, LAYER, N_OUT, N_DOM};
    memcpy(buf, "JANASGT1", 8), p += 8;
    memcpy(buf + p, h, sizeof(h)), p += sizeof(h);
    const char *names[N_OUT + N_DOM] = {"go on", "Specialist", "I don't know",
                                        "fisica", "diritto"};
    for (int r = 0; r < N_OUT + N_DOM; r++) {
        char name[32] = {0};
        float w[D + 1] = {0};
        snprintf(name, sizeof(name), "%s", names[r]);
        w[r < N_OUT ? r : 3 + (r - N_OUT)] = 1.0f;
        w[D] = -0.1f;
        memcpy(buf + p, name, 32), p += 32;
        memcpy(buf + p, w, sizeof(w)), p += sizeof(w);
    }
    return p;
}

int main(void)
{
    static unsigned char img[4096], mut[4096];
    char err[256];
    size_t len = build(img);
    struct janas_gate *g =
        janas_gate_parse(img, len, D, N_LAYER, err, sizeof(err));
    CHECK(g, "a valid gate refused: %s", err);
    if (g) {
        CHECK(janas_gate_layer(g) == LAYER, "layer %u", janas_gate_layer(g));
        CHECK(janas_gate_domains(g) == N_DOM, "domains %u",
              janas_gate_domains(g));
        CHECK(!strcmp(janas_gate_domain_name(g, 1), "diritto"), "name %s",
              janas_gate_domain_name(g, 1));
        float x[D] = {0};
        double sc[JANAS_GATE_OUTCOMES];
        uint32_t dom = 9;
        x[2] = 3.0f, x[4] = 1.0f; /* I don't know, diritto */
        int o = janas_gate_decide(g, x, &dom, sc);
        CHECK(o == JANAS_GATE_UNKNOWN, "outcome %d", o);
        CHECK(dom == 1, "domain %u", dom);
        CHECK(fabs(sc[JANAS_GATE_UNKNOWN] - 2.9) < 1e-6, "score %g",
              sc[JANAS_GATE_UNKNOWN]);
        CHECK(isinf(sc[JANAS_GATE_CORRECT]), "a missing reader scored %g",
              sc[JANAS_GATE_CORRECT]);
        x[2] = 0.0f, x[1] = 1.0f;
        CHECK(janas_gate_decide(g, x, NULL, NULL) == JANAS_GATE_SPECIALIST,
              "the Specialist");
        janas_gate_free(g);
    }
    /* another model's gate */
    CHECK(!janas_gate_parse(img, len, D + 1, N_LAYER, err, sizeof(err)),
          "a wider model accepted");
    CHECK(!janas_gate_parse(img, len, D, LAYER, err, sizeof(err)),
          "a model with fewer layers accepted");
    /* truncations and byte flips */
    for (size_t t = 0; t < len; t += 7)
        CHECK(!janas_gate_parse(img, t, D, N_LAYER, err, sizeof(err)),
              "truncated to %zu accepted", t);
    srand(7);
    int accepted = 0;
    for (int it = 0; it < 20000; it++) {
        memcpy(mut, img, len);
        int flips = 1 + rand() % 4;
        for (int k = 0; k < flips; k++)
            mut[rand() % len] ^= (unsigned char)(1 + rand() % 255);
        struct janas_gate *m =
            janas_gate_parse(mut, len, D, N_LAYER, err, sizeof(err));
        if (!m)
            continue;
        accepted++;
        CHECK(janas_gate_layer(m) < N_LAYER, "layer out of range");
        float x[D];
        for (int i = 0; i < D; i++)
            x[i] = (float)(rand() % 100) / 10.0f;
        uint32_t dom = 0;
        int o = janas_gate_decide(m, x, &dom, NULL);
        CHECK(o >= 0 && o < JANAS_GATE_OUTCOMES, "outcome %d", o);
        CHECK(dom < janas_gate_domains(m) || janas_gate_domains(m) == 0,
              "domain %u", dom);
        CHECK(strlen(janas_gate_domain_name(m, dom)) < 32, "name");
        janas_gate_free(m);
    }
    printf("test_gate: %d mutations accepted (weights and names), %s\n",
           accepted, failures ? "FAILED" : "ok");
    return failures != 0;
}
