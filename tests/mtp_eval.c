/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mtp_eval.c - how often the MTP block guesses what the main model will
 * choose two tokens ahead (the acceptance rate of its drafts under greedy
 * decoding), and how often it guesses the text.
 *
 * Usage: mtp_eval <model.jns> <mtp.jns> <tokens.int32> [pos=h|pos=t]
 * pos=h (default): the row pairing token t_{i+1} with hidden state h_i runs
 * at position i; pos=t: at position i + 1 (row 0 then pairs t_0 with a zero
 * state).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/model.h"

static uint32_t argmax(const float *v, uint32_t n)
{
    uint32_t b = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[b])
            b = i;
    return b;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: mtp_eval <model.jns> <mtp.jns> <tokens> "
                        "[pos=h|pos=t]\n");
        return 2;
    }
    int pos_t = argc > 4 && strcmp(argv[4], "pos=t") == 0;
    FILE *f = fopen(argv[3], "rb");
    if (!f)
        return 1;
    int32_t tok[4096];
    uint32_t n = (uint32_t)fread(tok, sizeof(int32_t), 4096, f);
    fclose(f);
    int ccpus[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int icpus[8] = {12, 13, 14, 15, 16, 17, 18, 19};
    struct janas_llm_options o = {.cache_bytes = 20480ull << 20,
                                  .n_ctx = n + 64,
                                  .compute_cpus = ccpus,
                                  .n_compute = 12,
                                  .io_cpus = icpus,
                                  .n_io = 8,
                                  .mtp_path = argv[2]};
    char err[256];
    struct janas_llm_model *m = janas_llm_model_load(argv[1], &o, err, 256);
    if (!m) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    uint32_t nv = janas_llm_model_n_vocab(m), dm = 2048, B = 64;
    float *logits = malloc((size_t)B * nv * sizeof(float));
    float *hid = calloc((size_t)(n + 1) * dm, sizeof(float));
    uint32_t *main_next = malloc(n * sizeof(uint32_t));
    /* main model: its choice after every position, and its hidden states */
    for (uint32_t p = 0; p < n; p += B) {
        uint32_t b = n - p < B ? n - p : B;
        if (janas_llm_model_forward(m, tok + p, b, p, logits, 1) != 0)
            return 1;
        for (uint32_t j = 0; j < b; j++)
            main_next[p + j] = argmax(logits + (size_t)j * nv, nv);
        memcpy(hid + (size_t)(p + 1) * dm, janas_llm_model_hidden(m),
               (size_t)b * dm * sizeof(float));
    }
    /* MTP rows i = 0 .. n - 2: token t_{i+1}, hidden h_i (row i + 1 of
       hid; row 0 is zero), guessing t_{i+2}; with pos=t a first row pairs
       t_0 with the zero state at position 0 */
    uint32_t agree = 0, text = 0, rows = 0;
    uint32_t start = pos_t ? 0 : 1; /* first hid row used */
    for (uint32_t r = start; r < n; r += B) {
        uint32_t b = n - r < B ? n - r : B;
        uint32_t pos0 = pos_t ? r : r - 1;
        /* row k: hidden hid[r + k] = h_{r+k-1}, token t_{r+k} */
        if (janas_llm_mtp_forward(m, tok + r, hid + (size_t)r * dm, b, pos0,
                                  logits, 1) != 0)
            return 1;
        for (uint32_t k = 0; k < b; k++) {
            uint32_t i = r + k; /* guesses t_{i+1} */
            if (i < 1 || i + 1 >= n)
                continue;
            uint32_t g = argmax(logits + (size_t)k * nv, nv);
            agree += g == main_next[i];
            text += g == (uint32_t)tok[i + 1];
            rows++;
        }
    }
    printf("MTP %s: agrees with the main model %u/%u (%.1f%%), with the "
           "text %u/%u (%.1f%%)\n",
           pos_t ? "pos=t" : "pos=h", agree, rows, 100.0 * agree / rows, text,
           rows, 100.0 * text / rows);
    free(logits);
    free(hid);
    free(main_next);
    janas_llm_model_free(m);
    return 0;
}
