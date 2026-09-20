/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jns_check.c - validate a JNS model file, optionally verifying every expert.
 *
 * Usage: jns_check <model.jns> [--verify]
 * With --verify every expert slot is read with O_DIRECT (the engine's I/O
 * path) and its CRC-32 compared with the table; the read rate is reported.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "llm/jns.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: jns_check <model.jns> [--verify]\n");
        return 2;
    }
    struct janas_jns j;
    char err[256];
    if (janas_jns_open(argv[1], &j, err, sizeof(err)) != 0) {
        fprintf(stderr, "%s: %s\n", argv[1], err);
        return 1;
    }
    const struct janas_jns_header *h = &j.h;
    printf("%s: arch %.32s, %u layers, %u experts (k %u), d_model %u\n",
           argv[1], h->arch, h->n_layer, h->n_expert, h->n_expert_used,
           h->d_model);
    printf("resident %.3f GB in %llu tensors, experts %.3f GB\n",
           h->resident_bytes / 1e9, (unsigned long long)h->n_tensor,
           h->experts_bytes / 1e9);
    char key[96], tok[64] = "?";
    float eps = 0, rope = 0;
    uint32_t heads = 0, kv_heads = 0;
    snprintf(key, sizeof(key), "%.32s.attention.layer_norm_rms_epsilon",
             h->arch);
    janas_jns_meta_f32(&j, key, &eps);
    snprintf(key, sizeof(key), "%.32s.rope.freq_base", h->arch);
    janas_jns_meta_f32(&j, key, &rope);
    snprintf(key, sizeof(key), "%.32s.attention.head_count", h->arch);
    janas_jns_meta_u32(&j, key, &heads);
    snprintf(key, sizeof(key), "%.32s.attention.head_count_kv", h->arch);
    janas_jns_meta_u32(&j, key, &kv_heads);
    janas_jns_meta_str(&j, "tokenizer.ggml.model", tok, sizeof(tok));
    printf("metadata %.2f MB: rms eps %g, rope base %g, heads %u/%u, "
           "tokenizer %s\n",
           h->metadata_bytes / 1e6, eps, rope, heads, kv_heads, tok);
    const struct janas_jns_layer *l0 = &j.layers[0];
    printf("layer 0 slot %.3f MB: gate type %u %ux%u, up type %u, down type "
           "%u %ux%u\n",
           l0->slot_bytes / 1e6, l0->m[0].type, l0->m[0].rows, l0->m[0].cols,
           l0->m[1].type, l0->m[2].type, l0->m[2].rows, l0->m[2].cols);
    if (j.zero_levels)
        printf("note: %u layer(s) with their levels written as zero, as "
               "hf2jns_mtp did before issue #6: read as the whole slot, and "
               "the file works as it is; converting it again with the "
               "current hf2jns_mtp is optional\n",
               j.zero_levels);

    int status = 0;
    if (argc > 2 && strcmp(argv[2], "--verify") == 0) {
        int fd = open(argv[1], O_RDONLY | O_DIRECT);
        uint64_t max_slot = 0;
        for (uint32_t l = 0; l < h->n_layer; l++)
            if (j.layers[l].slot_bytes > max_slot)
                max_slot = j.layers[l].slot_bytes;
        void *buf = NULL;
        if (fd < 0 || posix_memalign(&buf, JANAS_JNS_ALIGN, max_slot) != 0) {
            fprintf(stderr, "cannot open with O_DIRECT\n");
            return 1;
        }
        size_t bad = 0;
        double bytes = 0, t0 = now();
        for (uint32_t l = 0; l < h->n_layer; l++)
            for (uint32_t e = 0; e < h->n_expert; e++) {
                uint64_t n = j.layers[l].slot_bytes;
                if (pread(fd, buf, n,
                          (off_t)janas_jns_expert_offset(&j, l, e)) !=
                    (ssize_t)n) {
                    fprintf(stderr, "read error at layer %u expert %u\n", l, e);
                    return 1;
                }
                bytes += (double)n;
                if (janas_crc32(0, buf, n) !=
                    j.expert_sum[(size_t)l * h->n_expert + e])
                    bad++;
            }
        double dt = now() - t0;
        printf("verify: %zu of %llu experts with a wrong checksum, %.2f GB in "
               "%.1f s (%.2f GB/s, one read at a time, CRC on one core)\n",
               bad, (unsigned long long)h->n_layer * h->n_expert, bytes / 1e9,
               dt, bytes / 1e9 / dt);
        status = bad ? 1 : 0;
        free(buf);
        close(fd);
    }
    janas_jns_close(&j);
    return status;
}
