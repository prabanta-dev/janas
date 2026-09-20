/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jns_planes.c - rewrite a model file with the experts' down matrix cut into
 * three two-bit planes (JNS version 3, see jns.h and quant.h).
 *
 * Nothing is lost: with all three planes every weight is the Q6_K one, bit
 * for bit. What changes is the order of the bytes inside an expert slot, so
 * that a machine short of memory can read a prefix of it - the base plane
 * alone (two bits per weight), or two planes (four) - instead of the whole
 * slot. Layers whose down matrix is not Q6_K are copied unchanged.
 *
 * Usage: jns_planes <in.jns> <out.jns>
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "llm/jns.h"
#include "llm/quant.h"

#define ALIGN JANAS_JNS_ALIGN

static uint64_t align_up(uint64_t v, uint64_t a)
{
    return (v + a - 1) / a * a;
}

/* Expert slots are sorted by the offset they have in the source file. */
static const uint64_t *sort_key;

static int by_offset(const void *a, const void *b)
{
    uint64_t x = sort_key[*(const uint32_t *)a];
    uint64_t y = sort_key[*(const uint32_t *)b];
    return x < y ? -1 : x > y ? 1 : 0;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int write_at(int fd, const void *buf, size_t n, uint64_t off)
{
    const char *p = buf;
    while (n) {
        ssize_t w = pwrite(fd, p, n, (off_t)off);
        if (w <= 0)
            return -1;
        p += w;
        n -= (size_t)w;
        off += (uint64_t)w;
    }
    return 0;
}

static int read_at(int fd, void *buf, size_t n, uint64_t off)
{
    char *p = buf;
    while (n) {
        ssize_t r = pread(fd, p, n, (off_t)off);
        if (r <= 0)
            return -1;
        p += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return 0;
}

/* Copies n bytes between two files, through a buffer. */
static int copy_region(int in, uint64_t src, int out, uint64_t dst, uint64_t n)
{
    enum { CHUNK = 8u << 20 };
    char *buf = malloc(CHUNK);
    if (!buf)
        return -1;
    while (n) {
        size_t c = n < CHUNK ? (size_t)n : CHUNK;
        if (read_at(in, buf, c, src) || write_at(out, buf, c, dst)) {
            free(buf);
            return -1;
        }
        src += c;
        dst += c;
        n -= c;
    }
    free(buf);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: jns_planes <in.jns> <out.jns>\n");
        return 2;
    }
    struct janas_jns j;
    char err[256];
    if (janas_jns_open(argv[1], &j, err, sizeof(err)) != 0) {
        fprintf(stderr, "%s: %s\n", argv[1], err);
        return 1;
    }
    uint32_t n_layer = j.h.n_layer, n_expert = j.h.n_expert;
    uint64_t n_slots = (uint64_t)n_layer * n_expert;

    /* the new layer table: gate, up and the base plane first, then the other
       two planes, each level padded to a whole number of pages */
    struct janas_jns_layer *ly = calloc(n_layer, sizeof(*ly));
    uint64_t *new_off = malloc(n_slots * sizeof(uint64_t));
    uint64_t *new_sum = malloc(n_slots * sizeof(uint64_t));
    uint32_t *order = malloc(n_slots * sizeof(uint32_t));
    if (!ly || !new_off || !new_sum || !order)
        return 1;
    uint32_t planed = 0;
    for (uint32_t l = 0; l < n_layer; l++) {
        const struct janas_jns_layer *src = &j.layers[l];
        ly[l] = *src;
        memset(ly[l].plane, 0, sizeof(ly[l].plane));
        const struct janas_jns_matrix *dn = &src->m[JANAS_JNS_DOWN];
        if (dn->type != JANAS_Q6_K) { /* nothing to cut: copied as it is */
            for (int i = 0; i < 3; i++)
                ly[l].level_bytes[i] = src->slot_bytes;
            continue;
        }
        planed++;
        uint64_t nb = (uint64_t)dn->rows * (dn->cols / JANAS_QK);
        uint64_t off = 0;
        for (int m = 0; m < 3; m++) {
            ly[l].m[m].offset = off;
            if (m == JANAS_JNS_DOWN) {
                ly[l].m[m].type = JANAS_Q6_K_P;
                ly[l].m[m].bytes = nb * sizeof(struct janas_block_q6kp);
            }
            off = align_up(off + ly[l].m[m].bytes, 256);
        }
        ly[l].level_bytes[0] = align_up(off, ALIGN);
        for (int p = 0; p < 2; p++) {
            ly[l].plane[p] = ly[l].m[JANAS_JNS_DOWN];
            ly[l].plane[p].offset = ly[l].level_bytes[p];
            ly[l].plane[p].bytes = nb * JANAS_Q6KP_PLANE;
            ly[l].level_bytes[p + 1] =
                align_up(ly[l].plane[p].offset + ly[l].plane[p].bytes, ALIGN);
        }
        ly[l].slot_bytes = ly[l].level_bytes[2];
    }

    /* header and tables: the layer table grew, so everything after it moves */
    struct janas_jns_header h = j.h;
    h.version = JANAS_JNS_VERSION;
    uint64_t layer_len = (uint64_t)n_layer * sizeof(struct janas_jns_layer);
    uint64_t slot_len = n_slots * sizeof(uint64_t);
    uint64_t dir_len = j.h.n_tensor * sizeof(struct janas_jns_tensor);
    h.layer_table_offset = ALIGN;
    h.expert_table_offset = h.layer_table_offset + layer_len;
    h.expert_sums_offset = h.expert_table_offset + slot_len;
    h.tensor_dir_offset = h.expert_sums_offset + slot_len;
    h.metadata_offset = h.tensor_dir_offset + dir_len;
    h.resident_offset = align_up(h.metadata_offset + h.metadata_bytes, ALIGN);
    h.experts_offset = align_up(h.resident_offset + h.resident_bytes, ALIGN);

    struct janas_jns_tensor *tensors =
        malloc(dir_len ? (size_t)dir_len : (size_t)1);
    if (!tensors)
        return 1;
    memcpy(tensors, j.tensors, (size_t)dir_len);
    int64_t shift = (int64_t)h.resident_offset - (int64_t)j.h.resident_offset;
    for (uint64_t t = 0; t < h.n_tensor; t++)
        tensors[t].offset = (uint64_t)((int64_t)tensors[t].offset + shift);

    /* the experts keep the order they have on disk (co-activation) */
    for (uint64_t i = 0; i < n_slots; i++)
        order[i] = (uint32_t)i;
    sort_key = j.expert_offset;
    qsort(order, (size_t)n_slots, sizeof(uint32_t), by_offset);

    uint64_t at = h.experts_offset;
    for (uint64_t i = 0; i < n_slots; i++) {
        uint32_t s = order[i];
        new_off[s] = at;
        at += ly[s / n_expert].slot_bytes;
    }
    h.experts_bytes = at - h.experts_offset;
    h.file_bytes = at;

    int out = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) {
        fprintf(stderr, "cannot create %s\n", argv[2]);
        return 1;
    }
    if (copy_region(j.fd, j.h.resident_offset, out, h.resident_offset,
                    h.resident_bytes) != 0) {
        fprintf(stderr, "cannot copy the resident weights\n");
        return 1;
    }

    /* the expert slots, one at a time: gate and up as they are, down cut */
    uint64_t max_slot = 0, max_src = 0;
    for (uint32_t l = 0; l < n_layer; l++) {
        if (ly[l].slot_bytes > max_slot)
            max_slot = ly[l].slot_bytes;
        if (j.layers[l].slot_bytes > max_src)
            max_src = j.layers[l].slot_bytes;
    }
    uint8_t *src = malloc(max_src), *dst = malloc(max_slot);
    if (!src || !dst)
        return 1;
    double t0 = now();
    for (uint64_t i = 0; i < n_slots; i++) {
        uint32_t s = order[i];
        uint32_t l = s / n_expert;
        const struct janas_jns_layer *so = &j.layers[l];
        memset(dst, 0, ly[l].slot_bytes);
        if (read_at(j.fd, src, so->slot_bytes, j.expert_offset[s]) != 0) {
            fprintf(stderr, "cannot read expert %u of layer %u\n", s % n_expert,
                    l);
            return 1;
        }
        if (ly[l].plane[0].type == 0) {
            memcpy(dst, src, so->slot_bytes);
        } else {
            for (int m = 0; m < 2; m++)
                memcpy(dst + ly[l].m[m].offset, src + so->m[m].offset,
                       so->m[m].bytes);
            uint64_t nb = ly[l].plane[0].bytes / JANAS_Q6KP_PLANE;
            janas_q6k_to_planes(
                (const struct janas_block_q6k *)(src +
                                                 so->m[JANAS_JNS_DOWN].offset),
                (size_t)nb,
                (struct janas_block_q6kp *)(dst +
                                            ly[l].m[JANAS_JNS_DOWN].offset),
                dst + ly[l].plane[0].offset, dst + ly[l].plane[1].offset);
        }
        new_sum[s] = janas_crc32(0, dst, (size_t)ly[l].slot_bytes);
        if (write_at(out, dst, (size_t)ly[l].slot_bytes, new_off[s]) != 0) {
            fprintf(stderr, "cannot write expert %u of layer %u\n",
                    s % n_expert, l);
            return 1;
        }
        if ((i & 1023) == 1023 || i + 1 == n_slots) {
            double dt = now() - t0;
            fprintf(stderr, "\r%5.1f%%  %.1f GB in %.0f s (%.0f MB/s)   ",
                    100.0 * (double)(i + 1) / (double)n_slots,
                    (double)(at - h.experts_offset) * (double)(i + 1) /
                        (double)n_slots / 1e9,
                    dt,
                    (double)(at - h.experts_offset) * (double)(i + 1) /
                        (double)n_slots / 1e6 / dt);
        }
    }
    fprintf(stderr, "\n");

    /* tables last, with the checksums over what was written */
    uint64_t sum = JANAS_FNV_INIT;
    sum = janas_fnv1a(ly, (size_t)layer_len, sum);
    sum = janas_fnv1a(new_off, (size_t)slot_len, sum);
    sum = janas_fnv1a(new_sum, (size_t)slot_len, sum);
    sum = janas_fnv1a(tensors, (size_t)dir_len, sum);
    h.tables_checksum = sum;
    h.metadata_checksum = janas_crc32(0, j.metadata, (size_t)h.metadata_bytes);
    h.header_checksum = 0;
    h.header_checksum = janas_fnv1a(&h, sizeof(h), JANAS_FNV_INIT);

    uint8_t head[ALIGN];
    memset(head, 0, sizeof(head));
    memcpy(head, &h, sizeof(h));
    if (write_at(out, head, sizeof(head), 0) ||
        write_at(out, ly, (size_t)layer_len, h.layer_table_offset) ||
        write_at(out, new_off, (size_t)slot_len, h.expert_table_offset) ||
        write_at(out, new_sum, (size_t)slot_len, h.expert_sums_offset) ||
        write_at(out, tensors, (size_t)dir_len, h.tensor_dir_offset) ||
        write_at(out, j.metadata, (size_t)h.metadata_bytes,
                 h.metadata_offset)) {
        fprintf(stderr, "cannot write the tables\n");
        return 1;
    }
    if (ftruncate(out, (off_t)h.file_bytes) != 0 || close(out) != 0) {
        fprintf(stderr, "cannot close %s\n", argv[2]);
        return 1;
    }
    printf("%s: %u layers, %u of them with the down matrix in planes, "
           "%.1f GB\n",
           argv[2], n_layer, planed, (double)h.file_bytes / 1e9);
    printf("levels per expert slot: %.3f / %.3f / %.3f MB\n",
           (double)ly[0].level_bytes[0] / 1e6,
           (double)ly[0].level_bytes[1] / 1e6,
           (double)ly[0].level_bytes[2] / 1e6);
    free(src);
    free(dst);
    free(ly);
    free(new_off);
    free(new_sum);
    free(order);
    free(tensors);
    janas_jns_close(&j);
    return 0;
}
