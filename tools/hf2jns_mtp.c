/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf2jns_mtp.c - the multi-token prediction (MTP) block of Qwen3-Next, or of
 * a dense Qwen3.5, from the original safetensors checkpoint into a one-layer
 * JNS file.
 *
 * GGUF conversions of these models drop the MTP block; the checkpoint keeps
 * it under mtp.* (bf16). This tool quantizes the matrices to Q4_K with
 * Janas's own quantizer, writes the routed experts as expert slots (so the
 * engine streams them through an expert cache, like every other layer) -
 * for a dense model, its feed-forward as the one slot, as gguf2jns does -
 * and everything else as resident tensors, named like the main model's
 * layer tensors ("blk.0.*") plus "mtp.*" for the block's own parts. The
 * metadata region is copied from the main model's JNS: the hyperparameters
 * are the same. Whether the model is dense, and how many experts it has,
 * is the main model's; the shapes are the checkpoint's.
 *
 * Conventions checked on the weights themselves: the checkpoint's RMS norms
 * are zero-centred (the effective scale is 1 + w; its norms have negative
 * means, and the main model's norms in the GGUF equal the checkpoint's plus
 * 1 exactly), and the projections keep the checkpoint's row order (every
 * row of attn_q and attn_k correlates 0.997 with the same checkpoint row).
 * The same holds for Qwen3.5-9B: its GGUF norms are the checkpoint's plus 1,
 * to the bit.
 *
 * Usage: hf2jns_mtp <main.jns> <shard.safetensors>... <out.jns>
 *        hf2jns_mtp <main.jns> hf:<owner>/<repo>[@revision] <out.jns>
 * The block may be spread over several shards: give every one that has a
 * part of it (or one file holding just the block's tensors). With hf: the
 * block's tensors alone are read from the repository on Hugging Face, with
 * range requests - Qwen3.5-9B's 487 MB instead of its 14 GB of shards -
 * into <out.jns>.mtp.safetensors, deleted once converted.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/hf_fetch.h"
#include "common/pool.h"
#include "llm/jns.h"
#include "llm/quant.h"

#define MATRIX_ALIGN 256

static uint64_t align_up(uint64_t x, uint64_t a)
{
    return (x + a - 1) / a * a;
}

static _Noreturn void die(const char *msg, const char *arg)
{
    fprintf(stderr, "hf2jns_mtp: %s%s%s\n", msg, arg ? ": " : "",
            arg ? arg : "");
    exit(1);
}

/* ---- safetensors: an 8-byte header length, a JSON header, the data ---- */

struct st_file {
    FILE *f;
    char *json;
    uint64_t data_start;
};

struct st_tensor {
    int dims;
    uint64_t shape[4];
    uint64_t begin, end; /* data offsets */
};

static void st_open(struct st_file *s, const char *path)
{
    s->f = fopen(path, "rb");
    uint64_t n;
    if (!s->f || fread(&n, 8, 1, s->f) != 1 || n > (1u << 30))
        die("cannot read", path);
    s->json = malloc(n + 1);
    if (!s->json || fread(s->json, 1, n, s->f) != n)
        die("cannot read the header of", path);
    s->json[n] = 0;
    s->data_start = 8 + n;
}

/* Parses the entry of one tensor: {"dtype":"BF16","shape":[..],
   "data_offsets":[b,e]}. Only bf16 is accepted. */
static int st_find(const struct st_file *s, const char *name,
                   struct st_tensor *t)
{
    char key[256];
    snprintf(key, sizeof(key), "\"%s\"", name);
    const char *p = strstr(s->json, key);
    if (!p)
        return -1;
    const char *obj = strchr(p + strlen(key), '{');
    const char *end = obj ? strchr(obj, '}') : NULL;
    if (!obj || !end)
        return -1;
    const char *dt = strstr(obj, "\"dtype\"");
    const char *sh = strstr(obj, "\"shape\"");
    const char *of = strstr(obj, "\"data_offsets\"");
    if (!dt || !sh || !of || dt > end || sh > end || of > end ||
        !strstr(dt, "\"BF16\"") || strstr(dt, "\"BF16\"") > end)
        return -1;
    sh = strchr(sh, '[');
    t->dims = 0;
    for (const char *q = sh + 1; *q && *q != ']' && t->dims < 4;) {
        t->shape[t->dims++] = strtoull(q, (char **)&q, 10);
        while (*q == ',' || *q == ' ')
            q++;
    }
    of = strchr(of, '[');
    char *q;
    t->begin = strtoull(of + 1, &q, 10);
    while (*q == ',' || *q == ' ')
        q++;
    t->end = strtoull(q, NULL, 10);
    uint64_t count = 1;
    for (int i = 0; i < t->dims; i++)
        count *= t->shape[i];
    return t->end - t->begin == 2 * count ? 0 : -1;
}

/* The shards given; a tensor is looked for in each. */
static struct st_file shards[64];
static int n_shards;

/* The tensor as float32 (malloc'd), with its element count. */
static float *st_load(const char *name, uint64_t *n, struct st_tensor *t)
{
    const struct st_file *s = NULL;
    for (int i = 0; i < n_shards && !s; i++)
        if (st_find(&shards[i], name, t) == 0)
            s = &shards[i];
    if (!s)
        die("tensor missing or not bf16", name);
    *n = (t->end - t->begin) / 2;
    uint16_t *raw = malloc(*n * 2);
    float *x = malloc(*n * sizeof(float));
    if (!raw || !x || fseeko(s->f, (off_t)(s->data_start + t->begin), 0) ||
        fread(raw, 2, *n, s->f) != *n)
        die("cannot read", name);
    for (uint64_t i = 0; i < *n; i++) {
        uint32_t u = (uint32_t)raw[i] << 16;
        memcpy(&x[i], &u, 4);
    }
    free(raw);
    return x;
}

/* ---- tensors of the output ---- */

enum { T_F32 = 0, T_F16 = 1, T_Q4K = JANAS_Q4_K };

struct out_tensor {
    char name[64];
    uint32_t type, n_dims;
    uint64_t dims[4];
    uint8_t *data;
    uint64_t bytes, offset;
};

static struct out_tensor outs[64];
static int n_outs;

/* Rows x cols matrix to Q4_K, rows split over the pool. */
struct quant_job {
    const float *x;
    struct janas_block_q4k *y;
    uint64_t rows, cols;
};

static void quant_worker(void *arg, int tid, int n_threads)
{
    struct quant_job *q = arg;
    for (uint64_t r = (uint64_t)tid; r < q->rows; r += (uint64_t)n_threads)
        janas_q4k_quantize(q->x + r * q->cols, q->y + r * (q->cols / JANAS_QK),
                           q->cols);
}

static uint8_t *to_q4k(struct janas_pool *pool, const float *x, uint64_t rows,
                       uint64_t cols, uint64_t *bytes)
{
    if (cols % JANAS_QK)
        die("row length not a multiple of 256", NULL);
    *bytes = rows * (cols / JANAS_QK) * sizeof(struct janas_block_q4k);
    struct janas_block_q4k *y = malloc(*bytes);
    if (!y)
        die("out of memory", NULL);
    struct quant_job q = {x, y, rows, cols};
    janas_pool_run(pool, quant_worker, &q);
    return (uint8_t *)y;
}

/* Adds a resident tensor from the checkpoint. kind: T_F32 (norms get
   +1 when norm is set), T_F16 or T_Q4K. Dims in ggml order. */
static void add(struct janas_pool *pool, const char *src, const char *dst,
                uint32_t kind, int norm)
{
    struct st_tensor t;
    uint64_t n;
    float *x = st_load(src, &n, &t);
    struct out_tensor *o = &outs[n_outs++];
    snprintf(o->name, sizeof(o->name), "%s", dst);
    o->type = kind;
    /* checkpoint shape [rows, cols] -> ggml dims {cols, rows} */
    o->n_dims = t.dims == 2 && t.shape[0] > 1 ? 2 : 1;
    o->dims[0] = t.shape[t.dims - 1];
    o->dims[1] = o->n_dims == 2 ? t.shape[0] : 1;
    o->dims[2] = o->dims[3] = 1;
    if (kind == T_Q4K) {
        o->data = to_q4k(pool, x, o->dims[1], o->dims[0], &o->bytes);
    } else if (kind == T_F16) {
        uint16_t *h = malloc(n * 2);
        for (uint64_t i = 0; i < n; i++)
            h[i] = janas_fp32_to_fp16(x[i]);
        o->data = (uint8_t *)h;
        o->bytes = n * 2;
    } else {
        if (norm)
            for (uint64_t i = 0; i < n; i++)
                x[i] += 1.0f;
        o->data = malloc(n * 4);
        memcpy(o->data, x, n * 4);
        o->bytes = n * 4;
    }
    free(x);
    printf("  %-34s <- %s\n", dst, src);
}

/* The checkpoint's name of a feed-forward matrix of the block: expert e's,
   or the dense one's. */
static void slot_name(char *nm, size_t len, const char *layer, int dense,
                      uint32_t e, const char *part)
{
    if (dense)
        snprintf(nm, len, "%smlp.%s.weight", layer, part);
    else
        snprintf(nm, len, "%smlp.experts.%u.%s.weight", layer, e, part);
}

/* The shape of a tensor, from whichever shard has it. */
static int shape_of(const char *name, struct st_tensor *t)
{
    for (int i = 0; i < n_shards; i++)
        if (st_find(&shards[i], name, t) == 0)
            return 1;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4 || argc - 3 > (int)(sizeof(shards) / sizeof(shards[0]))) {
        fprintf(stderr, "usage: hf2jns_mtp <main.jns> <shard.safetensors>... "
                        "<out.jns>\n");
        return 2;
    }
    const char *out_path = argv[argc - 1];
    struct janas_jns main_j;
    char err[256];
    if (janas_jns_open(argv[1], &main_j, err, sizeof(err)) != 0)
        die(err, argv[1]);
    char fetched[4096] = "";
    for (int i = 2; i < argc - 1; i++) {
        if (strncmp(argv[i], "hf:", 3) == 0) {
            if (*fetched)
                die("one hf: repository at a time", argv[i]);
            char repo[512], *rev;
            snprintf(repo, sizeof(repo), "%s", argv[i] + 3);
            if ((rev = strchr(repo, '@')))
                *rev++ = 0;
            snprintf(fetched, sizeof(fetched), "%s.mtp.safetensors", out_path);
            if (janas_hf_fetch_tensors(repo, rev, "mtp.", fetched, err,
                                       sizeof(err)) < 0) {
                remove(fetched);
                die(err, NULL);
            }
            st_open(&shards[n_shards++], fetched);
        } else {
            st_open(&shards[n_shards++], argv[i]);
        }
    }
    /* a dense model is one expert a layer with no router (gguf2jns) */
    const uint32_t n_exp = (uint32_t)main_j.h.n_expert;
    const int dense = n_exp == 1;
    int cpus[64], nt = 12;
    for (int i = 0; i < nt; i++)
        cpus[i] = i;
    struct janas_pool *pool = janas_pool_create(nt, cpus);

    const char *L = "mtp.layers.0.";
    char a[160];
#define SRC(x) (snprintf(a, sizeof(a), "%s%s", L, x), a)
    add(pool, SRC("input_layernorm.weight"), "blk.0.attn_norm.weight", T_F32,
        1);
    add(pool, SRC("self_attn.q_proj.weight"), "blk.0.attn_q.weight", T_Q4K, 0);
    add(pool, SRC("self_attn.k_proj.weight"), "blk.0.attn_k.weight", T_Q4K, 0);
    add(pool, SRC("self_attn.v_proj.weight"), "blk.0.attn_v.weight", T_Q4K, 0);
    add(pool, SRC("self_attn.o_proj.weight"), "blk.0.attn_output.weight", T_Q4K,
        0);
    add(pool, SRC("self_attn.q_norm.weight"), "blk.0.attn_q_norm.weight", T_F32,
        1);
    add(pool, SRC("self_attn.k_norm.weight"), "blk.0.attn_k_norm.weight", T_F32,
        1);
    add(pool, SRC("post_attention_layernorm.weight"),
        "blk.0.post_attention_norm.weight", T_F32, 1);
    if (!dense) {
        add(pool, SRC("mlp.gate.weight"), "blk.0.ffn_gate_inp.weight", T_F32,
            0);
        add(pool, SRC("mlp.shared_expert.gate_proj.weight"),
            "blk.0.ffn_gate_shexp.weight", T_Q4K, 0);
        add(pool, SRC("mlp.shared_expert.up_proj.weight"),
            "blk.0.ffn_up_shexp.weight", T_Q4K, 0);
        add(pool, SRC("mlp.shared_expert.down_proj.weight"),
            "blk.0.ffn_down_shexp.weight", T_Q4K, 0);
        add(pool, SRC("mlp.shared_expert_gate.weight"),
            "blk.0.ffn_gate_inp_shexp.weight", T_F16, 0);
    }
    add(pool, "mtp.fc.weight", "mtp.eh_proj.weight", T_Q4K, 0);
    add(pool, "mtp.pre_fc_norm_embedding.weight", "mtp.enorm.weight", T_F32, 1);
    add(pool, "mtp.pre_fc_norm_hidden.weight", "mtp.hnorm.weight", T_F32, 1);
    add(pool, "mtp.norm.weight", "mtp.norm.weight", T_F32, 1);

    /* expert slot: gate, up, down in Q4_K, shaped as the checkpoint's first
       expert (Qwen3-Next: 512 x 2048, 512 x 2048, 2048 x 512), or as its
       dense feed-forward */
    struct janas_jns_layer ly = {0};
    uint64_t off = 0;
    const char *parts[3] = {"gate_proj", "up_proj", "down_proj"};
    uint32_t rows[3], cols[3];
    for (int p = 0; p < 3; p++) {
        char nm[160];
        slot_name(nm, sizeof(nm), L, dense, 0, parts[p]);
        struct st_tensor t;
        if (n_shards == 0 || !shape_of(nm, &t) || t.dims != 2)
            die("feed-forward matrix missing or not 2-D", nm);
        rows[p] = (uint32_t)t.shape[0];
        cols[p] = (uint32_t)t.shape[1];
    }
    for (int p = 0; p < 3; p++) {
        off = align_up(off, MATRIX_ALIGN);
        ly.m[p] = (struct janas_jns_matrix){
            .type = JANAS_Q4_K,
            .rows = rows[p],
            .cols = cols[p],
            .offset = off,
            .bytes = (uint64_t)rows[p] * cols[p] / JANAS_QK *
                     sizeof(struct janas_block_q4k)};
        off += ly.m[p].bytes;
    }
    ly.slot_bytes = align_up(off, JANAS_JNS_ALIGN);
    /* no bit planes: the three levels of a version 3 slot are all of it
       (left at zero, the file was refused: issue #6) */
    for (int i = 0; i < 3; i++)
        ly.level_bytes[i] = ly.slot_bytes;

    /* layout, as tools/gguf2jns.c */
    uint64_t A = JANAS_JNS_ALIGN, layer_table = A;
    uint64_t expert_table = layer_table + sizeof(struct janas_jns_layer);
    uint64_t expert_sums = expert_table + 8 * (uint64_t)n_exp;
    uint64_t tensor_dir = expert_sums + 8 * (uint64_t)n_exp;
    uint64_t meta_off = align_up(
        tensor_dir + sizeof(struct janas_jns_tensor) * (uint64_t)n_outs, A);
    uint64_t meta_bytes = main_j.h.metadata_bytes;
    uint64_t resident_off = align_up(meta_off + meta_bytes, A), pos;
    pos = resident_off;
    for (int i = 0; i < n_outs; i++) {
        outs[i].offset = pos;
        pos = align_up(pos + outs[i].bytes, A);
    }
    uint64_t resident_bytes = pos - resident_off, experts_off = pos;
    uint64_t *slot_off = calloc(n_exp, 8), *sums = calloc(n_exp, 8);
    if (!slot_off || !sums)
        die("out of memory", NULL);
    for (uint32_t e = 0; e < n_exp; e++) {
        slot_off[e] = pos;
        pos += ly.slot_bytes;
    }
    uint64_t file_bytes = pos;

    FILE *out = fopen(out_path, "w+b");
    if (!out || ftruncate(fileno(out), (off_t)file_bytes))
        die("cannot create", out_path);
    for (int i = 0; i < n_outs; i++)
        if (fseeko(out, (off_t)outs[i].offset, 0) ||
            fwrite(outs[i].data, 1, outs[i].bytes, out) != outs[i].bytes)
            die("write failed", out_path);

    /* experts, one slot each, with the CRC-32 of the whole slot */
    uint8_t *slot = calloc(1, ly.slot_bytes);
    for (uint32_t e = 0; e < n_exp; e++) {
        memset(slot, 0, ly.slot_bytes);
        for (int p = 0; p < 3; p++) {
            char nm[160];
            slot_name(nm, sizeof(nm), L, dense, e, parts[p]);
            struct st_tensor t;
            uint64_t n, bytes;
            float *x = st_load(nm, &n, &t);
            if (t.dims != 2 || t.shape[0] != rows[p] || t.shape[1] != cols[p])
                die("unexpected expert shape", nm);
            uint8_t *q = to_q4k(pool, x, rows[p], cols[p], &bytes);
            memcpy(slot + ly.m[p].offset, q, bytes);
            free(q);
            free(x);
        }
        sums[e] = janas_crc32(0, slot, ly.slot_bytes);
        if (fseeko(out, (off_t)slot_off[e], 0) ||
            fwrite(slot, 1, ly.slot_bytes, out) != ly.slot_bytes)
            die("write failed", out_path);
        if ((e + 1) % 64 == 0)
            printf("  experts %u/%u\n", e + 1, n_exp);
    }
    free(slot);

    /* tables, in file order, and their checksum */
    uint64_t tbl_len =
        tensor_dir + sizeof(struct janas_jns_tensor) * n_outs - layer_table;
    uint8_t *tbl = calloc(1, tbl_len);
    memcpy(tbl, &ly, sizeof(ly));
    memcpy(tbl + (expert_table - layer_table), slot_off, 8 * (size_t)n_exp);
    memcpy(tbl + (expert_sums - layer_table), sums, 8 * (size_t)n_exp);
    for (int i = 0; i < n_outs; i++) {
        struct janas_jns_tensor t = {0};
        memcpy(t.name, outs[i].name, sizeof(t.name));
        t.type = outs[i].type;
        t.n_dims = outs[i].n_dims;
        memcpy(t.dims, outs[i].dims, sizeof(t.dims));
        t.offset = outs[i].offset;
        t.bytes = outs[i].bytes;
        memcpy(tbl + (tensor_dir - layer_table) + i * sizeof(t), &t, sizeof(t));
    }
    fseeko(out, (off_t)layer_table, 0);
    fwrite(tbl, 1, tbl_len, out);
    fseeko(out, (off_t)meta_off, 0);
    fwrite(main_j.metadata, 1, meta_bytes, out);

    struct janas_jns_header h = {0};
    memcpy(h.magic, JANAS_JNS_MAGIC, 8);
    h.version = JANAS_JNS_VERSION;
    h.header_bytes = JANAS_JNS_ALIGN;
    h.file_bytes = file_bytes;
    memcpy(h.arch, main_j.h.arch, sizeof(h.arch));
    h.n_layer = 1;
    h.n_expert = n_exp;
    h.n_expert_used = main_j.h.n_expert_used;
    h.d_model = main_j.h.d_model;
    h.layer_table_offset = layer_table;
    h.expert_table_offset = expert_table;
    h.expert_sums_offset = expert_sums;
    h.tensor_dir_offset = tensor_dir;
    h.n_tensor = (uint64_t)n_outs;
    h.metadata_offset = meta_off;
    h.metadata_bytes = meta_bytes;
    h.resident_offset = resident_off;
    h.resident_bytes = resident_bytes;
    h.experts_offset = experts_off;
    h.experts_bytes = file_bytes - experts_off;
    h.tables_checksum = janas_fnv1a(tbl, tbl_len, JANAS_FNV_INIT);
    h.metadata_checksum = janas_crc32(0, main_j.metadata, meta_bytes);
    h.header_checksum = janas_fnv1a(&h, sizeof(h), JANAS_FNV_INIT);
    fseeko(out, 0, 0);
    fwrite(&h, 1, sizeof(h), out);
    if (fclose(out) != 0)
        die("write failed", out_path);
    printf("written %s: %.2f GB (resident %.1f MB, experts %.1f MB)\n",
           out_path, file_bytes / 1e9, resident_bytes / 1e6,
           (file_bytes - experts_off) / 1e6);
    free(tbl);
    free(slot_off);
    free(sums);
    for (int i = 0; i < n_outs; i++)
        free(outs[i].data);
    janas_pool_destroy(pool);
    janas_jns_close(&main_j);
    if (*fetched)
        remove(fetched);
    return 0;
}
