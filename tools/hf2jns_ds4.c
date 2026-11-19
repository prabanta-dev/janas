/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf2jns_ds4.c - DeepSeek-V4-Flash, from deepseek-ai's own checkpoint on
 * Hugging Face, into a JNS file (src/llm/jns.h) of architecture deepseek4,
 * its tensors named as llama.cpp's converter names them in a GGUF, so that
 * the engine reads the same file from the checkpoint and, through gguf2jns,
 * from a GGUF of it.
 *
 * Usage: hf2jns_ds4 hf:<owner>/<repo>[@revision] <out.jns>
 *
 * Nothing is stored but the JNS: the checkpoint's shards are read from the
 * Hub by ranges - runs of contiguous tensors, 64 MB at most a request - and
 * every byte of each shard goes through SHA-256, checked against the one the
 * Hub gives for it. Its 160 GB never sit on the disk beside the file made
 * from them.
 *
 * The weights:
 *  - the routed experts, FP4 (E2M1) two a byte with a UE8M0 scale per 32,
 *    are MXFP4 already: the same codes and the same scale byte, the nibbles
 *    only reordered (two consecutive weights a byte in the checkpoint; the
 *    first 16 of a block in the low nibbles and the next 16 in the high ones
 *    in ggml's MXFP4) - the same weights to the bit, one slot per (layer,
 *    expert): w1 the gate, w3 the up, w2 the down;
 *  - the dense matrices, FP8 E4M3 with a UE8M0 scale per 128 x 128 block,
 *    become Q8_0 (blocks of 32, fp16 scale, rounded to nearest), from the
 *    exact value of each weight - as llama.cpp's converter writes them;
 *  - the token embeddings and the head, BF16, become Q8_0 too, as in
 *    llama.cpp's Q8_0 files: the engine's products and its other heads
 *    (drafts, a subset of the vocabulary) read them as any other; the
 *    router becomes F32, which the engine turns into fp16 at load as every
 *    mixture's; the other BF16 matrices (the compressors, the indexer's
 *    head weights) stay BF16; the norms become F32, F32 stays F32, the hash
 *    layers' token-to-expert tables (I64) become I32;
 *  - the multi-token prediction layer (mtp.*) is left out: a file of its own,
 *    later.
 * The metadata are llama.cpp's deepseek4 keys, from config.json, and the
 * tokenizer from tokenizer.json: byte-level BPE (pre-tokenizer joyai-llm, as
 * llama.cpp names DeepSeek-V4's), the vocabulary, the added
 * tokens (special ones as control tokens), the merges.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/hf_fetch.h"
#include "common/sha256.h"
#include "llm/jns.h"
#include "llm/json.h"
#include "llm/quant.h"

#define ALIGN 4096
#define MATRIX_ALIGN 256
#define BATCH ((int64_t)64 << 20)
#define MAX_SHARDS 128
#define N_LAYER_MAX 64

/* ggml's types written */
enum { T_F32 = 0, T_Q8_0 = 8, T_I32 = 26, T_BF16 = 30, T_MXFP4 = 39 };
/* the checkpoint's dtypes */
enum { D_F32, D_BF16, D_F8, D_E8M0, D_I8, D_I64, D_OTHER };
/* GGUF's metadata types */
enum { G_U32 = 4, G_I32 = 5, G_F32 = 6, G_BOOL = 7, G_STR = 8, G_ARR = 9 };

static _Noreturn void die(const char *fmt, const char *arg)
{
    fprintf(stderr, "hf2jns_ds4: ");
    fprintf(stderr, fmt, arg ? arg : "");
    fprintf(stderr, "\n");
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p)
        die("out of memory%s", "");
    return p;
}

static uint64_t align_up(uint64_t x, uint64_t a)
{
    return (x + a - 1) / a * a;
}

/* ---- the repository ---- */

static char repo[256], rev[128];

static void fetch(const char *path, const int64_t *from, const int64_t *to,
                  struct janas_buf *out)
{
    char err[300];
    out->n = 0;
    if (janas_hf_get(repo, rev[0] ? rev : NULL, path, from, to, out, err,
                     sizeof err) != 0 ||
        out->oom)
        die("%s", err);
}

static struct janas_json_doc *fetch_json(const char *path)
{
    struct janas_buf b = {0};
    char err[200];
    fetch(path, NULL, NULL, &b);
    struct janas_json_doc *d = janas_json_parse(b.p, b.n, err, sizeof err);
    if (!d)
        die("%s: not JSON", path);
    /* the strings point into the document's own copy */
    janas_buf_free(&b);
    return d;
}

/* ---- the checkpoint's tensors ---- */

struct shard {
    char path[128];
    uint64_t data; /* where its tensors' bytes start */
    uint64_t size;
    char sha[65];
    struct janas_buf head; /* its first 8 + header bytes, for the hash */
    char url[4096];        /* where the Hub sent the last request, or "" */
};

struct src {
    char name[128];
    int shard, dtype, nd;
    uint64_t shape[4];
    uint64_t off, bytes; /* in the shard, absolute */
};

static struct shard shards[MAX_SHARDS];
static int n_shards;
static struct src *srcs;
static size_t n_srcs, cap_srcs;

static int dtype_of(const char *s)
{
    return !strcmp(s, "F32")       ? D_F32
           : !strcmp(s, "BF16")    ? D_BF16
           : !strcmp(s, "F8_E4M3") ? D_F8
           : !strcmp(s, "F8_E8M0") ? D_E8M0
           : !strcmp(s, "I8")      ? D_I8
           : !strcmp(s, "I64")     ? D_I64
                                   : D_OTHER;
}

/* bytes [from, to) of a shard: from the address on the CDN the last
   request ended at, found again through the Hub when it fails (signed
   addresses expire), each range tried five times */
static void fetch_range(int si, int64_t from, int64_t to, struct janas_buf *out)
{
    struct shard *s = &shards[si];
    char err[300] = "";
    for (int attempt = 0; attempt < 5; attempt++) {
        char start[4096];
        if (s->url[0])
            snprintf(start, sizeof start, "%s", s->url);
        else
            snprintf(start, sizeof start,
                     "https://huggingface.co/%s/resolve/%s/%s", repo,
                     rev[0] ? rev : "main", s->path);
        out->n = 0;
        if (janas_hf_get_url(start, &from, &to, out, s->url, sizeof s->url, err,
                             sizeof err) == 0 &&
            !out->oom && (int64_t)out->n == to - from)
            return;
        s->url[0] = 0;
        if (out->oom)
            break;
        fprintf(stderr, "\n%s: %s, trying again\n", s->path, err);
    }
    die("%s", err[0] ? err : "a range came short");
}

static void read_header(int si)
{
    struct shard *s = &shards[si];
    char err[200];
    fetch_range(si, 0, 8, &s->head);
    uint64_t hn;
    memcpy(&hn, s->head.p, 8);
    if (hn > (64u << 20))
        die("%s: a header too large", s->path);
    struct janas_buf h = {0};
    fetch_range(si, 8, 8 + (int64_t)hn, &h);
    janas_buf_put(&s->head, h.p, h.n);
    s->data = 8 + hn;
    struct janas_json_doc *d = janas_json_parse(h.p, h.n, err, sizeof err);
    if (!d)
        die("%s: its header is not JSON", s->path);
    for (const struct janas_json *t = janas_json_root(d)->child; t;
         t = t->next) {
        if (t->key_n == 12 && !memcmp(t->key, "__metadata__", 12))
            continue;
        if (n_srcs == cap_srcs) {
            cap_srcs = cap_srcs ? 2 * cap_srcs : 4096;
            srcs = realloc(srcs, cap_srcs * sizeof(*srcs));
            if (!srcs)
                die("out of memory%s", "");
        }
        struct src *x = &srcs[n_srcs++];
        memset(x, 0, sizeof *x);
        if (t->key_n >= sizeof x->name)
            die("a tensor name too long in %s", s->path);
        memcpy(x->name, t->key, t->key_n);
        x->shard = si;
        const char *dt = janas_json_str(janas_json_get(t, "dtype"));
        x->dtype = dtype_of(dt ? dt : "");
        const struct janas_json *sh = janas_json_get(t, "shape");
        for (const struct janas_json *e = sh ? sh->child : NULL; e && x->nd < 4;
             e = e->next)
            x->shape[x->nd++] = (uint64_t)janas_json_num(e, 0);
        const struct janas_json *o = janas_json_get(t, "data_offsets");
        if (!o || o->n != 2)
            die("%s: a tensor without its offsets", s->path);
        uint64_t b0 = (uint64_t)janas_json_num(o->child, 0),
                 b1 = (uint64_t)janas_json_num(o->child->next, 0);
        x->off = s->data + b0;
        x->bytes = b1 - b0;
    }
    janas_json_free(d);
    janas_buf_free(&h);
}

static int by_name(const void *a, const void *b)
{
    return strcmp(((const struct src *)a)->name, ((const struct src *)b)->name);
}

/* by name, once the headers are read and srcs sorted */
static const struct src *find_src(const char *name)
{
    struct src key;
    snprintf(key.name, sizeof key.name, "%s", name);
    return bsearch(&key, srcs, n_srcs, sizeof *srcs, by_name);
}

/* ---- conversions ---- */

static float bf16_f32(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static float e4m3_f32(uint8_t b)
{
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 15 && m == 7)
        v = NAN;
    else if (e == 0)
        v = ldexpf((float)m, -9);
    else
        v = ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}

static float e8m0_f32(uint8_t e)
{
    return e == 255 ? NAN : ldexpf(1.0f, (int)e - 127);
}

/* a row of n floats as Q8_0: per 32, d = max |x| / 127 as fp16, the
   weights rounded to nearest, ties away from zero (as ggml's) */
static void q8_0_row(const float *x, size_t n, uint8_t *out)
{
    for (size_t b = 0; b < n / 32; b++, out += 34) {
        float amax = 0;
        for (int i = 0; i < 32; i++)
            amax = fmaxf(amax, fabsf(x[32 * b + i]));
        float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        uint16_t h = janas_fp32_to_fp16(d);
        memcpy(out, &h, 2);
        for (int i = 0; i < 32; i++)
            out[2 + i] = (uint8_t)(int8_t)roundf(x[32 * b + i] * id);
    }
}

/* FP8 rows x cols with its 128 x 128 scales into Q8_0 */
static void fp8_to_q8_0(const uint8_t *w, const uint8_t *sc, uint64_t rows,
                        uint64_t cols, uint8_t *out)
{
    float *row = xmalloc(cols * sizeof(float));
    uint64_t sc_cols = (cols + 127) / 128, rb = cols / 32 * 34;
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t c = 0; c < cols; c++)
            row[c] = e4m3_f32(w[r * cols + c]) *
                     e8m0_f32(sc[(r / 128) * sc_cols + c / 128]);
        q8_0_row(row, cols, out + r * rb);
    }
    free(row);
}

/* FP4 rows x cols (cols / 2 bytes a row, two weights a byte, the first in
   the low nibble) with a scale byte per 32 into MXFP4 blocks: the scale,
   then weights j and j + 16 of the block in byte j's low and high nibble */
static void fp4_to_mxfp4(const uint8_t *w, const uint8_t *sc, uint64_t rows,
                         uint64_t cols, uint8_t *out)
{
    uint64_t nb = cols / 32;
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t b = 0; b < nb; b++, out += 17) {
            const uint8_t *q = w + r * (cols / 2) + b * 16;
            uint8_t code[32];
            for (int i = 0; i < 16; i++) {
                code[2 * i] = q[i] & 15;
                code[2 * i + 1] = q[i] >> 4;
            }
            out[0] = sc[r * nb + b];
            for (int j = 0; j < 16; j++)
                out[1 + j] = (uint8_t)(code[j] | code[j + 16] << 4);
        }
}

/* ---- what is written ---- */

enum { K_COPY, K_BF16_F32, K_FP8_Q8, K_I64_I32, K_BF16_Q8, K_FP8_HALF };

struct res {
    char name[64];
    int type, kind, nd;
    uint64_t dims[4]; /* ggml order */
    uint64_t bytes, offset;
    const struct src *w, *sc;
    int done;
    /* K_FP8_HALF: half 0 or 1 of each row of the tensor res[of] (two FP8
       matrices side by side: the MTP block's e_proj and h_proj as eh_proj);
       the second has no entry of its own in the directory (hidden) */
    int half, hidden;
    size_t of;
};

/* --mtp: the multi-token prediction block (mtp.0) alone, as a file of one
   layer that the engine opens beside the model */
static int mtp_mode;

static struct res *res;
static size_t n_res, cap_res;

static void add_res(const char *name, const struct src *w, const struct src *sc,
                    int kind)
{
    if (n_res == cap_res) {
        cap_res = cap_res ? 2 * cap_res : 1024;
        res = realloc(res, cap_res * sizeof(*res));
        if (!res)
            die("out of memory%s", "");
    }
    struct res *r = &res[n_res++];
    memset(r, 0, sizeof *r);
    snprintf(r->name, sizeof r->name, "%s", name);
    r->w = w;
    r->sc = sc;
    r->kind = kind;
    r->nd = w->nd;
    for (int i = 0; i < w->nd; i++) /* ggml order: the last dimension first */
        r->dims[i] = w->shape[w->nd - 1 - i];
    for (int i = w->nd; i < 4; i++)
        r->dims[i] = 1;
    uint64_t rows = w->nd >= 2 ? w->shape[0] : 1;
    uint64_t cols = w->shape[w->nd - 1];
    uint64_t count = 1;
    for (int i = 0; i < w->nd; i++)
        count *= w->shape[i];
    switch (kind) {
    case K_COPY:
        r->type = w->dtype == D_BF16 ? T_BF16 : T_F32;
        r->bytes = w->bytes;
        break;
    case K_BF16_F32:
        r->type = T_F32;
        r->bytes = count * 4;
        break;
    case K_FP8_HALF:
        r->dims[0] *= 2; /* the row of the pair */
        r->type = T_Q8_0;
        r->bytes = rows * (2 * cols / 32) * 34;
        if (cols % 32)
            die("%s: a row not of whole blocks of 32", name);
        break;
    case K_FP8_Q8:
    case K_BF16_Q8:
        if (cols % 32)
            die("%s: a row not of whole blocks of 32", name);
        r->type = T_Q8_0;
        r->bytes = rows * (cols / 32) * 34;
        break;
    case K_I64_I32:
        r->type = T_I32;
        r->bytes = count * 4;
        break;
    }
}

/* the GGUF name of a checkpoint tensor (what follows "layers.L." or the
   global name), its conversion; 0 when it is not one written here */
static const struct {
    const char *hf, *gguf;
    int kind;
} MAP[] = {
    {"attn.attn_sink", "attn_sinks.weight", K_COPY},
    {"attn.compressor.ape", "attn_compressor_ape.weight", K_COPY},
    {"attn.compressor.norm.weight", "attn_compressor_norm.weight", K_BF16_F32},
    {"attn.compressor.wgate.weight", "attn_compressor_gate.weight", K_COPY},
    {"attn.compressor.wkv.weight", "attn_compressor_kv.weight", K_COPY},
    {"attn.indexer.compressor.ape", "indexer_compressor_ape.weight", K_COPY},
    {"attn.indexer.compressor.norm.weight", "indexer_compressor_norm.weight",
     K_BF16_F32},
    {"attn.indexer.compressor.wgate.weight", "indexer_compressor_gate.weight",
     K_COPY},
    {"attn.indexer.compressor.wkv.weight", "indexer_compressor_kv.weight",
     K_COPY},
    {"attn.indexer.weights_proj.weight", "indexer.proj.weight", K_COPY},
    {"attn.indexer.wq_b.weight", "indexer.attn_q_b.weight", K_FP8_Q8},
    {"attn.kv_norm.weight", "attn_kv_a_norm.weight", K_BF16_F32},
    {"attn_norm.weight", "attn_norm.weight", K_BF16_F32},
    {"attn.q_norm.weight", "attn_q_a_norm.weight", K_BF16_F32},
    {"attn.wkv.weight", "attn_kv.weight", K_FP8_Q8},
    {"attn.wo_a.weight", "attn_output_a.weight", K_FP8_Q8},
    {"attn.wo_b.weight", "attn_output_b.weight", K_FP8_Q8},
    {"attn.wq_a.weight", "attn_q_a.weight", K_FP8_Q8},
    {"attn.wq_b.weight", "attn_q_b.weight", K_FP8_Q8},
    {"ffn.gate.tid2eid", "ffn_gate_tid2eid.weight", K_I64_I32},
    {"ffn.gate.weight", "ffn_gate_inp.weight", K_BF16_F32},
    {"ffn.gate.bias", "exp_probs_b.bias", K_COPY},
    {"ffn_norm.weight", "ffn_norm.weight", K_BF16_F32},
    {"ffn.shared_experts.w1.weight", "ffn_gate_shexp.weight", K_FP8_Q8},
    {"ffn.shared_experts.w3.weight", "ffn_up_shexp.weight", K_FP8_Q8},
    {"ffn.shared_experts.w2.weight", "ffn_down_shexp.weight", K_FP8_Q8},
    {"hc_attn_base", "hc_attn_base.weight", K_COPY},
    {"hc_attn_fn", "hc_attn_fn.weight", K_COPY},
    {"hc_attn_scale", "hc_attn_scale.weight", K_COPY},
    {"hc_ffn_base", "hc_ffn_base.weight", K_COPY},
    {"hc_ffn_fn", "hc_ffn_fn.weight", K_COPY},
    {"hc_ffn_scale", "hc_ffn_scale.weight", K_COPY},
};
static const struct {
    const char *hf, *gguf;
    int kind;
} GLOBAL[] = {
    {"embed.weight", "token_embd.weight", K_BF16_Q8},
    {"head.weight", "output.weight", K_BF16_Q8},
    {"norm.weight", "output_norm.weight", K_BF16_F32},
    {"hc_head_fn", "output_hc_fn.weight", K_COPY},
    {"hc_head_base", "output_hc_base.weight", K_COPY},
    {"hc_head_scale", "output_hc_scale.weight", K_COPY},
};

/* --mtp: the block's own tensors (what follows "mtp.0." is as a layer's) */
static const struct {
    const char *hf, *gguf;
    int kind;
} MTP_GLOBAL[] = {
    {"mtp.0.enorm.weight", "mtp.enorm.weight", K_BF16_F32},
    {"mtp.0.hnorm.weight", "mtp.hnorm.weight", K_BF16_F32},
    {"mtp.0.norm.weight", "mtp.norm.weight", K_BF16_F32},
    {"mtp.0.hc_head_fn", "mtp.hc_head_fn.weight", K_COPY},
    {"mtp.0.hc_head_base", "mtp.hc_head_base.weight", K_COPY},
    {"mtp.0.hc_head_scale", "mtp.hc_head_scale.weight", K_COPY},
};

/* the expert matrices of a slot: w1 the gate, w3 the up, w2 the down */
#define EXPERT_ROWS_GU 2048 /* filled from the shapes */
struct slotmat {
    uint8_t *mx[3]; /* converted, waiting for the slot's other matrices */
};

static int n_layer, n_expert, n_used, d_model, ff;
static uint64_t mat_off[3], mat_bytes[3], slot_bytes, experts_offset;
static struct slotmat *pend; /* n_layer x n_expert */
static uint64_t *slot_off, *sums;
static int out_fd;
static uint64_t written;

static void put(const void *p, uint64_t n, uint64_t at)
{
    const uint8_t *b = p;
    while (n) {
        ssize_t w =
            pwrite(out_fd, b, n > (1u << 30) ? (1u << 30) : n, (off_t)at);
        if (w <= 0)
            die("write failed: %s", strerror(errno));
        b += w;
        n -= (uint64_t)w;
        at += (uint64_t)w;
        written += (uint64_t)w;
    }
}

/* the slot written when its three matrices are there, its CRC kept */
static void slot_done(int l, int e)
{
    struct slotmat *s = &pend[(size_t)l * n_expert + e];
    for (int p = 0; p < 3; p++)
        if (!s->mx[p])
            return;
    uint8_t *slot = calloc(1, slot_bytes);
    if (!slot)
        die("out of memory%s", "");
    for (int p = 0; p < 3; p++) {
        memcpy(slot + mat_off[p], s->mx[p], mat_bytes[p]);
        free(s->mx[p]);
        s->mx[p] = NULL;
    }
    uint64_t at = slot_off[(size_t)l * n_expert + e];
    put(slot, slot_bytes, at);
    sums[(size_t)l * n_expert + e] = janas_crc32(0, slot, slot_bytes);
    free(slot);
}

/* the bytes of tensors waiting for their partner (a weight for its scale) */
struct held {
    const struct src *s;
    uint8_t *p;
};
static struct held *held;
static size_t n_held, cap_held;

static uint8_t *take_held(const struct src *s)
{
    for (size_t i = 0; i < n_held; i++)
        if (held[i].s == s) {
            uint8_t *p = held[i].p;
            held[i] = held[--n_held];
            return p;
        }
    return NULL;
}

static void hold(const struct src *s, const uint8_t *p)
{
    if (n_held == cap_held) {
        cap_held = cap_held ? 2 * cap_held : 256;
        held = realloc(held, cap_held * sizeof(*held));
        if (!held)
            die("out of memory%s", "");
    }
    held[n_held].s = s;
    held[n_held].p = xmalloc(s->bytes);
    memcpy(held[n_held].p, p, s->bytes);
    n_held++;
}

/* the partner of a weight (its .scale) or of a scale (its .weight) */
static const struct src *partner(const struct src *s)
{
    char n[160];
    size_t l = strlen(s->name);
    if (l > 6 && !strcmp(s->name + l - 6, ".scale"))
        snprintf(n, sizeof n, "%.*s.weight", (int)(l - 6), s->name);
    else if (l > 7 && !strcmp(s->name + l - 7, ".weight"))
        snprintf(n, sizeof n, "%.*s.scale", (int)(l - 7), s->name);
    else
        return NULL;
    return find_src(n);
}

/* layer, expert and matrix (0 gate, 1 up, 2 down) of an expert weight or
   scale; 0 when it is not one */
static int expert_of(const char *name, int *l, int *e, int *p)
{
    int w;
    char tail[16];
    if (sscanf(name,
               mtp_mode ? "mtp.%d.ffn.experts.%d.w%d.%15s"
                        : "layers.%d.ffn.experts.%d.w%d.%15s",
               l, e, &w, tail) != 4)
        return 0;
    *p = w == 1 ? 0 : w == 3 ? 1 : w == 2 ? 2 : -1;
    return *p >= 0;
}

static const struct res *res_of(const struct src *s)
{
    for (size_t i = 0; i < n_res; i++)
        if (res[i].w == s || res[i].sc == s)
            return &res[i];
    return NULL;
}

/* one tensor of the checkpoint, its bytes in p */
static void convert(const struct src *s, const uint8_t *p)
{
    int l, e, m;
    if (expert_of(s->name, &l, &e, &m)) {
        const struct src *o = partner(s);
        uint8_t *op = o ? take_held(o) : NULL;
        if (!op) {
            hold(s, p);
            return;
        }
        int is_scale = s->dtype == D_E8M0;
        const struct src *w = is_scale ? o : s;
        const uint8_t *wp = is_scale ? op : p, *sp = is_scale ? p : op;
        uint64_t rows = w->shape[0], cols = w->shape[1] * 2;
        uint8_t *mx = xmalloc(mat_bytes[m]);
        fp4_to_mxfp4(wp, sp, rows, cols, mx);
        free(op);
        pend[(size_t)l * n_expert + e].mx[m] = mx;
        slot_done(l, e);
        return;
    }
    const struct res *r = res_of(s);
    if (!r)
        return; /* not written (mtp.*) */
    if (r->kind == K_FP8_HALF) {
        /* half of each row of the pair's tensor */
        const struct src *o = s == r->w ? r->sc : r->w;
        uint8_t *op = take_held(o);
        if (!op) {
            hold(s, p);
            return;
        }
        const uint8_t *wp = s == r->w ? p : op, *sp = s == r->w ? op : p;
        uint64_t rows = r->w->shape[0], cols = r->w->shape[1];
        uint64_t hb = cols / 32 * 34;
        uint8_t *q = xmalloc(rows * hb);
        fp8_to_q8_0(wp, sp, rows, cols, q);
        const struct res *t = &res[r->of];
        for (uint64_t i = 0; i < rows; i++)
            put(q + i * hb, hb,
                t->offset + i * 2 * hb + (uint64_t)r->half * hb);
        free(q);
        free(op);
        return;
    }
    if (r->kind == K_FP8_Q8) {
        const struct src *o = s == r->w ? r->sc : r->w;
        uint8_t *op = take_held(o);
        if (!op) {
            hold(s, p);
            return;
        }
        const uint8_t *wp = s == r->w ? p : op, *sp = s == r->w ? op : p;
        uint8_t *q = xmalloc(r->bytes);
        fp8_to_q8_0(wp, sp, r->w->shape[0], r->w->shape[1], q);
        put(q, r->bytes, r->offset);
        free(q);
        free(op);
        return;
    }
    if (r->kind == K_COPY) {
        put(p, r->bytes, r->offset);
        return;
    }
    if (r->kind == K_BF16_Q8) {
        uint64_t rows = r->w->shape[0], cols = r->w->shape[1];
        uint64_t rb = cols / 32 * 34;
        float *row = xmalloc(cols * sizeof(float));
        uint8_t *q = xmalloc(r->bytes);
        for (uint64_t i = 0; i < rows; i++) {
            for (uint64_t c = 0; c < cols; c++) {
                uint16_t h;
                memcpy(&h, p + 2 * (i * cols + c), 2);
                row[c] = bf16_f32(h);
            }
            q8_0_row(row, cols, q + i * rb);
        }
        put(q, r->bytes, r->offset);
        free(q);
        free(row);
        return;
    }
    uint64_t n = r->bytes / 4;
    uint8_t *o = xmalloc(r->bytes);
    if (r->kind == K_BF16_F32)
        for (uint64_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, p + 2 * i, 2);
            float f = bf16_f32(h);
            memcpy(o + 4 * i, &f, 4);
        }
    else
        for (uint64_t i = 0; i < n; i++) {
            int64_t v;
            memcpy(&v, p + 8 * i, 8);
            int32_t x = (int32_t)v;
            memcpy(o + 4 * i, &x, 4);
        }
    put(o, r->bytes, r->offset);
    free(o);
}

static int by_offset(const void *a, const void *b)
{
    const struct src *x = *(const struct src *const *)a,
                     *y = *(const struct src *const *)b;
    return x->off < y->off ? -1 : x->off > y->off;
}

/* a shard read whole, by runs of tensors, its hash checked */
static void read_shard(int si)
{
    struct shard *sh = &shards[si];
    const struct src **list = xmalloc(n_srcs * sizeof(*list));
    size_t n = 0;
    for (size_t i = 0; i < n_srcs; i++)
        if (srcs[i].shard == si)
            list[n++] = &srcs[i];
    qsort(list, n, sizeof(*list), by_offset);
    struct janas_sha256 h;
    janas_sha256_init(&h);
    janas_sha256_update(&h, sh->head.p, sh->head.n);
    uint64_t at = sh->data;
    int contiguous = 1;
    struct janas_buf b = {0};
    for (size_t i = 0; i < n;) {
        size_t j = i;
        int64_t from = (int64_t)list[i]->off, to = from;
        while (j < n &&
               (j == i ||
                (int64_t)(list[j]->off + list[j]->bytes) - from <= BATCH)) {
            if ((int64_t)list[j]->off != to)
                contiguous = 0;
            to = (int64_t)(list[j]->off + list[j]->bytes);
            j++;
        }
        if ((uint64_t)from != at)
            contiguous = 0;
        fetch_range(si, from, to, &b);
        printf("\rshard %d/%d: %.0f of %.0f MB", si + 1, n_shards,
               (double)to / 1e6, (double)sh->size / 1e6);
        fflush(stdout);
        janas_sha256_update(&h, b.p, b.n);
        at = (uint64_t)to;
        for (size_t k = i; k < j; k++)
            convert(list[k], (const uint8_t *)b.p + (list[k]->off - from));
        i = j;
    }
    janas_buf_free(&b);
    free(list);
    char got[65];
    janas_sha256_hex(&h, got);
    if (contiguous && at == sh->size && sh->sha[0]) {
        if (strcmp(got, sh->sha))
            die("%s: its SHA-256 is not the Hub's: a corrupt read", sh->path);
    } else
        fprintf(stderr,
                "\n%s: not checked against the Hub's SHA-256 (its "
                "tensors are not contiguous)\n",
                sh->path);
}

/* ---- the metadata, as a GGUF's key/value section ---- */

static struct janas_buf meta;
static uint64_t n_kv;

static void m_str(const char *s, size_t n)
{
    uint64_t l = n;
    janas_buf_put(&meta, (const char *)&l, 8);
    janas_buf_put(&meta, s, n);
}
static void kv_head(const char *k, uint32_t type)
{
    m_str(k, strlen(k));
    janas_buf_put(&meta, (const char *)&type, 4);
    n_kv++;
}
static void kv_u32(const char *k, uint32_t v)
{
    kv_head(k, G_U32);
    janas_buf_put(&meta, (const char *)&v, 4);
}
static void kv_f32(const char *k, float v)
{
    kv_head(k, G_F32);
    janas_buf_put(&meta, (const char *)&v, 4);
}
static void kv_bool(const char *k, int v)
{
    kv_head(k, G_BOOL);
    uint8_t b = (uint8_t)(v != 0);
    janas_buf_put(&meta, (const char *)&b, 1);
}
static void kv_str(const char *k, const char *v)
{
    kv_head(k, G_STR);
    m_str(v, strlen(v));
}
static void kv_arr(const char *k, uint32_t type, uint64_t n)
{
    kv_head(k, G_ARR);
    janas_buf_put(&meta, (const char *)&type, 4);
    janas_buf_put(&meta, (const char *)&n, 8);
}

static double cnum(const struct janas_json *c, const char *k)
{
    const struct janas_json *v = janas_json_get(c, k);
    if (!v || v->type != JANAS_JSON_NUMBER)
        die("config.json: no %s", k);
    return janas_json_num(v, 0);
}

static void write_meta(const struct janas_json *c, const struct janas_json *tok)
{
    char k[96];
#define K(x) (snprintf(k, sizeof k, "deepseek4.%s", x), k)
    kv_str("general.architecture", "deepseek4");
    kv_str("general.name", repo);
    kv_u32(K("block_count"), (uint32_t)n_layer);
    kv_u32(K("context_length"), (uint32_t)cnum(c, "max_position_embeddings"));
    kv_u32(K("embedding_length"), (uint32_t)d_model);
    kv_u32(K("attention.head_count"), (uint32_t)cnum(c, "num_attention_heads"));
    kv_u32(K("attention.head_count_kv"),
           (uint32_t)cnum(c, "num_key_value_heads"));
    const struct janas_json *rs = janas_json_get(c, "rope_scaling");
    kv_str(K("rope.scaling.type"), "yarn");
    kv_f32(K("rope.scaling.factor"), (float)cnum(rs, "factor"));
    kv_u32(K("rope.scaling.original_context_length"),
           (uint32_t)cnum(rs, "original_max_position_embeddings"));
    kv_f32(K("rope.scaling.yarn_beta_fast"), (float)cnum(rs, "beta_fast"));
    kv_f32(K("rope.scaling.yarn_beta_slow"), (float)cnum(rs, "beta_slow"));
    kv_f32(K("rope.freq_base"), (float)cnum(c, "rope_theta"));
    kv_f32(K("attention.layer_norm_rms_epsilon"),
           (float)cnum(c, "rms_norm_eps"));
    kv_u32(K("expert_count"), (uint32_t)n_expert);
    kv_u32(K("expert_used_count"), (uint32_t)n_used);
    kv_u32(K("expert_gating_func"), 4); /* llama.cpp: sqrt(softplus) */
    kv_u32(K("attention.key_length"), (uint32_t)cnum(c, "head_dim"));
    kv_u32(K("attention.value_length"), (uint32_t)cnum(c, "head_dim"));
    kv_u32(K("rope.dimension_count"), (uint32_t)cnum(c, "qk_rope_head_dim"));
    kv_u32(K("attention.q_lora_rank"), (uint32_t)cnum(c, "q_lora_rank"));
    kv_u32(K("attention.sliding_window"), (uint32_t)cnum(c, "sliding_window"));
    kv_u32(K("expert_feed_forward_length"), (uint32_t)ff);
    kv_u32(K("expert_shared_count"), (uint32_t)cnum(c, "n_shared_experts"));
    kv_f32(K("expert_weights_scale"), (float)cnum(c, "routed_scaling_factor"));
    const struct janas_json *nt = janas_json_get(c, "norm_topk_prob");
    kv_bool(K("expert_weights_norm"), nt && nt->type == JANAS_JSON_TRUE);
    const char *clamp[2] = {"swiglu_clamp_exp", "swiglu_clamp_shexp"};
    for (int i = 0; i < 2; i++) {
        kv_arr(K(clamp[i]), G_F32, (uint64_t)n_layer);
        float v = (float)cnum(c, "swiglu_limit");
        for (int l = 0; l < n_layer; l++)
            janas_buf_put(&meta, (const char *)&v, 4);
    }
    kv_u32(K("embedding_length_out"),
           (uint32_t)(d_model * (int)cnum(c, "hc_mult")));
    kv_u32(K("attention.indexer.head_count"),
           (uint32_t)cnum(c, "index_n_heads"));
    kv_u32(K("attention.indexer.key_length"),
           (uint32_t)cnum(c, "index_head_dim"));
    kv_u32(K("attention.indexer.top_k"), (uint32_t)cnum(c, "index_topk"));
    kv_u32(K("attention.output_group_count"), (uint32_t)cnum(c, "o_groups"));
    kv_u32(K("attention.output_lora_rank"), (uint32_t)cnum(c, "o_lora_rank"));
    const struct janas_json *cr = janas_json_get(c, "compress_ratios");
    if (!cr || cr->type != JANAS_JSON_ARRAY || (int)cr->n < n_layer)
        die("config.json: no compress_ratios for every layer%s", "");
    kv_arr(K("attention.compress_ratios"), G_I32, (uint64_t)n_layer);
    const struct janas_json *x = cr->child;
    for (int i = 0; i < n_layer; i++, x = x->next) {
        int32_t v = (int32_t)janas_json_num(x, 0);
        janas_buf_put(&meta, (const char *)&v, 4);
    }
    kv_f32(K("attention.compress_rope_freq_base"),
           (float)cnum(c, "compress_rope_theta"));
    kv_u32(K("hyper_connection.count"), (uint32_t)cnum(c, "hc_mult"));
    kv_u32(K("hyper_connection.sinkhorn_iterations"),
           (uint32_t)cnum(c, "hc_sinkhorn_iters"));
    kv_f32(K("hyper_connection.epsilon"), (float)cnum(c, "hc_eps"));
    kv_u32(K("hash_layer_count"), (uint32_t)cnum(c, "num_hash_layers"));
#undef K

    /* the tokenizer: byte-level BPE */
    uint32_t rows = (uint32_t)cnum(c, "vocab_size");
    const struct janas_json *model = janas_json_get(tok, "model");
    if (!janas_json_is(janas_json_get(model, "type"), "BPE"))
        die("tokenizer.json: not a BPE model%s", "");
    const char **tk = calloc(rows, sizeof(*tk));
    size_t *tl = calloc(rows, sizeof(*tl));
    int32_t *tt = calloc(rows, sizeof(*tt));
    if (!tk || !tl || !tt)
        die("out of memory%s", "");
    for (const struct janas_json *t = janas_json_get(model, "vocab")->child; t;
         t = t->next) {
        double id = janas_json_num(t, -1);
        if (id < 0 || id >= rows)
            continue;
        tk[(uint32_t)id] = t->key;
        tl[(uint32_t)id] = t->key_n;
        tt[(uint32_t)id] = 1;
    }
    const struct janas_json *added = janas_json_get(tok, "added_tokens");
    for (const struct janas_json *t = added ? added->child : NULL; t;
         t = t->next) {
        double id = janas_json_num(janas_json_get(t, "id"), -1);
        const struct janas_json *ct = janas_json_get(t, "content");
        if (id < 0 || id >= rows || !ct)
            continue;
        const struct janas_json *sp = janas_json_get(t, "special");
        tk[(uint32_t)id] = ct->s;
        tl[(uint32_t)id] = ct->n;
        tt[(uint32_t)id] = sp && sp->type == JANAS_JSON_TRUE ? 3 : 4;
    }
    /* ids past the vocabulary (the embeddings' padding): unused tokens */
    char pad[48];
    kv_str("tokenizer.ggml.model", "gpt2");
    kv_str("tokenizer.ggml.pre", "joyai-llm");
    kv_arr("tokenizer.ggml.tokens", G_STR, rows);
    for (uint32_t i = 0; i < rows; i++) {
        if (tk[i])
            m_str(tk[i], tl[i]);
        else {
            snprintf(pad, sizeof pad, "[PAD%u]", i);
            m_str(pad, strlen(pad));
        }
    }
    kv_arr("tokenizer.ggml.token_type", G_I32, rows);
    for (uint32_t i = 0; i < rows; i++) {
        int32_t v = tk[i] ? tt[i] : 5; /* unused */
        janas_buf_put(&meta, (const char *)&v, 4);
    }
    const struct janas_json *merges = janas_json_get(model, "merges");
    kv_arr("tokenizer.ggml.merges", G_STR, merges ? merges->n : 0);
    for (const struct janas_json *m = merges ? merges->child : NULL; m;
         m = m->next) {
        if (m->type == JANAS_JSON_STRING) {
            m_str(m->s, m->n);
            continue;
        }
        const struct janas_json *a = m->child, *b = a ? a->next : NULL;
        if (!a || !b)
            die("tokenizer.json: a bad merge%s", "");
        uint64_t n = a->n + 1 + b->n;
        janas_buf_put(&meta, (const char *)&n, 8);
        janas_buf_put(&meta, a->s, a->n);
        janas_buf_put(&meta, " ", 1);
        janas_buf_put(&meta, b->s, b->n);
    }
    kv_u32("tokenizer.ggml.bos_token_id", (uint32_t)cnum(c, "bos_token_id"));
    kv_u32("tokenizer.ggml.eos_token_id", (uint32_t)cnum(c, "eos_token_id"));
    kv_bool("tokenizer.ggml.add_bos_token", 1);
    free(tk);
    free(tl);
    free(tt);
    if (meta.oom)
        die("out of memory%s", "");
}

int main(int argc, char **argv)
{
    int meta_only = argc == 4 && !strcmp(argv[1], "--tokenizer-only");
    mtp_mode = argc == 4 && !strcmp(argv[1], "--mtp");
    if (meta_only || mtp_mode)
        argv++, argc--;
    if (argc != 3 || strncmp(argv[1], "hf:", 3)) {
        fprintf(stderr,
                "usage: hf2jns_ds4 [--tokenizer-only | --mtp] "
                "hf:<owner>/<repo>[@revision] <out.jns>\n"
                "  DeepSeek-V4-Flash from its checkpoint on Hugging Face;\n"
                "  --tokenizer-only: the metadata alone, no weights (a small "
                "file\n  to check the tokenizer with)\n"
                "  --mtp: the multi-token prediction block alone, the file "
                "the\n  engine finds beside the model (<model>-mtp.jns)\n");
        return 2;
    }
    snprintf(repo, sizeof repo, "%s", argv[1] + 3);
    char *at = strchr(repo, '@');
    if (at) {
        *at = 0;
        snprintf(rev, sizeof rev, "%s", at + 1);
    }
    const char *dst = argv[2];

    /* the configuration, the tokenizer, the shards and their tensors */
    struct janas_json_doc *cfg = fetch_json("config.json");
    struct janas_json_doc *tokd = fetch_json("tokenizer.json");
    struct janas_json_doc *idx = fetch_json("model.safetensors.index.json");
    const struct janas_json *c = janas_json_root(cfg);
    n_layer = mtp_mode ? 1 : (int)cnum(c, "num_hidden_layers");
    n_expert = (int)cnum(c, "n_routed_experts");
    n_used = (int)cnum(c, "num_experts_per_tok");
    d_model = (int)cnum(c, "hidden_size");
    ff = (int)cnum(c, "moe_intermediate_size");
    if (n_layer < 1 || n_layer > N_LAYER_MAX || n_expert < 1 ||
        n_expert > JANAS_JNS_MAX_EXPERTS || ff % 32 || d_model % 32)
        die("config.json: a shape this converter does not take%s", "");
    if (meta_only) {
        /* the header, the metadata and nothing else: no layers */
        write_meta(c, janas_json_root(tokd));
        uint64_t mb = 8 + meta.n, mo = ALIGN;
        uint64_t fb = align_up(mo + mb, ALIGN);
        out_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (out_fd < 0 || ftruncate(out_fd, (off_t)fb) != 0)
            die("%s: cannot write", dst);
        uint8_t *mreg = xmalloc(mb);
        memcpy(mreg, &n_kv, 8);
        memcpy(mreg + 8, meta.p, meta.n);
        put(mreg, mb, mo);
        struct janas_jns_header h = {0};
        memcpy(h.magic, JANAS_JNS_MAGIC, 8);
        h.version = JANAS_JNS_VERSION;
        h.header_bytes = ALIGN;
        h.file_bytes = fb;
        memcpy(h.arch, "deepseek4", 9);
        h.d_model = (uint32_t)d_model;
        h.layer_table_offset = h.expert_table_offset = h.expert_sums_offset =
            h.tensor_dir_offset = ALIGN;
        h.metadata_offset = mo;
        h.metadata_bytes = mb;
        h.resident_offset = h.experts_offset = fb;
        h.tables_checksum = janas_fnv1a(NULL, 0, JANAS_FNV_INIT);
        h.metadata_checksum = janas_crc32(0, mreg, mb);
        h.header_checksum = janas_fnv1a(&h, sizeof h, JANAS_FNV_INIT);
        put(&h, sizeof h, 0);
        if (close(out_fd) != 0)
            die("write failed: %s", strerror(errno));
        printf("written %s: the metadata alone\n", dst);
        return 0;
    }
    const struct janas_json *wm =
        janas_json_get(janas_json_root(idx), "weight_map");
    for (const struct janas_json *e = wm ? wm->child : NULL; e; e = e->next) {
        const char *f = janas_json_str(e);
        /* --mtp: only the shards that hold the block */
        if (mtp_mode && (e->key_n < 4 || memcmp(e->key, "mtp.", 4)))
            continue;
        int k = 0;
        while (k < n_shards && strcmp(shards[k].path, f))
            k++;
        if (k == n_shards) {
            if (n_shards == MAX_SHARDS)
                die("too many shards%s", "");
            snprintf(shards[n_shards++].path, sizeof shards[0].path, "%s", f);
        }
    }
    if (!n_shards)
        die("model.safetensors.index.json lists no shards%s", "");
    for (int i = 0; i < n_shards; i++) {
        char err[300];
        if (janas_hf_info(repo, rev[0] ? rev : NULL, shards[i].path,
                          &shards[i].size, shards[i].sha, err, sizeof err))
            die("%s", err);
        read_header(i);
        printf("\rheaders %d/%d", i + 1, n_shards);
        fflush(stdout);
    }
    printf("\n");
    qsort(srcs, n_srcs, sizeof *srcs, by_name);

    /* the resident tensors, and the experts' slots */
    char name[160], gg[64];
    for (size_t g = 0; !mtp_mode && g < sizeof GLOBAL / sizeof *GLOBAL; g++) {
        const struct src *s = find_src(GLOBAL[g].hf);
        if (!s)
            die("%s missing", GLOBAL[g].hf);
        add_res(GLOBAL[g].gguf, s, NULL, GLOBAL[g].kind);
    }
    if (mtp_mode) {
        /* the block's own: its norms, its final mix, and [e_proj | h_proj]
           as one matrix of rows of 2 d_model (the engine's eh_proj) */
        for (size_t g = 0; g < sizeof MTP_GLOBAL / sizeof *MTP_GLOBAL; g++) {
            const struct src *s = find_src(MTP_GLOBAL[g].hf);
            if (!s)
                die("%s missing", MTP_GLOBAL[g].hf);
            add_res(MTP_GLOBAL[g].gguf, s, NULL, MTP_GLOBAL[g].kind);
        }
        const struct src *ep = find_src("mtp.0.e_proj.weight"),
                         *hp = find_src("mtp.0.h_proj.weight");
        if (!ep || !hp || !partner(ep) || !partner(hp))
            die("mtp.0.e_proj or h_proj missing%s", "");
        add_res("mtp.eh_proj.weight", ep, partner(ep), K_FP8_HALF);
        size_t first = n_res - 1;
        add_res("", hp, partner(hp), K_FP8_HALF);
        res[n_res - 1].half = 1;
        res[n_res - 1].hidden = 1;
        res[n_res - 1].of = first;
        res[first].of = first;
    }
    for (int l = 0; l < n_layer; l++)
        for (size_t m = 0; m < sizeof MAP / sizeof *MAP; m++) {
            snprintf(name, sizeof name, mtp_mode ? "mtp.%d.%s" : "layers.%d.%s",
                     l, MAP[m].hf);
            const struct src *s = find_src(name);
            if (!s)
                continue; /* not in every layer (indexer, bias, tid2eid) */
            const struct src *sc = NULL;
            if (MAP[m].kind == K_FP8_Q8 && !(sc = partner(s)))
                die("%s: its scale missing", name);
            snprintf(gg, sizeof gg, "blk.%d.%s", l, MAP[m].gguf);
            add_res(gg, s, sc, MAP[m].kind);
        }
    {
        int l, e, p;
        size_t ne = 0;
        for (size_t i = 0; i < n_srcs; i++)
            if (expert_of(srcs[i].name, &l, &e, &p) && srcs[i].dtype == D_I8) {
                if (l >= n_layer || e >= n_expert)
                    die("%s: beyond the configuration", srcs[i].name);
                uint64_t rows = srcs[i].shape[0], cols = srcs[i].shape[1] * 2;
                uint64_t mb = rows * (cols / 32) * 17;
                if (mat_bytes[p] && mat_bytes[p] != mb)
                    die("%s: experts of different shapes", srcs[i].name);
                mat_bytes[p] = mb;
                ne++;
            }
        if (ne != (size_t)n_layer * n_expert * 3)
            die("not every expert matrix is there%s", "");
    }
    uint64_t off = 0;
    for (int p = 0; p < 3; p++) {
        off = align_up(off, MATRIX_ALIGN);
        mat_off[p] = off;
        off += mat_bytes[p];
    }
    slot_bytes = align_up(off, ALIGN);

    /* the layout */
    struct janas_jns_layer *layers = calloc((size_t)n_layer, sizeof(*layers));
    if (!layers)
        die("out of memory%s", "");
    for (int l = 0; l < n_layer; l++) {
        layers[l].slot_bytes = slot_bytes;
        for (int v = 0; v < 3; v++)
            layers[l].level_bytes[v] = slot_bytes;
        for (int p = 0; p < 3; p++)
            layers[l].m[p] = (struct janas_jns_matrix){
                .type = T_MXFP4,
                .rows = (uint32_t)(p < 2 ? ff : d_model),
                .cols = (uint32_t)(p < 2 ? d_model : ff),
                .offset = mat_off[p],
                .bytes = mat_bytes[p]};
    }
    write_meta(c, janas_json_root(tokd));
    uint64_t metadata_bytes = 8 + meta.n;
    uint64_t layer_table = ALIGN;
    uint64_t expert_table = layer_table + sizeof(*layers) * (uint64_t)n_layer;
    uint64_t expert_sums = expert_table + 8ull * n_layer * n_expert;
    uint64_t tensor_dir = expert_sums + 8ull * n_layer * n_expert;
    size_t n_dir = 0; /* the tensors with an entry of their own */
    for (size_t i = 0; i < n_res; i++)
        n_dir += !res[i].hidden;
    uint64_t metadata_offset =
        align_up(tensor_dir + sizeof(struct janas_jns_tensor) * n_dir, ALIGN);
    uint64_t resident_offset =
        align_up(metadata_offset + metadata_bytes, ALIGN);
    uint64_t pos = resident_offset;
    for (size_t i = 0; i < n_res; i++) {
        if (res[i].hidden) { /* the second half of another's rows */
            res[i].offset = res[res[i].of].offset;
            continue;
        }
        res[i].offset = pos;
        pos = align_up(pos + res[i].bytes, ALIGN);
    }
    uint64_t resident_bytes = pos - resident_offset;
    experts_offset = pos;
    slot_off = xmalloc(8ull * n_layer * n_expert);
    sums = calloc((size_t)n_layer * n_expert, 8);
    pend = calloc((size_t)n_layer * n_expert, sizeof(*pend));
    if (!sums || !pend)
        die("out of memory%s", "");
    for (int l = 0; l < n_layer; l++)
        for (int e = 0; e < n_expert; e++) {
            slot_off[(size_t)l * n_expert + e] = pos;
            pos += slot_bytes;
        }
    uint64_t experts_bytes = pos - experts_offset, file_bytes = pos;
    printf("deepseek4: %d layers, %d experts (k %d), d_model %d; %zu resident "
           "tensors %.2f GB, experts %.2f GB, %.2f GB in all\n",
           n_layer, n_expert, n_used, d_model, n_res, resident_bytes / 1e9,
           experts_bytes / 1e9, file_bytes / 1e9);

    out_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (out_fd < 0 || ftruncate(out_fd, (off_t)file_bytes) != 0)
        die("%s: cannot write", dst);

    /* every shard, in order */
    for (int i = 0; i < n_shards; i++) {
        read_shard(i);
        printf("\rshard %d/%d, %.1f GB written", i + 1, n_shards,
               written / 1e9);
        fflush(stdout);
    }
    printf("\n");
    if (n_held)
        die("a weight without its scale, or an expert matrix alone%s", "");
    for (size_t i = 0; i < (size_t)n_layer * n_expert; i++)
        if (!sums[i])
            die("an expert slot left unwritten%s", "");

    /* the tables, the metadata, the header */
    uint64_t tbl_len =
        tensor_dir + sizeof(struct janas_jns_tensor) * n_dir - layer_table;
    uint8_t *tbl = calloc(1, tbl_len);
    if (!tbl)
        die("out of memory%s", "");
    memcpy(tbl, layers, sizeof(*layers) * (size_t)n_layer);
    memcpy(tbl + (expert_table - layer_table), slot_off,
           8ull * n_layer * n_expert);
    memcpy(tbl + (expert_sums - layer_table), sums, 8ull * n_layer * n_expert);
    for (size_t i = 0, k = 0; i < n_res; i++) {
        if (res[i].hidden)
            continue;
        struct janas_jns_tensor d = {0};
        snprintf(d.name, sizeof d.name, "%s", res[i].name);
        d.type = (uint32_t)res[i].type;
        d.n_dims = (uint32_t)res[i].nd;
        memcpy(d.dims, res[i].dims, sizeof d.dims);
        d.offset = res[i].offset;
        d.bytes = res[i].bytes;
        memcpy(tbl + (tensor_dir - layer_table) + k++ * sizeof d, &d, sizeof d);
    }
    uint8_t *mreg = xmalloc(metadata_bytes);
    memcpy(mreg, &n_kv, 8);
    memcpy(mreg + 8, meta.p, meta.n);
    put(tbl, tbl_len, layer_table);
    put(mreg, metadata_bytes, metadata_offset);

    struct janas_jns_header h = {0};
    memcpy(h.magic, JANAS_JNS_MAGIC, 8);
    h.version = JANAS_JNS_VERSION;
    h.header_bytes = ALIGN;
    h.file_bytes = file_bytes;
    memcpy(h.arch, "deepseek4", 9);
    h.n_layer = (uint32_t)n_layer;
    h.n_expert = (uint32_t)n_expert;
    h.n_expert_used = (uint32_t)n_used;
    h.d_model = (uint32_t)d_model;
    h.layer_table_offset = layer_table;
    h.expert_table_offset = expert_table;
    h.expert_sums_offset = expert_sums;
    h.tensor_dir_offset = tensor_dir;
    h.n_tensor = n_dir;
    h.metadata_offset = metadata_offset;
    h.metadata_bytes = metadata_bytes;
    h.resident_offset = resident_offset;
    h.resident_bytes = resident_bytes;
    h.experts_offset = experts_offset;
    h.experts_bytes = experts_bytes;
    h.tables_checksum = janas_fnv1a(tbl, tbl_len, JANAS_FNV_INIT);
    h.metadata_checksum = janas_crc32(0, mreg, metadata_bytes);
    h.header_checksum = janas_fnv1a(&h, sizeof h, JANAS_FNV_INIT);
    put(&h, sizeof h, 0);
    if (close(out_fd) != 0)
        die("write failed: %s", strerror(errno));
    printf("written %s: %.2f GB\n", dst, file_bytes / 1e9);
    janas_json_free(cfg);
    janas_json_free(tokd);
    janas_json_free(idx);
    return 0;
}
