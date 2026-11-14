/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_qsa.c - qwen4exp's sparse attention: past ix_top + ix_ratio - 1
 * positions a query attends only to the ix_top / ix_ratio blocks of
 * ix_ratio positions its indexer scores highest, and to the positions of
 * its own block while that is incomplete.
 *
 * The indexer, in each attention layer: a key of ix_dim numbers per position
 * (index_k_proj of the layer's input), kept raw; a block's ix_ratio keys
 * pooled into one, their mean normalized (index_k_norm) and turned for the
 * block's first position; a query per head (index_q_proj, normalized and
 * turned for its own position); a block's score the sum over the heads of
 * relu(q_h . key). The tail - positions (q + 1) / ratio * ratio to q, the
 * query's own block until it is complete - is always attended. With no more
 * complete blocks than ix_top / ix_ratio every position is, and the
 * attention is the dense one: only past that is a query's row worked out
 * again here, over the positions chosen.
 *
 * llama.cpp's build_qsa_top_k (src/models/qwen4exp.cpp) takes its top_k +
 * ratio - 1 positions by score, so where the tail is shorter than ratio - 1
 * it adds a few positions of one more block, chosen among equal scores by
 * its sort; the reference it follows, and this file, keep whole blocks.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "common/aligned.h"
#include "common/pool.h"
#include "jmath.h"
#include "vmath.h"

#include "model_impl.h"

int janas_m_qsa_load(struct janas_llm_model *m, char *err, size_t err_len)
{
    char key[128];
    uint32_t heads = 0, dim = 0, top = 0;
    snprintf(key, sizeof(key), "%.32s.attention.indexer.head_count",
             m->j.h.arch);
    if (janas_jns_meta_u32(&m->j, key, &heads) != 0)
        return 0; /* no indexer: dense attention throughout */
    snprintf(key, sizeof(key), "%.32s.attention.indexer.key_length",
             m->j.h.arch);
    janas_jns_meta_u32(&m->j, key, &dim);
    snprintf(key, sizeof(key), "%.32s.attention.indexer.top_k", m->j.h.arch);
    janas_jns_meta_u32(&m->j, key, &top);
    /* one ratio for every attention layer that has one: llama.cpp's input
       set is per ratio, and Flash-Next's are all 4 */
    uint32_t ratio = 0;
    snprintf(key, sizeof(key), "%.32s.attention.compress_ratios", m->j.h.arch);
    for (uint32_t l = 0; l < m->n_layer; l++) {
        uint64_t r = 0;
        if (janas_jns_meta_uint_at(&m->j, key, l, &r, NULL) != 0)
            break;
        if (m->layers[l].rec || r == 0)
            continue;
        if (ratio && r != ratio) {
            snprintf(err, err_len, "sparse attention: ratios differ");
            return -1;
        }
        ratio = (uint32_t)r;
    }
    if (!ratio)
        return 0;
    if (heads == 0 || heads > 64 || dim == 0 || dim % 8 || dim > m->head_dim ||
        top == 0 || top % ratio || ratio > 64 || m->n_rot > dim ||
        top + ratio > JANAS_QSA_MAX) {
        snprintf(err, err_len, "sparse attention: inconsistent sizes");
        return -1;
    }
    m->ix_heads = heads;
    m->ix_dim = dim;
    m->ix_top = top;
    m->ix_ratio = ratio;
    uint32_t dm = m->d_model;
    char name[64];
    for (uint32_t l = 0; l < m->n_layer; l++) {
        struct layer *ly = &m->layers[l];
        if (ly->rec)
            continue;
        static const char *const nm[4] = {"q_proj", "k_proj", "q_norm",
                                          "k_norm"};
        const struct janas_jns_tensor *t[4];
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "blk.%u.indexer.%s.weight", l, nm[i]);
            t[i] = janas_jns_tensor(&m->j, name);
            uint64_t d0 = i < 2 ? dm : dim, d1 = i == 0 ? heads * dim : dim;
            int ok = t[i] && t[i]->dims[0] == d0 &&
                     (i < 2 ? t[i]->dims[1] == d1 && t[i]->type == 30
                            : t[i]->n_dims == 1 && t[i]->type == 0);
            if (!ok) {
                snprintf(err, err_len, "tensor %s missing or unexpected", name);
                return -1;
            }
        }
        ly->ix_q = t[0];
        ly->ix_k = t[1];
        ly->ix_qn = t[2];
        ly->ix_kn = t[3];
    }
    /* the raw keys of every position, at half precision as llama.cpp keeps
       them, the pooled keys of every block, a block's queries and keys */
    size_t B = m->blk, ctx = m->n_ctx, nbk = (ctx + ratio - 1) / ratio;
    /* on whole pages: the GPU's whole token imports them (model_gpu.c) */
    m->ix_raw = janas_aligned_alloc(
        4096, ((size_t)m->n_attn * ctx * dim * sizeof(uint16_t) + 4095) &
                  ~(size_t)4095);
    m->ix_pool = janas_aligned_alloc(
        4096,
        ((size_t)m->n_attn * nbk * dim * sizeof(float) + 4095) & ~(size_t)4095);
    /* the bf16 products' (janas_gpu_dense_mm: a prompt's blocks) */
    m->ix_qs = janas_m_out_alloc(m, B * heads * dim * sizeof(float));
    m->ix_kb = janas_m_out_alloc(m, B * dim * sizeof(float));
    m->ix_xb = janas_m_out_alloc(m, B * dm * sizeof(float));
    m->ix_score = malloc(nbk * sizeof(float));
    m->ix_sel = malloc((nbk + 1) * sizeof(uint32_t));
    m->ix_pos = malloc(((size_t)top + ratio) * sizeof(uint32_t));
    if (!m->ix_raw || !m->ix_pool || !m->ix_qs || !m->ix_kb || !m->ix_xb ||
        !m->ix_score || !m->ix_sel || !m->ix_pos) {
        snprintf(err, err_len, "out of memory for the sparse attention");
        return -1;
    }
    return 0;
}

void janas_m_qsa_free(struct janas_llm_model *m)
{
    janas_aligned_free(m->ix_raw);
    janas_aligned_free(m->ix_pool);
    janas_aligned_free(m->ix_qs);
    janas_aligned_free(m->ix_kb);
    janas_aligned_free(m->ix_xb);
    free(m->ix_score);
    free(m->ix_sel);
    free(m->ix_pos);
}

/* NeoX rotation of the first n_rot numbers of v for position p, as the
   attention turns its heads (model.c rope_table, model_attn.c) */
static void rope_at(const struct janas_llm_model *m, float *v, uint32_t p)
{
    uint32_t half = m->n_rot / 2;
    for (uint32_t i = 0; i < half; i++) {
        float s, c,
            theta = (float)p * janas_powf(m->rope_base,
                                          -2.0f * (float)i / (float)m->n_rot);
        janas_sincosf(theta, &s, &c);
        float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

void janas_m_qsa_rope(const struct janas_llm_model *m, float *out,
                      uint32_t pos0, uint32_t n)
{
    uint32_t half = m->n_rot / 2, r = m->ix_ratio;
    for (uint32_t j = 0; j < n + n / r + 2; j++) {
        uint32_t p = j < n ? pos0 + j : (pos0 / r + (j - n)) * r;
        float *row = out + (size_t)j * m->n_rot;
        for (uint32_t i = 0; i < half; i++) {
            float theta =
                (float)p *
                janas_powf(m->rope_base, -2.0f * (float)i / (float)m->n_rot);
            janas_sincosf(theta, &row[half + i], &row[i]);
        }
    }
}

/* block b of slot sl pooled: the mean of its raw keys, normalized, turned
   for its first position */
static void pool_block(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t b)
{
    uint32_t d = m->ix_dim, r = m->ix_ratio;
    size_t nbk = (m->n_ctx + r - 1) / r;
    const uint16_t *raw =
        m->ix_raw + ((size_t)ly->slot * m->n_ctx + (size_t)b * r) * d;
    float *out = m->ix_pool + ((size_t)ly->slot * nbk + b) * d;
    for (uint32_t i = 0; i < d; i++) {
        float s = 0.0f;
        for (uint32_t k = 0; k < r; k++)
            s += janas_fp16_to_fp32(raw[(size_t)k * d + i]);
        out[i] = s / (float)r;
    }
    rms_norm(out, out, f32(m, ly->ix_kn), d, m->eps);
    rope_at(m, out, b * r);
}

void janas_m_qsa_restore(struct janas_llm_model *m, uint32_t n)
{
    for (uint32_t l = 0; m->ix_heads && l < m->n_layer; l++)
        if (m->layers[l].ix_q)
            for (uint32_t b = 0; (b + 1) * m->ix_ratio <= n; b++)
                pool_block(m, &m->layers[l], b);
}

/* The context moved along: the raw keys after keep + drop come down drop
   places (they carry no RoPE), and every block from keep's on is pooled
   again, at its new place */
void janas_m_qsa_shift(struct janas_llm_model *m, uint32_t keep, uint32_t drop,
                       uint32_t n_kv)
{
    if (!m->ix_heads)
        return;
    uint32_t d = m->ix_dim, r = m->ix_ratio, n = n_kv - drop;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        const struct layer *ly = &m->layers[l];
        if (!ly->ix_q)
            continue;
        uint16_t *raw = m->ix_raw + (size_t)ly->slot * m->n_ctx * d;
        memmove(raw + (size_t)keep * d, raw + (size_t)(keep + drop) * d,
                (size_t)(n - keep) * d * sizeof(uint16_t));
        for (uint32_t b = keep / r; (b + 1) * r <= n; b++)
            pool_block(m, ly, b);
    }
}

struct score_job {
    struct janas_llm_model *m;
    const float *pool, *q; /* the layer's pooled keys, the query's heads */
    uint32_t n_blocks;
};

static void score_worker(void *arg, int tid, int n_threads)
{
    struct score_job *s = arg;
    struct janas_llm_model *m = s->m;
    uint32_t d = m->ix_dim;
    uint32_t per =
        (s->n_blocks + (uint32_t)n_threads - 1) / (uint32_t)n_threads;
    uint32_t a = (uint32_t)tid * per,
             e = a + per < s->n_blocks ? a + per : s->n_blocks;
    for (uint32_t b = a; b < e; b++) {
        const float *k = s->pool + (size_t)b * d;
        float sum = 0.0f;
        for (uint32_t h = 0; h < m->ix_heads; h++) {
            float dot = janas_dot_f32(s->q + (size_t)h * d, k, d);
            sum += dot > 0.0f ? dot : 0.0f;
        }
        m->ix_score[b] = sum;
    }
}

/* The k-th largest of v[0 .. n) (k < n), v reordered: Hoare's selection */
static float kth_largest(float *v, uint32_t n, uint32_t k)
{
    uint32_t lo = 0, hi = n - 1;
    while (lo < hi) {
        float pivot = v[lo + (hi - lo) / 2];
        uint32_t i = lo, j = hi;
        while (i <= j) {
            while (v[i] > pivot)
                i++;
            while (v[j] < pivot)
                j--;
            if (i <= j) {
                float t = v[i];
                v[i] = v[j];
                v[j] = t;
                i++;
                if (j == 0)
                    break;
                j--;
            }
        }
        if (k <= j)
            hi = j;
        else if (k >= i)
            lo = i;
        else
            break;
    }
    return v[k];
}

/* The keep blocks of highest score among blocks 0 .. n - 1 (n > keep), in
   their order into sel: those above the keep-th score, then among those
   equal to it the earliest */
static void choose(struct janas_llm_model *m, uint32_t n, uint32_t keep)
{
    float *tmp = (float *)(void *)m->ix_sel; /* as many floats as blocks */
    memcpy(tmp, m->ix_score, n * sizeof(float));
    float th = kth_largest(tmp, n, keep - 1);
    uint32_t above = 0;
    for (uint32_t b = 0; b < n; b++)
        above += m->ix_score[b] > th;
    uint32_t equal = keep - above, k = 0;
    for (uint32_t b = 0; b < n && k < keep; b++)
        if (m->ix_score[b] > th || (m->ix_score[b] == th && equal && equal--))
            m->ix_pos[k++] = b;
}

struct head_job {
    struct janas_llm_model *m;
    const struct layer *ly;
    const float *q; /* the token's query heads, after norm and RoPE */
    float *out;     /* its attention row */
    const uint32_t *pos;
    uint32_t n_pos;
};

/* one query head over the chosen positions */
static void head_worker(void *arg, int tid, int n_threads)
{
    struct head_job *t = arg;
    struct janas_llm_model *m = t->m;
    const struct layer *ly = t->ly;
    uint32_t hd = ly->hd, nkv = ly->n_kv, rep = m->n_head / nkv;
    uint32_t cap = m->kv_cap[ly->slot];
    float scale = 1.0f / sqrtf((float)hd);
    float sc[JANAS_QSA_MAX], kf[512];
    for (uint32_t h = (uint32_t)tid; h < m->n_head; h += (uint32_t)n_threads) {
        uint32_t g = h / rep;
        const float *q = t->q + (size_t)h * hd;
        const float *ks =
            m->cur->kscale + m->ks_off[ly->slot] + (size_t)g * cap;
        const float *vs =
            m->cur->vscale + m->ks_off[ly->slot] + (size_t)g * cap;
        float mx = -INFINITY;
        for (uint32_t i = 0; i < t->n_pos; i++) {
            uint32_t p = t->pos[i];
            size_t at = ((size_t)g * cap + p) * hd;
            if (m->kv16) {
                const int16_t *k =
                    (const int16_t *)(const void *)(m->cur->kcache +
                                                    m->kv_off[ly->slot]) +
                    at;
                for (uint32_t e = 0; e < hd; e++)
                    kf[e] = (float)k[e];
            } else {
                const int8_t *k = m->cur->kcache + m->kv_off[ly->slot] + at;
                for (uint32_t e = 0; e < hd; e++)
                    kf[e] = (float)k[e];
            }
            sc[i] = janas_dot_f32(q, kf, hd) * ks[p] * scale;
            if (sc[i] > mx)
                mx = sc[i];
        }
        double sum = 0.0;
        for (uint32_t i = 0; i < t->n_pos; i++) {
            sc[i] = janas_expf(sc[i] - mx);
            sum += sc[i];
        }
        float *out = t->out + (size_t)h * hd;
        memset(out, 0, hd * sizeof(float));
        for (uint32_t i = 0; i < t->n_pos; i++) {
            uint32_t p = t->pos[i];
            size_t at = ((size_t)g * cap + p) * hd;
            float w = (float)(sc[i] / sum) * vs[p];
            if (m->kv16) {
                const int16_t *v =
                    (const int16_t *)(const void *)(m->cur->vcache +
                                                    m->kv_off[ly->slot]) +
                    at;
                for (uint32_t e = 0; e < hd; e++)
                    out[e] += w * (float)v[e];
            } else {
                const int8_t *v = m->cur->vcache + m->kv_off[ly->slot] + at;
                for (uint32_t e = 0; e < hd; e++)
                    out[e] += w * (float)v[e];
            }
        }
    }
}

void janas_m_qsa_keys(struct janas_llm_model *m, const struct layer *ly,
                      const float *xb, uint32_t n, uint32_t pos0)
{
    if (!m->ix_heads || !ly->ix_q)
        return;
    uint32_t dm = m->d_model, d = m->ix_dim, r = m->ix_ratio;
    janas_m_bf16_product(m, data(m, ly->ix_k), xb, m->ix_kb, d, dm, n, 1.0f);
    uint16_t *raw = m->ix_raw + (size_t)ly->slot * m->n_ctx * d;
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = 0; i < d; i++)
            raw[(size_t)(pos0 + j) * d + i] =
                janas_fp32_to_fp16(m->ix_kb[(size_t)j * d + i]);
    for (uint32_t b = pos0 / r; (b + 1) * r <= pos0 + n; b++)
        pool_block(m, ly, b);
}

void janas_m_qsa(struct janas_llm_model *m, const struct layer *ly, uint32_t n,
                 uint32_t pos0)
{
    if (!m->ix_heads || !ly->ix_q)
        return;
    uint32_t dm = m->d_model, d = m->ix_dim, r = m->ix_ratio, nh = m->ix_heads;
    size_t nbk = (m->n_ctx + r - 1) / r;
    /* the indexer's keys and queries from the layer's input, at bf16 */
    for (size_t i = 0; i < (size_t)n * dm; i++)
        m->ix_xb[i] = janas_m_round_bf16(m->xn[i]);
    janas_m_bf16_product(m, data(m, ly->ix_k), m->ix_xb, m->ix_kb, d, dm, n,
                         1.0f);
    janas_m_bf16_product(m, data(m, ly->ix_q), m->ix_xb, m->ix_qs, nh * d, dm,
                         n, 1.0f);
    uint16_t *raw = m->ix_raw + (size_t)ly->slot * m->n_ctx * d;
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < d; i++)
            raw[(size_t)(pos0 + j) * d + i] =
                janas_fp32_to_fp16(m->ix_kb[(size_t)j * d + i]);
        for (uint32_t h = 0; h < nh; h++) {
            float *q = m->ix_qs + ((size_t)j * nh + h) * d;
            rms_norm(q, q, f32(m, ly->ix_qn), d, m->eps);
            rope_at(m, q, pos0 + j);
        }
    }
    /* the blocks this call completed, or wrote again */
    for (uint32_t b = pos0 / r; (b + 1) * r <= pos0 + n; b++)
        pool_block(m, ly, b);
    const float *pool = m->ix_pool + (size_t)ly->slot * nbk * d;
    uint32_t keep = m->ix_top / r, qd = m->n_head * ly->hd;
    for (uint32_t j = 0; j < n; j++) {
        uint32_t p = pos0 + j, tail = (p + 1) / r * r, blocks = tail / r;
        if (blocks <= keep)
            continue; /* every position: the dense attention stands */
        struct score_job sj = {.m = m,
                               .pool = pool,
                               .q = m->ix_qs + (size_t)j * nh * d,
                               .n_blocks = blocks};
        janas_pool_run(m->compute, score_worker, &sj);
        /* the blocks chosen, then their positions, in place from the end */
        choose(m, blocks, keep);
        uint32_t np = keep * r;
        for (uint32_t i = keep; i-- > 0;)
            for (uint32_t k = r; k-- > 0;)
                m->ix_pos[i * r + k] = m->ix_pos[i] * r + k;
        for (uint32_t t = tail; t <= p; t++)
            m->ix_pos[np++] = t;
        struct head_job hj = {.m = m,
                              .ly = ly,
                              .q = m->q + (size_t)j * qd,
                              .out = m->att + (size_t)j * qd,
                              .pos = m->ix_pos,
                              .n_pos = np};
        janas_pool_run(m->compute, head_worker, &hj);
    }
}
