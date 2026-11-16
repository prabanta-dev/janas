/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_nh.c - NVIDIA's Nemotron-H with LatentMoE (nemotron_h_moe, the
 * Nemotron 3.5 Bolt): its hyperparameters, its layers and their loop.
 *
 * Every layer is one part with its own RMS norm (attn_norm) and its own
 * residual: a Mamba-2 mixer (model_mamba.c), an attention or a mixture of
 * experts. The file says which by layer: KV heads, an attention; a
 * feed-forward width, a mixture; neither, a Mamba-2 layer.
 *
 * The attention has no RoPE and no norm on q and k: 32 heads of 128 over 2
 * KV heads, scores scaled by 1 / sqrt(128).
 *
 * The mixture: a router of the layer's own number of experts on the
 * normalized input, sigmoid scores; the experts chosen by score plus a
 * bias (exp_probs_b), their weights the scores normalized to one, times
 * expert_weights_scale. The experts work in a latent space: the input
 * brought down to moe_latent_size (ffn_latent_down) before them, their
 * weighted sum brought up (ffn_latent_up) after; each is up and down with
 * relu^2 between them, no gate. A shared expert of the same kind works on
 * the whole input and is added.
 *
 * What is computed is llama.cpp's src/models/nemotron-h.cpp and
 * build_mamba2_layer, the reference the logits are compared against.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matvec.h"
#include "vmath.h"

#include "model_impl.h"

static int meta_u32(struct janas_llm_model *m, const char *suffix, uint32_t *v)
{
    char key[128];
    snprintf(key, sizeof(key), "%.32s.%s", m->j.h.arch, suffix);
    return janas_jns_meta_u32(&m->j, key, v);
}

static int meta_f32(struct janas_llm_model *m, const char *suffix, float *v)
{
    char key[128];
    snprintf(key, sizeof(key), "%.32s.%s", m->j.h.arch, suffix);
    return janas_jns_meta_f32(&m->j, key, v);
}

/* Element i of an integer array of the architecture's metadata, 0 if
   there is none. */
static uint64_t meta_at(struct janas_llm_model *m, const char *suffix,
                        uint64_t i)
{
    char key[128];
    uint64_t v = 0, count = 0;
    snprintf(key, sizeof(key), "%.32s.%s", m->j.h.arch, suffix);
    if (janas_jns_meta_uint_at(&m->j, key, i, &v, &count) != 0)
        return 0;
    return v;
}

int janas_m_nh_kind(struct janas_llm_model *m, uint32_t l)
{
    if (meta_at(m, "attention.head_count_kv", l))
        return JANAS_NH_ATTN;
    if (meta_at(m, "feed_forward_length", l))
        return JANAS_NH_MOE;
    return JANAS_NH_MAMBA;
}

int janas_m_nh_params(struct janas_llm_model *m)
{
    uint32_t heads_kv = 0;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        uint32_t h = (uint32_t)meta_at(m, "attention.head_count_kv", l);
        if (h > heads_kv)
            heads_kv = h;
    }
    m->n_head_kv = heads_kv;
    if (meta_u32(m, "attention.head_count", &m->n_head) ||
        meta_u32(m, "attention.key_length", &m->head_dim) ||
        meta_f32(m, "attention.layer_norm_rms_epsilon", &m->eps) ||
        meta_u32(m, "ssm.conv_kernel", &m->d_conv) ||
        meta_u32(m, "ssm.state_size", &m->ds) ||
        meta_u32(m, "ssm.group_count", &m->n_kh) ||
        meta_u32(m, "ssm.time_step_rank", &m->n_vh) ||
        meta_u32(m, "ssm.inner_size", &m->d_inner) ||
        meta_u32(m, "moe_latent_size", &m->moe_lat) ||
        meta_f32(m, "expert_weights_scale", &m->route_scale))
        return -1;
    /* no RoPE: the rotation tables are made for the head, never used */
    m->n_rot = m->head_dim;
    m->rope_base = 10000.0f;
    if (m->n_head_kv == 0 || m->n_head % m->n_head_kv || m->head_dim % 8 ||
        m->head_dim > 512 || m->d_model % JANAS_QK || m->k > MAX_K ||
        (m->n_head * m->head_dim) % JANAS_QK || m->d_conv < 2 ||
        m->d_conv > MAX_CONV || m->n_vh == 0 || m->n_kh == 0 ||
        m->n_vh % m->n_kh || m->d_inner % m->n_vh || m->d_inner % m->n_kh ||
        m->d_inner % JANAS_QK || m->ds % 8 || m->moe_lat == 0 ||
        m->moe_lat % JANAS_QK)
        return -1;
    m->ssm_hd = m->d_inner / m->n_vh;
    m->conv_ch = m->d_inner + 2 * m->n_kh * m->ds;
    /* z | x B C | dt */
    m->qkvz_dim = m->conv_ch + m->d_inner + m->n_vh;
    /* float states, two uint16 units a number */
    m->ssm_per = (size_t)m->n_vh * m->ssm_hd * m->ds * 2;
    /* the widest expert, as the experts' buffers take it */
    m->d_ff = 0;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        const struct janas_jns_layer *jl = &m->j.layers[l];
        if (!jl->slot_bytes)
            continue;
        const struct janas_jns_matrix *up = &jl->m[JANAS_JNS_UP],
                                      *dn = &jl->m[JANAS_JNS_DOWN];
        if (up->cols != m->moe_lat || dn->rows != m->moe_lat ||
            dn->cols != up->rows || up->rows % JANAS_QK)
            return -1;
        if (up->rows > m->d_ff)
            m->d_ff = up->rows;
    }
    return m->d_ff ? 0 : -1;
}

/* A tensor of the layer, of type type (0 f32, -1 any the products read)
   and shape d0 x d1 (d1 0: one dimension). */
static const struct janas_jns_tensor *need(struct janas_llm_model *m,
                                           uint32_t l, const char *suffix,
                                           int type, uint64_t d0, uint64_t d1,
                                           char *err, size_t err_len)
{
    char name[96];
    snprintf(name, sizeof(name), "blk.%u.%s", l, suffix);
    const struct janas_jns_tensor *t = janas_jns_tensor(&m->j, name);
    int ok = t && t->dims[0] == d0 && (d1 ? t->dims[1] == d1 : 1);
    if (ok)
        ok = type == 0 ? t->type == 0
                       : janas_qtype_block_size((int)t->type) != 0;
    if (!ok)
        snprintf(err, err_len, "tensor %s missing or unexpected", name);
    return ok ? t : NULL;
}

/* The scale of an NVFP4 matrix (the file's name.scale), 1 if none. */
static float scale_of(struct janas_llm_model *m, uint32_t l, const char *suffix)
{
    char name[96];
    snprintf(name, sizeof(name), "blk.%u.%s", l, suffix);
    const struct janas_jns_tensor *t = janas_jns_tensor(&m->j, name);
    return t && t->type == 0 && t->dims[0] == 1 ? f32(m, t)[0] : 1.0f;
}

int janas_m_nh_layer_load(struct janas_llm_model *m, uint32_t l,
                          struct layer *ly, char *err, size_t err_len)
{
    uint32_t dm = m->d_model;
#define NEED(var, suffix, type, d0, d1)                                        \
    if (!(var = need(m, l, suffix, type, d0, d1, err, err_len)))               \
        return -1;
    NEED(ly->attn_norm, "attn_norm.weight", 0, dm, 0)
    if (ly->nh_kind == JANAS_NH_MAMBA) {
        const struct janas_jns_tensor *cb, *dd;
        NEED(ly->ssm_in, "ssm_in.weight", -1, dm, m->qkvz_dim)
        NEED(ly->conv, "ssm_conv1d.weight", 0, m->d_conv, m->conv_ch)
        NEED(cb, "ssm_conv1d.bias", 0, m->conv_ch, 0)
        NEED(ly->dt_bias, "ssm_dt.bias", 0, m->n_vh, 0)
        NEED(ly->ssm_a, "ssm_a", 0, 1, m->n_vh)
        NEED(dd, "ssm_d", 0, 1, m->n_vh)
        NEED(ly->ssm_norm, "ssm_norm.weight", 0, m->d_inner / m->n_kh, m->n_kh)
        NEED(ly->ssm_out, "ssm_out.weight", -1, m->d_inner, dm)
        ly->conv_b = f32(m, cb);
        ly->ssm_dd = f32(m, dd);
        ly->in_scale = scale_of(m, l, "ssm_in.scale");
        ly->out_scale2 = scale_of(m, l, "ssm_out.scale");
        /* the conv's weights transposed: tap after tap, channels together */
        ly->conv_t = malloc((size_t)m->d_conv * m->conv_ch * sizeof(float));
        if (!ly->conv_t)
            return -1;
        for (uint32_t c = 0; c < m->conv_ch; c++)
            for (uint32_t i = 0; i < m->d_conv; i++)
                ly->conv_t[(size_t)i * m->conv_ch + c] =
                    f32(m, ly->conv)[(size_t)c * m->d_conv + i];
    } else if (ly->nh_kind == JANAS_NH_ATTN) {
        uint32_t qd = m->n_head * ly->hd, kvd = ly->n_kv * ly->hd;
        NEED(ly->wq, "attn_q.weight", -1, dm, qd)
        NEED(ly->wk, "attn_k.weight", -1, dm, kvd)
        NEED(ly->wv, "attn_v.weight", -1, dm, kvd)
        NEED(ly->wo, "attn_output.weight", -1, qd, dm)
    } else {
        const struct janas_jns_layer *jl = &m->j.layers[l];
        const struct janas_jns_tensor *eb;
        NEED(ly->gate_inp, "ffn_gate_inp.weight", 0, dm, 0)
        ly->n_exp = (uint32_t)ly->gate_inp->dims[1];
        /* the router's every expert present in the file (an absent one
           would be read from offset 0), and at least the k it chooses */
        int absent = 0;
        for (uint32_t e = 0; e < ly->n_exp && e < m->n_expert; e++)
            absent |= janas_jns_expert_offset(&m->j, l, e) == 0;
        if (!jl->slot_bytes || ly->n_exp == 0 || ly->n_exp > m->n_expert ||
            ly->n_exp < m->k || absent || jl->m[JANAS_JNS_GATE].type != 0) {
            snprintf(err, err_len, "layer %u: unexpected experts", l);
            return -1;
        }
        NEED(eb, "exp_probs_b.bias", 0, ly->n_exp, 0)
        ly->exp_bias = f32(m, eb);
        NEED(ly->lat_down, "ffn_latent_down.weight", -1, dm, m->moe_lat)
        NEED(ly->lat_up, "ffn_latent_up.weight", -1, m->moe_lat, dm)
        NEED(ly->sh_up, "ffn_up_shexp.weight", -1, dm, 0)
        uint32_t sf = (uint32_t)ly->sh_up->dims[1];
        NEED(ly->sh_down, "ffn_down_shexp.weight", -1, sf, dm)
        if (sf % JANAS_QK) {
            snprintf(err, err_len, "layer %u: shared expert's width", l);
            return -1;
        }
        if (sf > m->sh_ff)
            m->sh_ff = sf;
        /* the router in half precision, converted in place, as the other
           mixtures' (model_load.c) */
        uint16_t *h16 = (uint16_t *)data(m, ly->gate_inp);
        const float *w = f32(m, ly->gate_inp);
        for (size_t i = 0; i < (size_t)dm * ly->n_exp; i++)
            h16[i] = janas_fp32_to_fp16(w[i]);
        ly->router16 = h16;
    }
#undef NEED
    return 0;
}

int janas_m_nh_alloc(struct janas_llm_model *m)
{
    size_t B = m->blk, L = m->moe_lat, S = m->sh_ff;
    m->lat = malloc(B * L * sizeof(float));
    m->lat_acc = malloc(B * L * sizeof(float));
    m->latq = malloc(B * L / JANAS_QK * sizeof(*m->latq));
    m->sh_h = malloc(B * S * sizeof(float));
    m->sh_hq = malloc(B * S / JANAS_QK * sizeof(*m->sh_hq));
    return m->lat && m->lat_acc && m->latq && m->sh_h && m->sh_hq ? 0 : -1;
}

void janas_m_nh_free(struct janas_llm_model *m)
{
    free(m->lat);
    free(m->lat_acc);
    free(m->latq);
    free(m->sh_h);
    free(m->sh_hq);
}

static void relu2_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    struct janas_llm_model *m = t->m;
    uint32_t sf = (uint32_t)t->ly->sh_up->dims[1];
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        float *h = m->sh_h + (size_t)j * sf;
        for (uint32_t i = 0; i < sf; i++)
            h[i] = h[i] > 0.0f ? h[i] * h[i] : 0.0f;
        janas_q8k_quantize(h, m->sh_hq + (size_t)j * (sf / JANAS_QK), sf);
    }
}

/* The shared expert on the normalized block (xq), added to x. */
static void shared_expert(struct janas_llm_model *m, const struct layer *ly,
                          uint32_t n)
{
    uint32_t dm = m->d_model, sf = (uint32_t)ly->sh_up->dims[1];
    struct janas_matvec_task up = {.type = (int)ly->sh_up->type,
                                   .w = data(m, ly->sh_up),
                                   .x = m->xq,
                                   .y = m->sh_h,
                                   .rows = sf,
                                   .cols = dm,
                                   .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 1), &up, 1);
    struct tok_job t = {.m = m, .ly = ly, .n = n};
    if (n < 2)
        relu2_worker(&t, 0, 1);
    else
        janas_pool_run(m->compute, relu2_worker, &t);
    struct janas_matvec_task down = {.type = (int)ly->sh_down->type,
                                     .w = data(m, ly->sh_down),
                                     .x = m->sh_hq,
                                     .y = m->xn,
                                     .rows = dm,
                                     .cols = sf,
                                     .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 1), &down, 1);
    for (size_t i = 0; i < (size_t)n * dm; i++)
        m->x[i] += m->xn[i];
}

int janas_m_nh_layers(struct janas_llm_model *m, uint32_t n, uint32_t pos0)
{
    uint32_t dm = m->d_model;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        const struct layer *ly = &m->layers[l];
        janas_m_norm_quant(m, f32(m, ly->attn_norm), 0, n);
        trace_row(m, "attn_norm", m->xn + (size_t)(n - 1) * dm, dm);
        if (ly->nh_kind == JANAS_NH_MAMBA) {
            janas_m_mamba_layer(m, ly, n);
        } else if (ly->nh_kind == JANAS_NH_ATTN) {
            janas_m_attn_layer(m, ly, n, pos0);
        } else {
            /* the routed experts first, then the shared one: each reads
               the normalized block, which the shared expert's down product
               overwrites */
            if (janas_m_moe_block(m, ly, m->cache, &m->j.layers[l], l, n,
                                  MOE_ROUTED) != 0)
                return -1;
            shared_expert(m, ly, n);
        }
        if (m->trace) {
            const float *v = m->x + (size_t)(n - 1) * dm;
            double nn = 0;
            for (uint32_t i = 0; i < dm; i++)
                nn += (double)v[i] * v[i];
            fprintf(stderr, "l_out-%-3u norm %10.4f  %9.4f %9.4f %9.4f %9.4f\n",
                    l, sqrt(nn), v[0], v[1], v[2], v[3]);
        }
        if (m->n_obs && janas_m_obs_layer_out(m, l, m->x, n, dm))
            break; /* an observer stops the forward here */
    }
    return 0;
}
