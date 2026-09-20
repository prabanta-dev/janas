/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_load.c - opening a model: its tensors layer by layer, the MTP
 * block, the buffers, and closing it (see model.h).
 */
#define _GNU_SOURCE
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "attention.h"
#include "common/pool.h"
#include "common/sysinfo.h"
#include "deltanet.h"
#include "tuner.h"
#include "gpu.h"
#include "matvec.h"
#include "quant.h"
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

enum { T_QUANT, T_F32, T_F16 };

/* Tensor by name with an expected type class and shape (d1 = 0: any). */
static const struct janas_jns_tensor *need(const struct janas_jns *j,
                                           const char *name, int kind,
                                           uint64_t d0, uint64_t d1, char *err,
                                           size_t err_len)
{
    const struct janas_jns_tensor *t = janas_jns_tensor(j, name);
    if (!t) {
        snprintf(err, err_len, "tensor %s missing", name);
        return NULL;
    }
    int ok = t->dims[0] == d0 && (d1 == 0 || t->dims[1] == d1);
    if (kind == T_F32)
        ok = ok && t->type == 0;
    else if (kind == T_F16)
        ok = ok && t->type == 1;
    else
        ok = ok && janas_qtype_block_size((int)t->type) != 0;
    if (!ok) {
        snprintf(err, err_len, "tensor %s: unexpected type or shape", name);
        return NULL;
    }
    return t;
}

/* Hyperparameters of the Gated DeltaNet layers. */
static int load_ssm_params(struct janas_llm_model *m)
{
    if (meta_u32(m, "ssm.conv_kernel", &m->d_conv) ||
        meta_u32(m, "ssm.state_size", &m->ds) ||
        meta_u32(m, "ssm.group_count", &m->n_kh) ||
        meta_u32(m, "ssm.time_step_rank", &m->n_vh) ||
        meta_u32(m, "ssm.inner_size", &m->d_inner))
        return -1;
    if (m->d_conv < 2 || m->d_conv > MAX_CONV || m->ds % SSM_ROWS ||
        m->ds > MAX_SSM_DIM || m->n_kh == 0 || m->n_vh % m->n_kh ||
        m->d_inner != m->n_vh * m->ds || m->d_inner % JANAS_QK)
        return -1;
    m->conv_ch = 2 * m->n_kh * m->ds + m->d_inner;
    m->qkvz_dim = 2 * m->n_kh * m->ds + 2 * m->d_inner;
    return 0;
}

/*
 * Gemma 4: sliding-window layers and full ones, each kind with its head
 * size, RoPE and KV heads; the KV heads come as an array, one per layer,
 * and have to be the same within a kind (one attention per kind). *pat:
 * the pattern, one byte per layer, nonzero for a sliding-window layer.
 */
static int load_swa_params(struct janas_llm_model *m, const uint8_t **pat)
{
    char key[128];
    uint64_t np = 0, nk = 0;
    const int32_t *kv = NULL;
    snprintf(key, sizeof(key), "%.32s.attention.sliding_window_pattern",
             m->j.h.arch);
    if (janas_jns_meta_bytes(&m->j, key, &np, pat) || np < m->n_layer)
        return -1;
    snprintf(key, sizeof(key), "%.32s.attention.head_count_kv", m->j.h.arch);
    if (janas_jns_meta_ints(&m->j, key, &nk, &kv) || nk < m->n_layer)
        return -1;
    m->n_head_kv = m->n_kv_swa = 0;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        int32_t h;
        memcpy(&h, kv + l, 4);
        uint32_t *want = (*pat)[l] ? &m->n_kv_swa : &m->n_head_kv;
        if (h <= 0 || (*want && *want != (uint32_t)h))
            return -1;
        *want = (uint32_t)h;
    }
    m->softcap = 0.0f;
    meta_f32(m, "final_logit_softcapping", &m->softcap);
    if (m->n_head_kv == 0 || m->n_kv_swa == 0 ||
        meta_u32(m, "attention.key_length_swa", &m->head_swa) ||
        meta_u32(m, "rope.dimension_count_swa", &m->n_rot_swa) ||
        meta_f32(m, "rope.freq_base_swa", &m->rope_base_swa) ||
        meta_u32(m, "attention.sliding_window", &m->window) ||
        m->head_swa % 16 || m->head_swa > 512 || m->n_rot_swa == 0 ||
        m->n_rot_swa % 2 || m->n_rot_swa > m->head_swa || m->window == 0 ||
        m->n_head % m->n_kv_swa || (m->n_head * m->head_swa) % JANAS_QK)
        return -1;
    return 0;
}

/*
 * Rows of small weights read as f32 (a DeltaNet layer's alpha and beta):
 * the file's own when they are f32, else made f32 once from whatever type
 * they come in (Q8_0 or F16 in unsloth's files, Q5_1 when a whole file was
 * requantized), exactly - d_model x value heads, too few to be worth a
 * kernel.
 */
static int rows_f32(struct janas_llm_model *m, const struct janas_jns *j,
                    uint32_t l, const char *suffix, uint64_t d0, uint64_t d1,
                    const float **out, float **own, char *err, size_t err_len)
{
    char name[64];
    snprintf(name, sizeof(name), "blk.%u.%s", l, suffix);
    const struct janas_jns_tensor *t = janas_jns_tensor(j, name);
    int half = t && (t->type == 1 || t->type == 30); /* f16, bf16 */
    if (!t || t->dims[0] != d0 || t->dims[1] != d1 ||
        (t->type != 0 && !half &&
         (janas_qtype_block_size((int)t->type) == 0 || d0 % JANAS_QK))) {
        snprintf(err, err_len, "tensor %s missing or unexpected", name);
        return -1;
    }
    if (t->type == 0) {
        *out = f32(m, t);
        return 0;
    }
    if (!(*own = malloc(d0 * d1 * sizeof(float)))) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    if (half) {
        const uint16_t *h = data(m, t);
        for (uint64_t i = 0; i < d0 * d1; i++)
            (*own)[i] = t->type == 1 ? janas_fp16_to_fp32(h[i])
                                     : janas_u2f((uint32_t)h[i] << 16);
    } else {
        janas_dequantize((int)t->type, data(m, t), *own, d0 * d1);
    }
    *out = *own;
    return 0;
}

static int load_layer(struct janas_llm_model *m, const struct janas_jns *j,
                      uint32_t l, struct layer *ly, char *err, size_t err_len)
{
    if (!ly->hd) { /* the model's geometry, unless the caller set it */
        ly->hd = m->head_dim;
        ly->n_kv = m->n_head_kv;
        ly->n_rot = m->n_rot;
    }
    uint32_t dm = m->d_model, qd = m->n_head * ly->hd, kvd = ly->n_kv * ly->hd,
             ff = m->d_ff;
    char name[64];
#define NEED(field, suffix, kind, d0, d1)                                      \
    snprintf(name, sizeof(name), "blk.%u." suffix, l);                         \
    if (!(ly->field = need(j, name, kind, d0, d1, err, err_len)))              \
        return -1;
    NEED(attn_norm, "attn_norm.weight", T_F32, dm, 0)
    if (ly->rec) {
        uint32_t s = m->ds;
        if (m->a->rec == JANAS_REC_Q35) {
            NEED(ssm_qkv, "attn_qkv.weight", T_QUANT, dm, m->conv_ch)
            NEED(ssm_z, "attn_gate.weight", T_QUANT, dm, m->d_inner)
            if (rows_f32(m, j, l, "ssm_alpha.weight", dm, m->n_vh,
                         &ly->ssm_alpha, &ly->alpha_own, err, err_len) ||
                rows_f32(m, j, l, "ssm_beta.weight", dm, m->n_vh, &ly->ssm_beta,
                         &ly->beta_own, err, err_len))
                return -1;
        } else {
            NEED(ssm_in, "ssm_in.weight", T_QUANT, dm, m->qkvz_dim)
            NEED(ssm_ba, "ssm_ba.weight", T_QUANT, dm, 2 * m->n_vh)
        }
        NEED(conv, "ssm_conv1d.weight", T_F32, m->d_conv, m->conv_ch)
        ly->conv_t = malloc((size_t)m->d_conv * m->conv_ch * sizeof(float));
        if (!ly->conv_t)
            return -1;
        for (uint32_t c = 0; c < m->conv_ch; c++)
            for (uint32_t i = 0; i < m->d_conv; i++)
                ly->conv_t[(size_t)i * m->conv_ch + c] =
                    f32(m, ly->conv)[(size_t)c * m->d_conv + i];
        NEED(dt_bias, "ssm_dt.bias", T_F32, m->n_vh, 0)
        NEED(ssm_a, "ssm_a", T_F32, m->n_vh, 0)
        NEED(ssm_norm, "ssm_norm.weight", T_F32, s, 0)
        NEED(ssm_out, "ssm_out.weight", T_QUANT, m->d_inner, dm)
    } else {
        NEED(wq, "attn_q.weight", T_QUANT, dm, qd * (m->a->attn_gate ? 2 : 1))
        NEED(wk, "attn_k.weight", T_QUANT, dm, kvd)
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l);
        if (!m->a->swa || janas_jns_tensor(j, name)) {
            NEED(wv, "attn_v.weight", T_QUANT, dm, kvd)
        }
        NEED(wo, "attn_output.weight", T_QUANT, qd, dm)
        NEED(q_norm, "attn_q_norm.weight", T_F32, ly->hd, 0)
        NEED(k_norm, "attn_k_norm.weight", T_F32, ly->hd, 0)
    }
    ly->out_scale = 1.0f;
    if (m->a->sandwich) {
        NEED(post_attn, "post_attention_norm.weight", T_F32, dm, 0)
        NEED(post_ffn, "post_ffw_norm.weight", T_F32, dm, 0)
        snprintf(name, sizeof(name), "blk.%u.layer_output_scale.weight", l);
        const struct janas_jns_tensor *os = janas_jns_tensor(j, name);
        if (os && os->type == 0 && os->dims[0] == 1)
            ly->out_scale = f32(m, os)[0];
    }
    if (m->a->post_attn_norm) {
        NEED(ffn_norm, "post_attention_norm.weight", T_F32, dm, 0)
    } else {
        NEED(ffn_norm, "ffn_norm.weight", T_F32, dm, 0)
    }
    snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", l);
    ly->gate_inp = janas_jns_tensor(j, name);
    if (!ly->gate_inp && m->n_expert == 1) {
        /* a dense feed-forward is the only expert of its layer and has no
           router: a zero logit comes out of the softmax as weight 1 exactly,
           so the routed path below serves it unchanged */
        ly->router_own = calloc(dm, sizeof(uint16_t));
        if (!ly->router_own)
            return -1;
        ly->router16 = ly->router_own;
    } else if (!ly->gate_inp || ly->gate_inp->dims[0] != dm ||
               ly->gate_inp->dims[1] != m->n_expert ||
               (ly->gate_inp->type != 0 && ly->gate_inp->type != 30)) {
        snprintf(err, err_len, "tensor %s missing or unexpected", name);
        return -1;
    } else {
        /* the router in half precision, converted in place: half the reads of
           a table read in full for every token (from f32 or bf16) */
        uint16_t *h16 = (uint16_t *)data(m, ly->gate_inp);
        size_t nw = (size_t)dm * m->n_expert;
        if (ly->gate_inp->type == 0) {
            const float *w = f32(m, ly->gate_inp);
            for (size_t i = 0; i < nw; i++)
                h16[i] = janas_fp32_to_fp16(w[i]);
        } else {
            for (size_t i = 0; i < nw; i++)
                h16[i] = janas_fp32_to_fp16(janas_u2f((uint32_t)h16[i] << 16));
        }
        ly->router16 = h16;
    }
    if (m->a->shared_expert) {
        const struct janas_jns_tensor *sg, *sw[3];
        static const char *const sh_names[3] = {"ffn_gate_shexp.weight",
                                                "ffn_up_shexp.weight",
                                                "ffn_down_shexp.weight"};
        for (int p = 0; p < 3; p++) {
            snprintf(name, sizeof(name), "blk.%u.%s", l, sh_names[p]);
            if (!(sw[p] = need(j, name, T_QUANT, p == 2 ? ff : dm,
                               p == 2 ? dm : ff, err, err_len)))
                return -1;
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp_shexp.weight", l);
        sg = janas_jns_tensor(j, name);
        if (!sg || sg->dims[0] != dm ||
            (sg->type != 0 && sg->type != 1 && sg->type != 30)) {
            snprintf(err, err_len, "tensor %s missing or unexpected", name);
            return -1;
        }
        for (int p = 0; p < 3; p++) {
            ly->sh_w[p] = data(m, sw[p]);
            ly->sh_mx[p] =
                (struct janas_jns_matrix){.type = sw[p]->type,
                                          .rows = (uint32_t)sw[p]->dims[1],
                                          .cols = (uint32_t)sw[p]->dims[0],
                                          .bytes = sw[p]->bytes};
        }
        ly->sh_gate = malloc(dm * sizeof(float));
        if (!ly->sh_gate)
            return -1;
        if (sg->type == 0) {
            memcpy(ly->sh_gate, data(m, sg), dm * sizeof(float));
        } else if (sg->type == 30) { /* bf16 */
            const uint16_t *b16 = data(m, sg);
            for (uint32_t i = 0; i < dm; i++)
                ly->sh_gate[i] = janas_u2f((uint32_t)b16[i] << 16);
        } else {
            const uint16_t *h16 = data(m, sg);
            for (uint32_t i = 0; i < dm; i++)
                ly->sh_gate[i] = janas_fp16_to_fp32(h16[i]);
        }
    }
#undef NEED
    return 0;
}

/* The MTP block: a one-layer JNS file converted by tools/hf2jns_mtp. */
static int load_mtp(struct janas_llm_model *m, const char *path, char *err,
                    size_t err_len)
{
    char why[256];
    if (janas_jns_open(path, &m->mtp_j, why, sizeof(why)) != 0) {
        /* the MTP file's own name: the caller names the model's */
        snprintf(err, err_len, "MTP block %s: %s", path, why);
        return -1;
    }
    m->mtp_open = 1;
    const struct janas_jns_header *h = &m->mtp_j.h;
    if (strncmp(h->arch, m->j.h.arch, sizeof(h->arch)) != 0 ||
        h->n_layer != 1 || h->n_expert != m->n_expert ||
        h->n_expert_used != m->k || h->d_model != m->d_model ||
        m->mtp_j.layers[0].m[JANAS_JNS_GATE].rows != m->d_ff ||
        m->mtp_j.layers[0].m[JANAS_JNS_DOWN].rows != m->d_model) {
        snprintf(err, err_len, "%s: not an MTP block of this model", path);
        return -1;
    }
    m->mtp_res = aligned_alloc(4096, h->resident_bytes);
    if (!m->mtp_res) {
        snprintf(err, err_len, "out of memory for the MTP block");
        return -1;
    }
    for (uint64_t done = 0; done < h->resident_bytes;) {
        ssize_t r =
            pread(m->mtp_j.fd, m->mtp_res + done, h->resident_bytes - done,
                  (off_t)(h->resident_offset + done));
        if (r <= 0) {
            snprintf(err, err_len, "cannot read the MTP block");
            return -1;
        }
        done += (uint64_t)r;
    }
    uint32_t dm = m->d_model;
    m->mtp = m->mtp_own_cache = 1;
    m->mtp_jl = &m->mtp_j.layers[0];
    m->mtp_cl = 0;
    m->mtp_layer.rec = 0;
    m->mtp_layer.slot = m->n_attn; /* the KV slot after the main model's */
    if (load_layer(m, &m->mtp_j, 0, &m->mtp_layer, err, err_len) != 0 ||
        !(m->eh_proj = need(&m->mtp_j, "mtp.eh_proj.weight", T_QUANT, 2 * dm,
                            dm, err, err_len)) ||
        !(m->enorm = need(&m->mtp_j, "mtp.enorm.weight", T_F32, dm, 0, err,
                          err_len)) ||
        !(m->hnorm = need(&m->mtp_j, "mtp.hnorm.weight", T_F32, dm, 0, err,
                          err_len)) ||
        !(m->mtp_norm =
              need(&m->mtp_j, "mtp.norm.weight", T_F32, dm, 0, err, err_len)))
        return -1;
    return 0;
}

/* The MTP block inside the main file (qwen35moe: the block after the
   model's layers, tensors nextn.*); its experts are in the main cache. */
static int load_mtp_internal(struct janas_llm_model *m, char *err,
                             size_t err_len)
{
    uint32_t l = m->n_layer, dm = m->d_model;
    char name[64];
    m->mtp_layer.rec = 0;
    m->mtp_layer.slot = m->n_attn;
    if (load_layer(m, &m->j, l, &m->mtp_layer, err, err_len) != 0)
        return -1;
#define NEXTN(field, suffix, kind, d0, d1)                                     \
    snprintf(name, sizeof(name), "blk.%u.nextn." suffix, l);                   \
    if (!(m->field = need(&m->j, name, kind, d0, d1, err, err_len)))           \
        return -1;
    NEXTN(eh_proj, "eh_proj.weight", T_QUANT, 2 * dm, dm)
    NEXTN(enorm, "enorm.weight", T_F32, dm, 0)
    NEXTN(hnorm, "hnorm.weight", T_F32, dm, 0)
    NEXTN(mtp_norm, "shared_head_norm.weight", T_F32, dm, 0)
#undef NEXTN
    m->mtp = 1;
    m->mtp_own_cache = 0;
    m->mtp_jl = &m->j.layers[l];
    m->mtp_cl = l;
    return 0;
}

/* The draft head with the base lowest ids, room for 8192 more tokens. */
static int init_draft_head(struct janas_llm_model *m, uint32_t base)
{
    if (base > m->n_vocab)
        base = m->n_vocab;
    m->dh_row =
        m->d_model / JANAS_QK * janas_qtype_block_size((int)m->output->type);
    m->dh_cap = base + 8192 < m->n_vocab ? base + 8192 : m->n_vocab;
    m->dh_w = malloc((size_t)m->dh_cap * m->dh_row);
    m->dh_ids = malloc(m->dh_cap * sizeof(int32_t));
    m->dh_in = calloc(m->n_vocab, 1);
    m->dh_logits = malloc(m->dh_cap * sizeof(float));
    if (!m->dh_w || !m->dh_ids || !m->dh_in || !m->dh_logits)
        return -1;
    memcpy(m->dh_w, data(m, m->output), (size_t)base * m->dh_row);
    for (uint32_t i = 0; i < base; i++) {
        m->dh_ids[i] = (int32_t)i;
        m->dh_in[i] = 1;
    }
    m->dh_n = base;
    return 0;
}

struct janas_llm_model *janas_llm_model_load(const char *path,
                                             const struct janas_llm_options *o,
                                             char *err, size_t err_len)
{
    struct janas_llm_model *m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;
    if (janas_jns_open(path, &m->j, err, err_len) != 0) {
        free(m);
        return NULL;
    }
    m->a = janas_arch_find(m->j.h.arch, sizeof(m->j.h.arch));
    if (!m->a) {
        snprintf(err, err_len, "architecture %.32s not supported", m->j.h.arch);
        goto fail;
    }
    m->n_layer = m->j.h.n_layer;
    uint32_t nextn = 0;
    if (m->a->nextn_in_file &&
        meta_u32(m, "nextn_predict_layers", &nextn) == 0 && nextn < m->n_layer)
        m->n_layer -= nextn; /* the file's last blocks predict ahead */
    m->d_model = m->j.h.d_model;
    m->n_expert = m->j.h.n_expert;
    m->k = m->j.h.n_expert_used;
    m->d_ff = m->j.layers[0].m[JANAS_JNS_GATE].rows;
    uint32_t interval = 4;
    const uint8_t *swa_pat = NULL;
    if (meta_u32(m, "attention.head_count", &m->n_head) ||
        (m->a->swa ? load_swa_params(m, &swa_pat)
                   : meta_u32(m, "attention.head_count_kv", &m->n_head_kv)) ||
        meta_u32(m, "attention.key_length", &m->head_dim) ||
        meta_f32(m, "attention.layer_norm_rms_epsilon", &m->eps) ||
        meta_f32(m, "rope.freq_base", &m->rope_base) || m->k > MAX_K ||
        m->n_head_kv == 0 || m->n_head % m->n_head_kv || m->head_dim % 8 ||
        m->d_model % JANAS_QK || (m->n_head * m->head_dim) % JANAS_QK ||
        m->d_ff % JANAS_QK || (m->a->rec && load_ssm_params(m) != 0)) {
        snprintf(err, err_len, "missing or inconsistent hyperparameters");
        goto fail;
    }
    if (meta_u32(m, "rope.dimension_count", &m->n_rot) != 0)
        m->n_rot = m->head_dim;
    if (m->a->rec)
        meta_u32(m, "full_attention_interval", &interval);
    if (m->n_rot == 0 || m->n_rot % 2 || m->n_rot > m->head_dim ||
        interval == 0 || m->head_dim > 512) {
        snprintf(err, err_len, "inconsistent rotary or layer pattern");
        goto fail;
    }
    m->hd_max = m->head_dim > m->head_swa ? m->head_dim : m->head_swa;
    m->kvd_max = m->n_head_kv * m->head_dim;
    if (m->n_kv_swa * m->head_swa > m->kvd_max)
        m->kvd_max = m->n_kv_swa * m->head_swa;
    for (uint32_t l = 0; l < m->n_layer; l++)
        if (m->j.layers[l].m[JANAS_JNS_GATE].rows != m->d_ff ||
            m->j.layers[l].m[JANAS_JNS_DOWN].rows != m->d_model) {
            snprintf(err, err_len, "layer %u: expert shapes differ", l);
            goto fail;
        }

    /* resident region into RAM; with the GPU, memory from its driver
       (JANAS_GPU_ALLOC) or imported at the end */
    const char *ge = getenv("JANAS_GPU");
    if (ge ? atoi(ge) > 0 : o->use_gpu) {
        char gerr[256];
        m->gpu = janas_gpu_create(gerr, sizeof(gerr));
        if (m->gpu && getenv("JANAS_GPU_ALLOC") &&
            (m->resident = janas_gpu_alloc(m->gpu, m->j.h.resident_bytes)))
            m->resident_gpu = 1;
    }
    if (!m->resident)
        m->resident = aligned_alloc(4096, m->j.h.resident_bytes);
    if (!m->resident) {
        snprintf(err, err_len, "out of memory for the resident region");
        goto fail;
    }
    for (uint64_t done = 0; done < m->j.h.resident_bytes;) {
        ssize_t r =
            pread(m->j.fd, m->resident + done, m->j.h.resident_bytes - done,
                  (off_t)(m->j.h.resident_offset + done));
        if (r <= 0) {
            snprintf(err, err_len, "cannot read the resident region");
            goto fail;
        }
        done += (uint64_t)r;
    }

    uint32_t dm = m->d_model, qd = m->n_head * m->hd_max, kvd = m->kvd_max,
             ff = m->d_ff;
    if (!(m->tok_embd =
              need(&m->j, "token_embd.weight", T_QUANT, dm, 0, err, err_len)) ||
        !(m->out_norm =
              need(&m->j, "output_norm.weight", T_F32, dm, 0, err, err_len)))
        goto fail;
    /* tied embeddings (the dense Qwen3): the table is also the output head */
    m->output = janas_jns_tensor(&m->j, "output.weight")
                    ? need(&m->j, "output.weight", T_QUANT, dm, 0, err, err_len)
                    : m->tok_embd;
    if (!m->output)
        goto fail;
    m->n_vocab = (uint32_t)m->output->dims[1];
    if (m->tok_embd->dims[1] != m->n_vocab) {
        snprintf(err, err_len, "embedding and output vocabularies differ");
        goto fail;
    }
    m->layers = calloc(m->n_layer, sizeof(*m->layers));
    if (!m->layers)
        goto fail;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        struct layer *ly = &m->layers[l];
        ly->rec = m->a->rec && (l + 1) % interval != 0;
        ly->slot = ly->rec ? m->n_rec++ : m->n_attn++;
        if (swa_pat) {
            ly->swa = swa_pat[l] != 0;
            ly->hd = ly->swa ? m->head_swa : m->head_dim;
            ly->n_kv = ly->swa ? m->n_kv_swa : m->n_head_kv;
            ly->n_rot = ly->swa ? m->n_rot_swa : m->n_rot;
        }
        if (load_layer(m, &m->j, l, ly, err, err_len) != 0)
            goto fail;
    }

    /* the length the model was trained for: asking for more without a rope
       scaling the file does not carry gives text that falls apart, so the
       context stops there and the caller is told (janas_llm_describe) */
    m->n_ctx_train = 0;
    meta_u32(m, "context_length", &m->n_ctx_train);
    m->n_ctx = o->n_ctx;
    if (m->n_ctx_train && m->n_ctx > m->n_ctx_train)
        m->n_ctx = m->n_ctx_train;
    if (o->mtp_path && load_mtp(m, o->mtp_path, err, err_len) != 0)
        goto fail;
    if (!m->mtp && m->j.h.n_layer > m->n_layer &&
        load_mtp_internal(m, err, err_len) != 0)
        goto fail;
    if (m->a->swa) {
        /* the full layers' RoPE factors (theta divided by them, pair by
           pair: Gemma 4 turns only the first quarter of a head) */
        const struct janas_jns_tensor *rf =
            janas_jns_tensor(&m->j, "rope_freqs.weight");
        if (rf && (rf->type != 0 || rf->dims[0] != m->n_rot / 2)) {
            snprintf(err, err_len, "tensor rope_freqs.weight: unexpected");
            goto fail;
        }
        m->rope_freqs = rf ? f32(m, rf) : NULL;
    }
    /* the KV cache, slot after slot, each with its own geometry */
    uint32_t n_slots = m->n_attn + (m->mtp ? 1 : 0);
    m->kv_off = malloc((n_slots + 1) * sizeof(size_t));
    m->ks_off = malloc((n_slots + 1) * sizeof(size_t));
    m->kv_hd = calloc(n_slots + 1, sizeof(uint32_t));
    m->kv_heads = calloc(n_slots + 1, sizeof(uint32_t));
    if (!m->kv_off || !m->ks_off || !m->kv_hd || !m->kv_heads) {
        snprintf(err, err_len, "out of memory");
        goto fail;
    }
    for (uint32_t l = 0; l <= m->n_layer; l++) {
        const struct layer *ly = l < m->n_layer ? &m->layers[l] : &m->mtp_layer;
        if (ly->rec || (l == m->n_layer && !m->mtp))
            continue;
        m->kv_hd[ly->slot] = ly->hd;
        m->kv_heads[ly->slot] = ly->n_kv;
    }
    size_t kv_len = 0, ks_len = 0;
    for (uint32_t s = 0; s < n_slots; s++) {
        m->kv_off[s] = kv_len;
        m->ks_off[s] = ks_len;
        kv_len += (size_t)m->n_ctx * m->kv_heads[s] * m->kv_hd[s];
        ks_len += (size_t)m->n_ctx * m->kv_heads[s];
    }
    m->kv_off[n_slots] = kv_len;
    m->ks_off[n_slots] = ks_len;
    m->kcache = malloc(kv_len ? kv_len : 1);
    m->vcache = malloc(kv_len ? kv_len : 1);
    m->kscale = malloc((ks_len ? ks_len : 1) * sizeof(float));
    m->vscale = malloc((ks_len ? ks_len : 1) * sizeof(float));
    {
        const char *pb = getenv("JANAS_PREFILL_BLOCK");
        long v = pb ? atol(pb) : 256;
        m->blk = v < JANAS_LLM_MAX_BLOCK ? JANAS_LLM_MAX_BLOCK
                 : v > 4096              ? 4096
                                         : (uint32_t)v;
    }
    size_t B = m->blk, P = B * (m->k + 1);
    size_t inner = m->d_inner > qd ? m->d_inner : qd;
    size_t bigw = m->qkvz_dim > 2 * qd ? m->qkvz_dim : 2 * qd;
    m->x = malloc(B * dm * sizeof(float));
    m->xn = malloc(B * dm * sizeof(float));
    m->q = malloc(B * qd * sizeof(float));
    m->kk = malloc(B * kvd * sizeof(float));
    m->vv = malloc(B * kvd * sizeof(float));
    m->att = malloc(B * inner * sizeof(float));
    m->router = malloc(B * m->n_expert * sizeof(float));
    m->rope_cos = malloc(B * m->n_rot / 2 * sizeof(float));
    m->rope_sin = malloc(B * m->n_rot / 2 * sizeof(float));
    if (m->a->swa) {
        m->rope_cos_swa = malloc(B * m->n_rot_swa / 2 * sizeof(float));
        m->rope_sin_swa = malloc(B * m->n_rot_swa / 2 * sizeof(float));
        m->xres = malloc(B * dm * sizeof(float));
        if (!m->rope_cos_swa || !m->rope_sin_swa || !m->xres) {
            snprintf(err, err_len, "out of memory");
            goto fail;
        }
    }
    m->xq = malloc(B * dm / JANAS_QK * sizeof(*m->xq));
    m->attq = malloc(B * inner / JANAS_QK * sizeof(*m->attq));
    m->pair_tok = malloc(P * sizeof(uint32_t));
    m->pair_exp = malloc(P * sizeof(uint32_t));
    m->order = malloc(P * sizeof(uint32_t));
    m->glist = malloc(P * sizeof(uint32_t));
    m->count = malloc((m->n_expert + 1) * sizeof(uint32_t));
    m->pair_w = malloc(P * sizeof(float));
    m->gate = malloc(P * ff * sizeof(float));
    m->up = malloc(P * ff * sizeof(float));
    m->h = malloc(P * ff * sizeof(float));
    m->dout = malloc(P * dm * sizeof(float));
    m->xg = malloc(P * dm / JANAS_QK * sizeof(*m->xg));
    m->hq = malloc(P * ff / JANAS_QK * sizeof(*m->hq));
    m->tok_first = malloc((B + 1) * sizeof(uint32_t));
    m->tok_fill = malloc(B * sizeof(uint32_t));
    m->tok_list = malloc(P * sizeof(uint32_t));
    if (!m->tok_first || !m->tok_fill || !m->tok_list || !m->kcache ||
        !m->vcache || !m->kscale || !m->vscale || !m->x || !m->xn || !m->q ||
        !m->kk || !m->vv || !m->att || !m->router || !m->rope_cos ||
        !m->rope_sin || !m->xq || !m->attq || !m->pair_tok || !m->pair_exp ||
        !m->order || !m->glist || !m->count || !m->pair_w || !m->gate ||
        !m->up || !m->h || !m->dout || !m->xg || !m->hq || m->n_expert > 4096) {
        snprintf(err, err_len, "out of memory");
        goto fail;
    }
#if defined(__x86_64__)
    m->avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    m->trace = getenv("JANAS_TRACE") ? atoi(getenv("JANAS_TRACE")) : 0;
    /* how much of an expert slot to read: the file may hold the down matrix
       in bit planes, and a machine short of memory reads fewer of them */
    {
        int bits = o->expert_bits;
        const char *eb = getenv("JANAS_EXPERT_BITS");
        if (eb)
            bits = atoi(eb);
        for (uint32_t l = 0; l < m->j.h.n_layer; l++)
            m->exp_planes |= m->j.layers[l].plane[0].type != 0;
        if (!bits && m->exp_planes) {
            /* nobody asked: keep every bit when the experts all fit in
               memory, since then nothing is read while generating anyway;
               otherwise four bits, which read a third to a half fewer bytes
               and cost five to seven positions in four hundred */
            bits = o->cache_bytes >= m->j.h.experts_bytes ? 6 : 4;
            m->exp_auto = 1;
        }
        m->exp_level = bits == 2 ? 1 : bits == 4 ? 2 : 3;
        if (!m->exp_planes) /* a file without planes: every level is the whole
                               slot, and the weights are the quantized ones */
            m->exp_level = 3;
    }
    m->k_use = m->k;
    /*
     * How many experts a token uses: all of them while the cache holds a
     * fair share of the model, fewer only where it does not and the machine
     * would wait for the disk at every token. Measured on 20 Sep 2026 with
     * Qwen3-Next (400 tokens of C, four bits): with a cache of 26% of the
     * experts, using six instead of ten gains 14%; with 13%, 24%; with 6%,
     * 46%. The quality given up is the same everywhere (in Italian prose,
     * 22 most-likely tokens in 400 with eight experts and 55 with six), so
     * it is only worth it where the gain is large.
     */
    if (o->cache_bytes && m->j.h.experts_bytes) {
        double cover = (double)o->cache_bytes / (double)m->j.h.experts_bytes;
        uint32_t k = m->k;
        if (cover < 0.10)
            k = m->k * 6 / 10;
        else if (cover < 0.20)
            k = m->k * 8 / 10;
        if (k < 1)
            k = 1;
        if (k != m->k) {
            janas_llm_model_set_experts(m, k);
            m->k_auto = 1;
        }
    }
    if (getenv("JANAS_EXPERTS")) { /* fewer experts per token, for measures */
        janas_llm_model_set_experts(m, (uint32_t)atoi(getenv("JANAS_EXPERTS")));
        m->k_auto = 0;
    }
    {
        const char *vm = getenv("JANAS_VMAP");
        m->vmap_mod = vm ? strcmp(vm, "mod") == 0 : m->a->vmap_mod;
    }
#if defined(__x86_64__)
    m->f16c = __builtin_cpu_supports("f16c");
#endif
    if (m->a->attn_gate || m->a->rec) {
        /* the gated queries, and the DeltaNet's fused projections */
        m->big = malloc(B * bigw * sizeof(float));
        m->qgate = malloc(B * qd * sizeof(float));
        if (!m->big || !m->qgate) {
            snprintf(err, err_len, "out of memory");
            goto fail;
        }
    }
    if (m->a->rec) {
        m->ssm_buf = calloc(2 * (size_t)m->n_rec * m->n_vh * m->ds * m->ds,
                            sizeof(uint16_t));
        m->conv_buf = calloc(
            2 * (size_t)m->n_rec * m->conv_ch * (m->d_conv - 1), sizeof(float));
        m->ba = malloc(B * 2 * m->n_vh * sizeof(float));
        /* the rows of the replayed tokens (a logged call, at most
           JANAS_LLM_MAX_BLOCK), then the block */
        size_t L = JANAS_LLM_MAX_BLOCK, R = L + B;
        m->mixed = malloc(R * m->conv_ch * sizeof(float));
        m->conv_out = malloc(R * m->conv_ch * sizeof(float));
        m->beta = malloc(R * m->n_vh * sizeof(float));
        m->decay = malloc(R * m->n_vh * sizeof(float));
        m->log_mixed =
            malloc((size_t)m->n_rec * L * m->conv_ch * sizeof(float));
        m->log_beta = malloc((size_t)m->n_rec * L * m->n_vh * sizeof(float));
        m->log_decay = malloc((size_t)m->n_rec * L * m->n_vh * sizeof(float));
        if (!m->ssm_buf || !m->conv_buf || !m->ba || !m->mixed ||
            !m->conv_out || !m->beta || !m->decay || !m->log_mixed ||
            !m->log_beta || !m->log_decay) {
            snprintf(err, err_len, "out of memory");
            goto fail;
        }
    }

    struct janas_cpu_layout lay;
    int pin = -1;
    if (o->compute_cpus &&
        janas_cpu_layout(o->compute_cpus, o->n_compute, &lay) == 0) {
        m->compute = o->shared_compute ? o->shared_compute
                                       : janas_pool_create(lay.n, lay.cpus);
        m->own_compute = !o->shared_compute;
        m->p_cores = lay.n_p_cores;
        m->p_threads = lay.n_p_threads;
        m->all_threads = lay.n;
        pin = o->shared_compute ? -1 : lay.cpus[0];
    } else {
        m->compute = o->shared_compute ? o->shared_compute
                                       : janas_pool_create(o->n_compute, NULL);
        m->own_compute = !o->shared_compute;
        m->p_cores = m->p_threads = m->all_threads = o->n_compute;
    }
    m->io = janas_pool_create(o->n_io, o->io_cpus);
    if (!m->compute || !m->io)
        goto fail;
    /* the I/O threads wait for NVMe reads of ~100 us: sleeping costs them
       nothing and leaves the power budget to the compute cores */
    janas_pool_set_spin(m->io, 0);
    m->attn = janas_attn_create(m->n_head, m->n_head_kv, m->head_dim, m->n_ctx,
                                janas_pool_size(m->compute), o->attn_scores);
    if (m->attn && m->a->swa) {
        /* Gemma 4: the q and k norms scale the scores; the sliding-window
           layers have an attention of their own */
        janas_attn_set_scale(m->attn, 1.0f);
        m->attn_swa =
            janas_attn_create(m->n_head, m->n_kv_swa, m->head_swa, m->n_ctx,
                              janas_pool_size(m->compute), o->attn_scores);
        if (m->attn_swa) {
            janas_attn_set_scale(m->attn_swa, 1.0f);
            janas_attn_set_window(m->attn_swa, m->window);
        }
    }
    if (!m->attn || (m->a->swa && !m->attn_swa)) {
        snprintf(err, err_len, "out of memory");
        goto fail;
    }
    if (pin >= 0)
        janas_pin_current_thread(pin);
    m->cache = janas_expert_cache_create(&m->j, o->cache_bytes, m->exp_level,
                                         m->io, err, err_len);
    if (!m->cache)
        goto fail;
    /* fill the cache with the experts this machine used most, before the
       first reply has to find them one token at a time. A caller that does
       not want this (a measurement) neither reads the profile nor adds to
       it, so its runs leave the machine's profile as it was */
    {
        const char *w = getenv("JANAS_WARM");
        size_t used = 0;
        m->warm_use = w ? atoi(w) != 0 : o->warm != 0;
        uint32_t *counts = m->warm_use ? janas_m_profile_read(m, &used) : NULL;
        if (m->warm_use) {
            m->use_count = calloc((size_t)m->j.h.n_layer * m->j.h.n_expert,
                                  sizeof(uint32_t));
            if (!m->use_count) {
                free(counts);
                goto fail;
            }
        }
        if (counts) {
            /* on a thread of its own: the first reply does not wait for it */
            if (janas_expert_cache_warm_background(m->cache, counts) == 0) {
                size_t slots = janas_expert_cache_slots(m->cache);
                m->warm_experts = used < slots ? used : slots;
            }
            free(counts);
        }
    }
    if (m->mtp) {
        m->hid = malloc(B * dm * sizeof(float));
        m->mtp_hid = malloc(B * dm * sizeof(float));
        m->cat = malloc(B * 2 * dm * sizeof(float));
        m->catq = malloc(B * 2 * dm / JANAS_QK * sizeof(*m->catq));
        /* its own experts' cache, or the main one (block in the file) */
        m->mtp_cache =
            m->mtp_own_cache
                ? janas_expert_cache_create(
                      &m->mtp_j,
                      o->mtp_cache_bytes ? o->mtp_cache_bytes : 256ull << 20,
                      m->exp_level, m->io, err, err_len)
                : m->cache;
        if (!m->hid || !m->mtp_hid || !m->cat || !m->catq || !m->mtp_cache ||
            init_draft_head(m, o->draft_vocab ? o->draft_vocab : 32768) != 0)
            goto fail;
    }
    if (m->gpu) {
        /* optional: without it (or if the weights cannot be shared with
           it) everything runs on the CPU, with the same results. Memory
           of its driver written by the CPU (reads, converted router) is
           flushed out of the CPU's caches before the GPU reads it */
        if (m->resident_gpu)
            janas_gpu_flush(m->gpu, m->resident, m->j.h.resident_bytes);
        else if (janas_gpu_import(m->gpu, m->resident, m->j.h.resident_bytes) !=
                 0)
            m->gpu_off = 1;
        if (m->mtp_res && janas_gpu_import(m->gpu, m->mtp_res,
                                           m->mtp_j.h.resident_bytes) != 0)
            m->gpu_off = 1;
    }
    janas_m_init_tuner(m);
    return m;

fail:
    janas_llm_model_free(m);
    return NULL;
}

void janas_llm_model_free(struct janas_llm_model *m)
{
    if (janas_m_exp_tracing())
        janas_m_exp_report();
    if (!m)
        return;
    if (m->tune_key[0])
        janas_tuner_save(&m->tuner, m->tune_key);
    if (m->use_count) {
        janas_m_profile_write(m);
        free(m->use_count);
        m->use_count = NULL;
    }
    janas_gpu_destroy(m->gpu); /* before the memory it imported; frees
                                  the resident region it allocated */
    janas_expert_cache_destroy(m->cache);
    if (m->mtp_own_cache)
        janas_expert_cache_destroy(m->mtp_cache);
    janas_attn_destroy(m->attn);
    janas_attn_destroy(m->attn_swa);
    if (m->own_compute)
        janas_pool_destroy(m->compute);
    janas_pool_destroy(m->io);
    if (!m->resident_gpu)
        free(m->resident);
    if (m->layers)
        for (uint32_t l = 0; l < m->n_layer; l++) {
            free(m->layers[l].sh_gate);
            free(m->layers[l].router_own);
            free(m->layers[l].conv_t);
            free(m->layers[l].alpha_own);
            free(m->layers[l].beta_own);
        }
    free(m->layers);
    float *fl[] = {
        m->conv_buf, m->x,     m->xn,        m->q,        m->kk,
        m->vv,       m->att,   m->router,    m->rope_cos, m->rope_sin,
        m->big,      m->qgate, m->ba,        m->mixed,    m->conv_out,
        m->beta,     m->decay, m->pair_w,    m->gate,     m->up,
        m->h,        m->dout,  m->log_mixed, m->log_beta, m->log_decay};
    for (size_t i = 0; i < sizeof(fl) / sizeof(fl[0]); i++)
        free(fl[i]);
    free(m->kscale);
    free(m->vscale);
    free(m->kv_off);
    free(m->ks_off);
    free(m->kv_hd);
    free(m->kv_heads);
    free(m->rope_cos_swa);
    free(m->rope_sin_swa);
    free(m->xres);
    free(m->kcache);
    free(m->vcache);
    free(m->ssm_buf);
    free(m->xq);
    free(m->attq);
    free(m->pair_tok);
    free(m->pair_exp);
    free(m->order);
    free(m->glist);
    free(m->count);
    free(m->xg);
    free(m->hq);
    free(m->tok_first);
    free(m->tok_fill);
    free(m->tok_list);
    if (m->mtp_open)
        janas_jns_close(&m->mtp_j);
    free(m->mtp_res);
    free(m->hid);
    free(m->mtp_hid);
    free(m->cat);
    free(m->catq);
    free(m->dh_w);
    free(m->dh_in);
    free(m->dh_ids);
    free(m->dh_logits);
    free(m->mtp_layer.sh_gate);
    janas_jns_close(&m->j);
    free(m);
}

uint32_t janas_llm_model_n_vocab(const struct janas_llm_model *m)
{
    return m->n_vocab;
}

uint32_t janas_llm_model_n_embd(const struct janas_llm_model *m)
{
    return m->d_model;
}

uint32_t janas_llm_model_experts(const struct janas_llm_model *m,
                                 uint32_t *most)
{
    if (most)
        *most = m->k;
    return m->k_use;
}

int janas_llm_model_set_experts(struct janas_llm_model *m, uint32_t n)
{
    if (n == 0)
        n = m->k;
    if (n > m->k)
        return 0;
    m->k_use = n;
    return 1;
}

struct janas_gpu *janas_llm_model_gpu(struct janas_llm_model *m)
{
    return m->gpu;
}

uint32_t janas_llm_model_n_ctx(const struct janas_llm_model *m)
{
    return m->n_ctx;
}

uint32_t janas_llm_model_n_ctx_train(const struct janas_llm_model *m)
{
    return m->n_ctx_train;
}

int janas_llm_model_attn_scores(const struct janas_llm_model *m)
{
    return janas_attn_scores_mode(m->attn);
}

int janas_llm_model_attn_scores_auto(const struct janas_llm_model *m)
{
    return janas_attn_scores_auto(m->attn);
}

int janas_llm_model_expert_bits(const struct janas_llm_model *m)
{
    return 2 * m->exp_level;
}

int janas_llm_model_expert_bits_auto(const struct janas_llm_model *m)
{
    return m->exp_auto;
}

int janas_llm_model_experts_auto(const struct janas_llm_model *m)
{
    return m->k_auto;
}

uint64_t janas_llm_model_warm(const struct janas_llm_model *m, double *seconds)
{
    if (seconds)
        *seconds = m->warm_seconds;
    return m->warm_experts;
}

int janas_llm_model_settling(const struct janas_llm_model *m)
{
    return m->cache && janas_expert_cache_warming(m->cache);
}

uint64_t janas_llm_model_warm_done(const struct janas_llm_model *m)
{
    uint64_t done = 0;
    if (m->cache)
        janas_expert_cache_warm_progress(m->cache, &done, NULL);
    return done;
}

struct janas_expert_cache *janas_llm_model_cache(struct janas_llm_model *m)
{
    return m->cache;
}
