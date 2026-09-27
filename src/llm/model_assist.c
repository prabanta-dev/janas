/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_assist.c - Gemma 4's assistant: the drafts of a small model of its
 * own (gemma4-assistant, a JNS file given as the MTP block).
 *
 * The assistant is narrow (256 wide for E2B and E4B) and has a few layers
 * (four) with queries only: it keeps no keys and values, and reads the main
 * model's - its sliding-window layers the main model's last sliding layer's
 * cache, its full layer the last full one's. Its input is the main model's
 * embedding of the last token beside the main model's final state (after
 * the final norm), projected to its width; its output, logits over the
 * whole vocabulary and a state projected back to the main model's width,
 * on which the next draft is chained.
 *
 * A draft of token t_{p+1} (Janas's row: the token with the main model's
 * state h_p, "at" p) runs at t_{p+1}'s own position, p + 1, and attends
 * the main model's cache up to p: the token itself is not in it yet. The
 * drafts chained after it keep that same position, as the model was
 * trained (Hugging Face's gemma4_assistant), and so read the same cache.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "matvec.h"
#include "vmath.h"

#include "model_impl.h"

#define ASST_MAX_LAYERS 8

struct asst_layer {
    int swa;
    uint32_t hd, n_rot, slot; /* slot: the main model's KV it reads */
    const struct janas_jns_tensor *attn_norm, *wq, *q_norm, *wo, *post_attn,
        *ffn_norm, *post_ffn;
    float out_scale;
    const uint8_t *ffn[3]; /* gate, up, down: in the slot as read */
    struct janas_jns_matrix mx[3];
};

struct janas_assist {
    struct janas_jns j;
    int open;
    uint8_t *res, *slots;
    uint32_t n_layer, d, n_head, ff, hd_max;
    float eps;
    const float *rope_freqs;
    float rope_base, rope_base_swa;
    const struct janas_jns_tensor *embd, *out_norm, *pre, *post;
    struct asst_layer L[ASST_MAX_LAYERS];
    struct janas_attn *attn, *attn_swa;
    /* one row's scratch */
    float *cat, *x, *xn, *q, *att, *g, *u, *logits, *cs, *sn, *cs_swa, *sn_swa;
    struct janas_block_q8k *catq, *xq, *attq, *hq;
};

static const void *adata(const struct janas_assist *a,
                         const struct janas_jns_tensor *t)
{
    return a->res + (t->offset - a->j.h.resident_offset);
}

static int ameta_u32(const struct janas_assist *a, const char *suffix,
                     uint32_t *v)
{
    char key[128];
    snprintf(key, sizeof(key), "gemma4-assistant.%s", suffix);
    return janas_jns_meta_u32(&a->j, key, v);
}

static int ameta_f32(const struct janas_assist *a, const char *suffix, float *v)
{
    char key[128];
    snprintf(key, sizeof(key), "gemma4-assistant.%s", suffix);
    return janas_jns_meta_f32(&a->j, key, v);
}

/* A tensor of the assistant's file, of the expected shape (d1 0: any);
   f32: of type f32, else of a type the products know. */
static const struct janas_jns_tensor *aneed(const struct janas_assist *a,
                                            const char *name, int f32,
                                            uint64_t d0, uint64_t d1)
{
    const struct janas_jns_tensor *t = janas_jns_tensor(&a->j, name);
    if (!t || t->dims[0] != d0 || (d1 && t->dims[1] != d1))
        return NULL;
    if (f32 ? t->type != 0 : janas_qtype_block_size((int)t->type) == 0)
        return NULL;
    return t;
}

int janas_m_assist_is(const char *path)
{
    struct janas_jns j;
    char err[256];
    if (janas_jns_open(path, &j, err, sizeof(err)) != 0)
        return 0;
    int is = strncmp(j.h.arch, "gemma4-assistant", sizeof(j.h.arch)) == 0;
    janas_jns_close(&j);
    return is;
}

void janas_m_assist_free(struct janas_llm_model *m)
{
    struct janas_assist *a = m->asst;
    if (!a)
        return;
    if (a->open)
        janas_jns_close(&a->j);
    janas_attn_destroy(a->attn);
    janas_attn_destroy(a->attn_swa);
    free(a->res);
    free(a->slots);
    free(a->cat);
    free(a->x);
    free(a->xn);
    free(a->q);
    free(a->att);
    free(a->g);
    free(a->u);
    free(a->logits);
    free(a->cs);
    free(a->sn);
    free(a->cs_swa);
    free(a->sn_swa);
    free(a->catq);
    free(a->xq);
    free(a->attq);
    free(a->hq);
    free(a);
    m->asst = NULL;
}

static int fail(char *err, size_t err_len, const char *path, const char *why)
{
    snprintf(err, err_len, "assistant %s: %s", path, why);
    return -1;
}

int janas_m_assist_load(struct janas_llm_model *m, const char *path,
                        int attn_scores, char *err, size_t err_len)
{
    struct janas_assist *a = calloc(1, sizeof(*a));
    if (!a)
        return fail(err, err_len, path, "out of memory");
    m->asst = a;
    char why[256];
    if (janas_jns_open(path, &a->j, why, sizeof(why)) != 0)
        return fail(err, err_len, path, why);
    a->open = 1;
    const struct janas_jns_header *h = &a->j.h;
    uint32_t dm = m->d_model, out = 0, kl = 0, kl_swa = 0, shared = 0;
    const uint8_t *pat = NULL;
    uint64_t np = 0;
    a->n_layer = h->n_layer;
    a->d = h->d_model;
    if (!m->a->swa || m->n_layer < 2 || a->n_layer == 0 ||
        a->n_layer > ASST_MAX_LAYERS ||
        ameta_u32(a, "embedding_length_out", &out) || out != dm ||
        ameta_u32(a, "attention.head_count", &a->n_head) ||
        ameta_u32(a, "attention.key_length", &kl) ||
        ameta_u32(a, "attention.key_length_swa", &kl_swa) ||
        ameta_u32(a, "attention.shared_kv_layers", &shared) ||
        ameta_f32(a, "attention.layer_norm_rms_epsilon", &a->eps) ||
        ameta_f32(a, "rope.freq_base", &a->rope_base) ||
        ameta_f32(a, "rope.freq_base_swa", &a->rope_base_swa) ||
        janas_jns_meta_bytes(
            &a->j, "gemma4-assistant.attention.sliding_window_pattern", &np,
            &pat) ||
        np < a->n_layer || shared != a->n_layer || kl != m->head_dim ||
        kl_swa != m->head_swa || a->d % JANAS_QK || a->n_head == 0 ||
        a->n_head % m->n_head_kv || a->n_head % m->n_kv_swa)
        return fail(err, err_len, path, "not an assistant of this model");
    /* its KV heads, one count or one per layer (12B, 26B): those of the
       main model's caches it reads */
    uint32_t kv_one = 0;
    uint64_t n_kv_arr = 0;
    const int32_t *kv_arr = NULL;
    if (janas_jns_meta_ints(&a->j, "gemma4-assistant.attention.head_count_kv",
                            &n_kv_arr, &kv_arr) != 0 ||
        n_kv_arr < a->n_layer) {
        kv_arr = NULL;
        if (ameta_u32(a, "attention.head_count_kv", &kv_one))
            return fail(err, err_len, path, "no KV head count");
    }
    a->ff = a->j.layers[0].m[JANAS_JNS_GATE].rows;
    a->hd_max = kl > kl_swa ? kl : kl_swa;
    /* the resident tensors, then each layer's feed-forward (its one slot) */
    a->res = aligned_alloc(4096, h->resident_bytes);
    uint64_t sb = 0;
    for (uint32_t l = 0; l < a->n_layer; l++)
        if (a->j.layers[l].slot_bytes > sb)
            sb = a->j.layers[l].slot_bytes;
    a->slots = aligned_alloc(4096, sb * a->n_layer);
    if (!a->res || !a->slots)
        return fail(err, err_len, path, "out of memory");
    struct {
        uint8_t *dst;
        uint64_t off, len;
    } rd[1 + ASST_MAX_LAYERS];
    rd[0].dst = a->res;
    rd[0].off = h->resident_offset;
    rd[0].len = h->resident_bytes;
    for (uint32_t l = 0; l < a->n_layer; l++) {
        rd[1 + l].dst = a->slots + (size_t)l * sb;
        rd[1 + l].off = janas_jns_expert_offset(&a->j, l, 0);
        rd[1 + l].len = a->j.layers[l].slot_bytes;
    }
    for (uint32_t i = 0; i <= a->n_layer; i++)
        for (uint64_t done = 0; done < rd[i].len;) {
            ssize_t r = pread(a->j.fd, rd[i].dst + done, rd[i].len - done,
                              (off_t)(rd[i].off + done));
            if (r <= 0)
                return fail(err, err_len, path, "cannot read it");
            done += (uint64_t)r;
        }
    uint32_t d = a->d;
    if (!(a->embd = aneed(a, "token_embd.weight", 0, d, m->n_vocab)) ||
        !(a->out_norm = aneed(a, "output_norm.weight", 1, d, 0)) ||
        !(a->pre = aneed(a, "nextn.pre_projection.weight", 0, 2 * dm, d)) ||
        !(a->post = aneed(a, "nextn.post_projection.weight", 0, d, dm)))
        return fail(err, err_len, path, "tensors missing or unexpected");
    const struct janas_jns_tensor *rf =
        janas_jns_tensor(&a->j, "rope_freqs.weight");
    if (rf && (rf->type != 0 || rf->dims[0] != m->n_rot / 2))
        return fail(err, err_len, path, "rope_freqs unexpected");
    a->rope_freqs = rf ? adata(a, rf) : NULL;
    for (uint32_t l = 0; l < a->n_layer; l++) {
        struct asst_layer *L = &a->L[l];
        L->swa = pat[l] != 0;
        L->hd = L->swa ? m->head_swa : m->head_dim;
        L->n_rot = L->swa ? m->n_rot_swa : m->n_rot;
        /* the main model's last layer of the same kind */
        const struct layer *src = &m->layers[m->n_layer - (L->swa ? 2 : 1)];
        int32_t kvh = (int32_t)kv_one;
        if (kv_arr)
            memcpy(&kvh, kv_arr + l, sizeof(kvh));
        if (src->swa != L->swa || kvh <= 0 || src->n_kv != (uint32_t)kvh)
            return fail(err, err_len, path, "its layers do not match");
        L->slot = src->slot;
        uint32_t qd = a->n_head * L->hd;
        char name[64];
#define T(field, suffix, f32, d0, d1)                                          \
    snprintf(name, sizeof(name), "blk.%u." suffix, l);                         \
    if (!(L->field = aneed(a, name, f32, d0, d1)))                             \
        return fail(err, err_len, path, "a layer's tensors");
        T(attn_norm, "attn_norm.weight", 1, d, 0)
        T(wq, "attn_q.weight", 0, d, qd)
        T(q_norm, "attn_q_norm.weight", 1, L->hd, 0)
        T(wo, "attn_output.weight", 0, qd, d)
        T(post_attn, "post_attention_norm.weight", 1, d, 0)
        T(ffn_norm, "ffn_norm.weight", 1, d, 0)
        T(post_ffn, "post_ffw_norm.weight", 1, d, 0)
#undef T
        snprintf(name, sizeof(name), "blk.%u.layer_output_scale.weight", l);
        const struct janas_jns_tensor *os = janas_jns_tensor(&a->j, name);
        L->out_scale = os && os->type == 0 && os->dims[0] == 1
                           ? *(const float *)adata(a, os)
                           : 1.0f;
        for (int p = 0; p < 3; p++) {
            L->mx[p] = a->j.layers[l].m[p];
            L->ffn[p] = a->slots + (size_t)l * sb + L->mx[p].offset;
        }
        if (L->mx[JANAS_JNS_GATE].rows != a->ff ||
            L->mx[JANAS_JNS_GATE].cols != d ||
            L->mx[JANAS_JNS_DOWN].rows != d || a->ff % JANAS_QK)
            return fail(err, err_len, path, "a feed-forward's shape");
    }
    /* attention over the main model's caches with the assistant's heads.
       Its query stands at p + 1 but, the token not being in the cache,
       is attended as p: one position less of window keeps what it sees
       p + 1 - window + 1 .. p, as for the main model at p + 1 */
    uint32_t s_full = m->layers[m->n_layer - 1].slot,
             s_swa = m->layers[m->n_layer - 2].slot;
    int n_thr = janas_pool_size(m->compute);
    a->attn = janas_attn_create(a->n_head, m->n_head_kv, m->head_dim,
                                m->kv_cap[s_full], n_thr, attn_scores);
    a->attn_swa = janas_attn_create(a->n_head, m->n_kv_swa, m->head_swa,
                                    m->kv_cap[s_swa], n_thr, attn_scores);
    if (!a->attn || !a->attn_swa)
        return fail(err, err_len, path, "out of memory");
    janas_attn_set_kv16(a->attn, m->kv16);
    janas_attn_set_scale(a->attn, 1.0f);
    janas_attn_set_ring(a->attn, m->kv_ring[s_full]);
    janas_attn_set_kv16(a->attn_swa, m->kv16);
    janas_attn_set_scale(a->attn_swa, 1.0f);
    janas_attn_set_ring(a->attn_swa, m->kv_ring[s_swa]);
    janas_attn_set_window(a->attn_swa, m->window > 1 ? m->window - 1 : 1);
    size_t qd = (size_t)a->n_head * a->hd_max;
    a->cat = malloc(2 * dm * sizeof(float));
    a->x = malloc(d * sizeof(float));
    a->xn = malloc((d > dm ? d : dm) * sizeof(float));
    a->q = malloc(qd * sizeof(float));
    a->att = malloc(qd * sizeof(float));
    a->g = malloc(a->ff * sizeof(float));
    a->u = malloc(a->ff * sizeof(float));
    a->logits = malloc((size_t)m->n_vocab * sizeof(float));
    a->cs = malloc(m->n_rot / 2 * sizeof(float));
    a->sn = malloc(m->n_rot / 2 * sizeof(float));
    a->cs_swa = malloc(m->n_rot_swa / 2 * sizeof(float));
    a->sn_swa = malloc(m->n_rot_swa / 2 * sizeof(float));
    a->catq = malloc(2 * dm / JANAS_QK * sizeof(*a->catq));
    a->xq = malloc(d / JANAS_QK * sizeof(*a->xq));
    a->attq = malloc(qd / JANAS_QK * sizeof(*a->attq));
    a->hq = malloc(a->ff / JANAS_QK * sizeof(*a->hq));
    if (!m->hid)
        m->hid = malloc((size_t)m->blk * dm * sizeof(float));
    if (!m->mtp_hid)
        m->mtp_hid = malloc((size_t)m->blk * dm * sizeof(float));
    if (!a->cat || !a->x || !a->xn || !a->q || !a->att || !a->g || !a->u ||
        !a->logits || !a->cs || !a->sn || !a->cs_swa || !a->sn_swa ||
        !a->catq || !a->xq || !a->attq || !a->hq || !m->hid || !m->mtp_hid)
        return fail(err, err_len, path, "out of memory");
    return 0;
}

/*
 * The draft head, as the MTP block's (model_load.c): the assistant's own
 * output rows for the base lowest token ids - SentencePiece's most frequent
 * pieces come first - and every token met in the conversation. Its full
 * head is the whole vocabulary, 67 MB read for every draft on E4B; 32768
 * rows are 8 MB. base >= the vocabulary keeps it whole.
 */
int janas_m_assist_head(struct janas_llm_model *m, uint32_t base)
{
    const struct janas_assist *a = m->asst;
    return janas_m_init_draft_head(
        m, a->res + (a->embd->offset - a->j.h.resident_offset),
        a->d / JANAS_QK * janas_qtype_block_size((int)a->embd->type), base);
}

/* y = W x, x in Q8_K, on the compute pool */
static void mv(struct janas_llm_model *m, int type, const void *w,
               const struct janas_block_q8k *x, float *y, uint32_t rows,
               uint32_t cols)
{
    struct janas_matvec_task t = {.type = type,
                                  .w = w,
                                  .x = x,
                                  .y = y,
                                  .rows = rows,
                                  .cols = cols,
                                  .n_vec = 1};
    janas_gpu_matvec_group(NULL, m->compute, &t, 1);
}

static void mvt(struct janas_llm_model *m, const struct janas_assist *a,
                const struct janas_jns_tensor *t,
                const struct janas_block_q8k *x, float *y)
{
    mv(m, (int)t->type, adata(a, t), x, y, (uint32_t)t->dims[1],
       (uint32_t)t->dims[0]);
}

static void rope_at(float *cs, float *sn, uint32_t pos, uint32_t n_rot,
                    float base, const float *ff)
{
    for (uint32_t i = 0; i < n_rot / 2; i++) {
        float theta = (float)pos * powf(base, -2.0f * (float)i / (float)n_rot);
        if (ff)
            theta /= ff[i];
        cs[i] = cosf(theta);
        sn[i] = sinf(theta);
    }
}

static void rope_neox(float *v, uint32_t n, const float *cs, const float *sn)
{
    uint32_t half = n / 2;
    for (uint32_t i = 0; i < half; i++) {
        float x0 = v[i], x1 = v[i + half];
        v[i] = x0 * cs[i] - x1 * sn[i];
        v[i + half] = x0 * sn[i] + x1 * cs[i];
    }
}

/*
 * The draft after token (whose state from the main model, or from the
 * assistant's previous draft, is h) for position p + 1, the main model's
 * cache holding up to p: the likeliest token, its probability, and the
 * state to chain the next draft on (hnext, d_model wide).
 */
int janas_m_assist_draft(struct janas_llm_model *m, int32_t token,
                         const float *h, uint32_t p, float *hnext, int32_t *out,
                         float *conf)
{
    struct janas_assist *a = m->asst;
    uint32_t dm = m->d_model, d = a->d;
    if (p + 1 >= m->n_ctx || token < 0 || (uint32_t)token >= m->n_vocab)
        return -1;
    double t0 = now();
    /* [embedding of the token, as the main model scales it; h] */
    janas_m_embed(m, token, a->cat);
    memcpy(a->cat + dm, h, dm * sizeof(float));
    janas_q8k_quantize(a->cat, a->catq, 2 * dm);
    mvt(m, a, a->pre, a->catq, a->x);
    rope_at(a->cs, a->sn, p + 1, m->n_rot, a->rope_base, a->rope_freqs);
    rope_at(a->cs_swa, a->sn_swa, p + 1, m->n_rot_swa, a->rope_base_swa, NULL);
    for (uint32_t l = 0; l < a->n_layer; l++) {
        const struct asst_layer *L = &a->L[l];
        uint32_t qd = a->n_head * L->hd;
        rms_norm(a->xn, a->x, adata(a, L->attn_norm), d, a->eps);
        janas_q8k_quantize(a->xn, a->xq, d);
        mvt(m, a, L->wq, a->xq, a->q);
        for (uint32_t hh = 0; hh < a->n_head; hh++) {
            float *v = a->q + (size_t)hh * L->hd;
            rms_norm(v, v, adata(a, L->q_norm), L->hd, a->eps);
            rope_neox(v, L->n_rot, L->swa ? a->cs_swa : a->cs,
                      L->swa ? a->sn_swa : a->sn);
        }
        janas_attn_run(L->swa ? a->attn_swa : a->attn, m->compute, a->q,
                       m->kcache + m->kv_off[L->slot],
                       m->kscale + m->ks_off[L->slot],
                       m->vcache + m->kv_off[L->slot],
                       m->vscale + m->ks_off[L->slot], 1, p, a->att);
        janas_q8k_quantize(a->att, a->attq, qd);
        mvt(m, a, L->wo, a->attq, a->xn);
        rms_norm(a->xn, a->xn, adata(a, L->post_attn), d, a->eps);
        for (uint32_t i = 0; i < d; i++)
            a->x[i] += a->xn[i];
        rms_norm(a->xn, a->x, adata(a, L->ffn_norm), d, a->eps);
        janas_q8k_quantize(a->xn, a->xq, d);
        mv(m, (int)L->mx[0].type, L->ffn[0], a->xq, a->g, a->ff, d);
        mv(m, (int)L->mx[1].type, L->ffn[1], a->xq, a->u, a->ff, d);
        for (uint32_t i = 0; i < a->ff; i++)
            a->g[i] = janas_gelu_mul_det(a->g[i], a->u[i]);
        janas_q8k_quantize(a->g, a->hq, a->ff);
        mv(m, (int)L->mx[2].type, L->ffn[2], a->hq, a->xn, d, a->ff);
        rms_norm(a->xn, a->xn, adata(a, L->post_ffn), d, a->eps);
        for (uint32_t i = 0; i < d; i++)
            a->x[i] = (a->x[i] + a->xn[i]) * L->out_scale;
    }
    rms_norm(a->xn, a->x, adata(a, a->out_norm), d, a->eps);
    janas_q8k_quantize(a->xn, a->xq, d);
    mv(m, (int)a->embd->type, m->dh_w, a->xq, a->logits, m->dh_n, d);
    mvt(m, a, a->post, a->xq, hnext);
    /* the likeliest token of the head and its probability within it */
    const float *lg = a->logits;
    uint32_t best = 0;
    for (uint32_t i = 1; i < m->dh_n; i++)
        if (lg[i] > lg[best])
            best = i;
    *out = m->dh_ids[best];
    *conf = (float)(1.0 / janas_m_exp_sum(lg, m->dh_n, lg[best]));
    m->phase[JANAS_PH_OUTPUT] += now() - t0;
    return 0;
}
