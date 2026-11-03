/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf2jns_whisper.c - a Whisper model (speech to text), from its Hugging
 * Face checkpoint (openai/whisper-*: safetensors, config.json,
 * generation_config.json, preprocessor_config.json, tokenizer.json) into a
 * JNS file of resident tensors: <name>.jns, read by src/llm/whisper.c.
 *
 * The encoder - two convolutions over the log-mel spectrogram, sinusoidal
 * positions, blocks of LayerNorm, attention and a GELU MLP - and the
 * decoder - token and learned position embeddings, blocks of causal
 * attention, attention over the encoder's output and an MLP, the token
 * embeddings again for the logits. The matrices are kept in half
 * precision, as OpenAI trained and released them (the checkpoints' float32
 * are their exact copies: a value that is not is rounded and counted), and
 * transposed - a row per input, its outputs contiguous - as the engine's
 * tiled products read them, the token embeddings too; the norms, biases
 * and positions in float. The metadata holds the sizes, the
 * spectrogram's mel filters (as the processor has them, or as librosa
 * makes them where it no longer does: large-v3), the special
 * tokens, the languages, the tokens never to be sampled, and the
 * vocabulary with its merges for the engine's tokenizer.
 *
 * Usage: hf2jns_whisper <checkpoint directory> <out.jns>
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/hf_safetensors.h"
#include "llm/jns_write.h"
#include "llm/json.h"
#include "llm/quant.h"

#define GGML_F32 0
#define GGML_F16 1

static _Noreturn void die(const char *msg, const char *arg);

/* Slaney's mel scale, librosa's (htk=False) */
static double hz_to_mel(double f)
{
    const double sp = 200.0 / 3, min_log = 1000, logstep = log(6.4) / 27;
    return f < min_log ? f / sp : min_log / sp + log(f / min_log) / logstep;
}

static double mel_to_hz(double m)
{
    const double sp = 200.0 / 3, min_log = 1000, logstep = log(6.4) / 27;
    return m < min_log / sp ? m * sp
                            : min_log * exp(logstep * (m - min_log / sp));
}

/* The filters OpenAI's mel_filters.npz holds, which the processors of the
   first checkpoints carried: librosa.filters.mel(sr=rate, n_fft, n_mels,
   norm="slaney") in float32 - triangles on the mel scale from 0 Hz to half
   the rate, rounded to float, then scaled by 2 / their width in Hz and
   rounded again, as librosa's float32 array does. */
static void mel_filters(float *w, uint32_t mels, uint32_t n_fft, uint32_t rate)
{
    uint32_t bins = n_fft / 2 + 1;
    double *mf = malloc(sizeof(double) * (mels + 2));
    if (!mf)
        die("out of memory", NULL);
    /* numpy's linspace and rfftfreq, operation for operation */
    double lo = hz_to_mel(0), hi = hz_to_mel(rate / 2.0),
           step = (hi - lo) / (mels + 1), df = 1.0 / (n_fft * (1.0 / rate));
    for (uint32_t i = 0; i < mels + 2; i++)
        mf[i] = mel_to_hz(i == mels + 1 ? hi : i * step + lo);
    for (uint32_t i = 0; i < mels; i++) {
        double enorm = 2.0 / (mf[i + 2] - mf[i]);
        for (uint32_t k = 0; k < bins; k++) {
            double f = k * df;
            double lower = -(mf[i] - f) / (mf[i + 1] - mf[i]),
                   upper = (mf[i + 2] - f) / (mf[i + 2] - mf[i + 1]);
            double v = lower < upper ? lower : upper;
            float t = (float)(v > 0 ? v : 0);
            w[i * bins + k] = (float)(t * enorm);
        }
    }
    free(mf);
}

static _Noreturn void die(const char *msg, const char *arg)
{
    fprintf(stderr, "hf2jns_whisper: %s%s%s\n", msg, arg ? ": " : "",
            arg ? arg : "");
    exit(1);
}

/* ---- the checkpoint ---- */

static struct janas_st shards[64];
static int n_shards;

static void open_shards(const char *dir)
{
    DIR *d = opendir(dir);
    if (!d)
        die("cannot open", dir);
    struct dirent *e;
    char p[4096], err[300];
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 12 || strcmp(e->d_name + n - 12, ".safetensors"))
            continue;
        if (n_shards == 64)
            die("more than 64 shards in", dir);
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        if (janas_st_open(&shards[n_shards], p, err, sizeof err) != 0)
            die(err, NULL);
        n_shards++;
    }
    closedir(d);
    if (!n_shards)
        die("no safetensors in", dir);
}

static const struct janas_st *find(const char *name, struct janas_st_tensor *t)
{
    for (int i = 0; i < n_shards; i++)
        if (janas_st_find(&shards[i], name, t) == 0)
            return &shards[i];
    return NULL;
}

static struct janas_json_doc *read_json_file(const char *dir, const char *name)
{
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f)
        die("cannot read", path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *t = malloc((size_t)n + 1);
    if (!t || fread(t, 1, (size_t)n, f) != (size_t)n)
        die("cannot read", path);
    fclose(f);
    char err[200];
    struct janas_json_doc *d = janas_json_parse(t, (size_t)n, err, sizeof err);
    free(t);
    if (!d)
        die(err, path);
    return d;
}

/* ---- the output ---- */

static struct janas_jw W;
static uint64_t inexact; /* values half precision does not hold exactly */

static void put(const char *dst, uint32_t type, uint32_t n_dims,
                const uint64_t *dims, void *data, uint64_t bytes)
{
    if (!data)
        die("cannot read or out of memory", dst);
    if (janas_jw_tensor(&W, dst, type, n_dims, dims, data, bytes) != 0)
        die("out of memory, or a name too long", dst);
}

/* a tensor of the checkpoint: a matrix (rows the outermost dimension, the
   rest flattened into the row) in half precision, transposed, anything
   else in float; its elements must be count (0: any) */
static void add(const char *src, const char *dst, int matrix, uint64_t count)
{
    struct janas_st_tensor t;
    const struct janas_st *s = find(src, &t);
    if (!s)
        die("tensor missing or not F32/F16/BF16", src);
    if (count && t.count != count)
        die("tensor of another size", src);
    float *x = janas_st_floats(s, &t);
    if (!x)
        die("cannot read", src);
    if (!matrix) {
        uint64_t dims[1] = {t.count};
        put(dst, GGML_F32, 1, dims, x, 4 * t.count);
        return;
    }
    if (t.dims < 2)
        die("not a matrix", src);
    uint64_t rows = t.shape[0], cols = t.count / rows;
    uint16_t *h = malloc(2 * t.count);
    if (!h)
        die("out of memory", NULL);
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t c = 0; c < cols; c++) {
            float f = x[r * cols + c];
            uint16_t v = janas_fp32_to_fp16(f);
            if (janas_fp16_to_fp32(v) != f)
                inexact++;
            h[c * rows + r] = v;
        }
    free(x);
    /* ggml order: the length of a stored row (the outputs) first */
    uint64_t dims[2] = {rows, cols};
    put(dst, GGML_F16, 2, dims, h, 2 * t.count);
}

static uint32_t cfg_u32(const struct janas_json *o, const char *k)
{
    const struct janas_json *v = janas_json_get(o, k);
    if (!v || v->type != JANAS_JSON_NUMBER)
        die("missing", k);
    return (uint32_t)janas_json_num(v, 0);
}

/* the numbers of a JSON array as int32 (malloc'd), their count in *n */
static int32_t *int_array(const struct janas_json *a, uint64_t *n)
{
    *n = 0;
    if (!a || a->type != JANAS_JSON_ARRAY)
        return NULL;
    int32_t *v = malloc(sizeof(int32_t) * (a->n ? a->n : 1));
    if (!v)
        die("out of memory", NULL);
    for (const struct janas_json *e = a->child; e; e = e->next)
        if (e->type == JANAS_JSON_NUMBER)
            v[(*n)++] = (int32_t)janas_json_num(e, 0);
    return v;
}

/* ---- the vocabulary ---- */

static char **tok;      /* by id */
static uint64_t *tok_n; /* their lengths */
static int32_t *tok_type;
static uint32_t n_vocab;

static void set_token(uint32_t id, const char *s, size_t n, int32_t type)
{
    if (id >= n_vocab)
        die("a token beyond the vocabulary's size", NULL);
    free(tok[id]);
    tok[id] = malloc(n + 1);
    if (!tok[id])
        die("out of memory", NULL);
    memcpy(tok[id], s, n);
    tok[id][n] = 0;
    tok_n[id] = n;
    tok_type[id] = type;
}

static int32_t token_id(const char *s)
{
    for (uint32_t i = 0; i < n_vocab; i++)
        if (tok[i] && tok_n[i] == strlen(s) && !memcmp(tok[i], s, tok_n[i]))
            return (int32_t)i;
    return -1;
}

static void vocabulary(const struct janas_json *tj)
{
    const struct janas_json *m = janas_json_get(tj, "model");
    const struct janas_json *v = janas_json_get(m, "vocab");
    const struct janas_json *mg = janas_json_get(m, "merges");
    if (!janas_json_is(janas_json_get(m, "type"), "BPE") || !v ||
        v->type != JANAS_JSON_OBJECT || !mg || mg->type != JANAS_JSON_ARRAY)
        die("tokenizer.json is not a BPE vocabulary", NULL);
    tok = calloc(n_vocab, sizeof *tok);
    tok_n = calloc(n_vocab, sizeof *tok_n);
    tok_type = calloc(n_vocab, sizeof *tok_type);
    if (!tok || !tok_n || !tok_type)
        die("out of memory", NULL);
    for (const struct janas_json *e = v->child; e; e = e->next)
        set_token((uint32_t)janas_json_num(e, -1), e->key, e->key_n, 1);
    const struct janas_json *add = janas_json_get(tj, "added_tokens");
    for (const struct janas_json *e = add ? add->child : NULL; e; e = e->next) {
        const struct janas_json *c = janas_json_get(e, "content");
        if (!c || c->type != JANAS_JSON_STRING)
            die("an added token without its text", NULL);
        /* the special tokens and the timestamps: never text */
        set_token((uint32_t)janas_json_num(janas_json_get(e, "id"), -1), c->s,
                  c->n, 3);
    }
    for (uint32_t i = 0; i < n_vocab; i++)
        if (!tok[i])
            die("the vocabulary has a hole", NULL);
    janas_jw_str(&W, "tokenizer.ggml.model", "gpt2");
    janas_jw_str(&W, "tokenizer.ggml.pre", "whisper");
    janas_jw_strings(&W, "tokenizer.ggml.tokens", n_vocab,
                     (const char *const *)tok, tok_n);
    janas_jw_ints(&W, "tokenizer.ggml.token_type", n_vocab, tok_type);
    /* merges: "a b", or [a, b] in the newer files */
    uint64_t nm = 0;
    char **ms = malloc(sizeof(char *) * (mg->n ? mg->n : 1));
    if (!ms)
        die("out of memory", NULL);
    for (const struct janas_json *e = mg->child; e; e = e->next) {
        if (e->type == JANAS_JSON_STRING) {
            ms[nm] = strndup(e->s, e->n);
        } else if (e->type == JANAS_JSON_ARRAY && e->n == 2 &&
                   e->child->type == JANAS_JSON_STRING &&
                   e->child->next->type == JANAS_JSON_STRING) {
            const struct janas_json *a = e->child, *b = a->next;
            ms[nm] = malloc(a->n + b->n + 2);
            if (ms[nm]) {
                memcpy(ms[nm], a->s, a->n);
                ms[nm][a->n] = ' ';
                memcpy(ms[nm] + a->n + 1, b->s, b->n);
                ms[nm][a->n + 1 + b->n] = 0;
            }
        } else {
            die("a merge of another form", NULL);
        }
        if (!ms[nm++])
            die("out of memory", NULL);
    }
    janas_jw_strings(&W, "tokenizer.ggml.merges", nm, (const char *const *)ms,
                     NULL);
    for (uint64_t i = 0; i < nm; i++)
        free(ms[i]);
    free(ms);
}

static uint32_t need_token(const char *s)
{
    int32_t id = token_id(s);
    if (id < 0)
        die("the vocabulary lacks", s);
    return (uint32_t)id;
}

/* ---- the conversion ---- */

static void block(const char *src, const char *dst, uint32_t d, int cross)
{
    char s[160], o[64];
    static const char *const P[4] = {"q", "k", "v", "out"};
    static const char *const Q[4] = {"q", "k", "v", "o"};
    const char *att = cross ? "encoder_attn" : "self_attn";
    const char *x = cross ? "x" : "";
    for (int i = 0; i < 4; i++) {
        snprintf(s, sizeof s, "%s.%s.%s_proj.weight", src, att, P[i]);
        snprintf(o, sizeof o, "%s.%s%s", dst, x, Q[i]);
        add(s, o, 1, (uint64_t)d * d);
        if (i == 1)
            continue; /* the keys have no bias */
        snprintf(s, sizeof s, "%s.%s.%s_proj.bias", src, att, P[i]);
        snprintf(o, sizeof o, "%s.%s%s.b", dst, x, Q[i]);
        add(s, o, 0, d);
    }
    snprintf(s, sizeof s, "%s.%s_layer_norm.weight", src, att);
    snprintf(o, sizeof o, "%s.%sln.w", dst, x);
    add(s, o, 0, d);
    snprintf(s, sizeof s, "%s.%s_layer_norm.bias", src, att);
    snprintf(o, sizeof o, "%s.%sln.b", dst, x);
    add(s, o, 0, d);
}

static void mlp(const char *src, const char *dst, uint32_t d, uint32_t ffn)
{
    char s[160], o[64];
    snprintf(s, sizeof s, "%s.fc1.weight", src);
    snprintf(o, sizeof o, "%s.fc1", dst);
    add(s, o, 1, (uint64_t)d * ffn);
    snprintf(s, sizeof s, "%s.fc1.bias", src);
    snprintf(o, sizeof o, "%s.fc1.b", dst);
    add(s, o, 0, ffn);
    snprintf(s, sizeof s, "%s.fc2.weight", src);
    snprintf(o, sizeof o, "%s.fc2", dst);
    add(s, o, 1, (uint64_t)d * ffn);
    snprintf(s, sizeof s, "%s.fc2.bias", src);
    snprintf(o, sizeof o, "%s.fc2.b", dst);
    add(s, o, 0, d);
    snprintf(s, sizeof s, "%s.final_layer_norm.weight", src);
    snprintf(o, sizeof o, "%s.mln.w", dst);
    add(s, o, 0, d);
    snprintf(s, sizeof s, "%s.final_layer_norm.bias", src);
    snprintf(o, sizeof o, "%s.mln.b", dst);
    add(s, o, 0, d);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: hf2jns_whisper <checkpoint directory> "
                        "<out.jns>\n");
        return 2;
    }
    const char *dir = argv[1], *out_path = argv[2];
    struct janas_json_doc *cd = read_json_file(dir, "config.json");
    struct janas_json_doc *gd = read_json_file(dir, "generation_config.json");
    struct janas_json_doc *pd = read_json_file(dir, "preprocessor_config.json");
    struct janas_json_doc *td = read_json_file(dir, "tokenizer.json");
    const struct janas_json *cfg = janas_json_root(cd);
    const struct janas_json *gen = janas_json_root(gd);
    const struct janas_json *pre = janas_json_root(pd);
    if (!janas_json_is(janas_json_get(cfg, "model_type"), "whisper"))
        die("not a Whisper model (config.json's model_type)", NULL);
    uint32_t d = cfg_u32(cfg, "d_model");
    uint32_t el = cfg_u32(cfg, "encoder_layers");
    uint32_t eh = cfg_u32(cfg, "encoder_attention_heads");
    uint32_t ef = cfg_u32(cfg, "encoder_ffn_dim");
    uint32_t dl = cfg_u32(cfg, "decoder_layers");
    uint32_t dh = cfg_u32(cfg, "decoder_attention_heads");
    uint32_t df = cfg_u32(cfg, "decoder_ffn_dim");
    uint32_t mels = cfg_u32(cfg, "num_mel_bins");
    uint32_t actx = cfg_u32(cfg, "max_source_positions");
    uint32_t tctx = cfg_u32(cfg, "max_target_positions");
    n_vocab = cfg_u32(cfg, "vocab_size");
    uint32_t n_fft = cfg_u32(pre, "n_fft"), hop = cfg_u32(pre, "hop_length");
    uint32_t rate = cfg_u32(pre, "sampling_rate");
    uint32_t chunk = cfg_u32(pre, "chunk_length");
    if (!d || !eh || !dh || d % eh || d % dh || !el || !dl || !mels ||
        cfg_u32(pre, "feature_size") != mels || n_fft < 2 || !hop || !rate ||
        !chunk || actx * 2 * hop != chunk * rate)
        die("sizes out of range", NULL);
    open_shards(dir);

    janas_jw_u32(&W, "whisper.dim", d);
    janas_jw_u32(&W, "whisper.audio.layers", el);
    janas_jw_u32(&W, "whisper.audio.heads", eh);
    janas_jw_u32(&W, "whisper.audio.ffn", ef);
    janas_jw_u32(&W, "whisper.audio.ctx", actx);
    janas_jw_u32(&W, "whisper.text.layers", dl);
    janas_jw_u32(&W, "whisper.text.heads", dh);
    janas_jw_u32(&W, "whisper.text.ffn", df);
    janas_jw_u32(&W, "whisper.text.ctx", tctx);
    janas_jw_u32(&W, "whisper.vocab", n_vocab);
    janas_jw_u32(&W, "whisper.n_mels", mels);
    janas_jw_u32(&W, "whisper.n_fft", n_fft);
    janas_jw_u32(&W, "whisper.hop", hop);
    janas_jw_u32(&W, "whisper.sample_rate", rate);
    janas_jw_u32(&W, "whisper.chunk_seconds", chunk);

    /* the mel filters, n_mels rows of n_fft / 2 + 1 */
    const struct janas_json *mf = janas_json_get(pre, "mel_filters");
    uint32_t bins = n_fft / 2 + 1;
    if (mf && (mf->type != JANAS_JSON_ARRAY || mf->n != mels))
        die("preprocessor_config.json has mel filters of another size", NULL);
    float *filt = malloc(sizeof(float) * mels * bins);
    if (!filt)
        die("out of memory", NULL);
    if (!mf)
        mel_filters(filt, mels, n_fft, rate);
    uint32_t m = 0;
    for (const struct janas_json *r = mf ? mf->child : NULL; r;
         r = r->next, m++) {
        if (r->type != JANAS_JSON_ARRAY || r->n != bins)
            die("a mel filter of another length", NULL);
        uint32_t k = 0;
        for (const struct janas_json *e = r->child; e; e = e->next)
            filt[m * bins + k++] = (float)janas_json_num(e, 0);
    }
    uint64_t fd[2] = {bins, mels};
    put("mel_filters", GGML_F32, 2, fd, filt, 4ull * mels * bins);

    vocabulary(janas_json_root(td));
    struct {
        const char *key, *tok, *alt;
    } specials[] = {
        {"whisper.token.eot", "<|endoftext|>", NULL},
        {"whisper.token.sot", "<|startoftranscript|>", NULL},
        {"whisper.token.translate", "<|translate|>", NULL},
        {"whisper.token.transcribe", "<|transcribe|>", NULL},
        {"whisper.token.sot_lm", "<|startoflm|>", NULL},
        {"whisper.token.sot_prev", "<|startofprev|>", NULL},
        {"whisper.token.no_speech", "<|nospeech|>", "<|nocaptions|>"},
        {"whisper.token.no_timestamps", "<|notimestamps|>", NULL},
        {"whisper.token.timestamp_begin", "<|0.00|>", NULL},
    };
    for (size_t i = 0; i < sizeof specials / sizeof *specials; i++) {
        int32_t id = token_id(specials[i].tok);
        if (id < 0 && specials[i].alt)
            id = token_id(specials[i].alt);
        if (id < 0)
            die("the vocabulary lacks", specials[i].tok);
        janas_jw_u32(&W, specials[i].key, (uint32_t)id);
    }
    /* the languages, in the order of their tokens: the first follows the
       start of transcript */
    uint32_t sot = need_token("<|startoftranscript|>");
    const char *langs[200];
    uint32_t nl = 0;
    for (uint32_t id = sot + 1; id < n_vocab && nl < 200; id++) {
        const char *t = tok[id];
        size_t n = tok_n[id];
        if (n < 5 || n > 12 || strncmp(t, "<|", 2) || strcmp(t + n - 2, "|>"))
            break;
        if (!strcmp(t, "<|translate|>"))
            break;
        static char codes[200][12];
        memcpy(codes[nl], t + 2, n - 4);
        codes[nl][n - 4] = 0;
        langs[nl] = codes[nl];
        nl++;
    }
    if (!nl)
        die("no languages after <|startoftranscript|>", NULL);
    janas_jw_strings(&W, "whisper.languages", nl, langs, NULL);
    janas_jw_u32(&W, "whisper.multilingual",
                 janas_json_get(gen, "is_multilingual") &&
                     janas_json_get(gen, "is_multilingual")->type ==
                         JANAS_JSON_TRUE);
    uint64_t n;
    int32_t *v = int_array(janas_json_get(gen, "suppress_tokens"), &n);
    if (!v)
        v = int_array(janas_json_get(cfg, "suppress_tokens"), &n);
    if (!v)
        die("no suppress_tokens", NULL);
    janas_jw_ints(&W, "whisper.suppress", n, v);
    free(v);
    v = int_array(janas_json_get(gen, "begin_suppress_tokens"), &n);
    if (v) {
        janas_jw_ints(&W, "whisper.begin_suppress", n, v);
        free(v);
    }
    const struct janas_json *mi =
        janas_json_get(gen, "max_initial_timestamp_index");
    janas_jw_u32(&W, "whisper.max_initial_timestamp",
                 mi ? (uint32_t)janas_json_num(mi, 50) : 50);
    /* the heads that align words with time: pairs (layer, head) */
    const struct janas_json *ah = janas_json_get(gen, "alignment_heads");
    if (ah && ah->type == JANAS_JSON_ARRAY) {
        int32_t *p = malloc(sizeof(int32_t) * 2 * (ah->n ? ah->n : 1));
        uint64_t k = 0;
        if (!p)
            die("out of memory", NULL);
        for (const struct janas_json *e = ah->child; e; e = e->next)
            if (e->type == JANAS_JSON_ARRAY && e->n == 2) {
                p[k++] = (int32_t)janas_json_num(e->child, 0);
                p[k++] = (int32_t)janas_json_num(e->child->next, 0);
            }
        janas_jw_ints(&W, "whisper.alignment_heads", k, p);
        free(p);
    }

    /* the encoder */
    add("model.encoder.conv1.weight", "e.conv1", 1, (uint64_t)d * mels * 3);
    add("model.encoder.conv1.bias", "e.conv1.b", 0, d);
    add("model.encoder.conv2.weight", "e.conv2", 1, (uint64_t)d * d * 3);
    add("model.encoder.conv2.bias", "e.conv2.b", 0, d);
    add("model.encoder.embed_positions.weight", "e.pos", 0, (uint64_t)actx * d);
    for (uint32_t l = 0; l < el; l++) {
        char s[96], o[32];
        snprintf(s, sizeof s, "model.encoder.layers.%u", l);
        snprintf(o, sizeof o, "e.%u", l);
        block(s, o, d, 0);
        mlp(s, o, d, ef);
    }
    add("model.encoder.layer_norm.weight", "e.ln.w", 0, d);
    add("model.encoder.layer_norm.bias", "e.ln.b", 0, d);
    /* the decoder */
    add("model.decoder.embed_tokens.weight", "d.tok", 1, (uint64_t)n_vocab * d);
    add("model.decoder.embed_positions.weight", "d.pos", 0, (uint64_t)tctx * d);
    for (uint32_t l = 0; l < dl; l++) {
        char s[96], o[32];
        snprintf(s, sizeof s, "model.decoder.layers.%u", l);
        snprintf(o, sizeof o, "d.%u", l);
        block(s, o, d, 0);
        block(s, o, d, 1);
        mlp(s, o, d, df);
    }
    add("model.decoder.layer_norm.weight", "d.ln.w", 0, d);
    add("model.decoder.layer_norm.bias", "d.ln.b", 0, d);

    uint64_t file_bytes = 0;
    char err[300];
    if (janas_jw_write(&W, out_path, "whisper", d, &file_bytes, err,
                       sizeof err) != 0)
        die(err, NULL);
    printf("written %s: %d tensors, %.1f MB (encoder %u x %u, decoder %u x "
           "%u, %u mels, %u languages, %u tokens)\n",
           out_path, W.n, file_bytes / 1e6, el, d, dl, d, mels, nl, n_vocab);
    if (inexact)
        printf("%llu weights rounded to half precision\n",
               (unsigned long long)inexact);
    janas_jw_free(&W);
    for (uint32_t i = 0; i < n_vocab; i++)
        free(tok[i]);
    free(tok);
    free(tok_n);
    free(tok_type);
    for (int i = 0; i < n_shards; i++)
        janas_st_close(&shards[i]);
    janas_json_free(cd);
    janas_json_free(gd);
    janas_json_free(pd);
    janas_json_free(td);
    return 0;
}
