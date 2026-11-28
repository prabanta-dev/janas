/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_template.c - a conversation through the model's chat template, as
 * Hugging Face's transformers hands it over: Janas's Jinja interpreter
 * (jinja.h) runs it. The messages (their pictures, their reasoning, their
 * tool calls) and the tools are the template's data, marked as the
 * conversation's; add_generation_prompt, bos_token, eos_token, the switch
 * of the reasoning in the form the template takes it, and any keyword the
 * caller adds are its variables. Its text then becomes tokens as the
 * tokenizer of the model would make them: the special tokens read only
 * where the template itself wrote them, the rest as plain text between
 * them; each picture's placeholder made its tokens, as the model's
 * processor does (Qwen's <|image_pad|> repeated, Gemma's <|image|> between
 * its <start_of_image> and <end_of_image>).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"
#include "llm/jinja.h"
#include "llm/jns.h"
#include "llm/vision.h"

/* ---- the special tokens, by their first byte */
struct janas_tspecials {
    uint32_t n;
    int32_t *id;
    char (*text)[64];
    uint8_t *len;
    uint32_t *order;     /* indices, by first byte, longest first */
    uint32_t start[257]; /* order[start[b] .. start[b + 1]) begin with b */
};

static int by_byte_len(const void *a, const void *b, void *ctx)
{
    const struct janas_tspecials *s = ctx;
    uint32_t i = *(const uint32_t *)a, j = *(const uint32_t *)b;
    unsigned char x = (unsigned char)s->text[i][0],
                  y = (unsigned char)s->text[j][0];
    if (x != y)
        return x < y ? -1 : 1;
    if (s->len[i] != s->len[j])
        return s->len[i] > s->len[j] ? -1 : 1;
    return i < j ? -1 : i > j;
}

/* qsort_r is not everywhere: a plain insertion into place, the specials
   being a few thousand at most */
static void sort_specials(struct janas_tspecials *s)
{
    for (uint32_t k = 1; k < s->n; k++) {
        uint32_t v = s->order[k], q = k;
        while (q > 0 && by_byte_len(&s->order[q - 1], &v, s) > 0) {
            s->order[q] = s->order[q - 1];
            q--;
        }
        s->order[q] = v;
    }
}

static void specials_free(struct janas_tspecials *s)
{
    if (!s)
        return;
    free(s->id);
    free(s->text);
    free(s->len);
    free(s->order);
    free(s);
}

static struct janas_tspecials *specials_load(const janas_llm *llm)
{
    const int32_t *ids;
    struct janas_tspecials *s = calloc(1, sizeof *s);
    if (!s)
        return NULL;
    uint32_t n = janas_tokenizer_specials(llm->tok, &ids);
    s->id = malloc((size_t)n * sizeof *s->id + 1);
    s->text = malloc((size_t)n * sizeof *s->text + 1);
    s->len = malloc((size_t)n + 1);
    s->order = malloc((size_t)n * sizeof *s->order + 1);
    if (!s->id || !s->text || !s->len || !s->order) {
        specials_free(s);
        return NULL;
    }
    for (uint32_t i = 0; i < n; i++) {
        size_t l = janas_tokenizer_decode(llm->tok, ids[i], s->text[s->n],
                                          sizeof s->text[0]);
        if (!l || l >= sizeof s->text[0])
            continue; /* none longer is written by a template */
        s->id[s->n] = ids[i];
        s->len[s->n] = (uint8_t)l;
        s->order[s->n] = s->n;
        s->n++;
    }
    sort_specials(s);
    uint32_t k = 0;
    for (int b = 0; b < 256; b++) {
        s->start[b] = k;
        while (k < s->n && (unsigned char)s->text[s->order[k]][0] == b)
            k++;
    }
    s->start[256] = s->n;
    return s;
}

/* ---- the templates: the model's, its user's in its place, a chat's */
static void tmpl_free(struct janas_tmpl *t)
{
    janas_jinja_free(t->t);
    free(t->src);
    *t = (struct janas_tmpl){0};
}

/* src parsed into *t: 0, or -1 with why */
static int tmpl_make(struct janas_tmpl *t, const char *src, size_t n, char *why,
                     size_t cap)
{
    char err[200];
    struct janas_jinja *j = janas_jinja_parse(src, n, err, sizeof err);
    char *copy = j ? malloc(n + 1) : NULL;
    if (!j || !copy) {
        janas_jinja_free(j);
        snprintf(why, cap, "%s", j ? "out of memory" : err);
        return -1;
    }
    memcpy(copy, src, n);
    copy[n] = 0;
    tmpl_free(t);
    t->t = j;
    t->src = copy;
    /* the switch of its reasoning, as the template names it */
    t->think_var = strstr(copy, "enable_thinking")    ? 1
                   : strstr(copy, "reasoning_effort") ? 2
                                                      : 0;
    return 0;
}

void janas_api_template_open(janas_llm *llm, const char *src)
{
    llm->tfile = llm->tover = (struct janas_tmpl){0};
    llm->tmpl_err[0] = 0;
    /* the special tokens, for any template: the model's or another */
    llm->tsp = specials_load(llm);
    if (!src || !*src) {
        snprintf(llm->tmpl_err, sizeof llm->tmpl_err,
                 "the model's file has no chat template");
        return;
    }
    char err[200];
    if (!llm->tsp)
        snprintf(llm->tmpl_err, sizeof llm->tmpl_err, "out of memory");
    else if (tmpl_make(&llm->tfile, src, strlen(src), err, sizeof err) != 0)
        snprintf(llm->tmpl_err, sizeof llm->tmpl_err,
                 "the model's chat template could not be read: %s", err);
}

void janas_api_template_close(janas_llm *llm)
{
    tmpl_free(&llm->tfile);
    tmpl_free(&llm->tover);
    specials_free(llm->tsp);
    llm->tsp = NULL;
}

void janas_api_template_chat_free(janas_llm_chat *c)
{
    tmpl_free(&c->town);
    free(c->kwargs);
    c->kwargs = NULL;
}

/* the template a chat uses: its own, its model's user's, the model's */
static const struct janas_tmpl *tmpl_of(const janas_llm_chat *c)
{
    return c->town.t         ? &c->town
           : c->llm->tover.t ? &c->llm->tover
           : c->llm->tfile.t ? &c->llm->tfile
                             : NULL;
}

int janas_api_template_on(const janas_llm_chat *c)
{
    return c->llm->tsp && tmpl_of(c) != NULL;
}

int32_t janas_llm_set_template(janas_llm *llm, const char *src, int32_t len)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    if (!src) {
        tmpl_free(&llm->tover); /* the model's own again */
        return JANAS_LLM_OK;
    }
    char why[200];
    if (!llm->tsp)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    if (tmpl_make(&llm->tover, src, janas_api_text_len(src, len), why,
                  sizeof why) != 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "the chat template: %s", why);
    return JANAS_LLM_OK;
}

int32_t janas_llm_template(const janas_llm *llm, int32_t which, char *buf,
                           int32_t cap, int32_t *len)
{
    if (len)
        *len = 0;
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    const struct janas_tmpl *t = which == JANAS_LLM_TEMPLATE_FILE ? &llm->tfile
                                 : llm->tover.t                   ? &llm->tover
                                                                  : &llm->tfile;
    if (!t->src)
        return janas_api_fail(JANAS_LLM_EMODEL, "%s",
                              llm->tmpl_err[0] ? llm->tmpl_err
                                               : "no chat template");
    return janas_api_copy_out(t->src, strlen(t->src), buf, cap, len);
}

int32_t janas_llm_chat_template(janas_llm_chat *c, const char *src, int32_t len)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    if (!src) {
        tmpl_free(&c->town);
        return JANAS_LLM_OK;
    }
    char why[200];
    if (!c->llm->tsp)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    if (tmpl_make(&c->town, src, janas_api_text_len(src, len), why,
                  sizeof why) != 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "the chat template: %s", why);
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_template_kwargs(janas_llm_chat *c, const char *json,
                                       int32_t len)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    free(c->kwargs);
    c->kwargs = NULL;
    if (!json)
        return JANAS_LLM_OK;
    size_t n = janas_api_text_len(json, len);
    char err[128];
    struct janas_json_doc *d = janas_json_parse(json, n, err, sizeof err);
    int ok = d && janas_json_root(d)->type == JANAS_JSON_OBJECT;
    janas_json_free(d);
    if (!ok)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "the template's keywords: not a JSON object");
    if (!(c->kwargs = malloc(n + 1)))
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    memcpy(c->kwargs, json, n);
    c->kwargs[n] = 0;
    return JANAS_LLM_OK;
}

/* ---- the template's data and variables */
static void put_str(struct janas_buf *b, const char *s, size_t n)
{
    janas_json_write_str(b, s ? s : "", s ? n : 0);
}

static const char *role_name(int32_t r)
{
    return r == JANAS_LLM_ROLE_SYSTEM      ? "system"
           : r == JANAS_LLM_ROLE_ASSISTANT ? "assistant"
           : r == JANAS_LLM_ROLE_TOOL      ? "tool"
                                           : "user";
}

/* a token's text, as the template writes it */
static void token_str(const janas_llm *llm, int32_t id, struct janas_buf *b)
{
    char t[256];
    size_t n = id >= 0 ? janas_tokenizer_decode(llm->tok, id, t, sizeof t) : 0;
    if (n >= sizeof t)
        n = 0;
    put_str(b, t, n);
}

/*
 * The reasoning of one of the last replies, as the model wrote it (the
 * memo keeps its tokens): the text between its markers, without the name
 * of Gemma's channel, its blank lines around it left out.
 */
static void memo_reasoning(const janas_llm_chat *c, const struct cmsg *m,
                           struct janas_buf *out)
{
    const janas_llm *llm = c->llm;
    const struct memo *mm = janas_api_memo_find(c, m);
    if (!mm || llm->think_open < 0 || llm->think_close < 0)
        return;
    uint32_t a = 0;
    while (a < mm->n && mm->ids[a] != llm->think_open)
        a++;
    uint32_t b = a + 1;
    while (b < mm->n && mm->ids[b] != llm->think_close)
        b++;
    if (a >= mm->n || b >= mm->n)
        return;
    for (uint32_t i = a + 1; i < b; i++) {
        char t[256];
        size_t n = janas_tokenizer_decode(llm->tok, mm->ids[i], t, sizeof t);
        if (n < sizeof t)
            janas_buf_put(out, t, n);
    }
    if (llm->format == FORMAT_GEMMA && out->n >= 7 &&
        !memcmp(out->p, "thought", 7)) {
        memmove(out->p, out->p + 7, out->n - 7);
        out->n -= 7;
    }
    size_t s = 0, e = out->n;
    while (s < e && (unsigned char)out->p[s] <= ' ')
        s++;
    while (e > s && (unsigned char)out->p[e - 1] <= ' ')
        e--;
    memmove(out->p, out->p + s, e - s);
    out->n = e - s;
}

/* the items of a JSON array, between its brackets */
static void items_of(const char *json, const char **p, size_t *n)
{
    const char *a = json ? strchr(json, '[') : NULL;
    const char *b = json ? strrchr(json, ']') : NULL;
    *p = "";
    *n = 0;
    if (!a || !b || b <= a)
        return;
    a++;
    while (a < b && (unsigned char)*a <= ' ')
        a++;
    while (b > a && (unsigned char)b[-1] <= ' ')
        b--;
    *p = a;
    *n = (size_t)(b - a);
}

static int has_tools(const janas_llm_chat *c)
{
    const char *p;
    size_t n, m;
    items_of(c->tools_json, &p, &n);
    items_of(c->more_json, &p, &m);
    return n || m;
}

static void put_message(const janas_llm_chat *c, const struct cmsg *m,
                        struct janas_buf *d)
{
    janas_buf_puts(d, "{\"role\": ");
    put_str(d, role_name(m->role), strlen(role_name(m->role)));
    janas_buf_puts(d, ", \"content\": ");
    if (m->n_pics) {
        /* its pictures, then its text, as a list of parts */
        janas_buf_puts(d, "[");
        for (size_t k = 0; k < m->n_pics; k++)
            janas_buf_puts(d, "{\"type\": \"image\"}, ");
        janas_buf_puts(d, "{\"type\": \"text\", \"text\": ");
        put_str(d, m->text, m->n);
        janas_buf_puts(d, "}]");
    } else {
        put_str(d, m->text, m->n);
    }
    if (m->role == JANAS_LLM_ROLE_ASSISTANT) {
        struct janas_buf r = {0};
        if (m->reason)
            janas_buf_put(&r, m->reason, m->n_reason);
        else
            memo_reasoning(c, m, &r);
        if (r.n) {
            janas_buf_puts(d, ", \"reasoning_content\": ");
            put_str(d, r.p, r.n);
        }
        janas_buf_free(&r);
    }
    if (m->n_calls) {
        janas_buf_puts(d, ", \"tool_calls\": [");
        for (size_t k = 0; k < m->n_calls; k++) {
            const struct janas_tool_call *t = &m->calls[k];
            janas_buf_puts(d, k ? ", " : "");
            janas_buf_puts(d, "{\"type\": \"function\", \"function\": "
                              "{\"name\": ");
            put_str(d, t->name, t->name_n);
            /* the arguments as an object, as the templates expect */
            janas_buf_puts(d, ", \"arguments\": ");
            if (t->args_n)
                janas_buf_put(d, t->args, t->args_n);
            else
                janas_buf_puts(d, "{}");
            janas_buf_puts(d, "}}");
        }
        janas_buf_puts(d, "]");
    }
    janas_buf_puts(d, "}");
}

int janas_api_template_vars(const janas_llm_chat *c, size_t upto, int gen,
                            int tools, const char *kwargs,
                            struct janas_buf *vars, struct janas_buf *data)
{
    const janas_llm *llm = c->llm;
    janas_buf_puts(data, "{\"messages\": [");
    for (size_t i = 0; i < upto && i < c->n_msgs; i++) {
        janas_buf_puts(data, i ? ", " : "");
        put_message(c, &c->msgs[i], data);
    }
    janas_buf_puts(data, "]");
    if (tools && has_tools(c)) {
        /* those given and those added since, one list: the model sees
           every tool it may call, as the template writes them */
        janas_buf_puts(data, ", \"tools\": [");
        const char *p;
        size_t n;
        items_of(c->tools_json, &p, &n);
        janas_buf_put(data, p, n);
        int first = n == 0;
        items_of(c->more_json, &p, &n);
        if (n && !first)
            janas_buf_puts(data, ", ");
        janas_buf_put(data, p, n);
        janas_buf_puts(data, "]");
    }
    janas_buf_puts(data, "}");
    janas_buf_printf(vars, "{\"add_generation_prompt\": %s",
                     gen ? "true" : "false");
    /* reasoning or not, when the caller says: otherwise the template's own
       default */
    const struct janas_tmpl *t = tmpl_of(c);
    int tv = t ? t->think_var : 0;
    if (c->p.thinking >= 0 && tv == 1)
        janas_buf_printf(vars, ", \"enable_thinking\": %s",
                         c->p.thinking ? "true" : "false");
    else if (c->p.thinking == 0 && tv == 2)
        janas_buf_puts(vars, ", \"reasoning_effort\": \"none\"");
    janas_buf_puts(vars, ", \"bos_token\": ");
    token_str(llm, llm->bos_tok, vars);
    janas_buf_puts(vars, ", \"eos_token\": ");
    token_str(llm, llm->eos_id, vars);
    /* the caller's keywords, over them (a JSON object's members) */
    if (kwargs && *kwargs) {
        char err[128];
        struct janas_json_doc *d =
            janas_json_parse(kwargs, strlen(kwargs), err, sizeof err);
        const struct janas_json *o = d ? janas_json_root(d) : NULL;
        if (!o || o->type != JANAS_JSON_OBJECT) {
            janas_json_free(d);
            return janas_api_fail(JANAS_LLM_EINVAL,
                                  "the template's keywords: not a JSON object");
        }
        for (const struct janas_json *e = o->child; e; e = e->next) {
            janas_buf_puts(vars, ", ");
            put_str(vars, e->key, e->key_n);
            janas_buf_puts(vars, ": ");
            janas_json_write(vars, e);
        }
        janas_json_free(d);
    }
    janas_buf_puts(vars, "}");
    if (vars->oom || data->oom)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    return JANAS_LLM_OK;
}

int janas_api_template_text(const janas_llm_chat *c, size_t upto, int gen,
                            int tools, struct janas_buf *out,
                            struct janas_buf *marks)
{
    const struct janas_tmpl *tm = tmpl_of(c);
    if (!tm)
        return janas_api_fail(JANAS_LLM_EMODEL, "%s", c->llm->tmpl_err);
    char err[512];
    struct janas_buf vars = {0}, data = {0};
    int32_t rc =
        janas_api_template_vars(c, upto, gen, tools, c->kwargs, &vars, &data);
    struct janas_json_doc *dv = NULL, *dd = NULL;
    if (rc == JANAS_LLM_OK) {
        dv = janas_json_parse(vars.p, vars.n, err, sizeof err);
        dd = dv ? janas_json_parse(data.p, data.n, err, sizeof err) : NULL;
        if (!dv || !dd)
            rc = janas_api_fail(JANAS_LLM_EINVAL, "the conversation: %s", err);
    }
    if (rc == JANAS_LLM_OK &&
        janas_jinja_render(tm->t, janas_json_root(dv), janas_json_root(dd), out,
                           marks, err, sizeof err) != 0)
        rc = janas_api_fail(JANAS_LLM_EINVAL, "the chat template: %s", err);
    janas_json_free(dv);
    janas_json_free(dd);
    janas_buf_free(&vars);
    janas_buf_free(&data);
    return rc;
}

/* ---- the text into tokens */
struct tids {
    int32_t *v;
    size_t n, cap;
    struct ppic *pp; /* where each picture's tokens begin */
    size_t npp;
};

static int tids_put(struct tids *o, const int32_t *v, size_t n)
{
    if (o->n + n > o->cap) {
        size_t cap = o->cap ? o->cap : 256;
        while (cap < o->n + n)
            cap *= 2;
        int32_t *t = realloc(o->v, cap * sizeof *t);
        if (!t)
            return -1;
        o->v = t, o->cap = cap;
    }
    memcpy(o->v + o->n, v, n * sizeof *v);
    o->n += n;
    return 0;
}

static int tids_run(const janas_llm *llm, struct tids *o, const char *s,
                    size_t n)
{
    if (!n)
        return 0;
    long k = janas_tokenizer_encode(llm->tok, s, n, 0, NULL, 0);
    if (k < 0)
        return -1;
    if (o->n + (size_t)k > o->cap) {
        size_t cap = o->cap ? o->cap : 256;
        while (cap < o->n + (size_t)k)
            cap *= 2;
        int32_t *t = realloc(o->v, cap * sizeof *t);
        if (!t)
            return -1;
        o->v = t, o->cap = cap;
    }
    janas_tokenizer_encode(llm->tok, s, n, 0, o->v + o->n, (size_t)k);
    o->n += (size_t)k;
    return 0;
}

/* a picture where the template put its placeholder */
static int tids_picture(const janas_llm *llm, struct tids *o,
                        const struct cpic *pic)
{
    int32_t img, vs, ve;
    janas_vision_tokens(llm->vis, &img, &vs, &ve);
    int framed = o->n && o->v[o->n - 1] == vs;
    if (!framed && tids_put(o, &vs, 1))
        return -1;
    struct ppic *t = realloc(o->pp, (o->npp + 1) * sizeof *t);
    if (!t)
        return -1;
    o->pp = t;
    o->pp[o->npp++] = (struct ppic){.off = (uint32_t)o->n, .pic = pic};
    for (uint32_t k = 0; k < pic->nx * pic->ny; k++)
        if (tids_put(o, &img, 1))
            return -1;
    return framed ? 0 : tids_put(o, &ve, 1);
}

/* The template's text in tokens; the pictures of msgs[0..upto) in the
   order their placeholders come. */
static int32_t text_tokens(const janas_llm_chat *c, size_t upto,
                           const char *text, size_t n, const uint8_t *marks,
                           struct tids *o)
{
    const janas_llm *llm = c->llm;
    const struct janas_tspecials *sp = llm->tsp;
    int32_t img = -1, vs, ve;
    if (llm->vis)
        janas_vision_tokens(llm->vis, &img, &vs, &ve);
    size_t mi = 0, pi = 0; /* the next picture: message, index */
    int err = 0;
    size_t run = 0;
    for (size_t i = 0; i < n && !err;) {
        int32_t hit = -1;
        size_t hl = 0;
        unsigned char b0 = (unsigned char)text[i];
        for (uint32_t q = marks[i] ? sp->start[b0 + 1] : sp->start[b0];
             q < sp->start[b0 + 1]; q++) {
            uint32_t k = sp->order[q];
            size_t l = sp->len[k];
            if (i + l > n || memcmp(text + i, sp->text[k], l))
                continue;
            /* every byte of it the template's own */
            size_t z = 0;
            while (z < l && !marks[i + z])
                z++;
            if (z == l) {
                hit = sp->id[k], hl = l;
                break; /* the longest, being first */
            }
        }
        if (hit < 0) {
            i++;
            continue;
        }
        err |= tids_run(llm, o, text + run, i - run);
        if (hit == img && img >= 0) {
            while (mi < upto && pi >= c->msgs[mi].n_pics)
                mi++, pi = 0;
            if (mi >= upto)
                return janas_api_fail(JANAS_LLM_EINVAL,
                                      "the chat template places more "
                                      "pictures than the conversation has");
            err |= tids_picture(llm, o, &c->msgs[mi].pics[pi++]);
        } else {
            err |= tids_put(o, &hit, 1);
        }
        i += hl;
        run = i;
    }
    err |= tids_run(llm, o, text + run, n - run);
    if (err)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    while (mi < upto && pi >= c->msgs[mi].n_pics)
        mi++, pi = 0;
    if (mi < upto)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "the chat template leaves out a picture of "
                              "the conversation");
    return JANAS_LLM_OK;
}

static int32_t render_ids(const janas_llm_chat *c, size_t upto, int gen,
                          int tools, struct tids *o)
{
    struct janas_buf out = {0}, marks = {0};
    int32_t rc = janas_api_template_text(c, upto, gen, tools, &out, &marks);
    if (rc == JANAS_LLM_OK && out.n)
        rc = text_tokens(c, upto, out.p, out.n, (const uint8_t *)marks.p, o);
    janas_buf_free(&out);
    janas_buf_free(&marks);
    return rc;
}

int janas_api_template_tokens(janas_llm_chat *c, const char *text, size_t n,
                              const char *marks)
{
    struct tids o = {0};
    int32_t rc = text_tokens(c, c->n_msgs, text, n, (const uint8_t *)marks, &o);
    if (rc == JANAS_LLM_OK)
        for (size_t i = 0; i < o.n && rc == JANAS_LLM_OK; i++)
            if (janas_api_add_special(c, o.v[i]) != 0)
                rc = janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    free(o.v);
    free(o.pp);
    return rc;
}

/* How far the conversation's first `upto` messages, as the template writes
   them alone, go along with the whole (`full`); *len their own length. */
static uint32_t along(const janas_llm_chat *c, size_t upto, int tools,
                      const struct tids *full, uint32_t *len)
{
    struct tids o = {0};
    uint32_t k = 0;
    if (len)
        *len = 0;
    if (render_ids(c, upto, 0, tools, &o) == JANAS_LLM_OK) {
        while (k < o.n && k < full->n && o.v[k] == full->v[k])
            k++;
        if (len)
            *len = (uint32_t)o.n;
    }
    free(o.v);
    free(o.pp);
    return k;
}

/*
 * The conversation through the template into c->ids, the reply's turn
 * begun (*reply_off: where), its pictures where their tokens are; what the
 * conversation's accounts need: the system message's tokens, the tools'
 * among them, where the last question begins, whether the reply starts
 * inside the reasoning (the template opened it).
 */
int32_t janas_api_template_build(janas_llm_chat *c, uint32_t *reply_off)
{
    const janas_llm *llm = c->llm;
    struct tids full = {0};
    int32_t rc = render_ids(c, c->n_msgs, 1, 1, &full);
    if (rc != JANAS_LLM_OK) {
        free(full.v);
        free(full.pp);
        return rc;
    }
    int tools = has_tools(c);
    *reply_off = along(c, c->n_msgs, 1, &full, NULL);
    size_t nsys = c->n_msgs && c->msgs[0].role == JANAS_LLM_ROLE_SYSTEM;
    uint32_t sys = 0, sys_len = 0, plain_len = 0;
    if (nsys || tools) {
        sys = along(c, nsys, 1, &full, &sys_len);
        if (tools)
            along(c, nsys, 0, &full, &plain_len);
    }
    size_t q = c->n_msgs;
    while (q > 0 && (c->msgs[q - 1].role == JANAS_LLM_ROLE_USER ||
                     c->msgs[q - 1].role == JANAS_LLM_ROLE_TOOL))
        q--;
    uint32_t last = q ? along(c, q, 1, &full, NULL) : sys;
    /* into the chat */
    c->n_ids = c->n_text = 0;
    if (full.n > c->cap_ids) {
        int32_t *t = realloc(c->ids, full.n * sizeof *t);
        if (!t)
            rc = JANAS_LLM_ENOMEM;
        else
            c->ids = t, c->cap_ids = full.n;
    }
    if (rc == JANAS_LLM_OK && full.npp) {
        struct ppic *t = realloc(c->idpics, full.npp * sizeof *t);
        if (!t)
            rc = JANAS_LLM_ENOMEM;
        else
            c->idpics = t;
    }
    if (rc != JANAS_LLM_OK) {
        free(full.v);
        free(full.pp);
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    }
    if (full.n)
        memcpy(c->ids, full.v, full.n * sizeof *full.v);
    c->n_ids = full.n;
    if (full.npp)
        memcpy(c->idpics, full.pp, full.npp * sizeof *full.pp);
    c->n_idpics = full.npp;
    free(full.v);
    free(full.pp);
    c->sys_tokens = sys;
    c->tools_tokens = sys_len > plain_len ? sys_len - plain_len : 0;
    c->last_at = last;
    /* the reply inside its reasoning when the template opened it */
    c->in_think = c->in_turn = c->chan_label = 0;
    for (size_t i = *reply_off; i < c->n_ids; i++)
        if (llm->think_open >= 0 && c->ids[i] == llm->think_open)
            c->in_think = 1;
        else if (llm->think_close >= 0 && c->ids[i] == llm->think_close)
            c->in_think = 0;
    return JANAS_LLM_OK;
}

/* The system message alone (and the tools), as the template writes it
   before any question, into c->ids: what can be read ahead. */
int32_t janas_api_template_system(janas_llm_chat *c)
{
    struct cmsg one = {.role = JANAS_LLM_ROLE_SYSTEM,
                       .text = c->system,
                       .n = c->system ? strlen(c->system) : 0};
    struct cmsg *msgs = c->msgs;
    size_t n_msgs = c->n_msgs;
    c->msgs = &one;
    c->n_msgs = c->system ? 1 : 0;
    struct tids o = {0};
    int32_t rc = render_ids(c, c->n_msgs, 0, 1, &o);
    c->msgs = msgs;
    c->n_msgs = n_msgs;
    c->n_ids = c->n_text = 0;
    for (size_t i = 0; rc == JANAS_LLM_OK && i < o.n; i++)
        if (janas_api_add_special(c, o.v[i]) != 0)
            rc = janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    free(o.v);
    free(o.pp);
    return rc;
}
