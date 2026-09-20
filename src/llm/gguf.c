/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gguf.c - the head of a GGUF file (see gguf.h).
 *
 * Layout: "GGUF", u32 version (2 or 3), u64 tensor count, u64 key-value
 * count; the key-value pairs (string key, u32 type, value); the tensor
 * directory (string name, u32 dims, u64 dims[], u32 type, u64 offset);
 * the data, from the next multiple of general.alignment (32 when unsaid).
 * Strings are a u64 length and the bytes; everything little-endian.
 */
#include "llm/gguf.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_DEPTH 4                /* arrays of arrays of ... */
#define MAX_HEAD ((size_t)1 << 30) /* a head larger is no head */

int janas_gguf_type(uint32_t type, uint64_t *blck, uint64_t *size)
{
    /* the ggml types the converter takes (as the Python script before it
       did), and src/llm/jns.c reads */
    static const struct {
        uint32_t type, blck, size;
    } t[] = {
        {0, 1, 4},      {1, 1, 2},      {2, 32, 18},    {3, 32, 20},
        {6, 32, 22},    {7, 32, 24},    {8, 32, 34},    {10, 256, 84},
        {11, 256, 110}, {12, 256, 144}, {13, 256, 176}, {14, 256, 210},
        {15, 256, 292}, {20, 32, 18},   {21, 256, 110}, {23, 256, 136},
        {30, 1, 2},     {39, 32, 17},
    };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++)
        if (t[i].type == type) {
            *blck = t[i].blck;
            *size = t[i].size;
            return 1;
        }
    return 0;
}

/* A cursor over the head: every read checks what is left. */
struct cur {
    const uint8_t *p;
    size_t n, at;
    int short_; /* ran past the end: more of the file may help */
};

static int take(struct cur *c, size_t n, const uint8_t **out)
{
    if (n > c->n - c->at) {
        c->short_ = 1;
        return -1;
    }
    *out = c->p + c->at;
    c->at += n;
    return 0;
}

static int u32(struct cur *c, uint32_t *v)
{
    const uint8_t *p;
    if (take(c, 4, &p))
        return -1;
    memcpy(v, p, 4);
    return 0;
}

static int u64(struct cur *c, uint64_t *v)
{
    const uint8_t *p;
    if (take(c, 8, &p))
        return -1;
    memcpy(v, p, 8);
    return 0;
}

static int str(struct cur *c, struct janas_gguf_str *s)
{
    if (u64(c, &s->n))
        return -1;
    if (s->n > MAX_HEAD) /* no string of a head is that long */
        return -1;
    return take(c, (size_t)s->n, &s->p);
}

/* Bytes of a scalar value type; 0 for strings and arrays, -1 unknown. */
static int scalar_bytes(uint32_t t)
{
    static const int8_t b[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    return t < sizeof(b) ? b[t] : -1;
}

static int skip_value(struct cur *c, uint32_t t, int depth)
{
    int sb = scalar_bytes(t);
    const uint8_t *p;
    if (sb < 0 || depth > MAX_DEPTH)
        return -1;
    if (sb > 0)
        return take(c, (size_t)sb, &p);
    if (t == 8) {
        struct janas_gguf_str s;
        return str(c, &s);
    }
    uint32_t et;
    uint64_t n;
    if (u32(c, &et) || u64(c, &n))
        return -1;
    int eb = scalar_bytes(et);
    if (eb < 0)
        return -1;
    if (eb > 0) { /* a run of scalars, checked in one go */
        uint64_t total;
        if (__builtin_mul_overflow(n, (uint64_t)eb, &total) || total > MAX_HEAD)
            return -1;
        return take(c, (size_t)total, &p);
    }
    for (uint64_t i = 0; i < n; i++) {
        if (skip_value(c, et, depth + 1))
            return -1;
        if (c->at == c->n && i + 1 < n) {
            c->short_ = 1;
            return -1;
        }
    }
    return 0;
}

void janas_gguf_free(struct janas_gguf *g)
{
    free(g->head);
    free(g->kv);
    free(g->t);
    memset(g, 0, sizeof(*g));
}

static int fail(struct janas_gguf *g, int rc, char *err, size_t err_len,
                const char *why)
{
    snprintf(err, err_len, "%s", why);
    janas_gguf_free(g);
    return rc;
}

int janas_gguf_parse(uint8_t *head, size_t n, uint64_t file_bytes,
                     struct janas_gguf *g, char *err, size_t err_len)
{
    memset(g, 0, sizeof(*g));
    g->head = head;
    g->n_head = n;
    g->file_bytes = file_bytes;
    struct cur c = {.p = head, .n = n};
    const uint8_t *magic;
    if (take(&c, 4, &magic) || memcmp(magic, "GGUF", 4) != 0)
        return fail(g, -1, err, err_len, "not a GGUF file");
    if (u32(&c, &g->version) || u64(&c, &g->n_tensors) || u64(&c, &g->n_kv))
        return fail(g, -1, err, err_len, "not a GGUF file");
    if (g->version < 2 || g->version > 3)
        return fail(g, -1, err, err_len,
                    "GGUF version not supported (2 and 3 are)");
    /* a pair takes at least 12 bytes, a tensor at least 24: counts beyond
       what the file could hold are false */
    if (g->n_kv > file_bytes / 12 || g->n_tensors > file_bytes / 24)
        return fail(g, -1, err, err_len, "counts larger than the file");
    g->kv = calloc(g->n_kv ? g->n_kv : 1, sizeof(*g->kv));
    g->t = calloc(g->n_tensors ? g->n_tensors : 1, sizeof(*g->t));
    if (!g->kv || !g->t)
        return fail(g, -1, err, err_len, "out of memory");
    g->kv_start = c.at;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        struct janas_gguf_kv *kv = &g->kv[i];
        if (str(&c, &kv->key) || u32(&c, &kv->type))
            goto bad;
        kv->val = c.p + c.at;
        if (skip_value(&c, kv->type, 0))
            goto bad;
    }
    g->kv_end = c.at;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        struct janas_gguf_tensor *t = &g->t[i];
        if (str(&c, &t->name) || u32(&c, &t->n_dims))
            goto bad;
        if (t->n_dims == 0 || t->n_dims > 4)
            return fail(g, -1, err, err_len,
                        "a tensor of 0 or more than 4 "
                        "dimensions");
        for (uint32_t d = 0; d < t->n_dims; d++)
            if (u64(&c, &t->dims[d]))
                goto bad;
        for (uint32_t d = t->n_dims; d < 4; d++)
            t->dims[d] = 1;
        if (u32(&c, &t->type) || u64(&c, &t->offset))
            goto bad;
        uint64_t blck, size, count = 1, bytes;
        if (!janas_gguf_type(t->type, &blck, &size))
            return fail(g, -1, err, err_len,
                        "a tensor of a ggml type not "
                        "supported");
        for (uint32_t d = 0; d < 4; d++)
            if (__builtin_mul_overflow(count, t->dims[d], &count))
                return fail(g, -1, err, err_len, "a tensor too large");
        if (t->dims[0] % blck ||
            __builtin_mul_overflow(count / blck, size, &bytes))
            return fail(g, -1, err, err_len,
                        "a tensor's rows are not whole blocks");
        t->bytes = bytes;
    }
    /* the data, aligned; every tensor inside the file */
    uint64_t align = 32;
    if (janas_gguf_uint(g, "general.alignment", &align) == 0 &&
        (align == 0 || align > 65536 || (align & (align - 1))))
        return fail(g, -1, err, err_len, "general.alignment not a power of 2");
    g->data_start = (c.at + align - 1) / align * align;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const struct janas_gguf_tensor *t = &g->t[i];
        uint64_t end;
        if (__builtin_add_overflow(g->data_start, t->offset, &end) ||
            __builtin_add_overflow(end, t->bytes, &end) || end > file_bytes)
            return fail(g, -1, err, err_len,
                        "a tensor's data past the end of the file");
    }
    return 0;
bad:
    if (c.short_ && n < file_bytes && n < MAX_HEAD) {
        /* the head is cut here: the caller reads more and asks again */
        g->head = NULL; /* still the caller's, to grow */
        janas_gguf_free(g);
        snprintf(err, err_len, "head cut short");
        return 1;
    }
    return fail(g, -1, err, err_len, "a malformed GGUF head");
}

int janas_gguf_open(const char *path, struct janas_gguf *g, char *err,
                    size_t err_len)
{
    memset(g, 0, sizeof(*g));
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) {
        snprintf(err, err_len, "%s: cannot open", path);
        if (fd >= 0)
            close(fd);
        return -1;
    }
    uint64_t file_bytes = (uint64_t)st.st_size;
    size_t want = (size_t)16 << 20;
    for (;;) {
        size_t n = want < file_bytes ? want : (size_t)file_bytes;
        uint8_t *head = malloc(n ? n : 1);
        size_t got = 0;
        while (head && got < n) {
            ssize_t r = pread(fd, head + got, n - got, (off_t)got);
            if (r <= 0)
                break;
            got += (size_t)r;
        }
        if (!head || got < n) {
            free(head);
            close(fd);
            snprintf(err, err_len, "%s: cannot read", path);
            return -1;
        }
        char why[160];
        int rc = janas_gguf_parse(head, n, file_bytes, g, why, sizeof(why));
        if (rc == 1 && want < MAX_HEAD) {
            free(head);
            want *= 4; /* a large vocabulary: read more of the head */
            continue;
        }
        close(fd);
        if (rc != 0) {
            if (rc == 1)
                free(head);
            snprintf(err, err_len, "%s: %s", path, why);
            return -1;
        }
        return 0;
    }
}

static int key_is(const struct janas_gguf_str *k, const char *key)
{
    size_t n = strlen(key);
    return k->n == n && memcmp(k->p, key, n) == 0;
}

int janas_gguf_uint(const struct janas_gguf *g, const char *key, uint64_t *out)
{
    for (uint64_t i = 0; i < g->n_kv; i++) {
        const struct janas_gguf_kv *kv = &g->kv[i];
        if (!key_is(&kv->key, key))
            continue;
        const uint8_t *p = kv->val;
        switch (kv->type) {
        case 0: /* u8 */
        case 7: /* bool */
            *out = p[0];
            return 0;
        case 2: { /* u16 */
            uint16_t v;
            memcpy(&v, p, 2);
            *out = v;
            return 0;
        }
        case 4: { /* u32 */
            uint32_t v;
            memcpy(&v, p, 4);
            *out = v;
            return 0;
        }
        case 10: /* u64 */
            memcpy(out, p, 8);
            return 0;
        case 1: /* the signed ones, when not negative */
        case 3:
        case 5:
        case 11: {
            int64_t v = 0;
            if (kv->type == 1)
                v = (int8_t)p[0];
            else if (kv->type == 3) {
                int16_t x;
                memcpy(&x, p, 2);
                v = x;
            } else if (kv->type == 5) {
                int32_t x;
                memcpy(&x, p, 4);
                v = x;
            } else
                memcpy(&v, p, 8);
            if (v < 0)
                return -1;
            *out = (uint64_t)v;
            return 0;
        }
        default:
            return -1;
        }
    }
    return -1;
}

int janas_gguf_string(const struct janas_gguf *g, const char *key,
                      struct janas_gguf_str *out)
{
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (key_is(&g->kv[i].key, key) && g->kv[i].type == 8) {
            memcpy(&out->n, g->kv[i].val, 8);
            out->p = g->kv[i].val + 8;
            return 0;
        }
    return -1;
}

const struct janas_gguf_tensor *janas_gguf_tensor(const struct janas_gguf *g,
                                                  const char *name)
{
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (key_is(&g->t[i].name, name))
            return &g->t[i];
    return NULL;
}
