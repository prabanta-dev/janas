/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * store.c - the index on disk (~/.local/share/janas/docs/index.bin): what
 * made its vectors and their length, then each document with its time and
 * size as last read and its pieces, each with its vector or without.
 * Little-endian, as the machines it runs on:
 *
 *   "JDOCIDX1"  u16 n, source[n]  u32 dim  u32 files
 *   each file:  u16 n, path[n]  i64 mtime  i64 size  u32 pieces
 *   each piece: u32 n, text[n]  u8 has_vector  [f32 vector[dim]]
 *
 * A file that is broken or of another version is read as no index at all:
 * the documents are read again, nothing worse. Every length is checked
 * against what is left before it is believed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "docs.h"

static const char MAGIC[8] = {'J', 'D', 'O', 'C', 'I', 'D', 'X', '1'};

struct rd {
    const uint8_t *p;
    size_t n, at;
    int bad;
};

static const uint8_t *take(struct rd *r, size_t k)
{
    if (r->bad || r->n - r->at < k) {
        r->bad = 1;
        return NULL;
    }
    const uint8_t *p = r->p + r->at;
    r->at += k;
    return p;
}

static uint64_t num(struct rd *r, int bytes)
{
    const uint8_t *p = take(r, (size_t)bytes);
    uint64_t v = 0;
    for (int i = 0; p && i < bytes; i++)
        v |= (uint64_t)p[i] << (8 * i);
    return v;
}

static char *str(struct rd *r, size_t n)
{
    const uint8_t *p = take(r, n);
    char *s = p ? malloc(n + 1) : NULL;
    if (s) {
        memcpy(s, p, n);
        s[n] = 0;
    } else {
        r->bad = 1;
    }
    return s;
}

void dz_index_free(struct dz_index *x)
{
    for (size_t i = 0; i < x->n; i++) {
        for (uint32_t k = 0; k < x->f[i].n_chunk; k++) {
            free(x->f[i].c[k].text);
            free(x->f[i].c[k].vec);
        }
        free(x->f[i].c);
        free(x->f[i].path);
    }
    free(x->f);
    x->f = NULL;
    x->n = x->cap = 0;
}

/* the files of the buffer into x; 0, or -1 */
static int parse(struct dz_index *x, struct rd *r)
{
    const uint8_t *m = take(r, 8);
    if (!m || memcmp(m, MAGIC, 8))
        return -1;
    size_t sn = (size_t)num(r, 2);
    char *src = str(r, sn);
    uint32_t dim = (uint32_t)num(r, 4), files = (uint32_t)num(r, 4);
    if (r->bad || dim > 65536 || files > DZ_MAX_FILES ||
        sn >= sizeof x->source) {
        free(src);
        return -1;
    }
    snprintf(x->source, sizeof x->source, "%s", src);
    free(src);
    x->dim = (int)dim;
    x->f = calloc(files ? files : 1, sizeof(*x->f));
    if (!x->f)
        return -1;
    x->cap = files ? files : 1;
    for (uint32_t i = 0; i < files && !r->bad; i++) {
        struct dz_file *f = &x->f[x->n];
        size_t pn = (size_t)num(r, 2);
        if (pn == 0 || pn >= PATH_MAX)
            return -1;
        f->path = str(r, pn);
        x->n++; /* freed with the rest from here on */
        f->mtime = (int64_t)num(r, 8);
        f->size = (int64_t)num(r, 8);
        uint32_t pieces = (uint32_t)num(r, 4);
        /* a piece takes 5 bytes at least: no more than there is room for */
        if (r->bad || pieces > (r->n - r->at) / 5 ||
            (x->n > 1 && strcmp(x->f[x->n - 2].path, f->path) >= 0))
            return -1;
        f->c = calloc(pieces ? pieces : 1, sizeof(*f->c));
        if (!f->c)
            return -1;
        for (uint32_t k = 0; k < pieces && !r->bad; k++) {
            struct dz_chunk *c = &f->c[k];
            uint32_t n = (uint32_t)num(r, 4);
            if (n > 4 * DZ_CHUNK)
                return -1;
            c->text = str(r, n);
            c->n = n;
            f->n_chunk++;
            uint8_t has = (uint8_t)num(r, 1);
            if (has > 1 || (has && !dim))
                return -1;
            if (has) {
                const uint8_t *v = take(r, (size_t)dim * 4);
                c->vec = v ? malloc((size_t)dim * 4) : NULL;
                if (!c->vec)
                    return -1;
                memcpy(c->vec, v, (size_t)dim * 4);
            }
        }
    }
    return r->bad || r->at != r->n ? -1 : 0;
}

int dz_index_parse(struct dz_index *x, const uint8_t *p, size_t n)
{
    struct rd r = {.p = p, .n = n};
    if (parse(x, &r) != 0) {
        dz_index_free(x);
        x->source[0] = 0;
        x->dim = 0;
        return -1;
    }
    return 0;
}

int dz_index_load(struct dz_index *x, const char *path, char *why, size_t wn)
{
    snprintf(x->path, sizeof x->path, "%s", path);
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0; /* none yet */
    struct stat st;
    uint8_t *p = NULL;
    size_t n = 0;
    if (!fstat(fileno(f), &st) && st.st_size > 0) {
        n = (size_t)st.st_size;
        p = malloc(n);
        if (p && fread(p, 1, n, f) != n) {
            free(p);
            p = NULL;
        }
    }
    fclose(f);
    if (!p) {
        snprintf(why, wn, "%s cannot be read", path);
        return -1;
    }
    int rc = dz_index_parse(x, p, n);
    free(p);
    if (rc != 0)
        snprintf(why, wn, "%s is not an index of this version: made again",
                 path);
    return rc;
}

static void wr(FILE *f, uint64_t v, int bytes)
{
    uint8_t b[8];
    for (int i = 0; i < bytes; i++)
        b[i] = (uint8_t)(v >> (8 * i));
    fwrite(b, 1, (size_t)bytes, f);
}

int dz_index_save(struct dz_index *x)
{
    char tmp[PATH_MAX + 8];
    snprintf(tmp, sizeof tmp, "%s.new", x->path);
    FILE *f = fopen(tmp, "wb");
    if (!f)
        return -1;
    fwrite(MAGIC, 1, 8, f);
    size_t sn = strlen(x->source);
    wr(f, sn, 2);
    fwrite(x->source, 1, sn, f);
    wr(f, (uint64_t)x->dim, 4);
    wr(f, x->n, 4);
    for (size_t i = 0; i < x->n; i++) {
        const struct dz_file *d = &x->f[i];
        size_t pn = strlen(d->path);
        wr(f, pn, 2);
        fwrite(d->path, 1, pn, f);
        wr(f, (uint64_t)d->mtime, 8);
        wr(f, (uint64_t)d->size, 8);
        wr(f, d->n_chunk, 4);
        for (uint32_t k = 0; k < d->n_chunk; k++) {
            const struct dz_chunk *c = &d->c[k];
            wr(f, c->n, 4);
            fwrite(c->text, 1, c->n, f);
            wr(f, c->vec && x->dim, 1);
            if (c->vec && x->dim)
                fwrite(c->vec, 4, (size_t)x->dim, f);
        }
    }
    int bad = ferror(f);
    bad |= fflush(f) != 0 || fsync(fileno(f)) != 0;
    bad |= fclose(f) != 0;
    if (bad || rename(tmp, x->path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

long dz_index_find(const struct dz_index *x, const char *path, size_t *at)
{
    size_t lo = 0, hi = x->n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = strcmp(x->f[mid].path, path);
        if (!c)
            return (long)mid;
        if (c < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (at)
        *at = lo;
    return -1;
}
