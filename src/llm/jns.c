/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jns.c - reader and validator of the Janas-LLM model file (see jns.h).
 */
#define _GNU_SOURCE
#include "jns.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Largest header + tables prefix the reader accepts, to bound allocations. */
#define MAX_TABLES_BYTES ((uint64_t)256 << 20)

uint64_t janas_fnv1a(const void *data, size_t n, uint64_t h)
{
    const uint8_t *p = data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

static uint32_t crc_table[256];

/* Filled before main, so concurrent readers never race with the writer. */
__attribute__((constructor)) static void crc_table_init(void)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
        crc_table[i] = c;
    }
}

uint32_t janas_crc32(uint32_t crc, const void *data, size_t n)
{
    const uint8_t *p = data;
    crc = ~crc;
    for (size_t i = 0; i < n; i++)
        crc = crc_table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

static int fail(char *err, size_t err_len, const char *fmt, ...)
{
    if (err && err_len) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, err_len, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* Block size in weights and bytes per block of a ggml type; 0 if unknown. */
static int type_geometry(uint32_t type, uint64_t *blck, uint64_t *size)
{
    static const struct {
        uint32_t type;
        uint32_t blck, size;
    } t[] = {
        {0, 1, 4},      /* F32 */
        {1, 1, 2},      /* F16 */
        {2, 32, 18},    /* Q4_0 */
        {3, 32, 20},    /* Q4_1 */
        {6, 32, 22},    /* Q5_0 */
        {7, 32, 24},    /* Q5_1 */
        {8, 32, 34},    /* Q8_0 */
        {10, 256, 84},  /* Q2_K */
        {11, 256, 110}, /* Q3_K */
        {12, 256, 144}, /* Q4_K */
        {13, 256, 176}, /* Q5_K */
        {14, 256, 210}, /* Q6_K */
        {214, 256, 82}, /* Q6_K_P, the base plane and the scales */
        {15, 256, 292}, /* Q8_K */
        {20, 32, 18},   /* IQ4_NL */
        {21, 256, 110}, /* IQ3_S */
        {23, 256, 136}, /* IQ4_XS */
        {30, 1, 2},     /* BF16 */
        {39, 32, 17},   /* MXFP4 */
    };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++)
        if (t[i].type == type) {
            *blck = t[i].blck;
            *size = t[i].size;
            return 1;
        }
    return 0;
}

/* [off, off + len) lies inside [lo, hi), without overflow. */
static int inside(uint64_t off, uint64_t len, uint64_t lo, uint64_t hi)
{
    uint64_t end;
    if (__builtin_add_overflow(off, len, &end))
        return 0;
    return off >= lo && end <= hi;
}

/* Bytes of an array of n entries of size sz, or 0 on overflow. */
static uint64_t array_bytes(uint64_t n, uint64_t sz)
{
    uint64_t r;
    return __builtin_mul_overflow(n, sz, &r) ? 0 : r;
}

/* GGUF value types */
enum {
    GG_U8 = 0,
    GG_I8,
    GG_U16,
    GG_I16,
    GG_U32,
    GG_I32,
    GG_F32,
    GG_BOOL,
    GG_STRING,
    GG_ARRAY,
    GG_U64,
    GG_I64,
    GG_F64
};

static uint64_t scalar_size(uint32_t t)
{
    switch (t) {
    case GG_U8:
    case GG_I8:
    case GG_BOOL:
        return 1;
    case GG_U16:
    case GG_I16:
        return 2;
    case GG_U32:
    case GG_I32:
    case GG_F32:
        return 4;
    case GG_U64:
    case GG_I64:
    case GG_F64:
        return 8;
    default:
        return 0;
    }
}

/* Reads a u64 at *pos if it fits before end. */
static int take_u64(const uint8_t *m, uint64_t *pos, uint64_t end, uint64_t *v)
{
    if (end - *pos < 8)
        return 0;
    memcpy(v, m + *pos, 8);
    *pos += 8;
    return 1;
}

static int take_u32(const uint8_t *m, uint64_t *pos, uint64_t end, uint32_t *v)
{
    if (end - *pos < 4)
        return 0;
    memcpy(v, m + *pos, 4);
    *pos += 4;
    return 1;
}

/* Skips a string (u64 length + bytes); returns 0 if it does not fit. */
static int skip_string(const uint8_t *m, uint64_t *pos, uint64_t end)
{
    uint64_t n;
    if (!take_u64(m, pos, end, &n) || n > end - *pos)
        return 0;
    *pos += n;
    return 1;
}

/* Skips one value of type t; arrays of arrays are refused. */
static int skip_value(const uint8_t *m, uint64_t *pos, uint64_t end, uint32_t t,
                      int in_array)
{
    uint64_t sz = scalar_size(t);
    if (sz) {
        if (end - *pos < sz)
            return 0;
        *pos += sz;
        return 1;
    }
    if (t == GG_STRING)
        return skip_string(m, pos, end);
    if (t == GG_ARRAY && !in_array) {
        uint32_t et;
        uint64_t n, bytes;
        if (!take_u32(m, pos, end, &et) || !take_u64(m, pos, end, &n))
            return 0;
        sz = scalar_size(et);
        if (sz) {
            if (__builtin_mul_overflow(n, sz, &bytes) || bytes > end - *pos)
                return 0;
            *pos += bytes;
            return 1;
        }
        if (et != GG_STRING || n > end - *pos) /* each string >= 8 bytes */
            return 0;
        for (uint64_t i = 0; i < n; i++)
            if (!skip_string(m, pos, end))
                return 0;
        return 1;
    }
    return 0;
}

/* Walks every entry; 0 if the region is well formed and fully used. */
static int check_metadata(const uint8_t *m, uint64_t bytes)
{
    uint64_t pos = 0, n;
    if (!take_u64(m, &pos, bytes, &n) || n > bytes / 13)
        return 0;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t t;
        if (!skip_string(m, &pos, bytes) || !take_u32(m, &pos, bytes, &t) ||
            !skip_value(m, &pos, bytes, t, 0))
            return 0;
    }
    return 1;
}

/* Finds a key; on success *type and *pos (start of the value) are set. */
static int find_key(const struct janas_jns *j, const char *key, uint32_t *type,
                    uint64_t *pos)
{
    const uint8_t *m = j->metadata;
    uint64_t end = j->h.metadata_bytes, p = 0, n, klen;
    size_t want = strlen(key);
    if (!m || !take_u64(m, &p, end, &n))
        return 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!take_u64(m, &p, end, &klen) || klen > end - p)
            return 0;
        int match = klen == want && memcmp(m + p, key, want) == 0;
        p += klen;
        if (!take_u32(m, &p, end, type))
            return 0;
        if (match) {
            *pos = p;
            return 1;
        }
        if (!skip_value(m, &p, end, *type, 0))
            return 0;
    }
    return 0;
}

int janas_jns_meta_u32(const struct janas_jns *j, const char *key,
                       uint32_t *out)
{
    uint32_t t;
    uint64_t p;
    if (!find_key(j, key, &t, &p))
        return -1;
    const uint8_t *v = j->metadata + p;
    uint64_t x = 0;
    switch (t) {
    case GG_U8:
    case GG_BOOL:
        x = v[0];
        break;
    case GG_U16: {
        uint16_t y;
        memcpy(&y, v, 2);
        x = y;
        break;
    }
    case GG_U32:
    case GG_I32: {
        uint32_t y;
        memcpy(&y, v, 4);
        x = y;
        break;
    }
    case GG_U64:
    case GG_I64:
        memcpy(&x, v, 8);
        break;
    default:
        return -1;
    }
    if (x > UINT32_MAX)
        return -1;
    *out = (uint32_t)x;
    return 0;
}

int janas_jns_meta_f32(const struct janas_jns *j, const char *key, float *out)
{
    uint32_t t;
    uint64_t p;
    if (!find_key(j, key, &t, &p))
        return -1;
    if (t == GG_F32) {
        memcpy(out, j->metadata + p, 4);
        return 0;
    }
    if (t == GG_F64) {
        double d;
        memcpy(&d, j->metadata + p, 8);
        *out = (float)d;
        return 0;
    }
    return -1;
}

int janas_jns_meta_strings(const struct janas_jns *j, const char *key,
                           uint64_t *count, const uint8_t **data)
{
    uint32_t t, et;
    uint64_t p;
    if (!find_key(j, key, &t, &p) || t != GG_ARRAY)
        return -1;
    /* validated at open: element type, count, then the elements */
    memcpy(&et, j->metadata + p, 4);
    if (et != GG_STRING)
        return -1;
    memcpy(count, j->metadata + p + 4, 8);
    *data = j->metadata + p + 12;
    return 0;
}

int janas_jns_meta_ints(const struct janas_jns *j, const char *key,
                        uint64_t *count, const int32_t **data)
{
    uint32_t t, et;
    uint64_t p;
    if (!find_key(j, key, &t, &p) || t != GG_ARRAY)
        return -1;
    memcpy(&et, j->metadata + p, 4);
    if (et != GG_I32 && et != GG_U32)
        return -1;
    memcpy(count, j->metadata + p + 4, 8);
    *data = (const int32_t *)(const void *)(j->metadata + p + 12);
    return 0;
}

int janas_jns_meta_bytes(const struct janas_jns *j, const char *key,
                         uint64_t *count, const uint8_t **data)
{
    uint32_t t, et;
    uint64_t p;
    if (!find_key(j, key, &t, &p) || t != GG_ARRAY)
        return -1;
    memcpy(&et, j->metadata + p, 4);
    if (et != GG_BOOL && et != GG_U8 && et != GG_I8)
        return -1;
    memcpy(count, j->metadata + p + 4, 8);
    *data = j->metadata + p + 12;
    return 0;
}

int janas_jns_meta_str(const struct janas_jns *j, const char *key, char *out,
                       size_t len)
{
    uint32_t t;
    uint64_t p, n;
    if (len == 0 || !find_key(j, key, &t, &p) || t != GG_STRING)
        return -1;
    memcpy(&n, j->metadata + p, 8);
    size_t c = n < len - 1 ? (size_t)n : len - 1;
    memcpy(out, j->metadata + p + 8, c);
    out[c] = 0;
    return 0;
}

const struct janas_jns_tensor *janas_jns_tensor(const struct janas_jns *j,
                                                const char *name)
{
    for (uint64_t t = 0; t < j->h.n_tensor; t++)
        if (strcmp(j->tensors[t].name, name) == 0)
            return &j->tensors[t];
    return NULL;
}

int janas_jns_parse(const uint8_t *image, size_t bytes, uint64_t file_bytes,
                    struct janas_jns *out, char *err, size_t err_len)
{
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    struct janas_jns_header *h = &out->h;

    if (bytes < JANAS_JNS_ALIGN)
        return fail(err, err_len, "file shorter than the header");
    memcpy(h, image, sizeof(*h));
    if (memcmp(h->magic, JANAS_JNS_MAGIC, 8) != 0)
        return fail(err, err_len, "not a JNS file");
    if (h->version != JANAS_JNS_VERSION && h->version != 2)
        return fail(err, err_len, "unsupported JNS version %u", h->version);
    if (h->header_bytes != JANAS_JNS_ALIGN || h->file_bytes != file_bytes)
        return fail(err, err_len, "header size or file size mismatch");

    struct janas_jns_header zeroed = *h;
    zeroed.header_checksum = 0;
    if (janas_fnv1a(&zeroed, sizeof(zeroed), JANAS_FNV_INIT) !=
        h->header_checksum)
        return fail(err, err_len, "header checksum mismatch");

    if (h->n_layer < 1 || h->n_layer > JANAS_JNS_MAX_LAYERS ||
        h->n_expert < 1 || h->n_expert > JANAS_JNS_MAX_EXPERTS ||
        h->n_expert_used < 1 || h->n_expert_used > h->n_expert ||
        h->n_tensor > JANAS_JNS_MAX_TENSORS)
        return fail(err, err_len, "model dimensions out of range");

    /* tables: all inside the in-memory image, in file order */
    uint64_t n_slots = (uint64_t)h->n_layer * h->n_expert; /* < 2^22 */
    /* the table as the file holds it: version 2 has no levels or planes */
    size_t layer_entry = h->version == 2 ? sizeof(struct janas_jns_layer_v2)
                                         : sizeof(struct janas_jns_layer);
    uint64_t layer_len = array_bytes(h->n_layer, layer_entry);
    uint64_t slot_len = array_bytes(n_slots, sizeof(uint64_t));
    uint64_t dir_len =
        array_bytes(h->n_tensor, sizeof(struct janas_jns_tensor));
    if (!inside(h->layer_table_offset, layer_len, JANAS_JNS_ALIGN, bytes) ||
        !inside(h->expert_table_offset, slot_len, JANAS_JNS_ALIGN, bytes) ||
        !inside(h->expert_sums_offset, slot_len, JANAS_JNS_ALIGN, bytes) ||
        !inside(h->tensor_dir_offset, dir_len, JANAS_JNS_ALIGN, bytes) ||
        h->expert_table_offset < h->layer_table_offset + layer_len ||
        h->expert_sums_offset < h->expert_table_offset + slot_len ||
        h->tensor_dir_offset < h->expert_sums_offset + slot_len)
        return fail(err, err_len, "tables outside the header area");

    uint64_t sum = JANAS_FNV_INIT;
    sum = janas_fnv1a(image + h->layer_table_offset, layer_len, sum);
    sum = janas_fnv1a(image + h->expert_table_offset, slot_len, sum);
    sum = janas_fnv1a(image + h->expert_sums_offset, slot_len, sum);
    sum = janas_fnv1a(image + h->tensor_dir_offset, dir_len, sum);
    if (sum != h->tables_checksum)
        return fail(err, err_len, "table checksum mismatch");

    /* metadata: after the tables, inside the image */
    if (!inside(h->metadata_offset, h->metadata_bytes, JANAS_JNS_ALIGN,
                bytes) ||
        h->metadata_offset < h->tensor_dir_offset + dir_len ||
        h->metadata_bytes < 8 ||
        janas_crc32(0, image + h->metadata_offset, h->metadata_bytes) !=
            h->metadata_checksum ||
        !check_metadata(image + h->metadata_offset, h->metadata_bytes))
        return fail(err, err_len, "bad metadata region");

    /* data regions */
    if (h->resident_offset % JANAS_JNS_ALIGN ||
        h->experts_offset % JANAS_JNS_ALIGN ||
        !inside(h->resident_offset, h->resident_bytes, bytes, file_bytes) ||
        !inside(h->experts_offset, h->experts_bytes, h->resident_offset,
                file_bytes) ||
        h->experts_offset < h->resident_offset + h->resident_bytes)
        return fail(err, err_len, "data regions out of place");

    out->layers = malloc(array_bytes(h->n_layer, sizeof(*out->layers)));
    out->expert_offset = malloc(slot_len);
    out->expert_sum = malloc(slot_len);
    out->tensors = malloc(dir_len ? dir_len : 1);
    out->metadata = malloc(h->metadata_bytes);
    if (!out->layers || !out->expert_offset || !out->expert_sum ||
        !out->tensors || !out->metadata) {
        janas_jns_close(out);
        return fail(err, err_len, "out of memory");
    }
    if (h->version == 2) {
        for (uint32_t l = 0; l < h->n_layer; l++) {
            struct janas_jns_layer_v2 v2;
            memcpy(&v2, image + h->layer_table_offset + (size_t)l * layer_entry,
                   sizeof(v2));
            struct janas_jns_layer *ly = &out->layers[l];
            memset(ly, 0, sizeof(*ly));
            ly->slot_bytes = v2.slot_bytes;
            for (int i = 0; i < 3; i++) {
                ly->level_bytes[i] = v2.slot_bytes;
                ly->m[i] = v2.m[i];
            }
        }
    } else {
        memcpy(out->layers, image + h->layer_table_offset, layer_len);
        /* hf2jns_mtp before issue #6 wrote version 3 layers with the levels
           left at zero: never a valid value, so read as what it wrote, a
           slot with no planes whose levels are all of it */
        for (uint32_t l = 0; l < h->n_layer; l++) {
            struct janas_jns_layer *ly = &out->layers[l];
            if (!ly->level_bytes[0] && !ly->level_bytes[1] &&
                !ly->level_bytes[2] && !ly->plane[0].type &&
                !ly->plane[1].type) {
                for (int i = 0; i < 3; i++)
                    ly->level_bytes[i] = ly->slot_bytes;
                out->zero_levels++;
            }
        }
    }
    memcpy(out->expert_offset, image + h->expert_table_offset, slot_len);
    memcpy(out->expert_sum, image + h->expert_sums_offset, slot_len);
    memcpy(out->tensors, image + h->tensor_dir_offset, dir_len);
    memcpy(out->metadata, image + h->metadata_offset, h->metadata_bytes);

    uint64_t experts_end = h->experts_offset + h->experts_bytes;
    for (uint32_t l = 0; l < h->n_layer; l++) {
        const struct janas_jns_layer *ly = &out->layers[l];
        if (ly->slot_bytes == 0 || ly->slot_bytes % JANAS_JNS_ALIGN) {
            janas_jns_close(out);
            return fail(err, err_len, "layer %u: bad slot size", l);
        }
        for (int m = 0; m < 3; m++) {
            const struct janas_jns_matrix *mx = &ly->m[m];
            uint64_t blck, size, row_bytes, total;
            if (!type_geometry(mx->type, &blck, &size) || mx->rows == 0 ||
                mx->cols == 0 || mx->cols % blck ||
                __builtin_mul_overflow(mx->cols / blck, size, &row_bytes) ||
                __builtin_mul_overflow(row_bytes, mx->rows, &total) ||
                total != mx->bytes || mx->offset % 256 ||
                !inside(mx->offset, mx->bytes, 0, ly->slot_bytes)) {
                janas_jns_close(out);
                return fail(err, err_len, "layer %u matrix %d: bad geometry", l,
                            m);
            }
        }
        /* levels: growing prefixes of the slot, the last the whole slot */
        if (ly->level_bytes[2] != ly->slot_bytes ||
            ly->level_bytes[0] % JANAS_JNS_ALIGN ||
            ly->level_bytes[1] % JANAS_JNS_ALIGN ||
            ly->level_bytes[0] > ly->level_bytes[1] ||
            ly->level_bytes[1] > ly->level_bytes[2]) {
            janas_jns_close(out);
            return fail(err, err_len, "layer %u: bad levels", l);
        }
        for (int m = 0; m < 3; m++)
            if (ly->m[m].offset + ly->m[m].bytes > ly->level_bytes[0]) {
                janas_jns_close(out);
                return fail(err, err_len, "layer %u: matrix %d past level 1", l,
                            m);
            }
        for (int m = 0; m < 2; m++) {
            const struct janas_jns_matrix *px = &ly->plane[m];
            if (px->type == 0) { /* no planes: the levels must be one */
                if (px->rows || px->cols || px->bytes ||
                    ly->level_bytes[m] != ly->slot_bytes) {
                    janas_jns_close(out);
                    return fail(err, err_len, "layer %u: bad plane %d", l, m);
                }
                continue;
            }
            uint64_t total;
            /* a bare plane: 64 bytes per 256 weights, no scales */
            if (ly->m[JANAS_JNS_DOWN].type != 214 ||
                px->type != ly->m[JANAS_JNS_DOWN].type ||
                px->rows != ly->m[JANAS_JNS_DOWN].rows ||
                px->cols != ly->m[JANAS_JNS_DOWN].cols || px->cols % 256 ||
                __builtin_mul_overflow((uint64_t)px->rows * (px->cols / 256),
                                       64, &total) ||
                total != px->bytes || px->offset % 256 ||
                px->offset < ly->level_bytes[m] ||
                px->offset + px->bytes > ly->level_bytes[m + 1]) {
                janas_jns_close(out);
                return fail(err, err_len, "layer %u: bad plane %d", l, m);
            }
        }
        for (uint32_t e = 0; e < h->n_expert; e++) {
            uint64_t off = out->expert_offset[(size_t)l * h->n_expert + e];
            if (off % JANAS_JNS_ALIGN ||
                !inside(off, ly->slot_bytes, h->experts_offset, experts_end)) {
                janas_jns_close(out);
                return fail(err, err_len, "layer %u expert %u: bad offset", l,
                            e);
            }
        }
    }
    for (uint64_t t = 0; t < h->n_tensor; t++) {
        const struct janas_jns_tensor *tn = &out->tensors[t];
        uint64_t blck, size;
        if (memchr(tn->name, 0, sizeof(tn->name)) == NULL || tn->n_dims < 1 ||
            tn->n_dims > 4 || !type_geometry(tn->type, &blck, &size) ||
            !inside(tn->offset, tn->bytes, h->resident_offset,
                    h->resident_offset + h->resident_bytes)) {
            janas_jns_close(out);
            return fail(err, err_len, "tensor %llu: bad entry",
                        (unsigned long long)t);
        }
    }
    return 0;
}

int janas_jns_open(const char *path, struct janas_jns *out, char *err,
                   size_t err_len)
{
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return fail(err, err_len, "cannot open %s", path);
    struct stat st;
    struct janas_jns_header h;
    if (fstat(fd, &st) != 0 ||
        pread(fd, &h, sizeof(h), 0) != (ssize_t)sizeof(h)) {
        close(fd);
        return fail(err, err_len, "cannot read the header of %s", path);
    }
    /* the tables end where the resident data begins */
    uint64_t prefix = h.resident_offset;
    if (prefix < JANAS_JNS_ALIGN || prefix > MAX_TABLES_BYTES ||
        prefix > (uint64_t)st.st_size) {
        close(fd);
        return fail(err, err_len, "implausible table area in %s", path);
    }
    uint8_t *image = malloc(prefix);
    if (!image || pread(fd, image, prefix, 0) != (ssize_t)prefix) {
        free(image);
        close(fd);
        return fail(err, err_len, "cannot read the tables of %s", path);
    }
    int rc =
        janas_jns_parse(image, prefix, (uint64_t)st.st_size, out, err, err_len);
    free(image);
    if (rc != 0) {
        close(fd);
        return rc;
    }
    out->fd = fd;
    return 0;
}

void janas_jns_close(struct janas_jns *j)
{
    if (j->fd >= 0)
        close(j->fd);
    free(j->layers);
    free(j->expert_offset);
    free(j->expert_sum);
    free(j->tensors);
    free(j->metadata);
    memset(j, 0, sizeof(*j));
    j->fd = -1;
}
