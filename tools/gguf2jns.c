/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gguf2jns - converts a GGUF model into the Janas-LLM file format (JNS, see
 * src/llm/jns.h).
 *
 * Routed experts are rewritten one slot per (layer, expert), with gate, up
 * and down back to back, so the engine can read an expert with a single
 * aligned O_DIRECT request. Every other tensor goes, unchanged, into the
 * resident region. A dense model has no experts to route: its feed-forward
 * becomes the single slot of the layer, written in the same way.
 *
 * A model split into several GGUF files (name-00001-of-0000N.gguf) is read
 * from all of them: give the first. Data is copied in streaming; the model
 * is never held in memory.
 *
 * The file written is the one the Python converter it replaced wrote, byte
 * for byte (JNS version 2): the fingerprints in MODELS.md hold for both.
 *
 * Usage: gguf2jns <model.gguf> <model.jns>
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "llm/gguf.h"
#include "llm/jns.h"

#define ALIGN 4096
#define MATRIX_ALIGN 256
#define COPY_CHUNK ((size_t)16 << 20)
#define MAX_PARTS 256

static uint64_t align_up(uint64_t x, uint64_t a)
{
    return (x + a - 1) / a * a;
}

static _Noreturn void die(const char *fmt, const char *arg)
{
    fprintf(stderr, "gguf2jns: ");
    fprintf(stderr, fmt, arg);
    fputc('\n', stderr);
    exit(1);
}

/* A tensor of the model: which part of a split model, and where. */
struct tensor {
    const struct janas_gguf_tensor *t;
    int part;
    uint64_t base; /* the part's data start */
    int expert;    /* one of the experts' matrices */
    uint64_t new_offset;
};

struct part {
    struct janas_gguf g;
    int fd;
};

static struct part parts[MAX_PARTS];
static int n_parts;
static struct tensor *tensors;
static size_t n_tensors;
static uint64_t copied;
static int out_fd;
static uint8_t *buf;

static char *str_dup(const struct janas_gguf_str *s)
{
    char *d = malloc(s->n + 1);
    if (!d)
        die("%s", "out of memory");
    memcpy(d, s->p, s->n);
    d[s->n] = 0;
    return d;
}

static uint64_t need_uint(const struct janas_gguf *g, const char *key)
{
    uint64_t v;
    if (janas_gguf_uint(g, key, &v) != 0)
        die("metadata %s missing, or not a number", key);
    return v;
}

/* The tensor of this name in any part; the last part that has it wins, as
   the dictionary of the Python script had it. */
static struct tensor *find(const char *name)
{
    size_t n = strlen(name);
    for (size_t i = n_tensors; i-- > 0;)
        if (tensors[i].t->name.n == n &&
            memcmp(tensors[i].t->name.p, name, n) == 0)
            return &tensors[i];
    return NULL;
}

static void open_part(const char *path)
{
    if (n_parts == MAX_PARTS)
        die("%s: too many parts", path);
    struct part *p = &parts[n_parts];
    char err[512];
    if (janas_gguf_open(path, &p->g, err, sizeof(err)) != 0)
        die("%s", err);
    p->fd = open(path, O_RDONLY | O_CLOEXEC);
    if (p->fd < 0)
        die("%s: cannot open", path);
    struct tensor *grown =
        realloc(tensors, (n_tensors + p->g.n_tensors) * sizeof(*grown));
    if (!grown && p->g.n_tensors)
        die("%s", "out of memory");
    tensors = grown;
    for (uint64_t i = 0; i < p->g.n_tensors; i++)
        tensors[n_tensors++] = (struct tensor){
            .t = &p->g.t[i], .part = n_parts, .base = p->g.data_start};
    n_parts++;
}

/* Copies n bytes of a tensor's data, from its offset src, to dst in the
   output; with crc, their CRC-32 carried on. */
static uint32_t copy(const struct tensor *t, uint64_t src, uint64_t n,
                     uint64_t dst, uint32_t crc, int with_crc)
{
    int fd = parts[t->part].fd;
    uint64_t from = t->base + src;
    while (n) {
        size_t k = n < COPY_CHUNK ? (size_t)n : COPY_CHUNK;
        ssize_t r = pread(fd, buf, k, (off_t)from);
        if (r <= 0)
            die("%s", "unexpected end of the GGUF file");
        if (with_crc)
            crc = janas_crc32(crc, buf, (size_t)r);
        for (ssize_t w = 0; w < r;) {
            ssize_t x = pwrite(out_fd, buf + w, (size_t)(r - w),
                               (off_t)(dst + (uint64_t)w));
            if (x <= 0)
                die("write failed: %s", strerror(errno));
            w += x;
        }
        from += (uint64_t)r;
        dst += (uint64_t)r;
        n -= (uint64_t)r;
        copied += (uint64_t)r;
    }
    return crc;
}

/* The CRC-32 of n zero bytes carried on. */
static uint32_t crc_zeros(uint32_t crc, uint64_t n)
{
    static const uint8_t zeros[4096];
    while (n) {
        size_t k = n < sizeof(zeros) ? (size_t)n : sizeof(zeros);
        crc = janas_crc32(crc, zeros, k);
        n -= k;
    }
    return crc;
}

static int by_place(const void *a, const void *b)
{
    const struct tensor *x = *(const struct tensor *const *)a;
    const struct tensor *y = *(const struct tensor *const *)b;
    if (x->part != y->part)
        return x->part < y->part ? -1 : 1;
    return x->t->offset < y->t->offset ? -1 : x->t->offset > y->t->offset;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: gguf2jns <model.gguf> <model.jns>\n"
                        "  a model split into name-0000N-of-0000M.gguf is "
                        "read from all its parts: give the first\n");
        return 2;
    }
    const char *src = argv[1], *dst = argv[2];
    open_part(src);
    const struct janas_gguf *g0 = &parts[0].g;

    /* the other parts of a split model */
    uint64_t count = 1, no = 0;
    janas_gguf_uint(g0, "split.count", &count);
    if (count > 1) {
        janas_gguf_uint(g0, "split.no", &no);
        size_t n = strlen(src);
        const char *tail =
            n >= 20 ? src + n - 20 : NULL; /* -00001-of-0000N.gguf */
        if (!tail || no != 0 || count > MAX_PARTS ||
            strncmp(tail, "-00001-of-", 10) != 0 ||
            strcmp(tail + 15, ".gguf") != 0)
            die("%s: give the first file of the split model", src);
        for (uint64_t i = 2; i <= count; i++) {
            char *part = malloc(n + 1);
            if (!part)
                die("%s", "out of memory");
            snprintf(part, n + 1, "%.*s-%05llu-of-%05llu.gguf", (int)(n - 20),
                     src, (unsigned long long)i, (unsigned long long)count);
            open_part(part);
            free(part);
        }
    }

    struct janas_gguf_str arch_s;
    if (janas_gguf_string(g0, "general.architecture", &arch_s) != 0)
        die("%s", "general.architecture missing");
    char *arch = str_dup(&arch_s);
    char key[160];
    snprintf(key, sizeof(key), "%s.block_count", arch);
    uint64_t n_layer = need_uint(g0, key);
    /* a dense model is a mixture with one expert and no router: the engine
       gives it weight 1 through the same softmax, and the feed-forward of a
       layer becomes its only slot */
    snprintf(key, sizeof(key), "%s.expert_count", arch);
    uint64_t n_expert = 1, n_used = 1;
    int dense = janas_gguf_uint(g0, key, &n_expert) != 0;
    if (dense)
        n_expert = 1;
    else {
        snprintf(key, sizeof(key), "%s.expert_used_count", arch);
        n_used = need_uint(g0, key);
    }
    snprintf(key, sizeof(key), "%s.embedding_length", arch);
    uint64_t d_model = need_uint(g0, key);
    if (n_layer == 0 || n_layer > 4096 || n_expert == 0 || n_expert > 65536)
        die("%s: layers or experts out of range", src);

    /* expert geometry per layer */
    struct janas_jns_layer_v2 *layers = calloc(n_layer, sizeof(*layers));
    struct tensor *(*mats)[3] = calloc(n_layer, sizeof(*mats));
    if (!layers || !mats)
        die("%s", "out of memory");
    static const char *const part_name[3] = {"gate", "up", "down"};
    for (uint64_t l = 0; l < n_layer; l++) {
        uint64_t off = 0;
        for (int p = 0; p < 3; p++) {
            snprintf(key, sizeof(key),
                     dense ? "blk.%llu.ffn_%s.weight"
                           : "blk.%llu.ffn_%s_exps.weight",
                     (unsigned long long)l, part_name[p]);
            struct tensor *t = find(key);
            if (!t)
                die("%s missing: only models with routed experts in every "
                    "layer, or dense ones, are supported",
                    key);
            uint64_t ne = t->t->n_dims > 2 ? t->t->dims[2] : 1;
            if (ne != n_expert || t->t->bytes % n_expert ||
                t->t->dims[0] > UINT32_MAX || t->t->dims[1] > UINT32_MAX ||
                t->t->dims[3] != 1)
                die("%s: unexpected shape", key);
            uint64_t per = t->t->bytes / n_expert;
            off = align_up(off, MATRIX_ALIGN);
            layers[l].m[p] =
                (struct janas_jns_matrix){.type = t->t->type,
                                          .rows = (uint32_t)t->t->dims[1],
                                          .cols = (uint32_t)t->t->dims[0],
                                          .offset = off,
                                          .bytes = per};
            off += per;
            t->expert = 1;
            mats[l][p] = t;
        }
        layers[l].slot_bytes = align_up(off, ALIGN);
    }

    /* the resident tensors, in the order of the files */
    struct tensor **resident = malloc((n_tensors + 1) * sizeof(*resident));
    size_t n_res = 0;
    if (!resident)
        die("%s", "out of memory");
    for (size_t i = 0; i < n_tensors; i++) {
        /* a name some later part has too counts once, the later one */
        const struct janas_gguf_str *nm = &tensors[i].t->name;
        char *name = str_dup(nm);
        int shadowed = find(name) != &tensors[i];
        free(name);
        if (!tensors[i].expert && !shadowed)
            resident[n_res++] = &tensors[i];
    }
    qsort(resident, n_res, sizeof(*resident), by_place);

    /* layout */
    uint64_t metadata_bytes = 8 + (g0->kv_end - g0->kv_start);
    uint64_t layer_table = ALIGN;
    uint64_t expert_table =
        layer_table + sizeof(struct janas_jns_layer_v2) * n_layer;
    uint64_t expert_sums = expert_table + 8 * n_layer * n_expert;
    uint64_t tensor_dir = expert_sums + 8 * n_layer * n_expert;
    uint64_t metadata_offset =
        align_up(tensor_dir + sizeof(struct janas_jns_tensor) * n_res, ALIGN);
    uint64_t resident_offset =
        align_up(metadata_offset + metadata_bytes, ALIGN);
    uint64_t pos = resident_offset;
    for (size_t i = 0; i < n_res; i++) {
        resident[i]->new_offset = pos;
        pos = align_up(pos + resident[i]->t->bytes, ALIGN);
    }
    uint64_t resident_bytes = pos - resident_offset;
    uint64_t experts_offset = pos;
    uint64_t *slot_off = malloc(8 * n_layer * n_expert);
    uint64_t *sums = malloc(8 * n_layer * n_expert);
    if (!slot_off || !sums)
        die("%s", "out of memory");
    for (uint64_t l = 0; l < n_layer; l++)
        for (uint64_t e = 0; e < n_expert; e++) {
            slot_off[l * n_expert + e] = pos;
            pos += layers[l].slot_bytes;
        }
    uint64_t experts_bytes = pos - experts_offset;
    uint64_t file_bytes = pos;

    char what[96];
    if (dense)
        snprintf(what, sizeof(what), "dense feed-forward");
    else
        snprintf(what, sizeof(what), "%llu experts (k %llu)",
                 (unsigned long long)n_expert, (unsigned long long)n_used);
    printf("%s: %llu layers, %s, d_model %llu; %zu resident tensors %.2f GB, "
           "experts %.2f GB\n",
           arch, (unsigned long long)n_layer, what, (unsigned long long)d_model,
           n_res, resident_bytes / 1e9, experts_bytes / 1e9);

    out_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (out_fd < 0 || ftruncate(out_fd, (off_t)file_bytes) != 0)
        die("%s: cannot write", dst);
    buf = malloc(COPY_CHUNK);
    if (!buf)
        die("%s", "out of memory");

    for (size_t i = 0; i < n_res; i++)
        copy(resident[i], resident[i]->t->offset, resident[i]->t->bytes,
             resident[i]->new_offset, 0, 0);

    for (uint64_t l = 0; l < n_layer; l++) {
        for (uint64_t e = 0; e < n_expert; e++) {
            uint64_t slot = slot_off[l * n_expert + e];
            uint32_t crc = 0;
            uint64_t end = 0;
            for (int p = 0; p < 3; p++) {
                const struct janas_jns_matrix *m = &layers[l].m[p];
                if (m->offset > end) /* padding inside the slot is zero */
                    crc = crc_zeros(crc, m->offset - end);
                crc = copy(mats[l][p], mats[l][p]->t->offset + e * m->bytes,
                           m->bytes, slot + m->offset, crc, 1);
                end = m->offset + m->bytes;
            }
            crc = crc_zeros(crc, layers[l].slot_bytes - end);
            sums[l * n_expert + e] = crc;
        }
        printf("\r  layer %llu/%llu, %.1f GB copied", (unsigned long long)l + 1,
               (unsigned long long)n_layer, copied / 1e9);
        fflush(stdout);
    }
    printf("\n");

    /* tables, in file order, and their checksum */
    uint64_t tbl_len =
        tensor_dir + sizeof(struct janas_jns_tensor) * n_res - layer_table;
    uint8_t *tbl = calloc(1, tbl_len);
    if (!tbl)
        die("%s", "out of memory");
    memcpy(tbl, layers, sizeof(*layers) * n_layer);
    memcpy(tbl + (expert_table - layer_table), slot_off,
           8 * n_layer * n_expert);
    memcpy(tbl + (expert_sums - layer_table), sums, 8 * n_layer * n_expert);
    for (size_t i = 0; i < n_res; i++) {
        const struct janas_gguf_tensor *t = resident[i]->t;
        struct janas_jns_tensor d = {0};
        if (t->name.n >= sizeof(d.name)) {
            char *name = str_dup(&t->name);
            die("tensor name too long: %s", name);
        }
        memcpy(d.name, t->name.p, t->name.n);
        d.type = t->type;
        d.n_dims = t->n_dims;
        memcpy(d.dims, t->dims, sizeof(d.dims));
        d.offset = resident[i]->new_offset;
        d.bytes = t->bytes;
        memcpy(tbl + (tensor_dir - layer_table) + i * sizeof(d), &d, sizeof(d));
    }
    uint8_t *meta = malloc(metadata_bytes);
    if (!meta)
        die("%s", "out of memory");
    memcpy(meta, &g0->n_kv, 8);
    memcpy(meta + 8, g0->head + g0->kv_start, metadata_bytes - 8);
    if (pwrite(out_fd, tbl, tbl_len, (off_t)layer_table) != (ssize_t)tbl_len ||
        pwrite(out_fd, meta, metadata_bytes, (off_t)metadata_offset) !=
            (ssize_t)metadata_bytes)
        die("write failed: %s", strerror(errno));

    struct janas_jns_header h = {0};
    memcpy(h.magic, JANAS_JNS_MAGIC, 8);
    h.version = 2;
    h.header_bytes = ALIGN;
    h.file_bytes = file_bytes;
    memcpy(h.arch, arch, strlen(arch) < 31 ? strlen(arch) : 31);
    h.n_layer = (uint32_t)n_layer;
    h.n_expert = (uint32_t)n_expert;
    h.n_expert_used = (uint32_t)n_used;
    h.d_model = (uint32_t)d_model;
    h.layer_table_offset = layer_table;
    h.expert_table_offset = expert_table;
    h.expert_sums_offset = expert_sums;
    h.tensor_dir_offset = tensor_dir;
    h.n_tensor = n_res;
    h.metadata_offset = metadata_offset;
    h.metadata_bytes = metadata_bytes;
    h.resident_offset = resident_offset;
    h.resident_bytes = resident_bytes;
    h.experts_offset = experts_offset;
    h.experts_bytes = experts_bytes;
    h.tables_checksum = janas_fnv1a(tbl, tbl_len, JANAS_FNV_INIT);
    h.metadata_checksum = janas_crc32(0, meta, metadata_bytes);
    h.header_checksum = janas_fnv1a(&h, sizeof(h), JANAS_FNV_INIT);
    if (pwrite(out_fd, &h, sizeof(h), 0) != (ssize_t)sizeof(h) ||
        close(out_fd) != 0)
        die("write failed: %s", strerror(errno));
    printf("written %s: %.2f GB\n", dst, file_bytes / 1e9);

    free(meta);
    free(tbl);
    free(buf);
    free(slot_off);
    free(sums);
    free(resident);
    free(mats);
    free(layers);
    free(arch);
    free(tensors);
    for (int i = 0; i < n_parts; i++) {
        close(parts[i].fd);
        janas_gguf_free(&parts[i].g);
    }
    return 0;
}
