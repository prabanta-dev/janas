/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jns.h - the Janas-LLM model file (JNS).
 *
 * The engine never reads GGUF: an offline converter (tools/gguf2jns.c)
 * rewrites a model into this layout, built for streaming experts with
 * O_DIRECT. All integers are little-endian; every region starts on a
 * JANAS_JNS_ALIGN boundary and has a length that is a multiple of it.
 *
 *   offset 0      header, JANAS_JNS_ALIGN bytes (struct janas_jns_header)
 *   layer table   n_layer entries (struct janas_jns_layer)
 *   expert table  n_layer * n_expert u64 absolute offsets, layer-major; the
 *                 order of the experts on disk is free (e.g. co-activation)
 *   expert sums   n_layer * n_expert u64 checksums, one per expert slot: the
 *                 CRC-32 of zlib (IEEE 802.3) of the slot's bytes
 *   tensor dir    n_tensor entries (struct janas_jns_tensor): everything
 *                 that is not a routed expert, read once at load time
 *   metadata      the key/value section of the source GGUF, byte for byte:
 *                 u64 count, then entries (u64 key length, key, u32 type,
 *                 value); hyperparameters and tokenizer
 *   resident      the bytes of those tensors
 *   experts       one slot per (layer, expert): gate, up and down back to
 *                 back, each matrix starting on a 256-byte boundary, the slot
 *                 padded to JANAS_JNS_ALIGN
 *
 * Version 3 adds the levels of an expert slot. Where the down matrix is kept
 * in bit planes (JANAS_Q6_K_P), the slot holds gate, up and the base plane
 * first, then the second plane, then the third, each group padded to
 * JANAS_JNS_ALIGN: reading only the first level_bytes[0] of the slot gives
 * two bits per weight of the down matrix, level_bytes[1] four, the whole slot
 * the six of Q6_K, bit for bit. A machine short of memory reads the prefix it
 * can afford, out of the same file. Version 2 files are read as if their
 * three levels were all the whole slot.
 *
 * Sizes are stored, never derived from products of fields read from the
 * file; the reader checks every offset and length against the file size with
 * overflow-checked arithmetic before using it.
 */
#ifndef JANAS_LLM_JNS_H
#define JANAS_LLM_JNS_H

#include <stddef.h>
#include <stdint.h>

#define JANAS_JNS_MAGIC "JANASLLM"
#define JANAS_JNS_VERSION 3
#define JANAS_JNS_ALIGN 4096
#define JANAS_JNS_MAX_LAYERS 1024
#define JANAS_JNS_MAX_EXPERTS 4096
#define JANAS_JNS_MAX_TENSORS 65536

enum { JANAS_JNS_GATE = 0, JANAS_JNS_UP = 1, JANAS_JNS_DOWN = 2 };

struct janas_jns_header {
    char magic[8];         /* JANAS_JNS_MAGIC, not NUL-terminated */
    uint32_t version;      /* JANAS_JNS_VERSION */
    uint32_t header_bytes; /* JANAS_JNS_ALIGN */
    uint64_t file_bytes;
    char arch[32]; /* GGUF general.architecture, NUL-padded */
    uint32_t n_layer;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t d_model;
    uint64_t layer_table_offset;
    uint64_t expert_table_offset;
    uint64_t expert_sums_offset;
    uint64_t tensor_dir_offset;
    uint64_t n_tensor;
    uint64_t metadata_offset;
    uint64_t metadata_bytes;
    uint64_t resident_offset;
    uint64_t resident_bytes;
    uint64_t experts_offset;
    uint64_t experts_bytes;
    uint64_t tables_checksum;   /* FNV-1a of layer, expert and sum tables and
                                   the tensor directory, in file order */
    uint64_t metadata_checksum; /* CRC-32 of the metadata region */
    uint64_t reserved;
    uint64_t header_checksum; /* FNV-1a of this header with this field 0 */
};

/* One matrix of an expert slot. */
struct janas_jns_matrix {
    uint32_t type; /* enum janas_qtype / ggml_type */
    uint32_t rows;
    uint32_t cols;
    uint32_t reserved;
    uint64_t offset; /* from the start of the slot */
    uint64_t bytes;
};

struct janas_jns_layer {
    uint64_t slot_bytes;     /* bytes of one expert slot, multiple of ALIGN */
    uint64_t level_bytes[3]; /* prefix of the slot each level needs, growing,
                                the last equal to slot_bytes; all three equal
                                when the down matrix has no planes */
    struct janas_jns_matrix m[3];     /* gate, up, down (its base plane) */
    struct janas_jns_matrix plane[2]; /* the other two planes of down, type 0
                                         when there are none */
};

/* A layer as version 2 wrote it, still read from those files. */
struct janas_jns_layer_v2 {
    uint64_t slot_bytes;
    uint64_t reserved;
    struct janas_jns_matrix m[3];
};

struct janas_jns_tensor {
    char name[64]; /* NUL-terminated */
    uint32_t type;
    uint32_t n_dims;
    uint64_t dims[4]; /* ggml order: dims[0] is the row length */
    uint64_t offset;  /* absolute */
    uint64_t bytes;
    uint64_t reserved;
};

_Static_assert(sizeof(struct janas_jns_header) == 192, "jns header");
_Static_assert(sizeof(struct janas_jns_layer) == 192, "jns layer");
_Static_assert(sizeof(struct janas_jns_layer_v2) == 112, "jns layer v2");
_Static_assert(sizeof(struct janas_jns_tensor) == 128, "jns tensor");

struct janas_jns {
    int fd; /* opened O_RDONLY; experts are read with O_DIRECT by the cache */
    struct janas_jns_header h;
    struct janas_jns_layer *layers;
    uint64_t *expert_offset; /* n_layer * n_expert */
    uint64_t *expert_sum;    /* n_layer * n_expert */
    struct janas_jns_tensor *tensors;
    uint8_t *metadata; /* metadata_bytes, validated entry by entry */
    /* layers whose levels were written as zero (hf2jns_mtp before issue #6)
       and are read as the whole slot: for jns_check to say */
    uint32_t zero_levels;
};

uint64_t janas_fnv1a(const void *data, size_t n, uint64_t h);
#define JANAS_FNV_INIT 0xcbf29ce484222325ull

/* CRC-32 as zlib's crc32(): start with crc = 0, chain calls to extend. */
uint32_t janas_crc32(uint32_t crc, const void *data, size_t n);

/*
 * Parses and validates the header and tables of a JNS image already in
 * memory (the first `bytes` bytes of the file, file_bytes being the size of
 * the whole file). Returns 0 or a negative error, with a message in err.
 * Exposed separately from janas_jns_open so the parser can be fuzzed.
 */
int janas_jns_parse(const uint8_t *image, size_t bytes, uint64_t file_bytes,
                    struct janas_jns *out, char *err, size_t err_len);

/* Opens and validates a file. Returns 0 or a negative error. */
int janas_jns_open(const char *path, struct janas_jns *out, char *err,
                   size_t err_len);
void janas_jns_close(struct janas_jns *j);

/*
 * Metadata lookups. Return 0 when the key exists with a compatible scalar
 * type (any integer type for _u32, f32 or f64 for _f32), -1 otherwise.
 * _str copies at most len - 1 bytes and NUL-terminates.
 */
int janas_jns_meta_u32(const struct janas_jns *j, const char *key,
                       uint32_t *out);
int janas_jns_meta_f32(const struct janas_jns *j, const char *key, float *out);
int janas_jns_meta_str(const struct janas_jns *j, const char *key, char *out,
                       size_t len);

/*
 * Metadata arrays. _strings: *count strings, *data at the first, each a u64
 * length then its bytes (validated at open). _ints: *count 32-bit integers
 * (i32 or u32), possibly unaligned: read with memcpy. Return 0 or -1.
 */
int janas_jns_meta_strings(const struct janas_jns *j, const char *key,
                           uint64_t *count, const uint8_t **data);
int janas_jns_meta_ints(const struct janas_jns *j, const char *key,
                        uint64_t *count, const int32_t **data);
/* An array of bytes: bool, u8 or i8 (sliding_window_pattern). */
int janas_jns_meta_bytes(const struct janas_jns *j, const char *key,
                         uint64_t *count, const uint8_t **data);

/* The tensor of the resident region with this name, or NULL. */
const struct janas_jns_tensor *janas_jns_tensor(const struct janas_jns *j,
                                                const char *name);

/* Absolute file offset of expert e of layer l. */
static inline uint64_t janas_jns_expert_offset(const struct janas_jns *j,
                                               uint32_t l, uint32_t e)
{
    return j->expert_offset[(size_t)l * j->h.n_expert + e];
}

#endif
