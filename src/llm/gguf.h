/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gguf.h - the head of a GGUF file (llama.cpp's format): its metadata and
 * its tensor directory, for the converter to JNS (tools/gguf2jns.c).
 *
 * A GGUF file comes from anywhere on the internet, so the reader takes
 * nothing on trust: every length and count is checked against the bytes
 * there are, with overflow-checked arithmetic, and the data of every tensor
 * against the size of the file. tests/test_gguf.c mutates heads to hold it
 * to that.
 */
#ifndef JANAS_LLM_GGUF_H
#define JANAS_LLM_GGUF_H

#include <stddef.h>
#include <stdint.h>

/* A GGUF string: its bytes (not NUL-terminated) and their count. */
struct janas_gguf_str {
    const uint8_t *p;
    uint64_t n;
};

struct janas_gguf_kv {
    struct janas_gguf_str key;
    uint32_t type;      /* GGUF value type: 0-12 */
    const uint8_t *val; /* its encoding, in the head */
};

struct janas_gguf_tensor {
    struct janas_gguf_str name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;   /* ggml type */
    uint64_t offset; /* from the start of the data */
    uint64_t bytes;
};

struct janas_gguf {
    uint8_t *head; /* the first bytes of the file, owned */
    size_t n_head;
    uint32_t version;
    uint64_t n_kv, n_tensors;
    size_t kv_start, kv_end; /* the key-value pairs, raw, in head */
    struct janas_gguf_kv *kv;
    struct janas_gguf_tensor *t;
    uint64_t data_start; /* where the tensors' data begins in the file */
    uint64_t file_bytes;
};

/*
 * Reads a head of n bytes from a file of file_bytes. 0: the head is taken
 * over, freed by janas_gguf_free. -1: it is invalid, the reason in err, and
 * the head is freed. 1: the head is cut short - it stays the caller's, to
 * read more of the file into and try again.
 */
int janas_gguf_parse(uint8_t *head, size_t n, uint64_t file_bytes,
                     struct janas_gguf *g, char *err, size_t err_len);
/* The same from a file, reading as much of its head as it takes. */
int janas_gguf_open(const char *path, struct janas_gguf *g, char *err,
                    size_t err_len);
void janas_gguf_free(struct janas_gguf *g);

/* A value of the metadata: an integer of any width (or a bool), 0 or -1
   when absent or of another type; a string, the same. */
int janas_gguf_uint(const struct janas_gguf *g, const char *key, uint64_t *out);
int janas_gguf_string(const struct janas_gguf *g, const char *key,
                      struct janas_gguf_str *out);
/* The tensor of this name, NULL when there is none. */
const struct janas_gguf_tensor *janas_gguf_tensor(const struct janas_gguf *g,
                                                  const char *name);

/* Weights per block and bytes per block of a ggml type (0: unknown). */
int janas_gguf_type(uint32_t type, uint64_t *blck, uint64_t *size);

#endif
