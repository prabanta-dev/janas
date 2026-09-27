/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf_fetch.h - parts of a Hugging Face repository, over HTTPS: a file, a
 * range of bytes of one, or the tensors of a safetensors checkpoint whose
 * names start with a prefix, gathered into one local safetensors file. For
 * the MTP block of a model, a few hundred MB spread over shards of several
 * GB each.
 *
 * Redirects are followed (the files are served from a CDN); a range the
 * server does not honour is an error, never the whole file. HF_TOKEN, when
 * set, is sent to huggingface.co only.
 */
#ifndef JANAS_HF_FETCH_H
#define JANAS_HF_FETCH_H

#include <stdint.h>

#include "llm/json.h"

/* The bytes [from, to) of file path of repo at rev (NULL: all of it) into
   out. 0, or -1 with the reason in err. */
int janas_hf_get(const char *repo, const char *rev, const char *path,
                 const int64_t *from, const int64_t *to, struct janas_buf *out,
                 char *err, size_t err_len);

/* The same by address (a file of the CDN, as a redirect gave it): with
   final, the address the request ended at after the redirects. */
int janas_hf_get_url(const char *url, const int64_t *from, const int64_t *to,
                     struct janas_buf *out, char *final, size_t final_len,
                     char *err, size_t err_len);

/* A file of repo at rev (NULL: main): its size and, for a file kept by
   LFS (weights), the SHA-256 Hugging Face gives for it (sha empty
   otherwise). 0, or -1 with the reason in err. */
int janas_hf_info(const char *repo, const char *rev, const char *path,
                  uint64_t *size, char sha[65], char *err, size_t err_len);

/* Downloads file path of repo at rev to out_path, resuming a partial file
   (its bytes hashed first), in pieces, each tried again if the connection
   breaks; the SHA-256 of the whole written to sha. size: the file's, from
   janas_hf_info. progress(done, size, arg) after each piece (may be NULL).
   0, or -1 with the reason in err. */
int janas_hf_download(const char *repo, const char *rev, const char *path,
                      const char *out_path, uint64_t size, char sha[65],
                      void (*progress)(uint64_t done, uint64_t size, void *arg),
                      void *arg, char *err, size_t err_len);

/* The bf16 tensors of repo's checkpoint (one model.safetensors, or shards
   listed by model.safetensors.index.json) whose names start with prefix,
   written to out_path as one safetensors file. The number of tensors, or
   -1 with the reason in err. Progress goes to stdout. */
int janas_hf_fetch_tensors(const char *repo, const char *rev,
                           const char *prefix, const char *out_path, char *err,
                           size_t err_len);

#endif
