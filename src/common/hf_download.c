/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf_download.c - whole files of a Hugging Face repository, for janas-get
 * (see hf_fetch.h): the size and SHA-256 Hugging Face lists for a file,
 * and a download in pieces that resumes and hashes as it goes.
 *
 * Pieces rather than one long transfer: each is a request of its own on
 * the address the redirect gave, so a broken connection costs one piece,
 * and the signed address of the CDN, if it expires in a long download, is
 * asked for again.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common/hf_fetch.h"
#include "common/sha256.h"

#define PIECE ((int64_t)128 << 20)
#define TRIES 6

int janas_hf_info(const char *repo, const char *rev, const char *path,
                  uint64_t *size, char sha[65], char *err, size_t err_len)
{
    /* the listing of the file's directory */
    char url[1536], dir[1024] = "";
    const char *slash = strrchr(path, '/');
    if (slash)
        snprintf(dir, sizeof(dir), "/%.*s", (int)(slash - path), path);
    int k = snprintf(url, sizeof(url),
                     "https://huggingface.co/api/models/%s/tree/%s%s", repo,
                     rev ? rev : "main", dir);
    if (k < 0 || (size_t)k >= sizeof(url)) {
        snprintf(err, err_len, "%s: a name too long", path);
        return -1;
    }
    struct janas_buf b = {0};
    if (janas_hf_get_url(url, NULL, NULL, &b, NULL, 0, err, err_len) != 0) {
        janas_buf_free(&b);
        return -1;
    }
    struct janas_json_doc *d = janas_json_parse(b.p, b.n, err, err_len);
    janas_buf_free(&b);
    if (!d)
        return -1;
    int found = -1;
    const struct janas_json *root = janas_json_root(d);
    for (const struct janas_json *e =
             root && root->type == JANAS_JSON_ARRAY ? root->child : NULL;
         e; e = e->next) {
        if (!janas_json_is(janas_json_get(e, "path"), path))
            continue;
        *size = (uint64_t)janas_json_num(janas_json_get(e, "size"), 0);
        const char *oid =
            janas_json_str(janas_json_get(janas_json_get(e, "lfs"), "oid"));
        snprintf(sha, 65, "%s", oid && strlen(oid) == 64 ? oid : "");
        found = 0;
        break;
    }
    janas_json_free(d);
    if (found != 0)
        snprintf(err, err_len, "%s: not in %s", path, repo);
    return found;
}

/* The address the file answers at once redirected */
static int resolve(const char *repo, const char *rev, const char *path,
                   char *final, size_t len, char *err, size_t err_len)
{
    char url[1536];
    int k =
        snprintf(url, sizeof(url), "https://huggingface.co/%s/resolve/%s/%s",
                 repo, rev ? rev : "main", path);
    if (k < 0 || (size_t)k >= sizeof(url)) {
        snprintf(err, err_len, "%s: a name too long", path);
        return -1;
    }
    struct janas_buf b = {0};
    int64_t a = 0, z = 1;
    int r = janas_hf_get_url(url, &a, &z, &b, final, len, err, err_len);
    janas_buf_free(&b);
    return r;
}

int janas_hf_download(const char *repo, const char *rev, const char *path,
                      const char *out_path, uint64_t size, char sha[65],
                      void (*progress)(uint64_t, uint64_t, void *), void *arg,
                      char *err, size_t err_len)
{
    int fd = open(out_path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        snprintf(err, err_len, "cannot write %s", out_path);
        return -1;
    }
    struct stat st;
    uint64_t done = fstat(fd, &st) == 0 ? (uint64_t)st.st_size : 0;
    if (done > size) { /* not this file: start again */
        if (ftruncate(fd, 0) != 0) {
            close(fd);
            snprintf(err, err_len, "cannot truncate %s", out_path);
            return -1;
        }
        done = 0;
    }
    struct janas_sha256 h;
    janas_sha256_init(&h);
    struct janas_buf b = {0};
    /* what is already there: hashed as it is, then continued */
    if (done) {
        size_t cap = (size_t)8 << 20;
        char *buf = malloc(cap);
        uint64_t at = 0;
        while (buf && at < done) {
            size_t want = done - at < cap ? (size_t)(done - at) : cap;
            ssize_t r = pread(fd, buf, want, (off_t)at);
            if (r <= 0)
                break;
            janas_sha256_update(&h, buf, (size_t)r);
            at += (uint64_t)r;
        }
        free(buf);
        if (at != done) {
            close(fd);
            snprintf(err, err_len, "cannot read %s", out_path);
            return -1;
        }
        if (progress)
            progress(done, size, arg);
    }
    char *final = malloc(8192);
    int ok = final != NULL, tries = 0;
    if (final)
        final[0] = 0;
    while (ok && done < size) {
        if (!final[0] &&
            resolve(repo, rev, path, final, 8192, err, err_len) != 0) {
            if (++tries >= TRIES)
                ok = 0;
            continue;
        }
        int64_t a = (int64_t)done, z = (int64_t)(size - done) < PIECE
                                           ? (int64_t)size
                                           : (int64_t)done + PIECE;
        b.n = 0;
        if (janas_hf_get_url(final, &a, &z, &b, NULL, 0, err, err_len) != 0 ||
            b.oom || (int64_t)b.n != z - a) {
            final[0] = 0; /* ask for the address again */
            if (++tries >= TRIES)
                ok = 0;
            continue;
        }
        tries = 0;
        size_t w = 0;
        while (w < b.n) {
            ssize_t r = pwrite(fd, b.p + w, b.n - w, (off_t)(done + w));
            if (r <= 0) {
                snprintf(err, err_len, "cannot write %s (disk full?)",
                         out_path);
                ok = 0;
                break;
            }
            w += (size_t)r;
        }
        if (!ok)
            break;
        janas_sha256_update(&h, b.p, b.n);
        done += b.n;
        if (progress)
            progress(done, size, arg);
    }
    free(final);
    janas_buf_free(&b);
    if (close(fd) != 0 && ok) {
        snprintf(err, err_len, "cannot write %s", out_path);
        ok = 0;
    }
    if (!ok)
        return -1;
    janas_sha256_hex(&h, sha);
    return 0;
}
