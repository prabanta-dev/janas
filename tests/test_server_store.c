/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_server_store.c - the server's store on disk (src/server/store.c,
 * store_disk.c), in a directory of its own under $TMPDIR (else /tmp),
 * removed at the end:
 *   1. documents put, changed and deleted come back after a reopening, in
 *      their order, with their second text, and the deleted ones do not;
 *   2. an id that is not a file name stays in memory and is not written;
 *   3. a second store cannot open the same directory while the first holds
 *      it;
 *   4. an unfinished write is removed and an unreadable file is left out;
 *   5. reopened with a smaller cap, the oldest go, from the disk too.
 */
#define _GNU_SOURCE
#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "server/store.c"
#include "server/store_disk.c"

static int failed;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_server_store: " __VA_ARGS__);                \
            fputc('\n', stderr);                                               \
            failed = 1;                                                        \
        }                                                                      \
    } while (0)

static int rm_one(const char *path, const struct stat *st, int flag,
                  struct FTW *ftw)
{
    (void)st;
    (void)flag;
    (void)ftw;
    return remove(path);
}

static int exists(const char *dir, const char *rel)
{
    char p[8192];
    snprintf(p, sizeof(p), "%s/%s", dir, rel);
    return access(p, F_OK) == 0;
}

/* the document's texts are these (aux NULL: none) */
static void expect(struct srv_store *st, const char *kind, const char *id,
                   const char *json, const char *aux)
{
    char *j = NULL, *a = NULL;
    int rc = srv_store_get(st, kind, id, &j, &a);
    CHECK(rc == 0, "%s %s: not found (%d)", kind, id, rc);
    if (rc == 0) {
        CHECK(!strcmp(j, json), "%s %s: json '%s'", kind, id, j);
        CHECK(aux ? a && !strcmp(a, aux) : !a, "%s %s: aux '%s'", kind, id,
              a ? a : "(none)");
    }
    free(j);
    free(a);
}

int main(void)
{
    const char *t = getenv("TMPDIR");
    char dir[4096];
    snprintf(dir, sizeof(dir), "%s/janas-store-test-%d", t && *t ? t : "/tmp",
             (int)getpid());
    char err[512];

    /* 1, 2, 3 */
    struct srv_store *st = srv_store_open(0, dir, err, sizeof(err));
    CHECK(st, "open: %s", err);
    if (!st)
        return 1;
    srv_store_put(st, "chat.completion", "chatcmpl-1", 100, "{\"a\":1}",
                  "[\"m\"]");
    srv_store_put(st, "response", "resp_1", 101, "{\"b\":2}", NULL);
    srv_store_put(st, "chat.completion", "chatcmpl-2", 102, "{\"c\":3}", NULL);
    srv_store_put(st, "response", "../escape", 103, "{\"x\":0}", NULL);
    srv_store_update(st, "response", "resp_1", "{\"b\":22}", "items");
    srv_store_del(st, "chat.completion", "chatcmpl-2");
    CHECK(srv_store_count(st) == 3, "count %zu before reopening",
          srv_store_count(st));
    CHECK(!exists(dir, "escape") && !exists(dir, "../escape"),
          "an id with a path was written");
    struct srv_store *other = srv_store_open(0, dir, err, sizeof(err));
    CHECK(!other && strstr(err, "another janas-server"),
          "a second store opened the same directory");
    srv_store_free(other);
    srv_store_free(st);

    st = srv_store_open(0, dir, err, sizeof(err));
    CHECK(st, "reopen: %s", err);
    if (!st)
        return 1;
    CHECK(srv_store_count(st) == 2, "count %zu after reopening",
          srv_store_count(st));
    expect(st, "chat.completion", "chatcmpl-1", "{\"a\":1}", "[\"m\"]");
    expect(st, "response", "resp_1", "{\"b\":22}", "items");
    CHECK(srv_store_get(st, "chat.completion", "chatcmpl-2", NULL, NULL) == -1,
          "a deleted document came back");
    /* the order: a new one goes after those read back */
    srv_store_put(st, "chat.completion", "chatcmpl-3", 104, "{\"d\":4}", NULL);
    char **ids;
    size_t n = srv_store_list(st, "chat.completion", &ids);
    CHECK(n == 2 && !strcmp(ids[0], "chatcmpl-1") &&
              !strcmp(ids[1], "chatcmpl-3"),
          "order after reopening");
    srv_store_list_free(ids, n);
    srv_store_free(st);

    /* 4 */
    char p[4096 + 64];
    snprintf(p, sizeof(p), "%s/response/.resp_9.tmp", dir);
    FILE *f = fopen(p, "w");
    if (f) {
        fputs("JANAS-STORE 1 9", f);
        fclose(f);
    }
    snprintf(p, sizeof(p), "%s/response/resp_bad", dir);
    f = fopen(p, "w");
    if (f) {
        fputs("JANAS-STORE 1 10 105 50 -1\n{}", f); /* shorter than said */
        fclose(f);
    }
    st = srv_store_open(0, dir, err, sizeof(err));
    CHECK(st, "reopen with bad files: %s", err);
    if (!st)
        return 1;
    CHECK(!exists(dir, "response/.resp_9.tmp"), "an unfinished write was left");
    CHECK(srv_store_count(st) == 3, "count %zu with bad files",
          srv_store_count(st));
    srv_store_free(st);

    /* 5 */
    st = srv_store_open(1, dir, err, sizeof(err));
    CHECK(st, "reopen with cap 1: %s", err);
    if (!st)
        return 1;
    CHECK(srv_store_get(st, "chat.completion", "chatcmpl-1", NULL, NULL) ==
                  -1 &&
              !exists(dir, "chat.completion/chatcmpl-1"),
          "the oldest completion stayed past the cap");
    expect(st, "chat.completion", "chatcmpl-3", "{\"d\":4}", NULL);
    expect(st, "response", "resp_1", "{\"b\":22}", "items");
    srv_store_free(st);

    nftw(dir, rm_one, 16, FTW_DEPTH | FTW_PHYS);
    if (!failed)
        printf("test_server_store: ok\n");
    return failed;
}
