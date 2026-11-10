/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_model_pick.c - the models janas-chat's menu lists (issue #62): the
 * ones here first, then the catalog's to download, each chat model of the
 * catalog once; a prediction file, a test model or a file that is no model
 * is not offered. In a directory of its own under $TMPDIR (else /tmp).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/model_pick.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void touch(const char *dir, const char *name, char *path, size_t len)
{
    snprintf(path, len, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (f) {
        fputs("not a model", f);
        fclose(f);
    }
}

static int chat_models(void)
{
    int k = 0;
    for (int i = 0; i < janas_catalog_n; i++)
        k += !janas_catalog[i].not_chat && !janas_catalog[i].test_only;
    return k;
}

int main(void)
{
    char dir[512];
    const char *tmp = getenv("TMPDIR");
    snprintf(dir, sizeof dir, "%s/janas_test_model_pick_XXXXXX",
             tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(dir)) {
        printf("FAIL: cannot make a directory in %s\n", dir);
        return 1;
    }
    const struct janas_model_entry *q = janas_catalog_find("qwen3-0.6b");
    CHECK(q != NULL, "qwen3-0.6b is not in the catalog");
    if (!q)
        return 1;

    /* nothing here: the whole list to download */
    int n, n_here;
    struct janas_model_item *it = janas_model_items(dir, &n, &n_here);
    CHECK(it && n == chat_models() && n_here == 0,
          "an empty folder: %d models, %d here (%d chat models)", n, n_here,
          chat_models());
    free(it);

    /* a model, a prediction file and a file that is no model */
    const struct janas_model_entry *x = NULL;
    for (int i = 0; i < janas_catalog_n && !x; i++)
        if (janas_catalog[i].extra_out && !janas_catalog[i].not_chat)
            x = &janas_catalog[i];
    char made[3][700];
    touch(dir, q->out, made[0], sizeof made[0]);
    touch(dir, x ? x->extra_out : "none.bin", made[1], sizeof made[1]);
    touch(dir, "other.jns", made[2], sizeof made[2]);
    it = janas_model_items(dir, &n, &n_here);
    CHECK(it && n == chat_models() && n_here == 1,
          "one model here: %d models, %d here", n, n_here);
    if (it && n_here == 1) {
        CHECK(it[0].entry == q && strcmp(it[0].path, made[0]) == 0,
              "the model here first: %s at %s", it[0].name, it[0].path);
        CHECK(it[0].ram_gb == q->ram_gb, "its memory: %d", it[0].ram_gb);
    }
    for (int i = 0; it && i < n; i++) {
        CHECK(it[i].entry || i < n_here, "%s: no entry, not here", it[i].name);
        CHECK(!it[i].entry ||
                  (!it[i].entry->not_chat && !it[i].entry->test_only),
              "%s is not a chat model of the list", it[i].name);
        CHECK(i < n_here ? it[i].path[0] != 0 : it[i].path[0] == 0,
              "%s: path '%s' at %d of %d here", it[i].name, it[i].path, i,
              n_here);
        CHECK(i == 0 || it[i].entry != q, "qwen3-0.6b listed twice");
    }
    free(it);

    for (int i = 0; i < 3; i++)
        unlink(made[i]);
    CHECK(rmdir(dir) == 0, "%s left with files in it", dir);
    if (failures) {
        printf("test_model_pick: %d FAILED\n", failures);
        return 1;
    }
    printf("test_model_pick: all passed\n");
    return 0;
}
