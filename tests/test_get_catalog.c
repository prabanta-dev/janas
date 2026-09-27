/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_get_catalog.c - janas-get's catalog against MODELS.md: every
 * SHA-256 in it 64 hex digits and written in MODELS.md, every name unique.
 * A digest copied by hand one digit short would otherwise turn good
 * downloads away (it happened while writing the catalog).
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/get/catalog.c"

static int failures;

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *p = malloc((size_t)n + 1);
    if (p && fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p);
        p = NULL;
    }
    if (p)
        p[n] = 0;
    fclose(f);
    return p;
}

static void digest(const char *models, const char *who, const char *d)
{
    if (!d)
        return;
    int hex = strlen(d) == 64;
    for (const char *c = d; hex && *c; c++)
        hex = isxdigit((unsigned char)*c) && !isupper((unsigned char)*c);
    if (!hex) {
        printf("FAIL %s: %s is not 64 hex digits\n", who, d);
        failures++;
    } else if (!strstr(models, d)) {
        printf("FAIL %s: %s is not in MODELS.md\n", who, d);
        failures++;
    }
}

int main(void)
{
    char *models = slurp("MODELS.md");
    if (!models) {
        printf("test_get_catalog: MODELS.md not found (run from the "
               "repository's root)\n");
        return 1;
    }
    for (int i = 0; i < get_catalog_n; i++) {
        const struct get_model *m = &get_catalog[i];
        for (int p = 0; p < GET_MAX_PARTS && m->parts[p]; p++)
            digest(models, m->name, m->part_sha[p]);
        digest(models, m->name, m->flat_sha);
        digest(models, m->name, m->jns_sha);
        digest(models, m->name, m->extra_src_sha);
        digest(models, m->name, m->extra_sha);
        if (!m->flat_sha || (m->extra != EXTRA_NONE && !m->extra_sha)) {
            printf("FAIL %s: a fingerprint missing\n", m->name);
            failures++;
        }
        for (int k = 0; k < i; k++)
            if (strcmp(get_catalog[k].name, m->name) == 0) {
                printf("FAIL %s: twice in the catalog\n", m->name);
                failures++;
            }
    }
    free(models);
    printf("test_get_catalog: %s (%d models)\n", failures ? "FAILED" : "ok",
           get_catalog_n);
    return failures ? 1 : 0;
}
