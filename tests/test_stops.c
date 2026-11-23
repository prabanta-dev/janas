/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_stops.c - a real model opened, its chat format and the tokens that
 * end a reply read (api.c, find_format), and closed, within a minute: on
 * 20 Nov 2026 add_stop called itself and every model's open spun for ever,
 * and no other test of the suite opened a model to see it. The model is
 * the catalog's smallest, Qwen3-0.6B, in the models' folder (JANAS_MODELS);
 * without it the test says so and passes (GitHub's runners have none).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/catalog.h"
#include "janas/llm.h"

int main(void)
{
#ifndef _WIN32
    alarm(60); /* an open that never comes back fails here, not for ever */
#endif
    char dir[4096], path[4200];
    if (janas_models_dir(dir, sizeof dir) != 0) {
        printf("test_stops: no models' folder, skipped\n");
        return 0;
    }
    snprintf(path, sizeof path, "%s/qwen3-0.6b-q4km.jns", dir);
    if (access(path, R_OK) != 0) {
        printf("test_stops: %s not found, skipped\n", path);
        return 0;
    }
    janas_llm *llm = NULL;
    if (janas_llm_open(path, NULL, &llm) != JANAS_LLM_OK || !llm) {
        printf("FAIL: %s did not open: %s\n", path, janas_llm_last_error());
        return 1;
    }
    janas_llm_close(llm);
    printf("test_stops: ok\n");
    return 0;
}
