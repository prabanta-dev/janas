/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_reason_room.c - reasoning with no budget closed while there is room
 * for the answer (api_reply.c, answer_room): on 23 Nov 2026 a model that
 * reasoned in circles filled the context, and the request ended with "the
 * context is full" and no answer. A real model reasons here in a context
 * too small for it - the catalog's smallest, Qwen3-0.6B, which reasons, in
 * the models' folder (JANAS_MODELS); without it the test says so and
 * passes (GitHub's runners have none).
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
    alarm(120); /* a reply that never ends fails here, not for ever */
#endif
    char dir[4096], path[4200];
    if (janas_models_dir(dir, sizeof dir) != 0) {
        printf("test_reason_room: no models' folder, skipped\n");
        return 0;
    }
    snprintf(path, sizeof path, "%s/qwen3-0.6b-q4km.jns", dir);
    if (access(path, R_OK) != 0) {
        printf("test_reason_room: %s not found, skipped\n", path);
        return 0;
    }
    struct janas_llm_params mp;
    janas_llm_params_default(&mp);
    mp.n_ctx = 2048; /* far too little for the reasoning asked below */
    mp.mtp_path = NULL;
    janas_llm *llm = NULL;
    if (janas_llm_open(path, &mp, &llm) != JANAS_LLM_OK || !llm) {
        printf("FAIL: %s did not open: %s\n", path, janas_llm_last_error());
        return 1;
    }
    struct janas_llm_chat_params cp;
    janas_llm_chat_params_default(&cp);
    cp.temperature = 0;
    cp.thinking = 1;
    cp.max_answer = 48;
    janas_llm_chat *c = NULL;
    int fail = 0;
    if (janas_llm_chat_create(llm, &cp, &c) != JANAS_LLM_OK) {
        printf("FAIL: no chat: %s\n", janas_llm_last_error());
        janas_llm_close(llm);
        return 1;
    }
    static const char ask[] =
        "Think carefully about every prime number below 3000, one by one, "
        "checking each divisor, then tell me how many there are.";
    int32_t rc = janas_llm_chat_send(c, ask, -1);
    size_t answer = 0, reasoning = 0;
    char buf[512];
    while (rc == JANAS_LLM_OK) {
        int32_t len = 0;
        rc = janas_llm_chat_next(c, buf, sizeof buf, &len);
        if (rc == JANAS_LLM_OK && len > 0) {
            if (janas_llm_chat_thinking(c))
                reasoning += (size_t)len;
            else
                answer += (size_t)len;
        }
    }
    struct janas_llm_chat_stats st = {.size = sizeof st};
    janas_llm_chat_stats(c, &st);
    if (rc != JANAS_LLM_DONE) {
        printf("FAIL: the reply ended with %d (%s), not with its answer\n",
               (int)rc, janas_llm_last_error());
        fail = 1;
    }
    if (!st.reasoning_cut) {
        printf("FAIL: the reasoning was not closed for the answer's room "
               "(%zu bytes of it, finish %d)\n",
               reasoning, (int)st.finish);
        fail = 1;
    }
    if (answer == 0) {
        printf("FAIL: no answer after %zu bytes of reasoning\n", reasoning);
        fail = 1;
    }
    if (st.finish == JANAS_LLM_FINISH_CONTEXT ||
        st.finish == JANAS_LLM_FINISH_ERROR) {
        printf("FAIL: finish %d: the context filled anyway\n", (int)st.finish);
        fail = 1;
    }
    janas_llm_chat_destroy(c);
    janas_llm_close(llm);
    if (!fail)
        printf("test_reason_room: ok (%zu bytes reasoned, %zu answered)\n",
               reasoning, answer);
    return fail;
}
