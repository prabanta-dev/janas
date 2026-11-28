/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_compat.c - what Janas cannot do of what a model can (janas_llm_limits
 * in include/janas/llm.h), said rather than worked around in silence: the
 * model's file gives no chat template, or one Janas cannot read; its
 * template writes tool calls, or reasoning, in a form Janas does not read
 * in the model's replies. Where each form Janas reads comes from: the
 * model's chat template itself, which writes the earlier turns' calls and
 * reasoning as the model is to write them (tests/template_check.c checks
 * the prompt against it, token for token).
 */
#include <stdio.h>
#include <string.h>

#include "llm/chat.h"

static void line(struct janas_buf *b, const char *s)
{
    janas_buf_puts(b, s);
    janas_buf_puts(b, "\n");
}

int32_t janas_llm_limits(const janas_llm *llm, char *buf, int32_t cap,
                         int32_t *len)
{
    if (len)
        *len = 0;
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    struct janas_buf b = {0};
    const char *src = llm->tover.src ? llm->tover.src : llm->tfile.src;
    if (!llm->tfile.t && !llm->tover.t) {
        if (llm->tmpl_err[0] &&
            strncmp(llm->tmpl_err, "the model's file has no", 23) != 0) {
            janas_buf_puts(&b, llm->tmpl_err);
            janas_buf_puts(&b, ": its chats are written in Janas's format for "
                               "its family instead\n");
        } else if (llm->format == FORMAT_OTHER) {
            line(&b, "the model's file has no chat template");
        } else {
            line(&b, "the model's file has no chat template: its chats are "
                     "written in Janas's format for its family, not as its "
                     "makers write them");
        }
    }
    if (src) {
        /* the template writes calls, which Janas does not read in replies */
        if (llm->tools == JANAS_TOOLS_NONE &&
            (strstr(src, "tool_calls") || strstr(src, "tool_call")))
            line(&b, "its chat template writes tool calls in a form Janas "
                     "does not read in its replies yet: tools are off for "
                     "it");
        /* the template knows reasoning, whose markers Janas does not */
        if ((llm->think_open < 0 || llm->think_close < 0) &&
            (strstr(src, "reasoning_content") ||
             strstr(src, "enable_thinking") || strstr(src, "thinking")))
            line(&b, "its chat template knows reasoning, whose markers Janas "
                     "does not know for it: its reasoning comes as part of "
                     "the answer");
    }
    const char *rope = llm->m ? janas_llm_model_rope_unread(llm->m) : "";
    if (rope && rope[0]) {
        char t[256];
        snprintf(t, sizeof t,
                 "its file asks for a rope scaling (%s) Janas does not apply "
                 "to this model yet: its context is held to the length it "
                 "was trained for before it",
                 rope);
        line(&b, t);
    }
    if (llm->clean_spaces)
        line(&b, "its tokenizer takes the spaces before punctuation away "
                 "when it decodes (clean_spaces), which Janas does not: its "
                 "replies may keep a space before a full stop or a comma");
    if (b.oom) {
        janas_buf_free(&b);
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    }
    int32_t rc = janas_api_copy_out(b.p ? b.p : "", b.n, buf, cap, len);
    janas_buf_free(&b);
    return rc;
}
