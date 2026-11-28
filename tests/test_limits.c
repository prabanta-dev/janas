/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_limits.c - what janas_llm_limits says of a model (api_compat.c): a
 * model of no template, of a template whose tool calls or reasoning Janas
 * does not read, and of one it reads whole, made by hand.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"

static int fails;

static void check(const char *what, janas_llm *llm, const char *want)
{
    char b[2048];
    int32_t n = 0;
    if (janas_llm_limits(llm, b, sizeof b, &n) != JANAS_LLM_OK) {
        printf("FAIL %s: %s\n", what, janas_llm_last_error());
        fails++;
        return;
    }
    int ok = want ? strstr(b, want) != NULL : n == 0;
    if (!ok) {
        printf("FAIL %s: [%s], wanted %s\n", what, b, want ? want : "none");
        fails++;
    }
}

int main(void)
{
    janas_llm m = {0};
    m.think_open = m.think_close = -1;
    m.tools = JANAS_TOOLS_NONE;
    snprintf(m.tmpl_err, sizeof m.tmpl_err,
             "the model's file has no chat template");
    check("no template", &m, "no chat template");
    m.tmpl_err[0] = 0;
    /* a template (its text alone matters here; t marks it as read) */
    m.tfile.t = (struct janas_jinja *)&m;
    m.tfile.src = (char *)"{% for m in messages %}{{ m.content }}{% endfor %}";
    check("plain template", &m, NULL);
    m.tfile.src = (char *)"{% if m.tool_calls %}{{ m.tool_calls|tojson }}"
                          "{% endif %}";
    check("tools not read", &m, "tool calls");
    m.tools = JANAS_TOOLS_JSON;
    check("tools read", &m, NULL);
    m.tfile.src = (char *)"{% if enable_thinking %}think{% endif %}";
    check("reasoning not read", &m, "reasoning");
    m.think_open = 1, m.think_close = 2;
    check("reasoning read", &m, NULL);
    m.clean_spaces = 1;
    check("spaces not cleaned", &m, "clean_spaces");
    if (!fails)
        printf("test_limits: ok\n");
    return fails != 0;
}
