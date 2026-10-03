/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-mcp - Janas as an MCP server: the model running on this machine
 * offered as tools to the clients of the Model Context Protocol (Claude
 * Code, editors, other agents), over stdio. The client starts it, writes
 * requests on its standard input and reads the answers on its standard
 * output, one JSON-RPC message per line; what it has to say otherwise goes
 * to its standard error.
 *
 * The protocol, both its eras, is common/mcp_server.c's.
 *
 * Tools: generate (a question to the model) and embed (the vector of a
 * text), each where a model for it was given; a notifications/cancelled
 * stops a generation.
 */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/env_names.h"
#include "common/words.h"
#include "serve.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static const char INSTRUCTIONS[] =
    "A language model running on the user's own machine, without a GPU "
    "needed: slower than a hosted one, and private. Give it work that is "
    "self-contained - a text to summarise, translate or rewrite, a question "
    "it can answer from what the prompt says - with everything it needs in "
    "the prompt.";

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-mcp [<model.jns>] [options]\n"
            "An MCP server over stdio: a client (Claude Code, an editor)\n"
            "starts it and calls its tools. generate asks the model given;\n"
            "embed needs an embedding model.\n"
            "  --embedding-model <file>  a model that gives embeddings\n"
            "                   (Qwen3-Embedding): the tool embed\n"
            "  --mtp <file>     the model's multi-token prediction block\n"
            "  --ctx <tokens>   context length (default 16384)\n"
            "  --cache <GiB>    RAM for streamed experts (default: automatic)\n"
            "  --reserve <GiB>  memory left to other programs\n"
            "  --temp <t>       default temperature (0.7)\n"
            "  --max <tokens>   default longest answer (2048)\n"
            "  --think <on|off> reasoning before answering, for models that\n"
            "                   do it (default off; the reasoning is never\n"
            "                   returned, only the answer)\n"
            "  --timeout <s>    longest a generation may run (default 600;\n"
            "                   0: no limit)\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"janas\": {\"command\": \"janas-mcp\",\n"
            "      \"args\": [\"/path/to/qwen3-4b.jns\"]}}}\n");
}

static void list(void *ctx, struct janas_buf *b)
{
    mcpd_tools_list(ctx, b);
}

static int call(void *ctx, const struct janas_json *params, struct janas_buf *b)
{
    return mcpd_tools_call(ctx, params, b);
}

static janas_llm *open_model(const char *path, const struct janas_llm_params *p,
                             char *name, size_t cap)
{
    janas_llm *m = NULL;
    fprintf(stderr, "janas-mcp: loading %s ...\n", path);
    if (janas_llm_open(path, p, &m) != JANAS_LLM_OK) {
        fprintf(stderr, "janas-mcp: %s\n", janas_llm_last_error());
        exit(1);
    }
    int32_t len;
    janas_llm_name(m, name, (int32_t)cap, &len);
    return m;
}

int main(int argc, char **argv)
{
    janas_env_check("janas-mcp");
    struct mcpd d = {.call_seconds = 600};
    struct janas_llm_params mp;
    janas_llm_params_default(&mp);
    janas_llm_chat_params_default(&d.cp);
    d.cp.thinking = 0;
    d.cp.max_reply = 2048;
    const char *model = NULL, *emb = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (a[0] != '-') {
            model = a;
            continue;
        }
        static const char *const options[] = {
            "--embedding-model", "--mtp",  "--ctx", "--cache",
            "--reserve",         "--temp", "--max", "--think",
            "--timeout",         "--help"};
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage();
            return 0;
        }
        char why[300] = "";
        long long x = 0;
        double r = 0;
        int known = 0;
        for (size_t k = 0; k + 1 < sizeof options / sizeof *options; k++)
            known |= strcmp(a, options[k]) == 0;
        if (!known)
            janas_unknown_word(why, sizeof why, a, "an option", options,
                               sizeof options / sizeof *options);
        else if (!v)
            snprintf(why, sizeof why, "%s wants a value", a);
        if (why[0]) {
            fprintf(stderr, "janas-mcp: %s\n", why);
            return 2;
        }
        i++;
        if (strcmp(a, "--embedding-model") == 0)
            emb = v;
        else if (strcmp(a, "--mtp") == 0)
            mp.mtp_path = v;
        else if (strcmp(a, "--ctx") == 0) {
            if (janas_word_int(v, 0, UINT32_MAX, &x, why, sizeof why) == 0)
                mp.n_ctx = (uint32_t)x;
        } else if (strcmp(a, "--cache") == 0) {
            if (janas_word_real(v, 0, 1 << 20, &r, why, sizeof why) == 0)
                mp.cache_bytes = (uint64_t)(r * (1 << 30));
        } else if (strcmp(a, "--reserve") == 0) {
            if (janas_word_real(v, 0, 1 << 20, &r, why, sizeof why) == 0)
                mp.reserve_bytes = (uint64_t)(r * (1 << 30));
        } else if (strcmp(a, "--temp") == 0) {
            if (janas_word_real(v, 0, 100, &r, why, sizeof why) == 0)
                d.cp.temperature = (float)r;
        } else if (strcmp(a, "--max") == 0) {
            if (janas_word_int(v, 0, INT32_MAX, &x, why, sizeof why) == 0)
                d.cp.max_reply = (int32_t)x;
        } else if (strcmp(a, "--think") == 0) {
            static const char *const on_off[] = {"off", "on"};
            int k = janas_word_choice(v, on_off, 2, why, sizeof why);
            if (k >= 0)
                d.cp.thinking = k;
        } else if (strcmp(a, "--timeout") == 0) {
            if (janas_word_int(v, 0, 86400 * 7, &x, why, sizeof why) == 0)
                d.call_seconds = (int)x;
        }
        if (why[0]) {
            fprintf(stderr, "janas-mcp: %s: %s\n", a, why);
            return 2;
        }
    }
    if (!model && !emb) {
        usage();
        return 2;
    }
    if (janas_llm_abi_version() != JANAS_LLM_ABI_VERSION) {
        fprintf(stderr, "janas-mcp: library ABI %d, expected %d\n",
                janas_llm_abi_version(), JANAS_LLM_ABI_VERSION);
        return 1;
    }
    d.srv = (struct janas_mcps){.name = "janas-mcp",
                                .version = JANAS_VERSION,
                                .instructions = INSTRUCTIONS,
                                .ctx = &d,
                                .tools_list = list,
                                .tools_call = call};
    if (janas_mcps_open(&d.srv) != 0) {
        fprintf(stderr, "janas-mcp: cannot set its output aside\n");
        return 1;
    }
    if (model) {
        d.llm = open_model(model, &mp, d.name, sizeof(d.name));
        if (janas_llm_chat_create(d.llm, &d.cp, &d.chat) != JANAS_LLM_OK) {
            fprintf(stderr, "janas-mcp: %s\n", janas_llm_last_error());
            return 1;
        }
        /* a client asks with the same system message many times */
        janas_llm_chat_keep(d.chat, 2, 0);
    }
    if (emb) {
        struct janas_llm_params ep;
        janas_llm_params_default(&ep);
        d.emb = open_model(emb, &ep, d.emb_name, sizeof(d.emb_name));
        if (janas_llm_embed_dim(d.emb) <= 0) {
            fprintf(stderr, "janas-mcp: %s gives no embeddings\n", emb);
            return 1;
        }
    }
    fprintf(stderr, "janas-mcp: ready, tools:%s%s\n", d.llm ? " generate" : "",
            d.emb ? " embed" : "");
    janas_mcps_run(&d.srv);
    fprintf(stderr, "janas-mcp: the client has gone\n");
    if (d.chat)
        janas_llm_chat_destroy(d.chat);
    if (d.llm)
        janas_llm_close(d.llm);
    if (d.emb)
        janas_llm_close(d.emb);
    return 0;
}
