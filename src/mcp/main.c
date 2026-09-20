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
 * Both eras of the protocol are served, as the specification allows a
 * server to: a request that carries its revision in _meta (2026-07-28) is
 * answered as it asks, statelessly, and a client that opens with
 * initialize gets the handshake of the revision it names (2025-11-25 and
 * the three before it), which most clients still speak.
 *
 * Tools: generate (a question to the model) and embed (the vector of a
 * text), each where a model for it was given. A request that comes while a
 * call runs waits for it, but a ping is answered at once and a
 * notifications/cancelled stops the call.
 */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "serve.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

#define MODERN "2026-07-28"
#define CACHE_MS "3600000" /* the tools do not change while it runs */
static const char *const LEGACY[] = {"2025-11-25", "2025-06-18", "2025-03-26",
                                     "2024-11-05", NULL};

static const char INSTRUCTIONS[] =
    "A language model running on the user's own machine, without a GPU "
    "needed: slower than a hosted one, and private. Give it work that is "
    "self-contained - a text to summarise, translate or rewrite, a question "
    "it can answer from what the prompt says - with everything it needs in "
    "the prompt.";

static int initialized; /* a legacy client has shaken hands */
static char legacy_version[32];

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

static void send_buf(struct mcpd *d, struct janas_buf *b)
{
    if (!b->oom) {
        fwrite(b->p, 1, b->n, d->out);
        fflush(d->out);
    }
    janas_buf_free(b);
}

void mcpd_result(struct mcpd *d, const struct janas_json *id, int modern,
                 const char *members, size_t n)
{
    struct janas_buf b = {0};
    janas_buf_puts(&b, "{\"jsonrpc\":\"2.0\",\"id\":");
    janas_json_write(&b, id);
    janas_buf_puts(&b, ",\"result\":{");
    if (modern) {
        janas_buf_puts(&b, "\"resultType\":\"complete\",\"_meta\":{\"io."
                           "modelcontextprotocol/serverInfo\":{\"name\":"
                           "\"janas-mcp\",\"version\":");
        janas_json_write_str(&b, JANAS_VERSION, strlen(JANAS_VERSION));
        janas_buf_puts(&b, n ? "}}," : "}}");
    }
    janas_buf_put(&b, members, n);
    janas_buf_puts(&b, "}}\n");
    send_buf(d, &b);
}

void mcpd_error(struct mcpd *d, const struct janas_json *id, int code,
                const char *message, const char *data)
{
    struct janas_buf b = {0};
    janas_buf_puts(&b, "{\"jsonrpc\":\"2.0\",\"id\":");
    if (id)
        janas_json_write(&b, id);
    else
        janas_buf_puts(&b, "null");
    janas_buf_printf(&b, ",\"error\":{\"code\":%d,\"message\":", code);
    janas_json_write_str(&b, message, strlen(message));
    if (data)
        janas_buf_printf(&b, ",\"data\":%s", data);
    janas_buf_puts(&b, "}}\n");
    send_buf(d, &b);
}

/* Two ids the same: of the same type, and written the same. */
static int same_id(const struct janas_json *a, const struct janas_json *b)
{
    if (!a || !b || a->type != b->type)
        return 0;
    if (a->type == JANAS_JSON_NUMBER)
        return janas_json_num(a, 0) == janas_json_num(b, 1);
    return a->n == b->n && memcmp(a->s, b->s, a->n) == 0;
}

int mcpd_poll(struct mcpd *d)
{
    const char *line;
    size_t n;
    while (!d->cancelled && janas_mcp_proc_line(&d->in, 0, &line, &n) == 1) {
        struct janas_rpc_msg m;
        char err[128];
        if (janas_rpc_parse(line, n, &m, err, sizeof(err)) == 0) {
            if (m.kind == JANAS_RPC_NOTIFY &&
                strcmp(m.method, "notifications/cancelled") == 0 &&
                same_id(janas_json_get(m.params, "requestId"), d->running)) {
                d->cancelled = 1;
            } else if (m.kind == JANAS_RPC_REQUEST &&
                       strcmp(m.method, "ping") == 0) {
                mcpd_result(d, m.id, janas_json_get(m.params, "_meta") != NULL,
                            "", 0);
            } else if (m.kind == JANAS_RPC_REQUEST) { /* after the call */
                char **g = realloc(d->queue, (d->n_queue + 1) * sizeof(*g));
                char *copy = malloc(n + 1);
                if (g)
                    d->queue = g;
                if (g && copy) {
                    memcpy(copy, line, n);
                    copy[n] = 0;
                    d->queue[d->n_queue++] = copy;
                } else {
                    free(copy);
                    mcpd_error(d, m.id, -32603, "out of memory", NULL);
                }
            }
        }
        janas_rpc_free(&m);
    }
    return d->cancelled;
}

static void discover(struct mcpd *d, const struct janas_json *id)
{
    struct janas_buf b = {0};
    janas_buf_puts(&b, "\"supportedVersions\":[\"" MODERN "\"");
    for (int i = 0; LEGACY[i]; i++)
        janas_buf_printf(&b, ",\"%s\"", LEGACY[i]);
    /* the caching hints every complete discover and list result carries
       from 2026-07-28 on: the same for everybody, and fixed while this
       process runs */
    janas_buf_puts(&b, "],\"ttlMs\":" CACHE_MS ",\"cacheScope\":\"public\","
                       "\"capabilities\":{\"tools\":{}},\"instructions\":");
    janas_json_write_str(&b, INSTRUCTIONS, strlen(INSTRUCTIONS));
    if (!b.oom)
        mcpd_result(d, id, 1, b.p, b.n);
    janas_buf_free(&b);
}

static void initialize(struct mcpd *d, const struct janas_rpc_msg *m)
{
    const char *want =
        janas_json_str(janas_json_get(m->params, "protocolVersion"));
    const char *v = LEGACY[0];
    for (int i = 0; want && LEGACY[i]; i++)
        if (strcmp(want, LEGACY[i]) == 0)
            v = LEGACY[i];
    snprintf(legacy_version, sizeof(legacy_version), "%s", v);
    initialized = 1;
    struct janas_buf b = {0};
    janas_buf_printf(&b,
                     "\"protocolVersion\":\"%s\",\"capabilities\":{\"tools\":"
                     "{}},\"serverInfo\":{\"name\":\"janas-mcp\",\"version\":",
                     v);
    janas_json_write_str(&b, JANAS_VERSION, strlen(JANAS_VERSION));
    janas_buf_puts(&b, "},\"instructions\":");
    janas_json_write_str(&b, INSTRUCTIONS, strlen(INSTRUCTIONS));
    if (!b.oom)
        mcpd_result(d, m->id, 0, b.p, b.n);
    janas_buf_free(&b);
    fprintf(stderr, "janas-mcp: a client of revision %s\n", v);
}

static void request(struct mcpd *d, const struct janas_rpc_msg *m)
{
    const struct janas_json *meta = janas_json_get(m->params, "_meta");
    const char *version = janas_json_str(
        janas_json_get(meta, "io.modelcontextprotocol/protocolVersion"));
    int modern = version != NULL;
    if (strcmp(m->method, "ping") == 0) {
        mcpd_result(d, m->id, modern, "", 0);
        return;
    }
    if (strcmp(m->method, "initialize") == 0) {
        initialize(d, m);
        return;
    }
    if (version && strcmp(version, MODERN) != 0) {
        struct janas_buf data = {0};
        janas_buf_puts(&data, "{\"supported\":[\"" MODERN "\"],\"requested\":");
        janas_json_write_str(&data, version,
                             strlen(version) < 64 ? strlen(version) : 64);
        janas_buf_puts(&data, "}");
        janas_buf_put(&data, "", 1);
        mcpd_error(d, m->id, -32022, "Unsupported protocol version",
                   data.oom ? NULL : data.p);
        janas_buf_free(&data);
        return;
    }
    if (!version && !initialized) {
        mcpd_error(d, m->id, -32602,
                   "no protocol version: put io.modelcontextprotocol/"
                   "protocolVersion in _meta (" MODERN
                   "), or send initialize first",
                   NULL);
        return;
    }
    if (strcmp(m->method, "server/discover") == 0) {
        discover(d, m->id);
    } else if (strcmp(m->method, "tools/list") == 0) {
        struct janas_buf b = {0};
        mcpd_tools_list(d, &b);
        if (modern)
            janas_buf_puts(&b, ",\"ttlMs\":" CACHE_MS ",\"cacheScope\":"
                               "\"public\"");
        if (!b.oom)
            mcpd_result(d, m->id, modern, b.p, b.n);
        janas_buf_free(&b);
    } else if (strcmp(m->method, "tools/call") == 0) {
        struct janas_buf b = {0};
        d->running = m->id;
        d->cancelled = 0;
        int rc = mcpd_tools_call(d, m->params, &b);
        d->running = NULL;
        if (rc < 0) {
            const char *t = janas_json_str(janas_json_get(m->params, "name"));
            char msg[200];
            snprintf(msg, sizeof(msg), "Unknown tool: %.100s", t ? t : "");
            mcpd_error(d, m->id, -32602, msg, NULL);
        } else if (rc == 0 && !b.oom) {
            mcpd_result(d, m->id, modern, b.p, b.n);
        } else if (rc == 0) {
            mcpd_error(d, m->id, -32603, "out of memory", NULL);
        } /* cancelled: nothing more is said about it */
        janas_buf_free(&b);
    } else {
        mcpd_error(d, m->id, -32601, "Method not found", NULL);
    }
}

static void take(struct mcpd *d, const char *line, size_t n)
{
    struct janas_rpc_msg m;
    char err[128];
    if (janas_rpc_parse(line, n, &m, err, sizeof(err)) != 0) {
        if (m.doc || n)
            mcpd_error(d, NULL, m.doc ? -32600 : -32700,
                       m.doc ? "Invalid Request" : "Parse error", NULL);
    } else if (m.kind == JANAS_RPC_REQUEST) {
        request(d, &m);
    }
    /* notifications (initialized, a cancel that came too late) and answers
       to requests never sent need nothing */
    janas_rpc_free(&m);
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
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0 || !v) {
            usage();
            return strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0 ? 0 : 2;
        }
        i++;
        if (strcmp(a, "--embedding-model") == 0)
            emb = v;
        else if (strcmp(a, "--mtp") == 0)
            mp.mtp_path = v;
        else if (strcmp(a, "--ctx") == 0)
            mp.n_ctx = (uint32_t)atol(v);
        else if (strcmp(a, "--cache") == 0)
            mp.cache_bytes = (uint64_t)(atof(v) * (1 << 30));
        else if (strcmp(a, "--reserve") == 0)
            mp.reserve_bytes = (uint64_t)(atof(v) * (1 << 30));
        else if (strcmp(a, "--temp") == 0)
            d.cp.temperature = (float)atof(v);
        else if (strcmp(a, "--max") == 0)
            d.cp.max_reply = atoi(v);
        else if (strcmp(a, "--think") == 0)
            d.cp.thinking = strcmp(v, "on") == 0;
        else if (strcmp(a, "--timeout") == 0)
            d.call_seconds = atoi(v);
        else {
            usage();
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
    /* standard output carries the protocol and nothing else: whatever else
       would be written there goes to standard error */
    int fd = dup(STDOUT_FILENO);
    d.out = fd >= 0 ? fdopen(fd, "w") : NULL;
    if (!d.out || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
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
    d.in = (struct janas_mcp_proc){.in = -1, .out = STDIN_FILENO};
    for (;;) {
        if (d.n_queue) { /* what came while a call ran, in its order */
            char *line = d.queue[0];
            memmove(d.queue, d.queue + 1, --d.n_queue * sizeof(*d.queue));
            take(&d, line, strlen(line));
            free(line);
            continue;
        }
        const char *line;
        size_t n;
        int r = janas_mcp_proc_line(&d.in, 60000, &line, &n);
        if (r < 0)
            break; /* the client closed our input: the end */
        if (r > 0)
            take(&d, line, n);
    }
    fprintf(stderr, "janas-mcp: the client has gone\n");
    if (d.chat)
        janas_llm_chat_destroy(d.chat);
    if (d.llm)
        janas_llm_close(d.llm);
    if (d.emb)
        janas_llm_close(d.emb);
    free(d.in.buf);
    free(d.queue);
    fclose(d.out);
    return 0;
}
