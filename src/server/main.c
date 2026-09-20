/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-server - a model behind an HTTP API that follows OpenAI's, so that
 * the clients written for it (SDKs, chat front ends, editors) can use a
 * model running on this machine. See server.h.
 */
#include "server.h"

#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *f)
{
    fprintf(
        f,
        "usage: janas-server <model.jns> [options]\n"
        "\n"
        "Serves the model through an HTTP API that follows OpenAI's: models,\n"
        "chat completions (tools, JSON schemas, log-probabilities, stored\n"
        "completions), completions, responses and conversations, and with\n"
        "an embedding model the embeddings; moderations ask the model.\n"
        "Every other operation of OpenAI's API answers 501 and says why.\n"
        "\n"
        "  --host ADDR       address to listen on (default 127.0.0.1: this\n"
        "                    machine only; 0.0.0.0 opens it to the network)\n"
        "  --port N          default 8080\n"
        "  --api-key KEY     ask clients for this key (also JANAS_API_KEY)\n"
        "  --name ID         the model's name for clients (default: the\n"
        "                    file's name)\n"
        "  --ctx N           context in tokens (default 16384)\n"
        "  --cache GIB       memory for the experts (default: automatic)\n"
        "  --reserve GIB     memory left to other programs when the cache\n"
        "                    is automatic (default: a fifth of the machine's)\n"
        "  --mtp FILE        the model's prediction block, for speed\n"
        "  --draft FILE      a small model of the same family, for speed\n"
        "  --mode M          auto, eco or max (default auto)\n"
        "  --no-gpu          never give the GPU work\n"
        "  --embedding-model FILE  a model for /v1/embeddings (such as\n"
        "                    Qwen3-Embedding), run beside the chat model\n"
        "  --think on|off    reasoning before replying, for models that do\n"
        "                    it, when a request does not say (default: the\n"
        "                    model's own)\n"
        "  --keep N          conversations kept computed besides the one\n"
        "                    in use, so that a client coming back is not\n"
        "                    read again (default 8; 0: none)\n"
        "  --keep-memory GIB memory for them (default: half of what the\n"
        "                    expert cache and a margin for other programs\n"
        "                    leave free at start, at least 256 MiB)\n"
        "  --store DIR       where stored completions, responses and\n"
        "                    conversations are kept, so that they outlive\n"
        "                    the server (default ~/.local/share/janas/server)\n"
        "  --no-store        keep them in memory only, until it stops\n"
        "  --keep-disk GIB   disk for them below the memory, so that they\n"
        "                    outlive the server: a long system message and\n"
        "                    tools are read once (default 8; 0: none)\n"
        "  --queue N         requests allowed to wait (default 16)\n"
        "  --max-body MIB    largest request body (default 32)\n"
        "  --connections N   open connections at most (default 64)\n"
        "  --max-input N     the longest input a request may have, in tokens,\n"
        "                    below the context: longer ones get a 400 that\n"
        "                    says by how much (nothing is ever cut)\n"
        "  --warn-input N    longer inputs are taken, and said on stderr\n"
        "  --metrics FILE    JSON lines of what every reply costs: prompt,\n"
        "                    progress of a long one, first token, the end\n"
        "                    (counters and times, never text)\n"
        "  --progress S      a line on stderr every S seconds while a long\n"
        "                    prompt is read (30 with --verbose)\n"
        "  --no-mcp          refuse tools of type mcp in /responses: by\n"
        "                    default the server reaches the MCP servers a\n"
        "                    request names, as OpenAI's does\n"
        "  --test            serve a chat page for trying the server out, at\n"
        "                    http://<host>:<port>/test\n"
        "  --verbose         one line per request on stderr\n"
        "  --version, --help\n");
}

static const char *arg(int argc, char **argv, int *i)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "janas-server: %s wants a value\n", argv[*i]);
        exit(2);
    }
    return argv[++*i];
}

/* Where the store goes when --store does not say: $XDG_DATA_HOME/janas/
   server, or ~/.local/share/janas/server. 0, or -1 with no home. */
static int default_store(char *buf, size_t len)
{
    const char *x = getenv("XDG_DATA_HOME"), *h = getenv("HOME");
    int n = x && *x   ? snprintf(buf, len, "%s/janas/server", x)
            : h && *h ? snprintf(buf, len, "%s/.local/share/janas/server", h)
                      : -1;
    return n > 0 && (size_t)n < len ? 0 : -1;
}

/* The file's name without its directory and its extension. */
static char *model_name(const char *path)
{
    const char *b = strrchr(path, '/');
    b = b ? b + 1 : path;
    char *n = strdup(b);
    char *dot = n ? strrchr(n, '.') : NULL;
    if (dot && dot != n)
        *dot = 0;
    return n;
}

int main(int argc, char **argv)
{
    struct srv s = {0};
    struct srv_config *c = &s.cfg;
    janas_llm_params_default(&c->llm);
    c->host = "127.0.0.1";
    c->port = 8080;
    c->max_queue = 16;
    c->keep = 8;
    c->keep_disk = (uint64_t)8 << 30;
    int no_store = 0;
    c->max_body = (size_t)32 << 20;
    c->max_connections = 64;
    c->api_key = getenv("JANAS_API_KEY");
    c->thinking = -1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(a, "--version")) {
            printf("janas-server %s\n", JANAS_VERSION);
            return 0;
        } else if (!strcmp(a, "--host")) {
            c->host = arg(argc, argv, &i);
        } else if (!strcmp(a, "--port")) {
            c->port = (uint16_t)atoi(arg(argc, argv, &i));
        } else if (!strcmp(a, "--api-key")) {
            c->api_key = arg(argc, argv, &i);
        } else if (!strcmp(a, "--name")) {
            c->model_id = arg(argc, argv, &i);
        } else if (!strcmp(a, "--ctx")) {
            c->llm.n_ctx = (uint32_t)atoi(arg(argc, argv, &i));
        } else if (!strcmp(a, "--reserve")) {
            c->llm.reserve_bytes =
                (uint64_t)(atof(arg(argc, argv, &i)) * (1 << 30));
        } else if (!strcmp(a, "--cache")) {
            c->llm.cache_bytes =
                (uint64_t)(atof(arg(argc, argv, &i)) * (1 << 30));
        } else if (!strcmp(a, "--mtp")) {
            c->llm.mtp_path = arg(argc, argv, &i);
        } else if (!strcmp(a, "--draft")) {
            c->llm.draft_path = arg(argc, argv, &i);
        } else if (!strcmp(a, "--mode")) {
            const char *m = arg(argc, argv, &i);
            c->llm.mode = !strcmp(m, "eco")   ? JANAS_LLM_MODE_ECO
                          : !strcmp(m, "max") ? JANAS_LLM_MODE_MAX
                                              : JANAS_LLM_MODE_AUTO;
        } else if (!strcmp(a, "--think")) {
            const char *v = arg(argc, argv, &i);
            if (strcmp(v, "on") && strcmp(v, "off")) {
                fprintf(stderr, "janas-server: --think is on or off\n");
                return 2;
            }
            c->thinking = strcmp(v, "off") ? 1 : 0;
        } else if (!strcmp(a, "--embedding-model")) {
            c->embed_path = arg(argc, argv, &i);
        } else if (!strcmp(a, "--no-gpu")) {
            c->no_gpu = 1;
        } else if (!strcmp(a, "--keep")) {
            c->keep = atoi(arg(argc, argv, &i));
        } else if (!strcmp(a, "--keep-memory")) {
            c->keep_bytes = (uint64_t)(atof(arg(argc, argv, &i)) * (1 << 30));
        } else if (!strcmp(a, "--keep-disk")) {
            c->keep_disk = (uint64_t)(atof(arg(argc, argv, &i)) * (1 << 30));
        } else if (!strcmp(a, "--store")) {
            c->store_dir = arg(argc, argv, &i);
            no_store = 0;
        } else if (!strcmp(a, "--no-store")) {
            no_store = 1;
        } else if (!strcmp(a, "--queue")) {
            c->max_queue = (uint32_t)atoi(arg(argc, argv, &i));
        } else if (!strcmp(a, "--max-body")) {
            c->max_body = (size_t)atoi(arg(argc, argv, &i)) << 20;
        } else if (!strcmp(a, "--connections")) {
            c->max_connections = atoi(arg(argc, argv, &i));
        } else if (!strcmp(a, "--max-input")) {
            c->max_input = (uint32_t)atol(arg(argc, argv, &i));
        } else if (!strcmp(a, "--warn-input")) {
            c->warn_input = (uint32_t)atol(arg(argc, argv, &i));
        } else if (!strcmp(a, "--metrics")) {
            setenv("JANAS_METRICS", arg(argc, argv, &i), 1);
        } else if (!strcmp(a, "--progress")) {
            setenv("JANAS_PROGRESS", arg(argc, argv, &i), 1);
        } else if (!strcmp(a, "--no-mcp")) {
            c->no_mcp = 1;
        } else if (!strcmp(a, "--test")) {
            c->test = 1;
        } else if (!strcmp(a, "--verbose")) {
            c->verbose = 1;
            setenv("JANAS_PROGRESS", "30", 0); /* unless one is asked */
        } else if (a[0] == '-') {
            fprintf(stderr, "janas-server: unknown option %s\n", a);
            usage(stderr);
            return 2;
        } else if (!c->model_path) {
            c->model_path = a;
        } else {
            fprintf(stderr, "janas-server: one model only\n");
            return 2;
        }
    }
    if (!c->model_path) {
        usage(stderr);
        return 2;
    }
    if (c->api_key && !*c->api_key)
        c->api_key = NULL;
    char *name = NULL, *embed_name = NULL;
    if (!c->model_id)
        c->model_id = name = model_name(c->model_path);
    if (c->embed_path)
        c->embed_id = embed_name = model_name(c->embed_path);
    if (!c->model_id || c->max_queue < 1 || c->keep < 0 || c->keep > 1024 ||
        c->max_body < 1024 || c->max_connections < 1) {
        fprintf(stderr, "janas-server: invalid options\n");
        return 2;
    }

    /* the address first: a mistake there should not wait for the model */
    char port[8];
    snprintf(port, sizeof(port), "%u", (unsigned)c->port);
    struct addrinfo hints = {.ai_family = AF_UNSPEC,
                             .ai_socktype = SOCK_STREAM,
                             .ai_flags = AI_PASSIVE | AI_NUMERICSERV},
                    *ai = NULL;
    int gai = getaddrinfo(c->host, port, &hints, &ai);
    if (gai != 0) {
        fprintf(stderr, "janas-server: %s: %s\n", c->host, gai_strerror(gai));
        return 1;
    }
    int local = !strcmp(c->host, "127.0.0.1") || !strcmp(c->host, "::1") ||
                !strcmp(c->host, "localhost");
    if (!local && !c->api_key)
        fprintf(stderr,
                "janas-server: warning: listening on %s with no API key: "
                "anyone who can reach this machine can use the model. "
                "Set one with --api-key.\n",
                c->host);

    /* the signals that stop it, blocked in every thread and awaited here */
    sigset_t stop;
    sigemptyset(&stop);
    sigaddset(&stop, SIGINT);
    sigaddset(&stop, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &stop, NULL);
    signal(SIGPIPE, SIG_IGN);

    /* the store too: a directory another server holds should not wait
       for the model either */
    char err[512], dir[4096];
    if (no_store)
        c->store_dir = NULL;
    else if (!c->store_dir && default_store(dir, sizeof(dir)) == 0)
        c->store_dir = dir;
    if (!(s.store = srv_store_open(0, c->store_dir, err, sizeof(err)))) {
        fprintf(stderr, "janas-server: %s\n", err);
        freeaddrinfo(ai);
        return 1;
    }
    if (c->store_dir) {
        int failed = srv_responses_recover(s.store);
        fprintf(stderr, "janas-server: %zu stored documents in %s",
                srv_store_count(s.store), c->store_dir);
        if (failed)
            fprintf(stderr,
                    " (%d responses left unfinished by a server "
                    "that did not stop: marked failed)",
                    failed);
        fputc('\n', stderr);
    }
    fprintf(stderr, "janas-server: opening %s...\n", c->model_path);
    s.engine = srv_engine_start(c, err, sizeof(err));
    if (!s.engine) {
        fprintf(stderr, "janas-server: %s\n", err);
        freeaddrinfo(ai);
        srv_store_free(s.store);
        return 1;
    }
    if (srv_http_start(&s, ai->ai_addr, err, sizeof(err)) != 0) {
        fprintf(stderr, "janas-server: %s\n", err);
        freeaddrinfo(ai);
        srv_engine_stop(s.engine);
        srv_store_free(s.store);
        return 1;
    }
    freeaddrinfo(ai);
    fprintf(stderr,
            "janas-server: %s\n"
            "janas-server: model '%s' on http://%s%s%s:%u/v1%s; Ctrl+C "
            "stops it\n",
            srv_engine_describe(s.engine), c->model_id,
            strchr(c->host, ':') ? "[" : "", c->host,
            strchr(c->host, ':') ? "]" : "", (unsigned)c->port,
            c->api_key ? ", with a key" : "");
    if (c->test)
        fprintf(stderr, "janas-server: test chat on http://%s%s%s:%u/test\n",
                strchr(c->host, ':') ? "[" : "",
                strcmp(c->host, "0.0.0.0") ? c->host : "127.0.0.1",
                strchr(c->host, ':') ? "]" : "", (unsigned)c->port);
    int sig;
    sigwait(&stop, &sig);
    fprintf(stderr, "janas-server: stopping\n");
    srv_http_stop(&s);
    srv_engine_stop(s.engine);
    srv_responses_shutdown();
    srv_store_free(s.store);
    free(name);
    free(embed_name);
    return 0;
}
