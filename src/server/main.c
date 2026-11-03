/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-server - a model behind an HTTP API that follows OpenAI's, so that
 * the clients written for it (SDKs, chat front ends, editors) can use a
 * model running on this machine. See server.h.
 */
#include "files_store.h"
#include "server.h"
#include "assistants.h"
#include "containers.h"
#include "vectors.h"

#include "common/devices.h"
#include "common/sandbox.h"
#include "common/env_names.h"
#include "common/sock.h"
#include "common/stop.h"
#include "common/sysinfo.h"
#include "common/words.h"

#include <netdb.h>
#include <limits.h>
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
        "an embedding model the embeddings, with a speech model the\n"
        "transcriptions and translations; moderations ask the model.\n"
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
        "                    (without it, the one beside the model that\n"
        "                    fits it: an MTP block or Gemma 4's assistant)\n"
        "  --no-mtp          none, not even one found beside the model\n"
        "  --draft FILE      a small model of the same family, for speed\n"
        "  --mode M          auto, eco or max (default auto)\n"
        "  --quality Q       the lowest quality the engine may choose to be\n"
        "                    faster where memory is short: low (default)\n"
        "                    the experts at four bits and the light ones not\n"
        "                    in memory left out, medium four bits only,\n"
        "                    high nothing lowered\n"
        "  --no-gpu          never give the GPU work\n"
        "  --prompt-on DEV   what reads the prompts: auto (default: what\n"
        "                    measures fastest here), cpu, gpu, npu, npu+gpu\n"
        "                    (both in each pass), or a list apart by commas\n"
        "                    for the engine to choose from (npu,gpu); the\n"
        "                    CPU takes part in every pass, the NPU in blocks\n"
        "                    of 64 tokens and more; a choice wins over the\n"
        "                    mode\n"
        "  --drafts-on DEV   the same for the passes that check guessed\n"
        "                    tokens (2 to 16)\n"
        "  --reply-on DEV    the same for a reply's tokens, one at a time\n"
        "  --npu-int8        the NPU reads prompts with eight-bit\n"
        "                    activations: faster, a little less exact\n"
        "                    (dense Qwen3 models)\n"
        "  --embedding-model FILE  a model for /v1/embeddings (such as\n"
        "                    Qwen3-Embedding), run beside the chat model\n"
        "  --speech-model FILE  a Whisper model (made by hf2jns_whisper)\n"
        "                    for /v1/audio/transcriptions and translations\n"
        "  --vad-model FILE  Silero VAD (made by silero2jns) for their\n"
        "                    chunking_strategy (default: silero-vad.jns\n"
        "                    beside the speech model)\n"
        "  --speaker-model FILE  speaker embeddings (WeSpeaker, made by\n"
        "                    wespeaker2jns) for diarized_json (default:\n"
        "                    wespeaker-resnet34.jns beside the speech model)\n"
        "  --think on|off    reasoning before replying, for models that do\n"
        "                    it, when a request does not say (default: the\n"
        "                    model's own)\n"
        "  --reasoning-budget N  the longest reasoning, in tokens, when a\n"
        "                    request does not say: there the reasoning is\n"
        "                    closed and the model answers (-1: no limit,\n"
        "                    the default; 0: no reasoning)\n"
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
        "  --fim-prefix N    completions with a suffix (an editor filling in\n"
        "                    code): the tokens before the cursor read at\n"
        "                    most, cut at a line and from a start kept\n"
        "                    across keystrokes (default 2048; 0: all)\n"
        "  --fim-suffix N    the lines after the cursor read (default 8;\n"
        "                    0: all)\n"
        "  --fim-max N       the tokens written there at most, whatever the\n"
        "                    request asks (default 256; 0: as asked): an\n"
        "                    editor asks for thousands, and the requests\n"
        "                    of the next keystrokes wait for them\n"
        "  --metrics FILE    JSON lines of what every reply costs: prompt,\n"
        "                    progress of a long one, first token, the end\n"
        "                    (counters and times, never text)\n"
        "  --progress S      a line on stderr every S seconds while a long\n"
        "                    prompt is read (30 with --verbose)\n"
        "  --no-mcp          refuse tools of type mcp in /responses: by\n"
        "                    default the server reaches the MCP servers a\n"
        "                    request names, as OpenAI's does, at public\n"
        "                    addresses only\n"
        "  --tls-cert FILE   serve HTTPS with this certificate (PEM, the\n"
        "                    chain after it), with --tls-key\n"
        "  --tls-key FILE    its private key (PEM, not encrypted)\n"
        "  --no-web          refuse the web_search tool in /responses: by\n"
        "                    default the server searches Wikipedia when a\n"
        "                    request gives the model that tool\n"
        "  --code-interpreter  run the code a model writes (code_interpreter\n"
        "                    in /responses and the Assistants, the\n"
        "                    Containers API) in a cage: no network, no\n"
        "                    other files or processes, bounded memory\n"
        "                    and time (Linux, user namespaces); off by\n"
        "                    default\n"
        "  --code-interpreter-python PATH  the python3 inside the cage (a\n"
        "                    path under /usr; the system's by default)\n"
        "  --mcp-allow HOST[:PORT]  an MCP server at a private or local\n"
        "                    address that requests may name all the same\n"
        "                    (again for more)\n"
        "  --allow-origin ORIGIN  a web page of another origin that may\n"
        "                    call the server, as http://host:port (again\n"
        "                    for more; * for any): others are refused\n"
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

/* every option, for the nearest to one mistyped */
static const char *const OPTIONS[] = {"--help",
                                      "--version",
                                      "--host",
                                      "--port",
                                      "--api-key",
                                      "--name",
                                      "--ctx",
                                      "--reserve",
                                      "--cache",
                                      "--mtp",
                                      "--draft",
                                      "--mode",
                                      "--quality",
                                      "--think",
                                      "--embedding-model",
                                      "--speech-model",
                                      "--vad-model",
                                      "--speaker-model",
                                      "--reasoning-budget",
                                      "--no-mtp",
                                      "--no-gpu",
                                      "--keep",
                                      "--keep-memory",
                                      "--keep-disk",
                                      "--store",
                                      "--no-store",
                                      "--queue",
                                      "--max-body",
                                      "--connections",
                                      "--max-input",
                                      "--warn-input",
                                      "--fim-prefix",
                                      "--fim-suffix",
                                      "--fim-max",
                                      "--metrics",
                                      "--progress",
                                      "--no-mcp",
                                      "--mcp-allow",
                                      "--no-web",
                                      "--code-interpreter",
                                      "--code-interpreter-python",
                                      "--tls-cert",
                                      "--tls-key",
                                      "--allow-origin",
                                      "--test",
                                      "--verbose",
                                      "--prompt-on",
                                      "--drafts-on",
                                      "--reply-on",
                                      "--npu-int8"};

/* the value of option argv[*i], read whole within [lo, hi]; a mistake
   ends the program, as a missing value does */
static long long num(int argc, char **argv, int *i, long long lo, long long hi)
{
    const char *a = argv[*i], *v = arg(argc, argv, i);
    long long x;
    char why[160];
    if (janas_word_int(v, lo, hi, &x, why, sizeof why) != 0) {
        fprintf(stderr, "janas-server: %s: %s\n", a, why);
        exit(2);
    }
    return x;
}

static double real(int argc, char **argv, int *i, double lo, double hi)
{
    const char *a = argv[*i], *v = arg(argc, argv, i);
    double x;
    char why[160];
    if (janas_word_real(v, lo, hi, &x, why, sizeof why) != 0) {
        fprintf(stderr, "janas-server: %s: %s\n", a, why);
        exit(2);
    }
    return x;
}

static int choice(int argc, char **argv, int *i, const char *const *names,
                  size_t n)
{
    const char *a = argv[*i], *v = arg(argc, argv, i);
    char why[300];
    int k = janas_word_choice(v, names, n, why, sizeof why);
    if (k < 0) {
        fprintf(stderr, "janas-server: %s: %s\n", a, why);
        exit(2);
    }
    return k;
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
    janas_sandbox_main(argc, argv); /* a cage of code_interpreter's */
    janas_env_check("janas-server");
    struct srv s = {0};
    struct srv_config *c = &s.cfg;
    janas_llm_params_default(&c->llm);
    c->host = "127.0.0.1";
    c->port = 8080;
    c->max_queue = 16;
    c->keep = 8;
    c->keep_disk = (uint64_t)8 << 30;
    c->fim_prefix = 2048;
    c->fim_suffix = 8;
    c->fim_max = 256;
    int no_store = 0;
    c->max_body = (size_t)32 << 20;
    c->max_connections = 64;
    c->api_key = getenv("JANAS_API_KEY");
    c->thinking = -1;
    c->reasoning_budget = -1;
    c->mcp_policy.public_only = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        char dwhy[240];
        int dk;
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(a, "--version")) {
            printf("janas-server %s\n", JANAS_VERSION);
            return 0;
        } else if (!strcmp(a, "--host")) {
            c->host = arg(argc, argv, &i);
        } else if (!strcmp(a, "--port")) {
            c->port = (uint16_t)num(argc, argv, &i, 1, 65535);
        } else if (!strcmp(a, "--api-key")) {
            c->api_key = arg(argc, argv, &i);
        } else if (!strcmp(a, "--name")) {
            c->model_id = arg(argc, argv, &i);
        } else if (!strcmp(a, "--ctx")) {
            c->llm.n_ctx = (uint32_t)num(argc, argv, &i, 0, UINT32_MAX);
        } else if (!strcmp(a, "--reserve")) {
            c->llm.reserve_bytes =
                (uint64_t)(real(argc, argv, &i, 0, 1 << 20) * (1 << 30));
        } else if (!strcmp(a, "--cache")) {
            c->llm.cache_bytes =
                (uint64_t)(real(argc, argv, &i, 0, 1 << 20) * (1 << 30));
        } else if (!strcmp(a, "--mtp")) {
            c->llm.mtp_path = arg(argc, argv, &i);
        } else if (!strcmp(a, "--draft")) {
            c->llm.draft_path = arg(argc, argv, &i);
        } else if (!strcmp(a, "--quality")) {
            static const char *const levels[] = {"low", "medium", "high"};
            c->llm.min_quality = choice(argc, argv, &i, levels, 3);
        } else if (!strcmp(a, "--mode")) {
            static const char *const modes[] = {"auto", "eco", "max"};
            int m = choice(argc, argv, &i, modes, 3);
            c->llm.mode = m == 1   ? JANAS_LLM_MODE_ECO
                          : m == 2 ? JANAS_LLM_MODE_MAX
                                   : JANAS_LLM_MODE_AUTO;
        } else if (!strcmp(a, "--think")) {
            static const char *const on_off[] = {"off", "on"};
            c->thinking = choice(argc, argv, &i, on_off, 2);
        } else if (!strcmp(a, "--reasoning-budget")) {
            c->reasoning_budget = (int)num(argc, argv, &i, -1, INT32_MAX);
        } else if (!strcmp(a, "--embedding-model")) {
            c->embed_path = arg(argc, argv, &i);
        } else if (!strcmp(a, "--speech-model")) {
            c->speech_path = arg(argc, argv, &i);
        } else if (!strcmp(a, "--vad-model")) {
            c->vad_path = arg(argc, argv, &i);
        } else if (!strcmp(a, "--speaker-model")) {
            c->speaker_path = arg(argc, argv, &i);
        } else if (!strcmp(a, "--no-mtp")) {
            c->no_mtp = 1;
        } else if (!strcmp(a, "--no-gpu")) {
            c->no_gpu = 1;
        } else if (!strcmp(a, "--npu-int8")) {
            setenv("JANAS_NPU_A8", "1", 1); /* read when the model opens */
        } else if (!strcmp(a, "--keep")) {
            c->keep = (int)num(argc, argv, &i, 0, 1 << 20);
        } else if (!strcmp(a, "--keep-memory")) {
            c->keep_bytes =
                (uint64_t)(real(argc, argv, &i, 0, 1 << 20) * (1 << 30));
        } else if (!strcmp(a, "--keep-disk")) {
            c->keep_disk =
                (uint64_t)(real(argc, argv, &i, 0, 1 << 20) * (1 << 30));
        } else if (!strcmp(a, "--store")) {
            c->store_dir = arg(argc, argv, &i);
            no_store = 0;
        } else if (!strcmp(a, "--no-store")) {
            no_store = 1;
        } else if (!strcmp(a, "--queue")) {
            c->max_queue = (uint32_t)num(argc, argv, &i, 0, 1 << 20);
        } else if (!strcmp(a, "--max-body")) {
            c->max_body = (size_t)num(argc, argv, &i, 1, 1 << 14) << 20;
        } else if (!strcmp(a, "--connections")) {
            c->max_connections = (int)num(argc, argv, &i, 1, 1 << 16);
        } else if (!strcmp(a, "--max-input")) {
            c->max_input = (uint32_t)num(argc, argv, &i, 0, UINT32_MAX);
        } else if (!strcmp(a, "--warn-input")) {
            c->warn_input = (uint32_t)num(argc, argv, &i, 0, UINT32_MAX);
        } else if (!strcmp(a, "--fim-prefix")) {
            c->fim_prefix = (uint32_t)num(argc, argv, &i, 0, UINT32_MAX);
        } else if (!strcmp(a, "--fim-suffix")) {
            c->fim_suffix = (uint32_t)num(argc, argv, &i, 0, UINT32_MAX);
        } else if (!strcmp(a, "--fim-max")) {
            c->fim_max = (uint32_t)num(argc, argv, &i, 0, UINT32_MAX);
        } else if (!strcmp(a, "--metrics")) {
            setenv("JANAS_METRICS", arg(argc, argv, &i), 1);
        } else if (!strcmp(a, "--progress")) {
            real(argc, argv, &i, 0, 86400);
            setenv("JANAS_PROGRESS", argv[i], 1);
        } else if (!strcmp(a, "--no-mcp")) {
            c->no_mcp = 1;
        } else if (!strcmp(a, "--no-web")) {
            c->no_web = 1;
        } else if (!strcmp(a, "--code-interpreter")) {
            c->code_interpreter = 1;
        } else if (!strcmp(a, "--code-interpreter-python")) {
            c->code_python = arg(argc, argv, &i);
            c->code_interpreter = 1;
        } else if (!strcmp(a, "--tls-cert")) {
            c->tls_cert = arg(argc, argv, &i);
        } else if (!strcmp(a, "--tls-key")) {
            c->tls_key = arg(argc, argv, &i);
        } else if (!strcmp(a, "--mcp-allow") || !strcmp(a, "--allow-origin")) {
            /* argv lives as long as the server: pointers into it */
            int mcp = a[2] == 'm';
            const char *v = arg(argc, argv, &i);
            size_t n = mcp ? c->mcp_policy.n_allow : (size_t)c->n_allow_origin;
            void *t = realloc(mcp ? (void *)c->mcp_policy.allow
                                  : (void *)c->allow_origin,
                              (n + 1) * sizeof(char *));
            if (!t) {
                fprintf(stderr, "janas-server: out of memory\n");
                return 1;
            }
            if (mcp) {
                c->mcp_policy.allow = t;
                c->mcp_policy.allow[c->mcp_policy.n_allow++] = (char *)v;
            } else {
                c->allow_origin = t;
                c->allow_origin[c->n_allow_origin++] = v;
            }
        } else if (!strcmp(a, "--test")) {
            c->test = 1;
        } else if (!strcmp(a, "--verbose")) {
            c->verbose = 1;
            setenv("JANAS_PROGRESS", "30", 0); /* unless one is asked */
        } else if (a[0] == '-' && (dk = janas_devices_option(
                                       a, i + 1 < argc ? argv[i + 1] : NULL,
                                       dwhy, sizeof dwhy)) != 0) {
            if (dk < 0) {
                fprintf(stderr, "janas-server: %s\n", dwhy);
                return 2;
            }
            i++; /* its value, put in the engine's variable */
        } else if (a[0] == '-') {
            char why[200];
            janas_unknown_word(why, sizeof why, a, "an option", OPTIONS,
                               sizeof OPTIONS / sizeof *OPTIONS);
            fprintf(stderr, "janas-server: %s\n", why);
            fprintf(stderr, "janas-server --help lists the options\n");
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
    char *name = NULL, *embed_name = NULL, *speech_name = NULL;
    if (!c->model_id)
        c->model_id = name = model_name(c->model_path);
    if (c->embed_path)
        c->embed_id = embed_name = model_name(c->embed_path);
    if (c->speech_path)
        c->speech_id = speech_name = model_name(c->speech_path);
    if (!c->model_id || c->max_queue < 1 || c->keep < 0 || c->keep > 1024 ||
        c->max_body < 1024 || c->max_connections < 1) {
        fprintf(stderr, "janas-server: invalid options\n");
        return 2;
    }
    /* the certificate read now: a mistake there should not wait for the
       model */
    if ((c->tls_cert != NULL) != (c->tls_key != NULL)) {
        fprintf(stderr, "janas-server: --tls-cert and --tls-key go "
                        "together\n");
        return 2;
    }
    if (c->tls_cert) {
        char terr[300];
        if (srv_tls_setup(c->tls_cert, c->tls_key, terr, sizeof terr) != 0) {
            fprintf(stderr, "janas-server: %s\n", terr);
            return 2;
        }
    }

    /* the address first: a mistake there should not wait for the model */
    char port[8];
    snprintf(port, sizeof(port), "%u", (unsigned)c->port);
    struct addrinfo hints = {.ai_family = AF_UNSPEC,
                             .ai_socktype = SOCK_STREAM,
                             .ai_flags = AI_PASSIVE | AI_NUMERICSERV},
                    *ai = NULL;
    janas_sock_init();
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

    /* the requests to stop, held back from every thread and awaited here */
    janas_stop_prepare();

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
    if (!(s.files = srv_files_open(c->store_dir, err, sizeof(err)))) {
        fprintf(stderr, "janas-server: %s\n", err);
        freeaddrinfo(ai);
        srv_store_free(s.store);
        return 1;
    }
    /* the speech model before the chat model, which pins this thread to
       one CPU: the speech model's threads would inherit it */
    if (c->speech_path)
        fprintf(stderr, "janas-server: opening %s...\n", c->speech_path);
    if (srv_speech_start(&s, err, sizeof err) != 0) {
        fprintf(stderr, "janas-server: %s\n", err);
        freeaddrinfo(ai);
        srv_files_close(s.files);
        srv_store_free(s.store);
        return 1;
    }
    fprintf(stderr, "janas-server: opening %s...\n", c->model_path);
    s.engine = srv_engine_start(c, err, sizeof(err));
    if (!s.engine) {
        fprintf(stderr, "janas-server: %s\n", err);
        freeaddrinfo(ai);
        srv_speech_stop(&s);
        srv_files_close(s.files);
        srv_store_free(s.store);
        return 1;
    }
    if (srv_containers_start(&s) != 0)
        fprintf(stderr, "janas-server: code_interpreter is off: %s\n",
                srv_containers_off());
    if (srv_vectors_start(&s) != 0)
        fprintf(stderr, "janas-server: vector stores cannot be indexed: no "
                        "directory or thread\n");
    if (srv_batches_start(&s) != 0)
        fprintf(stderr, "janas-server: batches cannot run: no work "
                        "directory or thread\n");
    srv_runs_start(&s);
    if (srv_evals_start(&s) != 0)
        fprintf(stderr, "janas-server: evals cannot run: no work "
                        "directory or thread\n");
    if (srv_http_start(&s, ai->ai_addr, err, sizeof(err)) != 0) {
        fprintf(stderr, "janas-server: %s\n", err);
        freeaddrinfo(ai);
        srv_runs_stop();
        srv_containers_stop();
        srv_evals_stop();
        srv_batches_stop();
        srv_vectors_stop();
        srv_engine_stop(s.engine);
        srv_speech_stop(&s);
        srv_files_close(s.files);
        srv_store_free(s.store);
        return 1;
    }
    freeaddrinfo(ai);
    char cpu[128], feat[512];
    janas_cpu_name(cpu, sizeof cpu);
    janas_cpu_features(feat, sizeof feat);
    fprintf(stderr, "janas-server: CPU %s; instructions: %s\n",
            cpu[0] ? cpu : "unknown", feat[0] ? feat : "not known");
    fprintf(stderr,
            "janas-server: %s\n"
            "janas-server: model '%s' on http%s://%s%s%s:%u/v1%s; Ctrl+C "
            "stops it\n",
            srv_engine_describe(s.engine), c->model_id, c->tls_cert ? "s" : "",
            strchr(c->host, ':') ? "[" : "", c->host,
            strchr(c->host, ':') ? "]" : "", (unsigned)c->port,
            c->api_key ? ", with a key" : "");
    if (c->test)
        fprintf(stderr, "janas-server: test chat on http://%s%s%s:%u/test\n",
                strchr(c->host, ':') ? "[" : "",
                strcmp(c->host, "0.0.0.0") ? c->host : "127.0.0.1",
                strchr(c->host, ':') ? "]" : "", (unsigned)c->port);
    janas_stop_wait();
    fprintf(stderr, "janas-server: stopping\n");
    srv_engine_cancel_clients(s.engine);
    srv_http_stop(&s);
    srv_runs_stop();       /* their passes cut short: again at next start */
    srv_containers_stop(); /* the Python sessions end; the files stay */
    srv_evals_stop(); /* a row being graded finishes; the rest at next start */
    srv_batches_stop(); /* a line running finishes; the rest at next start */
    srv_vectors_stop(); /* a file being indexed: again at next start */
    srv_engine_stop(s.engine);
    srv_speech_stop(&s);
    srv_responses_shutdown();
    srv_uploads_shutdown();
    srv_files_close(s.files);
    srv_store_free(s.store);
    free(name);
    free(embed_name);
    free(speech_name);
    free(c->mcp_policy.allow);
    free(c->allow_origin);
    return 0;
}
