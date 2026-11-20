/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-docs (docs.h): an MCP server over stdio with its index
 * kept up to date in the background, or, with --index or --search, the
 * same work done once from the command line.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include "common/paths.h"
#include "common/mcp_server.h"
#include "docs.h"
#include "services/common/conf.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

/* INSTRUCTIONS, from SKILL.md beside this file (build.sh) */
#include "skill.h"

static void usage(void)
{
    fprintf(
        stderr,
        "usage: janas-docs [--server URL [--key KEY] | --embedding-model "
        "FILE]\n"
        "                  [--ocr-model FILE]\n"
        "                  [--index | --search TEXT]\n"
        "       janas-docs --text FILE\n"
        "An MCP server over stdio: a client starts it and calls its tools,\n"
        "which only read - docs_search, the passages of the user's\n"
        "documents that answer a question; docs_read, a document's text;\n"
        "docs_status, what is indexed. It indexes in the background the\n"
        "folders listed, a path a line, in ~/.config/janas/docs-dirs.txt\n"
        "(the Documents folder without it): text, Markdown, PDF, Word,\n"
        "Excel and PowerPoint (docx, xlsx, pptx), LibreOffice's (odt, ods,\n"
        "odp), RTF, web pages and EPUB books,\n"
        "never the private files (keys, passwords, the browsers' data).\n"
        "The index is kept in ~/.local/share/janas/docs/.\n"
        "Searched by their words always, and by their meaning with an\n"
        "embedding model: a janas-server started with --embedding-model\n"
        "(--server, or server = in ~/.config/janas/docs.conf, with\n"
        "server_key = when it wants one), or one janas-docs opens itself\n"
        "(--embedding-model, or embedding_model = there), such as\n"
        "qwen3-embedding-0.6b-q8.jns from janas-get.\n"
        "A PDF's pages without text, scanned, are read from their\n"
        "pictures by a model that sees them, with its .vision.jns beside\n"
        "it (--ocr-model, or ocr_model = in docs.conf), such as\n"
        "gemma-4-e2b-it-q4km.jns from janas-get.\n"
        "  --index        index now, say what is there, and end\n"
        "  --search TEXT  search the index as it is, and end\n"
        "  --text FILE    the text read from one document, as indexed\n"
        "Configured in a client as, for instance:\n"
        "  {\"mcpServers\": {\"docs\": {\"command\": \"janas-docs\"}}}\n");
}

static void mkdirs(char *p)
{
    /* from the first folder: "C:/" is a drive's, there already */
    char *s = p + (p[0] && p[1] == ':' && p[2] ? 3 : 1);
    for (; *s; s++)
        if (*s == '/') {
            *s = 0;
            if (mkdir(p, 0700) != 0 && errno != EEXIST)
                fprintf(stderr, "janas-docs: %s: %s\n", p, strerror(errno));
            *s = '/';
        }
}

int main(int argc, char **argv)
{
    const char *server = NULL, *key = NULL, *model = NULL, *search = NULL,
               *one = NULL, *ocr = NULL;
    int index_only = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--server") && v)
            server = v, i++;
        else if (!strcmp(a, "--key") && v)
            key = v, i++;
        else if (!strcmp(a, "--embedding-model") && v)
            model = v, i++;
        else if (!strcmp(a, "--ocr-model") && v)
            ocr = v, i++;
        else if (!strcmp(a, "--search") && v)
            search = v, i++;
        else if (!strcmp(a, "--index"))
            index_only = 1;
        else if (!strcmp(a, "--text") && v)
            one = v, i++;
        else {
            usage();
            return !strcmp(a, "-h") || !strcmp(a, "--help") ? 0 : 2;
        }
    }
    dz_ocr_set(ocr); /* --text reads scanned pages too, with --ocr-model */
    if (one) {       /* a document's text, as the index reads it */
        struct janas_buf t = {0};
        char why[256];
        if (dz_extract(one, &t, why, sizeof why) != 0) {
            fprintf(stderr, "janas-docs: %s: %s\n", one, why);
            janas_buf_free(&t);
            return 1;
        }
        fwrite(t.p, 1, t.n, stdout);
        if (t.n)
            fputc('\n', stdout);
        janas_buf_free(&t);
        return 0;
    }
    static struct dz_ctx c;
    if (dz_roots_load(&c.roots) != 0) {
        fprintf(stderr, "janas-docs: no home (HOME)\n");
        return 1;
    }
    /* the settings file, for what the command line did not say */
    static const char *const NAMES[] = {"server", "server_key",
                                        "embedding_model", "ocr_model"};
    char conf[PATH_MAX + 64], cs[1024], ck[512], cm[PATH_MAX], why[256];
    static char cm_home[2 * PATH_MAX];
    const char *own_conf = getenv("JANAS_DOCS_CONF"); /* an app's own */
    if (own_conf && janas_path_abs(own_conf))
        snprintf(conf, sizeof conf, "%s", own_conf);
    else
        snprintf(conf, sizeof conf, "%s/.config/janas/docs.conf", c.roots.home);
    if (!server && !model &&
        janas_conf_get(conf, NAMES, 4, "server", cs, sizeof cs, why,
                       sizeof why))
        server = cs;
    if (why[0])
        fprintf(stderr, "janas-docs: %s\n", why);
    if (server && !key &&
        janas_conf_get(conf, NAMES, 4, "server_key", ck, sizeof ck, why,
                       sizeof why))
        key = ck;
    if (!server && !model &&
        janas_conf_get(conf, NAMES, 4, "embedding_model", cm, sizeof cm, why,
                       sizeof why)) {
        model = cm;
        if (cm[0] == '~' && cm[1] == '/') {
            snprintf(cm_home, sizeof cm_home, "%s%s", c.roots.home, cm + 1);
            model = cm_home;
        }
    }
    /* the model that reads the pictures of scanned pages */
    char co[PATH_MAX];
    static char co_home[2 * PATH_MAX];
    if (!ocr && janas_conf_get(conf, NAMES, 4, "ocr_model", co, sizeof co, why,
                               sizeof why)) {
        ocr = co;
        if (co[0] == '~' && co[1] == '/') {
            snprintf(co_home, sizeof co_home, "%s%s", c.roots.home, co + 1);
            ocr = co_home;
        }
    }
    dz_ocr_set(ocr);
    /* a server's indexing gives way to the programs the user is using - a
       chat's model first: the threads made from here on (the embedding
       model's among them) take this priority */
    if (!index_only && !search && setpriority(PRIO_PROCESS, 0, 10) != 0)
        fprintf(stderr, "janas-docs: its priority stays: %s\n",
                strerror(errno));
    char err[512];
    if (dz_embedder_open(&c.emb, server, key, model, err, sizeof err) != 0) {
        fprintf(stderr, "janas-docs: %s\n", err);
        return 1;
    }

    struct dz_index *x = &c.index;
    pthread_mutex_init(&x->mu, NULL);
    pthread_cond_init(&x->wake, NULL);
    x->roots = &c.roots;
    x->emb = &c.emb;
    char path[PATH_MAX + 64];
    const char *data = getenv("XDG_DATA_HOME");
    /* (on Windows XDG_DATA_HOME is %LOCALAPPDATA%: common/win_compat.c) */
    if (data && janas_path_abs(data))
        snprintf(path, sizeof path, "%s/janas/docs/index.bin", data);
    else
        snprintf(path, sizeof path, "%s/.local/share/janas/docs/index.bin",
                 c.roots.home);
    janas_path_slashes(path);
    mkdirs(path);
    if (dz_index_load(x, path, err, sizeof err) != 0)
        fprintf(stderr, "janas-docs: %s\n", err);

    int rc = 0;
    if (index_only || search) {
        struct janas_buf o = {0};
        if (index_only) {
            dz_index_round(x);
            rc = dz_tool_status(&c, &o);
        } else {
            rc = dz_tool_search(&c, search, NULL, 6, &o);
        }
        printf("%.*s\n", (int)o.n, o.p ? o.p : "");
        janas_buf_free(&o);
    } else {
        struct janas_mcps srv = {.name = "janas-docs",
                                 .version = JANAS_VERSION,
                                 .instructions = INSTRUCTIONS,
                                 .ctx = &c,
                                 .tools_list = dz_tools_list,
                                 .tools_call = dz_tools_call};
        if (janas_mcps_open(&srv) != 0) {
            fprintf(stderr, "janas-docs: cannot set its output aside\n");
            return 1;
        }
        char name[sizeof c.emb.name];
        int ready = dz_embed_state(&c.emb, name, sizeof name, NULL);
        fprintf(stderr,
                "janas-docs %s: ready, 3 tools (docs_search, docs_read, "
                "docs_status), %d folder%s, %zu documents indexed, by %s\n",
                JANAS_VERSION, c.roots.n_root, c.roots.n_root == 1 ? "" : "s",
                x->n,
                c.emb.kind == DZ_EMB_NONE ? "their words"
                : ready                   ? name
                                          : "their words until the server "
                                            "answers");
        if (dz_indexer_start(x) != 0)
            fprintf(stderr, "janas-docs: no thread to index with\n");
        janas_mcps_run(&srv);
        dz_indexer_stop(x);
    }
    dz_index_free(x);
    dz_embedder_close(&c.emb);
    return rc != 0;
}
