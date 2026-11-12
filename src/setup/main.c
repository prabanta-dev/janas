/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-setup - Janas in the editor: the models installed here given to
 * the extensions of VS Code and VSCodium, the servers that serve them.
 *
 *   janas-setup editor [--list | --undo] [--code MODEL] [--complete MODEL|none]
 *                      [--yes] [--user-data-dir DIR] [--extensions-dir DIR]
 *                      [--editor vscode|vscodium]
 *   janas-setup serve
 *
 * editor finds the editors and their extensions, proposes a model for code
 * and one for completion from the memory of the machine (the reasons
 * said), and configures the extensions it can - Continue, by a file of its
 * own beside the user's - adding Janas beside what is there, never in its
 * place. Where none is installed it asks what the user wants to do and
 * installs the extension for it from the editor's own gallery. For the
 * others it says what to fill in. serve starts the servers.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/catalog.h"
#include "common/sysinfo.h"
#include "continue_cfg.h"
#include "editors.h"
#include "common/model_scan.h"
#include "plan.h"
#include "serve.h"
#include "vscode_cfg.h"

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-setup editor [--list | --undo] [--code MODEL]\n"
            "                          [--complete MODEL|none] [--yes]\n"
            "                          [--user-data-dir DIR] "
            "[--extensions-dir DIR]\n"
            "                          [--editor vscode|vscodium]\n"
            "       janas-setup serve [--monitor]\n"
            "  editor   give the models installed here to VS Code's and\n"
            "           VSCodium's extensions (Continue, llama.vscode, "
            "Cline...)\n"
            "  --list   only say what is here: models, editors, extensions\n"
            "  --undo   take Janas away from the extensions' settings\n"
            "  --code, --complete  the model for code (chat, edits, the "
            "agent)\n"
            "           and for completion as you type, instead of the ones\n"
            "           proposed\n"
            "  --yes    take the proposals without asking\n"
            "  --user-data-dir, --extensions-dir  passed on to the editor\n"
            "  --editor this editor only, where both are installed\n"
            "  serve    start the servers editor set up, until Ctrl+C\n"
            "  --monitor  each server shows the engine at work at /monitor\n");
}

/* a line from the user; "" at the end of the input */
static void ask(const char *question, char *buf, size_t len)
{
    printf("%s ", question);
    fflush(stdout);
    if (!fgets(buf, (int)len, stdin))
        buf[0] = 0;
    buf[strcspn(buf, "\r\n")] = 0;
}

/* something listening on 127.0.0.1:port */
static int listening(int port)
{
#ifdef _WIN32
    (void)port;
    return 0;
#else
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return 0;
    struct sockaddr_in a = {.sin_family = AF_INET,
                            .sin_port = htons((uint16_t)port)};
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int ok = connect(fd, (struct sockaddr *)&a, sizeof a) == 0;
    close(fd);
    return ok;
#endif
}

/* a model by its file's name, or by its name in the catalog
   (qwen3-0.6b for qwen3-0.6b-q4km.jns) */
static const struct janas_model_file *by_name(const struct janas_model_file *m,
                                              int n, const char *name)
{
    for (int i = 0; i < n; i++)
        if (strcmp(m[i].name, name) == 0)
            return &m[i];
    const struct janas_model_entry *e = janas_catalog_find(name);
    if (!e)
        return NULL;
    size_t k = strlen(e->out) - 4; /* without .jns */
    for (int i = 0; i < n; i++)
        if (strlen(m[i].name) == k && strncmp(m[i].name, e->out, k) == 0)
            return &m[i];
    return NULL;
}

static void list_models(const struct janas_model_file *m, int n)
{
    for (int i = 0; i < n; i++)
        printf("   %-34s %6.1f GB  %s%s%s\n", m[i].name,
               (double)m[i].bytes / 1e9,
               m[i].coder ? "made for code" : "general",
               janas_model_completes(&m[i]) ? ", completes at the cursor" : "",
               m[i].fim && m[i].recurrent && m[i].coder
                   ? ", fills in but recurrent: not for completion"
                   : "");
}

static void list_editors(const struct setup_editor *e, int n)
{
    if (!n)
        printf("   none found (codium, code)\n");
    for (int i = 0; i < n; i++) {
        printf("   %s (%s, extensions from %s)\n", e[i].name, e[i].cli,
               e[i].open_vsx ? "Open VSX" : "Microsoft's gallery");
        int any = 0;
        for (int k = 0; k < setup_known_n; k++) {
            const char *v = setup_editor_has(&e[i], setup_known_ext[k].id);
            if (v) {
                printf("      %s %s\n", setup_known_ext[k].name, v);
                any = 1;
            }
        }
        if (!any)
            printf("      no extension Janas knows (%d others)\n", e[i].n_ext);
    }
}

/* the extension the user wants, installed in an editor: its index in
   setup_known_ext, or -1 */
static int offer(struct setup_editor *e, int n_e, int yes)
{
    if (!n_e) {
        printf("No VS Code or VSCodium found: install one, then run "
               "janas-setup editor again.\n");
        return -1;
    }
    struct setup_editor *ed = &e[0];
    if (n_e > 1 && !yes) {
        char a[64];
        for (int i = 0; i < n_e; i++)
            printf("   %d) %s\n", i + 1, e[i].name);
        ask("Which editor? [1]", a, sizeof a);
        int k = atoi(a);
        if (k >= 1 && k <= n_e)
            ed = &e[k - 1];
    }
    int idx[8], n = 0;
    printf("What would you like Janas to do in %s?\n", ed->name);
    for (int k = 0; k < setup_known_n && n < 8; k++) {
        const struct setup_known *x = &setup_known_ext[k];
        if (x->archived || (ed->open_vsx && !x->on_open_vsx))
            continue;
        idx[n++] = k;
        printf("   %d) %s - %s%s\n", n, x->does, x->name,
               k == 0 ? " (recommended: one extension does it all, and "
                        "Janas sets it up)"
                      : "");
    }
    char a[64] = "";
    if (!yes)
        ask("[1]", a, sizeof a);
    int k = a[0] ? atoi(a) : 1;
    if (k < 1 || k > n)
        return -1;
    const struct setup_known *x = &setup_known_ext[idx[k - 1]];
    printf("== installing %s in %s\n", x->name, ed->name);
    return setup_editor_install(ed, x->id) == 0 ? idx[k - 1] : -1;
}

static int editor(int argc, char **argv)
{
    int list = 0, undo = 0, yes = 0;
    const char *code = NULL, *complete = NULL, *udd = NULL, *xd = NULL,
               *only = NULL;
    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
        if (!strcmp(a, "--list"))
            list = 1;
        else if (!strcmp(a, "--undo"))
            undo = 1;
        else if (!strcmp(a, "--yes"))
            yes = 1;
        else if (!strcmp(a, "--code") && more)
            code = argv[++i];
        else if (!strcmp(a, "--complete") && more)
            complete = argv[++i];
        else if (!strcmp(a, "--user-data-dir") && more)
            udd = argv[++i];
        else if (!strcmp(a, "--extensions-dir") && more)
            xd = argv[++i];
        else if (!strcmp(a, "--editor") && more)
            only = argv[++i];
        else {
            usage();
            return 2;
        }
    }
    if (undo) {
        char file[4200], dir[4096];
        int r = setup_continue_remove();
        int v = setup_vscode_undo();
        if (setup_plan_file(file, sizeof file) == 0 && remove(file) == 0)
            printf("   %s removed\n", file);
        if (setup_conf_dir(dir, sizeof dir) == 0) {
            snprintf(file, sizeof file, "%s/roo-code.json", dir);
            if (remove(file) == 0)
                printf("   %s removed (a profile Roo Code imported stays "
                       "in it: delete it there)\n",
                       file);
        }
        return r < 0 || v < 0 ? 1 : 0;
    }

    char dir[4096];
    static struct janas_model_file m[JANAS_MODELS_MAX];
    int n = janas_models_dir(dir, sizeof dir) == 0
                ? janas_models_scan(dir, m, JANAS_MODELS_MAX)
                : 0;
    static struct setup_editor e[4];
    int n_e = setup_editors_find(e, 4, udd, xd);
    if (only) {
        /* "VS Code" is vscode, "VSCodium" vscodium */
        int k = 0;
        for (int i = 0; i < n_e; i++) {
            char low[32];
            size_t w = 0;
            for (const char *c = e[i].name; *c && w + 1 < sizeof low; c++)
                if (*c != ' ')
                    low[w++] = (char)tolower((unsigned char)*c);
            low[w] = 0;
            if (!strcmp(low, only))
                e[k++] = e[i];
        }
        if (!k) {
            fprintf(stderr, "janas-setup: no editor %s here\n", only);
            return 1;
        }
        n_e = k;
    }
    uint64_t mem = janas_mem_total();
    if (list) {
        printf("Memory: %.0f GB\nModels in %s:\n", (double)mem / 1e9, dir);
        list_models(m, n);
        printf("Editors:\n");
        list_editors(e, n_e);
        static const struct {
            int port;
            const char *who;
        } other[] = {{11434, "Ollama's port"},
                     {1234, "LM Studio's port"},
                     {8080, "llama.cpp's port"}};
        for (size_t i = 0; i < sizeof other / sizeof *other; i++)
            if (listening(other[i].port))
                printf("Something answers on %s, %d\n", other[i].who,
                       other[i].port);
        return 0;
    }
    if (!n) {
        printf("No model in %s yet. For code, janas-get list shows the "
               "models; janas-get qwen3-coder-30b-a3b downloads one made "
               "for code that also completes at the cursor (32 GB of "
               "memory).\n",
               dir);
        return 1;
    }

    struct setup_choice c = {0};
    if (code && !(c.code = by_name(m, n, code))) {
        fprintf(stderr, "janas-setup: no model %s in %s\n", code, dir);
        return 1;
    }
    if (complete && !strcmp(complete, "none"))
        c.no_complete = 1;
    else if (complete && !(c.complete = by_name(m, n, complete))) {
        fprintf(stderr, "janas-setup: no model %s in %s\n", complete, dir);
        return 1;
    }
    struct setup_plan p;
    setup_plan_propose(&p, m, n, mem, &c);
    if (!yes && !code && !complete) {
        char a[16];
        ask("Take these? [Y/n]", a, sizeof a);
        if (a[0] == 'n' || a[0] == 'N') {
            printf("Name them with --code and --complete; the models:\n");
            list_models(m, n);
            return 1;
        }
    }

    int known = 0;
    for (int i = 0; i < n_e; i++)
        for (int k = 0; k < setup_known_n; k++)
            known |= setup_editor_has(&e[i], setup_known_ext[k].id) != NULL;
    if (!known && offer(e, n_e, yes) < 0)
        return 1;

    printf("== settings\n");
    int done_continue = 0, bad = 0;
    for (int i = 0; i < n_e; i++)
        for (int k = 0; k < setup_known_n; k++) {
            const char *id = setup_known_ext[k].id;
            if (!setup_editor_has(&e[i], id))
                continue;
            if (strcmp(id, "continue.continue") != 0) {
                bad |= setup_vscode_write(&e[i], id, &p) < 0;
                continue;
            }
            /* one file for every editor */
            if (done_continue++)
                continue;
            struct setup_served sc = {.name = p.code->name,
                                      .port = p.code_port,
                                      .ctx = p.ctx},
                                sm = {0};
            if (p.complete)
                sm = (struct setup_served){.name = p.complete->name,
                                           .port = p.complete_port,
                                           .ctx = p.ctx};
            const char *note;
            if (setup_continue_write(&sc, p.complete ? &sm : NULL, &note) != 0)
                bad = 1;
            else if (note)
                printf("   Note: %s.\n", note);
        }
    if (bad)
        return 1;
    if (setup_plan_save(&p) != 0)
        return 1;
    printf("Start the servers with: janas-setup serve\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "editor"))
        return editor(argc - 2, argv + 2);
    if (argc >= 2 && !strcmp(argv[1], "serve") &&
        (argc == 2 || (argc == 3 && !strcmp(argv[2], "--monitor"))))
        return setup_serve(argc == 3);
    usage();
    return 2;
}
