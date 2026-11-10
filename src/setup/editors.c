/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * editors.c - the editors of the VS Code family and their extensions (see
 * editors.h). An editor is found in PATH, then where its installer puts
 * it (Windows: under the user's or the system's programs; Linux: a
 * Flatpak), and its command's output is read from a file, which works the
 * same on every system.
 */
#include "editors.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common/catalog.h"
#include "common/proc.h"
#include "plan.h"

const struct setup_known setup_known_ext[] = {
    {.id = "continue.continue",
     .name = "Continue",
     .does = "completion as you type, chat, edits and an agent",
     .configured = 1,
     .on_open_vsx = 1},
    {.id = "ggml-org.llama-vscode",
     .name = "llama.vscode",
     .does = "completion as you type",
     .configured = 1,
     .on_open_vsx = 1},
    {.id = "saoudrizwan.claude-dev",
     .name = "Cline",
     .does = "an agent that plans, then writes and runs, step by step",
     .configured = 1,
     .on_open_vsx = 1},
    {.id = "rooveterinaryinc.roo-cline",
     .name = "Roo Code",
     .does = "an agent with modes (architect, code, ask)",
     .configured = 1,
     .on_open_vsx = 1,
     .archived = 1},
    {.id = "github.copilot-chat",
     .name = "GitHub Copilot Chat",
     .does = "VS Code's own chat, with a model of this machine",
     .configured = 1},
};
const int setup_known_n = sizeof setup_known_ext / sizeof *setup_known_ext;

#ifdef _WIN32
#define PATH_SEP ';'
#define CLI_EXT ".cmd"
#else
#define PATH_SEP ':'
#define CLI_EXT ""
#endif

static const struct {
    const char *name, *cli;
    int open_vsx;
    const char *config;  /* its folder of settings, under the user's */
    const char *win_dir; /* Windows: its folder under Programs */
    const char *flatpak; /* Linux: its Flatpak */
} family[] = {
    {"VSCodium", "codium", 1, "VSCodium", "VSCodium", "com.vscodium.codium"},
    {"VSCodium Insiders", "codium-insiders", 1, "VSCodium - Insiders",
     "VSCodium Insiders", NULL},
    {"VS Code", "code", 0, "Code", "Microsoft VS Code",
     "com.visualstudio.code"},
    {"VS Code Insiders", "code-insiders", 0, "Code - Insiders",
     "Microsoft VS Code Insiders", NULL},
};

static int runnable(const char *file)
{
    struct stat st;
    if (stat(file, &st) != 0 || !S_ISREG(st.st_mode))
        return 0;
#ifdef _WIN32
    return 1; /* a .cmd: Windows has no mode that says it runs */
#else
    return access(file, X_OK) == 0;
#endif
}

/* cli in PATH: its path in buf */
static int in_path(const char *cli, char *buf, size_t len)
{
    const char *path = getenv("PATH");
    while (path && *path) {
        size_t n = strcspn(path, (char[]){PATH_SEP, 0});
        if (n && n < 4000) {
            snprintf(buf, len, "%.*s/%s%s", (int)n, path, cli, CLI_EXT);
            if (runnable(buf))
                return 1;
        }
        path += n;
        if (*path == PATH_SEP)
            path++;
    }
    return 0;
}

/* where the installer puts it */
static int installed(int i, char *buf, size_t len)
{
#ifdef _WIN32
    const char *root[] = {getenv("LOCALAPPDATA"), getenv("ProgramFiles")};
    for (int r = 0; r < 2; r++) {
        if (!root[r] || !*root[r])
            continue;
        snprintf(buf, len, "%s%s/%s/bin/%s.cmd", root[r],
                 r == 0 ? "/Programs" : "", family[i].win_dir, family[i].cli);
        if (runnable(buf))
            return 1;
    }
#else
    (void)i;
    (void)buf;
    (void)len;
#endif
    return 0;
}

/* the folder of the user's settings */
static void user_dir(struct setup_editor *e, int i, int flatpak)
{
    const char *home = getenv("HOME"), *conf = getenv("XDG_CONFIG_HOME");
    if (e->user_data_dir)
        snprintf(e->user_dir, sizeof e->user_dir, "%s/User", e->user_data_dir);
    else if (flatpak)
        snprintf(e->user_dir, sizeof e->user_dir,
                 "%s/.var/app/%s/config/%s/User", home ? home : "",
                 family[i].flatpak, family[i].config);
    else if (conf && *conf)
        snprintf(e->user_dir, sizeof e->user_dir, "%s/%s/User", conf,
                 family[i].config);
    else
        snprintf(e->user_dir, sizeof e->user_dir, "%s/.config/%s/User",
                 home ? home : "", family[i].config);
}

/* the editor's command line with its folders, then extra */
static void args(const struct setup_editor *e, const char **argv, int max,
                 const char *const *extra)
{
    int n = 0;
    argv[n++] = e->cli;
    if (e->pre) {
        argv[n++] = "run";
        argv[n++] = e->pre;
    }
    if (e->user_data_dir) {
        argv[n++] = "--user-data-dir";
        argv[n++] = e->user_data_dir;
    }
    if (e->extensions_dir) {
        argv[n++] = "--extensions-dir";
        argv[n++] = e->extensions_dir;
    }
    for (; *extra && n < max - 1; extra++)
        argv[n++] = *extra;
    argv[n] = NULL;
}

/* Runs the editor's command with extra, its output (both streams) in a
   file beside janas-setup's plan, then in *out (malloc'd, ended by a 0):
   its exit status, or -1. */
static int run(const struct setup_editor *e, const char *const *extra,
               int timeout_ms, char **out)
{
    const char *argv[16];
    args(e, argv, 16, extra);
    *out = NULL;
    char dir[4096], file[4200], err[256];
    if (setup_conf_dir(dir, sizeof dir) != 0 || janas_mkdirs(dir) != 0)
        return -1;
    snprintf(file, sizeof file, "%s/editor-cli.out", dir);
    int fd = open(file, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0)
        return -1;
    int nul = janas_null_fd(0);
    struct janas_proc p;
    int rc = janas_proc_start(&p, argv, NULL, NULL, nul, fd, fd,
                              JANAS_PROC_DETACHED, err, sizeof err);
    if (nul >= 0)
        close(nul);
    close(fd);
    int code = -1;
    if (rc == 0 && janas_proc_wait(&p, timeout_ms, &code) != 1) {
        janas_proc_kill(&p, 1);
        janas_proc_wait(&p, -1, NULL);
        code = -1;
    }
    FILE *f = fopen(file, "rb");
    if (f) {
        size_t cap = 1 << 16, n = 0;
        char *b = malloc(cap + 1);
        size_t r;
        while (b && (r = fread(b + n, 1, cap - n, f)) > 0) {
            n += r;
            if (n == cap) {
                char *g = realloc(b, cap * 2 + 1);
                if (!g)
                    break;
                b = g;
                cap *= 2;
            }
        }
        fclose(f);
        if (b)
            b[n] = 0;
        *out = b;
    }
    remove(file);
    return rc == 0 ? code : -1;
}

static void list_ext(struct setup_editor *e)
{
    const char *extra[] = {"--list-extensions", "--show-versions", NULL};
    char *out;
    e->n_ext = 0;
    if (run(e, extra, 60000, &out) == 0 && out)
        for (char *line = strtok(out, "\r\n"); line && e->n_ext < SETUP_EXT_MAX;
             line = strtok(NULL, "\r\n")) {
            /* publisher.name@version, nothing else */
            if (!strchr(line, '.') || strchr(line, ' '))
                continue;
            char *x = e->ext[e->n_ext++];
            snprintf(x, 128, "%s", line);
            for (; *x && *x != '@'; x++)
                *x = (char)tolower((unsigned char)*x);
        }
    free(out);
}

int setup_editors_find(struct setup_editor *out, int max,
                       const char *user_data_dir, const char *extensions_dir)
{
    int n = 0;
    for (int i = 0; i < (int)(sizeof family / sizeof *family) && n < max; i++) {
        struct setup_editor *e = &out[n];
        memset(e, 0, sizeof *e);
        int flatpak = 0;
        if (!in_path(family[i].cli, e->cli, sizeof e->cli) &&
            !installed(i, e->cli, sizeof e->cli)) {
            /* a Flatpak, when flatpak is there and the application has
               its folder */
            char app[4200];
            const char *home = getenv("HOME");
            if (!family[i].flatpak || !home ||
                !in_path("flatpak", e->cli, sizeof e->cli))
                continue;
            snprintf(app, sizeof app, "%s/.var/app/%s", home,
                     family[i].flatpak);
            struct stat st;
            if (stat(app, &st) != 0)
                continue;
            e->pre = family[i].flatpak;
            flatpak = 1;
        }
        n++;
        e->name = family[i].name;
        e->open_vsx = family[i].open_vsx;
        e->user_data_dir = user_data_dir;
        e->extensions_dir = extensions_dir;
        user_dir(e, i, flatpak);
        list_ext(e);
    }
    return n;
}

const char *setup_editor_has(const struct setup_editor *e, const char *id)
{
    size_t n = strlen(id);
    for (int i = 0; i < e->n_ext; i++)
        if (strncmp(e->ext[i], id, n) == 0 &&
            (e->ext[i][n] == '@' || e->ext[i][n] == 0))
            return e->ext[i][n] == '@' ? e->ext[i] + n + 1 : "";
    return NULL;
}

int setup_editor_install(struct setup_editor *e, const char *id)
{
    const char *extra[] = {"--install-extension", id, NULL};
    char *out;
    printf("   %s%s%s --install-extension %s\n", e->cli, e->pre ? " run " : "",
           e->pre ? e->pre : "", id);
    fflush(stdout);
    int rc = run(e, extra, 600000, &out);
    if (rc != 0)
        fprintf(stderr, "janas-setup: %s could not install %s%s%s\n", e->name,
                id, out && *out ? ": " : "", out ? out : "");
    free(out);
    if (rc != 0)
        return -1;
    list_ext(e);
    return setup_editor_has(e, id) ? 0 : -1;
}
