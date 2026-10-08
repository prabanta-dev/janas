/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * models.c - janas-chat's models by name (see models.h).
 */
#define _GNU_SOURCE
#include "models.h"

#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "janas/llm.h"

extern char **environ;

static int is_file(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int is_dir(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/* A line from the user, without its newline; NULL at the end of input. */
static char *ask_line(const char *question, char *buf, size_t len)
{
    fputs(question, stderr);
    fflush(stderr);
    if (!fgets(buf, (int)len, stdin))
        return NULL;
    buf[strcspn(buf, "\n")] = 0;
    return buf;
}

int chat_ask_yes(const char *question)
{
    char b[64];
    fprintf(stderr, "%s [y/N] ", question);
    fflush(stderr);
    if (!fgets(b, sizeof(b), stdin))
        return 0;
    return b[0] == 'y' || b[0] == 'Y';
}

static void set_key(struct chat_model *cm, const char *base)
{
    snprintf(cm->key, sizeof(cm->key), "%s", base);
    size_t n = strlen(cm->key);
    if (n > 4 && strcmp(cm->key + n - 4, ".jns") == 0)
        cm->key[n - 4] = 0;
}

/* The catalog's chat models, numbered from 1, with what is already here. */
static void list(const char *dir)
{
    fprintf(stderr, "The models janas-chat knows (in %s):\n\n", dir);
    int k = 0;
    for (int i = 0; i < janas_catalog_n; i++) {
        const struct janas_model_entry *m = &janas_catalog[i];
        if (m->not_chat)
            continue;
        char p[4200];
        snprintf(p, sizeof(p), "%s/%s", dir, m->out);
        fprintf(stderr,
                "%2d. %s%s\n    %s; %.1f GB to download, %d GB of "
                "memory or more\n",
                ++k, m->name, is_file(p) ? "  (here)" : "", m->what, m->gguf_gb,
                m->ram_gb);
    }
    fputc('\n', stderr);
}

/* The k-th chat model of the list, from 1. */
static const struct janas_model_entry *nth(int k)
{
    for (int i = 0; i < janas_catalog_n; i++)
        if (!janas_catalog[i].not_chat && --k == 0)
            return &janas_catalog[i];
    return NULL;
}

/* janas-get, beside this program, for the model into dir; 0 if it made it */
static int fetch(const struct janas_model_entry *m, const char *dir)
{
    char self[4096], tool[4200];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0)
        return -1;
    self[n] = 0;
    char *slash = strrchr(self, '/');
    if (!slash)
        return -1;
    *slash = 0;
    snprintf(tool, sizeof(tool), "%s/janas-get", self);
    char *argv[] = {"janas-get", (char *)m->name, "--dir", (char *)dir, NULL};
    pid_t pid;
    if (posix_spawn(&pid, tool, NULL, NULL, argv, environ) != 0) {
        fprintf(stderr, "janas-chat: cannot start %s\n", tool);
        return -1;
    }
    int st = 0;
    while (waitpid(pid, &st, 0) < 0)
        ;
    return WIFEXITED(st) && WEXITSTATUS(st) == 0 ? 0 : -1;
}

/*
 * The folder for a first download: the proposed one, or another the user
 * types - and then a word on how janas-chat will find it again, since only
 * JANAS_MODELS or --models tell it.
 */
static int choose_dir(char *dir, size_t len)
{
    const char *e = getenv("JANAS_MODELS");
    if ((e && *e) || is_dir(dir) || !isatty(STDIN_FILENO))
        return 0;
    char b[4096];
    fprintf(stderr, "The models will go to %s.\n", dir);
    if (!ask_line("Press Enter to accept, or type another folder: ", b,
                  sizeof(b)))
        return -1;
    if (b[0]) {
        snprintf(dir, len, "%s", b);
        fprintf(stderr,
                "Note: janas-chat looks in %s only when told - start it with "
                "--models %s, or set JANAS_MODELS=%s.\n",
                dir, dir, dir);
    }
    return 0;
}

int chat_model_resolve(const char *arg, struct chat_model *cm)
{
    memset(cm, 0, sizeof(*cm));
    if (arg && is_file(arg)) {
        snprintf(cm->path, sizeof(cm->path), "%s", arg);
        snprintf(cm->dir, sizeof(cm->dir), "%s", arg);
        char *slash = strrchr(cm->dir, '/');
        const char *base = slash ? slash + 1 : arg;
        if (slash)
            *slash = 0;
        else
            snprintf(cm->dir, sizeof(cm->dir), ".");
        for (int i = 0; i < janas_catalog_n; i++)
            if (strcmp(janas_catalog[i].out, base) == 0)
                cm->entry = &janas_catalog[i];
        set_key(cm, cm->entry ? cm->entry->name : base);
        return 0;
    }
    if (arg && (strchr(arg, '/') || strstr(arg, ".jns"))) {
        fprintf(stderr, "janas-chat: there is no file %s\n", arg);
        return -1;
    }
    if (janas_models_dir(cm->dir, sizeof(cm->dir)) != 0) {
        fprintf(stderr, "janas-chat: no home directory for the models: give "
                        "a model's path\n");
        return -1;
    }
    const struct janas_model_entry *m = NULL;
    if (arg) {
        m = janas_catalog_find(arg);
        if (!m || m->not_chat) {
            fprintf(stderr,
                    "janas-chat: no model called %s (janas-chat with no "
                    "model lists them)\n",
                    arg);
            return -1;
        }
    } else {
        if (!isatty(STDIN_FILENO)) {
            fprintf(stderr, "janas-chat: no model given\n");
            return -1;
        }
        list(cm->dir);
        char b[128];
        while (!m) {
            if (!ask_line("Which one (number or name, Enter to leave)? ", b,
                          sizeof(b)) ||
                !b[0])
                return -1;
            int k = atoi(b);
            m = k > 0 ? nth(k) : janas_catalog_find(b);
            if (m && m->not_chat)
                m = NULL;
            if (!m)
                fprintf(stderr, "not in the list: %s\n", b);
        }
    }
    cm->entry = m;
    set_key(cm, m->name);
    snprintf(cm->path, sizeof(cm->path), "%s/%s", cm->dir, m->out);
    if (is_file(cm->path))
        return 0;
    if (!isatty(STDIN_FILENO)) {
        fprintf(stderr, "janas-chat: %s is not here yet: janas-get %s\n",
                m->name, m->name);
        return -1;
    }
    if (choose_dir(cm->dir, sizeof(cm->dir)) != 0)
        return -1;
    snprintf(cm->path, sizeof(cm->path), "%s/%s", cm->dir, m->out);
    if (is_file(cm->path))
        return 0;
    char q[512];
    snprintf(q, sizeof(q),
             "%s is not here: download it (%.1f GB, and about %.0f GB free "
             "while it is converted) with its prediction file?",
             m->name, m->gguf_gb, m->gguf_gb * 2.0 + 0.5);
    if (!chat_ask_yes(q))
        return -1;
    if (fetch(m, cm->dir) != 0 || !is_file(cm->path)) {
        fprintf(stderr, "janas-chat: %s was not made\n", cm->path);
        return -1;
    }
    return 0;
}

const char *chat_model_draft(const struct chat_model *cm, char *buf, size_t len)
{
    const struct janas_model_entry *d =
        cm->entry && cm->entry->draft ? janas_catalog_find(cm->entry->draft)
                                      : NULL;
    if (!d)
        return NULL;
    snprintf(buf, len, "%s/%s", cm->dir, d->out);
    return is_file(buf) ? buf : NULL;
}

int chat_words_any(int n, char *const *w, const char *const *opts)
{
    for (int i = 0; i < n; i++)
        for (int k = 0; opts[k]; k++)
            if (strcmp(w[i], opts[k]) == 0)
                return 1;
    return 0;
}

const char *chat_model_mtp(const struct chat_model *cm, char *buf,
                           size_t len)
{
    if (cm->entry) {
        if (cm->entry->extra == JANAS_EXTRA_NONE || !cm->entry->extra_out)
            return NULL;
        snprintf(buf, len, "%s/%s", cm->dir, cm->entry->extra_out);
        return is_file(buf) ? buf : NULL;
    }
    return janas_llm_find_mtp(cm->path, buf, len) == JANAS_LLM_OK ? buf
                                                                   : NULL;
}

const char *chat_words_last(int n, char *const *w, const char *opt)
{
    const char *v = NULL;
    for (int i = 0; i + 1 < n; i++)
        if (strcmp(w[i], opt) == 0)
            v = w[i + 1];
    return v;
}
