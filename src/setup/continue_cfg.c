/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * continue_cfg.c - Janas's models in Continue (see continue_cfg.h).
 */
#include "continue_cfg.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "common/catalog.h"

int setup_continue_dir(char *buf, size_t len)
{
    /* on Windows getenv gives USERPROFILE for HOME, as os.homedir() */
    const char *g = getenv("CONTINUE_GLOBAL_DIR"), *h = getenv("HOME");
    int n;
    if (g && *g)
        n = snprintf(buf, len, "%s", g);
    else if (h && *h)
        n = snprintf(buf, len, "%s/.continue", h);
    else
        return -1;
    return n > 0 && (size_t)n < len ? 0 : -1;
}

static int exists(const char *dir, const char *file)
{
    char p[4200];
    struct stat st;
    snprintf(p, sizeof p, "%s/%s", dir, file);
    return stat(p, &st) == 0;
}

/* one model of the block */
static void model(FILE *f, const struct setup_served *s, const char *what,
                  const char *roles, int completes)
{
    fprintf(f,
            "  - name: \"Janas %s (%s)\"\n"
            "    provider: ollama\n"
            "    model: \"%s\"\n"
            "    apiBase: \"http://127.0.0.1:%d\"\n"
            "    roles: [%s]\n",
            s->name, what, s->name, s->port, roles);
    if (strstr(roles, "chat"))
        fprintf(f, "    capabilities: [tool_use]\n");
    if (s->ctx)
        fprintf(f,
                "    defaultCompletionOptions:\n"
                "      contextLength: %u\n",
                s->ctx);
    /* Continue keeps a completion 150 ms by default, then cuts it: a model
       on this machine takes longer over a prompt */
    if (completes)
        fprintf(f, "    autocompleteOptions:\n"
                   "      modelTimeout: 20000\n");
}

int setup_continue_write(const struct setup_served *code,
                         const struct setup_served *complete, const char **note)
{
    char dir[4096], models[4200], file[4300], tmp[4400];
    *note = NULL;
    if (setup_continue_dir(dir, sizeof dir) != 0) {
        fprintf(stderr, "janas-setup: no home folder for Continue's\n");
        return -1;
    }
    if (!exists(dir, "config.yaml") && exists(dir, "config.json"))
        *note = "Continue here reads its old config.json, and the models "
                "of models/ only once there is a config.yaml: Continue's "
                "own \"Migrate to config.yaml\" makes one";
    snprintf(models, sizeof models, "%s/models", dir);
    if (janas_mkdirs(models) != 0) {
        fprintf(stderr, "janas-setup: cannot make %s: %s\n", models,
                strerror(errno));
        return -1;
    }
    snprintf(file, sizeof file, "%s/janas.yaml", models);
    snprintf(tmp, sizeof tmp, "%s.tmp", file);
    FILE *f = fopen(tmp, "w");
    if (!f) {
        fprintf(stderr, "janas-setup: cannot write %s: %s\n", tmp,
                strerror(errno));
        return -1;
    }
    fprintf(f, "# Written by janas-setup: the models Janas serves on this "
               "machine.\n# janas-setup editor --undo removes it.\n"
               "name: Janas\nversion: 1.0.0\nschema: v1\nmodels:\n");
    int same = complete && complete->port == code->port &&
               strcmp(complete->name, code->name) == 0;
    model(f, code, same ? "code and completion" : "code",
          same ? "chat, edit, apply, summarize, autocomplete"
               : "chat, edit, apply, summarize",
          same);
    if (complete && !same)
        model(f, complete, "completion", "autocomplete", 1);
    int bad = ferror(f);
    if (fclose(f) != 0 || bad) {
        fprintf(stderr, "janas-setup: cannot write %s\n", tmp);
        remove(tmp);
        return -1;
    }
    remove(file); /* Windows does not rename over a file */
    if (rename(tmp, file) != 0) {
        fprintf(stderr, "janas-setup: cannot write %s: %s\n", file,
                strerror(errno));
        remove(tmp);
        return -1;
    }
    printf("   Continue: %s\n", file);
    return 0;
}

int setup_continue_remove(void)
{
    char dir[4096], file[4300];
    if (setup_continue_dir(dir, sizeof dir) != 0)
        return 0;
    snprintf(file, sizeof file, "%s/models/janas.yaml", dir);
    if (remove(file) == 0) {
        printf("   Continue: %s removed\n", file);
        return 1;
    }
    return errno == ENOENT ? 0 : -1;
}
