/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * plan.c - which installed model does what (see plan.h).
 */
#include "plan.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/catalog.h"

/* what the system, the editor and the context take beside the models */
#define RESERVE (6ull << 30)

static double gb(uint64_t b)
{
    return (double)b / 1e9;
}

/* the largest of the models that pick takes that fits in room, else the
   smallest of them; NULL when pick takes none */
static const struct janas_model_file *best(const struct janas_model_file *m,
                                           int n, uint64_t room, int coder_only,
                                           int *fits)
{
    const struct janas_model_file *big = NULL, *small = NULL;
    for (int i = 0; i < n; i++) {
        if (coder_only && !m[i].coder)
            continue;
        if (!small || m[i].bytes < small->bytes)
            small = &m[i];
        if (m[i].bytes <= room && (!big || m[i].bytes > big->bytes))
            big = &m[i];
    }
    *fits = big != NULL;
    return big ? big : small;
}

/* the completion the user named, said */
static void named_complete(struct setup_plan *p,
                           const struct janas_model_file *c, uint64_t room,
                           uint64_t mem)
{
    p->complete = c;
    p->complete_port = c == p->code ? p->code_port : SETUP_PORT + 1;
    printf("For completion as you type: %s, %.1f GB, as asked", c->name,
           gb(c->bytes));
    if (!janas_model_completes(c))
        printf(" - it is not made to fill in at the cursor: completion may "
               "be poor or slow");
    else if (c != p->code && p->code->bytes + c->bytes > room)
        printf(" - the two models together are more than this machine's "
               "%.0f GB of memory hold beside the system: Janas reads what "
               "does not fit from the disk, so it answers more slowly",
               gb(mem));
    printf(".\n");
}

void setup_plan_propose(struct setup_plan *p, const struct janas_model_file *m,
                        int n, uint64_t mem, const struct setup_choice *c)
{
    memset(p, 0, sizeof *p);
    p->ctx = SETUP_CTX;
    uint64_t room = mem > RESERVE ? mem - RESERVE : 0;
    int fits = 0;
    if (c && c->code) {
        p->code = c->code;
        fits = p->code->bytes <= room;
    } else {
        p->code = best(m, n, room, 1, &fits);
        if (!p->code)
            p->code = best(m, n, room, 0, &fits);
    }
    if (!p->code)
        return;
    p->code_port = SETUP_PORT;
    printf("For code (chat, edits, the agent): %s, %.1f GB%s", p->code->name,
           gb(p->code->bytes), c && c->code ? ", as asked" : "");
    if (fits)
        printf(" - this machine's %.0f GB of memory hold it whole.\n", gb(mem));
    else
        printf(" - more than this machine's %.0f GB of memory hold beside the "
               "system: Janas reads what does not fit from the disk, so it "
               "answers more slowly.\n",
               gb(mem));
    if (!p->code->coder && !(c && c->code))
        printf("   None of the models installed is made for code: this one "
               "is a general model.\n");
    if (c && c->no_complete) {
        printf("For completion as you type: none, as asked.\n");
        return;
    }
    if (c && c->complete) {
        named_complete(p, c->complete, room, mem);
        return;
    }

    /* completion: a smaller model of its own when both fit, otherwise the
       same one when it can */
    const struct janas_model_file *small = NULL;
    for (int i = 0; i < n; i++)
        if (&m[i] != p->code && janas_model_completes(&m[i]) &&
            m[i].bytes < p->code->bytes &&
            (!small || m[i].bytes < small->bytes))
            small = &m[i];
    int code_can = janas_model_completes(p->code);
    if (small && p->code->bytes + small->bytes <= room) {
        p->complete = small;
        p->complete_port = SETUP_PORT + 1;
        printf("For completion as you type: %s, %.1f GB - small, so it "
               "answers quickly; memory holds both.\n",
               small->name, gb(small->bytes));
    } else if (code_can) {
        p->complete = p->code;
        p->complete_port = p->code_port;
        printf("For completion as you type: the same model - one model in "
               "memory%s.\n",
               small ? ": a second one would not fit beside it" : "");
    } else if (small) {
        printf("For completion as you type: %s could, but memory does not "
               "hold it beside %s; none is set.\n",
               small->name, p->code->name);
    } else {
        printf("For completion as you type: no model installed can (it takes "
               "one trained to fill in the middle, as the Qwen coders"
               "%s); none is set.\n",
               p->code->fim && p->code->recurrent && p->code->coder
                   ? ", without recurrent layers: every keystroke would read "
                     "the window again from its start"
                   : "");
    }
}

int setup_conf_dir(char *buf, size_t len)
{
    /* on Windows getenv gives APPDATA for XDG_CONFIG_HOME
       (src/common/win_compat.c) */
    const char *x = getenv("XDG_CONFIG_HOME"), *h = getenv("HOME");
    int n;
    if (x && *x)
        n = snprintf(buf, len, "%s/janas", x);
    else if (h && *h)
        n = snprintf(buf, len, "%s/.config/janas", h);
    else
        return -1;
    return n > 0 && (size_t)n < len ? 0 : -1;
}

int setup_plan_file(char *buf, size_t len)
{
    char dir[4096];
    if (setup_conf_dir(dir, sizeof dir) != 0)
        return -1;
    int n = snprintf(buf, len, "%s/editor.conf", dir);
    return n > 0 && (size_t)n < len ? 0 : -1;
}

int setup_plan_save(const struct setup_plan *p)
{
    char file[4096], dir[4096];
    if (setup_plan_file(file, sizeof file) != 0)
        return -1;
    snprintf(dir, sizeof dir, "%s", file);
    char *slash = strrchr(dir, '/');
    if (slash)
        *slash = 0;
    if (janas_mkdirs(dir) != 0)
        return -1;
    FILE *f = fopen(file, "w");
    if (!f) {
        fprintf(stderr, "janas-setup: cannot write %s: %s\n", file,
                strerror(errno));
        return -1;
    }
    fprintf(f,
            "# Written by janas-setup editor; janas-setup serve starts "
            "these.\nctx=%u\ncode=%s\ncode_name=%s\ncode_port=%d\n",
            p->ctx, p->code->path, p->code->name, p->code_port);
    if (p->complete && p->complete != p->code)
        fprintf(f, "complete=%s\ncomplete_name=%s\ncomplete_port=%d\n",
                p->complete->path, p->complete->name, p->complete_port);
    return fclose(f) == 0 ? 0 : -1;
}

int setup_plan_load(struct setup_saved *s)
{
    char file[4096], line[4600];
    memset(s, 0, sizeof *s);
    if (setup_plan_file(file, sizeof file) != 0)
        return -1;
    FILE *f = fopen(file, "r");
    if (!f)
        return -1;
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        char *v = strchr(line, '=');
        if (line[0] == '#' || !v)
            continue;
        *v++ = 0;
        if (!strcmp(line, "ctx"))
            s->ctx = (unsigned)strtoul(v, NULL, 10);
        else if (!strcmp(line, "code"))
            snprintf(s->code_path, sizeof s->code_path, "%s", v);
        else if (!strcmp(line, "code_name"))
            snprintf(s->code_name, sizeof s->code_name, "%s", v);
        else if (!strcmp(line, "code_port"))
            s->code_port = atoi(v);
        else if (!strcmp(line, "complete"))
            snprintf(s->complete_path, sizeof s->complete_path, "%s", v);
        else if (!strcmp(line, "complete_name"))
            snprintf(s->complete_name, sizeof s->complete_name, "%s", v);
        else if (!strcmp(line, "complete_port"))
            s->complete_port = atoi(v);
    }
    fclose(f);
    return s->code_path[0] ? 0 : -1;
}
