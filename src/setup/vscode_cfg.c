/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vscode_cfg.c - Janas in the extensions' JSON settings (see
 * vscode_cfg.h). What a write replaced goes to editor-undo.conf beside the
 * plan, one line each - "K", the file, the path's keys joined by |, and
 * the old value on one line, or "-" when there was none; "A", the file,
 * for Janas's item of an array - the first time only, so that running
 * janas-setup again never takes Janas's own value for the user's.
 */
#include "vscode_cfg.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/catalog.h"
#include "common/jsonc.h"

static char *slurp(const char *file, size_t *n)
{
    FILE *f = fopen(file, "rb");
    *n = 0;
    if (!f)
        return NULL;
    size_t cap = 1 << 14;
    char *b = malloc(cap + 1);
    size_t r;
    while (b && (r = fread(b + *n, 1, cap - *n, f)) > 0) {
        *n += r;
        if (*n == cap) {
            char *g = realloc(b, cap * 2 + 1);
            if (!g) {
                free(b);
                b = NULL;
                break;
            }
            b = g;
            cap *= 2;
        }
    }
    fclose(f);
    if (b)
        b[*n] = 0;
    return b;
}

/* written whole or not at all: a file beside it, then renamed over it */
static int spill(const char *file, const char *text, size_t n)
{
    char dir[4096], tmp[4200];
    snprintf(dir, sizeof dir, "%s", file);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = 0;
        if (janas_mkdirs(dir) != 0)
            return -1;
    }
    snprintf(tmp, sizeof tmp, "%s.janas-tmp", file);
    FILE *f = fopen(tmp, "wb");
    if (!f)
        return -1;
    int bad = fwrite(text, 1, n, f) != n;
    if (fclose(f) != 0 || bad) {
        remove(tmp);
        return -1;
    }
    remove(file); /* Windows does not rename over a file */
    if (rename(tmp, file) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

static int undo_file(char *buf, size_t len)
{
    char dir[4096];
    if (setup_conf_dir(dir, sizeof dir) != 0 || janas_mkdirs(dir) != 0)
        return -1;
    int n = snprintf(buf, len, "%s/editor-undo.conf", dir);
    return n > 0 && (size_t)n < len ? 0 : -1;
}

/* a line of the log for (kind, file, keys) is already there */
static int logged(const char *kind, const char *file, const char *keys)
{
    char log[4200], want[8400];
    size_t n;
    if (undo_file(log, sizeof log) != 0)
        return 0;
    char *t = slurp(log, &n);
    if (!t)
        return 0;
    snprintf(want, sizeof want, "%s\t%s\t%s\t", kind, file, keys);
    int found = 0;
    for (char *l = t; l && *l;) {
        if (strncmp(l, want, strlen(want)) == 0)
            found = 1;
        l = strchr(l, '\n');
        if (l)
            l++;
    }
    free(t);
    return found;
}

static int note(const char *kind, const char *file, const char *keys,
                const char *old, size_t old_n)
{
    char log[4200];
    if (logged(kind, file, keys))
        return 0;
    if (undo_file(log, sizeof log) != 0)
        return -1;
    FILE *f = fopen(log, "ab");
    if (!f)
        return -1;
    fprintf(f, "%s\t%s\t%s\t", kind, file, keys);
    if (!old)
        fputc('-', f);
    /* a value's line breaks are spaces between its tokens: one line */
    for (size_t i = 0; old && i < old_n; i++)
        fputc(old[i] == '\n' || old[i] == '\r' ? ' ' : old[i], f);
    fputc('\n', f);
    return fclose(f) == 0 ? 0 : -1;
}

/* keys "a|b" split into path (pointers into buf) */
static int split(char *buf, const char **path, int max)
{
    int n = 0;
    for (char *k = strtok(buf, "|"); k && n < max; k = strtok(NULL, "|"))
        path[n++] = k;
    return n;
}

/* the member at keys of file set to value, what it held written down */
static int set_member(const char *file, const char *keys, const char *value)
{
    char kb[512];
    const char *path[8];
    snprintf(kb, sizeof kb, "%s", keys);
    int np = split(kb, path, 8);
    size_t n, at, len;
    char *t = slurp(file, &n);
    if (!t && errno != ENOENT) {
        fprintf(stderr, "janas-setup: cannot read %s\n", file);
        return -1;
    }
    int had = t ? janas_jsonc_find(t, n, path, np, &at, &len) : 0;
    if (had > 0 && len == strlen(value) && memcmp(t + at, value, len) == 0) {
        free(t);
        return 0; /* already so */
    }
    if (had > 0 && !logged("K", file, keys))
        printf("   %s in %s was %.*s: janas-setup editor --undo puts it "
               "back\n",
               path[np - 1], file, (int)(len < 200 ? len : 200), t + at);
    if (had < 0) {
        fprintf(stderr,
                "janas-setup: %s is not JSON as the editor writes it: left "
                "as it is\n",
                file);
        free(t);
        return -1;
    }
    char *out;
    size_t out_n;
    int rc = janas_jsonc_set(t ? t : "", n, path, np, value, &out, &out_n);
    if (rc == 0 &&
        note("K", file, keys, had ? t + at : NULL, had ? len : 0) != 0)
        rc = -1;
    if (rc == 0)
        rc = spill(file, out, out_n);
    if (rc != 0)
        fprintf(stderr, "janas-setup: cannot write %s\n", file);
    free(out);
    free(t);
    return rc;
}

static int llama_vscode(const struct setup_editor *e,
                        const struct setup_plan *p)
{
    char file[4200], url[64];
    snprintf(file, sizeof file, "%s/settings.json", e->user_dir);
    if (p->complete) {
        snprintf(url, sizeof url, "\"http://127.0.0.1:%d\"", p->complete_port);
        if (set_member(file, "llama-vscode.endpoint", url) != 0)
            return -1;
    }
    snprintf(url, sizeof url, "\"http://127.0.0.1:%d\"", p->code_port);
    if (set_member(file, "llama-vscode.endpoint_chat", url) != 0 ||
        set_member(file, "llama-vscode.endpoint_tools", url) != 0)
        return -1;
    printf("   llama.vscode: %s%s\n", file,
           p->complete ? "" : " (chat only: no model here completes code)");
    return 0;
}

/* VS Code's own chat: a provider of its Custom Endpoint kind */
static int copilot(const struct setup_editor *e, const struct setup_plan *p)
{
    char file[4200];
    snprintf(file, sizeof file, "%s/chatLanguageModels.json", e->user_dir);
    size_t n;
    char *t = slurp(file, &n);
    char item[1024];
    snprintf(item, sizeof item,
             "{\"name\": \"Janas\", \"vendor\": \"customendpoint\", "
             "\"apiKey\": \"janas\", \"apiType\": \"chat-completions\", "
             "\"models\": [{\"id\": \"%s\", \"name\": \"Janas %s\", \"url\": "
             "\"http://127.0.0.1:%d/v1/chat/completions\", \"toolCalling\": "
             "true, \"maxInputTokens\": %u, \"maxOutputTokens\": 8192}]}",
             p->code->name, p->code->name, p->code_port,
             p->ctx > 8192 ? p->ctx - 8192 : p->ctx);
    char *out;
    size_t out_n;
    int rc = janas_jsonc_array_put(t ? t : "", n, "name", "Janas", item, &out,
                                   &out_n);
    if (rc != 0)
        fprintf(stderr,
                "janas-setup: %s is not a list of models as VS Code writes "
                "it: left as it is\n",
                file);
    if (rc == 0 && (note("A", file, "Janas", NULL, 0) != 0 ||
                    spill(file, out, out_n) != 0)) {
        fprintf(stderr, "janas-setup: cannot write %s\n", file);
        rc = -1;
    }
    if (rc == 0)
        printf("   VS Code's chat: %s (model \"Janas %s\")\n", file,
               p->code->name);
    free(out);
    free(t);
    return rc;
}

static void cline_dir(char *buf, size_t len)
{
    const char *d = getenv("CLINE_DATA_DIR"), *c = getenv("CLINE_DIR"),
               *h = getenv("HOME");
    if (d && *d)
        snprintf(buf, len, "%s", d);
    else if (c && *c)
        snprintf(buf, len, "%s/data", c);
    else
        snprintf(buf, len, "%s/.cline/data", h ? h : ".");
}

/* Cline: its own file, in its versions that keep the providers there */
static int cline(const struct setup_plan *p)
{
    char dir[4096], file[4200];
    cline_dir(dir, sizeof dir);
    snprintf(file, sizeof file, "%s/settings/providers.json", dir);
    size_t n, at, len;
    char *t = slurp(file, &n);
    const char *path[] = {"providers", "ollama"};
    int had = t ? janas_jsonc_find(t, n, path, 2, &at, &len) : -1;
    free(t);
    /* Janas's own, from an earlier run, is Janas's to set again */
    if (had == 1 && logged("K", file, "providers|ollama"))
        had = 0;
    if (had != 0) {
        /* none (its settings in the editor's own storage), or an Ollama of
           the user's already set: nothing replaced */
        printf("   Cline: set it in its settings - provider Ollama, base URL "
               "http://127.0.0.1:%d, model %s, context %u%s\n",
               p->code_port, p->code->name, p->ctx,
               had == 1 ? " (its Ollama provider is set already: left as it "
                          "is)"
                        : "");
        return 1;
    }
    char item[1024], when[32];
    time_t now = time(NULL);
    struct tm tm;
#ifdef _WIN32
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    strftime(when, sizeof when, "%Y-%m-%dT%H:%M:%S.000Z", &tm);
    snprintf(item, sizeof item,
             "{\"settings\": {\"provider\": \"ollama\", \"model\": \"%s\", "
             "\"baseUrl\": \"http://127.0.0.1:%d\"}, \"updatedAt\": \"%s\", "
             "\"tokenSource\": \"manual\"}",
             p->code->name, p->code_port, when);
    if (set_member(file, "providers|ollama", item) != 0)
        return -1;
    printf("   Cline: %s (its Ollama provider; restart the editor)\n", file);
    return 0;
}

/* Roo Code imports a file of its profiles when asked: written here, the
   import left to the user, since an import at every start (its
   autoImportSettingsPath) would make Janas its current profile every
   time */
static int roo(const struct setup_plan *p)
{
    char dir[4096], file[4200], text[1024];
    if (setup_conf_dir(dir, sizeof dir) != 0)
        return -1;
    snprintf(file, sizeof file, "%s/roo-code.json", dir);
    int n = snprintf(text, sizeof text,
                     "{\"providerProfiles\": {\"currentApiConfigName\": "
                     "\"Janas\", \"apiConfigs\": {\"Janas\": {\"id\": "
                     "\"janas\", \"apiProvider\": \"ollama\", "
                     "\"ollamaBaseUrl\": \"http://127.0.0.1:%d\", "
                     "\"ollamaModelId\": \"%s\"}}, \"modeApiConfigs\": {}}}\n",
                     p->code_port, p->code->name);
    if (n <= 0 || (size_t)n >= sizeof text ||
        spill(file, text, (size_t)n) != 0) {
        fprintf(stderr, "janas-setup: cannot write %s\n", file);
        return -1;
    }
    printf("   Roo Code: import %s (its settings, Import): a profile "
           "\"Janas\" beside yours, made the one in use\n",
           file);
    return 1;
}

int setup_vscode_write(const struct setup_editor *e, const char *id,
                       const struct setup_plan *p)
{
    if (!strcmp(id, "ggml-org.llama-vscode"))
        return llama_vscode(e, p);
    if (!strcmp(id, "github.copilot-chat"))
        return copilot(e, p);
    if (!strcmp(id, "saoudrizwan.claude-dev"))
        return cline(p);
    if (!strcmp(id, "rooveterinaryinc.roo-cline"))
        return roo(p);
    return 1;
}

int setup_vscode_undo(void)
{
    char log[4200];
    size_t n;
    if (undo_file(log, sizeof log) != 0)
        return 0;
    char *t = slurp(log, &n);
    if (!t)
        return 0;
    int bad = 0;
    /* lines walked by hand: split() below uses strtok */
    for (char *l = t, *next; l && *l; l = next) {
        next = strchr(l, '\n');
        if (next)
            *next++ = 0;
        char *f[4] = {l, NULL, NULL, NULL};
        for (int k = 1; k < 4; k++) {
            char *tab = f[k - 1] ? strchr(f[k - 1], '\t') : NULL;
            if (tab) {
                *tab = 0;
                f[k] = tab + 1;
            }
        }
        if (!f[3])
            continue;
        size_t fn;
        char *text = slurp(f[1], &fn), *out = NULL;
        size_t out_n;
        int r;
        if (!text)
            continue; /* gone: nothing of Janas's left in it */
        if (!strcmp(f[0], "A")) {
            r = janas_jsonc_array_remove(text, fn, "name", f[2], &out, &out_n);
        } else {
            char kb[512];
            const char *path[8];
            snprintf(kb, sizeof kb, "%s", f[2]);
            int np = split(kb, path, 8);
            if (!strcmp(f[3], "-"))
                r = janas_jsonc_remove(text, fn, path, np, &out, &out_n);
            else
                r = janas_jsonc_set(text, fn, path, np, f[3], &out, &out_n) == 0
                        ? 1
                        : -1;
        }
        if (r == 1 && spill(f[1], out, out_n) == 0)
            printf("   %s: put back\n", f[1]);
        else if (r < 0) {
            fprintf(stderr, "janas-setup: %s could not be put back\n", f[1]);
            bad = 1;
        }
        free(out);
        free(text);
    }
    free(t);
    if (!bad)
        remove(log);
    return bad ? -1 : 0;
}
