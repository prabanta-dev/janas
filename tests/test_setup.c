/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_setup.c - janas-setup editor (src/setup/): the models proposed for
 * code and completion as memory allows, Continue's file, the settings of
 * llama.vscode, Cline and VS Code's chat written and put back byte for
 * byte, and on Linux an editor's extensions read from its command line (a
 * script standing for it). Everything in a directory of its own under
 * $TMPDIR (else /tmp).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "setup/continue_cfg.c"
#include "setup/editors.c"
#include "setup/plan.c"
#include "setup/vscode_cfg.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

#define GB (1000ull * 1000 * 1000)

static void mk(struct janas_model_file *m, const char *name, double gb,
               int coder, int fim, int recurrent)
{
    memset(m, 0, sizeof *m);
    snprintf(m->name, sizeof m->name, "%s", name);
    snprintf(m->path, sizeof m->path, "/models/%s.jns", name);
    m->bytes = (uint64_t)(gb * (double)GB);
    m->coder = coder;
    m->fim = fim;
    m->recurrent = recurrent;
}

static void proposals(void)
{
    struct janas_model_file m[4];
    mk(&m[0], "coder-30b", 18.6, 1, 1, 0);
    mk(&m[1], "coder-1.5b", 1.0, 1, 1, 0);
    mk(&m[2], "general-4b", 2.5, 0, 1, 0);
    mk(&m[3], "coder-next", 45.0, 1, 1, 1);
    struct setup_plan p;

    /* 32 GB: the largest coder that fits, and the small one beside it */
    setup_plan_propose(&p, m, 4, 32 * GB, NULL);
    CHECK(p.code == &m[0] && p.complete == &m[1] &&
              p.complete_port == p.code_port + 1,
          "32 GB: %s and %s", p.code ? p.code->name : "-",
          p.complete ? p.complete->name : "-");
    /* 26 GB: the small one does not fit beside it - the same model */
    setup_plan_propose(&p, m, 3, 26 * GB, NULL);
    CHECK(p.code == &m[0] && p.complete == &m[0] &&
              p.complete_port == p.code_port,
          "26 GB: %s and %s", p.code ? p.code->name : "-",
          p.complete ? p.complete->name : "-");
    /* 64 GB: the 45 GB coder, recurrent: completion from the small one */
    setup_plan_propose(&p, m, 4, 64 * GB, NULL);
    CHECK(p.code == &m[3] && p.complete == &m[1], "64 GB: %s and %s",
          p.code ? p.code->name : "-", p.complete ? p.complete->name : "-");
    /* 8 GB: the coders that fit */
    setup_plan_propose(&p, m, 4, 8 * GB, NULL);
    CHECK(p.code == &m[1] && p.complete == &m[1], "8 GB: %s and %s",
          p.code ? p.code->name : "-", p.complete ? p.complete->name : "-");
    /* a general model alone: code, no completion (no coder to fill in) */
    setup_plan_propose(&p, &m[2], 1, 32 * GB, NULL);
    CHECK(p.code == &m[2] && !p.complete, "general alone: %s and %s",
          p.code ? p.code->name : "-", p.complete ? p.complete->name : "-");
    /* named: the code model, completion proposed around it */
    struct setup_choice c = {.code = &m[1]};
    setup_plan_propose(&p, m, 4, 32 * GB, &c);
    CHECK(p.code == &m[1] && p.complete == &m[1] &&
              p.complete_port == p.code_port,
          "named code: %s and %s", p.code ? p.code->name : "-",
          p.complete ? p.complete->name : "-");
    c = (struct setup_choice){.code = &m[0], .complete = &m[1]};
    setup_plan_propose(&p, m, 4, 32 * GB, &c);
    CHECK(p.code == &m[0] && p.complete == &m[1] &&
              p.complete_port == p.code_port + 1,
          "named both");
    c = (struct setup_choice){.no_complete = 1};
    setup_plan_propose(&p, m, 4, 32 * GB, &c);
    CHECK(p.code == &m[0] && !p.complete, "no completion");
    CHECK(!janas_model_completes(&m[3]) && !janas_model_completes(&m[2]) &&
              janas_model_completes(&m[0]),
          "what completes");
}

static char *read_all(const char *file)
{
    size_t n;
    return slurp(file, &n);
}

static void put(const char *file, const char *text)
{
    FILE *f = fopen(file, "wb");
    if (f) {
        fputs(text, f);
        fclose(f);
    }
}

static void settings(const char *dir)
{
    char path[1024], file[4200], cline_file[1100], clm[4200];
    snprintf(path, sizeof path, "%s/config", dir);
    setenv("XDG_CONFIG_HOME", path, 1);
    snprintf(path, sizeof path, "%s/continue", dir);
    setenv("CONTINUE_GLOBAL_DIR", path, 1);
    snprintf(path, sizeof path, "%s/cline", dir);
    setenv("CLINE_DATA_DIR", path, 1);

    struct janas_model_file m[1];
    mk(&m[0], "coder", 1.0, 1, 1, 0);
    struct setup_plan p = {.code = &m[0],
                           .complete = &m[0],
                           .code_port = 11435,
                           .complete_port = 11435,
                           .ctx = 32768};

    /* Continue: a file of Janas's, then none */
    struct setup_served s = {.name = "coder", .port = 11435, .ctx = 32768};
    const char *note;
    CHECK(setup_continue_write(&s, &s, &note) == 0 && !note, "continue write");
    snprintf(file, sizeof file, "%s/continue/models/janas.yaml", dir);
    char *y = read_all(file);
    CHECK(
        y && strstr(y, "roles: [chat, edit, apply, summarize, autocomplete]") &&
            strstr(y, "apiBase: \"http://127.0.0.1:11435\"") &&
            strstr(y, "modelTimeout: 20000"),
        "continue file: %s", y ? y : "(none)");
    free(y);
    CHECK(setup_continue_remove() == 1 && access(file, F_OK) != 0,
          "continue remove");
    /* only the old config.json: said */
    snprintf(file, sizeof file, "%s/continue/config.json", dir);
    put(file, "{}");
    CHECK(setup_continue_write(&s, NULL, &note) == 0 && note,
          "config.json not said");
    setup_continue_remove();
    remove(file);

    /* llama.vscode, Cline, VS Code's chat: written, then put back */
    struct setup_editor e = {.name = "VS Code"};
    snprintf(e.user_dir, sizeof e.user_dir, "%s/User", dir);
    janas_mkdirs(e.user_dir);
    snprintf(path, sizeof path, "%s/cline/settings", dir);
    janas_mkdirs(path);
    const char *orig_settings =
        "{\n    // mine\n    \"editor.fontSize\": 14, "
        "/* big */\n    \"llama-vscode.endpoint_chat\": "
        "\"http://mine:9\",\n}\n";
    const char *orig_cline =
        "{\"version\": 1, \"providers\": {\"anthropic\": "
        "{\"settings\": {\"provider\": \"anthropic\"}}}}\n";
    const char *orig_clm = "[\n  {\"name\": \"Mine\", \"models\": []}\n]\n";
    snprintf(file, sizeof file, "%s/settings.json", e.user_dir);
    snprintf(cline_file, sizeof cline_file, "%s/providers.json", path);
    snprintf(clm, sizeof clm, "%s/chatLanguageModels.json", e.user_dir);
    put(file, orig_settings);
    put(cline_file, orig_cline);
    put(clm, orig_clm);
    for (int run = 0; run < 2; run++) {
        CHECK(setup_vscode_write(&e, "ggml-org.llama-vscode", &p) == 0,
              "llama.vscode write %d", run);
        CHECK(setup_vscode_write(&e, "saoudrizwan.claude-dev", &p) == 0,
              "cline write %d", run);
        CHECK(setup_vscode_write(&e, "github.copilot-chat", &p) == 0,
              "copilot write %d", run);
    }
    char *t = read_all(file);
    CHECK(t &&
              strstr(t,
                     "\"llama-vscode.endpoint\": \"http://127.0.0.1:11435\"") &&
              strstr(t, "// mine"),
          "settings: %s", t ? t : "(none)");
    free(t);
    t = read_all(cline_file);
    CHECK(t &&
              strstr(t, "\"ollama\": {\"settings\": {\"provider\": \"ollama\""),
          "cline: %s", t ? t : "(none)");
    free(t);
    t = read_all(clm);
    CHECK(t && strstr(t, "\"name\": \"Janas\"") && strstr(t, "\"Mine\""),
          "chat models: %s", t ? t : "(none)");
    free(t);
    CHECK(setup_vscode_undo() == 0, "undo");
    t = read_all(file);
    CHECK(t && strcmp(t, orig_settings) == 0, "settings back: %s",
          t ? t : "(none)");
    free(t);
    t = read_all(cline_file);
    CHECK(t && strcmp(t, orig_cline) == 0, "cline back: %s", t ? t : "(none)");
    free(t);
    t = read_all(clm);
    CHECK(t && strcmp(t, orig_clm) == 0, "chat models back: %s",
          t ? t : "(none)");
    free(t);

    /* a broken settings.json is left as it is */
    put(file, "{\"a\": ");
    CHECK(setup_vscode_write(&e, "ggml-org.llama-vscode", &p) < 0,
          "broken settings written");
    t = read_all(file);
    CHECK(t && strcmp(t, "{\"a\": ") == 0, "broken settings changed");
    free(t);
    setup_vscode_undo();
    remove(file);
    remove(cline_file);
    remove(clm);
}

#ifndef _WIN32
/* an editor's command line, a script that says what is installed */
static void editors(const char *dir)
{
    char bin[1024], cli[1100], path[2200];
    snprintf(bin, sizeof bin, "%s/bin", dir);
    janas_mkdirs(bin);
    snprintf(cli, sizeof cli, "%s/codium", bin);
    put(cli, "#!/bin/sh\nfor a in \"$@\"; do [ \"$a\" = --list-extensions ] && "
             "{ echo Continue.continue@2.1.0; echo Some.Other@1.0; exit 0; "
             "}; done\nexit 1\n");
    chmod(cli, 0755);
    const char *old = getenv("PATH");
    char *keep = old ? strdup(old) : NULL;
    snprintf(path, sizeof path, "%s", bin);
    setenv("PATH", path, 1);
    static struct setup_editor e[4];
    int n = setup_editors_find(e, 4, "/nowhere/data", NULL);
    CHECK(n == 1 && !strcmp(e[0].name, "VSCodium") && e[0].open_vsx,
          "editors found: %d", n);
    if (n == 1) {
        const char *v = setup_editor_has(&e[0], "continue.continue");
        CHECK(v && !strcmp(v, "2.1.0"), "continue version: %s", v ? v : "-");
        CHECK(setup_editor_has(&e[0], "some.other") &&
                  !setup_editor_has(&e[0], "saoudrizwan.claude-dev"),
              "other extensions");
        CHECK(!strcmp(e[0].user_dir, "/nowhere/data/User"), "user dir: %s",
              e[0].user_dir);
    }
    if (keep)
        setenv("PATH", keep, 1);
    free(keep);
    remove(cli);
    rmdir(bin);
}
#endif

int main(void)
{
    proposals();
    char dir[512];
    const char *tmp = getenv("TMPDIR");
    snprintf(dir, sizeof dir, "%s/janas_test_setup_XXXXXX",
             tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(dir)) {
        printf("FAIL: cannot make a directory in %s\n", dir);
        return 1;
    }
    settings(dir);
#ifndef _WIN32
    editors(dir);
#endif
    /* what the run made, then the directory */
    static const char *const made[] = {
        "continue/models", "continue", "cline/settings", "cline", "User",
        "config/janas",    "config"};
    for (size_t i = 0; i < sizeof made / sizeof *made; i++) {
        char p[700];
        snprintf(p, sizeof p, "%s/%s", dir, made[i]);
        rmdir(p);
    }
    CHECK(rmdir(dir) == 0, "%s left with files in it", dir);
    if (failures) {
        printf("test_setup: %d FAILED\n", failures);
        return 1;
    }
    printf("test_setup: all passed\n");
    return 0;
}
