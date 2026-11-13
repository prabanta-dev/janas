/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_chatfile.c - conversations written down and in files
 * (src/llm/api_transcript.c, api_chatfile.c), without a model: the title
 * from the first question and given, the saved form written and read back
 * whole (reasoning, calls, title), the Markdown export read back with its
 * reasoning and calls, the plain text without, a Markdown written by hand,
 * what is not a conversation, saving as it goes (a file made at the first
 * question, a new one after a reset), the list newest first and what a file
 * says of itself. In a directory of its own under $TMPDIR (else /tmp).
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "llm/chat.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static char dir[512];

/* A chat with no model behind it: what the record and the files use. */
static janas_llm_chat *fake(janas_llm *llm)
{
    memset(llm, 0, sizeof(*llm));
    snprintf(llm->name, sizeof llm->name, "Tiny-Test");
    llm->path = "/models/tiny.jns";
    janas_llm_chat *c = calloc(1, sizeof(*c));
    c->llm = llm;
    return c;
}

static void push(janas_llm_chat *c, int32_t role, const char *text,
                 const char *why)
{
    struct tmsg *m = janas_api_tr_push(c, role, text, strlen(text));
    if (m && why)
        m->reasoning = strdup(why);
}

static void fill(janas_llm_chat *c)
{
    janas_api_tr_system(c, "You are terse.");
    push(c, JANAS_LLM_ROLE_USER,
         "How is bread made at home, with flour, water, salt and yeast, "
         "without a machine?\nAnd how long does it rise?",
         NULL);
    push(c, JANAS_LLM_ROLE_ASSISTANT, "Mix, knead, let it rise, bake.",
         "The user wants steps.\n\nShort ones.");
    push(c, JANAS_LLM_ROLE_USER, "What is the weather in Cagliari?", NULL);
    push(c, JANAS_LLM_ROLE_ASSISTANT, "", NULL);
    struct tmsg *m = &c->tr[c->n_tr - 1];
    const char *args = "{\"city\":\"Cagliari\",\"note\":\"```quoted```\"}";
    janas_api_tr_call(m, "weather", 7, args, strlen(args));
    push(c, JANAS_LLM_ROLE_TOOL, "{\"temp\":21}", NULL);
    push(c, JANAS_LLM_ROLE_ASSISTANT, "21 degrees.\n\n## A heading inside",
         NULL);
}

static char *text_of(janas_llm_chat *c, int32_t i, int32_t what, int32_t *role)
{
    static char buf[4][1024];
    static int k;
    char *b = buf[k++ & 3];
    int32_t len;
    if (janas_llm_chat_message(c, i, what, role, b, 1024, &len) != JANAS_LLM_OK)
        snprintf(b, 1024, "(error)");
    return b;
}

/* The same messages, the reasoning and calls too unless plain. */
static void same(janas_llm_chat *a, janas_llm_chat *b, int plain,
                 const char *what)
{
    int32_t n = janas_llm_chat_messages(a);
    CHECK(janas_llm_chat_messages(b) == n, "%s: %d messages, not %d", what,
          (int)janas_llm_chat_messages(b), (int)n);
    for (int32_t i = 0; i < n && i < janas_llm_chat_messages(b); i++) {
        int32_t ra, rb;
        char *ta = text_of(a, i, JANAS_LLM_MESSAGE_TEXT, &ra);
        char *tb = text_of(b, i, JANAS_LLM_MESSAGE_TEXT, &rb);
        CHECK(ra == rb, "%s: message %d role %d, not %d", what, (int)i, (int)rb,
              (int)ra);
        CHECK(!strcmp(ta, tb), "%s: message %d text [%s], not [%s]", what,
              (int)i, tb, ta);
        if (plain)
            continue;
        ta = text_of(a, i, JANAS_LLM_MESSAGE_REASONING, NULL);
        tb = text_of(b, i, JANAS_LLM_MESSAGE_REASONING, NULL);
        CHECK(!strcmp(ta, tb), "%s: message %d reasoning [%s], not [%s]", what,
              (int)i, tb, ta);
        ta = text_of(a, i, JANAS_LLM_MESSAGE_CALLS, NULL);
        tb = text_of(b, i, JANAS_LLM_MESSAGE_CALLS, NULL);
        CHECK(!strcmp(ta, tb), "%s: message %d calls %s, not %s", what, (int)i,
              tb, ta);
    }
}

static int read_back(janas_llm_chat *c, const char *path, int *saved)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    static char s[65536];
    size_t n = fread(s, 1, sizeof s - 1, f);
    fclose(f);
    s[n] = 0;
    janas_api_tr_clear(c);
    return janas_api_tr_read(c, s, n, path, saved);
}

static void write_text(const char *path, const char *s)
{
    FILE *f = fopen(path, "wb");
    if (f) {
        fputs(s, f);
        fclose(f);
    }
}

/* A folder and what is in it, one level of folders below. */
static void rm_tree(const char *d)
{
    DIR *dd = opendir(d);
    struct dirent *e;
    while (dd && (e = readdir(dd))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        char p[1200];
        snprintf(p, sizeof p, "%s/%s", d, e->d_name);
        if (remove(p) != 0)
            rm_tree(p);
    }
    if (dd)
        closedir(dd);
    rmdir(d);
}

static int files_in(const char *d)
{
    char list[8192];
    int32_t len;
    if (janas_llm_chats_list(d, list, sizeof list, &len) != JANAS_LLM_OK)
        return -1;
    int k = 0;
    for (char *p = list; *p; p++)
        k += *p == '\n';
    return k;
}

int main(void)
{
    const char *t = getenv("TMPDIR");
    snprintf(dir, sizeof dir, "%s/janas-chatfile-test-%d", t && *t ? t : "/tmp",
             (int)getpid());
    mkdir(dir, 0700);
    janas_llm la, lb;
    janas_llm_chat *a = fake(&la), *b = fake(&lb);
    fill(a);
    char title[128], path[700];
    int32_t len;
    janas_llm_chat_title(a, title, sizeof title, &len);
    CHECK(!strcmp(title, "How is bread made at home, with flour, water, salt "
                         "and"),
          "title from the first question: [%s]", title);
    CHECK(janas_llm_chat_messages(a) == 7, "%d messages, not 7",
          (int)janas_llm_chat_messages(a));
    CHECK(!strcmp(text_of(a, 4, JANAS_LLM_MESSAGE_CALLS, NULL),
                  "[{\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":"
                  "\\\"Cagliari\\\",\\\"note\\\":\\\"```quoted```\\\"}\"}]"),
          "calls as JSON: %s", text_of(a, 4, JANAS_LLM_MESSAGE_CALLS, NULL));

    /* the saved form, whole */
    snprintf(path, sizeof path, "%s/one.jsonl", dir);
    CHECK(janas_llm_chat_save(a, path) == JANAS_LLM_OK, "save: %s",
          janas_llm_last_error());
    int saved = 0;
    CHECK(read_back(b, path, &saved) == 0 && saved, "the saved form not read");
    same(a, b, 0, "saved");
    janas_llm_chat_title(b, title, sizeof title, &len);
    CHECK(!b->tr_titled && !strncmp(title, "How is bread", 12),
          "saved title [%s]", title);
    janas_llm_chat_set_title(a, "Bread and weather", -1);
    janas_llm_chat_save(a, path);
    read_back(b, path, &saved);
    janas_llm_chat_title(b, title, sizeof title, &len);
    CHECK(b->tr_titled && !strcmp(title, "Bread and weather"),
          "a title given, read back as [%s]", title);
    char tn[64] = "", mn[64] = "", mf[64] = "";
    int64_t upd = 0;
    int32_t nm = -1;
    CHECK(janas_llm_chat_file_info(path, tn, sizeof tn, mn, sizeof mn, mf,
                                   sizeof mf, &upd, &nm) == JANAS_LLM_OK &&
              !strcmp(tn, "Bread and weather") && !strcmp(mn, "Tiny-Test") &&
              !strcmp(mf, "/models/tiny.jns") && upd > 0 && nm == 6,
          "file info: [%s] [%s] %lld %d", tn, mn, (long long)upd, (int)nm);
    char cut[8];
    janas_llm_chat_file_info(path, cut, sizeof cut, NULL, 0, NULL, 0, NULL,
                             NULL);
    CHECK(!strcmp(cut, "Bread a"), "title cut to its buffer: [%s]", cut);

    /* Markdown, with its reasoning and calls */
    snprintf(path, sizeof path, "%s/one.md", dir);
    CHECK(janas_llm_chat_export(a, path, JANAS_LLM_EXPORT_MARKDOWN) ==
              JANAS_LLM_OK,
          "export md: %s", janas_llm_last_error());
    CHECK(read_back(b, path, &saved) == 0 && !saved, "the Markdown not read");
    same(a, b, 0, "markdown");
    janas_llm_chat_title(b, title, sizeof title, &len);
    CHECK(!strcmp(title, "Bread and weather"), "markdown title [%s]", title);

    /* plain text: the messages and calls, no reasoning */
    snprintf(path, sizeof path, "%s/one.txt", dir);
    CHECK(janas_llm_chat_export(a, path, JANAS_LLM_EXPORT_TEXT) == JANAS_LLM_OK,
          "export text: %s", janas_llm_last_error());
    CHECK(read_back(b, path, &saved) == 0, "the text not read");
    same(a, b, 1, "text");
    CHECK(!strcmp(text_of(b, 4, JANAS_LLM_MESSAGE_CALLS, NULL),
                  text_of(a, 4, JANAS_LLM_MESSAGE_CALLS, NULL)),
          "text calls %s", text_of(b, 4, JANAS_LLM_MESSAGE_CALLS, NULL));

    /* Markdown by hand: by its headings */
    snprintf(path, sizeof path, "%s/hand.md", dir);
    write_text(path, "# Notes\n\nSome words first.\n\n## User\n\nHello?\n\n"
                     "## Assistant\n\nHi.\nTwo lines.\n\n## user\n\nBye.\n");
    CHECK(read_back(b, path, &saved) == 0 && janas_llm_chat_messages(b) == 3,
          "hand-written markdown: %d messages",
          (int)janas_llm_chat_messages(b));
    CHECK(
        !strcmp(text_of(b, 1, JANAS_LLM_MESSAGE_TEXT, NULL), "Hi.\nTwo lines."),
        "hand-written reply [%s]", text_of(b, 1, JANAS_LLM_MESSAGE_TEXT, NULL));
    janas_llm_chat_title(b, title, sizeof title, &len);
    CHECK(!strcmp(title, "Notes"), "hand-written title [%s]", title);

    /* not a conversation */
    snprintf(path, sizeof path, "%s/none.txt", dir);
    write_text(path, "just some words\nand more\n");
    CHECK(read_back(b, path, &saved) != 0 && janas_llm_chat_messages(b) == 0,
          "words that are no conversation were read as one");

    /* saved as it goes */
    char auto_dir[600];
    snprintf(auto_dir, sizeof auto_dir, "%s/auto", dir);
    janas_api_tr_clear(b);
    CHECK(janas_llm_chat_autosave(b, auto_dir) == JANAS_LLM_OK, "autosave: %s",
          janas_llm_last_error());
    janas_api_tr_system(b, "Be kind.");
    janas_api_tr_saved(b);
    CHECK(files_in(auto_dir) == 0, "a file before the first question");
    janas_api_tr_turn(b, "First?", 6, 0, NULL, NULL);
    CHECK(files_in(auto_dir) == 1, "%d files after the first question",
          files_in(auto_dir));
    char file1[700];
    janas_llm_chat_file(b, file1, sizeof file1, &len);
    janas_api_tr_turn(b, "Second?", 7, 0, NULL, NULL);
    CHECK(janas_llm_chat_file_info(file1, NULL, 0, NULL, 0, NULL, 0, NULL,
                                   &nm) == JANAS_LLM_OK &&
              nm == 2,
          "the file holds %d messages, not 2", (int)nm);
    janas_api_tr_clear(b); /* as a reset does */
    b->tr_created = 1;     /* another time, another name */
    janas_api_tr_turn(b, "Another?", 8, 0, NULL, NULL);
    CHECK(files_in(auto_dir) == 2, "%d files after a new conversation",
          files_in(auto_dir));
    char list[4096];
    janas_llm_chats_list(auto_dir, list, sizeof list, &len);
    CHECK(strstr(list, file1) != NULL, "the list [%s] lacks %s", list, file1);

    janas_api_tr_free(a);
    janas_api_tr_free(b);
    free(a);
    free(b);
    rm_tree(dir);
    if (failures) {
        printf("test_chatfile: %d failures\n", failures);
        return 1;
    }
    printf("test_chatfile: ok\n");
    return 0;
}
