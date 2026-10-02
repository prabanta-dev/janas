/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_layout.c - a layout put into another language by janas-chat,
 * without a model: a translator that gives the text back as it came, so
 * that what the program puts on its own shows: the tags and keys back in
 * place, the decimal comma, the terms of the glossary (never asked of the
 * model) and the file kept, in a directory of its own under ~/tmp.
 * The sources belong to the program, so they are compiled in here.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "chat/glossary.c"
#include "chat/layout.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char EN[] = "Wind {{wind}} km/h ({{bft|bft}}), sea "
                         "{{wave}} m ({{code|sea}}).\n"
                         "---\n"
                         "bft.1 = light air\n"
                         "bft.4 = moderate breeze\n"
                         "sea.2 = smooth\n"
                         "wd.Mon = Mon\n";

static char asked[4096];
static int calls;

/* The text back as it came: "Translate into ...:\n\n" cut off. */
static int echo(const char *name, const char *system, const char *text,
                char **out)
{
    (void)name;
    (void)system;
    calls++;
    snprintf(asked, sizeof asked, "%s", text);
    const char *t = strstr(text, "\n\n");
    *out = strdup(t ? t + 2 : text);
    return *out ? 0 : -1;
}

int main(void)
{
    CHECK(strcmp(glossary_word("it", "bft.1", 5), "bava di vento") == 0 &&
              strcmp(glossary_word("it", "sea.4", 5), "molto mosso") == 0 &&
              strcmp(glossary_word("es", "bft.12", 6), "temporal huracanado") ==
                  0,
          "the glossary's terms");
    CHECK(!glossary_word("it", "bft.13", 6) &&
              !glossary_word("it", "sea.", 4) &&
              !glossary_word("it", "sea.1x", 6) &&
              !glossary_word("it", "wd.Mon", 6) &&
              !glossary_word("en", "bft.1", 5) &&
              !glossary_word("xx", "bft.1", 5),
          "words the glossary has not");
    CHECK(glossary_mark("it", 1) != glossary_mark("fr", 1) &&
              glossary_mark("en", 1) == 1,
          "the glossary's mark");

    char dir[512];
    const char *home = getenv("HOME");
    snprintf(dir, sizeof dir, "%s/tmp/janas-layout-test-%d", home ? home : ".",
             (int)getpid());
    if (mkdir(dir, 0700) != 0) {
        printf("cannot make %s\n", dir);
        return 1;
    }
    setenv("XDG_CONFIG_HOME", dir, 1);
    layout_set_lang("it");
    layout_set_translator(echo);

    char *it = translated("test", EN);
    CHECK(it != NULL, "no layout");
    if (it) {
        CHECK(strstr(it, "Wind {{wind}} km/h ({{bft|bft}}), sea {{wave}} m "
                         "({{code|sea}}).") == it,
              "the tags back in place:\n%s", it);
        CHECK(strstr(it, "bft.1 = bava di vento\n") &&
                  strstr(it, "bft.4 = vento moderato\n") &&
                  strstr(it, "sea.2 = poco mosso\n") &&
                  strstr(it, "wd.Mon = Mon\n") && strstr(it, "decimal = ,\n"),
              "the words:\n%s", it);
    }
    CHECK(strstr(asked, "Italian, with these terms: swell = mare lungo") &&
              strstr(asked, ": Mon   [day of the week Mon]") &&
              !strstr(asked, "light air") && !strstr(asked, "smooth"),
          "the request:\n%s", asked);

    /* the second time, from the file */
    char *again = translated("test", EN);
    CHECK(calls == 1 && again && it && strcmp(again, it) == 0,
          "the layout kept (%d requests)", calls);
    free(again);
    free(it);

    /* lines added by the model between two tags are not kept: those
       between tags alone come from English */
    {
        static const char LEGS[] = "Trips:\n{{#legs}}\n- {{to}}\n{{/legs}}\n"
                                   "{{#cut}} and {{cut}} more{{/cut}}\n";
        struct coded c;
        CHECK(encode(LEGS, &c) == 0, "encode");
        char why[200] = "";
        char *d = decode(&c,
                         "Viaggi:\n{1}\n\n- {2}\n{3}\n\n\n{4} e altri {5} "
                         "{6}\n=== words\n",
                         why, sizeof why);
        CHECK(d && strcmp(d, "Viaggi:\n{{#legs}}\n- {{to}}\n{{/legs}}\n"
                             "{{#cut}} e altri {{cut}} {{/cut}}\n---\n"
                             "decimal = ,\n") == 0,
              "the lines between tags:\n%s%s", d ? d : "NULL ", why);
        free(d);
        coded_free(&c);
    }

    /* tidy up: the one file and the two directories made here */
    char path[1200];
    snprintf(path, sizeof path, "%s/janas/layouts/test.it.%016llx.txt", dir,
             (unsigned long long)glossary_mark("it", mark(EN)));
    CHECK(unlink(path) == 0, "no file at %s", path);
    snprintf(path, sizeof path, "%s/janas/layouts", dir);
    rmdir(path);
    snprintf(path, sizeof path, "%s/janas", dir);
    rmdir(path);
    rmdir(dir);

    if (failures) {
        printf("test_layout: %d failures\n", failures);
        return 1;
    }
    printf("test_layout: ok\n");
    return 0;
}
