/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_template.c - layouts filled with data (src/services/common/template.c):
 * fields, words, lists and their numbers, parts shown or not, lines of a
 * part's tag left out, the decimal point of the words, what is wrong in a
 * layout, and a translation's shape against its original.
 */
#include <stdio.h>
#include <string.h>

#include "services/common/template.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char DATA[] =
    "{\"sea\": \"Tyrrhenian Sea\", \"count\": 3, \"temp\": 21.7, \"none\": 0,"
    " \"empty\": \"\", \"tags\": [\"a\", \"b\"], \"aircraft\": [{\"call\": "
    "\"AZ1\", \"dir\": \"NE\", \"route\": {\"from\": \"Rome\", \"to\": "
    "\"Milan\"}}, {\"call\": \"FR2\", \"dir\": \"SW\"}, {\"call\": \"U23\", "
    "\"dir\": \"N\", \"emergency\": \"7700\"}]}";

/* The layout filled with DATA, or "ERROR: why". */
static const char *fill(const char *tpl)
{
    static char text[2048];
    struct janas_json_doc *d = janas_json_parse(DATA, strlen(DATA), NULL, 0);
    struct janas_buf out = {0};
    char err[200];
    if (!d || janas_tpl_render(tpl, strlen(tpl), janas_json_root(d), &out, err,
                               sizeof err) != 0)
        snprintf(text, sizeof text, "ERROR: %s", d ? err : "data");
    else
        snprintf(text, sizeof text, "%.*s", (int)out.n, out.p ? out.p : "");
    janas_buf_free(&out);
    janas_json_free(d);
    return text;
}

#define SAME(tpl, want)                                                        \
    do {                                                                       \
        const char *got = fill(tpl);                                           \
        CHECK(strcmp(got, want) == 0, "%s\n--- got\n%s\n--- wanted\n%s", tpl,  \
              got, want);                                                      \
    } while (0)

static void test_fill(void)
{
    SAME("{{count}} over the {{sea}}.", "3 over the Tyrrhenian Sea.");
    SAME("{{temp}} C\n---\ndecimal = ,", "21,7 C");
    SAME("[{{missing}}][{{none}}]", "[][0]");
    SAME("{{#aircraft}}{{@n}}. {{call}} {{dir|dir}}; {{/aircraft}}\n---\n"
         "dir.NE = nord-est\ndir.SW = sud-ovest",
         "1. AZ1 nord-est; 2. FR2 sud-ovest; 3. U23 N; ");
    SAME("{{#aircraft}}{{call}}{{#route}} {{from}}-{{to}}{{/route}}"
         "{{^route}} -{{/route}}{{#emergency}} SOS {{emergency}}"
         "{{/emergency}},{{/aircraft}}",
         "AZ1 Rome-Milan,FR2 -,U23 - SOS 7700,");
    SAME("{{#tags}}<{{.}}>{{/tags}}", "<a><b>");
    SAME("{{^none}}no{{/none}}{{^empty}} empty{{/empty}}{{#count}} "
         "some{{/count}}",
         "no empty some");
    /* a line of a part's tag alone is left out whole */
    SAME("List:\n{{#aircraft}}\n- {{call}}\n{{/aircraft}}\nEnd.",
         "List:\n- AZ1\n- FR2\n- U23\nEnd.");
    SAME("{{#aircraft}}{{#route}}{{sea}}: {{route.from}}{{/route}}"
         "{{/aircraft}}",
         "Tyrrhenian Sea: Rome");
    SAME("a word with a new line: {{dir|w}}\n---\nw = x\n# a note\n"
         "w.3 = no",
         "a word with a new line: ");
    SAME("{{count|n}}\n---\nn.3 = three\\nlines", "three\nlines");
    CHECK(strncmp(fill("{{#aircraft}}open"), "ERROR", 5) == 0,
          "a part not closed");
    CHECK(strncmp(fill("{{#aircraft}}x{{/route}}"), "ERROR", 5) == 0,
          "closed by another name");
    CHECK(strncmp(fill("x {{count"), "ERROR", 5) == 0, "a tag not ended");
    CHECK(strncmp(fill("{{/count}}"), "ERROR", 5) == 0, "a close alone");
}

static void test_shape(void)
{
    const char *en = "{{count}} aircraft over the {{sea}}:\n{{#aircraft}}\n"
                     "{{@n}}. {{call}} heading {{dir|dir}}\n{{/aircraft}}\n"
                     "---\ndir.NE = north-east\ndir.SW = south-west";
    const char *it = "Sul {{sea}} volano {{count}} aerei:\n{{#aircraft}}\n"
                     "{{@n}}. {{call}} verso {{dir|dir}}\n{{/aircraft}}\n"
                     "---\ndir.SW = sud-ovest\ndir.NE = nord-est\ndir.X "
                     "= y\ndecimal = ,";
    char why[200];
    CHECK(!janas_tpl_same_shape(en, strlen(en), it, strlen(it), why,
                                sizeof why) &&
              strstr(why, "words"),
          "a word more: %s", why);
    const char *it2 = "Sul {{sea}} volano {{count}} aerei:\n{{#aircraft}}\n"
                      "{{@n}}. {{call}} verso {{dir|dir}}\n{{/aircraft}}\n"
                      "---\ndir.SW = sud-ovest\ndir.NE = nord-est";
    CHECK(
        janas_tpl_same_shape(en, strlen(en), it2, strlen(it2), why, sizeof why),
        "fields moved, words in another order: %s", why);
    const char *it3 = "Sul {{sea}} volano {{count}} aerei:\n{{#aircraft}}\n"
                      "{{@n}}. {{call}} verso {{dir|dir}}\n{{/aircraft}}\n"
                      "---\ndir.SW = sud-ovest\ndir.NE = nord-est\n"
                      "decimal = ,";
    CHECK(
        janas_tpl_same_shape(en, strlen(en), it3, strlen(it3), why, sizeof why),
        "a decimal point added: %s", why);
    const char *bad = "Sul {{mare}} volano {{count}} aerei:\n{{#aircraft}}\n"
                      "{{@n}}. {{call}} verso {{dir|dir}}\n{{/aircraft}}\n"
                      "---\ndir.SW = sud-ovest\ndir.NE = nord-est";
    CHECK(!janas_tpl_same_shape(en, strlen(en), bad, strlen(bad), why,
                                sizeof why) &&
              strstr(why, "fields"),
          "a field renamed: %s", why);
    const char *lost = "Sul {{sea}} volano {{count}} aerei.\n---\n"
                       "dir.SW = sud-ovest\ndir.NE = nord-est";
    CHECK(!janas_tpl_same_shape(en, strlen(en), lost, strlen(lost), why,
                                sizeof why),
          "a part lost");
}

int main(void)
{
    test_fill();
    test_shape();
    if (failures) {
        printf("test_template: %d failures\n", failures);
        return 1;
    }
    printf("test_template: ok\n");
    return 0;
}
