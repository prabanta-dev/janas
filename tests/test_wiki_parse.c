/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_wiki_parse.c - janas-wiki's reading of Wikipedia's answers, without
 * the network: a search and a geosearch as they came on 1 October 2026
 * (cut to three pages, given by page id and not in their order), a page
 * reached through a redirect, a disambiguation page, a missing one and an
 * error of the API; and a page's text cut by its headings.
 * The sources belong to the program, so they are compiled in here.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "services/wiki/read.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char SEARCH[] =
    "{\"batchcomplete\": true, \"continue\": {\"gsroffset\": 3, "
    "\"continue\": \"gsroffset||\"}, \"query\": {\"pages\": [{\"pageid\": "
    "4173, \"ns\": 0, \"title\": \"Trapani\", \"index\": 2, \"coordinates\": "
    "[{\"lat\": 38.0175, \"lon\": 12.515, \"primary\": true, \"globe\": "
    "\"earth\"}], \"description\": \"comune italiano\", \"extract\": "
    "\"Trapani \\u00e8 un comune italiano di 54 546 abitanti.\"}, "
    "{\"pageid\": 232190, \"ns\": 0, \"title\": \"I promessi sposi (film "
    "1941)\", \"index\": 3, \"description\": \"film del 1941 diretto da "
    "Mario Camerini\", \"extract\": \"I promessi sposi \\u00e8 un film del "
    "1941.\"}, {\"pageid\": 2719455, \"ns\": 0, \"title\": \"I promessi "
    "sposi\", \"index\": 1, \"extract\": \"I promessi sposi \\u00e8 un "
    "romanzo storico di Alessandro Manzoni.\"}]}}";

static const char NEAR[] =
    "{\"batchcomplete\": true, \"query\": {\"pages\": [{\"pageid\": 1, "
    "\"title\": \"Castello di Terra\", \"index\": 5, \"coordinates\": "
    "[{\"lat\": 38.019264, \"lon\": 12.514415}], \"description\": "
    "\"castello di Trapani, Italia\"}, {\"pageid\": 2, \"title\": "
    "\"Trapani\", \"index\": -1, \"coordinates\": [{\"lat\": 38.0175, "
    "\"lon\": 12.515}]}, {\"pageid\": 3, \"title\": \"Palazzo D'Al\\u00ec\", "
    "\"index\": 2, \"coordinates\": [{\"lat\": 38.01856, \"lon\": "
    "12.5136}]}]}}";

static const char PAGE[] =
    "{\"batchcomplete\": true, \"query\": {\"redirects\": [{\"from\": "
    "\"Manzoni\", \"to\": \"Alessandro Manzoni\"}], \"pages\": "
    "[{\"pageid\": 2719455, \"ns\": 0, \"title\": \"Alessandro Manzoni\", "
    "\"description\": \"scrittore, poeta e drammaturgo italiano\", "
    "\"fullurl\": \"https://it.wikipedia.org/wiki/Alessandro_Manzoni\", "
    "\"canonicalurl\": \"https://it.wikipedia.org/wiki/Alessandro_Manzoni\", "
    "\"extract\": \"Alessandro Manzoni \\u00e8 stato uno scrittore.\\nNato "
    "a Milano.\\n\\n\\n== Biografia ==\\nLa vita.\\n\\n=== Infanzia "
    "===\\nI primi anni.\\n\\n=== La conversione ===\\nNel 1810.\\n\\n\\n== "
    "Opere ==\\nI promessi sposi.\\n\\n== Note ==\\n\"}]}}";

static const char DISAMB[] =
    "{\"query\": {\"pages\": [{\"pageid\": 2786, \"title\": \"Mercurio\", "
    "\"pageprops\": {\"disambiguation\": \"\"}, \"extract\": \"Mercurio "
    "\\u2013 pianeta\\nMercurio \\u2013 elemento chimico\"}]}}";

static const char MISSING[] =
    "{\"batchcomplete\": true, \"query\": {\"pages\": [{\"ns\": 0, "
    "\"title\": \"Xyzzyqq\", \"missing\": true}]}}";

static const char ERROR[] =
    "{\"error\": {\"code\": \"badvalue\", \"info\": \"Unrecognized value "
    "for parameter \\\"prop\\\": nope.\"}, \"servedby\": \"mw-api\"}";

static void test_hits(void)
{
    struct wiki_hit h[WIKI_HITS];
    char err[256];
    int n =
        wiki_read_hits(SEARCH, strlen(SEARCH), h, WIKI_HITS, err, sizeof err);
    CHECK(n == 3, "search: %d (%s)", n, n < 0 ? err : "");
    if (n == 3) {
        CHECK(strcmp(h[0].title, "I promessi sposi") == 0 &&
                  strcmp(h[1].title, "Trapani") == 0 &&
                  strcmp(h[2].title, "I promessi sposi (film 1941)") == 0,
              "search order: %s / %s / %s", h[0].title, h[1].title, h[2].title);
        CHECK(isnan(h[0].lat) && fabs(h[1].lat - 38.0175) < 1e-9 &&
                  fabs(h[1].lon - 12.515) < 1e-9,
              "search points");
        CHECK(strcmp(h[1].desc, "comune italiano") == 0 &&
                  strstr(h[1].text, "54 546") && h[0].desc[0] == 0,
              "search texts: '%s' '%s'", h[1].desc, h[1].text);
    }
    n = wiki_read_hits(SEARCH, strlen(SEARCH), h, 2, err, sizeof err);
    CHECK(n == 2, "search, at most 2: %d", n);
    n = wiki_read_hits(NEAR, strlen(NEAR), h, WIKI_HITS, err, sizeof err);
    CHECK(n == 3 && strcmp(h[0].title, "Trapani") == 0 &&
              strcmp(h[1].title, "Palazzo D'Alì") == 0,
          "geosearch: %d %s %s", n, n > 1 ? h[0].title : "",
          n > 1 ? h[1].title : "");
    n = wiki_read_hits(ERROR, strlen(ERROR), h, WIKI_HITS, err, sizeof err);
    CHECK(n < 0 && strstr(err, "Unrecognized value"), "error: %d %s", n, err);
    n = wiki_read_hits("<html>", 6, h, WIKI_HITS, err, sizeof err);
    CHECK(n < 0, "not json: %d", n);
    n = wiki_read_hits("{\"batchcomplete\": true}", 23, h, WIKI_HITS, err,
                       sizeof err);
    CHECK(n == 0, "nothing found: %d", n);
}

static void test_page(void)
{
    struct wiki_page p;
    char err[256];
    CHECK(wiki_read_page(PAGE, strlen(PAGE), &p, err, sizeof err) == 0,
          "page: %s", err);
    CHECK(strcmp(p.title, "Alessandro Manzoni") == 0 &&
              strcmp(p.from, "Manzoni") == 0 && !p.missing && !p.disamb &&
              isnan(p.lat) && strstr(p.url, "Alessandro_Manzoni"),
          "page fields: %s %s %s", p.title, p.from, p.url);
    size_t intro = wiki_intro_len(p.text, p.len);
    CHECK(intro == strlen("Alessandro Manzoni è stato uno scrittore.\n"
                          "Nato a Milano."),
          "intro: %zu", intro);
    struct janas_buf list = {0};
    wiki_section_list(p.text, p.len, &list);
    CHECK(list.p && strcmp(list.p, "Biografia (Infanzia, La conversione); "
                                   "Opere; Note") == 0,
          "sections: %s", list.p ? list.p : "(none)");
    janas_buf_free(&list);
    size_t at, n;
    CHECK(wiki_section(p.text, p.len, "biografia", &at, &n) == 0 &&
              strncmp(p.text + at, "== Biografia ==", 15) == 0 &&
              strstr(p.text + at, "Nel 1810.") &&
              memcmp(p.text + at + n - 9, "Nel 1810.", 9) == 0,
          "section Biografia: %.*s", (int)n, p.text + at);
    CHECK(wiki_section(p.text, p.len, "La conv", &at, &n) == 0 &&
              n == strlen("=== La conversione ===\nNel 1810."),
          "section by its start: %zu", n);
    CHECK(wiki_section(p.text, p.len, "conversione", &at, &n) == 0,
          "section by a part");
    CHECK(wiki_section(p.text, p.len, "Opere", &at, &n) == 0 &&
              n == strlen("== Opere ==\nI promessi sposi."),
          "section Opere: %zu", n);
    CHECK(wiki_section(p.text, p.len, "Discografia", &at, &n) != 0,
          "no such section");
    CHECK(strstr(p.text, "Milano.\n\n== Biografia ==\nLa vita.\n\n=== "
                         "Infanzia") &&
              p.text[p.len - 1] == '=',
          "squeezed: %s", p.text);
    const char *cut = "Testo.\n\n== A ==\nDi A.\n\n== B ==\n\n=== B1 ===\n";
    CHECK(wiki_drop_heads(cut, strlen(cut)) == strlen("Testo.\n\n== A "
                                                      "==\nDi A."),
          "headings dropped: %zu", wiki_drop_heads(cut, strlen(cut)));
    wiki_page_free(&p);

    CHECK(wiki_read_page(DISAMB, strlen(DISAMB), &p, err, sizeof err) == 0 &&
              p.disamb && strstr(p.text, "elemento chimico"),
          "disambiguation");
    wiki_page_free(&p);
    CHECK(wiki_read_page(MISSING, strlen(MISSING), &p, err, sizeof err) == 0 &&
              p.missing && !p.text,
          "missing");
    wiki_page_free(&p);
    CHECK(wiki_read_page(ERROR, strlen(ERROR), &p, err, sizeof err) != 0,
          "page error");
}

static void test_cut(void)
{
    const char *s = "Uno due tre.\n\nQuattro cinque. Sei sette otto.";
    size_t n = strlen(s);
    CHECK(wiki_cut(s, n, 100) == n, "no cut");
    CHECK(wiki_cut(s, n, 20) == 12, "cut at the paragraph: %zu",
          wiki_cut(s, n, 20));
    CHECK(wiki_cut(s, n, 35) == 29, "cut at the sentence: %zu",
          wiki_cut(s, n, 35));
    const char *u = "aaaa è è è è è è è è";
    size_t k = wiki_cut(u, strlen(u), 8);
    CHECK(k == 7, "cut at a word: %zu", k); /* "aaaa è" */
    const char *v = "èèèèèèèèèè";
    k = wiki_cut(v, strlen(v), 7);
    CHECK(k == 6, "cut not inside a character: %zu", k);
}

int main(void)
{
    test_hits();
    test_page();
    test_cut();
    if (failures) {
        printf("test_wiki_parse: %d failures\n", failures);
        return 1;
    }
    printf("test_wiki_parse: ok\n");
    return 0;
}
