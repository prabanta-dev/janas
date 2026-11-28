/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* What Brebus's updates read of a release (update_rel.h): versions, which
   is newer, GitHub's answer for a release, SHA256SUMS's lines. */
#include "common/update_rel.h"

#include <stdio.h>
#include <string.h>

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

static void versions(void)
{
    int p[4];
    CHECK(janas_rel_version("v2026.11.26", p) == 0 && p[0] == 2026 &&
              p[1] == 11 && p[2] == 26 && p[3] == 0,
          "v2026.11.26");
    CHECK(janas_rel_version("2026.11.24.1", p) == 0 && p[3] == 1,
          "2026.11.24.1");
    static const char *const no[] = {"v2026.11.18-6-g75f29a2",
                                     "v2026.11.26-dirty",
                                     "unknown",
                                     "",
                                     "v",
                                     "75f29a2",
                                     "v2026.11",
                                     "v2026.13.01",
                                     "v2026.11.26.1.2",
                                     "v2026..26",
                                     "v2026.11.26.",
                                     "v1999.01.01",
                                     "v2026.11.0",
                                     "v2026.11.26 "};
    for (size_t i = 0; i < sizeof no / sizeof *no; i++)
        CHECK(janas_rel_version(no[i], p) != 0, "taken for a release: '%s'",
              no[i]);
    CHECK(janas_rel_version(NULL, p) != 0, "NULL taken");

    CHECK(janas_rel_newer("v2026.11.27", "v2026.11.26"), "a day later");
    CHECK(janas_rel_newer("v2026.11.26.1", "v2026.11.26"), "the day's second");
    CHECK(janas_rel_newer("v2026.12.01", "v2026.11.30"), "a month later");
    CHECK(janas_rel_newer("v2027.01.01", "v2026.12.31"), "a year later");
    CHECK(janas_rel_newer("v2026.11.26.10", "v2026.11.26.9"),
          "10 after 9, as numbers");
    CHECK(!janas_rel_newer("v2026.11.26", "v2026.11.26"), "the same");
    CHECK(!janas_rel_newer("v2026.11.25", "v2026.11.26"), "older");
    CHECK(!janas_rel_newer("v2026.11.26", "v2026.11.26.1"), "older, by .1");
    /* a build between releases is never updated over */
    CHECK(!janas_rel_newer("v2026.11.27", "v2026.11.26-3-gabc1234"),
          "a developer's build taken for a release");
    CHECK(!janas_rel_newer("v2026.11.27", "unknown"), "unknown taken");
    CHECK(!janas_rel_newer("nightly", "v2026.11.26"), "a tag of no date");
}

static void release_json(void)
{
    static const char gh[] =
        "{\"tag_name\": \"v2026.11.27\", \"draft\": false, \"prerelease\": "
        "false, \"assets\": ["
        "{\"name\": \"brebus-x86_64.AppImage\", \"size\": 39127544, "
        "\"browser_download_url\": \"https://github.com/prabanta-dev/brebus/"
        "releases/download/v2026.11.27/brebus-x86_64.AppImage\"},"
        "{\"name\": \"plain.bin\", \"size\": 1, \"browser_download_url\": "
        "\"http://example.com/plain.bin\"},"
        "{\"name\": \"local\", \"size\": 2, \"browser_download_url\": "
        "\"http://127.0.0.1:8080/local\"},"
        "{\"name\": \"nosize\", \"browser_download_url\": "
        "\"https://example.com/nosize\"},"
        "{\"name\": \"SHA256SUMS\", \"size\": 725, \"browser_download_url\": "
        "\"https://github.com/prabanta-dev/brebus/releases/download/"
        "v2026.11.27/SHA256SUMS\"}]}";
    char tag[64], err[300];
    struct janas_rel_asset a[8];
    int n = -1;
    CHECK(janas_rel_parse(gh, sizeof gh - 1, tag, sizeof tag, a, 8, &n, err,
                          sizeof err) == 0,
          "parse: %s", err);
    CHECK(strcmp(tag, "v2026.11.27") == 0, "tag %s", tag);
    CHECK(n == 3,
          "%d files, not 3 (one over HTTP elsewhere, one without its "
          "size)",
          n);
    CHECK(n >= 2 && strcmp(a[0].name, "brebus-x86_64.AppImage") == 0 &&
              a[0].size == 39127544 && strcmp(a[1].name, "local") == 0 &&
              strcmp(a[2].name, "SHA256SUMS") == 0,
          "the files");
    CHECK(janas_rel_parse(gh, sizeof gh - 1, tag, sizeof tag, a, 1, &n, err,
                          sizeof err) == 0 &&
              n == 1,
          "max not kept: %d", n);

    static const char pre[] = "{\"tag_name\": \"v2026.11.27\", \"prerelease\": "
                              "true, \"assets\": []}";
    CHECK(janas_rel_parse(pre, sizeof pre - 1, tag, sizeof tag, a, 8, &n, err,
                          sizeof err) != 0,
          "a pre-release taken");
    static const char draft[] =
        "{\"tag_name\": \"v2026.11.27\", \"draft\": true, \"assets\": []}";
    CHECK(janas_rel_parse(draft, sizeof draft - 1, tag, sizeof tag, a, 8, &n,
                          err, sizeof err) != 0,
          "a draft taken");
    static const char limit[] =
        "{\"message\": \"API rate limit exceeded\", \"documentation_url\": "
        "\"https://docs.github.com\"}";
    CHECK(janas_rel_parse(limit, sizeof limit - 1, tag, sizeof tag, a, 8, &n,
                          err, sizeof err) != 0 &&
              strstr(err, "rate limit"),
          "GitHub's message not said: %s", err);
    CHECK(janas_rel_parse("[", 1, tag, sizeof tag, a, 8, &n, err, sizeof err) !=
              0,
          "broken JSON taken");
    char small[4];
    CHECK(janas_rel_parse(gh, sizeof gh - 1, small, sizeof small, a, 8, &n, err,
                          sizeof err) != 0,
          "a tag longer than its buffer taken");
}

static void sums(void)
{
    /* v2026.11.26's own, two of its lines, and the ways a line may come */
    static const char s[] =
        "4b1b3c6c4f0b4b06e0f8de4a43c3b4bfd6e7b1f4d6a5c8e96f5b8e3ef2e1d0c9  "
        "brebus-x86_64.AppImage\n"
        "DAE06CA9C5267C6C8501FE1E9F766B0000000000000000000000000000000001 "
        "*brebus_amd64.deb\r\n"
        "0123  short.bin\n"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  "
        "twice\n"
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  "
        "twice\n"
        "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz  "
        "nothex";
    char sha[65];
    CHECK(janas_rel_sum(s, sizeof s - 1, "brebus-x86_64.AppImage", sha) == 0 &&
              strcmp(sha, "4b1b3c6c4f0b4b06e0f8de4a43c3b4bfd6e7b1f4d6a5c8e96f"
                          "5b8e3ef2e1d0c9") == 0,
          "AppImage: %s", sha);
    CHECK(janas_rel_sum(s, sizeof s - 1, "brebus_amd64.deb", sha) == 0 &&
              strcmp(sha, "dae06ca9c5267c6c8501fe1e9f766b00000000000000000000"
                          "00000000000001") == 0,
          "deb (binary mark, CR, upper case): %s", sha);
    CHECK(janas_rel_sum(s, sizeof s - 1, "brebus", sha) != 0,
          "a name's beginning taken");
    CHECK(janas_rel_sum(s, sizeof s - 1, "x86_64.AppImage", sha) != 0,
          "a name's end taken");
    CHECK(janas_rel_sum(s, sizeof s - 1, "short.bin", sha) != 0,
          "a short sum taken");
    CHECK(janas_rel_sum(s, sizeof s - 1, "twice", sha) != 0,
          "a name listed twice taken");
    CHECK(janas_rel_sum(s, sizeof s - 1, "nothex", sha) != 0,
          "a sum not hex taken");
    CHECK(janas_rel_sum(s, sizeof s - 1, "missing", sha) != 0,
          "a missing name found");
}

int main(void)
{
    versions();
    release_json();
    sums();
    if (failures) {
        printf("test_update_rel: %d failures\n", failures);
        return 1;
    }
    printf("test_update_rel: ok\n");
    return 0;
}
