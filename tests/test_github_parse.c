/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_github_parse.c - janas-github's reading of GitHub's answers, without
 * the network, and the layouts filled with what was read: Janas's issues
 * and a pull request of llama.cpp as the search gives them, a release of
 * llama.cpp (its notes opening with HTML), Janas's commits and repository,
 * a search of repositories and the user's own, as they came on 2 October
 * 2026 (cut to a few items); and when the issues were last seen, in a
 * directory of its own under ~/tmp.
 * The sources belong to the program, so they are compiled in here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "services/common/template.h"
#include "services/github/data.c"
#include "services/github/layouts.c"
#include "services/github/read.c"
#include "services/github/seen.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char S_ISSUES[] =
    "{\"total_count\": 14, \"incomplete_results\": false, \"items\": [{\"numb"
    "er\": 15, \"title\": \"Test report: AMD Ryzen 9 9950X3D 16-Core Processo"
    "r, medium\", \"user\": {\"login\": \"bfg9000it\"}, \"html_url\": \"https"
    "://github.com/prabanta-dev/janas/issues/15\", \"created_at\": \"2026-09-"
    "27T10:50:23Z\", \"closed_at\": null, \"comments\": 7, \"state\": \"open\""
    ", \"labels\": []}, {\"number\": 14, \"title\": \"Test report: AMD Ryzen "
    "9 9950X3D 16-Core Processor, medium\", \"user\": {\"login\": \"bfg9000it"
    "\"}, \"html_url\": \"https://github.com/prabanta-dev/janas/issues/14\", "
    "\"created_at\": \"2026-09-27T10:17:33Z\", \"closed_at\": null, \"comment"
    "s\": 0, \"state\": \"open\", \"labels\": []}, {\"number\": 13, \"title\""
    ": \"Test report: Intel(R) Core(TM) i9-14900HX, full\", \"user\": {\"logi"
    "n\": \"smallrobots\"}, \"html_url\": \"https://github.com/prabanta-dev/j"
    "anas/issues/13\", \"created_at\": \"2026-09-27T09:51:34Z\", \"closed_at\""
    ": \"2026-09-27T18:33:41Z\", \"comments\": 5, \"state\": \"closed\", \"la"
    "bels\": [{\"name\": \"test-report\"}]}, {\"number\": 29846, \"title\": \""
    "CUDA: fuse KV cache dequantization into fattn-mma-f16\", \"user\": {\"lo"
    "gin\": \"Abai\"}, \"html_url\": \"https://github.com/ggml-org/llama.cpp/"
    "pull/29846\", \"created_at\": \"2026-10-02T11:03:32Z\", \"closed_at\": n"
    "ull, \"comments\": 3, \"state\": \"open\", \"labels\": [], \"pull_reques"
    "t\": {\"merged_at\": null}, \"draft\": false}]}";
static const char S_RELEASES[] =
    "[{\"tag_name\": \"b11344\", \"name\": \"b11344\", \"html_url\": \"https:"
    "//github.com/ggml-org/llama.cpp/releases/tag/b11344\", \"published_at\":"
    " \"2026-10-02T08:54:49Z\", \"prerelease\": true, \"draft\": false, \"bod"
    "y\": \"<details open>\\n\\nCUDA: fix 2 broken Volta FA cases (#29803)\\n"
    "\\n</details>\\n\\n**Website:**\\n- <https://llama.app>\\n\\n**Attestati"
    "ons:**\\n- <https://github.com/ggml-org/llama.cpp/attestations/52093363>"
    "\\n\\n**macOS/iOS:**\\n- [macOS Apple Silicon (arm64)](https://github.co"
    "m/ggml-org/llama.cpp/releases/download/b11344/llama-b11344-bin-macos-arm"
    "64.tar.gz)\\n- macOS Apple Silicon (arm64, KleidiAI enabled) [DISABLED]("
    "https://github.com/ggml-org/llama.cpp/pull/23780)\\n- [macOS Intel (x64)"
    "](https://github.com/ggml-org/llama.cpp/releases/download/b11344/llama-b"
    "11344-bin-macos-x64.tar.gz)\\n- [iOS XCFramework](https://github.com/ggm"
    "l-org/llama.cpp/releases/download/b11344/llama-b11344-xcframework.zip)\\n"
    "\\n**Linux:**\\n- [Ubunt\", \"author\": {\"login\": \"github-actions[bot"
    "]\"}, \"assets\": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {"
    "}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {"
    "}, {}, {}, {}]}]";
static const char S_COMMITS[] =
    "[{\"sha\": \"d6b2cf29252ba3244b8c27d5cdd0721ac70fd9be\", \"html_url\": \""
    "https://github.com/prabanta-dev/janas/commit/d6b2cf29252ba3244b8c27d5cdd"
    "0721ac70fd9be\", \"commit\": {\"message\": \"Release 2026-10-02: janas-m"
    "aps - roads and places as an MCP service (routes, public transport, what"
    " is near, addresses), tried in janas-chat\\n\\nSigned-off-by: camauri <m"
    "aurizio.cammalleri@gmail.com>\", \"author\": {\"name\": \"camauri\", \"d"
    "ate\": \"2026-10-02T11:18:31Z\"}}}, {\"sha\": \"8eb26dcb83bfef4e6d188a16"
    "a1ddc0c843709589\", \"html_url\": \"https://github.com/prabanta-dev/jana"
    "s/commit/8eb26dcb83bfef4e6d188a16a1ddc0c843709589\", \"commit\": {\"mess"
    "age\": \"Release 2026-10-02: layouts - the services' data in the user's "
    "language, a brief to the model; a glossary of the scales' terms\\n\\nSig"
    "ned-off-by: camauri <maurizio.cammalleri@gmail.com>\", \"author\": {\"na"
    "me\": \"camauri\", \"date\": \"2026-10-01T22:11:17Z\"}}}, {\"sha\": \"45"
    "0b548e585d20144fc15f68aa6db09a722d14b1\", \"html_url\": \"https://github"
    ".com/prabanta-dev/janas/commit/450b548e585d20144fc15f68aa6db09a722d14b1\""
    ", \"commit\": {\"message\": \"Release 2026-10-01: janas-wiki, the servic"
    "es' tools given when needed, the system message read ahead\", \"author\""
    ": {\"name\": \"camauri\", \"date\": \"2026-10-01T19:32:44Z\"}}}]";
static const char S_REPO[] =
    "{\"full_name\": \"prabanta-dev/janas\", \"description\": \"Janas-LLM: la"
    "rge mixture-of-experts language models on ordinary computers, in C. Expe"
    "rts streamed from NVMe, with or without a GPU.\", \"language\": \"C\", \""
    "license\": {\"spdx_id\": \"GPL-3.0\", \"name\": \"GNU General Public Lic"
    "ense v3.0\"}, \"default_branch\": \"main\", \"homepage\": null, \"html_u"
    "rl\": \"https://github.com/prabanta-dev/janas\", \"topics\": [\"avx2\", "
    "\"c\", \"cpu-inference\", \"gguf\", \"inference-engine\", \"linux\", \"l"
    "lm\", \"local-llm\", \"mixture-of-experts\", \"moe\", \"nvme\", \"quanti"
    "zation\", \"qwen3\", \"simd\", \"speculative-decoding\", \"vulkan\"], \""
    "stargazers_count\": 29, \"forks_count\": 5, \"subscribers_count\": 1, \""
    "open_issues_count\": 4, \"pushed_at\": \"2026-10-02T11:18:34Z\", \"creat"
    "ed_at\": \"2026-09-20T09:55:09Z\", \"archived\": false, \"private\": fal"
    "se, \"fork\": false}";
static const char S_SEARCH[] =
    "{\"total_count\": 2721, \"items\": [{\"full_name\": \"ggml-org/llama.cpp"
    "\", \"stargazers_count\": 130126}, {\"full_name\": \"abetlen/llama-cpp-p"
    "ython\", \"stargazers_count\": 10636}, {\"full_name\": \"ikawrakow/ik_ll"
    "ama.cpp\", \"stargazers_count\": 3273}, {\"full_name\": \"withcatai/node"
    "-llama-cpp\", \"stargazers_count\": 2187}, {\"full_name\": \"go-skynet/g"
    "o-llama.cpp\", \"stargazers_count\": 940}]}";
static const char S_MINE[] =
    "[{\"full_name\": \"prabanta-dev/janas\"}, {\"full_name\": \"prabanta-dev"
    "/limba\"}, {\"full_name\": \"prabanta-dev/meri\"}, {\"full_name\": \"cam"
    "auri/SedaiBasic2\"}, {\"full_name\": \"prabanta-dev/.github\"}, {\"full_"
    "name\": \"camauri/SedaiBasic2-Deps\"}, {\"full_name\": \"camauri/SedaiAu"
    "dio\"}, {\"full_name\": \"camauri/PxLab\"}, {\"full_name\": \"camauri/Ka"
    "irosProject\"}]";

/* The layout filled with the data: the text, or "" when it failed. */
static char text[16384];

static const char *fill(const char *layout, const struct janas_buf *d)
{
    text[0] = 0;
    struct janas_json_doc *doc =
        janas_json_parse(d->p ? d->p : "", d->n, NULL, 0);
    CHECK(doc != NULL, "the data is not JSON:\n%.*s", (int)d->n, d->p);
    if (!doc)
        return text;
    struct janas_buf out = {0};
    char err[200];
    if (janas_tpl_render(layout, strlen(layout), janas_json_root(doc), &out,
                         err, sizeof err) == 0)
        snprintf(text, sizeof text, "%.*s", (int)out.n, out.p);
    else
        CHECK(0, "a layout not filled: %s", err);
    janas_buf_free(&out);
    janas_json_free(doc);
    return text;
}

static void issues(void)
{
    struct gh_issues v;
    char err[200] = "";
    CHECK(gh_read_issues(S_ISSUES, strlen(S_ISSUES), 10, &v, err, sizeof err) ==
                  0 &&
              v.total == 14 && v.n == 4,
          "the issues: %s (%ld, %d)", err, v.total, v.n);
    CHECK(v.i[0].number == 15 && strcmp(v.i[0].user, "bfg9000it") == 0 &&
              v.i[0].open && !v.i[0].pr && v.i[0].comments == 7 &&
              v.i[0].created == gh_utc("2026-09-27T10:50:23Z"),
          "the first issue");
    CHECK(v.i[2].number == 13 && !v.i[2].open &&
              strcmp(v.i[2].labels, "test-report") == 0,
          "a closed issue with its label: %s", v.i[2].labels);
    CHECK(v.i[3].pr && !v.i[3].merged && v.i[3].number == 29846,
          "a pull request");
    struct janas_buf d = {0};
    gh_issues_data(&d, "prabanta-dev/janas", "yours", 0, "all",
                   gh_utc("2026-09-25T12:00:00Z"), 1, &v);
    const char *t = fill(GH_ISSUES_LAYOUT, &d);
    CHECK(strstr(t, "prabanta-dev/janas (one of yours): 14 new issues since "
                    "25 September") &&
              strstr(t, "(the first time you ask: the last 7 days); the "
                        "latest 4:\n") &&
              strstr(t, "  #15 Test report: AMD Ryzen 9 9950X3D 16-Core "
                        "Processor, medium\n      open, by bfg9000it, 27 "
                        "September") &&
              strstr(t, "closed, by smallrobots, 27 September") &&
              strstr(t, ", comments: 5, labels: test-report\n") &&
              strstr(t, "      https://github.com/prabanta-dev/janas/issues/"
                        "15\n"),
          "the issues' layout:\n%s", t);
    /* a label is the issue's own: #15 has none */
    const char *i15 = strstr(t, "#15");
    const char *i14 = strstr(t, "#14");
    CHECK(i15 && i14 && !memmem(i15, (size_t)(i14 - i15), "labels", 6),
          "#15 without labels");
    t = fill(GH_ISSUES_BRIEF, &d);
    CHECK(strstr(t, "say which repository was taken") &&
              strstr(t, "14 new_issues (of those listed: 3 open, 1 "
                        "closed) since 25/9") &&
              strstr(t, "#15 "),
          "the issues' brief:\n%s", t);
    janas_buf_free(&d);
    struct gh_issues none = {0};
    gh_issues_data(&d, "a/b", "", 1, "open", 0, 0, &none);
    t = fill(GH_ISSUES_LAYOUT, &d);
    CHECK(strcmp(t, "a/b: no open pull requests.\n") == 0, "none:\n%s", t);
    janas_buf_free(&d);
}

static void releases(void)
{
    struct gh_releases v;
    char err[200] = "";
    CHECK(gh_read_releases(S_RELEASES, strlen(S_RELEASES), 3, &v, err,
                           sizeof err) == 0 &&
              v.n == 1,
          "the releases: %s", err);
    CHECK(strcmp(v.r[0].tag, "b11344") == 0 && !v.r[0].name[0] && v.r[0].pre &&
              v.r[0].assets == 35 &&
              strcmp(v.r[0].author, "github-actions[bot]") == 0,
          "the release");
    CHECK(strncmp(v.r[0].notes, "CUDA: fix 2 broken Volta FA cases (#29803)\n",
                  43) == 0 &&
              strstr(v.r[0].notes, "- https://llama.app\n") &&
              !strchr(v.r[0].notes, '<'),
          "the notes without HTML, links kept:\n%s", v.r[0].notes);
    struct janas_buf d = {0};
    gh_releases_data(&d, "ggml-org/llama.cpp", "found by its name", &v);
    const char *t = fill(GH_RELEASES_LAYOUT, &d);
    CHECK(strstr(t, "ggml-org/llama.cpp (the one of that name with the most "
                    "stars): the latest release:\n\nb11344 (pre-release), 2 "
                    "October") &&
              strstr(t, ", by github-actions[bot], files: 35\nCUDA: fix") &&
              strstr(t, "https://github.com/ggml-org/llama.cpp/releases/tag/"
                        "b11344\n"),
          "the releases' layout:\n%s", t);
    janas_buf_free(&d);
    CHECK(gh_read_tags("[{\"name\": \"v1.2\"}]", 18, 3, &v, err, sizeof err) ==
                  0 &&
              v.tags && v.n == 1,
          "the tags");
    gh_releases_data(&d, "a/b", "", &v);
    t = fill(GH_RELEASES_LAYOUT, &d);
    CHECK(strcmp(t, "a/b: no releases; its latest tags:\n\nv1.2\n") == 0,
          "tags:\n%s", t);
    janas_buf_free(&d);
}

static void commits(void)
{
    struct gh_commits v;
    char err[200] = "";
    CHECK(gh_read_commits(S_COMMITS, strlen(S_COMMITS), 2, &v, err,
                          sizeof err) == 0 &&
              v.n == 2 && v.more,
          "the commits: %s", err);
    CHECK(strcmp(v.c[0].sha, "d6b2cf2") == 0 &&
              strncmp(v.c[0].message, "Release 2026-10-02: janas-maps", 30) ==
                  0 &&
              !strchr(v.c[0].message, '\n') &&
              strcmp(v.c[0].author, "camauri") == 0,
          "the first commit");
    struct janas_buf d = {0};
    gh_commits_data(&d, "prabanta-dev/janas", "", "main", 3, &v);
    const char *t = fill(GH_COMMITS_LAYOUT, &d);
    CHECK(strstr(t, "prabanta-dev/janas, branch main: the latest 2 commits in "
                    "the last 3 days:\n  d6b2cf2 Release") &&
              strstr(t, "\n      camauri, 2 October"),
          "the commits' layout:\n%s", t);
    janas_buf_free(&d);
}

static void repo(void)
{
    struct gh_repo r;
    char err[200] = "", full[128] = "";
    CHECK(gh_read_repo(S_REPO, strlen(S_REPO), &r, err, sizeof err) == 0 &&
              strcmp(r.full, "prabanta-dev/janas") == 0 && r.stars == 29 &&
              strcmp(r.license, "GPL-3.0") == 0 &&
              strncmp(r.topics, "avx2, c, ", 9) == 0 && !r.archived,
          "the repository: %s", err);
    struct janas_buf d = {0};
    gh_repo_data(&d, "", &r);
    const char *t = fill(GH_REPO_LAYOUT, &d);
    CHECK(strstr(t, "prabanta-dev/janas\nJanas-LLM:") &&
              strstr(t, "\nstars: 29, forks: 5, watching: 1; open issues and "
                        "pull requests: 4\nlanguage: C\nlicence: GPL-3.0\n"
                        "main branch: main; last push: 2 October") &&
              !strstr(t, "site:"),
          "the repository's layout:\n%s", t);
    janas_buf_free(&d);
    CHECK(gh_read_search_repo(S_SEARCH, strlen(S_SEARCH), "LLAMA.cpp", full,
                              sizeof full, err, sizeof err) == 0 &&
              strcmp(full, "ggml-org/llama.cpp") == 0,
          "the search: %s", full);
    CHECK(gh_read_search_repo(S_SEARCH, strlen(S_SEARCH), "llama", full,
                              sizeof full, err, sizeof err) == 0 &&
              strcmp(full, "ggml-org/llama.cpp") == 0,
          "the search, no name alike: the first, %s", full);
    CHECK(gh_read_mine(S_MINE, strlen(S_MINE), "Janas", full, sizeof full, err,
                       sizeof err) == 1 &&
              strcmp(full, "prabanta-dev/janas") == 0 &&
              gh_read_mine(S_MINE, strlen(S_MINE), "llama.cpp", full,
                           sizeof full, err, sizeof err) == 0,
          "the user's own");
}

static void seen(void)
{
    char dir[512];
    const char *home = getenv("HOME");
    snprintf(dir, sizeof dir, "%s/tmp/janas-github-test-%d", home ? home : ".",
             (int)getpid());
    if (mkdir(dir, 0700) != 0) {
        CHECK(0, "cannot make %s", dir);
        return;
    }
    setenv("XDG_CONFIG_HOME", dir, 1);
    CHECK(gh_seen("a/b", "issues") == 0, "never seen");
    gh_seen_set("a/b", "issues", 100);
    gh_seen_set("a/b", "pulls", 200);
    gh_seen_set("c/d", "issues", 300);
    gh_seen_set("a/b", "issues", 400);
    CHECK(gh_seen("a/b", "issues") == 400 && gh_seen("a/b", "pulls") == 200 &&
              gh_seen("c/d", "issues") == 300 && gh_seen("c/d", "pulls") == 0,
          "seen");
    char path[1200];
    snprintf(path, sizeof path, "%s/janas/github-seen.txt", dir);
    CHECK(unlink(path) == 0, "no file at %s", path);
    snprintf(path, sizeof path, "%s/janas", dir);
    rmdir(path);
    rmdir(dir);
}

int main(void)
{
    setenv("TZ", "UTC", 1);
    CHECK(gh_utc("2026-10-02T11:18:31Z") == 1790939911 && gh_utc(NULL) == -1,
          "gh_utc");
    issues();
    releases();
    commits();
    repo();
    seen();
    if (failures) {
        printf("test_github_parse: %d failures\n", failures);
        return 1;
    }
    printf("test_github_parse: ok\n");
    return 0;
}
