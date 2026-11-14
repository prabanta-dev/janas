/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_bench_report.c - janas-bench's reports found in a text
 * (src/common/bench_report.c): a tester's report read with its names
 * cleaned, quoted and indented, two in one text, one without its result
 * left out; then the text cut short at every byte, mutated 20000 times and
 * given lines longer than any buffer. Nothing may crash or read past the
 * text (the asan build sees to the reading).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/bench_report.h"

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

static const char REPORT[] =
    "### The benchmark report\n\n```text\n"
    "janas-bench 6b8ba9c, 20261112-224547\n"
    "system:  Linux 7.3.3-1-cachyos x86_64\n"
    "CPU:     Intel(R) Core(TM) Ultra 5 125U (2 performance cores, 4 "
    "threads on them, 12 usable in all)\n"
    "memory:  30.7 GiB, 27.6 GiB available; power: mains\n"
    "GPU:     Intel(R) Graphics (MTL)\n"
    "model:   Qwen3 Next 80B A3B Instruct (qwen3next), context 4096 "
    "tokens\n"
    "result:  prefill 703.0 tok/s, decode 41.2 tok/s, with MTP 49.1 tok/s "
    "greedy, 54.2 tok/s sampled\n```\n";

static struct janas_bench_row got[8];
static int n_got;

static int keep(void *ctx, const struct janas_bench_row *r)
{
    (void)ctx;
    if (n_got < 8)
        got[n_got] = *r;
    n_got++;
    return 0;
}

static int scan(const char *s, size_t n)
{
    n_got = 0;
    return janas_bench_report_scan(s, n, keep, NULL);
}

int main(void)
{
    /* a report, its names as people say them */
    CHECK(scan(REPORT, strlen(REPORT)) == 1, "one report: %d", n_got);
    const struct janas_bench_row *r = &got[0];
    CHECK(!strcmp(r->version, "6b8ba9c"), "version [%s]", r->version);
    CHECK(!strcmp(r->sys, "Linux"), "system [%s]", r->sys);
    CHECK(!strcmp(r->cpu, "Intel Core Ultra 5 125U"), "cpu [%s]", r->cpu);
    CHECK(!strcmp(r->ram, "30.7 GiB"), "ram [%s]", r->ram);
    CHECK(!strcmp(r->gpu, "Intel Graphics"), "gpu [%s]", r->gpu);
    CHECK(!strcmp(r->model, "Qwen3-Next-80B-A3B"), "model [%s]", r->model);
    CHECK(r->prefill == 703.0 && r->decode == 41.2 && r->mtp_greedy == 49.1 &&
              r->mtp_sampled == 54.2,
          "speeds %g %g %g %g", r->prefill, r->decode, r->mtp_greedy,
          r->mtp_sampled);

    /* quoted in a reply, twice in one text, a third without its result */
    char two[4096];
    snprintf(two, sizeof two,
             "> janas-bench v2026.11.13, 1\n> CPU:     AMD Ryzen 9 9950X3D "
             "16-Core Processor\n> model:   Qwen3.6 35B A3B (qwen35moe)\n>"
             " result:  prefill 317.4 tok/s, decode 22.8 tok/s\n\n%s"
             "janas-bench x, 2\nCPU: A\nmodel: B\n",
             REPORT);
    CHECK(scan(two, strlen(two)) == 2 &&
              !strcmp(got[0].cpu, "AMD Ryzen 9 9950X3D") &&
              !strcmp(got[0].model, "Qwen3.6-35B-A3B") &&
              got[0].decode == 22.8 && got[0].mtp_greedy == 0,
          "quoted and twice: %d [%s] [%s] %g", n_got, got[0].cpu, got[0].model,
          got[0].decode);

    /* cut short at every byte: at most the one report, never past it */
    size_t n = strlen(REPORT);
    char *cut = malloc(n);
    for (size_t k = 0; k <= n && cut; k++) {
        memcpy(cut, REPORT, k);
        int f = scan(cut, k);
        CHECK(f == 0 || f == 1, "cut at %zu: %d", k, f);
    }
    free(cut);

    /* mutated: bytes changed anywhere, NULs and newlines among them */
    srand(12345);
    char *m = malloc(n);
    int accepted = 0;
    for (int t = 0; t < 20000 && m; t++) {
        memcpy(m, REPORT, n);
        int k = 1 + rand() % 8;
        for (int i = 0; i < k; i++)
            m[rand() % n] = (char)(rand() % 256);
        accepted += scan(m, n) > 0;
    }
    free(m);

    /* lines longer than every buffer */
    size_t big = 200000;
    char *l = malloc(big);
    if (l) {
        memset(l, 'x', big);
        memcpy(l, "janas-bench ", 12);
        l[5000] = ',';
        l[60000] = '\n';
        memcpy(l + 60001, "CPU: ", 5);
        l[150000] = '\n';
        memcpy(l + 150001, "model: ", 7);
        l[160000] = '\n';
        memcpy(l + 160001, "result: decode 5", 16);
        CHECK(scan(l, big) <= 1, "long lines");
        free(l);
    }

    printf("test_bench_report: %d of 20000 mutated reports read, %s\n",
           accepted, failures ? "FAILED" : "ok");
    return failures != 0;
}
