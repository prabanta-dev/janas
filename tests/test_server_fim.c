/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_server_fim.c - the window of a fill-in-the-middle request
 * (src/server/fim.c), with a token every four bytes:
 *   1. the suffix is cut after its lines;
 *   2. an editor that sends the prefix from the start of the file keeps
 *      its start;
 *   3. an editor that slides a window of 200 lines: the start is put a
 *      quarter ahead at its first slide, and kept for the slides after;
 *   4. a prefix over the budget is cut at a line to three quarters of it,
 *      and the start kept while the prefix grows;
 *   5. two files in turn keep a start each;
 *   6. as 3, with a newline in front of every prefix, as Continue sends.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "server/fim.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static int32_t by4(void *ctx, const char *s, size_t n)
{
    (void)ctx, (void)s;
    return (int32_t)((n + 3) / 4);
}

/* lines from..to-1 of a file named name, into a buffer of its own */
static char *lines(const char *name, int from, int to, size_t *n)
{
    size_t cap = (size_t)(to - from) * 64 + 1;
    char *b = malloc(cap), *p = b;
    for (int i = from; i < to; i++)
        p += snprintf(p, cap - (size_t)(p - b),
                      "%s line %04d: x = f(x, %d); /* code */\n", name, i, i);
    *n = (size_t)(p - b);
    return b;
}

/* the line number a window starts at */
static int first_line(const char *s)
{
    const char *p = strstr(s, " line ");
    return p ? atoi(p + 6) : -1;
}

int main(void)
{
    alarm(60);
    size_t off, sl, n;
    const char *how;

    /* 1. */
    struct srv_fim *f = srv_fim_new(0, 8);
    char *suf = lines("s", 0, 20, &n);
    char *pre = lines("a", 0, 10, &n);
    srv_fim_window(f, by4, NULL, pre, n, suf, strlen(suf), &off, &sl, &how);
    size_t want = 0;
    for (int i = 0; i < 8; i++)
        want += strchr(suf + want, '\n') - (suf + want) + 1;
    CHECK(sl == want, "suffix %zu bytes, not %zu", sl, want);
    free(suf);
    free(pre);
    srv_fim_free(f);

    /* 2. */
    f = srv_fim_new(0, 0);
    for (int end = 100; end < 160; end += 7) {
        pre = lines("a", 0, end, &n);
        srv_fim_window(f, by4, NULL, pre, n, "", 0, &off, &sl, &how);
        CHECK(off == 0, "from the start, %d lines: cut at %zu (%s)", end, off,
              how);
        free(pre);
    }
    srv_fim_free(f);

    /* 3. */
    f = srv_fim_new(0, 0);
    int start = -1, reads = 0;
    for (int top = 0; top < 120; top++) {
        pre = lines("a", top, top + 200, &n);
        srv_fim_window(f, by4, NULL, pre, n, "", 0, &off, &sl, &how);
        CHECK(off == 0 || pre[off - 1] == '\n', "slide %d: not at a line", top);
        int at = first_line(pre + off);
        if (at != start) {
            reads++;
            start = at;
        }
        if (top == 1)
            CHECK(at > 40 && at < 60 &&
                      !strcmp(how, "ahead of a sliding window"),
                  "first slide: line %d (%s)", at, how);
        CHECK(at >= top, "slide %d: line %d", top, at);
        free(pre);
    }
    /* starts at 0, 51, then every 50 lines or so: four or five readings
       of the whole window in 120 lines, not 120 */
    CHECK(reads <= 5, "120 slides read the window %d times", reads);
    srv_fim_free(f);

    /* 4. lines of 42 bytes, 11 tokens: a budget of 1000 takes 90 */
    f = srv_fim_new(1000, 0);
    start = -1;
    reads = 0;
    for (int end = 300; end < 400; end++) {
        pre = lines("a", 0, end, &n);
        srv_fim_window(f, by4, NULL, pre, n, "", 0, &off, &sl, &how);
        int32_t t = by4(NULL, pre + off, n - off);
        CHECK(t <= 1000 && (off == 0 || pre[off - 1] == '\n'),
              "%d lines: %d tokens from %zu", end, t, off);
        int at = first_line(pre + off);
        if (at != start) {
            CHECK(t <= 750 && t > 650, "%d lines: cut to %d tokens", end, t);
            reads++;
            start = at;
        }
        free(pre);
    }
    CHECK(reads <= 6, "100 lines more read the window %d times", reads);
    srv_fim_free(f);

    /* 5. */
    f = srv_fim_new(0, 0);
    int sa = -1, sb = -1;
    for (int top = 0; top < 6; top++) {
        pre = lines("a", top, top + 200, &n);
        srv_fim_window(f, by4, NULL, pre, n, "", 0, &off, &sl, &how);
        int a = first_line(pre + off);
        free(pre);
        pre = lines("b", top + 1000, top + 1200, &n);
        srv_fim_window(f, by4, NULL, pre, n, "", 0, &off, &sl, &how);
        int b = first_line(pre + off);
        free(pre);
        if (top >= 2) {
            CHECK(a == sa && b == sb, "two files, slide %d: %d %d, not %d %d",
                  top, a, b, sa, sb);
        }
        sa = a;
        sb = b;
    }
    srv_fim_free(f);

    /* 6. */
    f = srv_fim_new(2048, 0);
    start = -1;
    reads = 0;
    for (int top = 0; top < 120; top++) {
        char *w = lines("a", top, top + 100, &n);
        pre = malloc(n + 2);
        pre[0] = '\n';
        memcpy(pre + 1, w, n + 1);
        free(w);
        srv_fim_window(f, by4, NULL, pre, n + 1, "", 0, &off, &sl, &how);
        CHECK(off >= 1 && pre[off - 1] == '\n' && pre[off] != '\n',
              "newline in front, slide %d: at %zu", top, off);
        int at = first_line(pre + off);
        if (at != start) {
            reads++;
            start = at;
        }
        CHECK(at >= top, "newline in front, slide %d: line %d", top, at);
        free(pre);
    }
    CHECK(reads <= 6, "newline in front: 120 slides read the window %d times",
          reads);
    srv_fim_free(f);

    if (failures) {
        printf("test_server_fim: %d failures\n", failures);
        return 1;
    }
    printf("test_server_fim: ok\n");
    return 0;
}
