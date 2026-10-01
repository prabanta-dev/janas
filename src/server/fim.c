/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fim.c - the window of a fill-in-the-middle request.
 *
 * An editor asks at every keystroke, with the text before the cursor and
 * after it. The session reuses what the last request shares from the
 * start, so the start of the prefix must not move. An editor that sends
 * "the last N lines" moves it at every new line, and the whole prefix was
 * read again - 81 s for 5300 tokens, Qwen3-Coder-30B on a laptop, against
 * 7 s for one keystroke more.
 *
 * So the head of each window is remembered (a few, for a few files):
 *  - a later prefix holding the start of one is cut there;
 *  - a prefix starting inside the head of one, past its start, comes from
 *    an editor that slides its window: it is cut a quarter of the way in,
 *    so that the next lines it slides keep that start in;
 *  - a prefix over the budget is cut to three quarters of it, at a line,
 *    which leaves room for many lines before the next cut.
 * Each cut costs one reading of the whole window, and the keystrokes after
 * it read only what they changed.
 *
 * The suffix comes after the prefix and is read again at every request:
 * only its first lines are kept.
 */
#include "fim.h"

#include <stdlib.h>
#include <string.h>

#define N_HEADS 8     /* windows remembered: a few files */
#define HEAD_MAX 4096 /* bytes of a window's head remembered */
#define START_MAX 160 /* of them, the bytes a start is matched by */
#define START_MIN 24  /* shorter ones match too easily */

struct head {
    char s[HEAD_MAX];
    size_t n;
    uint64_t used;
};

struct srv_fim {
    uint32_t prefix_tokens, suffix_lines;
    uint64_t clock;
    struct head h[N_HEADS];
};

struct srv_fim *srv_fim_new(uint32_t prefix_tokens, uint32_t suffix_lines)
{
    struct srv_fim *f = calloc(1, sizeof(*f));
    if (f) {
        f->prefix_tokens = prefix_tokens;
        f->suffix_lines = suffix_lines;
    }
    return f;
}

void srv_fim_free(struct srv_fim *f)
{
    free(f);
}

/* The start of the line at off or the first after it; without one, off
   itself moved past the bytes that continue a UTF-8 character. */
static size_t line_from(const char *p, size_t n, size_t off)
{
    if (off == 0 || off >= n)
        return off < n ? off : n;
    if (p[off - 1] == '\n')
        return off;
    const char *nl = memchr(p + off, '\n', n - off);
    if (nl && (size_t)(nl - p) + 1 < n)
        return (size_t)(nl - p) + 1;
    while (off < n && ((unsigned char)p[off] & 0xC0) == 0x80)
        off++;
    return off;
}

/* Where the k bytes of s are at the start of a line of t (n bytes), from
   the line at or after from on: the first place, or SIZE_MAX. */
static size_t at_line(const char *t, size_t n, size_t from, const char *s,
                      size_t k)
{
    for (size_t o = line_from(t, n, from); o < n;) {
        if (n - o >= k && memcmp(t + o, s, k) == 0)
            return o;
        const char *nl = memchr(t + o, '\n', n - o);
        if (!nl)
            break;
        o = (size_t)(nl - t) + 1;
    }
    return SIZE_MAX;
}

static size_t start_len(const struct head *h)
{
    return h->n < START_MAX ? h->n : START_MAX;
}

/* The prefix cut at a line so that its tokens are want at most, from off
   on: the new offset. */
static size_t cut_to(srv_fim_count count, void *ctx, const char *pre, size_t n,
                     size_t off, int32_t t, uint32_t want)
{
    for (int tries = 0; tries < 16 && t > (int32_t)want; tries++) {
        size_t rest = n - off;
        /* the bytes in proportion, a little less each try */
        size_t keep = (size_t)((double)rest * want / t * (1.0 - tries / 16.0));
        size_t to = line_from(pre, n, n - keep);
        if (to <= off)
            to = line_from(pre, n, off + 1);
        off = to;
        if (off >= n)
            break;
        t = count(ctx, pre + off, n - off);
        if (t < 0)
            break;
    }
    return off;
}

void srv_fim_window(struct srv_fim *f, srv_fim_count count, void *ctx,
                    const char *pre, size_t n_pre, const char *suf,
                    size_t n_suf, size_t *pre_off, size_t *suf_len,
                    const char **how)
{
    *how = "";
    *suf_len = n_suf;
    if (f->suffix_lines) {
        uint32_t lines = 0;
        for (size_t i = 0; i < n_suf; i++)
            if (suf[i] == '\n' && ++lines == f->suffix_lines) {
                *suf_len = i + 1;
                break;
            }
    }

    /* blank lines in front are left out: Continue joins its snippets of
       other files to the prefix with a newline, and sends one even with
       none, which put each window's start at the end of a line */
    size_t lead = 0;
    while (lead < n_pre && (pre[lead] == '\n' || pre[lead] == '\r'))
        lead++;
    pre += lead;
    n_pre -= lead;

    /* the start of a window before, the one used last if several */
    int hit = -1;
    size_t off = 0;
    for (int i = 0; i < N_HEADS; i++) {
        if (!f->h[i].n)
            continue;
        size_t o = at_line(pre, n_pre, 0, f->h[i].s, start_len(&f->h[i]));
        if (o != SIZE_MAX && (hit < 0 || f->h[i].used > f->h[hit].used)) {
            hit = i;
            off = o;
        }
    }
    if (off)
        *how = "at the start of the window before";

    /* none: does this prefix start inside the head of one, past its start?
       Then the editor slides its window, and the start is put ahead */
    size_t k = n_pre < START_MAX ? n_pre : START_MAX;
    if (hit < 0 && k >= START_MIN)
        for (int i = 0; i < N_HEADS; i++)
            if (f->h[i].n > k &&
                at_line(f->h[i].s, f->h[i].n, 1, pre, k) != SIZE_MAX) {
                off = line_from(pre, n_pre, n_pre / 4);
                *how = "ahead of a sliding window";
                break;
            }

    if (f->prefix_tokens && count) {
        int32_t t = count(ctx, pre + off, n_pre - off);
        uint32_t want = f->prefix_tokens / 4 * 3;
        if (t > (int32_t)f->prefix_tokens) {
            off = cut_to(count, ctx, pre, n_pre, off, t, want);
            *how = "over the budget";
            hit = -1;
        }
    }
    /* a start that would leave less than half of the text, and under a
       budget less than three eighths of it, is too far (the cursor went
       up): the window starts again */
    int32_t kept = f->prefix_tokens && count && hit >= 0
                       ? count(ctx, pre + off, n_pre - off)
                       : 0;
    if (hit >= 0 && off > n_pre / 2 &&
        (!f->prefix_tokens || kept < (int32_t)(f->prefix_tokens / 8 * 3))) {
        off = 0;
        *how = "";
        hit = -1;
        int32_t t = f->prefix_tokens && count ? count(ctx, pre, n_pre) : 0;
        if (t > (int32_t)f->prefix_tokens) {
            off =
                cut_to(count, ctx, pre, n_pre, 0, t, f->prefix_tokens / 4 * 3);
            *how = "over the budget";
        }
    }

    size_t rest = n_pre - off;
    size_t len = rest / 2 < HEAD_MAX ? rest / 2 : HEAD_MAX;
    if (hit >= 0) {
        f->h[hit].used = ++f->clock;
    } else if (len >= START_MIN) {
        int slot = 0;
        for (int i = 1; i < N_HEADS; i++)
            if (f->h[i].used < f->h[slot].used)
                slot = i;
        memcpy(f->h[slot].s, pre + off, len);
        f->h[slot].n = len;
        f->h[slot].used = ++f->clock;
    }
    *pre_off = lead + off;
}
