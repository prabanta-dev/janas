/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * edit.h - reading a line in janas-chat (included by main.c alone).
 *
 * A small editor of our own: the arrows, Home and End, the usual control
 * keys, and the previous prompts, kept between sessions in the same place
 * the engine keeps what it learns about this machine. Characters are moved
 * over whole, not byte by byte, so accented letters and anything else in
 * UTF-8 behave. When the input is not a terminal the line is simply read,
 * as before.
 */
#ifndef JANAS_CHAT_EDIT_H
#define JANAS_CHAT_EDIT_H

#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "scroll.h"

#define HIST_MAX 500

struct janas_hist {
    char *line[HIST_MAX];
    int n; /* lines held, oldest first */
    char path[512] /* empty: not kept on disk */;
};

static void hist_add(struct janas_hist *h, const char *s)
{
    if (!*s || (h->n && !strcmp(h->line[h->n - 1], s)))
        return;
    if (h->n == HIST_MAX) {
        free(h->line[0]);
        memmove(h->line, h->line + 1, (HIST_MAX - 1) * sizeof(*h->line));
        h->n--;
    }
    h->line[h->n] = strdup(s);
    if (h->line[h->n])
        h->n++;
}

static void hist_load(struct janas_hist *h, const char *path)
{
    memset(h, 0, sizeof(*h));
    snprintf(h->path, sizeof(h->path), "%s", path ? path : "");
    FILE *f = h->path[0] ? fopen(h->path, "r") : NULL;
    if (!f)
        return;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r'))
            line[--n] = 0;
        hist_add(h, line);
    }
    fclose(f);
}

static void hist_save(const struct janas_hist *h)
{
    if (!h->path[0])
        return;
    FILE *f = fopen(h->path, "w");
    if (!f)
        return;
    for (int i = 0; i < h->n; i++)
        fprintf(f, "%s\n", h->line[i]);
    fclose(f);
}

static void hist_free(struct janas_hist *h)
{
    for (int i = 0; i < h->n; i++)
        free(h->line[i]);
    h->n = 0;
}

/* The store the conversation is kept in, when there is one. */
static struct janas_scroll *ed_store(struct janas_term *t)
{
    return t->sink == scroll_sink ? (struct janas_scroll *)t->sink_arg : NULL;
}

static size_t ed_left(const char *s, size_t pos)
{
    while (pos && (s[--pos] & 0xc0) == 0x80)
        ;
    return pos;
}

static size_t ed_right(const char *s, size_t len, size_t pos)
{
    if (pos < len)
        do
            pos++;
        while (pos < len && (s[pos] & 0xc0) == 0x80);
    return pos;
}

/*
 * The bytes at the start of the line that name a command the chat really
 * has: the first word must be one of them, whole. Nothing while it is still
 * being typed, so the colour appears exactly when the command is there.
 */
static int ed_command(const char *const *cmds, const char *buf, size_t len)
{
    if (!cmds || !len || buf[0] != '/')
        return 0;
    size_t w = 0;
    while (w < len && buf[w] != ' ')
        w++;
    for (int i = 0; cmds[i]; i++)
        if (strlen(cmds[i]) == w && !strncmp(cmds[i], buf, w))
            return (int)w;
    return 0;
}

/* Draws prompt and line in the box, scrolled sideways to keep the cursor in. */
static void ed_draw(struct janas_term *t, const char *prompt, const char *buf,
                    size_t len, size_t pos, int hl)
{
    /*
     * The box takes the rows the message needs, up to the cap, and the
     * conversation takes the rest: growing it moves the scrolling region,
     * so what is above has to be drawn again at its new size. It goes back
     * to one row of its own accord when the line is short again.
     */
    int cap = 1;
    int rows = term_box_rows(t, prompt, buf, (int)len, &cap);
    if (rows > cap)
        rows = cap;
    int want = rows + 3; /* the two rules and the footer */
    if (t->footer_rows != want) {
        t->footer_rows = want;
        term_region(t);
        struct janas_scroll *st = ed_store(t);
        if (st)
            scroll_anchor(t, st);
    }
    term_box(t, prompt, buf, (int)len, (int)pos, hl, rows);
}

/*
 * Tab: the commands that start like what is written. One of them is put in
 * whole; several put in what they have in common and are listed in the
 * conversation above the box.
 */
static size_t ed_complete(struct janas_term *t, const char *const *cmds,
                          char *buf, size_t cap, size_t len)
{
    if (!cmds || buf[0] != '/' || memchr(buf, ' ', len))
        return len;
    size_t n = 0, common = 0;
    const char *first = NULL;
    for (int i = 0; cmds[i]; i++)
        if (!strncmp(cmds[i], buf, len)) {
            if (!n)
                first = cmds[i], common = strlen(cmds[i]);
            else
                while (common > len && strncmp(first, cmds[i], common))
                    common--;
            n++;
        }
    if (!n)
        return len;
    if (common > len && common + 2 < cap) {
        memcpy(buf, first, common);
        len = common;
        if (n == 1)
            buf[len++] = ' ';
        buf[len] = 0;
        return len;
    }
    printf("\0338%s", T_DIM(t)); /* back to the conversation for the list */
    for (int i = 0; cmds[i]; i++)
        if (!strncmp(cmds[i], buf, len))
            printf("%s%s", i ? "  " : "", cmds[i]);
    printf("%s\n\0337", T_RESET(t));
    return len;
}

/*
 * Called twice a second while the editor waits for a key, when set: what
 * the caller shows that changes on its own (the filling of the expert
 * cache, in the footer). Returns 1 when it drew something, and the editor
 * then draws its line again, the cursor with it.
 */
static int (*line_idle)(void);

/*
 * One line from the user. Returns 1 with the line in buf, 0 at the end of
 * the input (Ctrl-D on an empty line), -1 when the line was given up
 * (Ctrl-C), which leaves buf empty.
 */
static int line_read(struct janas_term *t, const char *prompt, char *buf,
                     size_t cap, struct janas_hist *h, const char *const *cmds)
{
    if (!t->tty) {
        printf("%s", prompt);
        fflush(stdout);
        if (!fgets(buf, (int)cap, stdin))
            return 0;
        size_t n = strlen(buf);
        while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
            buf[--n] = 0;
        return 1;
    }
    size_t len = 0, pos = 0;
    int hi = h->n; /* h->n: the line being written */
    char *saved = NULL;
    buf[0] = 0;
    term_raw(t, 1);
    printf("\0337"); /* where the conversation is, to come back to */
    ed_draw(t, prompt, buf, len, pos, ed_command(cmds, buf, len));
    for (;;) {
        unsigned char c;
        if (line_idle) {
            struct pollfd pf = {STDIN_FILENO, POLLIN, 0};
            int pr = poll(&pf, 1, 500);
            if (pr < 0 && errno == EINTR) { /* the window changed size */
                term_size(t);
                ed_draw(t, prompt, buf, len, pos, ed_command(cmds, buf, len));
                fflush(stdout);
                continue;
            }
            if (pr == 0) {
                t->editing = 1;
                int drew = line_idle();
                t->editing = 0;
                if (drew) {
                    ed_draw(t, prompt, buf, len, pos,
                            ed_command(cmds, buf, len));
                    fflush(stdout);
                }
                continue;
            }
            /* a key: read() below takes it */
        }
        ssize_t r = read(STDIN_FILENO, &c, 1);
        if (r < 0 && errno == EINTR) { /* the window changed size */
            term_size(t);
            ed_draw(t, prompt, buf, len, pos, ed_command(cmds, buf, len));
            continue;
        }
        if (r <= 0) {
            term_box_clear(t);
            printf("\0338");
            term_raw(t, 0);
            free(saved);
            return 0;
        }
        if (c == '\r' || c == '\n') {
            /*
             * A message can be more than a line: a backslash at the end and
             * Enter puts a newline in it and goes on, the box growing under
             * the conversation as it does. The backslash is not part of
             * what is written - it asked for the line, and it is gone once
             * it has been given, the way it would be in an editor.
             */
            if (len && buf[len - 1] == '\\') {
                buf[len - 1] = '\n';
                if (pos > len - 1)
                    pos = len;
                ed_draw(t, prompt, buf, len, pos, ed_command(cmds, buf, len));
                continue;
            }
            struct janas_scroll *st_send = ed_store(t);
            buf[len] = 0;
            term_box_clear(t);
            if (t->footer_rows != 4) { /* back to one row for the next one */
                t->footer_rows = 4;
                term_region(t);
                if (st_send)
                    scroll_anchor(t, st_send);
            }
            /* the line as it was sent, on the same ground as the box it
               was written in. The blank line that separates it from what
               comes next is not printed here: term_gap puts one between any
               two blocks, and two of them printing one each gave two. */
            printf("\0338");
            /* sending is asking to see the answer: whoever was reading the
               conversation further up is brought back to the end */
            if (st_send)
                scroll_to_end(t, st_send);
            term_gap(t);
            term_echo(t, prompt, buf, ed_command(cmds, buf, len));
            fflush(stdout);
            term_raw(t, 0);
            free(saved);
            hist_add(h, buf);
            return 1;
        }
        if (c == 3) { /* Ctrl-C: give up this line */
            term_box_clear(t);
            printf("\0338");
            term_raw(t, 0);
            free(saved);
            buf[0] = 0;
            return -1;
        }
        if (c == 4 && len == 0) { /* Ctrl-D on an empty line */
            term_box_clear(t);
            printf("\0338");
            term_raw(t, 0);
            free(saved);
            return 0;
        }
        if (c == 127 || c == 8) { /* Backspace */
            if (pos) {
                size_t p = ed_left(buf, pos);
                memmove(buf + p, buf + pos, len - pos);
                len -= pos - p;
                pos = p;
            }
        } else if (c == 1) { /* Ctrl-A */
            pos = 0;
        } else if (c == 5) { /* Ctrl-E */
            pos = len;
        } else if (c == 11) { /* Ctrl-K: to the end */
            len = pos;
        } else if (c == 21) { /* Ctrl-U: to the start */
            memmove(buf, buf + pos, len - pos);
            len -= pos;
            pos = 0;
        } else if (c == 23) { /* Ctrl-W: the word before */
            size_t p = pos;
            while (p && buf[p - 1] == ' ')
                p--;
            while (p && buf[p - 1] != ' ')
                p--;
            memmove(buf + p, buf + pos, len - pos);
            len -= pos - p;
            pos = p;
        } else if (c == 9) { /* Tab: the commands */
            len = ed_complete(t, cmds, buf, cap, len);
            pos = len;
        } else if (c == 12) { /* Ctrl-L: a clean window */
            printf("\033[2J\033[H");
            term_region(t);
            printf("\0337");
        } else if (c == 27) { /* a key that arrives as an escape sequence */
            /*
             * Read the whole of it: the parameters, then the byte that
             * names the key. Three bytes were enough for the arrows and
             * Home, but Ctrl-Home sends five, and what was left over used
             * to end up typed into the line.
             */
            struct pollfd pf = {STDIN_FILENO, POLLIN, 0};
            unsigned char lead = 0, final = 0;
            char par[16];
            int pn = 0;
            if (poll(&pf, 1, 50) <= 0 || read(STDIN_FILENO, &lead, 1) != 1 ||
                (lead != '[' && lead != 'O'))
                continue;
            for (;;) {
                unsigned char b;
                if (read(STDIN_FILENO, &b, 1) != 1)
                    break;
                if (b >= 0x40 && b <= 0x7e) {
                    final = b;
                    break;
                }
                if (pn < (int)sizeof(par) - 1)
                    par[pn++] = (char)b;
            }
            par[pn] = 0;
            if (!final)
                continue;
            struct janas_scroll *st = ed_store(t);
            int page = t->rows - t->footer_rows - 1;
            int ctrl = strstr(par, ";5") != NULL;
            if (st && final == '~' && strcmp(par, "5") == 0)
                scroll_move(t, st, page > 1 ? page : 1); /* Page Up */
            else if (st && final == '~' && strcmp(par, "6") == 0)
                scroll_move(t, st, -(page > 1 ? page : 1)); /* Page Down */
            else if (st && ctrl && final == 'H')
                scroll_to_start(t, st); /* Ctrl-Home */
            else if (st && ctrl && final == 'F')
                scroll_to_end(t, st);                         /* Ctrl-End */
            else if (final == '~' && strcmp(par, "3") == 0) { /* Delete */
                if (pos < len) {
                    size_t p = ed_right(buf, len, pos);
                    memmove(buf + pos, buf + p, len - p);
                    len -= p - pos;
                }
            } else if (final == '~' &&
                       (strcmp(par, "1") == 0 || strcmp(par, "7") == 0))
                pos = 0;
            else if (final == '~' &&
                     (strcmp(par, "4") == 0 || strcmp(par, "8") == 0))
                pos = len;
            else if (final == 'D')
                pos = ed_left(buf, pos);
            else if (final == 'C')
                pos = ed_right(buf, len, pos);
            else if (final == 'H')
                pos = 0;
            else if (final == 'F')
                pos = len;
            else if (final == 'A' || final == 'B') {
                /*
                 * Up and down move through the message while there is one
                 * to move through, and bring back earlier prompts once the
                 * cursor is off its first or last row. The column is kept,
                 * as an editor keeps it.
                 */
                int col = 0, total = 1;
                int row = term_box_find(t, prompt, buf, (int)len, (int)pos,
                                        &col, &total);
                int want = final == 'A' ? row - 1 : row + 1;
                if (total > 1 && want >= 0 && want < total) {
                    pos = (size_t)term_box_at(t, prompt, buf, (int)len, want,
                                              col);
                    ed_draw(t, prompt, buf, len, pos,
                            ed_command(cmds, buf, len));
                    continue;
                }
                /* history */
                int to = hi + (final == 'A' ? -1 : 1);
                if (to < 0 || to > h->n)
                    continue;
                if (hi == h->n) { /* keep what was being written */
                    free(saved);
                    buf[len] = 0;
                    saved = strdup(buf);
                }
                hi = to;
                const char *src =
                    hi == h->n ? (saved ? saved : "") : h->line[hi];
                snprintf(buf, cap, "%s", src);
                len = strlen(buf);
                pos = len;
            }
        } else if (c >= 32) { /* a character to insert */
            if (len + 1 < cap) {
                memmove(buf + pos + 1, buf + pos, len - pos);
                buf[pos++] = (char)c;
                len++;
            }
        }
        buf[len] = 0;
        ed_draw(t, prompt, buf, len, pos, ed_command(cmds, buf, len));
    }
}

#endif
