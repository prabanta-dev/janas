/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * scroll.h - the conversation kept, so that it can be read again.
 *
 * A terminal with a scrolling region throws away the lines that leave the
 * top of it: they never reach its own history, and Page Up finds nothing.
 * So the chat keeps them itself and draws a window on what it kept. What is
 * kept are the lines as they were written, escape sequences and all, not
 * broken into screen rows: the window wraps them when it draws, so making
 * the terminal narrower re-wraps the conversation instead of ruining it.
 *
 * The window follows the end while it is at the end, and stays where it was
 * put while it is not, so a reply that arrives does not throw the reader
 * back to the bottom.
 */
#ifndef JANAS_CHAT_SCROLL_H
#define JANAS_CHAT_SCROLL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "term.h"

#define SCROLL_MAX_LINES 4096 /* the oldest are let go */
#define SCROLL_SGR_MAX 64     /* the colours carried from line to line */

struct janas_scroll {
    struct janas_term *t; /* the window it is drawn in */
    char **line;          /* the lines written, oldest first */
    int n, cap;
    char *pend; /* the line being written, no newline yet */
    size_t pend_n, pend_cap;
    /* the colours in force where the next line begins: a window is drawn a
       row at a time, each reset first, so a line that does not carry its own
       colour comes out in the terminal's. The reasoning opened its grey once
       and everything under that first line turned white. */
    char sgr[SCROLL_SGR_MAX];
    size_t sgr_n;
    int back; /* rows above the end the window sits at; 0: at the end */
    /* the bytes each row of the window was last drawn with, for the partial
       redraw of scroll_sink: trusted while the window's size and the
       terminal's count of clears are those of that drawing */
    char **shown;
    size_t *shown_n;
    int shown_view, shown_cols; /* 0: nothing known */
    int shown_full;             /* every row of the window had text */
    unsigned shown_cleared;
};

static void scroll_init(struct janas_scroll *s)
{
    memset(s, 0, sizeof(*s));
}

static void scroll_free(struct janas_scroll *s)
{
    for (int i = 0; i < s->n; i++)
        free(s->line[i]);
    free(s->line);
    free(s->pend);
    for (int r = 0; r < s->shown_view; r++)
        free(s->shown[r]);
    free(s->shown);
    free(s->shown_n);
    memset(s, 0, sizeof(*s));
}

/* One finished line into the store, the oldest let go when it is full. */
static void sc_push(struct janas_scroll *s, char *owned)
{
    if (s->n == s->cap) {
        int cap = s->cap ? s->cap * 2 : 128;
        char **l = realloc(s->line, (size_t)cap * sizeof(*l));
        if (!l) {
            free(owned);
            return;
        }
        s->line = l;
        s->cap = cap;
    }
    if (s->n == SCROLL_MAX_LINES) {
        free(s->line[0]);
        memmove(s->line, s->line + 1, (size_t)(s->n - 1) * sizeof(*s->line));
        s->n--;
        if (s->back > 0)
            s->back--; /* the window keeps looking at the same text */
    }
    s->line[s->n++] = owned;
}

/*
 * What is in force after this escape sequence. A full reset clears what was
 * held; any other sequence that ends in 'm' is a colour or an attribute and
 * is added to it. Everything else - moving the cursor, clearing a line -
 * says nothing about how the next line looks and is let go. When more is
 * held than there is room for, the oldest goes: a line drawn in the last
 * few colours is right far more often than one drawn in none.
 */
static void sc_sgr(struct janas_scroll *s, const char *e, size_t n)
{
    if (n < 3 || e[1] != '[' || e[n - 1] != 'm')
        return;
    if (n == 3 || (n == 4 && e[2] == '0')) { /* ESC [ m, ESC [ 0 m */
        s->sgr_n = 0;
        return;
    }
    if (n > SCROLL_SGR_MAX)
        return;
    while (s->sgr_n + n > SCROLL_SGR_MAX) { /* drop the oldest whole one */
        size_t k = sc_esc(s->sgr, s->sgr_n);
        if (!k || k > s->sgr_n) {
            s->sgr_n = 0;
            break;
        }
        memmove(s->sgr, s->sgr + k, s->sgr_n - k);
        s->sgr_n -= k;
    }
    memcpy(s->sgr + s->sgr_n, e, n);
    s->sgr_n += n;
}

/* Room for one more byte in the line being written. */
static int sc_room(struct janas_scroll *s, size_t more)
{
    if (s->pend_n + more + 1 <= s->pend_cap)
        return 1;
    size_t cap = s->pend_cap ? s->pend_cap * 2 : 256;
    while (cap < s->pend_n + more + 1)
        cap *= 2;
    char *p = realloc(s->pend, cap);
    if (!p)
        return 0;
    s->pend = p;
    s->pend_cap = cap;
    return 1;
}

/* What the chat prints, kept as well: split where the newlines are. Each
   line begins with the colours it is written in, so that it can be drawn
   on its own. */
static void scroll_add(struct janas_scroll *s, const char *text, size_t n)
{
    for (size_t i = 0; i < n;) {
        size_t e = sc_esc(text + i, n - i);
        if (e) {
            if (!sc_room(s, e))
                return;
            sc_sgr(s, text + i, e);
            memcpy(s->pend + s->pend_n, text + i, e);
            s->pend_n += e;
            s->pend[s->pend_n] = 0;
            i += e;
            continue;
        }
        if (!sc_room(s, 1))
            return;
        if (text[i] == '\n') {
            s->pend[s->pend_n] = 0;
            char *line = strdup(s->pend);
            if (line)
                sc_push(s, line);
            s->pend_n = 0;
            if (s->sgr_n && sc_room(s, s->sgr_n)) { /* the next line's own */
                memcpy(s->pend, s->sgr, s->sgr_n);
                s->pend_n = s->sgr_n;
            }
        } else if (text[i] != '\r')
            s->pend[s->pend_n++] = text[i];
        s->pend[s->pend_n] = 0; /* always a string, like the others */
        i++;
    }
}

/* The text of line i, the one still being written included (i == n). */
static const char *sc_text(const struct janas_scroll *s, int i)
{
    if (i < s->n)
        return s->line[i];
    return s->pend ? s->pend : "";
}

#define SC_MAX_ROWS 1024 /* screen rows one written line may take */

/* How many rows the whole conversation takes at this width. */
static long scroll_rows(const struct janas_scroll *s, int cols)
{
    int start[1];
    long rows = 0;
    for (int i = 0; i <= s->n; i++)
        rows += sc_wrap(sc_text(s, i), cols, start, 1);
    return rows;
}

struct sc_row {
    int line;     /* which written line it belongs to */
    int off, end; /* its bytes within that line, end < 0: to the end */
    int hang;     /* columns it is pushed right by, to line up under the
                     text of a list item it continues; 0 on a first row */
};

/*
 * The view rows that sit `back` rows above the end, newest first, so that
 * walking back stops as soon as there are enough of them: a conversation of
 * four thousand lines is not wrapped to show twenty.
 */
static int sc_collect(const struct janas_scroll *s, int cols, int back,
                      int view, struct sc_row *out)
{
    int skipped = 0, got = 0;
    static int start[SC_MAX_ROWS];
    for (int i = s->n; i >= 0 && got < view; i--) {
        const char *text = sc_text(s, i);
        int r = sc_wrap(text, cols, start, SC_MAX_ROWS);
        if (r > SC_MAX_ROWS)
            r = SC_MAX_ROWS;
        int hang = sc_hang(text);
        if (hang >= cols - 8) /* the same guard sc_wrap uses */
            hang = 0;
        for (int k = r - 1; k >= 0 && got < view; k--) {
            if (skipped < back) {
                skipped++;
                continue;
            }
            out[got].line = i;
            out[got].off = start[k];
            out[got].end = k + 1 < r ? start[k + 1] : -1;
            out[got].hang = k ? hang : 0;
            got++;
        }
    }
    return got;
}

/* A growing buffer of bytes. */
struct sc_buf {
    char *b;
    size_t n, cap;
};

static void sc_put(struct sc_buf *o, const char *p, size_t n)
{
    if (o->n + n > o->cap) {
        size_t cap = o->cap ? o->cap * 2 : 256;
        while (cap < o->n + n)
            cap *= 2;
        char *b = realloc(o->b, cap);
        if (!b)
            return;
        o->b = b;
        o->cap = cap;
    }
    memcpy(o->b + o->n, p, n);
    o->n += n;
}

/*
 * One row, with the colours it inherits: a row that starts in the middle of
 * a written line would otherwise lose the escape that coloured it, so the
 * escapes before it are sent again, and nothing else of what they preceded.
 * Into o, which then holds exactly what the row is drawn with.
 */
static void sc_put_row(struct sc_buf *o, const char *text,
                       const struct sc_row *r, int cols)
{
    size_t n = strlen(text);
    for (int k = 0; k < r->hang; k++)
        sc_put(o, " ", 1);
    for (size_t i = 0; i < (size_t)r->off && i < n;) {
        size_t e = sc_esc(text + i, n - i);
        if (e) {
            sc_put(o, text + i, e);
            i += e;
        } else {
            size_t bytes;
            term_char_cols(text + i, n - i, &bytes);
            i += bytes ? bytes : 1;
        }
    }
    /* no further than the window's edge: a row is wrapped to fit, but the
       space it was broken after can stay at its end, one column past the
       edge, where the terminal would wrap it again - on the region's last
       row by moving the whole window up one */
    size_t from = (size_t)r->off;
    size_t to = r->end < 0 ? n : (size_t)r->end;
    int used = r->hang;
    for (size_t i = from; i < to;) {
        size_t e = sc_esc(text + i, to - i);
        if (e) {
            sc_put(o, text + i, e);
            i += e;
            continue;
        }
        size_t bytes;
        int w = term_char_cols(text + i, to - i, &bytes);
        if (!bytes)
            bytes = 1;
        if (used + w <= cols) {
            sc_put(o, text + i, bytes);
            used += w;
        }
        i += bytes;
    }
}

/*
 * The window, drawn over the scrolling region. Returns the columns the
 * bottom row holds, which is where the next text goes. With `partial`, the
 * rows the terminal already shows are left alone: while a reply streams in,
 * a piece of it changes the row being written and, when the text wraps, moves
 * every row up. Redrawing the whole window for each piece sent 12 KB a token
 * to the terminal and cost Konsole 3% of the reply's speed (2 Oct 2026). Each
 * row is laid out into the bytes it is drawn with; a row whose bytes are
 * those it was drawn with last time is not sent again, and when the rows
 * match those shown k rows lower the terminal moves them itself - a newline
 * on the region's last row - and only the rows that are new are drawn.
 */
static int scroll_draw_rows(struct janas_term *t, struct janas_scroll *s,
                            int partial)
{
    if (!t->tty)
        return 0;
    int view = t->rows - t->footer_rows;
    if (view < 1)
        view = 1;
    struct sc_row *rows = malloc((size_t)view * sizeof(*rows));
    struct sc_buf *now = calloc((size_t)view, sizeof(*now));
    if (!rows || !now) {
        free(rows);
        free(now);
        return 0;
    }
    int got = sc_collect(s, t->cols, s->back, view, rows);
    /*
     * A window that is not full fills from the top, as a terminal does: the
     * banner of a conversation just begun belongs up there with the room
     * under it, not pushed to the bottom of the screen.
     */
    int bottom = got < view ? got : view;
    for (int r = 1; r <= view; r++) {
        int i = bottom - r; /* rows[0] is the newest, and sits at bottom */
        if (i >= 0 && i < got)
            sc_put_row(&now[r - 1], sc_text(s, rows[i].line), &rows[i],
                       t->cols);
    }
    int known = partial && s->shown && s->shown_view == view &&
                s->shown_cols == t->cols && s->shown_cleared == t->cleared;
    if (!known) { /* everything drawn, and remembered from here */
        for (int r = 0; r < s->shown_view; r++)
            free(s->shown[r]);
        free(s->shown);
        free(s->shown_n);
        s->shown = calloc((size_t)view, sizeof(*s->shown));
        s->shown_n = calloc((size_t)view, sizeof(*s->shown_n));
        s->shown_view = s->shown && s->shown_n ? view : 0;
        s->shown_cols = t->cols;
        s->shown_cleared = t->cleared;
    }
    printf("\0337"); /* the box and the footer keep their cursor */
    /* a window that was not full held its rows at the top, with room
       below: a move is a move only from full to full */
    if (known && got >= view && s->shown_full) {
        int best = 0, most = -1;
        for (int k = 0; k <= 4 && k < view; k++) {
            int match = 0;
            for (int r = 1; r + k <= view; r++)
                match +=
                    s->shown[r + k - 1] &&
                    s->shown_n[r + k - 1] == now[r - 1].n &&
                    !memcmp(s->shown[r + k - 1], now[r - 1].b, now[r - 1].n);
            if (match > most) { /* ties: the smaller move */
                most = match;
                best = k;
            }
        }
        if (best > 0) {
            /* the region again, should anything have reset it: a newline
               outside one would scroll the box and the footer away too */
            printf("\033[1;%dr\033[%d;1H", view, view);
            for (int k = 0; k < best; k++)
                putchar('\n');
            for (int r = 0; r < best; r++)
                free(s->shown[r]);
            memmove(s->shown, s->shown + best,
                    (size_t)(view - best) * sizeof(*s->shown));
            memmove(s->shown_n, s->shown_n + best,
                    (size_t)(view - best) * sizeof(*s->shown_n));
            for (int r = view - best; r < view; r++) {
                s->shown[r] = NULL; /* blank now: a row unknown */
                s->shown_n[r] = 0;
            }
        }
    }
    for (int r = 1; r <= view; r++) {
        struct sc_buf *o = &now[r - 1];
        if (known && s->shown[r - 1] && s->shown_n[r - 1] == o->n &&
            !memcmp(s->shown[r - 1], o->b, o->n))
            continue; /* the terminal shows these bytes already */
        printf("\033[%d;1H\033[K\033[0m", r);
        if (o->n)
            fwrite(o->b, 1, o->n, stdout);
        if (s->shown_view) {
            free(s->shown[r - 1]);
            s->shown[r - 1] = o->b ? o->b : calloc(1, 1);
            s->shown_n[r - 1] = o->n;
            o->b = NULL; /* kept */
        }
    }
    for (int r = 0; r < view; r++)
        free(now[r].b);
    free(now);
    printf("\033[0m\0338");
    fflush(stdout);
    s->shown_full = got >= view;
    int cols = 0;
    if (got > 0) { /* the bottom row: what is written there already */
        const char *text = sc_text(s, rows[0].line);
        size_t n = strlen(text);
        cols = rows[0].hang;
        size_t to = rows[0].end < 0 ? n : (size_t)rows[0].end;
        for (size_t i = (size_t)rows[0].off; i < to;) {
            size_t e = sc_esc(text + i, to - i);
            if (e) {
                i += e;
                continue;
            }
            size_t bytes;
            cols += term_char_cols(text + i, to - i, &bytes);
            i += bytes ? bytes : 1;
        }
    }
    free(rows);
    return cols;
}

static int scroll_draw(struct janas_term *t, struct janas_scroll *s)
{
    return scroll_draw_rows(t, s, 0);
}

/*
 * The same, and the place the conversation writes from put back where the
 * window now ends: the scrolling region moves when the box grows, and the
 * cursor the conversation had saved may be under it by then.
 */
static void scroll_anchor(struct janas_term *t, struct janas_scroll *s)
{
    int cols = scroll_draw(t, s);
    if (!t->tty)
        return;
    int view = t->rows - t->footer_rows;
    if (view < 1)
        view = 1;
    long total = scroll_rows(s, t->cols);
    int bottom = total < view ? (int)total : view;
    if (bottom < 1)
        bottom = 1;
    if (cols >= t->cols)
        cols = t->cols - 1;
    printf("\033[%d;%dH\0337", bottom, cols + 1);
    fflush(stdout);
}

/*
 * Moving the window. Up is towards the beginning, and it stops there; down
 * stops at the end, where the window starts following again.
 */
static void scroll_move(struct janas_term *t, struct janas_scroll *s, int rows)
{
    int view = t->rows - t->footer_rows;
    if (view < 1)
        view = 1;
    long total = scroll_rows(s, t->cols);
    long most = total - view;
    if (most < 0)
        most = 0;
    long to = (long)s->back + rows;
    if (to < 0)
        to = 0;
    if (to > most)
        to = most;
    if (to == s->back)
        return;
    s->back = (int)to;
    scroll_draw(t, s);
}

static void scroll_to_end(struct janas_term *t, struct janas_scroll *s)
{
    if (s->back == 0)
        return;
    s->back = 0;
    scroll_draw(t, s);
}

static void scroll_to_start(struct janas_term *t, struct janas_scroll *s)
{
    scroll_move(t, s, 1 << 20);
}

/*
 * The store as the terminal's sink: everything the conversation prints is
 * kept, and shown as it comes while the window is at the end. Scrolled back,
 * the reader is left where they are and the text waits below.
 */
static void scroll_sink(void *arg, const char *text, size_t n)
{
    struct janas_scroll *s = arg;
    scroll_add(s, text, n);
    /*
     * Drawn, not printed. Letting the text go straight out left the wrapping
     * to the terminal, which breaks a line at its edge whatever is there -
     * "per" came out as "p" and "er" on the next row. Drawing the window
     * instead puts one rule in charge of where a row ends, the same one that
     * lays the conversation out again when it is read back or the window is
     * made narrower.
     */
    if (s->back == 0 && s->t)
        scroll_draw_rows(s->t, s, 1);
}

static void scroll_attach(struct janas_term *t, struct janas_scroll *s)
{
    if (!t->tty)
        return; /* a pipe keeps nothing: it has the whole thing already */
    s->t = t;
    t->sink = scroll_sink;
    t->sink_arg = s;
}

#endif
