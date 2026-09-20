/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * term.h - the terminal janas-chat draws on (included by main.c alone).
 *
 * When both ends are a terminal the chat takes the window: a banner at the
 * top, the conversation scrolling in the middle, and a footer that stays at
 * the bottom with what is worth having under the eye - the model, the
 * context used, the speed of the last reply, the state of the expert cache.
 * The footer keeps its place because the text scrolls inside a region that
 * stops above it (DECSTBM), so nothing has to be redrawn as replies stream.
 *
 * Piped in or out, none of this happens: the chat reads lines and writes
 * plain text, so scripts and measurements see what they saw before.
 */
#ifndef JANAS_CHAT_TERM_H
#define JANAS_CHAT_TERM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <stdarg.h>
#include <stdint.h>
#include <termios.h>
#include <wchar.h>
#include <unistd.h>

struct janas_term {
    int tty;    /* stdin and stdout are both a terminal */
    int colour; /* and colours are welcome there */
    int rows, cols;
    int footer_rows;
    int raw;     /* the terminal is in raw mode right now */
    int editing; /* the line editor holds the saved cursor (ESC 7): the
                    footer must not save its own over it */
    int gapped;  /* a block has been printed: the next one needs a line */
    /* where the conversation goes. NULL: straight out, as it always did.
       The store that keeps it (scroll.h) puts itself here, so that what
       scrolls away can be read again; the footer and the box never go
       through it, they are drawn where they are and kept nowhere. */
    void (*sink)(void *, const char *, size_t);
    void *sink_arg;
    struct termios saved;
};

/* Colours: empty strings when they are not wanted, so every use is one
   printf argument either way. */
#define T_RESET(t) ((t)->colour ? "\033[0m" : "")
#define T_DIM(t) ((t)->colour ? "\033[2m" : "")
#define T_BOLD(t) ((t)->colour ? "\033[1m" : "")
#define T_USER(t) ((t)->colour ? "\033[38;5;179m" : "")  /* amber */
#define T_MODEL(t) ""                                    /* the terminal's */
#define T_BRAND(t) ((t)->colour ? "\033[38;5;179m" : "") /* the mascot */
#define T_THINK(t) ((t)->colour ? "\033[38;5;245m" : "") /* grey */
#define T_WARN(t) ((t)->colour ? "\033[38;5;203m" : "")  /* red */
#define T_OK(t) ((t)->colour ? "\033[38;5;108m" : "")    /* green */
#define T_BAR(t) ((t)->colour ? "\033[48;5;236m" : "")   /* the line written */
#define T_CMD(t) ((t)->colour ? "\033[38;5;109m" : "")   /* a real command */
#define T_FG(t) ((t)->colour ? "\033[39m" : "") /* the foreground alone */

/*
 * Columns, which are neither bytes nor characters: a tick or an ideogram
 * takes two of them, a combining accent none. The chat used to count the
 * lead bytes of the UTF-8, so a reply with a tick in it - and a model puts
 * them everywhere - wrapped short and the box drifted. wcwidth() knows, once
 * the locale says UTF-8, which is why the chat asks for the user's own.
 *
 * term_char_cols reads one character and says how wide it is and how many
 * bytes it took; where wchar_t is not Unicode the old count is kept, which
 * is right for everything but the wide and the combining.
 */
static int term_char_cols(const char *s, size_t n, size_t *bytes)
{
    unsigned char c = (unsigned char)s[0];
    size_t len = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
    if (c >= 0x80 && c < 0xc0) /* a stray continuation byte: one at a time */
        len = 1;
    if (len > n)
        len = n;
    if (bytes)
        *bytes = len;
#ifdef __STDC_ISO_10646__
    uint32_t cp = c < 0x80   ? c
                  : c < 0xe0 ? (uint32_t)(c & 0x1f)
                  : c < 0xf0 ? (uint32_t)(c & 0x0f)
                             : (uint32_t)(c & 0x07);
    for (size_t i = 1; i < len; i++)
        cp = (cp << 6) | ((unsigned char)s[i] & 0x3f);
    int w = wcwidth((wchar_t)cp);
    return w < 0 ? 0 : w; /* unprintable: it takes no room of its own */
#else
    return 1;
#endif
}

/* The columns of n bytes of UTF-8. */
static int term_cols(const char *s, size_t n)
{
    int cols = 0;
    for (size_t i = 0; i < n;) {
        size_t used;
        cols += term_char_cols(s + i, n - i, &used);
        i += used ? used : 1;
    }
    return cols;
}

/*
 * The rule that lays text out, shared by everything that lays text out:
 * the conversation, what is read back again, and the banner. Two rules
 * would be two answers to the same question.
 */
/* The bytes of the escape sequence at s, or 0 where there is none. */
static size_t sc_esc(const char *s, size_t n)
{
    if (n < 2 || (unsigned char)s[0] != 0x1b)
        return 0;
    if (s[1] != '[')
        return 2; /* ESC 7, ESC 8 and the like: two bytes */
    size_t i = 2;
    while (i < n && (unsigned char)s[i] >= 0x20 && (unsigned char)s[i] < 0x40)
        i++;
    return i < n ? i + 1 : n;
}

/*
 * Where each screen row of one line begins, in bytes. Returns how many rows
 * the line takes (at least one, even when empty), and fills start[] with the
 * first max of them.
 */
/*
 * A list item's continuations line up under its text, not under its mark:
 * the columns taken by the indent, the mark and the space after it. Zero
 * for anything that is not a list item, so nothing else moves. The marks
 * are the bullet the markdown machine writes and a number with a dot or a
 * bracket, which models write and nothing translates.
 */
static int sc_hang(const char *s)
{
    size_t n = strlen(s), i = 0;
    int cols = 0, digits = 0;
    while (i < n) {
        size_t e = sc_esc(s + i, n - i);
        if (e) {
            i += e;
            continue;
        }
        if (s[i] == ' ' || s[i] == '\t') {
            cols++;
            i++;
            continue;
        }
        break;
    }
    if (i >= n)
        return 0;
    if (strncmp(s + i, "\342\200\242", 3) == 0) { /* the bullet */
        i += 3;
        cols += 1;
    } else {
        while (i < n && s[i] >= '0' && s[i] <= '9') {
            digits++;
            cols++;
            i++;
        }
        if (!digits || i >= n || (s[i] != '.' && s[i] != ')'))
            return 0;
        i++;
        cols++;
    }
    if (i >= n || s[i] != ' ')
        return 0;
    while (i < n && s[i] == ' ') { /* the space, and any that follow it */
        cols++;
        i++;
    }
    return cols;
}

static int sc_wrap(const char *s, int cols, int *start, int max)
{
    size_t n = strlen(s);
    int rows = 1, used = 0;
    size_t i = 0, space = 0; /* the last place this row could break at */
    int hang = sc_hang(s), limit = cols;
    if (hang >= cols - 8) /* not at a width where it would help */
        hang = 0;
    if (max > 0)
        start[0] = 0;
    while (i < n) {
        size_t e = sc_esc(s + i, n - i);
        if (e) { /* colours take no room */
            i += e;
            continue;
        }
        size_t bytes;
        int w = term_char_cols(s + i, n - i, &bytes);
        if (!bytes)
            bytes = 1;
        /* a space at the end of a row is not seen: let it hang over the
           edge, or a word that fits exactly would be pushed down by it */
        if (used + w > limit && used > 0 && s[i] != ' ') {
            /*
             * The row is full. It breaks after the last space if there was
             * one, so a word is never cut in half; a word longer than the
             * whole row has nowhere to break and is cut where it is.
             */
            size_t at = space ? space : i;
            if (rows < max)
                start[rows] = (int)at;
            rows++;
            used = 0;
            space = 0;
            limit = cols - hang; /* the continuations are pushed right */
            if (at != i) { /* the word goes on the new row, from its start */
                i = at;
                continue;
            }
        }
        if (s[i] == ' ')
            space = i + 1; /* the next row would start after it */
        used += w;
        i += bytes;
    }
    return rows;
}

/*
 * One blank line between one block of output and the next - the line the
 * user wrote, the reply, the counters, what a command answers - so that the
 * eye finds where each begins. Every block calls term_gap() before printing
 * anything: the first one of the session prints nothing, the others a
 * newline. Not on a pipe, where a script reads the lines and the spacing is
 * not for it to guess at.
 */
/* Into the conversation: through the store if there is one. */
static void term_write(struct janas_term *t, const char *s, size_t n)
{
    if (t->sink)
        t->sink(t->sink_arg, s, n);
    else
        fwrite(s, 1, n, stdout);
}

static void term_puts(struct janas_term *t, const char *s)
{
    term_write(t, s, strlen(s));
}

__attribute__((format(printf, 2, 3))) static void
term_printf(struct janas_term *t, const char *fmt, ...)
{
    char small[1024], *buf = small;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(small, sizeof(small), fmt, ap);
    va_end(ap);
    if (n < 0)
        return;
    if ((size_t)n >= sizeof(small)) {
        buf = malloc((size_t)n + 1);
        if (!buf)
            return;
        va_start(ap, fmt);
        vsnprintf(buf, (size_t)n + 1, fmt, ap);
        va_end(ap);
    }
    term_write(t, buf, (size_t)n);
    if (buf != small)
        free(buf);
}

static void term_gap(struct janas_term *t)
{
    if (!t->tty)
        return;
    if (t->gapped)
        term_write(t, "\n", 1);
    t->gapped = 1;
}

static void term_size(struct janas_term *t)
{
    struct winsize ws;
    t->rows = 24;
    t->cols = 80;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 4 &&
        ws.ws_col > 20) {
        t->rows = ws.ws_row;
        t->cols = ws.ws_col;
    }
}

static void term_init(struct janas_term *t)
{
    memset(t, 0, sizeof(*t));
    t->tty = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
    const char *no = getenv("NO_COLOR"), *term = getenv("TERM");
    t->colour = t->tty && !(no && *no) && !(term && !strcmp(term, "dumb"));
    /* the box the line is written in, and the status line under it */
    t->footer_rows = t->tty ? 4 : 0;
    term_size(t);
}

/* Raw mode for the line editor: keys as they are pressed, no echo. */
static void term_raw(struct janas_term *t, int on)
{
    if (!t->tty || t->raw == on)
        return;
    if (on) {
        if (tcgetattr(STDIN_FILENO, &t->saved) != 0)
            return;
        struct termios r = t->saved;
        r.c_lflag &= (tcflag_t) ~(ICANON | ECHO);
        r.c_cc[VMIN] = 1;
        r.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &r) != 0)
            return;
    } else if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &t->saved) != 0)
        return;
    t->raw = on;
}

/*
 * The scrolling region: everything the conversation prints stays above the
 * footer. Called again after a resize.
 */
static void term_region(struct janas_term *t)
{
    if (!t->tty)
        return;
    int last = t->rows - t->footer_rows;
    if (last < 2)
        last = 2;
    printf("\033[1;%dr", last);
    fflush(stdout);
}

static void term_region_off(struct janas_term *t)
{
    if (!t->tty)
        return;
    printf("\033[r\033[%d;1H\033[0m\n", t->rows);
    fflush(stdout);
}

/*
 * The footer and the input rule are drawn outside the scrolling region, by
 * addressing their rows and putting the cursor back: the region itself is
 * set once (term_region) and never touched again, or the conversation would
 * scroll over them.
 */
/* The most rows one line of text may take when it is laid out. */
#define JANAS_BANNER_ROWS 32

/* The bytes of s that fit in `cols` columns, cut at a character. */
static size_t term_cut(const char *s, int cols)
{
    size_t n = 0, left = strlen(s);
    int w = 0;
    while (left) {
        size_t took;
        int cw = term_char_cols(s + n, left, &took);
        if (w + cw > cols)
            break;
        w += cw;
        n += took;
        left -= took;
    }
    return n;
}

static void term_footer(struct janas_term *t, const char *text)
{
    if (!t->tty)
        return;
    /* cut where the columns run out, not where the bytes do: a name with
       an accent in it took more bytes than columns, so the line was cut
       short and then padded past the edge of the window */
    size_t n = term_cut(text, t->cols);
    int w = term_cols(text, n);
    /* reset first: saving the cursor saves the colours with it, so a
       footer drawn while the model is writing its reasoning in grey came
       out grey, and white the rest of the time */
    /* one place to save the cursor in: while the editor holds it, the
       footer moves freely and the editor puts the cursor back itself */
    printf("%s\033[%d;1H%s%s%.*s%*s%s%s", t->editing ? "" : "\0337", t->rows,
           T_RESET(t), T_DIM(t), (int)n, text, t->cols - w, "", T_RESET(t),
           t->editing ? "" : "\0338");
    fflush(stdout);
}

/*
 * The line being written, between two rules: no sides, so a long line has
 * the whole width. Leaves the cursor on it, `at` columns after the prompt.
 */
/* hl: bytes at the start of text that name a command the chat knows. */
/*
 * How many rows of text the line needs, and how many the box may give it:
 * a third of the window at most, so that a long message never leaves the
 * conversation without room. The prompt holds the first columns of the
 * first row and the rest is written under it, so every row has the same
 * width to fill.
 */
/*
 * How many rows of text the message needs, and how many the box may give
 * it: a third of the window at most, so that a long one never leaves the
 * conversation without room. The prompt holds the first columns of the
 * first row and the rest is written under it, so every row has the same
 * width to fill. A newline in the message breaks a row wherever it is: a
 * message of several lines is written as several lines.
 */
#define TERM_BOX_MAXROWS 512

static int term_box_layout(struct janas_term *t, const char *prompt,
                           const char *text, int nbytes, int *begin, int *stop);

static int term_box_rows(struct janas_term *t, const char *prompt,
                         const char *text, int nbytes, int *most)
{
    int cap = (t->rows - 4) / 3;
    if (cap < 1)
        cap = 1;
    if (most)
        *most = cap;
    /* counted by the rule that draws it, not by one of its own: two of them
       disagreed about where a word goes and the box came out a row short */
    static int begin[TERM_BOX_MAXROWS], stop[TERM_BOX_MAXROWS];
    return term_box_layout(t, prompt, text, nbytes, begin, stop);
}

/* Where each row of the message begins, and where it ends: a row broken by
   a newline ends before it, so the newline itself is never printed. */
static int term_box_layout(struct janas_term *t, const char *prompt,
                           const char *text, int nbytes, int *begin, int *stop)
{
    int pc = term_cols(prompt, strlen(prompt));
    int room = t->cols - pc - 1;
    if (room < 4)
        room = 4;
    int rows = 1, used = 0, space = 0; /* after the last space on this row */
    begin[0] = 0;
    for (int i = 0; i < nbytes;) {
        if (text[i] == '\n') {
            if (rows < TERM_BOX_MAXROWS) {
                stop[rows - 1] = i;
                begin[rows] = i + 1;
            }
            rows++;
            used = 0;
            space = 0;
            i++;
            continue;
        }
        size_t bytes;
        int w = term_char_cols(text + i, (size_t)(nbytes - i), &bytes);
        /* the same rule the conversation wraps by: the row breaks after the
           last space, so a word is never cut in half, and a space at the
           end of a row hangs over the edge instead of pushing a word that
           fits exactly down to the next one */
        if (used + w > room && used > 0 && text[i] != ' ') {
            int at = space ? space : i;
            if (rows < TERM_BOX_MAXROWS) {
                stop[rows - 1] = at;
                begin[rows] = at;
            }
            rows++;
            used = 0;
            space = 0;
            if (at != i) { /* the word goes down whole, from its start */
                i = at;
                continue;
            }
        }
        if (text[i] == ' ')
            space = i + 1;
        used += w;
        i += (int)(bytes ? bytes : 1);
    }
    if (rows <= TERM_BOX_MAXROWS)
        stop[rows - 1] = nbytes;
    return rows;
}

/*
 * Where the cursor is in the message: which row of the box, which column of
 * that row, and how many rows there are in all. Moving up and down a message
 * of several lines is done on these, so it follows what is on the screen and
 * not what is in the buffer.
 */
static int term_box_find(struct janas_term *t, const char *prompt,
                         const char *text, int nbytes, int pos, int *col,
                         int *total)
{
    static int begin[TERM_BOX_MAXROWS], stop[TERM_BOX_MAXROWS];
    int rows = term_box_layout(t, prompt, text, nbytes, begin, stop);
    if (rows > TERM_BOX_MAXROWS)
        rows = TERM_BOX_MAXROWS;
    if (total)
        *total = rows;
    for (int r = 0; r < rows; r++)
        if (pos >= begin[r] && pos <= stop[r]) {
            if (col)
                *col = term_cols(text + begin[r], (size_t)(pos - begin[r]));
            return r;
        }
    if (col)
        *col = 0;
    return rows - 1;
}

/* The byte at that column of that row, or the end of the row if it is
   shorter: a column kept while moving up and down, as an editor does. */
static int term_box_at(struct janas_term *t, const char *prompt,
                       const char *text, int nbytes, int row, int col)
{
    static int begin[TERM_BOX_MAXROWS], stop[TERM_BOX_MAXROWS];
    int rows = term_box_layout(t, prompt, text, nbytes, begin, stop);
    if (rows > TERM_BOX_MAXROWS)
        rows = TERM_BOX_MAXROWS;
    if (row < 0)
        row = 0;
    if (row >= rows)
        row = rows - 1;
    int at = begin[row], used = 0;
    while (at < stop[row]) {
        size_t bytes;
        int w = term_char_cols(text + at, (size_t)(stop[row] - at), &bytes);
        if (used + w > col)
            break;
        used += w;
        at += (int)(bytes ? bytes : 1);
    }
    return at;
}

/*
 * The message being written, between two rules, over as many rows as it
 * takes up to the cap: the box grows under the conversation while the
 * message does and goes back to one row when it has been sent. `pos` is
 * where the cursor is, in bytes; `rows` is what the caller has already made
 * room for (see term_box_rows).
 */
static void term_box(struct janas_term *t, const char *prompt, const char *text,
                     int nbytes, int pos, int hl, int rows)
{
    if (!t->tty)
        return;
    if (rows < 1)
        rows = 1;
    int pc = term_cols(prompt, strlen(prompt));
    int top = t->rows - 2 - rows; /* the rule above the text */
    printf("%s%s\033[%d;1H\033[K", T_RESET(t), T_DIM(t), top);
    for (int i = 0; i < t->cols; i++)
        printf("─");
    printf("\033[%d;1H\033[K", t->rows - 1);
    for (int i = 0; i < t->cols; i++)
        printf("─");
    printf("%s", T_RESET(t));

    static int begin[TERM_BOX_MAXROWS], stop[TERM_BOX_MAXROWS];
    int total = term_box_layout(t, prompt, text, nbytes, begin, stop);
    if (total > TERM_BOX_MAXROWS)
        total = TERM_BOX_MAXROWS;
    /* the row the cursor is on, and its column there */
    int at_row = total - 1, at_col = 0;
    for (int r = 0; r < total; r++)
        if (pos >= begin[r] && pos <= stop[r]) {
            at_row = r;
            at_col = term_cols(text + begin[r], (size_t)(pos - begin[r]));
            break;
        }
    /* the space a break leaves at the end of a row hangs over the edge, so
       a cursor just after it would be drawn outside the box: it rests on
       the last column instead, which is where the space is */
    if (1 + pc + at_col > t->cols)
        at_col = t->cols - pc - 1;
    if (at_col < 0)
        at_col = 0;
    int first = total > rows ? total - rows : 0;
    if (at_row < first) /* keep the cursor in sight */
        first = at_row;
    for (int r = 0; r < rows; r++) {
        int line = first + r;
        printf("\033[%d;1H\033[K", top + 1 + r);
        if (line >= total)
            continue;
        if (line == 0)
            printf("%s", prompt);
        else
            printf("%*s", pc, "");
        int from = begin[line], to = stop[line];
        int h = hl - from; /* the command, where it is still in sight */
        if (h < 0)
            h = 0;
        if (h > to - from)
            h = to - from;
        if (h > 0)
            printf("%s%.*s%s", T_CMD(t), h, text + from, T_FG(t));
        printf("%.*s", to - from - h, text + from + h);
    }
    printf("\033[%d;%dH", top + 1 + (at_row - first), 1 + pc + at_col);
    fflush(stdout);
}

/*
 * The line as it was sent, repeated in the conversation: the prompt in the
 * first columns, the text wrapped under itself so the prompt's column stays
 * empty, all of it on the faint ground.
 */
static void term_echo(struct janas_term *t, const char *prompt,
                      const char *text, int hl)
{
    int pc = term_cols(prompt, strlen(prompt));
    int room = t->cols - pc;
    if (room < 8 || !t->tty)
        room = 1 << 30; /* narrow or not a terminal: one line, as it comes */
    const char *at = text;
    int first = 1;
    do {
        int cols = 0;
        const char *end = at, *space = NULL;
        while (*end && *end != '\n') { /* a newline breaks the row here */
            size_t used;
            int w = term_char_cols(end, strlen(end), &used);
            if (cols + w > room && end != at && *end != ' ') {
                /* after the last space, so a word is never cut in half; a
                   word longer than the row has nowhere to break */
                if (space && space > at)
                    end = space;
                break; /* never nothing: a wide character on a narrow line */
            }
            if (*end == ' ')
                space = end + (used ? used : 1);
            cols += w;
            end += used ? used : 1;
        }
        int h = (int)(hl - (at - text)); /* of this piece, the command */
        if (h < 0)
            h = 0;
        if (h > (int)(end - at))
            h = (int)(end - at);
        if (first)
            term_printf(t, "%s%s", T_BAR(t), prompt);
        else /* under the column the text starts at */
            term_printf(t, "%s%*s", T_BAR(t), pc, "");
        if (h > 0)
            term_printf(t, "%s%.*s%s", T_CMD(t), h, at, T_FG(t));
        term_printf(t, "%.*s%s\n", (int)(end - at) - h, at + h, T_RESET(t));
        first = 0;
        at = end;
        if (*at == '\n')
            at++; /* it did its work: it is not written */
    } while (*at);
}

/* Wipes the rules and the line, for when the line has been taken. */
static void term_box_clear(struct janas_term *t)
{
    if (!t->tty)
        return;
    for (int r = t->rows - t->footer_rows + 1; r <= t->rows - 1; r++)
        printf("\033[%d;1H\033[K", r);
    fflush(stdout);
}

/* A page of its own for long text (/help): the window is given back
   untouched when it is done. */
/*
 * The keys a page reads: a byte as it came, or one of these where it
 * arrived as an escape sequence. Escape on its own is a key too, and is
 * told from the start of a sequence by waiting a moment for what follows.
 */
enum {
    TERM_KEY_ESC = 0x1000,
    TERM_KEY_UP,
    TERM_KEY_DOWN,
    TERM_KEY_PGUP,
    TERM_KEY_PGDN,
    TERM_KEY_HOME,
    TERM_KEY_END,
    TERM_KEY_OTHER
};

static int term_key(void)
{
    unsigned char c;
    if (read(STDIN_FILENO, &c, 1) != 1)
        return TERM_KEY_ESC; /* the input ended: let the page close */
    if (c != 27)
        return c;
    struct pollfd pf = {STDIN_FILENO, POLLIN, 0};
    unsigned char lead = 0, final = 0;
    char par[16];
    int pn = 0;
    if (poll(&pf, 1, 50) <= 0 || read(STDIN_FILENO, &lead, 1) != 1 ||
        (lead != '[' && lead != 'O'))
        return TERM_KEY_ESC; /* nothing followed it: Escape itself */
    for (;;) {               /* the parameters, then the byte that names it */
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
    switch (final) {
    case 'A':
        return TERM_KEY_UP;
    case 'B':
        return TERM_KEY_DOWN;
    case 'H':
        return TERM_KEY_HOME;
    case 'F':
        return TERM_KEY_END;
    case '~':
        if (strcmp(par, "5") == 0)
            return TERM_KEY_PGUP;
        if (strcmp(par, "6") == 0)
            return TERM_KEY_PGDN;
        if (strcmp(par, "1") == 0 || strcmp(par, "7") == 0)
            return TERM_KEY_HOME;
        if (strcmp(par, "4") == 0 || strcmp(par, "8") == 0)
            return TERM_KEY_END;
        break;
    default:
        break;
    }
    return TERM_KEY_OTHER;
}

/*
 * A page laid over the conversation: /help and /about are read here. It
 * was a screenful printed once and given back on any key, which in a
 * narrow window meant a text the terminal broke where it liked, no way
 * down it, and a page that shut at the first key touched while looking
 * for one. So: the text is laid out to the window with the rule everything
 * else uses, the arrows and the page keys move in it, and ESC - and
 * only ESC, or q - closes it.
 */
#define TERM_PAGE_ROWS 4096

static void term_pager(struct janas_term *t, const char *text)
{
    if (!t->tty) {
        fputs(text, stdout);
        return;
    }
    char *copy = strdup(text);
    if (!copy) {
        fputs(text, stdout);
        return;
    }
    struct {
        const char *s;
        int len;
    } *row = malloc(sizeof(*row) * 64);
    int n = 0, cap = row ? 64 : 0;
    for (char *line = copy;;) { /* an empty line is a row too */
        char *nl = strchr(line, '\n');
        if (nl)
            *nl = 0;
        /* but the nothing after the last newline is not a line: a text
           that ends properly was leaving a blank row at the foot */
        if (!nl && !*line)
            break;
        for (const char *p = line;;) {
            int start[2];
            int rows = sc_wrap(p, t->cols, start, 2);
            int len = rows > 1 ? start[1] : (int)strlen(p);
            if (n == cap) {
                void *bigger = realloc(row, sizeof(*row) * (size_t)cap * 2);
                if (!bigger)
                    break;
                row = bigger;
                cap *= 2;
            }
            /* the space that hangs over the edge of a row is not part of
               it: kept, it would be copied out with the text */
            while (len > 0 && p[len - 1] == ' ')
                len--;
            row[n].s = p;
            row[n].len = len;
            n++;
            if (rows <= 1 || n >= TERM_PAGE_ROWS)
                break;
            p += len;
            /* a space that a break carried to the head of the next row is
               not a space the writer wrote: it is the one that ended the
               row above, and at the start of a row it only shifts the text */
            while (*p == ' ')
                p++;
        }
        if (!nl)
            break;
        line = nl + 1;
    }
    int page = t->rows - 1; /* the last row says the way out */
    if (page < 1)
        page = 1;
    int top = 0, most = n > page ? n - page : 0;
    term_raw(t, 1);
    for (;;) {
        printf("\033[2J\033[H");
        for (int i = 0; i < page && top + i < n; i++)
            printf("%.*s%s\n", row[top + i].len, row[top + i].s, T_RESET(t));
        /* the way out, said as fully as the window lets it be said */
        char hint[128];
        const char *say = " ESC closes, arrows and Page Up/Down move";
        if (term_cols(say, strlen(say)) + 6 > t->cols)
            say = " ESC closes";
        if (term_cols(say, strlen(say)) + 6 > t->cols)
            say = " ESC";
        if (most)
            snprintf(hint, sizeof(hint), "%s   %d%%", say,
                     (int)(100.0 * top / most + 0.5));
        else
            snprintf(hint, sizeof(hint), "%s", say);
        printf("\033[%d;1H%s%.*s%s", t->rows, T_DIM(t),
               (int)term_cut(hint, t->cols), hint, T_RESET(t));
        fflush(stdout);
        int k = term_key();
        if (k == TERM_KEY_ESC || k == 'q' || k == 'Q' || k == 3)
            break;
        switch (k) {
        case TERM_KEY_UP:
            top--;
            break;
        case TERM_KEY_DOWN:
            top++;
            break;
        case TERM_KEY_PGUP:
            top -= page - 1 > 0 ? page - 1 : 1;
            break;
        case TERM_KEY_PGDN:
        case ' ':
            top += page - 1 > 0 ? page - 1 : 1;
            break;
        case TERM_KEY_HOME:
            top = 0;
            break;
        case TERM_KEY_END:
            top = most;
            break;
        default: /* any other key does nothing: it is not a way out */
            break;
        }
        if (top > most)
            top = most;
        if (top < 0)
            top = 0;
    }
    term_raw(t, 0);
    free(row);
    free(copy);
}

static void term_page_begin(struct janas_term *t)
{
    if (t->tty) {
        printf("\033[?1049h\033[r\033[2J\033[H");
        fflush(stdout);
    }
}

static void term_page_end(struct janas_term *t)
{
    if (t->tty) {
        printf("\033[?1049l");
        term_region(t);
        fflush(stdout);
    }
}

/*
 * The banner, at the top of the window. The mascot is a small octopus drawn
 * in block characters: four rows of the same width, easy to replace.
 */
#define JANAS_MASCOT_ROWS 8
#define JANAS_MASCOT_COLS 19
static const char *const janas_mascot[JANAS_MASCOT_ROWS] = {
    "  \u2584\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2584    ",
    "\u2584\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2584  ",
    "\u2588\u2588\u2588 \u2584 \u2588\u2588\u2588\u2588\u2588 \u2584 \u2588\u2588\u2588 \u2584",
    "\u2588\u2588\u2588\u2584\u2584\u2584\u2588\u2588\u2588\u2588\u2588\u2584\u2584\u2584\u2588\u2588\u2588\u2588 ",
    "\u2580\u2588\u2588\u2588\u2588\u2588\u2584\u2580\u2580\u2580\u2584\u2588\u2588\u2588\u2588\u2588\u2580 \u2580",
    "  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588    ",
    " \u2588 \u2588 \u2588 \u2588 \u2588 \u2588 \u2588 \u2588   ",
    "\u2580 \u2580 \u2580 \u2580 \u2580 \u2580 \u2580 \u2580    "};

/*
 * Text beside the mascot, laid out rather than left to the edge of the
 * window: what does not fit beside the drawing goes on under the line
 * above, from the same column, and the block grows taller than the mascot
 * when there is more text than drawing. A big font is a narrow window, and
 * there the name of a model is three rows on its own. Where even that is
 * too narrow the drawing goes: a banner is worth nothing if it costs the
 * words their room. `most` is the rows it may take in all.
 */
/* The widest run without a space in it: what the layout cannot break. */
static int term_longest_word(const char *s)
{
    int most = 0, run = 0;
    for (size_t i = 0, n = strlen(s); i < n;) {
        size_t e = sc_esc(s + i, n - i);
        if (e) {
            i += e;
            continue;
        }
        size_t bytes;
        int w = term_char_cols(s + i, n - i, &bytes);
        if (!bytes)
            bytes = 1;
        run = s[i] == ' ' ? 0 : run + w;
        if (run > most)
            most = run;
        i += bytes;
    }
    return most;
}

static void term_beside(struct janas_term *t, const char *const *text,
                        const char *const *face, int n, int most, int stack)
{
    int room = t->cols - (JANAS_MASCOT_COLS + 2);
    int drawn = room >= 12;
    /* and it goes as well where it would cost a word its letters: a URL or
       a licence has no space to break at, and cut in three it says
       nothing. The drawing is the thing that can be done without */
    for (int k = 0; k < n && drawn; k++)
        if (term_longest_word(text[k]) > room &&
            term_longest_word(text[k]) <= t->cols)
            drawn = 0;
    /* where the drawing cannot stand beside the words it can stand over
       them, for a page whose point is to be looked at; the banner, which
       has a conversation to leave room for, does without it instead */
    if (!drawn && stack && t->cols >= JANAS_MASCOT_COLS) {
        for (int r = 0; r < JANAS_MASCOT_ROWS && r + 1 < most; r++)
            term_printf(t, "%s%s%s\n", T_BRAND(t), janas_mascot[r], T_RESET(t));
        term_printf(t, "\n");
        most -= JANAS_MASCOT_ROWS + 1;
    }
    if (!drawn)
        room = t->cols;
    int src[JANAS_BANNER_ROWS], from[JANAS_BANNER_ROWS], to[JANAS_BANNER_ROWS];
    int nrow = 0;
    for (int k = 0; k < n; k++) {
        int start[JANAS_BANNER_ROWS];
        int rows = sc_wrap(text[k], room, start, JANAS_BANNER_ROWS);
        if (rows > JANAS_BANNER_ROWS)
            rows = JANAS_BANNER_ROWS;
        for (int r = 0; r < rows && nrow < JANAS_BANNER_ROWS; r++) {
            src[nrow] = k;
            from[nrow] = start[r];
            to[nrow] = r + 1 < rows ? start[r + 1] : (int)strlen(text[k]);
            nrow++;
        }
    }
    /* the text sits beside the middle of the mascot, and pushes the block
       taller than the drawing when there is more of it than drawing */
    int rows = drawn && nrow < JANAS_MASCOT_ROWS ? JANAS_MASCOT_ROWS : nrow;
    int first =
        drawn && nrow < JANAS_MASCOT_ROWS ? (JANAS_MASCOT_ROWS - nrow) / 2 : 0;
    if (rows > most)
        rows = most > 1 ? most : 1;
    for (int r = 0; r < rows; r++) {
        int i = r - first;
        if (drawn) {
            if (r < JANAS_MASCOT_ROWS)
                term_printf(t, "%s%s%s  ", T_BRAND(t), janas_mascot[r],
                            T_RESET(t));
            else
                term_printf(t, "%*s  ", JANAS_MASCOT_COLS, "");
        }
        if (i >= 0 && i < nrow)
            term_printf(t, "%s%.*s%s", face[src[i]], to[i] - from[i],
                        text[src[i]] + from[i], T_RESET(t));
        term_printf(t, "\n");
    }
}

/*
 * A section of a page: its name on a line of its own, in the colour of the
 * program, then the text, then a blank line. Two columns side by side were
 * a table, and a table in a narrow window is a third of the room gone to
 * the labels; this reads the same at any width, and the page it goes on
 * lays the text out.
 */
static void term_field(struct janas_term *t, const char *name, const char *text)
{
    term_printf(t, "%s%s%s\n%s\n\n", T_BRAND(t), name, T_RESET(t), text);
}

static void term_banner(struct janas_term *t, const char *product,
                        const char *model, const char *line3)
{
    if (!t->tty) {
        term_printf(t, "%s\n%s\n", product, model);
        return;
    }
    /* the model is not here: the bar a row below the conversation says it
       and goes on saying it, and /about has the whole of it. Said in all
       three places it was said twice too often */
    const char *text[2] = {product, line3};
    const char *face[2] = {T_BOLD(t), T_DIM(t)};
    printf("\033[2J\033[H"); /* the cursor at the top: the banner goes there */
    term_region(t);
    printf("\033[1;1H");
    term_beside(t, text, face, 2, t->rows - t->footer_rows - 1, 0);
    term_printf(t, "\n");
    fflush(stdout);
}

#endif
