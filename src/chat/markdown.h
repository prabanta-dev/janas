/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * markdown.h - the marks a model writes, shown instead of printed.
 *
 * Models mark their replies up: **bold**, `code`, fenced blocks, headings,
 * bullets. Printed as they come the marks are noise; dropped, the reply
 * loses what they meant - a block of code is worth telling from prose. So
 * they are read and turned into what the terminal can do.
 *
 * The text arrives a piece at a time and a mark can be cut in half by the
 * end of a piece, so a few bytes are held back until they are either a mark
 * or ordinary text. What is held is never lost: the end of the reply flushes
 * it.
 *
 * An underscore is italic only where a word begins or ends, never inside
 * one, and only alone. In a conversation about code snake_case is
 * everywhere, and so are __init__ and friends: a renderer that treats every
 * _ as italic turns half of it into slanted nonsense. So a_b stays text, and
 * so does any run of two or more; _this_ and **_this_** are marks.
 */
#ifndef JANAS_CHAT_MARKDOWN_H
#define JANAS_CHAT_MARKDOWN_H

#include <string.h>

#include "term.h"

#define T_ITAL(t) ((t)->colour ? "\033[3m" : "")
#define T_NOBOLD(t) ((t)->colour ? "\033[22m" : "")
#define T_NOITAL(t) ((t)->colour ? "\033[23m" : "")
#define T_CODE(t) ((t)->colour ? "\033[38;5;109m" : "") /* as a command */

struct janas_md {
    struct janas_term *t;
    const char *base; /* the colour the reply is written in */
    int on;           /* marks are read, not printed */
    int fence;        /* inside a fenced block */
    int fence_info;   /* swallowing the language written after the fence */
    int bold, ital, mono;
    int bol;      /* the next byte starts a line */
    int pend_bol; /* what is held back was at the start of one */
    char last;    /* the last byte of text let through */
    int heading;  /* bold until the end of this line */
    char pend[8]; /* bytes held back: they may yet be a mark */
    int pn;
    /*
     * The spaces a line begins with, held until it is known whether a fence
     * opens after them. A fence that opens is swallowed whole, its line with
     * it, and a model that puts a code block inside a numbered list indents
     * it: printed, that indent stayed behind on a line whose end had been
     * swallowed, and the first line of code came out with its own indent
     * added to it.
     */
    char indent[16];
    int in;
};

static void md_init(struct janas_md *m, struct janas_term *t, int on)
{
    memset(m, 0, sizeof(*m));
    m->t = t;
    m->on = on;
    m->base = "";
    m->bol = 1;
}

/* The colour the text goes back to when a mark ends. */
static void md_plain(struct janas_md *m)
{
    term_printf(m->t, "%s%s", m->t->colour ? "\033[39m" : "", m->base);
}

static void md_flush_indent(struct janas_md *m)
{
    if (m->in) {
        term_write(m->t, m->indent, (size_t)m->in);
        m->in = 0;
    }
}

static void md_flush_pend(struct janas_md *m)
{
    md_flush_indent(m);
    if (m->pn) {
        term_write(m->t, m->pend, (size_t)m->pn);
        m->pn = 0;
    }
}

/* Where a line ends, whatever was open on it closes. */
static void md_end_line(struct janas_md *m)
{
    if (m->heading) {
        term_printf(m->t, "%s", T_NOBOLD(m->t));
        m->heading = 0;
    }
    if (m->ital) {
        term_printf(m->t, "%s", T_NOITAL(m->t));
        m->ital = 0;
    }
    if (m->bold) {
        term_printf(m->t, "%s", T_NOBOLD(m->t));
        m->bold = 0;
    }
    if (m->mono) {
        md_plain(m);
        m->mono = 0;
    }
}

/*
 * One byte through the machine. Returns the bytes it wants held back: the
 * caller passes them again with what follows.
 */
static void md_byte(struct janas_md *m, char c);

static void md_write(struct janas_md *m, const char *s, size_t n)
{
    if (!m->on) {
        term_write(m->t, s, n);
        return;
    }
    for (size_t i = 0; i < n; i++)
        md_byte(m, s[i]);
}

/* The reply is over. What was held back is decided as if the text ended
   there: a mark that closes is closed, not printed - a reply that ends in
   **bold** or `code` showed the last two stars or the last backtick. */
static void md_end(struct janas_md *m)
{
    if (!m->on)
        return;
    if (m->pn)
        md_byte(m, 0); /* 0: the end, decided but never printed */
    md_flush_indent(m);
    md_end_line(m);
    if (m->fence) {
        md_plain(m);
        m->fence = 0;
    }
}

static int md_alnum(char c)
{
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z');
}

static void md_byte(struct janas_md *m, char c)
{
    struct janas_term *t = m->t;
    if (c == '\n' && m->fence_info) { /* the rest of the fence line goes */
        m->fence_info = 0;
        m->bol = 1;
        m->last = 0;
        return;
    }
    if (m->fence_info)
        return;
    /*
     * What is being held is decided first, the newline included: a mark
     * that closes at the end of a line - and ** at the end of a line is
     * where a model puts most of them - arrives with the newline behind
     * it, and reading the newline first printed the two stars as text.
     */
    /* what is being held: decide it now that one more byte is known */
    if (m->pn) {
        m->pend[m->pn++] = c;
        if (m->pend[0] == '`') {
            if (m->pn == 2 && c == '`')
                return;                   /* two: it may still be three */
            if (m->pn == 3 && c == '`') { /* a fence */
                m->pn = 0;
                m->in = 0; /* the fence's own indent goes with it */
                m->fence = !m->fence;
                if (m->fence) {
                    term_printf(t, "%s", T_CODE(t));
                    m->fence_info = 1; /* the language named after it */
                } else
                    md_plain(m);
                return;
            }
            /* one or two backticks and then something else */
            int ticks = m->pn - 1;
            char last = c;
            m->pn = 0;
            for (int k = 0; k < ticks; k++) {
                m->mono = !m->mono;
                if (m->mono)
                    term_printf(t, "%s", T_CODE(t));
                else
                    md_plain(m);
            }
            if (last)
                md_byte(m, last);
            return;
        }
        if (m->pend[0] == '_') { /* see the comment at the top */
            m->pn = 0;
            if (c == '_') { /* a run of them: text, all of it */
                term_write(t, "__", 2);
                m->last = '_';
                return;
            }
            int opens = !m->ital && c && c != ' ' && c != '\t' && c != '\n' &&
                        !md_alnum(m->last);
            int closes = m->ital && m->last && m->last != ' ' &&
                         m->last != '\t' && !md_alnum(c);
            if (opens || closes) {
                m->ital = opens;
                term_printf(t, "%s", opens ? T_ITAL(t) : T_NOITAL(t));
            } else {
                term_write(t, "_", 1);
                m->last = '_';
            }
            if (c)
                md_byte(m, c);
            return;
        }
        if (m->pend[0] == '*') {
            if (m->pend_bol && m->pn == 2 && c == ' ') { /* a bullet */
                m->pn = 0;
                term_write(t, "\342\200\242 ", strlen("\342\200\242 "));
                return;
            }
            if (m->pn == 2 && c == '*')
                return; /* two stars: what follows says what they are */
            /*
             * A mark is attached to the text it marks: it opens only when
             * something follows it and closes only when something came
             * before. Without that rule a lone star in the prose - 2*3, a
             * footnote - turned the rest of the line slanted.
             */
            int stars = m->pn - 1;
            /* and never inside a word: 2*3 is a product, not emphasis,
               and a model writing prose about code writes it often */
            int opens = c && c != ' ' && c != '\t' && !md_alnum(m->last);
            int closes = m->last && m->last != ' ' && m->last != '\t';
            m->pn = 0;
            int *flag = stars == 2 ? &m->bold : &m->ital;
            if (!*flag && opens) {
                *flag = 1;
                term_printf(t, "%s", stars == 2 ? T_BOLD(t) : T_ITAL(t));
            } else if (*flag && closes) {
                *flag = 0;
                term_printf(t, "%s", stars == 2 ? T_NOBOLD(t) : T_NOITAL(t));
            } else { /* not a mark after all: the stars are text */
                for (int k = 0; k < stars; k++)
                    term_write(t, "*", 1);
                m->last = '*';
            }
            if (c)
                md_byte(m, c);
            return;
        }
        if (m->pend[0] == '#') { /* a heading, or a hash in the text */
            if (c == '#' && m->pn < 7)
                return;
            int hashes = m->pn - 1;
            m->pn = 0;
            if (c == ' ') {
                m->heading = 1;
                term_printf(t, "%s", T_BOLD(t));
                return;
            }
            term_write(t, "#######", (size_t)hashes); /* all of them */
            if (c)
                md_byte(m, c);
            return;
        }
        if (m->pend[0] == '-' || m->pend[0] == '+') { /* a bullet */
            char lead = m->pend[0];
            m->pn = 0;
            if (c == ' ') {
                term_write(t, "• ", strlen("• "));
                return;
            }
            term_write(t, &lead, 1);
            if (c)
                md_byte(m, c);
            return;
        }
        md_flush_pend(m);
        if (!c)
            return;
    }
    if (c == '\n') {
        md_end_line(m);
        m->in = 0; /* spaces at the end of a line are not worth keeping */
        term_write(t, "\n", 1);
        m->bol = 1;
        m->last = 0;
        return;
    }
    if (m->fence) { /* inside a block only the closing fence is a mark */
        if (m->bol && c == '`') {
            m->pend[m->pn++] = c;
            m->pend_bol = 1;
            m->bol = 0;
            return;
        }
        /*
         * Spaces do not end the start of a line, here as everywhere else.
         * A model that puts a code block inside a numbered list indents the
         * fence that closes it, and closing only on a backtick in the first
         * column left the rest of the reply inside the block: its stars and
         * its bullets came out as written, because inside a block nothing
         * else is a mark.
         */
        if (c == ' ' || c == '\t') {
            if (m->bol && m->in < (int)sizeof(m->indent)) {
                m->indent[m->in++] = c;
                m->last = c;
                return;
            }
        } else
            m->bol = 0;
        md_flush_indent(m);
        m->last = c;
        term_write(t, &c, 1);
        return;
    }
    if (m->bol && (c == '#' || c == '-' || c == '+' || c == '*')) {
        md_flush_indent(m); /* only a fence may still swallow its indent */
        m->pend[m->pn++] = c;
        m->pend_bol = 1;
        m->bol = 0;
        return;
    }
    if (c == '*' || c == '`' || (c == '_' && m->last != '_')) {
        if (c != '`')
            md_flush_indent(m);
        m->pend[m->pn++] = c;
        m->pend_bol = 0;
        m->bol = 0;
        return;
    }
    if (c == ' ' || c == '\t') {
        if (m->bol && m->in < (int)sizeof(m->indent)) {
            m->indent[m->in++] = c; /* it may be a fence's indent */
            m->last = c;
            return;
        }
    } else
        m->bol = 0;
    md_flush_indent(m);
    m->last = c;
    term_write(t, &c, 1);
}

#endif
