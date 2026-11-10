/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_menu.c - see model_menu.h.
 *
 * The menu takes the terminal's other screen, as a pager does, so that
 * leaving it gives the user back the window as it was. It is drawn whole at
 * every key: a few dozen short lines. Nothing is laid out in fixed columns:
 * a model's name on one line and what it asks for under it, the help at the
 * bottom wrapped by words, so that 34 columns are as good as 200.
 */
#include "model_menu.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/sysinfo.h"
#include "tty.h"

struct menu {
    const struct janas_model_item *it;
    int n, n_here;
    int tab; /* 0: the models here, 1: the ones to download */
    int sel[2], top[2];
    int info; /* the page about the chosen model is open */
    int rows, cols;
    int colour;
    int mem_gb; /* this machine's memory */
    int row;    /* the next line drawn, from 0 */
};

static volatile sig_atomic_t interrupted;

static void on_interrupt(int sig)
{
    (void)sig;
    interrupted = 1;
}

static const char *c_reset(const struct menu *m)
{
    return m->colour ? "\033[0m" : "";
}

static const char *c_bold(const struct menu *m)
{
    return m->colour ? "\033[1m" : "";
}

static const char *c_dim(const struct menu *m)
{
    return m->colour ? "\033[2m" : "";
}

static const char *c_mark(const struct menu *m) /* amber, as the user's */
{
    return m->colour ? "\033[1;38;5;179m" : "";
}

static const char *c_warn(const struct menu *m)
{
    return m->colour ? "\033[38;5;203m" : "";
}

static int list_n(const struct menu *m, int tab)
{
    return tab ? m->n - m->n_here : m->n_here;
}

static const struct janas_model_item *chosen(const struct menu *m)
{
    if (list_n(m, m->tab) == 0)
        return NULL;
    return &m->it[(m->tab ? m->n_here : 0) + m->sel[m->tab]];
}

/* One line of the screen in a style, cut at the window's width (columns
   counted as characters: the menu's own words and the catalog's are
   all narrow). */
static void line(struct menu *m, const char *style, const char *text)
{
    if (m->row >= m->rows)
        return;
    printf("\033[%d;1H%s", m->row + 1, style);
    int c = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if ((*p & 0xC0) != 0x80 && ++c > m->cols)
            break;
        putchar(*p);
    }
    printf("%s\033[K", c_reset(m));
    m->row++;
}

/* A paragraph wrapped by words at the window's width, from the left
   edge plus indent; at most max lines (-1: as many as there is room
   for). The number of lines it took, or would take when draw is 0. */
static int para(struct menu *m, const char *style, const char *text, int indent,
                int max, int draw)
{
    int width = m->cols - indent;
    if (width < 8)
        width = 8;
    int lines = 0;
    const char *p = text;
    while (*p && (max < 0 || lines < max)) {
        while (*p == ' ')
            p++;
        if (!*p)
            break;
        /* as many words as fit; a word longer than the line is cut */
        const char *end = p, *fit = NULL;
        int c = 0;
        while (*end) {
            const char *w = end;
            int wc = 0;
            while (*w && *w != ' ') {
                if ((*w & 0xC0) != 0x80)
                    wc++;
                w++;
            }
            if (c + (c ? 1 : 0) + wc > width)
                break;
            c += (c ? 1 : 0) + wc;
            fit = w;
            end = w;
            while (*end == ' ')
                end++;
        }
        if (!fit) { /* one word wider than the line */
            fit = p;
            for (int k = 0; *fit && *fit != ' ' && k < width; fit++)
                if ((*fit & 0xC0) != 0x80)
                    k++;
            end = fit;
        }
        if (draw) {
            char buf[1024];
            int len = (int)(fit - p);
            if (len > (int)sizeof buf - 64)
                len = (int)sizeof buf - 64;
            snprintf(buf, sizeof buf, "%*s%.*s", indent, "", len, p);
            line(m, style, buf);
        }
        lines++;
        p = end;
    }
    return lines;
}

static int too_big(const struct menu *m, const struct janas_model_item *it)
{
    return it->ram_gb && m->mem_gb && it->ram_gb > m->mem_gb;
}

/* What a model asks for, under its name. */
static const char *detail(const struct menu *m,
                          const struct janas_model_item *it, char *buf,
                          size_t len)
{
    int n = snprintf(buf, len, "%.1f GB%s", it->gb,
                     it->path[0] ? "" : " to download");
    if (it->ram_gb && n > 0 && (size_t)n < len)
        n += snprintf(buf + n, len - (size_t)n, ", for %d GB of memory or more",
                      it->ram_gb);
    if (too_big(m, it) && n > 0 && (size_t)n < len)
        n += snprintf(buf + n, len - (size_t)n, ": this machine has %d GB",
                      m->mem_gb);
    if (!it->entry && n > 0 && (size_t)n < len)
        snprintf(buf + n, len - (size_t)n, ", not in Janas' list");
    return buf;
}

static const char *help(const struct menu *m)
{
    if (m->info)
        return chosen(m) && !chosen(m)->path[0] ? "Enter download · Esc back"
                                                : "Enter open · Esc back";
    return m->tab ? "↑↓ move · ←→ the models here · Enter download · "
                    "I about · Esc leave"
                  : "↑↓ move · ←→ the models to download · Enter open · "
                    "I about · Esc leave";
}

static void draw_list(struct menu *m)
{
    char buf[512];
    line(m, c_bold(m), "Janas · choose a model");
    snprintf(buf, sizeof buf,
             m->tab ? " Here: %d   [To download: %d]"
                    : "[Here: %d]   To download: %d ",
             m->n_here, m->n - m->n_here);
    line(m, "", buf);
    line(m, "", "");
    int foot = para(m, "", help(m), 0, -1, 0);
    int room = m->rows - m->row - foot - 1;
    int t = m->tab, k = list_n(m, t);
    if (k == 0)
        para(m, c_dim(m),
             t ? "Every model of Janas' list is here."
               : "No model here yet: → shows the ones to download.",
             0, 2, 1);
    /* the first model shown: the chosen one and those above it fit, each
       its name and what it asks for wrapped under it */
    const struct janas_model_item *base = &m->it[t ? m->n_here : 0];
    char d[512];
    if (m->sel[t] < m->top[t])
        m->top[t] = m->sel[t];
    for (;;) {
        int used = 0;
        for (int i = m->top[t]; i <= m->sel[t]; i++)
            used += 1 + para(m, "", detail(m, &base[i], d, sizeof d), 4, -1, 0);
        if (used <= room || m->top[t] >= m->sel[t])
            break;
        m->top[t]++;
    }
    for (int i = m->top[t]; i < k; i++) {
        const struct janas_model_item *it = &base[i];
        detail(m, it, d, sizeof d);
        int h = 1 + para(m, "", d, 4, -1, 0);
        if (i > m->top[t] && h > room)
            break;
        room -= h;
        int on = i == m->sel[t];
        snprintf(buf, sizeof buf, "%s %s", on ? "›" : " ", it->name);
        line(m, on ? c_mark(m) : "", buf);
        para(m, too_big(m, it) ? c_warn(m) : c_dim(m), d, 4, -1, 1);
    }
    while (m->row < m->rows - foot)
        line(m, "", "");
    para(m, c_dim(m), help(m), 0, -1, 1);
}

static void draw_info(struct menu *m)
{
    const struct janas_model_item *it = chosen(m);
    char buf[4600];
    int foot = para(m, "", help(m), 0, -1, 0);
    line(m, c_bold(m), it->name);
    line(m, "", "");
    para(m, "",
         it->entry ? it->entry->what
                   : "A model of no entry of Janas' list, found in the "
                     "models' folder.",
         0, -1, 1);
    line(m, "", "");
    if (it->path[0])
        snprintf(buf, sizeof buf, "Here: %.1f GB, %s", it->gb, it->path);
    else if (it->entry->jns_repo)
        snprintf(buf, sizeof buf,
                 "To download: %.1f GB, converted already, from "
                 "huggingface.co/%s",
                 it->gb, it->entry->jns_repo);
    else
        snprintf(buf, sizeof buf,
                 "To download: %.1f GB from huggingface.co/%s, converted "
                 "here (about %.0f GB free while it is)",
                 it->gb, it->entry->repo, it->gb * 2.0 + 0.5);
    para(m, "", buf, 0, -1, 1);
    if (it->ram_gb) {
        line(m, "", "");
        snprintf(buf, sizeof buf,
                 "Memory: meant for %d GB or more; this machine has %d GB.",
                 it->ram_gb, m->mem_gb);
        para(m, "", buf, 0, -1, 1);
        if (too_big(m, it))
            para(m, c_warn(m),
                 "With less, it reads from the disk what memory does not "
                 "hold, and runs slower.",
                 0, -1, 1);
    }
    while (m->row < m->rows - foot)
        line(m, "", "");
    para(m, c_dim(m), help(m), 0, -1, 1);
}

static void draw(struct menu *m)
{
    m->row = 0;
    if (m->info && chosen(m))
        draw_info(m);
    else
        draw_list(m);
    fflush(stdout);
}

enum {
    K_NONE,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_PGUP,
    K_PGDN,
    K_HOME,
    K_END,
    K_ENTER,
    K_ESC,
    K_INFO
};

/* A key, the arrows and the like read from the sequences an xterm (and
   Windows' console in its virtual-terminal mode) sends. */
static int key(void)
{
    unsigned char c;
    if (tty_byte(&c) != 1)
        return K_ESC;
    if (c == '\r' || c == '\n')
        return K_ENTER;
    if (c == 'i' || c == 'I')
        return K_INFO;
    if (c == '\t')
        return K_RIGHT;
    if (c != 27)
        return K_NONE;
    if (tty_wait(50) <= 0)
        return K_ESC; /* Esc alone */
    unsigned char b;
    if (tty_byte(&b) != 1 || (b != '[' && b != 'O'))
        return K_NONE;
    int num = 0;
    for (;;) {
        if (tty_wait(50) <= 0 || tty_byte(&b) != 1)
            return K_NONE;
        if (b >= '0' && b <= '9')
            num = num * 10 + (b - '0');
        else if (b >= 0x40 && b <= 0x7e)
            break;
    }
    switch (b) {
    case 'A':
        return K_UP;
    case 'B':
        return K_DOWN;
    case 'C':
        return K_RIGHT;
    case 'D':
        return K_LEFT;
    case 'H':
        return K_HOME;
    case 'F':
        return K_END;
    case '~':
        return num == 5               ? K_PGUP
               : num == 6             ? K_PGDN
               : num == 1 || num == 7 ? K_HOME
               : num == 4 || num == 8 ? K_END
                                      : K_NONE;
    default:
        return K_NONE;
    }
}

int chat_model_menu(const char *dir, const struct janas_model_item *items,
                    int n, int n_here)
{
    (void)dir;
    const char *no = getenv("NO_COLOR"), *term = getenv("TERM");
    struct menu m = {.it = items,
                     .n = n,
                     .n_here = n_here,
                     .tab = n_here ? 0 : 1,
                     .colour = !(no && *no) && !(term && !strcmp(term, "dumb")),
                     /* in GB as the catalog has them: a machine of
                        32 GB shows 31 GiB, and is meant for its 32 GB
                        models */
                     .mem_gb = (int)((double)janas_mem_total() / 1e9 + 0.5)};
    tty_setup();
    if (tty_raw(1) != 0)
        return -1;
    interrupted = 0;
    void (*old)(int) = signal(SIGINT, on_interrupt);
    fputs("\033[?1049h\033[?25l", stdout); /* the other screen, no cursor */
    int pick = -1, done = 0, dirty = 1;
    while (!done && !interrupted) {
        int rows = m.rows, cols = m.cols;
        if (tty_size(&m.rows, &m.cols) != 0 || m.rows < 4 || m.cols < 8)
            m.rows = 24, m.cols = 80;
        if (dirty || rows != m.rows || cols != m.cols)
            draw(&m);
        dirty = 0;
        /* a change of size is seen within a quarter of a second, where
           no signal says it */
        int r = tty_wait(250);
        if (r <= 0)
            continue;
        dirty = 1;
        int t = m.tab, k = list_n(&m, t), page = m.rows / 2 - 3;
        if (page < 1)
            page = 1;
        switch (key()) {
        case K_UP:
            if (m.sel[t] > 0)
                m.sel[t]--;
            break;
        case K_DOWN:
            if (m.sel[t] + 1 < k)
                m.sel[t]++;
            break;
        case K_PGUP:
            m.sel[t] = m.sel[t] > page ? m.sel[t] - page : 0;
            break;
        case K_PGDN:
            m.sel[t] = m.sel[t] + page < k ? m.sel[t] + page : k ? k - 1 : 0;
            break;
        case K_HOME:
            m.sel[t] = 0;
            break;
        case K_END:
            m.sel[t] = k ? k - 1 : 0;
            break;
        case K_LEFT:
        case K_RIGHT:
            if (!m.info)
                m.tab = !m.tab;
            break;
        case K_INFO:
            if (k)
                m.info = !m.info;
            break;
        case K_ENTER:
            if (k) {
                pick = (t ? n_here : 0) + m.sel[t];
                done = 1;
            }
            break;
        case K_ESC:
            if (m.info)
                m.info = 0;
            else
                done = 1;
            break;
        default:
            break;
        }
    }
    fputs("\033[?25h\033[?1049l", stdout);
    fflush(stdout);
    signal(SIGINT, old == SIG_ERR ? SIG_DFL : old);
    tty_raw(0);
    return interrupted ? -1 : pick;
}
