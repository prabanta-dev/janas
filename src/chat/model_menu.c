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
#include "menu_draw.h"
#include "tty.h"

struct menu {
    struct menu_screen s; /* first: the drawing takes a menu as one */
    const struct janas_model_item *it;
    int n, n_here;
    int tab; /* 0: the models here, 1: the ones to download */
    int sel[2], top[2];
    int info;   /* the page about the chosen model is open */
    int mem_gb; /* this machine's memory */
};

static volatile sig_atomic_t interrupted;

static void on_interrupt(int sig)
{
    (void)sig;
    interrupted = 1;
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
    menu_line(&m->s, menu_bold(&m->s), "Janas · choose a model");
    snprintf(buf, sizeof buf,
             m->tab ? " Here: %d   [To download: %d]"
                    : "[Here: %d]   To download: %d ",
             m->n_here, m->n - m->n_here);
    menu_line(&m->s, "", buf);
    menu_line(&m->s, "", "");
    int foot = menu_para(&m->s, "", help(m), 0, -1, 0);
    int room = m->s.rows - m->s.row - foot - 1;
    int t = m->tab, k = list_n(m, t);
    if (k == 0)
        menu_para(&m->s, menu_dim(&m->s),
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
            used += 1 + menu_para(&m->s, "", detail(m, &base[i], d, sizeof d),
                                  4, -1, 0);
        if (used <= room || m->top[t] >= m->sel[t])
            break;
        m->top[t]++;
    }
    for (int i = m->top[t]; i < k; i++) {
        const struct janas_model_item *it = &base[i];
        detail(m, it, d, sizeof d);
        int h = 1 + menu_para(&m->s, "", d, 4, -1, 0);
        if (i > m->top[t] && h > room)
            break;
        room -= h;
        int on = i == m->sel[t];
        snprintf(buf, sizeof buf, "%s %s", on ? "›" : " ", it->name);
        menu_line(&m->s, on ? menu_mark(&m->s) : "", buf);
        menu_para(&m->s, too_big(m, it) ? menu_warn(&m->s) : menu_dim(&m->s), d,
                  4, -1, 1);
    }
    while (m->s.row < m->s.rows - foot)
        menu_line(&m->s, "", "");
    menu_para(&m->s, menu_dim(&m->s), help(m), 0, -1, 1);
}

static void draw_info(struct menu *m)
{
    const struct janas_model_item *it = chosen(m);
    char buf[4600];
    int foot = menu_para(&m->s, "", help(m), 0, -1, 0);
    menu_line(&m->s, menu_bold(&m->s), it->name);
    menu_line(&m->s, "", "");
    menu_para(&m->s, "",
              it->entry ? it->entry->what
                        : "A model of no entry of Janas' list, found in the "
                          "models' folder.",
              0, -1, 1);
    menu_line(&m->s, "", "");
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
    menu_para(&m->s, "", buf, 0, -1, 1);
    if (it->ram_gb) {
        menu_line(&m->s, "", "");
        snprintf(buf, sizeof buf,
                 "Memory: meant for %d GB or more; this machine has %d GB.",
                 it->ram_gb, m->mem_gb);
        menu_para(&m->s, "", buf, 0, -1, 1);
        if (too_big(m, it))
            menu_para(&m->s, menu_warn(&m->s),
                      "With less, it reads from the disk what memory does not "
                      "hold, and runs slower.",
                      0, -1, 1);
    }
    while (m->s.row < m->s.rows - foot)
        menu_line(&m->s, "", "");
    menu_para(&m->s, menu_dim(&m->s), help(m), 0, -1, 1);
}

static void draw(struct menu *m)
{
    m->s.row = 0;
    if (m->info && chosen(m))
        draw_info(m);
    else
        draw_list(m);
    fflush(stdout);
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
                     .s.colour =
                         !(no && *no) && !(term && !strcmp(term, "dumb")),
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
        int rows = m.s.rows, cols = m.s.cols;
        if (tty_size(&m.s.rows, &m.s.cols) != 0 || m.s.rows < 4 || m.s.cols < 8)
            m.s.rows = 24, m.s.cols = 80;
        if (dirty || rows != m.s.rows || cols != m.s.cols)
            draw(&m);
        dirty = 0;
        /* a change of size is seen within a quarter of a second, where
           no signal says it */
        int r = tty_wait(250);
        if (r <= 0)
            continue;
        dirty = 1;
        int t = m.tab, k = list_n(&m, t), page = m.s.rows / 2 - 3;
        if (page < 1)
            page = 1;
        switch (menu_key()) {
        case MENU_UP:
            if (m.sel[t] > 0)
                m.sel[t]--;
            break;
        case MENU_DOWN:
            if (m.sel[t] + 1 < k)
                m.sel[t]++;
            break;
        case MENU_PGUP:
            m.sel[t] = m.sel[t] > page ? m.sel[t] - page : 0;
            break;
        case MENU_PGDN:
            m.sel[t] = m.sel[t] + page < k ? m.sel[t] + page : k ? k - 1 : 0;
            break;
        case MENU_HOME:
            m.sel[t] = 0;
            break;
        case MENU_END:
            m.sel[t] = k ? k - 1 : 0;
            break;
        case MENU_LEFT:
        case MENU_RIGHT:
            if (!m.info)
                m.tab = !m.tab;
            break;
        case MENU_INFO:
            if (k)
                m.info = !m.info;
            break;
        case MENU_ENTER:
            if (k) {
                pick = (t ? n_here : 0) + m.sel[t];
                done = 1;
            }
            break;
        case MENU_ESC:
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
