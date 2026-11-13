/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * chats_menu.c - see chats_menu.h.
 *
 * Drawn as the models' menu is (menu_draw.h): the terminal's other screen,
 * the whole of it at every key, a conversation's title on one line and
 * what it is under it, wrapped by words.
 */
#include "chats_menu.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "janas/llm.h"
#include "menu_draw.h"
#include "tty.h"

int chat_list_load(struct chat_list *l)
{
    memset(l, 0, sizeof(*l));
    int32_t len = 0;
    janas_llm_chats_list(NULL, NULL, 0, &len);
    char *buf = malloc((size_t)len + 1);
    if (!buf ||
        janas_llm_chats_list(NULL, buf, len + 1, &len) != JANAS_LLM_OK) {
        free(buf);
        return -1;
    }
    int cap = 0;
    for (char *p = buf, *e; *p; p = e + 1) {
        if (!(e = strchr(p, '\n')))
            break;
        *e = 0;
        struct chat_item it = {0};
        int32_t nm = 0;
        if (janas_llm_chat_file_info(p, it.title, sizeof it.title, it.model,
                                     sizeof it.model, it.model_file,
                                     sizeof it.model_file, &it.updated,
                                     &nm) != JANAS_LLM_OK)
            continue;
        it.messages = nm;
        snprintf(it.path, sizeof it.path, "%s", p);
        if (!it.title[0])
            snprintf(it.title, sizeof it.title, "(no title)");
        if (l->n == cap) {
            cap = cap ? 2 * cap : 32;
            struct chat_item *t = realloc(l->it, (size_t)cap * sizeof(*t));
            if (!t)
                break;
            l->it = t;
        }
        l->it[l->n++] = it;
    }
    free(buf);
    return 0;
}

void chat_list_free(struct chat_list *l)
{
    free(l->it);
    memset(l, 0, sizeof(*l));
}

void chat_item_detail(const struct chat_item *it, char *buf, size_t len)
{
    time_t t = (time_t)it->updated;
    struct tm tm;
    localtime_r(&t, &tm);
    char when[40];
    strftime(when, sizeof when, "%Y-%m-%d %H:%M", &tm);
    snprintf(buf, len, "%s · %s · %d message%s", when,
             it->model[0] ? it->model : "?", it->messages,
             it->messages == 1 ? "" : "s");
}

static volatile sig_atomic_t interrupted;

static void on_interrupt(int sig)
{
    (void)sig;
    interrupted = 1;
}

struct menu {
    struct menu_screen s;
    const struct chat_list *l;
    int sel, top;
};

static const char HELP[] = "↑↓ move · Enter take it up · Esc leave";

static void draw(struct menu *m)
{
    m->s.row = 0;
    char buf[600], d[400];
    menu_line(&m->s, menu_bold(&m->s), "Janas · take up a conversation");
    menu_line(&m->s, "", "");
    int foot = menu_para(&m->s, "", HELP, 0, -1, 0);
    int room = m->s.rows - m->s.row - foot - 1;
    if (m->sel < m->top)
        m->top = m->sel;
    for (;;) { /* the chosen one and those above it fit */
        int used = 0;
        for (int i = m->top; i <= m->sel; i++) {
            chat_item_detail(&m->l->it[i], d, sizeof d);
            used += 1 + menu_para(&m->s, "", d, 4, -1, 0);
        }
        if (used <= room || m->top >= m->sel)
            break;
        m->top++;
    }
    for (int i = m->top; i < m->l->n; i++) {
        const struct chat_item *it = &m->l->it[i];
        chat_item_detail(it, d, sizeof d);
        int h = 1 + menu_para(&m->s, "", d, 4, -1, 0);
        if (i > m->top && h > room)
            break;
        room -= h;
        int on = i == m->sel;
        snprintf(buf, sizeof buf, "%s %s", on ? "›" : " ", it->title);
        menu_line(&m->s, on ? menu_mark(&m->s) : "", buf);
        menu_para(&m->s, menu_dim(&m->s), d, 4, -1, 1);
    }
    while (m->s.row < m->s.rows - foot)
        menu_line(&m->s, "", "");
    menu_para(&m->s, menu_dim(&m->s), HELP, 0, -1, 1);
    fflush(stdout);
}

int chat_list_menu(const struct chat_list *l)
{
    if (l->n == 0)
        return -1;
    const char *no = getenv("NO_COLOR"), *term = getenv("TERM");
    struct menu m = {
        .l = l, .s.colour = !(no && *no) && !(term && !strcmp(term, "dumb"))};
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
        if (tty_wait(250) <= 0)
            continue;
        dirty = 1;
        int page = m.s.rows / 2 - 2;
        if (page < 1)
            page = 1;
        switch (menu_key()) {
        case MENU_UP:
            if (m.sel > 0)
                m.sel--;
            break;
        case MENU_DOWN:
            if (m.sel + 1 < l->n)
                m.sel++;
            break;
        case MENU_PGUP:
            m.sel = m.sel > page ? m.sel - page : 0;
            break;
        case MENU_PGDN:
            m.sel = m.sel + page < l->n ? m.sel + page : l->n - 1;
            break;
        case MENU_HOME:
            m.sel = 0;
            break;
        case MENU_END:
            m.sel = l->n - 1;
            break;
        case MENU_ENTER:
            pick = m.sel;
            done = 1;
            break;
        case MENU_ESC:
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

char *chat_message_dup(janas_llm_chat *c, int32_t i, int32_t what,
                       int32_t *role)
{
    int32_t len = 0;
    janas_llm_chat_message(c, i, what, role, NULL, 0, &len);
    char *s = malloc((size_t)len + 1);
    if (s && janas_llm_chat_message(c, i, what, role, s, len + 1, &len) !=
                 JANAS_LLM_OK) {
        free(s);
        s = NULL;
    }
    return s;
}

/* The next word of *p, or the words in quotes; NULL at the end. */
static char *word(char **p)
{
    char *s = *p;
    while (*s == ' ')
        s++;
    if (!*s)
        return NULL;
    char *e;
    if (*s == '"' && (e = strchr(s + 1, '"'))) {
        *e = 0;
        *p = e + 1;
        return s + 1;
    }
    e = s + strcspn(s, " ");
    *p = *e ? e + 1 : e;
    *e = 0;
    return s;
}

/* A file's name from the title: its letters, digits and a few signs, the
   rest a dash; the time when there is no title. */
static void file_name(janas_llm_chat *c, int markdown, char *buf, size_t len)
{
    char t[256] = "";
    int32_t tl;
    janas_llm_chat_title(c, t, sizeof t, &tl);
    size_t n = 0;
    for (const unsigned char *p = (const unsigned char *)t;
         *p && n + 8 < len && n < 80; p++) {
        int ok = *p >= 0x80 || (*p >= '0' && *p <= '9') ||
                 (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                 *p == '.' || *p == '_' || *p == '-';
        if (ok)
            buf[n++] = (char)*p;
        else if (n && buf[n - 1] != '-')
            buf[n++] = '-';
    }
    while (n && (buf[n - 1] == '-' || buf[n - 1] == '.'))
        n--;
    buf[n] = 0;
    if (!n) {
        time_t now = time(NULL);
        struct tm tm;
        localtime_r(&now, &tm);
        strftime(buf, len, "chat-%Y-%m-%d-%H%M%S", &tm);
        n = strlen(buf);
    }
    snprintf(buf + n, len - n, "%s", markdown ? ".md" : ".txt");
}

int chat_export(janas_llm_chat *c, char *arg, int markdown, char *say,
                size_t say_len)
{
    int over = 0;
    const char *title = NULL, *file = NULL;
    char *p = arg, *w;
    /* --title="multiseat systems" as --title "multiseat systems" */
    for (char *t = strstr(arg, "--title="); t; t = strstr(t + 8, "--title="))
        t[7] = ' ';
    while ((w = word(&p))) {
        if (!strcmp(w, "--overwrite")) {
            over = 1;
        } else if (!strcmp(w, "--title")) {
            if (!(title = word(&p))) {
                snprintf(say, say_len, "--title: which?");
                return -1;
            }

        } else if (!file) {
            file = w;
        } else {
            snprintf(say, say_len, "/export: one file, not %s too", w);
            return -1;
        }
    }
    if (title && janas_llm_chat_set_title(c, title, -1) != JANAS_LLM_OK) {
        snprintf(say, say_len, "%s", janas_llm_last_error());
        return -1;
    }
    char name[512];
    if (file)
        snprintf(name, sizeof name, "%s", file);
    else
        file_name(c, markdown, name, sizeof name);
    size_t nl = strlen(name);
    int text = nl > 4 && !strcmp(name + nl - 4, ".txt");
    int md = nl > 3 && !strcmp(name + nl - 3, ".md");
    int format = text       ? JANAS_LLM_EXPORT_TEXT
                 : md       ? JANAS_LLM_EXPORT_MARKDOWN
                 : markdown ? JANAS_LLM_EXPORT_MARKDOWN
                            : JANAS_LLM_EXPORT_TEXT;
    struct stat st;
    if (!over && stat(name, &st) == 0) {
        snprintf(say, say_len,
                 "%s is there already: --overwrite to write it "
                 "again",
                 name);
        return -1;
    }
    if (janas_llm_chat_export(c, name, format) != JANAS_LLM_OK) {
        snprintf(say, say_len, "%s", janas_llm_last_error());
        return -1;
    }
    snprintf(say, say_len, "written to %s", name);
    return 0;
}
