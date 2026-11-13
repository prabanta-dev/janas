/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * menu_draw.c - see menu_draw.h.
 */
#include "menu_draw.h"

#include <stdio.h>

#include "tty.h"

const char *menu_reset(const struct menu_screen *m)
{
    return m->colour ? "\033[0m" : "";
}

const char *menu_bold(const struct menu_screen *m)
{
    return m->colour ? "\033[1m" : "";
}

const char *menu_dim(const struct menu_screen *m)
{
    return m->colour ? "\033[2m" : "";
}

const char *menu_mark(const struct menu_screen *m) /* amber, as the user's */
{
    return m->colour ? "\033[1;38;5;179m" : "";
}

const char *menu_warn(const struct menu_screen *m)
{
    return m->colour ? "\033[38;5;203m" : "";
}

/* One line of the screen in a style, cut at the window's width (columns
   counted as characters: the menu's own words and the catalog's are
   all narrow). */
void menu_line(struct menu_screen *m, const char *style, const char *text)
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
    printf("%s\033[K", menu_reset(m));
    m->row++;
}

/* A paragraph wrapped by words at the window's width, from the left
   edge plus indent; at most max lines (-1: as many as there is room
   for). The number of lines it took, or would take when draw is 0. */
int menu_para(struct menu_screen *m, const char *style, const char *text,
              int indent, int max, int draw)
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
            menu_line(m, style, buf);
        }
        lines++;
        p = end;
    }
    return lines;
}

/* A key, the arrows and the like read from the sequences an xterm (and
   Windows' console in its virtual-terminal mode) sends. */
int menu_key(void)
{
    unsigned char c;
    if (tty_byte(&c) != 1)
        return MENU_ESC;
    if (c == '\r' || c == '\n')
        return MENU_ENTER;
    if (c == 'i' || c == 'I')
        return MENU_INFO;
    if (c == '\t')
        return MENU_RIGHT;
    if (c != 27)
        return MENU_NONE;
    if (tty_wait(50) <= 0)
        return MENU_ESC; /* Esc alone */
    unsigned char b;
    if (tty_byte(&b) != 1 || (b != '[' && b != 'O'))
        return MENU_NONE;
    int num = 0;
    for (;;) {
        if (tty_wait(50) <= 0 || tty_byte(&b) != 1)
            return MENU_NONE;
        if (b >= '0' && b <= '9')
            num = num * 10 + (b - '0');
        else if (b >= 0x40 && b <= 0x7e)
            break;
    }
    switch (b) {
    case 'A':
        return MENU_UP;
    case 'B':
        return MENU_DOWN;
    case 'C':
        return MENU_RIGHT;
    case 'D':
        return MENU_LEFT;
    case 'H':
        return MENU_HOME;
    case 'F':
        return MENU_END;
    case '~':
        return num == 5               ? MENU_PGUP
               : num == 6             ? MENU_PGDN
               : num == 1 || num == 7 ? MENU_HOME
               : num == 4 || num == 8 ? MENU_END
                                      : MENU_NONE;
    default:
        return MENU_NONE;
    }
}
