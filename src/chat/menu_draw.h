/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * menu_draw.h - what janas-chat's menus share (the models', model_menu.c;
 * the conversations', chats_menu.c): a screen drawn a line at a time from
 * the top, lines cut and paragraphs wrapped at the window's width, the
 * styles, and the keys read from the sequences a terminal sends.
 */
#ifndef JANAS_CHAT_MENU_DRAW_H
#define JANAS_CHAT_MENU_DRAW_H

struct menu_screen {
    int rows, cols;
    int colour;
    int row; /* the next line drawn, from 0 */
};

/* The styles, empty without colour. */
const char *menu_reset(const struct menu_screen *m);
const char *menu_bold(const struct menu_screen *m);
const char *menu_dim(const struct menu_screen *m);
const char *menu_mark(const struct menu_screen *m); /* amber, as the user's */
const char *menu_warn(const struct menu_screen *m);

/* One line of the screen in a style, cut at the window's width. */
void menu_line(struct menu_screen *m, const char *style, const char *text);
/* A paragraph wrapped by words at the window's width, from the left edge
   plus indent; at most max lines (-1: as many as there is room for). The
   number of lines it took, or would take when draw is 0. */
int menu_para(struct menu_screen *m, const char *style, const char *text,
              int indent, int max, int draw);

enum {
    MENU_NONE,
    MENU_UP,
    MENU_DOWN,
    MENU_LEFT,
    MENU_RIGHT,
    MENU_PGUP,
    MENU_PGDN,
    MENU_HOME,
    MENU_END,
    MENU_ENTER,
    MENU_ESC,
    MENU_INFO
};

/* A key: one of MENU_..., MENU_NONE for any other. */
int menu_key(void);

#endif
