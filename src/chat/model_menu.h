/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_menu.h - janas-chat's menu of the models, when it is started with
 * none and both ends are a terminal (issue #62): the models here and the
 * ones to download in two lists, the arrows to move, I for what a model is
 * and what it asks of the machine, Enter to open it, Esc to leave without
 * opening any. It stands in for the numbered list of model_pick.c, which
 * stays for pipes and for janas-bench.
 */
#ifndef JANAS_CHAT_MODEL_MENU_H
#define JANAS_CHAT_MODEL_MENU_H

#include "common/model_pick.h"

/* janas_model_menu's shape: the index of the item chosen, or -1. */
int chat_model_menu(const char *dir, const struct janas_model_item *items,
                    int n, int n_here);

#endif
