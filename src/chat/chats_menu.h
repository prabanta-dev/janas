/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * chats_menu.h - janas-chat's saved conversations (issue #69): the list of
 * the default folder's, the most recent first, and the menu to choose one
 * from when both ends are a terminal (--resume, /resume): the arrows to
 * move, Enter to take it up, Esc to leave. And /export's words.
 */
#ifndef JANAS_CHAT_CHATS_MENU_H
#define JANAS_CHAT_CHATS_MENU_H

#include <stddef.h>
#include <stdint.h>

#include "janas/llm.h"

struct chat_item {
    char path[1024];
    char title[256];
    char model[96];
    char model_file[1024];
    int64_t updated;
    int messages;
};

struct chat_list {
    struct chat_item *it;
    int n;
};

/* The default folder's conversations into l: 0, or -1. */
int chat_list_load(struct chat_list *l);
void chat_list_free(struct chat_list *l);
/* What a conversation is, on a line: when, the model, how many messages. */
void chat_item_detail(const struct chat_item *it, char *buf, size_t len);
/* The menu: the index chosen, or -1 (none, or left). */
int chat_list_menu(const struct chat_list *l);

/* Message i's part (JANAS_LLM_MESSAGE_...), allocated; NULL on failure. */
char *chat_message_dup(janas_llm_chat *c, int32_t i, int32_t what,
                       int32_t *role);
/* /export [--overwrite] [--title <title>] [file]: the title a word or in
   quotes, given to the conversation; the file by default the title's
   words in the current folder, .md (.txt with markdown off, or a file so
   named). What was done, or what was wrong, in say: 0 or -1. */
int chat_export(janas_llm_chat *c, char *arg, int markdown, char *say,
                size_t say_len);

#endif
