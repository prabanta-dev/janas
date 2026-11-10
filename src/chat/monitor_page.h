/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * monitor_page.h - janas-chat's /monitor: the engine at work as a page of
 * the terminal, made anew every second while it is open - what janas-server
 * --monitor shows in a browser, from the same counters (janas_llm_monitor,
 * janas_llm_monitor_reply). Counters and times, never the text.
 */
#ifndef JANAS_CHAT_MONITOR_PAGE_H
#define JANAS_CHAT_MONITOR_PAGE_H

#include "janas/llm.h"

/* The page's state between two samples, for what a second changed. */
struct chat_monitor {
    janas_llm *llm;
    int have;              /* a sample before this one */
    double t, cpu, joules; /* its time, processor seconds and energy */
    uint64_t bytes, used, hit;
};

/* The page's text now (malloc'd, NULL out of memory); m is a struct
   chat_monitor, as term_pager_live calls it. */
char *chat_monitor_text(void *m);

#endif
