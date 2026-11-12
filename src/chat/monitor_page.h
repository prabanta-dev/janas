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
    /* the look inside the layers (/monitor layers): on, its shape, and
       what it gathered since it was turned on - the means weighted by the
       passes, the experts' counts */
    int layers;
    uint32_t shape[3];
    double *ffn, *res, *chosen, *read, weight;
    uint64_t passes;
    float *f_ffn, *f_res, *f_str;
    uint32_t *u_chosen, *u_read;
};

/* The look inside the layers turned on (1) or off; 0, -1 out of memory. */
int chat_monitor_layers(struct chat_monitor *m, int on);

/* The page's text now (malloc'd, NULL out of memory); m is a struct
   chat_monitor, as term_pager_live calls it. */
char *chat_monitor_text(void *m);

#endif
