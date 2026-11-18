/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * window_place.c - where Brebus's window was and how big, kept in its data
 * folder (window.json) when it closes, given back at the next start; the
 * first start, with nothing kept, has it maximized. Shared by the window
 * of Linux (window.c) and of Windows (window_win.c).
 */
#include <stdio.h>
#include <string.h>

#include "brebus.h"

static char place_file[4300];

void brebus_place_dir(const char *data_dir)
{
    snprintf(place_file, sizeof place_file, "%s/window.json", data_dir);
}

int brebus_place_read(struct brebus_place *p)
{
    memset(p, 0, sizeof *p);
    FILE *f = place_file[0] ? fopen(place_file, "rb") : NULL;
    if (!f)
        return -1;
    char buf[512], err[160];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    struct janas_json_doc *d = janas_json_parse(buf, n, err, sizeof err);
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    if (r) {
        p->x = (int)janas_json_num(janas_json_get(r, "x"), 0);
        p->y = (int)janas_json_num(janas_json_get(r, "y"), 0);
        p->w = (int)janas_json_num(janas_json_get(r, "width"), 0);
        p->h = (int)janas_json_num(janas_json_get(r, "height"), 0);
        const struct janas_json *m = janas_json_get(r, "maximized");
        p->max = m && m->type == JANAS_JSON_TRUE;
    }
    janas_json_free(d);
    /* a size too small to be one the user chose is not kept */
    return r && p->w >= 400 && p->h >= 300 ? 0 : -1;
}

void brebus_place_write(const struct brebus_place *p)
{
    char tmp[4400];
    if (!place_file[0] || p->w < 400 || p->h < 300)
        return;
    snprintf(tmp, sizeof tmp, "%s.new", place_file);
    FILE *f = fopen(tmp, "wb");
    if (!f)
        return;
    int bad = fprintf(f,
                      "{\"x\": %d, \"y\": %d, \"width\": %d, \"height\": %d, "
                      "\"maximized\": %s}\n",
                      p->x, p->y, p->w, p->h, p->max ? "true" : "false") < 0;
    bad |= fclose(f) != 0;
    if (bad || rename(tmp, place_file) != 0)
        remove(tmp);
}
