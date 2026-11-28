/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tiles.c - the tiles of maps made from OpenStreetMap's data, for the
 * views' maps when the user chooses one there (the map's layers):
 * /brebus/tile/<layer>/<z>/<x>/<y>.png, asked by Brebus itself, never by
 * the view (which reaches nothing). Only servers whose terms allow an
 * application to use them, each as its terms ask: OpenStreetMap's own
 * (the OSMF's tile usage policy: a User-Agent that names the program, a
 * tile kept and used again for a week, none asked before it is seen) and
 * OpenTopoMap's (CC-BY-SA, apps welcome while they do not load its server
 * heavily: the same care). Kept in the cache folder, janas/tiles/<layer>.
 * The view says each map's attribution on it.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/mcp_http.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

#define WEEK (7 * 24 * 3600)
#define MAX_TILE (1u << 20)

/* the cache's folder: $XDG_CACHE_HOME/janas/tiles or ~/.cache/janas/tiles
   (Windows: the local application data, through win_compat's getenv) */
static int cache_dir(char *out, size_t n)
{
    const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
    if (xdg && *xdg)
        snprintf(out, n, "%s/janas/tiles", xdg);
    else if (home && *home)
        snprintf(out, n, "%s/.cache/janas/tiles", home);
    else
        return -1;
    return 0;
}

static int slurp(const char *path, struct janas_buf *out)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    char buf[16384];
    size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0 && out->n <= MAX_TILE)
        janas_buf_put(out, buf, r);
    fclose(f);
    return out->n && out->n <= MAX_TILE && !out->oom ? 0 : -1;
}

/* the maps there are: a layer's name, its server, its deepest zoom */
static const struct layer {
    const char *name, *url;
    int maxz;
} LAYERS[] = {
    {"osm", "https://tile.openstreetmap.org/%d/%ld/%ld.png", 19},
    {"topo", "https://a.tile.opentopomap.org/%d/%ld/%ld.png", 17},
};

/* the tile asked of the layer's server: 0, or -1 with why */
static int fetch(const struct layer *l, int z, long x, long y,
                 struct janas_buf *out, char *why, size_t wn)
{
    char url[200], agent[200];
    snprintf(url, sizeof url, l->url, z, x, y);
    snprintf(agent, sizeof agent,
             "User-Agent: Brebus/%s (Janas; "
             "https://github.com/prabanta-dev/janas)",
             JANAS_VERSION);
    struct janas_url u;
    if (janas_url_parse(url, &u, why, wn) != 0)
        return -1;
    const char *hdr[] = {agent};
    struct janas_http h;
    if (janas_http_request(&h, &u, "GET", hdr, 1, NULL, 0, 20000, NULL, why,
                           wn) != 0)
        return -1;
    int rc = h.head.status == 200 ? 0 : -1;
    if (rc)
        snprintf(why, wn, "the map's server answered %d", h.head.status);
    while (rc == 0) {
        int k = janas_http_body(&h, out, 20000);
        if (k == 0)
            break;
        if (k < 0 || out->n > MAX_TILE) {
            snprintf(why, wn, "the tile did not come whole");
            rc = -1;
        }
    }
    janas_http_close(&h);
    /* a PNG, nothing else */
    if (rc == 0 && (out->n < 8 || memcmp(out->p, "\x89PNG", 4))) {
        snprintf(why, wn, "not a picture");
        rc = -1;
    }
    return rc;
}

int brebus_tile(struct brebus *b, const char *path, struct janas_buf *png,
                char *why, size_t wn)
{
    (void)b;
    int z;
    long x, y;
    char tail[8] = "", name[16] = "";
    const struct layer *l = NULL;
    if (path && sscanf(path, "/brebus/tile/%15[a-z]/%d/%ld/%ld%7s", name, &z,
                       &x, &y, tail) == 5)
        for (size_t i = 0; i < sizeof LAYERS / sizeof *LAYERS; i++)
            if (!strcmp(LAYERS[i].name, name))
                l = &LAYERS[i];
    if (!l || strcmp(tail, ".png") || z < 0 || z > l->maxz || x < 0 || y < 0 ||
        x >= (1L << z) || y >= (1L << z)) {
        snprintf(why, wn, "no such tile");
        return -1;
    }
    char dir[4200], file[4400], tmp[4500];
    int cached = cache_dir(dir, sizeof dir) == 0;
    if (cached) {
        snprintf(dir + strlen(dir), sizeof dir - strlen(dir), "/%s/%d/%ld",
                 l->name, z, x);
        snprintf(file, sizeof file, "%s/%ld.png", dir, y);
    }
    struct stat st;
    int have = cached && stat(file, &st) == 0;
    if (have && time(NULL) - st.st_mtime < WEEK && slurp(file, png) == 0)
        return 0;
    janas_buf_free(png);
    *png = (struct janas_buf){0};
    if (fetch(l, z, x, y, png, why, wn) != 0) {
        /* out of reach: the old one, if there is one */
        janas_buf_free(png);
        *png = (struct janas_buf){0};
        return have && slurp(file, png) == 0 ? 0 : -1;
    }
    if (cached && janas_mkdirs(dir) == 0) {
        snprintf(tmp, sizeof tmp, "%s.new", file);
        FILE *f = fopen(tmp, "wb");
        int bad = !f;
        if (f) {
            bad |= fwrite(png->p, 1, png->n, f) != png->n;
            bad |= fclose(f) != 0;
        }
        if (bad || rename(tmp, file) != 0)
            remove(tmp);
    }
    return 0;
}
