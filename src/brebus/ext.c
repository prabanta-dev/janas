/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * ext.c - what Brebus offers to extend the model, from outside: the lists
 * Janas publishes (src/brebus/catalog/connectors.json and skills.json, read
 * from GitHub as the repository's main branch has them, kept a day in the
 * data folder; the page carries the copy it was built with, for when
 * neither is there), and a search in MCP's official registry of servers
 * (registry.modelcontextprotocol.io), its answer handed to the page as it
 * comes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "brebus.h"
#include "common/hf_fetch.h"

#define CATALOG_URL                                                            \
    "https://raw.githubusercontent.com/prabanta-dev/janas/main/src/brebus/"    \
    "catalog/"
#define REGISTRY_URL "https://registry.modelcontextprotocol.io/v0/servers"
#define DAY (24 * 3600)

/* the file into out (at most max bytes): 0, or -1 */
static int slurp(const char *path, size_t max, struct janas_buf *out)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    char buf[16384];
    size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0 && out->n < max)
        janas_buf_put(out, buf, r);
    fclose(f);
    return out->n && out->n <= max && !out->oom ? 0 : -1;
}

/* a list Janas publishes: an object with its "items" */
static int catalog_ok(const struct janas_buf *t)
{
    char err[160];
    struct janas_json_doc *d = janas_json_parse(t->p, t->n, err, sizeof err);
    const struct janas_json *it =
        d ? janas_json_get(janas_json_root(d), "items") : NULL;
    int ok = it && it->type == JANAS_JSON_ARRAY;
    janas_json_free(d);
    return ok;
}

int brebus_ext_catalog(struct brebus *b, const char *kind,
                       struct janas_buf *out)
{
    if (!kind || (strcmp(kind, "connectors") && strcmp(kind, "skills")))
        return -1;
    char path[4300], tmp[4400], url[512], err[200];
    snprintf(path, sizeof path, "%s/catalog-%s.json", b->data_dir, kind);
    struct stat st;
    int have = stat(path, &st) == 0;
    /* a day old at most; else asked again, the old one kept if it fails */
    if (!have || time(NULL) - st.st_mtime > DAY) {
        struct janas_buf t = {0};
        snprintf(url, sizeof url, CATALOG_URL "%s.json", kind);
        if (janas_hf_get_url(url, NULL, NULL, &t, NULL, 0, err, sizeof err) ==
                0 &&
            t.n < (1u << 20) && catalog_ok(&t)) {
            snprintf(tmp, sizeof tmp, "%s.new", path);
            FILE *f = fopen(tmp, "wb");
            int bad = !f;
            if (f) {
                bad |= fwrite(t.p, 1, t.n, f) != t.n;
                bad |= fclose(f) != 0;
            }
            if (bad || rename(tmp, path) != 0)
                remove(tmp);
            else
                have = 1;
        }
        janas_buf_free(&t);
    }
    struct janas_buf t = {0};
    int rc = have && slurp(path, 1u << 20, &t) == 0 && catalog_ok(&t) ? 0 : -1;
    if (rc == 0)
        janas_buf_put(out, t.p, t.n);
    janas_buf_free(&t);
    return rc;
}

/* q as a URL's query wants it */
static void encode(const char *q, char *out, size_t n)
{
    static const char HEX[] = "0123456789ABCDEF";
    size_t k = 0;
    for (const unsigned char *c = (const unsigned char *)q; *c && k + 4 < n;
         c++)
        if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
            (*c >= '0' && *c <= '9') || *c == '-' || *c == '_' || *c == '.')
            out[k++] = (char)*c;
        else {
            out[k++] = '%';
            out[k++] = HEX[*c >> 4];
            out[k++] = HEX[*c & 15];
        }
    out[k] = 0;
}

int brebus_ext_registry(struct brebus *b, const char *q, const char *cursor,
                        struct janas_buf *out)
{
    (void)b;
    char eq[400], ec[400], url[1200], err[200];
    encode(q ? q : "", eq, sizeof eq);
    encode(cursor ? cursor : "", ec, sizeof ec);
    snprintf(url, sizeof url, REGISTRY_URL "?limit=40&version=latest%s%s%s%s",
             *eq ? "&search=" : "", eq, *ec ? "&cursor=" : "", ec);
    struct janas_buf t = {0};
    int rc = janas_hf_get_url(url, NULL, NULL, &t, NULL, 0, err, sizeof err);
    struct janas_json_doc *d = rc == 0 && t.n < (8u << 20)
                                   ? janas_json_parse(t.p, t.n, err, sizeof err)
                                   : NULL;
    if (d)
        janas_buf_put(out, t.p, t.n);
    else {
        const char *m = rc ? err : "the registry's answer is not JSON";
        janas_buf_puts(out, "{\"error\": ");
        janas_json_write_str(out, m, strlen(m));
        janas_buf_puts(out, "}");
    }
    janas_json_free(d);
    janas_buf_free(&t);
    return d ? 0 : -1;
}
