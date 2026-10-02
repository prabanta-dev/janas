/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * data.c - what janas-git read, as the data of its layouts (see git.h,
 * layouts.c and common/template.h). Every field is written, even empty: a
 * layout looks a name up in the item, then in what holds it.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "git.h"

void gt_jstr(struct janas_buf *b, const char *key, const char *v)
{
    if (!v)
        v = "";
    janas_buf_printf(b, ", \"%s\": ", key);
    janas_json_write_str(b, v, strlen(v));
}

void gt_jwhen(struct janas_buf *b, const char *key, time_t t)
{
    if (t < 0) {
        janas_buf_printf(b,
                         ", \"%s\": {\"day\": 0, \"mon\": 0, \"year\": 0, "
                         "\"at\": \"\", \"ago_d\": 0}",
                         key);
        return;
    }
    struct tm lt, nt;
    time_t now = time(NULL);
    localtime_r(&t, &lt);
    localtime_r(&now, &nt);
    long ago = (long)((now - t) / 86400);
    janas_buf_printf(b,
                     ", \"%s\": {\"day\": %d, \"mon\": %d, \"year\": %d, "
                     "\"at\": \"%02d:%02d\", \"ago_d\": %ld}",
                     key, lt.tm_mday, lt.tm_mon + 1,
                     lt.tm_year == nt.tm_year ? 0 : lt.tm_year + 1900,
                     lt.tm_hour, lt.tm_min, ago < 0 ? 0 : ago);
}

void gt_jhead(struct janas_buf *b, const char *top)
{
    char shown[1024];
    gt_short_path(top, shown, sizeof shown);
    const char *last = strrchr(top, '/');
    last = last && last[1] ? last + 1 : top;
    janas_buf_puts(b, "{\"repo\": ");
    janas_json_write_str(b, last, strlen(last));
    gt_jstr(b, "path", shown);
}

void gt_status_data(struct janas_buf *b, const char *top,
                    const struct gt_status *s)
{
    gt_jhead(b, top);
    gt_jstr(b, "branch", s->branch);
    gt_jstr(b, "upstream", s->upstream);
    janas_buf_printf(b,
                     ", \"detached\": %s, \"ahead\": %d, \"behind\": %d, "
                     "\"staged\": %d, \"changed\": %d, \"untracked\": %d, "
                     "\"conflicts\": %d, \"clean\": %s, \"more\": %d",
                     s->detached ? "true" : "false", s->ahead, s->behind,
                     s->n_staged, s->n_changed, s->n_untracked, s->n_conflicts,
                     s->n || s->more ? "false" : "true", s->more);
    janas_buf_puts(b, ", \"files\": [");
    for (int i = 0; i < s->n; i++) {
        const struct gt_file *f = &s->f[i];
        char st[2] = {f->staged, 0}, ch[2] = {f->changed, 0};
        janas_buf_printf(b, "%s{\"path\": ", i ? ", " : "");
        janas_json_write_str(b, f->path, strlen(f->path));
        gt_jstr(b, "in_index", st);
        gt_jstr(b, "in_tree", ch);
        janas_buf_printf(b, ", \"new\": %s, \"conflict\": %s}",
                         f->untracked ? "true" : "false",
                         f->conflict ? "true" : "false");
    }
    janas_buf_puts(b, "]}");
}

void gt_log_data(struct janas_buf *b, const char *top, const char *branch,
                 const struct gt_log *l)
{
    gt_jhead(b, top);
    gt_jstr(b, "branch", branch);
    janas_buf_printf(b, ", \"n\": %d, \"more\": %s", l->n,
                     l->more ? "true" : "false");
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < l->n; i++) {
        const struct gt_commit *c = &l->c[i];
        janas_buf_printf(b, "%s{\"sha\": ", i ? ", " : "");
        janas_json_write_str(b, c->sha, strlen(c->sha));
        gt_jstr(b, "message", c->message);
        gt_jstr(b, "author", c->author);
        gt_jstr(b, "refs", c->refs);
        gt_jwhen(b, "when", c->at);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}

void gt_branches_data(struct janas_buf *b, const char *top,
                      const struct gt_branches *v)
{
    gt_jhead(b, top);
    janas_buf_printf(b, ", \"n\": %d, \"more\": %d", v->n, v->more);
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < v->n; i++) {
        const struct gt_branch *x = &v->b[i];
        janas_buf_printf(b, "%s{\"name\": ", i ? ", " : "");
        janas_json_write_str(b, x->name, strlen(x->name));
        gt_jstr(b, "upstream", x->upstream);
        gt_jstr(b, "track", x->track);
        gt_jstr(b, "message", x->message);
        gt_jwhen(b, "when", x->at);
        janas_buf_printf(b, ", \"current\": %s}",
                         x->current ? "true" : "false");
    }
    janas_buf_puts(b, "]}");
}

void gt_diff_data(struct janas_buf *b, const char *top, int staged,
                  const char *path, const struct gt_diff *d, const char *patch,
                  size_t patch_n, size_t brief_n)
{
    gt_jhead(b, top);
    gt_jstr(b, "only", path);
    janas_buf_printf(b,
                     ", \"staged\": %s, \"n\": %d, \"more\": %d, \"add\": "
                     "%ld, \"del\": %ld",
                     staged ? "true" : "false", d->n, d->more, d->add, d->del);
    janas_buf_puts(b, ", \"files\": [");
    for (int i = 0; i < d->n; i++) {
        const struct gt_numstat *f = &d->f[i];
        janas_buf_printf(b, "%s{\"path\": ", i ? ", " : "");
        janas_json_write_str(b, f->path, strlen(f->path));
        janas_buf_printf(b, ", \"binary\": %s, \"plus\": %ld, \"minus\": %ld}",
                         f->add < 0 ? "true" : "false", f->add < 0 ? 0 : f->add,
                         f->del < 0 ? 0 : f->del);
    }
    janas_buf_puts(b, "], \"patch\": ");
    janas_json_write_str(b, patch ? patch : "", patch ? patch_n : 0);
    janas_buf_puts(b, ", \"start\": ");
    janas_json_write_str(b, patch ? patch : "",
                         patch ? (brief_n < patch_n ? brief_n : patch_n) : 0);
    janas_buf_printf(b, ", \"cut\": %s}",
                     patch && brief_n < patch_n ? "true" : "false");
}

void gt_done_data(struct janas_buf *b, const char *top, const char *what,
                  const char *branch, const char *sha, const char *message,
                  const char *output)
{
    gt_jhead(b, top);
    gt_jstr(b, "what", what);
    gt_jstr(b, "branch", branch);
    gt_jstr(b, "sha", sha);
    gt_jstr(b, "message", message);
    gt_jstr(b, "output", output);
    janas_buf_puts(b, "}");
}
