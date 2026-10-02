/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * read.c - janas-git reads git's output (see git.h), in the formats meant
 * for programs: status --porcelain=v2 -z, log and for-each-ref with
 * fields between 0x1f and records ended by 0x1e, diff --numstat -z. No git
 * here, so that the tests can read output kept as it came.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "git.h"

static void copy(char *to, size_t cap, const char *s, size_t n)
{
    snprintf(to, cap, "%.*s", (int)(n < cap ? n : cap - 1), s);
}

/* The field after skip spaces of a status line: where the path begins */
static const char *after_fields(const char *s, const char *end, int skip)
{
    while (skip-- > 0) {
        const char *sp = memchr(s, ' ', (size_t)(end - s));
        if (!sp)
            return NULL;
        s = sp + 1;
    }
    return s;
}

int gt_read_status(const char *text, size_t n, struct gt_status *s)
{
    memset(s, 0, sizeof *s);
    const char *p = text, *end = text + n;
    while (p < end) {
        const char *z = memchr(p, 0, (size_t)(end - p));
        if (!z)
            z = end;
        size_t len = (size_t)(z - p);
        const char *path = NULL;
        struct gt_file f = {.staged = 0};
        if (len > 16 && strncmp(p, "# branch.head ", 14) == 0) {
            copy(s->branch, sizeof s->branch, p + 14, len - 14);
            s->detached = strcmp(s->branch, "(detached)") == 0;
        } else if (len > 15 && strncmp(p, "# branch.oid ", 13) == 0) {
            copy(s->oid, 8, p + 13, len - 13);
        } else if (len > 18 && strncmp(p, "# branch.upstream ", 18) == 0) {
            copy(s->upstream, sizeof s->upstream, p + 18, len - 18);
        } else if (len > 12 && strncmp(p, "# branch.ab ", 12) == 0) {
            sscanf(p + 12, "+%d -%d", &s->ahead, &s->behind);
        } else if (len > 4 && (p[0] == '1' || p[0] == '2') && p[1] == ' ') {
            /* 1 XY sub mH mI mW hH hI path; 2 adds a score, then the
               path it came from after the next 0 */
            f.staged = p[2] != '.' ? p[2] : 0;
            f.changed = p[3] != '.' ? p[3] : 0;
            path = after_fields(p, z, p[0] == '1' ? 8 : 9);
            if (f.staged)
                s->n_staged++;
            if (f.changed)
                s->n_changed++;
        } else if (len > 4 && p[0] == 'u' && p[1] == ' ') {
            f.conflict = 1;
            path = after_fields(p, z, 10);
            s->n_conflicts++;
        } else if (len > 2 && p[0] == '?' && p[1] == ' ') {
            f.untracked = 1;
            path = p + 2;
            s->n_untracked++;
        }
        if (path) {
            if (s->n < GT_FILES) {
                copy(f.path, sizeof f.path, path, (size_t)(z - path));
                s->f[s->n++] = f;
            } else {
                s->more++;
            }
        }
        if (p[0] == '2') { /* the path it came from */
            const char *z2 =
                z < end ? memchr(z + 1, 0, (size_t)(end - z - 1)) : NULL;
            z = z2 ? z2 : end;
        }
        p = z + 1;
    }
    return s->branch[0] ? 0 : -1;
}

/* The n-th field (0x1f between) of a record, as a pointer and a length */
static const char *field(const char *r, size_t rn, int k, size_t *len)
{
    const char *e = r + rn;
    while (k-- > 0) {
        const char *u = memchr(r, 0x1f, (size_t)(e - r));
        if (!u)
            return NULL;
        r = u + 1;
    }
    const char *u = memchr(r, 0x1f, (size_t)(e - r));
    *len = (size_t)((u ? u : e) - r);
    return r;
}

/* The next record (0x1e after it, line ends before it left out): its
   start and length, NULL at the end */
static const char *next_record(const char **p, const char *end, size_t *rn)
{
    while (*p < end && (**p == '\n' || **p == '\r'))
        (*p)++;
    const char *z = *p < end ? memchr(*p, 0x1e, (size_t)(end - *p)) : NULL;
    if (!z)
        return NULL;
    const char *r = *p;
    *rn = (size_t)(z - r);
    *p = z + 1;
    return r;
}

int gt_read_log(const char *text, size_t n, int max, struct gt_log *l)
{
    memset(l, 0, sizeof *l);
    if (max > GT_ITEMS)
        max = GT_ITEMS;
    const char *p = text, *r;
    size_t rn;
    while ((r = next_record(&p, text + n, &rn))) {
        if (l->n >= max) {
            l->more = 1;
            continue;
        }
        struct gt_commit *c = &l->c[l->n++];
        size_t fl;
        const char *f;
        if ((f = field(r, rn, 0, &fl)))
            copy(c->sha, sizeof c->sha, f, fl);
        if ((f = field(r, rn, 1, &fl)))
            copy(c->author, sizeof c->author, f, fl);
        c->at = (f = field(r, rn, 2, &fl)) ? (time_t)strtoll(f, NULL, 10) : -1;
        if ((f = field(r, rn, 3, &fl)))
            copy(c->message, sizeof c->message, f, fl);
        if ((f = field(r, rn, 4, &fl)))
            copy(c->refs, sizeof c->refs, f, fl);
    }
    return 0;
}

int gt_read_branches(const char *text, size_t n, int max, struct gt_branches *b)
{
    memset(b, 0, sizeof *b);
    if (max > GT_ITEMS)
        max = GT_ITEMS;
    const char *p = text, *r;
    size_t rn;
    while ((r = next_record(&p, text + n, &rn))) {
        size_t fl;
        const char *f = field(r, rn, 0, &fl);
        int current = f && fl && *f == '*';
        /* the branch one is on is listed whatever its place */
        if (b->n >= max && !current) {
            b->more++;
            continue;
        }
        if (b->n >= max)
            b->more++; /* the last listed gives way to it */
        struct gt_branch *x = &b->b[b->n < max ? b->n++ : max - 1];
        memset(x, 0, sizeof *x);
        x->current = current;
        if ((f = field(r, rn, 1, &fl)))
            copy(x->name, sizeof x->name, f, fl);
        if ((f = field(r, rn, 2, &fl)))
            copy(x->upstream, sizeof x->upstream, f, fl);
        if ((f = field(r, rn, 3, &fl)))
            copy(x->track, sizeof x->track, f, fl);
        x->at = (f = field(r, rn, 4, &fl)) ? (time_t)strtoll(f, NULL, 10) : -1;
        if ((f = field(r, rn, 5, &fl)))
            copy(x->message, sizeof x->message, f, fl);
    }
    return 0;
}

int gt_read_numstat(const char *text, size_t n, struct gt_diff *d)
{
    memset(d, 0, sizeof *d);
    const char *p = text, *end = text + n;
    while (p < end) {
        const char *z = memchr(p, 0, (size_t)(end - p));
        if (!z)
            z = end;
        /* "add\tdel\tpath", or "add\tdel\t" and the two paths of a
           rename, each after a 0 */
        char *t1, *t2;
        long add = strtol(p, &t1, 10), del = 0;
        int binary = *p == '-';
        if (binary)
            t1 = (char *)p + 1;
        if (*t1 != '\t')
            break;
        del = binary ? 0 : strtol(t1 + 1, &t2, 10);
        if (binary)
            t2 = t1 + 2;
        if (*t2 != '\t')
            break;
        const char *path = t2 + 1;
        if (path == z) { /* a rename: from, then to */
            const char *z1 =
                z + 1 < end ? memchr(z + 1, 0, (size_t)(end - z - 1)) : NULL;
            const char *z2 = z1 && z1 + 1 < end
                                 ? memchr(z1 + 1, 0, (size_t)(end - z1 - 1))
                                 : NULL;
            if (!z1 || !z2)
                break;
            path = z1 + 1;
            z = z2;
        }
        if (!binary) {
            d->add += add;
            d->del += del;
        }
        if (d->n < GT_FILES) {
            struct gt_numstat *f = &d->f[d->n++];
            copy(f->path, sizeof f->path, path, (size_t)(z - path));
            f->add = binary ? -1 : add;
            f->del = binary ? -1 : del;
        } else {
            d->more++;
        }
        p = z + 1;
    }
    return 0;
}
