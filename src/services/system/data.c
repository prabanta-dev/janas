/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * data.c - what janas-system read, as the data of its layouts (see
 * system.h, layouts.c and services/common/template.h). Every field is
 * written, even empty: a layout looks a name up in the item, then in what
 * holds it. Sizes, percentages and times are worked out here, rounded, so
 * that neither the layout nor the model does sums: a size is {"v": 12.5,
 * "u": "GB"} (of 1024, as the file managers count).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "system.h"

void sy_jstr(struct janas_buf *b, const char *key, const char *v)
{
    if (!v)
        v = "";
    janas_buf_printf(b, ", \"%s\": ", key);
    janas_json_write_str(b, v, strlen(v));
}

static double round1(double x)
{
    return floor(x * 10 + 0.5) / 10;
}

static void jsize(struct janas_buf *b, const char *key, uint64_t bytes)
{
    static const char *const unit[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    double v = (double)bytes;
    int u = 0;
    while (v >= 1024 && u < 5)
        v /= 1024, u++;
    /* 1 decimal below 100, none above */
    v = v < 100 && u ? round1(v) : floor(v + 0.5);
    janas_buf_printf(b, ", \"%s\": {\"v\": %g, \"u\": \"%s\"}", key, v,
                     unit[u]);
}

static int pct(uint64_t part, uint64_t whole)
{
    return whole ? (int)((double)part * 100 / (double)whole + 0.5) : 0;
}

static void jspan(struct janas_buf *b, const char *key, long s)
{
    if (s < 0)
        s = 0;
    janas_buf_printf(b, ", \"%s\": {\"d\": %ld, \"h\": %ld, \"m\": %ld}", key,
                     s / 86400, s / 3600 % 24, s / 60 % 60);
}

static void jwhen(struct janas_buf *b, const char *key, long long at)
{
    time_t t = (time_t)at, now = time(NULL);
    struct tm lt, nt;
#ifdef _WIN32
    localtime_s(&lt, &t);
    localtime_s(&nt, &now);
#else
    localtime_r(&t, &lt);
    localtime_r(&now, &nt);
#endif
    long ago = at > 0 ? (long)((now - t) / 86400) : 0;
    janas_buf_printf(
        b,
        ", \"%s\": {\"has\": %s, \"day\": %d, \"mon\": %d, "
        "\"at\": \"%02d:%02d\", \"today\": %s, \"ago_d\": %ld}",
        key, at > 0 ? "true" : "false", lt.tm_mday, lt.tm_mon + 1, lt.tm_hour,
        lt.tm_min,
        lt.tm_yday == nt.tm_yday && lt.tm_year == nt.tm_year ? "true" : "false",
        ago < 0 ? 0 : ago);
}

static const char *tf(int v)
{
    return v ? "true" : "false";
}

/* a path with the home as "~" */
static void jpath(struct janas_buf *b, const char *key, const char *path)
{
    const char *home = sy_home();
    size_t h = strlen(home);
    char out[1100];
    if (h > 1 && !strncmp(path, home, h) && (path[h] == '/' || !path[h]))
        snprintf(out, sizeof out, "~%s", path + h);
    else
        sy_put(out, sizeof out, path);
    sy_jstr(b, key, out);
}

static void disk(struct janas_buf *b, const struct sy_disk *d, int first)
{
    uint64_t used = d->total > d->avail ? d->total - d->avail : 0;
    int p = pct(used, d->total);
    janas_buf_puts(b, first ? "{" : ", {");
    janas_buf_puts(b, "\"mount\": ");
    janas_json_write_str(b, d->mount, strlen(d->mount));
    sy_jstr(b, "dev", d->dev);
    sy_jstr(b, "fs", d->fs);
    jsize(b, "used", used);
    jsize(b, "total", d->total);
    jsize(b, "avail", d->avail);
    janas_buf_printf(b, ", \"pct\": %d, \"full\": %s, \"ro\": %s}", p,
                     tf(p >= SY_FULL_PCT), tf(d->ro));
}

void sy_status_data(struct janas_buf *b, const struct sy_status *s)
{
    janas_buf_puts(b, "{\"host\": ");
    janas_json_write_str(b, s->host, strlen(s->host));
    sy_jstr(b, "os", s->os);
    sy_jstr(b, "kernel", s->kernel);
    sy_jstr(b, "cpu", s->cpu);
    sy_jstr(b, "platform", s->platform);
    janas_buf_printf(b, ", \"cpus\": %d", s->cpus);
    jspan(b, "up", s->uptime_s);
    janas_buf_printf(b,
                     ", \"has_load\": %s, \"load1\": %.2f, \"load5\": %.2f, "
                     "\"load15\": %.2f",
                     tf(s->has_load), s->load[0], s->load[1], s->load[2]);
    janas_buf_printf(b, ", \"has_cpu\": %s, \"cpu_pct\": %d, \"sys_pct\": %d",
                     tf(s->cpu_pct >= 0), (int)(s->cpu_pct + 0.5),
                     (int)(s->sys_pct + 0.5));
    uint64_t used =
        s->mem_total > s->mem_avail ? s->mem_total - s->mem_avail : 0;
    janas_buf_puts(b, ", \"mem\": {\"x\": 1");
    jsize(b, "used", used);
    jsize(b, "total", s->mem_total);
    jsize(b, "avail", s->mem_avail);
    janas_buf_printf(b, ", \"pct\": %d}", pct(used, s->mem_total));
    uint64_t sw =
        s->swap_total > s->swap_free ? s->swap_total - s->swap_free : 0;
    janas_buf_printf(b, ", \"swap\": {\"has\": %s", tf(s->swap_total > 0));
    jsize(b, "used", sw);
    jsize(b, "total", s->swap_total);
    janas_buf_printf(b, ", \"pct\": %d}", pct(sw, s->swap_total));
    janas_buf_printf(
        b,
        ", \"psi\": {\"has\": %s, \"cpu\": %g, \"mem\": %g, "
        "\"io\": %g, \"high\": %s}",
        tf(s->psi_cpu >= 0), round1(s->psi_cpu > 0 ? s->psi_cpu : 0),
        round1(s->psi_mem > 0 ? s->psi_mem : 0),
        round1(s->psi_io > 0 ? s->psi_io : 0),
        tf(s->psi_cpu >= 20 || s->psi_mem >= 10 || s->psi_io >= 20));
    int full = 0;
    janas_buf_puts(b, ", \"disks\": [");
    for (int i = 0; i < s->disks.n; i++) {
        disk(b, &s->disks.d[i], i == 0);
        const struct sy_disk *d = &s->disks.d[i];
        full += pct(d->total - d->avail, d->total) >= SY_FULL_PCT;
    }
    janas_buf_printf(b, "], \"full_n\": %d", full);
    const struct sy_battery *t = &s->bat;
    janas_buf_printf(b,
                     ", \"bat\": {\"present\": %s, \"pct\": %d, \"state\": "
                     "\"%s\", \"mains\": %s, \"draining\": %s, "
                     "\"has_time\": %s",
                     tf(t->present), t->pct < 0 ? 0 : t->pct, t->state,
                     tf(t->mains == 1), tf(!strcmp(t->state, "discharging")),
                     tf(t->minutes > 0));
    jspan(b, "time", t->minutes > 0 ? (long)t->minutes * 60 : 0);
    janas_buf_printf(b, ", \"health\": %d, \"worn\": %s}",
                     t->health < 0 ? 0 : t->health,
                     tf(t->health >= 0 && t->health < 80));
    janas_buf_printf(b, ", \"n_temps\": %d, \"temps\": [", s->n_temps);
    int hot = 0;
    for (int i = 0; i < s->n_temps; i++) {
        const struct sy_temp *x = &s->t[i];
        int h = x->high > 0 && x->c >= x->high - 10;
        hot += h;
        janas_buf_printf(b, "%s{\"kind\": \"%s\"", i ? ", " : "", x->kind);
        sy_jstr(b, "label", x->label);
        janas_buf_printf(b, ", \"c\": %d, \"high\": %d, \"hot\": %s}",
                         (int)(x->c + 0.5), (int)(x->high + 0.5), tf(h));
    }
    janas_buf_printf(b, "], \"hot_n\": %d}", hot);
}

void sy_disks_data(struct janas_buf *b, const struct sy_disks *d)
{
    janas_buf_printf(b, "{\"n\": %d, \"more\": %d, \"disks\": [", d->n,
                     d->more);
    for (int i = 0; i < d->n; i++)
        disk(b, &d->d[i], i == 0);
    janas_buf_puts(b, "]}");
}

void sy_procs_data(struct janas_buf *b, const struct sy_procs *p)
{
    janas_buf_printf(b, "{\"by\": \"%s\"", p->by);
    sy_jstr(b, "name", p->name);
    janas_buf_printf(b,
                     ", \"total\": %d, \"matched\": %d, \"n\": %d, \"more\": "
                     "%d, \"items\": [",
                     p->total, p->matched, p->n, p->matched - p->n);
    for (int i = 0; i < p->n; i++) {
        const struct sy_proc *x = &p->p[i];
        janas_buf_printf(b, "%s{\"pid\": %d", i ? ", " : "", x->pid);
        sy_jstr(b, "pname", x->name);
        sy_jstr(b, "user", x->user);
        janas_buf_printf(b, ", \"cpu\": %g", round1(x->cpu_pct));
        jsize(b, "mem", x->mem);
        janas_buf_printf(
            b, ", \"mem_pct\": %g, \"threads\": %d",
            p->mem_total ? round1((double)x->mem * 100 / (double)p->mem_total)
                         : 0,
            x->threads);
        jspan(b, "age", x->age_s);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}

void sy_errors_data(struct janas_buf *b, const struct sy_errors *e)
{
    janas_buf_printf(b,
                     "{\"hours\": %d, \"boot\": %s, \"user_only\": %s, "
                     "\"how\": \"%s\", \"total\": %d, \"cut\": %s, \"n\": %d, "
                     "\"more\": %d",
                     e->hours, tf(e->hours <= 0), tf(e->user_only), e->how,
                     e->total, tf(e->cut), e->n, e->more);
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < e->n; i++) {
        const struct sy_err *x = &e->e[i];
        janas_buf_puts(b, i ? ", {\"who\": " : "{\"who\": ");
        janas_json_write_str(b, x->who, strlen(x->who));
        sy_jstr(b, "text", x->text);
        janas_buf_printf(b, ", \"count\": %d, \"again\": %s", x->count,
                         tf(x->count > 1));
        jwhen(b, "when", x->at);
        janas_buf_puts(b, "}");
    }
    janas_buf_printf(b, "], \"n_failed\": %d, \"more_failed\": %d", e->n_failed,
                     e->more_failed);
    janas_buf_puts(b, ", \"failed\": [");
    for (int i = 0; i < e->n_failed; i++) {
        janas_buf_puts(b, i ? ", {\"unit\": " : "{\"unit\": ");
        janas_json_write_str(b, e->failed[i], strlen(e->failed[i]));
        janas_buf_printf(b, ", \"count\": %d, \"again\": %s}", e->failed_n[i],
                         tf(e->failed_n[i] > 1));
    }
    janas_buf_puts(b, "]}");
}

void sy_updates_data(struct janas_buf *b, const struct sy_updates *u)
{
    janas_buf_printf(b,
                     "{\"tool\": \"%s\", \"total\": %d, \"n\": %d, \"more\": "
                     "%d, \"security\": %d",
                     u->tool, u->total, u->n, u->total - u->n, u->security);
    jwhen(b, "checked", u->checked);
    /* how the list is fetched again: a command, never translated */
    sy_jstr(b, "refresh",
            !strcmp(u->tool, "apt")      ? "sudo apt update"
            : !strcmp(u->tool, "dnf")    ? "dnf makecache"
            : !strcmp(u->tool, "pacman") ? "checkupdates (pacman-contrib)"
                                         : "");
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < u->n; i++) {
        const struct sy_update *x = &u->u[i];
        janas_buf_puts(b, i ? ", {\"pkg\": " : "{\"pkg\": ");
        janas_json_write_str(b, x->name, strlen(x->name));
        sy_jstr(b, "to", x->to);
        sy_jstr(b, "from", x->from);
        janas_buf_printf(b, ", \"sec\": %s}", tf(x->security));
    }
    janas_buf_puts(b, "]}");
}

void sy_net_data(struct janas_buf *b, const struct sy_net *n)
{
    janas_buf_puts(b, "{\"gateway\": ");
    janas_json_write_str(b, n->gateway, strlen(n->gateway));
    sy_jstr(b, "gw_iface", n->gw_iface);
    janas_buf_puts(b, ", \"dns\": [");
    for (int i = 0; i < n->n_dns; i++) {
        janas_buf_puts(b, i ? ", " : "");
        janas_json_write_str(b, n->dns[i], strlen(n->dns[i]));
    }
    janas_buf_printf(b,
                     "], \"n_dns\": %d, \"n\": %d, \"more\": %d, \"items\": [",
                     n->n_dns, n->n, n->more);
    for (int i = 0; i < n->n; i++) {
        const struct sy_iface *x = &n->i[i];
        janas_buf_puts(b, i ? ", {\"name\": " : "{\"name\": ");
        janas_json_write_str(b, x->name, strlen(x->name));
        janas_buf_printf(b, ", \"kind\": \"%s\", \"state\": \"%s\"", x->kind,
                         x->state);
        janas_buf_printf(b, ", \"up\": %s", tf(!strcmp(x->state, "up")));
        janas_buf_puts(b, ", \"addrs\": [");
        for (int k = 0; k < x->n_addr; k++) {
            janas_buf_puts(b, k ? ", " : "");
            janas_json_write_str(b, x->addr[k], strlen(x->addr[k]));
        }
        janas_buf_printf(b,
                         "], \"speed\": %ld, \"signal\": %d, \"wifi_pct\": "
                         "%d, \"gw\": %s",
                         x->speed_mbps, x->signal_dbm, x->wifi_pct,
                         tf(!strcmp(x->name, n->gw_iface)));
        jsize(b, "rx", x->rx);
        jsize(b, "tx", x->tx);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}

void sy_usage_data(struct janas_buf *b, const struct sy_usage *u)
{
    janas_buf_puts(b, "{\"x\": 1");
    jpath(b, "path", u->path);
    jsize(b, "total", u->total);
    jsize(b, "fs_total", u->fs_total);
    jsize(b, "fs_avail", u->fs_avail);
    janas_buf_printf(b,
                     ", \"files\": %ld, \"unread\": %ld, \"cut\": %s, "
                     "\"secs\": %g, \"n\": %d, \"more\": %d, \"items\": [",
                     u->files, u->unread, tf(u->cut), round1(u->secs), u->n,
                     u->more);
    for (int i = 0; i < u->n; i++) {
        const struct sy_entry *x = &u->e[i];
        janas_buf_puts(b, i ? ", {\"entry\": " : "{\"entry\": ");
        janas_json_write_str(b, x->name, strlen(x->name));
        janas_buf_printf(b, ", \"dir\": %s, \"files\": %ld", tf(x->dir),
                         x->files);
        jsize(b, "size", x->bytes);
        janas_buf_printf(b, ", \"pct\": %d}", pct(x->bytes, u->total));
    }
    janas_buf_puts(b, "]}");
}
