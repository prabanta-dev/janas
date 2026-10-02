/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * read.c - janas-system reads the texts of Linux (see system.h): /proc's
 * files, journalctl's and systemctl's output, the package managers' lists.
 * Plain C over text, nothing read from the system here: the tests give it
 * texts kept from a machine.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "system.h"

void sy_put(char *out, size_t cap, const char *s)
{
    size_t n = strlen(s);
    if (n >= cap)
        n = cap - 1;
    memcpy(out, s, n);
    out[n] = 0;
}

/* the text after "key:" on its line, or NULL */
static const char *after(const char *t, const char *key)
{
    size_t k = strlen(key);
    for (const char *p = t; p && *p;) {
        if (!strncmp(p, key, k))
            return p + k;
        p = strchr(p, '\n');
        p = p ? p + 1 : NULL;
    }
    return NULL;
}

static uint64_t kib(const char *t, const char *key)
{
    const char *p = after(t, key);
    return p ? strtoull(p, NULL, 10) * 1024 : 0;
}

/* a field of a line into out, at most cap - 1 bytes; the rest after it */
static const char *word(const char *p, char *out, size_t cap)
{
    while (*p == ' ' || *p == '\t')
        p++;
    size_t n = 0;
    while (*p && *p != ' ' && *p != '\t' && *p != '\n') {
        if (n + 1 < cap)
            out[n++] = *p;
        p++;
    }
    out[n] = 0;
    return p;
}

static void copy(char *out, size_t cap, const char *s, size_t n)
{
    if (n >= cap)
        n = cap - 1;
    memcpy(out, s, n);
    out[n] = 0;
}

int sy_read_meminfo(const char *t, struct sy_status *s)
{
    s->mem_total = kib(t, "MemTotal:");
    s->mem_avail = kib(t, "MemAvailable:");
    s->swap_total = kib(t, "SwapTotal:");
    s->swap_free = kib(t, "SwapFree:");
    return s->mem_total ? 0 : -1;
}

int sy_read_cpu_ticks(const char *t, uint64_t *busy, uint64_t *all,
                      uint64_t *sys)
{
    if (strncmp(t, "cpu ", 4))
        return -1;
    /* user nice system idle iowait irq softirq steal (guest is in user) */
    uint64_t v[8] = {0};
    const char *p = t + 4;
    for (int i = 0; i < 8; i++) {
        char *e;
        v[i] = strtoull(p, &e, 10);
        if (e == p)
            break;
        p = e;
    }
    *all = 0;
    for (int i = 0; i < 8; i++)
        *all += v[i];
    *busy = *all - v[3] - v[4];
    *sys = v[2] + v[5] + v[6];
    return 0;
}

int sy_read_loadavg(const char *t, double load[3])
{
    return sscanf(t, "%lf %lf %lf", &load[0], &load[1], &load[2]) == 3 ? 0 : -1;
}

double sy_read_psi(const char *t)
{
    const char *p = t ? strstr(t, "some avg10=") : NULL;
    return p ? atof(p + 11) : -1;
}

int sy_read_pid_stat(const char *t, char *name, size_t cap, uint64_t *ticks,
                     int *threads, uint64_t *start)
{
    /* "pid (name) state ..."; the name may hold spaces and parentheses */
    const char *a = strchr(t, '('), *z = strrchr(t, ')');
    if (!a || !z || z < a)
        return -1;
    copy(name, cap, a + 1, (size_t)(z - a - 1));
    /* after ")": state is field 3; utime 14, stime 15, threads 20, start
       22 */
    const char *p = z + 1;
    uint64_t f[23] = {0};
    for (int i = 3; i <= 22 && *p; i++) {
        while (*p == ' ')
            p++;
        if (i > 3)
            f[i] = strtoull(p, NULL, 10);
        while (*p && *p != ' ')
            p++;
    }
    *ticks = f[14] + f[15];
    *threads = (int)f[20];
    *start = f[22];
    return 0;
}

/* the filesystems that hold files on a disk or a share */
static int real_fs(const char *fs)
{
    static const char *const ok[] = {
        "ext2",     "ext3",     "ext4",  "xfs",  "btrfs", "vfat",
        "exfat",    "ntfs",     "ntfs3", "f2fs", "zfs",   "jfs",
        "reiserfs", "hfsplus",  "nfs",   "nfs4", "cifs",  "smb3",
        "fuseblk",  "bcachefs", "apfs",  NULL};
    for (int i = 0; ok[i]; i++)
        if (!strcmp(fs, ok[i]))
            return 1;
    return 0;
}

/* mountinfo writes a space in a path as \040 */
static void unescape(char *s)
{
    char *w = s;
    for (char *r = s; *r; r++) {
        if (r[0] == '\\' && r[1] >= '0' && r[1] <= '3' && isdigit(r[2]) &&
            isdigit(r[3])) {
            *w++ = (char)((r[1] - '0') * 64 + (r[2] - '0') * 8 + (r[3] - '0'));
            r += 3;
        } else
            *w++ = *r;
    }
    *w = 0;
}

int sy_read_mountinfo(const char *t, struct sy_disks *d)
{
    memset(d, 0, sizeof *d);
    for (const char *line = t; line && *line;) {
        const char *end = strchr(line, '\n');
        size_t len = end ? (size_t)(end - line) : strlen(line);
        char l[2048];
        copy(l, sizeof l, line, len);
        line = end ? end + 1 : NULL;
        /* id parent maj:min root mount options [optional...] - fs source
           super-options */
        char id[16], parent[16], dev_no[32], root[512], mount[512], opts[256];
        const char *p = word(l, id, sizeof id);
        p = word(p, parent, sizeof parent);
        p = word(p, dev_no, sizeof dev_no);
        p = word(p, root, sizeof root);
        p = word(p, mount, sizeof mount);
        p = word(p, opts, sizeof opts);
        const char *dash = strstr(p, " - ");
        if (!dash)
            continue;
        char fs[32], source[256];
        p = word(dash + 3, fs, sizeof fs);
        word(p, source, sizeof source);
        if (!real_fs(fs))
            continue;
        unescape(mount);
        unescape(source);
        int seen = 0; /* a device mounted twice (btrfs subvolumes, binds) */
        for (int i = 0; i < d->n; i++)
            seen |= !strcmp(d->d[i].dev, source);
        if (seen)
            continue;
        if (d->n == SY_DISKS) {
            d->more++;
            continue;
        }
        struct sy_disk *k = &d->d[d->n++];
        sy_put(k->mount, sizeof k->mount, mount);
        sy_put(k->dev, sizeof k->dev, source);
        sy_put(k->fs, sizeof k->fs, fs);
        k->ro = !strncmp(opts, "ro", 2) && (opts[2] == ',' || !opts[2]);
    }
    return 0;
}

/* the same text but for its numbers (a process's id, a time, a count) */
static int same_but_numbers(const char *a, const char *b)
{
    while (*a && *b) {
        if (isdigit((unsigned char)*a) && isdigit((unsigned char)*b)) {
            while (isdigit((unsigned char)*a))
                a++;
            while (isdigit((unsigned char)*b))
                b++;
        } else if (*a++ != *b++)
            return 0;
    }
    return !*a && !*b;
}

void sy_err_add(struct sy_errors *e, long long at, const char *who,
                const char *text)
{
    e->total++;
    /* its first line: the rest (a core dump's modules and stack) is for
       the log's own reader */
    char t[240];
    while (*text == ' ' || *text == '\n')
        text++;
    size_t first = strcspn(text, "\n");
    copy(t, sizeof t, text, first);
    for (char *c = t; *c; c++)
        if (*c == '\r' || *c == '\t')
            *c = ' ';
    for (int i = 0; i < e->n; i++)
        if (!strcmp(e->e[i].who, who) && same_but_numbers(e->e[i].text, t)) {
            e->e[i].count++;
            if (at > e->e[i].at) { /* the latest's words */
                e->e[i].at = at;
                sy_put(e->e[i].text, sizeof e->e[i].text, t);
            }
            return;
        }
    if (e->n == SY_ERRS) { /* the oldest out, the newest in */
        int old = 0;
        for (int i = 1; i < e->n; i++)
            if (e->e[i].at < e->e[old].at)
                old = i;
        if (e->e[old].at > at) {
            e->more++;
            return;
        }
        e->more += e->e[old].count;
        e->e[old] = e->e[--e->n];
    }
    struct sy_err *x = &e->e[e->n++];
    x->at = at;
    x->count = 1;
    sy_put(x->who, sizeof x->who, who);
    sy_put(x->text, sizeof x->text, t);
}

static int later_first(const void *a, const void *b)
{
    long long x = ((const struct sy_err *)a)->at,
              y = ((const struct sy_err *)b)->at;
    return x < y ? 1 : x > y ? -1 : 0;
}

void sy_err_order(struct sy_errors *e)
{
    qsort(e->e, (size_t)e->n, sizeof *e->e, later_first);
}

int sy_read_journal(const char *t, size_t n, struct sy_errors *e)
{
    const char *end = t + n;
    for (const char *line = t; line < end;) {
        const char *nl = memchr(line, '\n', (size_t)(end - line));
        size_t len = nl ? (size_t)(nl - line) : (size_t)(end - line);
        char why[100];
        struct janas_json_doc *d =
            len ? janas_json_parse(line, len, why, sizeof why) : NULL;
        line = nl ? nl + 1 : end;
        if (!d)
            continue;
        const struct janas_json *r = janas_json_root(d);
        const char *msg = janas_json_str(janas_json_get(r, "MESSAGE"));
        const char *ts =
            janas_json_str(janas_json_get(r, "__REALTIME_TIMESTAMP"));
        /* who: the program, else its unit */
        const char *who =
            janas_json_str(janas_json_get(r, "SYSLOG_IDENTIFIER"));
        if (!who || !*who)
            who = janas_json_str(janas_json_get(r, "_SYSTEMD_USER_UNIT"));
        if (!who || !*who)
            who = janas_json_str(janas_json_get(r, "_SYSTEMD_UNIT"));
        if (msg) /* a message of bytes not text comes as a list: left out */
            sy_err_add(e, ts ? atoll(ts) / 1000000 : 0, who ? who : "", msg);
        janas_json_free(d);
    }
    return 0;
}

int sy_read_failed(const char *t, struct sy_errors *e)
{
    for (const char *line = t; line && *line;) {
        char unit[256];
        const char *p = line;
        while (*p == ' ' || *p == '*' || (unsigned char)*p >= 0x80)
            p++; /* the mark of a failed unit, "●" */
        word(p, unit, sizeof unit);
        line = strchr(line, '\n');
        line = line ? line + 1 : NULL;
        if (!unit[0])
            continue;
        /* the instances of a template, "name@a.service", counted once */
        char *at = strchr(unit, '@');
        if (at) {
            char *dot = strrchr(at, '.');
            memmove(at + 1, dot ? dot : at + strlen(at),
                    strlen(dot ? dot : at + strlen(at)) + 1);
        }
        int i = 0;
        while (i < e->n_failed && strcmp(e->failed[i], unit))
            i++;
        if (i < e->n_failed) {
            e->failed_n[i]++;
            continue;
        }
        if (e->n_failed == SY_FAILED) {
            e->more_failed++;
            continue;
        }
        sy_put(e->failed[i], sizeof e->failed[i], unit);
        e->failed_n[i] = 1;
        e->n_failed++;
    }
    return 0;
}

static void update_add(struct sy_updates *u, const char *name, const char *to,
                       const char *from, int security)
{
    u->total++;
    u->security += security;
    if (u->n == SY_UPDATES)
        return;
    struct sy_update *x = &u->u[u->n++];
    sy_put(x->name, sizeof x->name, name);
    sy_put(x->to, sizeof x->to, to);
    sy_put(x->from, sizeof x->from, from);
    x->security = security;
}

int sy_read_apt(const char *t, struct sy_updates *u)
{
    /* "name/suite version arch [upgradable from: old]" */
    for (const char *line = t; line && *line;) {
        char l[1024];
        const char *nl = strchr(line, '\n');
        copy(l, sizeof l, line, nl ? (size_t)(nl - line) : strlen(line));
        line = nl ? nl + 1 : NULL;
        char *slash = strchr(l, '/'), *from = strstr(l, "[upgradable from: ");
        if (!slash || !from || slash > from)
            continue;
        *slash = 0;
        char suite[128], ver[64];
        const char *p = word(slash + 1, suite, sizeof suite);
        word(p, ver, sizeof ver);
        char *old = from + 18, *close = strchr(old, ']');
        if (close)
            *close = 0;
        update_add(u, l, ver, old, strstr(suite, "-security") != NULL);
    }
    snprintf(u->tool, sizeof u->tool, "apt");
    return 0;
}

int sy_read_dnf(const char *t, struct sy_updates *u)
{
    /* after a blank line: "name.arch  version  repo"; "Obsoleting
       Packages" ends the list */
    for (const char *line = t; line && *line;) {
        char l[1024];
        const char *nl = strchr(line, '\n');
        copy(l, sizeof l, line, nl ? (size_t)(nl - line) : strlen(line));
        line = nl ? nl + 1 : NULL;
        if (!strncmp(l, "Obsoleting", 10))
            break;
        char name[128], ver[64], repo[64];
        const char *p = word(l, name, sizeof name);
        p = word(p, ver, sizeof ver);
        word(p, repo, sizeof repo);
        char *dot = strrchr(name, '.');
        if (!repo[0] || !dot || !isdigit((unsigned char)ver[0]))
            continue;
        *dot = 0;
        update_add(u, name, ver, "", 0);
    }
    snprintf(u->tool, sizeof u->tool, "dnf");
    return 0;
}

int sy_read_pacman(const char *t, struct sy_updates *u)
{
    for (const char *line = t; line && *line;) {
        char l[1024];
        const char *nl = strchr(line, '\n');
        copy(l, sizeof l, line, nl ? (size_t)(nl - line) : strlen(line));
        line = nl ? nl + 1 : NULL;
        char name[128], old[64], arrow[8], ver[64];
        const char *p = word(l, name, sizeof name);
        p = word(p, old, sizeof old);
        p = word(p, arrow, sizeof arrow);
        word(p, ver, sizeof ver);
        if (strcmp(arrow, "->") || !ver[0])
            continue;
        update_add(u, name, ver, old, 0);
    }
    snprintf(u->tool, sizeof u->tool, "pacman");
    return 0;
}

int sy_read_wireless(const char *t, const char *iface, int *dbm, int *pct)
{
    char key[80];
    snprintf(key, sizeof key, "%s:", iface);
    for (const char *line = t; line && *line;) {
        const char *p = line;
        while (*p == ' ')
            p++;
        if (!strncmp(p, key, strlen(key))) {
            int status;
            double link, level;
            if (sscanf(p + strlen(key), "%x %lf %lf", (unsigned *)&status,
                       &link, &level) != 3)
                return -1;
            *dbm = (int)level;
            *pct = (int)(link * 100 / 70 + 0.5); /* the quality is of 70 */
            if (*pct > 100)
                *pct = 100;
            return 0;
        }
        line = strchr(line, '\n');
        line = line ? line + 1 : NULL;
    }
    return -1;
}

int sy_read_route(const char *t, char *gw, size_t gw_cap, char *iface,
                  size_t iface_cap)
{
    for (const char *line = t; line && *line;) {
        char ifc[64];
        unsigned dest, via, flags;
        if (sscanf(line, "%63s %x %x %x", ifc, &dest, &via, &flags) == 4 &&
            dest == 0 && (flags & 2)) { /* RTF_GATEWAY */
            /* in the host's order: little-endian, the first byte first */
            snprintf(gw, gw_cap, "%u.%u.%u.%u", via & 255, (via >> 8) & 255,
                     (via >> 16) & 255, via >> 24);
            snprintf(iface, iface_cap, "%s", ifc);
            return 0;
        }
        line = strchr(line, '\n');
        line = line ? line + 1 : NULL;
    }
    return -1;
}

int sy_read_resolv(const char *t, struct sy_net *n)
{
    for (const char *line = t; line && *line;) {
        char key[32], v[64];
        if (sscanf(line, "%31s %63s", key, v) == 2 &&
            !strcmp(key, "nameserver") && n->n_dns < 3)
            sy_put(n->dns[n->n_dns++], sizeof n->dns[0], v);
        line = strchr(line, '\n');
        line = line ? line + 1 : NULL;
    }
    return 0;
}

int sy_read_os_release(const char *t, char *out, size_t cap)
{
    const char *p = after(t, "PRETTY_NAME=");
    if (!p)
        return -1;
    if (*p == '"')
        p++;
    size_t n = strcspn(p, "\"\n");
    copy(out, cap, p, n);
    return 0;
}

const char *sy_sensor_kind(const char *h)
{
    static const struct {
        const char *name, *kind;
    } K[] = {{"coretemp", "cpu"},    {"k10temp", "cpu"},  {"zenpower", "cpu"},
             {"cpu_thermal", "cpu"}, {"nvme", "disk"},    {"drivetemp", "disk"},
             {"spd5118", "memory"},  {"jc42", "memory"},  {"iwlwifi", "wifi"},
             {"ath", "wifi"},        {"mt7", "wifi"},     {"amdgpu", "gpu"},
             {"nouveau", "gpu"},     {"radeon", "gpu"},   {"i915", "gpu"},
             {"xe", "gpu"},          {"acpitz", "board"}, {"pch_", "chipset"},
             {"BAT", "battery"}};
    for (size_t i = 0; i < sizeof K / sizeof *K; i++)
        if (!strncmp(h, K[i].name, strlen(K[i].name)))
            return K[i].kind;
    return "other";
}
