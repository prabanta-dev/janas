/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * linux.c - janas-system's backend for Linux (see system.h): /proc and /sys
 * read as files, journalctl and systemctl for the errors, the package
 * manager (apt, dnf, pacman) for the updates, each from what it already
 * knows: nothing is fetched, nothing needs root.
 */
#ifdef __linux__
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include "common/sysinfo.h"
#include "services/common/run.h"
#include "system.h"

/* a small file of /proc or /sys whole into buf; its length, or -1 */
static long slurp(const char *path, char *buf, size_t cap)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    size_t n = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[n] = 0;
    return (long)n;
}

static long long num_file(const char *path, long long def)
{
    char b[64];
    return slurp(path, b, sizeof b) > 0 ? atoll(b) : def;
}

static void str_file(const char *path, char *out, size_t cap)
{
    out[0] = 0;
    if (slurp(path, out, cap) > 0)
        out[strcspn(out, "\n")] = 0;
}

/* a large file of /proc (mountinfo, many lines) into b */
static int slurp_buf(const char *path, struct janas_buf *b)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char chunk[8192];
    size_t n;
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0)
        janas_buf_put(b, chunk, n);
    fclose(f);
    janas_buf_put(b, "", 1);
    return b->oom ? -1 : 0;
}

const char *sy_home(void)
{
    const char *h = getenv("HOME");
    if (h && *h)
        return h;
    struct passwd *pw = getpwuid(getuid());
    return pw && pw->pw_dir ? pw->pw_dir : "";
}

static void sleep_ms(int ms)
{
    struct timespec t = {.tv_sec = ms / 1000,
                         .tv_nsec = (long)(ms % 1000) * 1000000L};
    while (nanosleep(&t, &t) != 0 && errno == EINTR)
        ;
}

int sy_get_disks(struct sy_disks *d, char *err, size_t err_len)
{
    struct janas_buf b = {0};
    if (slurp_buf("/proc/self/mountinfo", &b) != 0) {
        janas_buf_free(&b);
        snprintf(err, err_len, "/proc/self/mountinfo could not be read");
        return -1;
    }
    sy_read_mountinfo(b.p, d);
    janas_buf_free(&b);
    for (int i = 0; i < d->n; i++) {
        struct statvfs v;
        if (statvfs(d->d[i].mount, &v) == 0) {
            d->d[i].total = (uint64_t)v.f_blocks * v.f_frsize;
            d->d[i].avail = (uint64_t)v.f_bavail * v.f_frsize;
        }
    }
    /* the empty ones (a filesystem unreadable) left out */
    int k = 0;
    for (int i = 0; i < d->n; i++)
        if (d->d[i].total)
            d->d[k++] = d->d[i];
    d->n = k;
    return 0;
}

static void battery(struct sy_battery *b)
{
    memset(b, 0, sizeof *b);
    b->pct = b->minutes = b->health = -1;
    b->mains = janas_on_mains();
    snprintf(b->state, sizeof b->state, "unknown");
    DIR *dir = opendir("/sys/class/power_supply");
    if (!dir)
        return;
    struct dirent *e;
    char p[512], v[64];
    while ((e = readdir(dir))) {
        snprintf(p, sizeof p, "/sys/class/power_supply/%s/type", e->d_name);
        str_file(p, v, sizeof v);
        snprintf(p, sizeof p, "/sys/class/power_supply/%s/scope", e->d_name);
        char scope[32];
        str_file(p, scope, sizeof scope);
        if (strcmp(v, "Battery") || !strcmp(scope, "Device"))
            continue; /* a mouse's battery is not the computer's */
        char base[400];
        snprintf(base, sizeof base, "/sys/class/power_supply/%s", e->d_name);
        b->present = 1;
        snprintf(p, sizeof p, "%s/capacity", base);
        b->pct = (int)num_file(p, -1);
        snprintf(p, sizeof p, "%s/status", base);
        str_file(p, v, sizeof v);
        snprintf(b->state, sizeof b->state, "%s",
                 !strcmp(v, "Charging")       ? "charging"
                 : !strcmp(v, "Discharging")  ? "discharging"
                 : !strcmp(v, "Full")         ? "full"
                 : !strcmp(v, "Not charging") ? "idle"
                                              : "unknown");
        /* energy (µWh, µW) or charge (µAh, µA) */
        long long now, full, design, rate;
        snprintf(p, sizeof p, "%s/energy_now", base);
        now = num_file(p, -1);
        const char *u = now >= 0 ? "energy" : "charge";
        if (now < 0) {
            snprintf(p, sizeof p, "%s/charge_now", base);
            now = num_file(p, -1);
        }
        snprintf(p, sizeof p, "%s/%s_full", base, u);
        full = num_file(p, -1);
        snprintf(p, sizeof p, "%s/%s_full_design", base, u);
        design = num_file(p, -1);
        snprintf(p, sizeof p, "%s/%s", base,
                 u[0] == 'e' ? "power_now" : "current_now");
        rate = num_file(p, -1);
        if (rate < 0)
            rate = -rate; /* some drivers count a discharge below zero */
        if (rate > 0 && now >= 0 && !strcmp(b->state, "discharging"))
            b->minutes = (int)(now * 60 / rate);
        else if (rate > 0 && full > now && !strcmp(b->state, "charging"))
            b->minutes = (int)((full - now) * 60 / rate);
        if (design > 0 && full > 0)
            b->health = (int)(full * 100 / design);
        break;
    }
    closedir(dir);
}

static int hotter(const void *a, const void *b)
{
    double x = ((const struct sy_temp *)a)->c,
           y = ((const struct sy_temp *)b)->c;
    return x < y ? 1 : x > y ? -1 : 0;
}

/* a sensor a device: the processor's package and its hottest core, a
   disk's composite, the others' first */
static void temperatures(struct sy_status *s)
{
    DIR *dir = opendir("/sys/class/hwmon");
    if (!dir)
        return;
    struct dirent *e;
    int disks = 0;
    while ((e = readdir(dir)) && s->n_temps < SY_TEMPS - 1) {
        if (e->d_name[0] == '.')
            continue;
        char base[300], p[400], name[64];
        snprintf(base, sizeof base, "/sys/class/hwmon/%s", e->d_name);
        snprintf(p, sizeof p, "%s/name", base);
        str_file(p, name, sizeof name);
        const char *kind = sy_sensor_kind(name);
        /* the package, the composite or Tctl over the others; the first
           of those */
        double first = -300, first_high = 0, core = -300, core_high = 0;
        int rank = 0;
        for (int i = 1; i <= 64; i++) {
            snprintf(p, sizeof p, "%s/temp%d_input", base, i);
            long long mc = num_file(p, -1000000);
            if (mc == -1000000)
                continue;
            char label[64];
            snprintf(p, sizeof p, "%s/temp%d_label", base, i);
            str_file(p, label, sizeof label);
            /* the critical limit, else the maximum (spd5118's maximum is a
               warning set low) */
            snprintf(p, sizeof p, "%s/temp%d_crit", base, i);
            long long hi = num_file(p, 0);
            if (hi <= 0 || hi > 200000) {
                snprintf(p, sizeof p, "%s/temp%d_max", base, i);
                hi = num_file(p, 0);
            }
            if (hi > 200000) /* no limit, written as a huge one */
                hi = 0;
            double c = (double)mc / 1000;
            if (!strncmp(label, "Core", 4)) {
                if (c > core)
                    core = c, core_high = (double)hi / 1000;
                continue;
            }
            int r = !strncmp(label, "Package", 7) ||
                            !strcmp(label, "Composite") ||
                            !strcmp(label, "Tctl")
                        ? 2
                        : 1;
            if (r > rank)
                rank = r, first = c, first_high = (double)hi / 1000;
        }
        if (first > -300) {
            struct sy_temp *t = &s->t[s->n_temps++];
            sy_put(t->kind, sizeof t->kind, kind);
            if (!strcmp(kind, "disk") && disks++)
                snprintf(t->label, sizeof t->label, "%d", disks);
            t->c = first;
            t->high = first_high;
        }
        if (core > -300 && s->n_temps < SY_TEMPS) {
            struct sy_temp *t = &s->t[s->n_temps++];
            snprintf(t->kind, sizeof t->kind, "core");
            t->c = core;
            t->high = core_high;
        }
    }
    closedir(dir);
    qsort(s->t, (size_t)s->n_temps, sizeof *s->t, hotter);
}

static int cpu_ticks(uint64_t *busy, uint64_t *all, uint64_t *sys)
{
    char b[512];
    uint64_t s;
    return slurp("/proc/stat", b, sizeof b) > 0
               ? sy_read_cpu_ticks(b, busy, all, sys ? sys : &s)
               : -1;
}

static double uptime(void)
{
    char b[128];
    return slurp("/proc/uptime", b, sizeof b) > 0 ? atof(b) : 0;
}

int sy_get_status(struct sy_status *s, char *err, size_t err_len)
{
    memset(s, 0, sizeof *s);
    snprintf(s->platform, sizeof s->platform, "linux");
    uint64_t b0 = 0, a0 = 0, s0 = 0, b1 = 0, a1 = 0, s1 = 0;
    int ticks = cpu_ticks(&b0, &a0, &s0);
    char buf[4096];
    if (slurp("/proc/meminfo", buf, sizeof buf) < 0 ||
        sy_read_meminfo(buf, s) != 0) {
        snprintf(err, err_len, "/proc/meminfo could not be read");
        return -1;
    }
    gethostname(s->host, sizeof s->host - 1);
    struct utsname u;
    if (uname(&u) == 0)
        sy_put(s->kernel, sizeof s->kernel, u.release);
    if (slurp("/etc/os-release", buf, sizeof buf) > 0 ||
        slurp("/usr/lib/os-release", buf, sizeof buf) > 0)
        sy_read_os_release(buf, s->os, sizeof s->os);
    janas_cpu_name(s->cpu, sizeof s->cpu);
    s->cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (slurp("/proc/uptime", buf, sizeof buf) > 0)
        s->uptime_s = atol(buf);
    s->has_load = slurp("/proc/loadavg", buf, sizeof buf) > 0 &&
                  sy_read_loadavg(buf, s->load) == 0;
    s->psi_cpu = slurp("/proc/pressure/cpu", buf, sizeof buf) > 0
                     ? sy_read_psi(buf)
                     : -1;
    s->psi_mem = slurp("/proc/pressure/memory", buf, sizeof buf) > 0
                     ? sy_read_psi(buf)
                     : -1;
    s->psi_io =
        slurp("/proc/pressure/io", buf, sizeof buf) > 0 ? sy_read_psi(buf) : -1;
    char e2[100];
    sy_get_disks(&s->disks, e2, sizeof e2);
    battery(&s->bat);
    temperatures(s);
    /* the processor over the time all that took, and a little more */
    sleep_ms(SY_SAMPLE_MS);
    s->cpu_pct = -1;
    if (ticks == 0 && cpu_ticks(&b1, &a1, &s1) == 0 && a1 > a0) {
        s->cpu_pct = (double)(b1 - b0) * 100 / (double)(a1 - a0);
        s->sys_pct = (double)(s1 - s0) * 100 / (double)(a1 - a0);
    }
    return 0;
}

/* ---- the processes ---- */

struct tick {
    int pid;
    uint64_t t;
};

static int by_pid(const void *a, const void *b)
{
    return ((const struct tick *)a)->pid - ((const struct tick *)b)->pid;
}

static int is_pid(const char *s)
{
    if (!*s)
        return 0;
    for (; *s; s++)
        if (*s < '0' || *s > '9')
            return 0;
    return 1;
}

/* every process's ticks, sorted by pid; their count, or -1 */
static int all_ticks(struct tick **out)
{
    DIR *d = opendir("/proc");
    if (!d)
        return -1;
    int n = 0, cap = 512;
    struct tick *v = malloc((size_t)cap * sizeof *v);
    struct dirent *e;
    while (v && (e = readdir(d))) {
        if (!is_pid(e->d_name))
            continue;
        char p[300], b[1024], name[64];
        snprintf(p, sizeof p, "/proc/%s/stat", e->d_name);
        uint64_t t, start;
        int th;
        if (slurp(p, b, sizeof b) <= 0 ||
            sy_read_pid_stat(b, name, sizeof name, &t, &th, &start) != 0)
            continue;
        if (n == cap) {
            struct tick *w = realloc(v, (size_t)(cap *= 2) * sizeof *v);
            if (!w)
                break;
            v = w;
        }
        v[n].pid = atoi(e->d_name);
        v[n++].t = t;
    }
    closedir(d);
    if (!v)
        return -1;
    qsort(v, (size_t)n, sizeof *v, by_pid);
    *out = v;
    return n;
}

static int ci_has(const char *s, const char *what)
{
    size_t n = strlen(what);
    for (; *s; s++)
        if (!strncasecmp(s, what, n))
            return 1;
    return 0;
}

static int more_cpu(const void *a, const void *b)
{
    double x = ((const struct sy_proc *)a)->cpu_pct,
           y = ((const struct sy_proc *)b)->cpu_pct;
    uint64_t m = ((const struct sy_proc *)a)->mem,
             k = ((const struct sy_proc *)b)->mem;
    return x < y ? 1 : x > y ? -1 : m < k ? 1 : m > k ? -1 : 0;
}

static int more_mem(const void *a, const void *b)
{
    uint64_t x = ((const struct sy_proc *)a)->mem,
             y = ((const struct sy_proc *)b)->mem;
    return x < y ? 1 : x > y ? -1 : 0;
}

int sy_get_procs(const char *by, const char *name, struct sy_procs *p,
                 char *err, size_t err_len)
{
    memset(p, 0, sizeof *p);
    sy_put(p->by, sizeof p->by, by);
    sy_put(p->name, sizeof p->name, name ? name : "");
    p->mem_total = janas_mem_total();
    struct tick *before = NULL;
    uint64_t b0 = 0, a0 = 0, b1, a1;
    cpu_ticks(&b0, &a0, NULL);
    double up0 = uptime();
    int n0 = all_ticks(&before);
    if (n0 < 0) {
        snprintf(err, err_len, "/proc could not be read");
        return -1;
    }
    sleep_ms(SY_SAMPLE_MS);
    double all =
        cpu_ticks(&b1, &a1, NULL) == 0 && a1 > a0 ? (double)(a1 - a0) : 0;
    long hz = sysconf(_SC_CLK_TCK), page = sysconf(_SC_PAGESIZE);
    double up = 0;
    char buf[1024];
    if (slurp("/proc/uptime", buf, sizeof buf) > 0)
        up = atof(buf);
    /* the whole list kept to sort, the first SY_PROCS shown */
    int cap = 1024, n = 0;
    struct sy_proc *v = malloc((size_t)cap * sizeof *v);
    DIR *d = opendir("/proc");
    struct dirent *e;
    while (v && d && (e = readdir(d))) {
        if (!is_pid(e->d_name))
            continue;
        char path[300], pname[64];
        snprintf(path, sizeof path, "/proc/%s/stat", e->d_name);
        uint64_t t, start;
        int th;
        if (slurp(path, buf, sizeof buf) <= 0 ||
            sy_read_pid_stat(buf, pname, sizeof pname, &t, &th, &start) != 0)
            continue;
        if (th == 0) /* a zombie */
            continue;
        p->total++;
        if (name && *name && !ci_has(pname, name))
            continue;
        p->matched++;
        if (n == cap) {
            struct sy_proc *w = realloc(v, (size_t)(cap *= 2) * sizeof *v);
            if (!w)
                break;
            v = w;
        }
        struct sy_proc *x = &v[n++];
        memset(x, 0, sizeof *x);
        x->pid = atoi(e->d_name);
        x->threads = th;
        sy_put(x->name, sizeof x->name, pname);
        struct tick k = {.pid = x->pid}, *was = bsearch(&k, before, (size_t)n0,
                                                        sizeof *before, by_pid);
        /* /proc/stat counts every CPU's ticks: the share of the whole */
        if (was && all > 0 && t >= was->t)
            x->cpu_pct = (double)(t - was->t) * 100 / all;
        else if (!was && all > 0 && hz > 0 &&
                 (double)start / (double)hz >= up0 - 0.01)
            x->cpu_pct = (double)t * 100 / all; /* born while measured */
        snprintf(path, sizeof path, "/proc/%s/statm", e->d_name);
        char sm[128];
        if (slurp(path, sm, sizeof sm) > 0) {
            unsigned long long size, res;
            if (sscanf(sm, "%llu %llu", &size, &res) == 2)
                x->mem = res * (uint64_t)page;
        }
        x->age_s = hz > 0 ? (long)(up - (double)start / (double)hz) : 0;
        struct stat st;
        snprintf(path, sizeof path, "/proc/%s", e->d_name);
        if (stat(path, &st) == 0) {
            struct passwd *pw = getpwuid(st.st_uid);
            if (pw)
                sy_put(x->user, sizeof x->user, pw->pw_name);
            else
                snprintf(x->user, sizeof x->user, "%u", (unsigned)st.st_uid);
        }
    }
    if (d)
        closedir(d);
    free(before);
    if (!v) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    qsort(v, (size_t)n, sizeof *v, strcmp(by, "memory") ? more_cpu : more_mem);
    p->n = n < SY_PROCS ? n : SY_PROCS;
    memcpy(p->p, v, (size_t)p->n * sizeof *v);
    free(v);
    return 0;
}

/* ---- the errors ---- */

int sy_get_errors(int hours, struct sy_errors *e, char *err, size_t err_len)
{
    memset(e, 0, sizeof *e);
    e->hours = hours;
    snprintf(e->how, sizeof e->how, "journal");
    static const char *const env[] = {"LC_ALL=C", "SYSTEMD_COLORS=0",
                                      "SYSTEMD_PAGER=", NULL};
    char since[48], why[300];
    if (hours > 0)
        snprintf(since, sizeof since, "--since=-%dh", hours);
    else
        snprintf(since, sizeof since, "--boot");
    const char *av[] = {"journalctl",
                        "--priority=0..3",
                        since,
                        "--no-pager",
                        "--output=json",
                        "--reverse", /* the latest first: a cut drops the
                                        oldest */
                        "--output-fields=MESSAGE,SYSLOG_IDENTIFIER,"
                        "_SYSTEMD_UNIT,_SYSTEMD_USER_UNIT",
                        "--lines=2000",
                        NULL};
    struct janas_buf out = {0}, er = {0};
    int rc = janas_run(NULL, av, env, SY_RUN_MS, &out, &er, why, sizeof why);
    if (rc < 0) {
        janas_buf_free(&out);
        janas_buf_free(&er);
        snprintf(err, err_len, "the journal could not be read: %s", why);
        return -1;
    }
    sy_read_journal(out.p ? out.p : "", out.n, e);
    e->cut = out.n >= JANAS_RUN_OUT_MAX || e->total >= 2000;
    /* without the groups adm or systemd-journal, only the user's own */
    e->user_only = er.p && strstr(er.p, "not seeing messages") != NULL;
    janas_buf_free(&out);
    janas_buf_free(&er);
    sy_err_order(e);
    for (int user = 0; user < 2; user++) {
        const char *sv[] = {"systemctl",   user ? "--user" : "--system",
                            "--failed",    "--plain",
                            "--no-legend", "--no-pager",
                            NULL};
        struct janas_buf o = {0}, x = {0};
        if (janas_run(NULL, sv, env, SY_RUN_MS, &o, &x, why, sizeof why) >= 0)
            sy_read_failed(o.p ? o.p : "", e);
        janas_buf_free(&o);
        janas_buf_free(&x);
    }
    return 0;
}

/* ---- the updates ---- */

static int has_program(const char *name)
{
    const char *path = getenv("PATH");
    char dir[4096];
    snprintf(dir, sizeof dir, "%s",
             path && *path ? path : "/usr/local/bin:/usr/bin:/bin");
    for (char *s = strtok(dir, ":"); s; s = strtok(NULL, ":")) {
        char p[4200];
        snprintf(p, sizeof p, "%s/%s", s, name);
        if (access(p, X_OK) == 0)
            return 1;
    }
    return 0;
}

static long long mtime(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 ? (long long)st.st_mtime : 0;
}

int sy_get_updates(struct sy_updates *u, char *err, size_t err_len)
{
    memset(u, 0, sizeof *u);
    static const char *const env[] = {"LC_ALL=C", NULL};
    char why[300];
    struct janas_buf out = {0}, er = {0};
    int rc;
    if (has_program("apt")) {
        const char *av[] = {"apt", "list", "--upgradable", NULL};
        rc = janas_run(NULL, av, env, SY_RUN_MS, &out, &er, why, sizeof why);
        if (rc >= 0)
            sy_read_apt(out.p ? out.p : "", u);
        u->checked = mtime("/var/cache/apt/pkgcache.bin");
    } else if (has_program("dnf")) {
        /* from the metadata it has: --cacheonly fetches nothing */
        const char *av[] = {"dnf", "check-update", "--cacheonly", "--quiet",
                            NULL};
        rc = janas_run(NULL, av, env, SY_RUN_MS, &out, &er, why, sizeof why);
        if (rc == 0 || rc == 100) {
            sy_read_dnf(out.p ? out.p : "", u);
            rc = 0;
        }
        snprintf(u->tool, sizeof u->tool, "dnf");
    } else if (has_program("pacman")) {
        /* against the database as last synchronised */
        const char *av[] = {"pacman", "-Qu", NULL};
        rc = janas_run(NULL, av, env, SY_RUN_MS, &out, &er, why, sizeof why);
        if (rc == 0 || rc == 1) { /* 1: nothing to update */
            sy_read_pacman(out.p ? out.p : "", u);
            rc = 0;
        }
        snprintf(u->tool, sizeof u->tool, "pacman");
        u->checked = mtime("/var/lib/pacman/sync");
    } else {
        snprintf(err, err_len,
                 "no package manager known here (apt, dnf, pacman)");
        return -1;
    }
    janas_buf_free(&out);
    janas_buf_free(&er);
    if (rc < 0) {
        snprintf(err, err_len, "%s", why);
        return -1;
    }
    return 0;
}

/* ---- the network ---- */

int sy_get_net(struct sy_net *n, char *err, size_t err_len)
{
    memset(n, 0, sizeof *n);
    struct ifaddrs *ifs;
    if (getifaddrs(&ifs) != 0) {
        snprintf(err, err_len, "the interfaces could not be read: %s",
                 strerror(errno));
        return -1;
    }
    char wireless[2048] = "";
    slurp("/proc/net/wireless", wireless, sizeof wireless);
    for (struct ifaddrs *a = ifs; a; a = a->ifa_next) {
        if (a->ifa_flags & IFF_LOOPBACK)
            continue;
        int i = 0;
        while (i < n->n && strcmp(n->i[i].name, a->ifa_name))
            i++;
        if (i == n->n) {
            if (n->n == SY_IFACES) {
                n->more++;
                continue;
            }
            struct sy_iface *x = &n->i[n->n++];
            sy_put(x->name, sizeof x->name, a->ifa_name);
            char p[300], v[32];
            snprintf(p, sizeof p, "/sys/class/net/%s/wireless", x->name);
            struct stat st;
            int wifi = stat(p, &st) == 0;
            snprintf(p, sizeof p, "/sys/class/net/%s/device", x->name);
            int dev = stat(p, &st) == 0;
            snprintf(x->kind, sizeof x->kind, "%s",
                     wifi  ? "wifi"
                     : dev ? "ethernet"
                           : "virtual");
            snprintf(p, sizeof p, "/sys/class/net/%s/operstate", x->name);
            str_file(p, v, sizeof v);
            sy_put(x->state, sizeof x->state, v[0] ? v : "unknown");
            snprintf(p, sizeof p, "/sys/class/net/%s/speed", x->name);
            long long sp = num_file(p, 0);
            x->speed_mbps = sp > 0 ? (long)sp : 0;
            snprintf(p, sizeof p, "/sys/class/net/%s/statistics/rx_bytes",
                     x->name);
            x->rx = (uint64_t)num_file(p, 0);
            snprintf(p, sizeof p, "/sys/class/net/%s/statistics/tx_bytes",
                     x->name);
            x->tx = (uint64_t)num_file(p, 0);
            if (wifi)
                sy_read_wireless(wireless, x->name, &x->signal_dbm,
                                 &x->wifi_pct);
        }
        struct sy_iface *x = &n->i[i];
        if (!a->ifa_addr || x->n_addr == SY_ADDRS)
            continue;
        int fam = a->ifa_addr->sa_family;
        if (fam != AF_INET && fam != AF_INET6)
            continue;
        char host[64];
        if (getnameinfo(a->ifa_addr,
                        fam == AF_INET ? sizeof(struct sockaddr_in)
                                       : sizeof(struct sockaddr_in6),
                        host, sizeof host, NULL, 0, NI_NUMERICHOST) != 0)
            continue;
        sy_put(x->addr[x->n_addr++], sizeof x->addr[0], host);
    }
    freeifaddrs(ifs);
    /* the virtual ones down and without addresses are noise */
    int k = 0;
    for (int i = 0; i < n->n; i++)
        if (strcmp(n->i[i].kind, "virtual") || n->i[i].n_addr)
            n->i[k++] = n->i[i];
    n->n = k;
    char buf[8192];
    if (slurp("/proc/net/route", buf, sizeof buf) > 0)
        sy_read_route(buf, n->gateway, sizeof n->gateway, n->gw_iface,
                      sizeof n->gw_iface);
    if (slurp("/etc/resolv.conf", buf, sizeof buf) > 0)
        sy_read_resolv(buf, n);
    return 0;
}
#endif
