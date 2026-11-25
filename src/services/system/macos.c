/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * macos.c - janas-system's backend on macOS (see system.h): sysctl and the
 * Mach kernel for the machine, getmntinfo for the disks, libproc for the
 * processes, getifaddrs for the network. The system's log and its updates
 * are not read here: each answer says where they are.
 */
#ifdef __APPLE__
#include <errno.h>
#include <ifaddrs.h>
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <netdb.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "common/sysinfo.h"
#include "system.h"

/* a string from sysctl, "" when none */
static void sysctl_str(const char *name, char *out, size_t cap)
{
    size_t n = cap;
    if (!cap)
        return;
    if (sysctlbyname(name, out, &n, NULL, 0) != 0)
        out[0] = 0;
    out[cap - 1] = 0;
}

int sy_get_disks(struct sy_disks *d, char *err, size_t err_len)
{
    memset(d, 0, sizeof *d);
    struct statfs *m = NULL;
    int n = getmntinfo(&m, MNT_NOWAIT);
    if (n <= 0) {
        snprintf(err, err_len, "the mounted filesystems could not be read");
        return -1;
    }
    for (int i = 0; i < n; i++) {
        /* the volumes a user sees: local, browsable, not empty */
        if (!(m[i].f_flags & MNT_LOCAL) || (m[i].f_flags & MNT_DONTBROWSE) ||
            !m[i].f_blocks)
            continue;
        if (d->n == SY_DISKS) {
            d->more++;
            continue;
        }
        struct sy_disk *k = &d->d[d->n++];
        sy_put(k->mount, sizeof k->mount, m[i].f_mntonname);
        sy_put(k->dev, sizeof k->dev, m[i].f_mntfromname);
        sy_put(k->fs, sizeof k->fs, m[i].f_fstypename);
        k->total = (uint64_t)m[i].f_blocks * m[i].f_bsize;
        k->avail = (uint64_t)m[i].f_bavail * m[i].f_bsize;
        k->ro = (m[i].f_flags & MNT_RDONLY) != 0;
    }
    return 0;
}

/* busy and all ticks of every core */
static int cpu_ticks(uint64_t *busy, uint64_t *sys, uint64_t *all)
{
    natural_t n = 0;
    processor_info_array_t info;
    mach_msg_type_number_t cnt;
    if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &n,
                            &info, &cnt) != KERN_SUCCESS)
        return -1;
    *busy = *sys = *all = 0;
    for (natural_t c = 0; c < n; c++) {
        const integer_t *t = info + c * CPU_STATE_MAX;
        uint64_t u = (uint32_t)t[CPU_STATE_USER],
                 s = (uint32_t)t[CPU_STATE_SYSTEM],
                 ni = (uint32_t)t[CPU_STATE_NICE],
                 id = (uint32_t)t[CPU_STATE_IDLE];
        *busy += u + s + ni;
        *sys += s;
        *all += u + s + ni + id;
    }
    vm_deallocate(mach_task_self(), (vm_address_t)info,
                  cnt * sizeof(integer_t));
    return 0;
}

int sy_get_status(struct sy_status *s, char *err, size_t err_len)
{
    memset(s, 0, sizeof *s);
    sy_put(s->platform, sizeof s->platform, "macos");
    gethostname(s->host, sizeof s->host);
    s->host[sizeof s->host - 1] = 0;
    char ver[32] = "", build[32] = "";
    sysctl_str("kern.osproductversion", ver, sizeof ver);
    sysctl_str("kern.osversion", build, sizeof build);
    snprintf(s->os, sizeof s->os, "macOS %s%s%s%s", ver, *build ? " (" : "",
             build, *build ? ")" : "");
    char rel[64] = "";
    sysctl_str("kern.osrelease", rel, sizeof rel);
    snprintf(s->kernel, sizeof s->kernel, "Darwin %s", rel);
    janas_cpu_name(s->cpu, sizeof s->cpu);
    s->cpus = janas_cpu_count();
    struct timeval boot;
    size_t bn = sizeof boot;
    if (sysctlbyname("kern.boottime", &boot, &bn, NULL, 0) == 0)
        s->uptime_s = (long)(time(NULL) - boot.tv_sec);
    s->has_load = getloadavg(s->load, 3) == 3;
    uint64_t b0, y0, a0, b1, y1, a1;
    if (cpu_ticks(&b0, &y0, &a0) == 0) {
        usleep(SY_SAMPLE_MS * 1000);
        if (cpu_ticks(&b1, &y1, &a1) == 0 && a1 > a0) {
            s->cpu_pct = 100.0 * (double)(b1 - b0) / (double)(a1 - a0);
            s->sys_pct = 100.0 * (double)(y1 - y0) / (double)(a1 - a0);
        }
    }
    s->mem_total = janas_mem_total();
    s->mem_avail = janas_mem_available();
    struct xsw_usage sw;
    size_t sn = sizeof sw;
    if (sysctlbyname("vm.swapusage", &sw, &sn, NULL, 0) == 0) {
        s->swap_total = sw.xsu_total;
        s->swap_free = sw.xsu_avail;
    }
    s->psi_cpu = s->psi_mem = s->psi_io = -1; /* Linux's pressure only */
    s->bat.pct = s->bat.minutes = s->bat.health = -1;
    s->bat.mains = janas_on_mains();
    sy_put(s->bat.state, sizeof s->bat.state, "unknown");
    /* the disks as part of the state, none when they cannot be read */
    if (sy_get_disks(&s->disks, err, err_len) != 0)
        err[0] = 0;
    return 0;
}

/* ---- the processes ---- */

static int by_cpu(const void *a, const void *b)
{
    double x = ((const struct sy_proc *)a)->cpu_pct,
           y = ((const struct sy_proc *)b)->cpu_pct;
    return x < y ? 1 : x > y ? -1 : 0;
}

static int by_mem(const void *a, const void *b)
{
    uint64_t x = ((const struct sy_proc *)a)->mem,
             y = ((const struct sy_proc *)b)->mem;
    return x < y ? 1 : x > y ? -1 : 0;
}

int sy_get_procs(const char *by, const char *name, struct sy_procs *p,
                 char *err, size_t err_len)
{
    memset(p, 0, sizeof *p);
    int mem = by && strcmp(by, "memory") == 0;
    sy_put(p->by, sizeof p->by, mem ? "memory" : "cpu");
    sy_put(p->name, sizeof p->name, name ? name : "");
    p->mem_total = janas_mem_total();
    int cap = proc_listallpids(NULL, 0);
    if (cap <= 0) {
        snprintf(err, err_len, "the processes could not be listed");
        return -1;
    }
    cap += 64;
    pid_t *pids = calloc((size_t)cap, sizeof *pids);
    int n = pids ? proc_listallpids(pids, cap * (int)sizeof *pids) : -1;
    struct sy_proc *all = n > 0 ? calloc((size_t)n, sizeof *all) : NULL;
    uint64_t *t0 = n > 0 ? calloc((size_t)n, sizeof *t0) : NULL;
    if (!all || !t0) {
        free(pids), free(all), free(t0);
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    /* the processor's time over SY_SAMPLE_MS, each process's */
    for (int i = 0; i < n; i++) {
        struct proc_taskinfo ti;
        if (proc_pidinfo(pids[i], PROC_PIDTASKINFO, 0, &ti, sizeof ti) ==
            (int)sizeof ti)
            t0[i] = ti.pti_total_user + ti.pti_total_system;
    }
    usleep(SY_SAMPLE_MS * 1000);
    int k = 0;
    time_t now = time(NULL);
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    int cpus = janas_cpu_count();
    for (int i = 0; i < n; i++) {
        struct proc_taskallinfo ai;
        if (proc_pidinfo(pids[i], PROC_PIDTASKALLINFO, 0, &ai, sizeof ai) !=
            (int)sizeof ai)
            continue;
        p->total++;
        const char *pn =
            ai.pbsd.pbi_name[0] ? ai.pbsd.pbi_name : ai.pbsd.pbi_comm;
        if (name && *name && !strstr(pn, name))
            continue;
        p->matched++;
        struct sy_proc *q = &all[k++];
        q->pid = pids[i];
        q->threads = ai.ptinfo.pti_threadnum;
        sy_put(q->name, sizeof q->name, pn);
        struct passwd *pw = getpwuid(ai.pbsd.pbi_uid);
        sy_put(q->user, sizeof q->user, pw ? pw->pw_name : "");
        q->mem = ai.ptinfo.pti_resident_size;
        uint64_t t1 = ai.ptinfo.pti_total_user + ai.ptinfo.pti_total_system;
        /* the task's times in Mach ticks: nanoseconds by the timebase */
        double ns = (double)(t1 - t0[i]) * tb.numer / tb.denom;
        q->cpu_pct =
            t0[i] && t1 >= t0[i] ? ns / (SY_SAMPLE_MS * 1e6) / cpus * 100.0 : 0;
        q->age_s = (long)(now - (time_t)ai.pbsd.pbi_start_tvsec);
    }
    qsort(all, (size_t)k, sizeof *all, mem ? by_mem : by_cpu);
    p->n = k < SY_PROCS ? k : SY_PROCS;
    memcpy(p->p, all, (size_t)p->n * sizeof *all);
    free(pids), free(all), free(t0);
    return 0;
}

int sy_get_errors(int hours, struct sy_errors *e, char *err, size_t err_len)
{
    memset(e, 0, sizeof *e);
    e->hours = hours;
    snprintf(err, err_len,
             "on macOS the system's log is not read here: it is in the "
             "Console app, or `log show --last 1h --predicate "
             "'messageType == error'` in a terminal");
    return -1;
}

int sy_get_updates(struct sy_updates *u, char *err, size_t err_len)
{
    memset(u, 0, sizeof *u);
    snprintf(err, err_len,
             "on macOS the updates are not read here: they are in System "
             "Settings, General, Software Update (`softwareupdate -l` in a "
             "terminal, `brew outdated` for Homebrew's)");
    return -1;
}

/* ---- the network ---- */

int sy_get_net(struct sy_net *n, char *err, size_t err_len)
{
    memset(n, 0, sizeof *n);
    struct ifaddrs *ifs = NULL;
    if (getifaddrs(&ifs) != 0) {
        snprintf(err, err_len, "the network interfaces could not be read");
        return -1;
    }
    for (struct ifaddrs *a = ifs; a; a = a->ifa_next) {
        if (!a->ifa_name || (a->ifa_flags & IFF_LOOPBACK))
            continue;
        struct sy_iface *f = NULL;
        for (int i = 0; i < n->n; i++)
            if (strcmp(n->i[i].name, a->ifa_name) == 0)
                f = &n->i[i];
        if (!f) {
            if (n->n == SY_IFACES) {
                n->more++;
                continue;
            }
            f = &n->i[n->n++];
            sy_put(f->name, sizeof f->name, a->ifa_name);
            sy_put(f->kind, sizeof f->kind,
                   strncmp(a->ifa_name, "en", 2) == 0 ? "ethernet" : "virtual");
            sy_put(f->state, sizeof f->state,
                   (a->ifa_flags & IFF_UP) && (a->ifa_flags & IFF_RUNNING)
                       ? "up"
                       : "down");
        }
        if (!a->ifa_addr)
            continue;
        int fam = a->ifa_addr->sa_family;
        if (fam == AF_LINK && a->ifa_data) {
            const struct if_data *d = a->ifa_data;
            f->rx = d->ifi_ibytes;
            f->tx = d->ifi_obytes;
            f->speed_mbps = (long)(d->ifi_baudrate / 1000000);
        } else if ((fam == AF_INET || fam == AF_INET6) &&
                   f->n_addr < SY_ADDRS) {
            char host[NI_MAXHOST];
            socklen_t len = fam == AF_INET ? sizeof(struct sockaddr_in)
                                           : sizeof(struct sockaddr_in6);
            if (getnameinfo(a->ifa_addr, len, host, sizeof host, NULL, 0,
                            NI_NUMERICHOST) == 0)
                sy_put(f->addr[f->n_addr++], sizeof f->addr[0], host);
        }
    }
    freeifaddrs(ifs);
    /* the interfaces with no address left out: macOS has many idle ones */
    int k = 0;
    for (int i = 0; i < n->n; i++)
        if (n->i[i].n_addr)
            n->i[k++] = n->i[i];
    n->n = k;
    return 0;
}

#endif /* __APPLE__ */
