/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * system.h - janas-system, this computer as an MCP service: how it is
 * (memory, processor, disks, battery, temperatures), what runs on it, its
 * errors, its updates, its network, what fills a directory. It only reads.
 *
 * What its parts share. The system is read by one backend a platform,
 * behind the sy_get_* calls: linux.c (from /proc and /sys, journalctl,
 * systemctl and the package manager) and windows.c (from the Win32 API);
 * usage.c walks a directory. read.c reads the texts of Linux (no system
 * here: the tests read kept texts), data.c and layouts.c write the
 * answers as data with a layout, tools.c offers the tools.
 */
#ifndef JANAS_SYSTEM_H
#define JANAS_SYSTEM_H

#include <stddef.h>
#include <stdint.h>

#include "llm/json.h"

#define SY_DISKS 16
#define SY_TEMPS 12
#define SY_PROCS 15
#define SY_ERRS 30
#define SY_FAILED 12
#define SY_UPDATES 40
#define SY_IFACES 12
#define SY_ADDRS 4
#define SY_USAGE 20

#define SY_SAMPLE_MS 500  /* between the two readings of the processor */
#define SY_USAGE_MS 8000  /* at most, to walk a directory */
#define SY_USAGE_DEPTH 64 /* levels at most below it */
#define SY_FULL_PCT 90    /* a disk this full is told */
#define SY_RUN_MS 20000   /* a program run (journalctl, apt) at most */

struct sy_disk {
    char mount[256], dev[128], fs[32];
    uint64_t total, avail; /* bytes; avail: what the user may write */
    int ro;
};

struct sy_disks {
    struct sy_disk d[SY_DISKS];
    int n, more;
};

/* kind: cpu, core (the hottest core), disk, memory, wifi, board, gpu,
   battery, chipset, other */
struct sy_temp {
    char kind[12], label[64];
    double c, high; /* high: the sensor's critical limit (else its maximum),
                       0 when none */
};

struct sy_battery {
    int present, pct; /* pct -1: not told */
    int mains;        /* 1 on mains, 0 not, -1 not known */
    char state[16];   /* charging, discharging, full, idle, unknown */
    int minutes;      /* to empty or to full, -1 when not known */
    int health;       /* % of the design capacity left, -1 */
};

struct sy_status {
    char host[64], os[128], kernel[96], cpu[128], platform[12];
    long uptime_s;
    int has_load, cpus;
    double load[3];
    double cpu_pct; /* of the whole processor, over SY_SAMPLE_MS */
    double sys_pct; /* of it, the kernel's */
    uint64_t mem_total, mem_avail, swap_total, swap_free;
    double psi_cpu, psi_mem, psi_io; /* % of time waited, 10 s; -1 none */
    struct sy_disks disks;
    struct sy_battery bat;
    struct sy_temp t[SY_TEMPS];
    int n_temps;
};

struct sy_proc {
    int pid, threads;
    char name[64], user[32];
    double cpu_pct; /* of the whole processor */
    uint64_t mem;   /* resident, bytes */
    long age_s;
};

struct sy_procs {
    struct sy_proc p[SY_PROCS];
    int n, total, matched;
    char by[8];    /* cpu or memory */
    char name[64]; /* the name asked, "" for all */
    uint64_t mem_total;
};

struct sy_err {
    long long at; /* seconds since 1970, the latest */
    char who[64], text[240];
    int count;
};

struct sy_errors {
    struct sy_err e[SY_ERRS];
    int n, more, total;
    char failed[SY_FAILED][96];
    int failed_n[SY_FAILED]; /* instances of a template unit (name@...) */
    int n_failed, more_failed;
    int hours, user_only;
    int cut;      /* the log read only in part: total is at least that */
    char how[16]; /* journal, eventlog */
};

struct sy_update {
    char name[96], to[64], from[64];
    int security;
};

struct sy_updates {
    struct sy_update u[SY_UPDATES];
    int n, total, security;
    long long checked; /* when the list was last fetched, 0 not known */
    char tool[16];     /* apt, dnf, pacman */
};

struct sy_iface {
    char name[64], kind[12], state[12]; /* kind: wifi, ethernet, virtual */
    char addr[SY_ADDRS][64];
    int n_addr;
    long speed_mbps;          /* 0 when not told */
    int signal_dbm, wifi_pct; /* 0 when not wireless */
    uint64_t rx, tx;          /* bytes since the start */
};

struct sy_net {
    struct sy_iface i[SY_IFACES];
    int n, more;
    char gateway[64], gw_iface[64];
    char dns[3][64];
    int n_dns;
};

struct sy_entry {
    char name[256];
    uint64_t bytes;
    long files;
    int dir;
};

struct sy_usage {
    char path[1024];
    struct sy_entry e[SY_USAGE];
    int n, more;
    uint64_t total, fs_total, fs_avail;
    long files, unread; /* unread: directories that could not be read */
    int cut;            /* the time ran out: sizes are at least these */
    double secs;
};

/* ---- the backend: linux.c or windows.c (and usage.c) ---- */

int sy_get_status(struct sy_status *s, char *err, size_t err_len);
int sy_get_disks(struct sy_disks *d, char *err, size_t err_len);
/* by "cpu" or "memory"; name: only the processes whose name has it */
int sy_get_procs(const char *by, const char *name, struct sy_procs *p,
                 char *err, size_t err_len);
int sy_get_errors(int hours, struct sy_errors *e, char *err, size_t err_len);
int sy_get_updates(struct sy_updates *u, char *err, size_t err_len);
int sy_get_net(struct sy_net *n, char *err, size_t err_len);
/* path: a directory (NULL: the home); its entries by size */
int sy_get_usage(const char *path, struct sy_usage *u, char *err,
                 size_t err_len);
/* the user's home directory, or "" */
const char *sy_home(void);

/* ---- read.c: the texts of Linux ---- */

/* s into out, cut to cap - 1 bytes */
void sy_put(char *out, size_t cap, const char *s);

/* /proc/meminfo into mem_* and swap_* */
int sy_read_meminfo(const char *t, struct sy_status *s);
/* /proc/stat's first line: ticks busy, in all, and the kernel's (system,
   interrupts) */
int sy_read_cpu_ticks(const char *t, uint64_t *busy, uint64_t *all,
                      uint64_t *sys);
/* /proc/loadavg */
int sy_read_loadavg(const char *t, double load[3]);
/* /proc/pressure/<x>: the "some avg10", -1 when not there */
double sy_read_psi(const char *t);
/* /proc/<pid>/stat: its name, its ticks (user and system), threads and
   start (ticks after the boot) */
int sy_read_pid_stat(const char *t, char *name, size_t cap, uint64_t *ticks,
                     int *threads, uint64_t *start);
/* /proc/self/mountinfo: the filesystems on a disk (or a network share),
   each device once */
int sy_read_mountinfo(const char *t, struct sy_disks *d);
/* journalctl -o json, a line an entry, the latest last */
int sy_read_journal(const char *t, size_t n, struct sy_errors *e);
/* systemctl --failed --plain --no-legend */
int sy_read_failed(const char *t, struct sy_errors *e);
/* apt list --upgradable, in LC_ALL=C */
int sy_read_apt(const char *t, struct sy_updates *u);
/* dnf check-update: a package a line, "name.arch version repo" */
int sy_read_dnf(const char *t, struct sy_updates *u);
/* pacman -Qu: "name old -> new" */
int sy_read_pacman(const char *t, struct sy_updates *u);
/* /proc/net/wireless: the signal of iface (dBm) and its quality (%) */
int sy_read_wireless(const char *t, const char *iface, int *dbm, int *pct);
/* /proc/net/route: the default gateway */
int sy_read_route(const char *t, char *gw, size_t gw_cap, char *iface,
                  size_t iface_cap);
/* resolv.conf's name servers */
int sy_read_resolv(const char *t, struct sy_net *n);
/* os-release's PRETTY_NAME */
int sy_read_os_release(const char *t, char *out, size_t cap);
/* the kind of a hwmon sensor from its name ("coretemp": cpu), "" to leave
   it out */
const char *sy_sensor_kind(const char *hwmon_name);
/* one error more into e: the same text from the same program counted */
void sy_err_add(struct sy_errors *e, long long at, const char *who,
                const char *text);
/* the latest errors first */
void sy_err_order(struct sy_errors *e);

/* ---- data.c ---- */

void sy_jstr(struct janas_buf *b, const char *key, const char *v);
void sy_status_data(struct janas_buf *b, const struct sy_status *s);
void sy_disks_data(struct janas_buf *b, const struct sy_disks *d);
void sy_procs_data(struct janas_buf *b, const struct sy_procs *p);
void sy_errors_data(struct janas_buf *b, const struct sy_errors *e);
void sy_updates_data(struct janas_buf *b, const struct sy_updates *u);
void sy_net_data(struct janas_buf *b, const struct sy_net *n);
void sy_usage_data(struct janas_buf *b, const struct sy_usage *u);

/* ---- layouts.c ---- */

extern const char SY_STATUS_LAYOUT[], SY_STATUS_BRIEF[];
extern const char SY_DISKS_LAYOUT[], SY_DISKS_BRIEF[];
extern const char SY_PROCS_LAYOUT[], SY_PROCS_BRIEF[];
extern const char SY_ERRORS_LAYOUT[], SY_ERRORS_BRIEF[];
extern const char SY_UPDATES_LAYOUT[], SY_UPDATES_BRIEF[];
extern const char SY_NET_LAYOUT[], SY_NET_BRIEF[];
extern const char SY_USAGE_LAYOUT[], SY_USAGE_BRIEF[];

/* ---- tools.c ---- */

void sy_tools_list(void *ctx, struct janas_buf *b);
int sy_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);

#endif
