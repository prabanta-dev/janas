/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_system_parse.c - janas-system's reading of Linux's texts, without
 * reading the system, and the layouts filled with what was read. The
 * texts are those of the development laptop (Debian 13) on 2 October
 * 2026, cut; the btrfs subvolume and the path with a space in mountinfo,
 * and the outputs of dnf and pacman, are written from their documentation
 * (no such system was at hand).
 * The sources belong to the program, so they are compiled in here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "services/common/template.h"
#include "services/system/data.c"
#include "services/system/layouts.c"
#include "services/system/read.c"

/* the backend is not compiled in: the home for the paths */
const char *sy_home(void)
{
    return "/home/someone";
}

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char S_MEMINFO[] = "MemTotal:       32280768 kB\n"
                                "MemFree:         2973200 kB\n"
                                "MemAvailable:   24510148 kB\n"
                                "Buffers:          524688 kB\n"
                                "SwapTotal:      32982012 kB\n"
                                "SwapFree:       31316232 kB\n";

static const char S_STAT[] =
    "cpu  333284235 3886676 54046493 3968942521 6156313 0 292631 0 0 0\n"
    "cpu0 1 2 3 4 5 6 7 8 9 10\n";

static const char S_PSI[] =
    "some avg10=0.01 avg60=0.02 avg300=0.00 total=8039391441\n"
    "full avg10=0.01 avg60=0.02 avg300=0.00 total=6399024250\n";

static const char S_PID_STAT[] =
    "2011430 (Isolated Web Co) S 1981282 1981282 1981282 0 -1 4194560 337955 "
    "0 4 0 4475 282 0 0 20 0 35 0 198506288 3139346432 118613 "
    "18446744073709551615 94824029778960 94824030202304 140726984299296 0 0 "
    "0 0 69634 1082134264 0 0 0 17 6 0 0 0 0 0\n";

static const char S_MOUNTINFO[] =
    "25 31 0:23 / /proc rw,nosuid,nodev,noexec,relatime shared:12 - proc "
    "proc rw\n"
    "28 31 0:25 / /run rw,nosuid,nodev,noexec,relatime shared:5 - tmpfs "
    "tmpfs rw,size=3228080k,mode=755,inode64\n"
    "31 1 259:5 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p5 "
    "rw,errors=remount-ro,stripe=128\n"
    "49 31 259:1 / /boot/efi rw,relatime shared:88 - vfat /dev/nvme0n1p1 "
    "rw,fmask=0077,dmask=0077,codepage=437,iocharset=ascii,shortname=mixed,"
    "utf8,errors=remount-ro\n"
    "191 161 0:69 / /run/user/1000/doc rw,nosuid,nodev,relatime shared:660 "
    "- fuse.portal portal rw,user_id=1000,group_id=1000\n"
    /* written for the test */
    "80 31 0:50 /@home /home rw,relatime shared:30 - btrfs /dev/sda2 rw\n"
    "81 31 0:50 /@snap /home/snap rw,relatime shared:31 - btrfs /dev/sda2 "
    "rw\n"
    "90 31 8:17 / /media/someone/My\\040Disk ro,nosuid shared:40 - exfat "
    "/dev/sdb1 ro\n";

static const char S_JOURNAL[] =
    "{\"_SYSTEMD_USER_UNIT\":\"plasma-kwin_wayland.service\",\"_SYSTEMD_UNIT\":"
    "\"user@1000.service\",\"__REALTIME_TIMESTAMP\":\"1790961552458210\","
    "\"PRIORITY\":\"3\",\"MESSAGE\":\"PAM adding faulty module: pam_sss.so\","
    "\"SYSLOG_IDENTIFIER\":\"kscreenlocker_greet\"}\n"
    "{\"_SYSTEMD_UNIT\":\"user@1000.service\",\"__REALTIME_TIMESTAMP\":"
    "\"1790967655461074\",\"MESSAGE\":\"PAM unable to dlopen(pam_sss.so): "
    "/usr/lib/security/pam_sss.so: cannot open shared object file\","
    "\"PRIORITY\":\"3\",\"SYSLOG_IDENTIFIER\":\"kscreenlocker_greet\"}\n"
    "{\"SYSLOG_IDENTIFIER\":\"kscreenlocker_greet\",\"MESSAGE\":\"PAM adding "
    "faulty module: pam_sss.so\",\"__REALTIME_TIMESTAMP\":"
    "\"1790967655464070\",\"PRIORITY\":\"3\"}\n"
    /* a message of bytes not text: journalctl writes it as a list */
    "{\"SYSLOG_IDENTIFIER\":\"x\",\"MESSAGE\":[80,65,77],"
    "\"__REALTIME_TIMESTAMP\":\"1790967655464071\"}\n";

static const char S_FAILED[] =
    "drkonqi-coredump-processor@1022-3261128-0.service loaded failed failed "
    "Pass systemd-coredump journal entries to relevant user for potential "
    "DrKonqi handling\n"
    "drkonqi-coredump-processor@1118-2837842-0.service loaded failed failed "
    "Pass systemd-coredump journal entries to relevant user for potential "
    "DrKonqi handling\n"
    "\xe2\x97\x8f cups.service loaded failed failed CUPS Scheduler\n";

static const char S_APT[] =
    "Listing...\n"
    "bind9-dnsutils/stable-security 1:9.20.29-1~deb13u1 amd64 [upgradable "
    "from: 1:9.20.26-1~deb13u1]\n"
    "chromium-common/stable-security 154.0.8037.92-1~deb13u1 amd64 "
    "[upgradable from: 152.0.7977.82-1~deb13u1]\n"
    "tzdata/stable 2026b-0+deb13u1 all [upgradable from: 2026a-0+deb13u1]\n";

static const char S_DNF[] = "\n"
                            "kernel.x86_64       6.11.4-301.fc41   updates\n"
                            "vim-minimal.x86_64  2:9.1.785-1.fc41  updates\n"
                            "Obsoleting Packages\n"
                            "grub2-tools.x86_64  1:2.12-10.fc41    updates\n";

static const char S_PACMAN[] = "linux 6.11.3.arch1-1 -> 6.11.4.arch1-1\n"
                               "firefox 131.0.2-1 -> 131.0.3-1\n";

static const char S_WIRELESS[] =
    "Inter-| sta-|   Quality        |   Discarded packets               | "
    "Missed | WE\n"
    " face | tus | link level noise |  nwid  crypt   frag  retry   misc | "
    "beacon | 22\n"
    "wlp0s20f3: 0000   50.  -60.  -256        0      0      0    343  10299 "
    "       0\n";

static const char S_ROUTE[] =
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\t"
    "Window\tIRTT\n"
    "wlp0s20f3\t0001A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0\n"
    "wlp0s20f3\t00000000\t0101A8C0\t0003\t0\t0\t600\t00000000\t0\t0\t0\n";

static const char S_OS[] = "PRETTY_NAME=\"Debian GNU/Linux 13 (trixie)\"\n"
                           "NAME=\"Debian GNU/Linux\"\n";

/* the layout filled with the data d; the text, or "" (counted) */
static char *fill(const char *layout, struct janas_buf *d)
{
    static char out[16384];
    char why[200];
    struct janas_json_doc *doc = janas_json_parse(d->p, d->n, why, sizeof why);
    CHECK(doc, "the data is not JSON: %s\n%.*s", why, (int)d->n, d->p);
    out[0] = 0;
    if (doc) {
        struct janas_buf o = {0};
        char err[200];
        int r = janas_tpl_render(layout, strlen(layout), janas_json_root(doc),
                                 &o, err, sizeof err);
        CHECK(r == 0, "not filled: %s", err);
        if (r == 0 && o.p)
            snprintf(out, sizeof out, "%.*s", (int)o.n, o.p);
        janas_buf_free(&o);
        janas_json_free(doc);
    }
    janas_buf_free(d);
    return out;
}

static void test_status(void)
{
    struct sy_status s;
    memset(&s, 0, sizeof s);
    CHECK(sy_read_meminfo(S_MEMINFO, &s) == 0, "meminfo");
    CHECK(s.mem_total == 32280768ull * 1024 &&
              s.mem_avail == 24510148ull * 1024,
          "memory %llu %llu", (unsigned long long)s.mem_total,
          (unsigned long long)s.mem_avail);
    CHECK(s.swap_total == 32982012ull * 1024, "swap");
    uint64_t busy, all, sys;
    CHECK(sy_read_cpu_ticks(S_STAT, &busy, &all, &sys) == 0, "stat");
    CHECK(sys == 54046493ull + 292631, "kernel ticks");
    CHECK(all == 333284235ull + 3886676 + 54046493 + 3968942521ull + 6156313 +
                     292631,
          "all ticks %llu", (unsigned long long)all);
    CHECK(busy == all - 3968942521ull - 6156313, "busy ticks");
    CHECK(sy_read_loadavg("0.11 0.36 0.47 1/1628 2650727\n", s.load) == 0 &&
              s.load[2] == 0.47,
          "loadavg");
    CHECK(sy_read_psi(S_PSI) == 0.01, "psi");
    CHECK(sy_read_psi("") == -1, "psi absent");
    CHECK(sy_read_os_release(S_OS, s.os, sizeof s.os) == 0 &&
              !strcmp(s.os, "Debian GNU/Linux 13 (trixie)"),
          "os-release: %s", s.os);
    CHECK(!strcmp(sy_sensor_kind("coretemp"), "cpu") &&
              !strcmp(sy_sensor_kind("nvme"), "disk") &&
              !strcmp(sy_sensor_kind("spd5118"), "memory") &&
              !strcmp(sy_sensor_kind("iwlwifi_1"), "wifi") &&
              !strcmp(sy_sensor_kind("acpitz"), "board"),
          "sensor kinds");

    /* a whole status, as the backend would give it */
    snprintf(s.host, sizeof s.host, "babay");
    snprintf(s.kernel, sizeof s.kernel, "6.12.107+deb13-amd64");
    snprintf(s.cpu, sizeof s.cpu, "Intel(R) Core(TM) Ultra 7 155H");
    s.cpus = 22;
    s.uptime_s = 22 * 86400 + 23 * 3600 + 28 * 60;
    s.has_load = 1;
    s.cpu_pct = 4.4;
    s.psi_cpu = 30, s.psi_mem = 0, s.psi_io = 0.5;
    CHECK(sy_read_mountinfo(S_MOUNTINFO, &s.disks) == 0, "mountinfo");
    s.disks.d[0].total = 900ull << 30, s.disks.d[0].avail = 50ull << 30;
    s.bat = (struct sy_battery){.present = 1,
                                .pct = 80,
                                .mains = 0,
                                .state = "discharging",
                                .minutes = 135,
                                .health = 75};
    s.t[0] = (struct sy_temp){.kind = "cpu", .c = 106, .high = 110};
    s.t[1] = (struct sy_temp){.kind = "disk", .c = 31.85, .high = 83.85};
    s.n_temps = 2;
    struct janas_buf d = {0};
    sy_status_data(&d, &s);
    const char *t = fill(SY_STATUS_LAYOUT, &d);
    CHECK(strstr(t, "babay: Debian GNU/Linux 13 (trixie), 6.12.107"), "%s", t);
    CHECK(strstr(t, "on for 22 d 23 h 28 min."), "uptime:\n%s", t);
    CHECK(strstr(t, "22 threads, 4% in use; load 0.11, 0.36, 0.47"),
          "processor:\n%s", t);
    CHECK(strstr(t, "Memory: 7.4 GB used of 30.8 GB (24%), 23.4 GB "
                    "available; swap: 1.6 GB of 31.5 GB."),
          "memory:\n%s", t);
    CHECK(strstr(t, "Waiting, of the last 10 seconds: for the processor 30%"),
          "pressure:\n%s", t);
    CHECK(strstr(t, "  /: 850 GB of 900 GB (94%), 50 GB free - nearly full\n"),
          "disk:\n%s", t);
    CHECK(strstr(t, "Battery: 80%, on battery, 2 h 15 min left; capacity: 75% "
                    "of what it was new."),
          "battery:\n%s", t);
    CHECK(strstr(t, "Temperatures:\n  processor: 106 °C, near its limit of "
                    "110 °C\n  disk: 32 °C\n"),
          "temperatures:\n%s", t);
    sy_status_data(&d, &s);
    t = fill(SY_STATUS_BRIEF, &d);
    CHECK(strstr(t, "/ 94% (50 GB free, NEARLY FULL)") &&
              strstr(t, "cpu 106 C NEAR ITS LIMIT") &&
              strstr(t, ", worn (75%)"),
          "status brief:\n%s", t);
}

static void test_disks(void)
{
    struct sy_disks d;
    sy_read_mountinfo(S_MOUNTINFO, &d);
    /* proc, tmpfs and the portal out; the second btrfs subvolume too */
    CHECK(d.n == 4, "disks: %d", d.n);
    CHECK(d.n == 4 && !strcmp(d.d[0].mount, "/") &&
              !strcmp(d.d[0].fs, "ext4") &&
              !strcmp(d.d[0].dev, "/dev/nvme0n1p5") && !d.d[0].ro,
          "the root");
    CHECK(d.n == 4 && !strcmp(d.d[2].mount, "/home"), "btrfs once");
    CHECK(d.n == 4 && !strcmp(d.d[3].mount, "/media/someone/My Disk") &&
              d.d[3].ro,
          "the space in a path, read only: %s", d.d[3].mount);
    for (int i = 0; i < d.n; i++)
        d.d[i].total = 100ull << 30, d.d[i].avail = 25ull << 30;
    struct janas_buf b = {0};
    sy_disks_data(&b, &d);
    const char *t = fill(SY_DISKS_LAYOUT, &b);
    CHECK(strstr(t, "  /media/someone/My Disk (/dev/sdb1, exfat, read only): "
                    "75 GB used of 100 GB (75%), 25 GB free\n"),
          "disks layout:\n%s", t);
}

static void test_procs(void)
{
    char name[64];
    uint64_t ticks, start;
    int threads;
    CHECK(sy_read_pid_stat(S_PID_STAT, name, sizeof name, &ticks, &threads,
                           &start) == 0,
          "pid stat");
    CHECK(!strcmp(name, "Isolated Web Co") && ticks == 4475 + 282 &&
              threads == 35 && start == 198506288,
          "pid stat: %s %llu %d %llu", name, (unsigned long long)ticks, threads,
          (unsigned long long)start);
    struct sy_procs p;
    memset(&p, 0, sizeof p);
    snprintf(p.by, sizeof p.by, "cpu");
    p.total = 412, p.matched = 412, p.n = 2;
    p.mem_total = 32ull << 30;
    p.p[0] = (struct sy_proc){.pid = 2011430,
                              .threads = 35,
                              .name = "Isolated Web Co",
                              .user = "someone",
                              .cpu_pct = 12.34,
                              .mem = 463ull << 20,
                              .age_s = 3 * 3600 + 5 * 60};
    p.p[1] = (struct sy_proc){.pid = 1, .name = "systemd", .user = "root"};
    struct janas_buf b = {0};
    sy_procs_data(&b, &p);
    const char *t = fill(SY_PROCS_LAYOUT, &b);
    CHECK(strstr(t, "The processes using the most processor, of 412:\n"
                    "  Isolated Web Co (pid 2011430, someone): processor "
                    "12.3%, memory 463 MB (1.4%), running for 3 h 5 min\n"),
          "processes:\n%s", t);
    p.n = p.matched = 0;
    snprintf(p.name, sizeof p.name, "firefox");
    sy_procs_data(&b, &p);
    t = fill(SY_PROCS_LAYOUT, &b);
    CHECK(strstr(t, "Processes with \"firefox\" in their name: 0 of 412:\n"
                    "  none.\n"),
          "none:\n%s", t);
}

static void test_errors(void)
{
    struct sy_errors e;
    memset(&e, 0, sizeof e);
    sy_read_journal(S_JOURNAL, sizeof S_JOURNAL - 1, &e);
    CHECK(e.total == 3 && e.n == 2, "journal: %d entries, %d kinds", e.total,
          e.n);
    sy_err_order(&e);
    CHECK(e.n == 2 && e.e[0].count == 2 &&
              !strcmp(e.e[0].who, "kscreenlocker_greet") &&
              e.e[0].at == 1790967655,
          "the same message counted, its latest time");
    sy_read_failed(S_FAILED, &e);
    CHECK(e.n_failed == 2 &&
              !strcmp(e.failed[0], "drkonqi-coredump-processor@.service") &&
              e.failed_n[0] == 2 && !strcmp(e.failed[1], "cups.service"),
          "failed: %d %s", e.n_failed, e.failed[0]);
    e.user_only = 1;
    struct janas_buf b = {0};
    sy_errors_data(&b, &e);
    const char *t = fill(SY_ERRORS_LAYOUT, &b);
    CHECK(strstr(t, "Errors since the computer started: 3.\nOnly those of "
                    "your own programs and session"),
          "errors head:\n%s", t);
    CHECK(strstr(t, " kscreenlocker_greet (2 times): PAM adding faulty module: "
                    "pam_sss.so\n"),
          "errors item:\n%s", t);
    CHECK(strstr(t, "Services that failed:\n  "
                    "drkonqi-coredump-processor@.service (2 times)\n  "
                    "cups.service\n"),
          "failed services:\n%s", t);

    /* a core dump: its first line, the same but for the numbers counted
       once */
    memset(&e, 0, sizeof e);
    sy_err_add(&e, 100, "systemd-coredump",
               "Process 2992988 (sb) of user 1000 dumped core.\n\nModule "
               "/home/x/sb without build-id.");
    sy_err_add(&e, 200, "systemd-coredump",
               "Process 2993006 (sb) of user 1000 dumped core.\n\nModule");
    sy_err_add(&e, 150, "systemd-coredump",
               "Process 2990885 (r11g) of user 1000 dumped core.");
    CHECK(e.n == 2 && e.e[0].count == 2 && e.e[0].at == 200 &&
              !strcmp(e.e[0].text,
                      "Process 2993006 (sb) of user 1000 dumped core."),
          "core dumps: %d kinds, %s", e.n, e.e[0].text);

    /* more kinds than kept: the oldest go */
    memset(&e, 0, sizeof e);
    for (int i = 0; i < SY_ERRS + 5; i++) {
        char text[32];
        snprintf(text, sizeof text, "error %c", 'A' + i);
        sy_err_add(&e, 1000 + i, "p", text);
    }
    sy_err_order(&e);
    CHECK(e.n == SY_ERRS && e.more == 5 && e.total == SY_ERRS + 5 &&
              e.e[0].at == 1000 + SY_ERRS + 4,
          "the latest kept: %d more %d", e.n, e.more);
}

static void test_updates(void)
{
    struct sy_updates u;
    memset(&u, 0, sizeof u);
    sy_read_apt(S_APT, &u);
    CHECK(u.total == 3 && u.security == 2 &&
              !strcmp(u.u[0].name, "bind9-dnsutils") &&
              !strcmp(u.u[0].to, "1:9.20.29-1~deb13u1") &&
              !strcmp(u.u[0].from, "1:9.20.26-1~deb13u1") && !u.u[2].security,
          "apt: %d %d %s", u.total, u.security, u.u[0].name);
    struct janas_buf b = {0};
    sy_updates_data(&b, &u);
    const char *t = fill(SY_UPDATES_LAYOUT, &b);
    CHECK(strstr(t, "Updates available: 3, 2 of them for security, by apt's "
                    "list.\n"),
          "updates head:\n%s", t);
    CHECK(strstr(t, "  tzdata 2026a-0+deb13u1 -> 2026b-0+deb13u1\n") &&
              strstr(t, "fetched again with: sudo apt update"),
          "updates:\n%s", t);
    memset(&u, 0, sizeof u);
    sy_read_dnf(S_DNF, &u);
    CHECK(u.total == 2 && !strcmp(u.u[0].name, "kernel") &&
              !strcmp(u.u[1].to, "2:9.1.785-1.fc41"),
          "dnf: %d %s", u.total, u.u[0].name);
    memset(&u, 0, sizeof u);
    sy_read_pacman(S_PACMAN, &u);
    CHECK(u.total == 2 && !strcmp(u.u[1].name, "firefox") &&
              !strcmp(u.u[1].from, "131.0.2-1") &&
              !strcmp(u.u[1].to, "131.0.3-1"),
          "pacman: %d", u.total);
}

static void test_net(void)
{
    int dbm = 0, q = 0;
    CHECK(sy_read_wireless(S_WIRELESS, "wlp0s20f3", &dbm, &q) == 0 &&
              dbm == -60 && q == 71,
          "wireless %d %d", dbm, q);
    CHECK(sy_read_wireless(S_WIRELESS, "wlan1", &dbm, &q) == -1,
          "no such interface");
    struct sy_net n;
    memset(&n, 0, sizeof n);
    CHECK(sy_read_route(S_ROUTE, n.gateway, sizeof n.gateway, n.gw_iface,
                        sizeof n.gw_iface) == 0 &&
              !strcmp(n.gateway, "192.168.1.1") &&
              !strcmp(n.gw_iface, "wlp0s20f3"),
          "route: %s %s", n.gateway, n.gw_iface);
    sy_read_resolv("# x\nnameserver 1.1.1.1\nnameserver 1.0.0.1\n", &n);
    CHECK(n.n_dns == 2 && !strcmp(n.dns[1], "1.0.0.1"), "resolv");
    n.n = 1;
    n.i[0] = (struct sy_iface){.name = "wlp0s20f3",
                               .kind = "wifi",
                               .state = "up",
                               .addr = {"192.168.1.20"},
                               .n_addr = 1,
                               .signal_dbm = -60,
                               .wifi_pct = 71,
                               .rx = 3ull << 30,
                               .tx = 200ull << 20};
    struct janas_buf b = {0};
    sy_net_data(&b, &n);
    const char *t = fill(SY_NET_LAYOUT, &b);
    CHECK(strstr(t, "wlp0s20f3 (Wi-Fi, connected), the way out to the "
                    "internet: 192.168.1.20\n  signal: -60 dBm (71%)\n  "
                    "received 3 GB, sent 200 MB\n"),
          "network:\n%s", t);
    CHECK(strstr(t, "Gateway: 192.168.1.1; name servers: 1.1.1.1 1.0.0.1."),
          "gateway:\n%s", t);
}

static void test_usage(void)
{
    struct sy_usage u;
    memset(&u, 0, sizeof u);
    snprintf(u.path, sizeof u.path, "/home/someone");
    u.total = 120ull << 30, u.fs_total = 900ull << 30, u.fs_avail = 50ull << 30;
    u.files = 123456, u.n = 2, u.more = 30, u.cut = 1, u.secs = 8.04;
    u.e[0] = (struct sy_entry){
        .name = ".cache", .bytes = 60ull << 30, .files = 100000, .dir = 1};
    u.e[1] =
        (struct sy_entry){.name = "film.mkv", .bytes = 4ull << 30, .files = 1};
    struct janas_buf b = {0};
    sy_usage_data(&b, &u);
    const char *t = fill(SY_USAGE_LAYOUT, &b);
    CHECK(strstr(t, "~: 120 GB in 123456 files, at least: the time ran out "
                    "after 8 s; the disk has 50 GB free of 900 GB.\n"
                    "  60 GB (50%)  .cache/, 100000 files\n"
                    "  4 GB (3%)  film.mkv\n  ... and 30 more entries.\n"),
          "usage:\n%s", t);
}

int main(void)
{
    test_status();
    test_disks();
    test_procs();
    test_errors();
    test_updates();
    test_net();
    test_usage();
    if (failures) {
        printf("test_system_parse: %d failures\n", failures);
        return 1;
    }
    printf("test_system_parse: ok\n");
    return 0;
}
