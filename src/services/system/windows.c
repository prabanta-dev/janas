/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * windows.c - janas-system's backend for Windows (see system.h), from the
 * Win32 API, no PowerShell and no WMI: the memory, the processor's times,
 * the drives, the power status, the processes (Tool Help), the event log
 * (System and Application, levels critical and error), the services set
 * to start by themselves and stopped with an error, the network adapters
 * (IP Helper) and the Wi-Fi signal (WLAN API); what fills a directory,
 * by the files' lengths (not the clusters they take, as on POSIX).
 * Not told on Windows: the temperatures (they need WMI and an
 * administrator), the pressure and the load, the updates (Windows
 * Update's own list).
 *
 * Linked with: iphlpapi ws2_32 wevtapi wlanapi advapi32 ole32.
 */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0A00
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <iphlpapi.h>
#include <netioapi.h>
#include <objbase.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <winevt.h>
#include <wlanapi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "system.h"

/* UTF-16 to UTF-8, cut to cap */
static void utf8(const wchar_t *w, char *out, size_t cap)
{
    out[0] = 0;
    if (w && !WideCharToMultiByte(CP_UTF8, 0, w, -1, out, (int)cap, NULL, NULL))
        out[cap - 1] = 0;
}

static uint64_t ft64(FILETIME f)
{
    return ((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime;
}

const char *sy_home(void)
{
    static char home[MAX_PATH * 3];
    if (!home[0]) {
        wchar_t w[MAX_PATH];
        DWORD n = GetEnvironmentVariableW(L"USERPROFILE", w, MAX_PATH);
        if (n && n < MAX_PATH)
            utf8(w, home, sizeof home);
    }
    return home;
}

static void reg_str(HKEY root, const char *key, const char *name, char *out,
                    size_t cap)
{
    DWORD n = (DWORD)cap;
    out[0] = 0;
    if (RegGetValueA(root, key, name, RRF_RT_REG_SZ, NULL, out, &n) !=
        ERROR_SUCCESS)
        out[0] = 0;
}

int sy_get_disks(struct sy_disks *d, char *err, size_t err_len)
{
    (void)err, (void)err_len;
    memset(d, 0, sizeof *d);
    DWORD mask = GetLogicalDrives();
    for (int i = 0; i < 26; i++) {
        if (!(mask & (1u << i)))
            continue;
        char root[4] = {(char)('A' + i), ':', '\\', 0};
        UINT type = GetDriveTypeA(root);
        if (type != DRIVE_FIXED && type != DRIVE_REMOVABLE &&
            type != DRIVE_REMOTE)
            continue;
        ULARGE_INTEGER avail, total, free_all;
        if (!GetDiskFreeSpaceExA(root, &avail, &total, &free_all))
            continue; /* a card reader without a card */
        if (d->n == SY_DISKS) {
            d->more++;
            continue;
        }
        struct sy_disk *k = &d->d[d->n++];
        sy_put(k->mount, sizeof k->mount, root);
        char label[MAX_PATH + 1] = "", fs[MAX_PATH + 1] = "";
        DWORD flags = 0;
        GetVolumeInformationA(root, label, sizeof label, NULL, NULL, &flags, fs,
                              sizeof fs);
        sy_put(k->dev, sizeof k->dev, label);
        sy_put(k->fs, sizeof k->fs, fs);
        k->ro = (flags & FILE_READ_ONLY_VOLUME) != 0;
        k->total = total.QuadPart;
        k->avail = avail.QuadPart;
    }
    return 0;
}

static void battery(struct sy_battery *b)
{
    memset(b, 0, sizeof *b);
    b->pct = b->minutes = b->health = -1;
    b->mains = -1;
    sy_put(b->state, sizeof b->state, "unknown");
    SYSTEM_POWER_STATUS p;
    if (!GetSystemPowerStatus(&p))
        return;
    b->mains = p.ACLineStatus == 1 ? 1 : p.ACLineStatus == 0 ? 0 : -1;
    if (p.BatteryFlag == 255 || (p.BatteryFlag & 128))
        return; /* not known, or no battery */
    b->present = 1;
    b->pct = p.BatteryLifePercent <= 100 ? p.BatteryLifePercent : -1;
    sy_put(b->state, sizeof b->state,
           (p.BatteryFlag & 8) ? "charging"
           : b->mains == 0     ? "discharging"
           : b->pct == 100     ? "full"
                               : "idle");
    if (b->mains == 0 && p.BatteryLifeTime != (DWORD)-1)
        b->minutes = (int)(p.BatteryLifeTime / 60);
}

static void os_name(struct sy_status *s)
{
    static const char key[] = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
    char product[96], display[32], build[16];
    reg_str(HKEY_LOCAL_MACHINE, key, "ProductName", product, sizeof product);
    reg_str(HKEY_LOCAL_MACHINE, key, "DisplayVersion", display, sizeof display);
    reg_str(HKEY_LOCAL_MACHINE, key, "CurrentBuildNumber", build, sizeof build);
    /* Windows 11 still calls itself Windows 10 there: its build tells */
    if (atoi(build) >= 22000 && !strncmp(product, "Windows 10", 10))
        product[8] = '1', product[9] = '1';
    snprintf(s->os, sizeof s->os, "%s%s%s", product, display[0] ? " " : "",
             display);
    snprintf(s->kernel, sizeof s->kernel, "build %s", build);
}

int sy_get_status(struct sy_status *s, char *err, size_t err_len)
{
    memset(s, 0, sizeof *s);
    sy_put(s->platform, sizeof s->platform, "windows");
    FILETIME i0, k0, u0, i1, k1, u1;
    int times = GetSystemTimes(&i0, &k0, &u0) != 0;
    MEMORYSTATUSEX m = {.dwLength = sizeof m};
    if (!GlobalMemoryStatusEx(&m)) {
        snprintf(err, err_len, "the memory could not be read");
        return -1;
    }
    s->mem_total = m.ullTotalPhys;
    s->mem_avail = m.ullAvailPhys;
    /* the page file: what the commit limit holds beyond the memory */
    if (m.ullTotalPageFile > m.ullTotalPhys) {
        s->swap_total = m.ullTotalPageFile - m.ullTotalPhys;
        uint64_t used = (m.ullTotalPageFile - m.ullAvailPageFile) >
                                (m.ullTotalPhys - m.ullAvailPhys)
                            ? (m.ullTotalPageFile - m.ullAvailPageFile) -
                                  (m.ullTotalPhys - m.ullAvailPhys)
                            : 0;
        s->swap_free = used < s->swap_total ? s->swap_total - used : 0;
    }
    DWORD n = sizeof s->host;
    GetComputerNameExA(ComputerNameDnsHostname, s->host, &n);
    os_name(s);
    reg_str(HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            "ProcessorNameString", s->cpu, sizeof s->cpu);
    s->cpus = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    s->uptime_s = (long)(GetTickCount64() / 1000);
    s->psi_cpu = s->psi_mem = s->psi_io = -1;
    char e2[100];
    sy_get_disks(&s->disks, e2, sizeof e2);
    battery(&s->bat);
    Sleep(SY_SAMPLE_MS);
    s->cpu_pct = -1;
    if (times && GetSystemTimes(&i1, &k1, &u1)) {
        /* the kernel's time holds the idle time */
        uint64_t idle = ft64(i1) - ft64(i0);
        uint64_t all = (ft64(k1) - ft64(k0)) + (ft64(u1) - ft64(u0));
        if (all > 0) {
            s->cpu_pct = (double)(all - idle) * 100 / (double)all;
            s->sys_pct =
                (double)(ft64(k1) - ft64(k0) - idle) * 100 / (double)all;
        }
    }
    return 0;
}

/* ---- the processes ---- */

struct ptime {
    DWORD pid;
    uint64_t t;
};

static int ci_has(const char *s, const char *what)
{
    size_t n = strlen(what);
    for (; *s; s++)
        if (!_strnicmp(s, what, n))
            return 1;
    return 0;
}

static uint64_t proc_time(HANDLE h, uint64_t *created)
{
    FILETIME c, e, k, u;
    if (!GetProcessTimes(h, &c, &e, &k, &u))
        return 0;
    if (created)
        *created = ft64(c);
    return ft64(k) + ft64(u);
}

static void proc_user(DWORD pid, char *out, size_t cap)
{
    out[0] = 0;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid), tok;
    if (!h)
        return;
    if (OpenProcessToken(h, TOKEN_QUERY, &tok)) {
        char buf[256];
        DWORD n;
        if (GetTokenInformation(tok, TokenUser, buf, sizeof buf, &n)) {
            wchar_t name[128], dom[128];
            DWORD nn = 128, dn = 128;
            SID_NAME_USE use;
            if (LookupAccountSidW(NULL, ((TOKEN_USER *)buf)->User.Sid, name,
                                  &nn, dom, &dn, &use))
                utf8(name, out, cap);
        }
        CloseHandle(tok);
    }
    CloseHandle(h);
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
    MEMORYSTATUSEX m = {.dwLength = sizeof m};
    if (GlobalMemoryStatusEx(&m))
        p->mem_total = m.ullTotalPhys;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        snprintf(err, err_len, "the processes could not be listed");
        return -1;
    }
    int cap = 512, n = 0;
    struct sy_proc *v = malloc((size_t)cap * sizeof *v);
    struct ptime *t0 = malloc((size_t)cap * sizeof *t0);
    PROCESSENTRY32W e = {.dwSize = sizeof e};
    for (BOOL ok = Process32FirstW(snap, &e); ok && v && t0;
         ok = Process32NextW(snap, &e)) {
        if (e.th32ProcessID == 0) /* the idle process */
            continue;
        p->total++;
        char pname[64];
        utf8(e.szExeFile, pname, sizeof pname);
        char *dot = strrchr(pname, '.');
        if (dot && !_stricmp(dot, ".exe"))
            *dot = 0;
        if (name && *name && !ci_has(pname, name))
            continue;
        p->matched++;
        if (n == cap) {
            cap *= 2;
            struct sy_proc *w = realloc(v, (size_t)cap * sizeof *v);
            struct ptime *u = w ? realloc(t0, (size_t)cap * sizeof *t0) : NULL;
            if (!w || !u) {
                if (w)
                    v = w;
                break;
            }
            v = w, t0 = u;
        }
        struct sy_proc *x = &v[n];
        memset(x, 0, sizeof *x);
        x->pid = (int)e.th32ProcessID;
        x->threads = (int)e.cntThreads;
        sy_put(x->name, sizeof x->name, pname);
        t0[n].pid = e.th32ProcessID;
        t0[n].t = 0;
        HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                               e.th32ProcessID);
        if (h) {
            uint64_t created = 0;
            t0[n].t = proc_time(h, &created);
            FILETIME now;
            GetSystemTimeAsFileTime(&now);
            if (created)
                x->age_s = (long)((ft64(now) - created) / 10000000);
            PROCESS_MEMORY_COUNTERS pm = {.cb = sizeof pm};
            if (GetProcessMemoryInfo(h, &pm, sizeof pm))
                x->mem = pm.WorkingSetSize;
            CloseHandle(h);
        }
        n++;
    }
    CloseHandle(snap);
    if (!v || !t0) {
        free(v);
        free(t0);
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    /* the processor's share: their times again after a while */
    ULONGLONG w0 = GetTickCount64();
    Sleep(SY_SAMPLE_MS);
    double wall = (double)(GetTickCount64() - w0) * 10000; /* 100 ns */
    int cpus = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    for (int i = 0; i < n; i++) {
        HANDLE h =
            OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, t0[i].pid);
        if (!h)
            continue;
        uint64_t t = proc_time(h, NULL);
        CloseHandle(h);
        if (t >= t0[i].t && t0[i].t && wall > 0 && cpus > 0)
            v[i].cpu_pct = (double)(t - t0[i].t) * 100 / (wall * cpus);
    }
    free(t0);
    qsort(v, (size_t)n, sizeof *v, strcmp(by, "memory") ? more_cpu : more_mem);
    p->n = n < SY_PROCS ? n : SY_PROCS;
    memcpy(p->p, v, (size_t)p->n * sizeof *v);
    free(v);
    for (int i = 0; i < p->n; i++) /* who: only for those shown, it costs */
        proc_user((DWORD)p->p[i].pid, p->p[i].user, sizeof p->p[i].user);
    return 0;
}

/* ---- the errors ---- */

/* an event's text in its publisher's words, its source and its time */
static void one_event(EVT_HANDLE ev, EVT_HANDLE ctx, struct sy_errors *e)
{
    DWORD used = 0, props = 0;
    char buf[2048];
    if (!EvtRender(ctx, ev, EvtRenderEventValues, sizeof buf, buf, &used,
                   &props))
        return;
    EVT_VARIANT *v = (EVT_VARIANT *)buf;
    char who[64] = "";
    if (v[0].Type == EvtVarTypeString)
        utf8(v[0].StringVal, who, sizeof who);
    long long at = 0;
    if (v[1].Type == EvtVarTypeFileTime) /* 100 ns since 1601 */
        at = (long long)(v[1].FileTimeVal / 10000000) - 11644473600LL;
    wchar_t text[1024] = L"";
    EVT_HANDLE pub =
        v[0].Type == EvtVarTypeString
            ? EvtOpenPublisherMetadata(NULL, v[0].StringVal, NULL, 0, 0)
            : NULL;
    DWORD n = 0;
    if (!pub || !EvtFormatMessage(pub, ev, 0, 0, NULL, EvtFormatMessageEvent,
                                  sizeof text / sizeof *text, text, &n))
        wcscpy(text, L"(no text: its publisher's message is not installed)");
    if (pub)
        EvtClose(pub);
    char t8[1024];
    utf8(text, t8, sizeof t8);
    sy_err_add(e, at, who, t8);
}

static void channel(const wchar_t *path, const wchar_t *query,
                    struct sy_errors *e)
{
    EVT_HANDLE q = EvtQuery(NULL, path, query,
                            EvtQueryChannelPath | EvtQueryReverseDirection);
    if (!q)
        return;
    LPCWSTR fields[] = {L"Event/System/Provider/@Name",
                        L"Event/System/TimeCreated/@SystemTime"};
    EVT_HANDLE ctx = EvtCreateRenderContext(2, fields, EvtRenderContextValues);
    EVT_HANDLE ev[16];
    DWORD got = 0;
    int seen = 0;
    while (ctx && seen < 2000 && EvtNext(q, 16, ev, 2000, 0, &got)) {
        for (DWORD i = 0; i < got; i++) {
            one_event(ev[i], ctx, e);
            EvtClose(ev[i]);
        }
        seen += (int)got;
    }
    if (seen >= 2000) /* the latest first: those left are older */
        e->cut = 1;
    if (ctx)
        EvtClose(ctx);
    EvtClose(q);
}

/* the services set to start by themselves, stopped with an error */
static void failed_services(struct sy_errors *e)
{
    SC_HANDLE sc = OpenSCManagerW(NULL, NULL, SC_MANAGER_ENUMERATE_SERVICE);
    if (!sc)
        return;
    DWORD need = 0, count = 0, resume = 0;
    EnumServicesStatusExW(sc, SC_ENUM_PROCESS_INFO, SERVICE_WIN32,
                          SERVICE_INACTIVE, NULL, 0, &need, &count, &resume,
                          NULL);
    BYTE *buf = need ? malloc(need) : NULL;
    resume = 0;
    if (buf && EnumServicesStatusExW(sc, SC_ENUM_PROCESS_INFO, SERVICE_WIN32,
                                     SERVICE_INACTIVE, buf, need, &need, &count,
                                     &resume, NULL)) {
        ENUM_SERVICE_STATUS_PROCESSW *s = (ENUM_SERVICE_STATUS_PROCESSW *)buf;
        for (DWORD i = 0; i < count; i++) {
            DWORD code = s[i].ServiceStatusProcess.dwWin32ExitCode;
            if (code == 0 || code == ERROR_SERVICE_NEVER_STARTED)
                continue;
            SC_HANDLE h =
                OpenServiceW(sc, s[i].lpServiceName, SERVICE_QUERY_CONFIG);
            if (!h)
                continue;
            DWORD cn = 0;
            QueryServiceConfigW(h, NULL, 0, &cn);
            QUERY_SERVICE_CONFIGW *cfg = cn ? malloc(cn) : NULL;
            int automatic = cfg && QueryServiceConfigW(h, cfg, cn, &cn) &&
                            cfg->dwStartType == SERVICE_AUTO_START;
            free(cfg);
            CloseServiceHandle(h);
            if (!automatic)
                continue;
            if (e->n_failed == SY_FAILED) {
                e->more_failed++;
                continue;
            }
            utf8(s[i].lpDisplayName, e->failed[e->n_failed],
                 sizeof e->failed[0]);
            e->failed_n[e->n_failed++] = 1;
        }
    }
    free(buf);
    CloseServiceHandle(sc);
}

int sy_get_errors(int hours, struct sy_errors *e, char *err, size_t err_len)
{
    (void)err, (void)err_len;
    memset(e, 0, sizeof *e);
    e->hours = hours;
    sy_put(e->how, sizeof e->how, "eventlog");
    unsigned long long ms =
        hours > 0 ? (unsigned long long)hours * 3600000ULL : GetTickCount64();
    wchar_t query[200];
    swprintf(query, sizeof query / sizeof *query,
             L"*[System[(Level=1 or Level=2) and "
             L"TimeCreated[timediff(@SystemTime) <= %llu]]]",
             ms);
    channel(L"System", query, e);
    channel(L"Application", query, e);
    sy_err_order(e);
    failed_services(e);
    return 0;
}

int sy_get_updates(struct sy_updates *u, char *err, size_t err_len)
{
    memset(u, 0, sizeof *u);
    snprintf(err, err_len,
             "on Windows the updates are Windows Update's, not read here: "
             "they are in Settings, Windows Update (and winget upgrade for "
             "the programs installed with winget)");
    return -1;
}

/* ---- the network ---- */

static void wifi_signal(struct sy_net *n, const char *const *guids)
{
    HANDLE h;
    DWORD ver;
    if (WlanOpenHandle(2, NULL, &ver, &h) != ERROR_SUCCESS)
        return;
    WLAN_INTERFACE_INFO_LIST *list = NULL;
    if (WlanEnumInterfaces(h, NULL, &list) == ERROR_SUCCESS) {
        for (DWORD i = 0; i < list->dwNumberOfItems; i++) {
            WLAN_INTERFACE_INFO *w = &list->InterfaceInfo[i];
            wchar_t g[64];
            char g8[64];
            StringFromGUID2(&w->InterfaceGuid, g, 64);
            utf8(g, g8, sizeof g8);
            int k = 0;
            while (k < n->n && _stricmp(guids[k], g8))
                k++;
            if (k == n->n)
                continue;
            DWORD sz = 0;
            WLAN_CONNECTION_ATTRIBUTES *c = NULL;
            if (WlanQueryInterface(h, &w->InterfaceGuid,
                                   wlan_intf_opcode_current_connection, NULL,
                                   &sz, (void **)&c, NULL) == ERROR_SUCCESS) {
                n->i[k].wifi_pct =
                    (int)c->wlanAssociationAttributes.wlanSignalQuality;
                WlanFreeMemory(c);
            }
            LONG *rssi = NULL;
            if (WlanQueryInterface(h, &w->InterfaceGuid, wlan_intf_opcode_rssi,
                                   NULL, &sz, (void **)&rssi,
                                   NULL) == ERROR_SUCCESS) {
                n->i[k].signal_dbm = (int)*rssi;
                WlanFreeMemory(rssi);
            }
        }
        WlanFreeMemory(list);
    }
    WlanCloseHandle(h, NULL);
}

static void addr_str(const SOCKET_ADDRESS *a, char *out, size_t cap)
{
    DWORD n = (DWORD)cap;
    wchar_t w[64];
    DWORD wn = 64;
    out[0] = 0;
    if (WSAAddressToStringW(a->lpSockaddr, (DWORD)a->iSockaddrLength, NULL, w,
                            &wn) == 0)
        utf8(w, out, n);
}

int sy_get_net(struct sy_net *n, char *err, size_t err_len)
{
    memset(n, 0, sizeof *n);
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    ULONG size = 32768;
    IP_ADAPTER_ADDRESSES *list = NULL;
    ULONG rc = ERROR_BUFFER_OVERFLOW;
    for (int tries = 0; rc == ERROR_BUFFER_OVERFLOW && tries < 3; tries++) {
        free(list);
        list = malloc(size);
        if (!list)
            break;
        rc = GetAdaptersAddresses(AF_UNSPEC,
                                  GAA_FLAG_INCLUDE_GATEWAYS |
                                      GAA_FLAG_SKIP_MULTICAST |
                                      GAA_FLAG_SKIP_ANYCAST,
                                  NULL, list, &size);
    }
    if (!list || rc != NO_ERROR) {
        free(list);
        snprintf(err, err_len, "the network adapters could not be read");
        return -1;
    }
    static char guid[SY_IFACES][64];
    const char *guids[SY_IFACES];
    for (IP_ADAPTER_ADDRESSES *a = list; a; a = a->Next) {
        if (a->IfType == IF_TYPE_SOFTWARE_LOOPBACK)
            continue;
        if (n->n == SY_IFACES) {
            n->more++;
            continue;
        }
        struct sy_iface *x = &n->i[n->n];
        utf8(a->FriendlyName, x->name, sizeof x->name);
        sy_put(guid[n->n], sizeof guid[0], a->AdapterName);
        guids[n->n] = guid[n->n];
        n->n++;
        sy_put(x->kind, sizeof x->kind,
               a->IfType == IF_TYPE_IEEE80211         ? "wifi"
               : a->IfType == IF_TYPE_ETHERNET_CSMACD ? "ethernet"
                                                      : "virtual");
        sy_put(x->state, sizeof x->state,
               a->OperStatus == IfOperStatusUp               ? "up"
               : a->OperStatus == IfOperStatusDormant        ? "dormant"
               : a->OperStatus == IfOperStatusLowerLayerDown ? "lowerlayerdown"
               : a->OperStatus == IfOperStatusDown ||
                       a->OperStatus == IfOperStatusNotPresent
                   ? "down"
                   : "unknown");
        if (a->TransmitLinkSpeed && a->TransmitLinkSpeed != (ULONG64)-1)
            x->speed_mbps = (long)(a->TransmitLinkSpeed / 1000000);
        for (IP_ADAPTER_UNICAST_ADDRESS *u = a->FirstUnicastAddress;
             u && x->n_addr < SY_ADDRS; u = u->Next)
            addr_str(&u->Address, x->addr[x->n_addr++], sizeof x->addr[0]);
        MIB_IF_ROW2 row = {.InterfaceLuid = a->Luid};
        if (GetIfEntry2(&row) == NO_ERROR) {
            x->rx = row.InOctets;
            x->tx = row.OutOctets;
        }
        if (!n->gateway[0] && a->FirstGatewayAddress &&
            a->OperStatus == IfOperStatusUp) {
            addr_str(&a->FirstGatewayAddress->Address, n->gateway,
                     sizeof n->gateway);
            sy_put(n->gw_iface, sizeof n->gw_iface, x->name);
        }
        for (IP_ADAPTER_DNS_SERVER_ADDRESS *d = a->FirstDnsServerAddress;
             d && n->n_dns < 3 && a->OperStatus == IfOperStatusUp;
             d = d->Next) {
            char s[64];
            addr_str(&d->Address, s, sizeof s);
            int seen = 0;
            for (int k = 0; k < n->n_dns; k++)
                seen |= !strcmp(n->dns[k], s);
            if (!seen && s[0])
                sy_put(n->dns[n->n_dns++], sizeof n->dns[0], s);
        }
    }
    free(list);
    /* the virtual ones without addresses are noise */
    int k = 0;
    for (int i = 0; i < n->n; i++)
        if (strcmp(n->i[i].kind, "virtual") || n->i[i].n_addr) {
            guids[k] = guids[i];
            n->i[k++] = n->i[i];
        }
    n->n = k;
    wifi_signal(n, guids);
    WSACleanup();
    return 0;
}

/* ---- what fills a directory ---- */

struct walk {
    ULONGLONG end;
    int cut;
    long unread;
};

/* the bytes under dir (UTF-16, without the final \); reparse points (links,
   junctions, mounted volumes) are not followed */
static uint64_t sum(struct walk *w, const wchar_t *dir, int depth, long *files)
{
    wchar_t pat[1100];
    if (_snwprintf(pat, 1100, L"%ls\\*", dir) < 0)
        return 0;
    WIN32_FIND_DATAW f;
    HANDLE h = FindFirstFileExW(pat, FindExInfoBasic, &f, FindExSearchNameMatch,
                                NULL, FIND_FIRST_EX_LARGE_FETCH);
    if (h == INVALID_HANDLE_VALUE) {
        w->unread++;
        return 0;
    }
    uint64_t total = 0;
    unsigned tick = 0;
    do {
        if ((++tick & 255) == 0 && GetTickCount64() > w->end)
            w->cut = 1;
        if (w->cut)
            break;
        if (!wcscmp(f.cFileName, L".") || !wcscmp(f.cFileName, L".."))
            continue;
        if (f.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)
            continue;
        if (!(f.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            total += ((uint64_t)f.nFileSizeHigh << 32) | f.nFileSizeLow;
            (*files)++;
            continue;
        }
        if (depth >= SY_USAGE_DEPTH)
            continue;
        wchar_t sub[1100];
        if (_snwprintf(sub, 1100, L"%ls\\%ls", dir, f.cFileName) > 0)
            total += sum(w, sub, depth + 1, files);
    } while (FindNextFileW(h, &f));
    FindClose(h);
    return total;
}

static int bigger(const void *a, const void *b)
{
    uint64_t x = ((const struct sy_entry *)a)->bytes,
             y = ((const struct sy_entry *)b)->bytes;
    return x < y ? 1 : x > y ? -1 : 0;
}

int sy_get_usage(const char *path, struct sy_usage *u, char *err,
                 size_t err_len)
{
    memset(u, 0, sizeof *u);
    ULONGLONG t0 = GetTickCount64();
    const char *home = sy_home();
    if (!path || !*path)
        path = home;
    if (path[0] == '~' && (path[1] == '/' || path[1] == '\\' || !path[1]))
        snprintf(u->path, sizeof u->path, "%s%s", home, path + 1);
    else
        sy_put(u->path, sizeof u->path, path);
    size_t len = strlen(u->path);
    while (len > 3 && (u->path[len - 1] == '\\' || u->path[len - 1] == '/'))
        u->path[--len] = 0;
    wchar_t dir[1100];
    if (!MultiByteToWideChar(CP_UTF8, 0, u->path, -1, dir, 1100)) {
        snprintf(err, err_len, "%s is not a path", u->path);
        return -1;
    }
    size_t dl = wcslen(dir);
    if (dl && dir[dl - 1] == L'\\') /* "C:\" */
        dir[dl - 1] = 0;
    ULARGE_INTEGER avail, total, free_all;
    wchar_t probe[1104];
    _snwprintf(probe, 1104, L"%ls\\", dir);
    if (GetDiskFreeSpaceExW(probe, &avail, &total, &free_all)) {
        u->fs_total = total.QuadPart;
        u->fs_avail = avail.QuadPart;
    }
    wchar_t pat[1104];
    _snwprintf(pat, 1104, L"%ls\\*", dir);
    WIN32_FIND_DATAW f;
    HANDLE h = FindFirstFileExW(pat, FindExInfoBasic, &f, FindExSearchNameMatch,
                                NULL, FIND_FIRST_EX_LARGE_FETCH);
    if (h == INVALID_HANDLE_VALUE) {
        snprintf(err, err_len, "%s cannot be read", u->path);
        return -1;
    }
    struct walk w = {.end = t0 + SY_USAGE_MS};
    int cap = 256, n = 0;
    struct sy_entry *all = malloc((size_t)cap * sizeof *all);
    do {
        if (!all)
            break;
        if (!wcscmp(f.cFileName, L".") || !wcscmp(f.cFileName, L".."))
            continue;
        if (n == cap) {
            struct sy_entry *m = realloc(all, (size_t)(cap *= 2) * sizeof *m);
            if (!m)
                break;
            all = m;
        }
        struct sy_entry *x = &all[n++];
        memset(x, 0, sizeof *x);
        utf8(f.cFileName, x->name, sizeof x->name);
        x->dir = (f.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        if (!x->dir) {
            x->bytes = ((uint64_t)f.nFileSizeHigh << 32) | f.nFileSizeLow;
            x->files = 1;
            continue;
        }
        if ((f.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) || w.cut)
            continue;
        wchar_t sub[1100];
        if (_snwprintf(sub, 1100, L"%ls\\%ls", dir, f.cFileName) > 0)
            x->bytes = sum(&w, sub, 1, &x->files);
    } while (FindNextFileW(h, &f));
    FindClose(h);
    if (!all) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    qsort(all, (size_t)n, sizeof *all, bigger);
    for (int i = 0; i < n; i++) {
        u->total += all[i].bytes;
        u->files += all[i].files;
    }
    u->n = n < SY_USAGE ? n : SY_USAGE;
    u->more = n - u->n;
    memcpy(u->e, all, (size_t)u->n * sizeof *all);
    free(all);
    u->cut = w.cut;
    u->unread = w.unread;
    u->secs = (double)(GetTickCount64() - t0) / 1e3;
    return 0;
}
#endif
