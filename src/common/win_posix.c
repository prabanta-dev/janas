/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * win_posix.c - the functions behind the headers of src/common/win/, which
 * stand in for POSIX's on Windows: statvfs, getrusage, uname, getrandom,
 * dlopen, setpriority, fnmatch, fsync, and poll over sockets, pipes and
 * the console. Nothing on other
 * systems.
 */
#ifdef _WIN32
#include "win/dlfcn.h"
#include "win/fnmatch.h"
#include "win/sys/random.h"
#include "win/sys/resource.h"
#include "win/sys/statvfs.h"
#include "win/sys/utsname.h"
#include "win/win_sockets.h"

#include <ctype.h>
#include <io.h>
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>

#undef poll

int setpriority(int which, int who, int nice)
{
    (void)which, (void)who;
    DWORD c = nice > 10  ? IDLE_PRIORITY_CLASS
              : nice > 0 ? BELOW_NORMAL_PRIORITY_CLASS
                         : NORMAL_PRIORITY_CLASS;
    return SetPriorityClass(GetCurrentProcess(), c) ? 0 : -1;
}

int janas_fsync(int fd)
{
    return _commit(fd);
}

static int fold(int c, int flags)
{
    return flags & FNM_CASEFOLD ? tolower((unsigned char)c) : c;
}

int fnmatch(const char *p, const char *s, int flags)
{
    for (;; p++, s++) {
        switch (*p) {
        case 0:
            return *s ? FNM_NOMATCH : 0;
        case '?':
            if (!*s)
                return FNM_NOMATCH;
            break;
        case '*':
            while (*p == '*')
                p++;
            if (!*p)
                return 0;
            for (; *s; s++)
                if (fnmatch(p, s, flags) == 0)
                    return 0;
            return FNM_NOMATCH;
        case '[': {
            if (!*s)
                return FNM_NOMATCH;
            const char *q = p + 1;
            int neg = *q == '!' || *q == '^', hit = 0;
            if (neg)
                q++;
            /* a ']' first is one of the set */
            for (int first = 1; *q && (*q != ']' || first); q++, first = 0) {
                if (q[1] == '-' && q[2] && q[2] != ']') {
                    hit |= fold(*s, flags) >= fold(*q, flags) &&
                           fold(*s, flags) <= fold(q[2], flags);
                    q += 2;
                } else
                    hit |= fold(*q, flags) == fold(*s, flags);
            }
            if (!*q) { /* no closing ']': the '[' as itself */
                if (*s != '[')
                    return FNM_NOMATCH;
                break;
            }
            if (hit == neg)
                return FNM_NOMATCH;
            p = q;
            break;
        }
        case '\\':
            if (!(flags & FNM_NOESCAPE) && p[1])
                p++;
            /* fall through */
        default:
            if (fold(*p, flags) != fold(*s, flags))
                return FNM_NOMATCH;
        }
    }
}

static int volume_room(const char *path, struct statvfs *s)
{
    ULARGE_INTEGER avail, total, free_;
    if (!GetDiskFreeSpaceExA(path, &avail, &total, &free_)) {
        errno = ENOENT;
        return -1;
    }
    /* in bytes: a block of one */
    *s = (struct statvfs){.f_bsize = 1,
                          .f_frsize = 1,
                          .f_blocks = total.QuadPart,
                          .f_bfree = free_.QuadPart,
                          .f_bavail = avail.QuadPart};
    return 0;
}

int statvfs(const char *path, struct statvfs *s)
{
    return volume_room(path, s);
}

int fstatvfs(int fd, struct statvfs *s)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    char p[MAX_PATH + 8];
    DWORD n =
        h == INVALID_HANDLE_VALUE
            ? 0
            : GetFinalPathNameByHandleA(h, p, sizeof p, FILE_NAME_NORMALIZED);
    if (n == 0 || n >= sizeof p) {
        errno = EBADF;
        return -1;
    }
    /* \\?\C:\dir\file: the volume is the path's root */
    char root[MAX_PATH + 8];
    if (!GetVolumePathNameA(p, root, sizeof root)) {
        errno = EBADF;
        return -1;
    }
    return volume_room(root, s);
}

static struct timeval filetime_tv(FILETIME f)
{
    uint64_t t = ((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime;
    t /= 10; /* 100 ns to us */
    return (struct timeval){.tv_sec = (long)(t / 1000000),
                            .tv_usec = (long)(t % 1000000)};
}

int getrusage(int who, struct rusage *u)
{
    (void)who;
    FILETIME c, e, k, us;
    PROCESS_MEMORY_COUNTERS m = {.cb = sizeof m};
    memset(u, 0, sizeof *u);
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &us)) {
        u->ru_utime = filetime_tv(us);
        u->ru_stime = filetime_tv(k);
    }
    if (GetProcessMemoryInfo(GetCurrentProcess(), &m, sizeof m))
        u->ru_maxrss = (long)(m.PeakWorkingSetSize / 1024);
    return 0;
}

int uname(struct utsname *u)
{
    memset(u, 0, sizeof *u);
    snprintf(u->sysname, sizeof u->sysname, "Windows");
    DWORD n = sizeof u->nodename;
    if (!GetComputerNameA(u->nodename, &n))
        snprintf(u->nodename, sizeof u->nodename, "?");
    /* the true version: GetVersionEx answers what the manifest says */
    RTL_OSVERSIONINFOW v = {.dwOSVersionInfoSize = sizeof v};
    typedef LONG(WINAPI * get_version)(RTL_OSVERSIONINFOW *);
    HMODULE nt = GetModuleHandleA("ntdll.dll");
    get_version gv =
        nt ? (get_version)(void (*)(void))GetProcAddress(nt, "RtlGetVersion")
           : NULL;
    if (gv && gv(&v) == 0) {
        snprintf(u->release, sizeof u->release, "%lu.%lu.%lu",
                 (unsigned long)v.dwMajorVersion,
                 (unsigned long)v.dwMinorVersion,
                 (unsigned long)v.dwBuildNumber);
        snprintf(u->version, sizeof u->version, "build %lu",
                 (unsigned long)v.dwBuildNumber);
    }
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);
    snprintf(u->machine, sizeof u->machine, "%s",
             si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64
                 ? "x86_64"
             : si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64
                 ? "aarch64"
                 : "unknown");
    return 0;
}

ssize_t getrandom(void *buf, size_t n, unsigned flags)
{
    (void)flags;
    if (n > 0xffffffffu)
        n = 0xffffffffu;
    if (BCryptGenRandom(NULL, buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG) !=
        0) {
        errno = EIO;
        return -1;
    }
    return (ssize_t)n;
}

static char dl_error[160];
static int dl_failed;

static void dl_fail(const char *what)
{
    snprintf(dl_error, sizeof dl_error, "%s: error %lu", what,
             (unsigned long)GetLastError());
    dl_failed = 1;
}

void *dlopen(const char *file, int mode)
{
    (void)mode;
    HMODULE h = LoadLibraryA(file);
    if (!h)
        dl_fail(file);
    return (void *)h;
}

void *dlsym(void *handle, const char *name)
{
    FARPROC f = GetProcAddress((HMODULE)handle, name);
    if (!f)
        dl_fail(name);
    return (void *)(intptr_t)f;
}

int dlclose(void *handle)
{
    return FreeLibrary((HMODULE)handle) ? 0 : -1;
}

char *dlerror(void)
{
    if (!dl_failed)
        return NULL;
    dl_failed = 0;
    return dl_error;
}

/* the events WSAPoll accepts as asked for (it refuses the others) */
#define ASKABLE (POLLRDNORM | POLLRDBAND | POLLWRNORM)

static int is_socket(int fd)
{
    int type;
    int len = sizeof type;
    return getsockopt((SOCKET)(intptr_t)fd, SOL_SOCKET, SO_TYPE, (char *)&type,
                      &len) == 0;
}

/* Whether the console has a key typed: what is not a key (the mouse, focus,
   a resize, a key let go) is taken out of the way, so that a read that
   follows does not wait. */
static int console_has_key(HANDLE h)
{
    for (;;) {
        INPUT_RECORD r;
        DWORD n = 0;
        if (!PeekConsoleInputW(h, &r, 1, &n) || n == 0)
            return 0;
        if (r.EventType == KEY_EVENT && r.Event.KeyEvent.bKeyDown)
            return 1;
        ReadConsoleInputW(h, &r, 1, &n);
    }
}

static short fd_ready(int fd, short events)
{
    if (fd < 0)
        return 0;
    if (is_socket(fd)) {
        WSAPOLLFD p = {.fd = (SOCKET)(intptr_t)fd,
                       .events = (short)(events & ASKABLE)};
        if (WSAPoll(&p, 1, 0) < 0)
            return POLLERR;
        return p.revents;
    }
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE)
        return POLLNVAL;
    short r = 0;
    switch (GetFileType(h)) {
    case FILE_TYPE_PIPE: {
        DWORD avail = 0;
        if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL))
            return POLLHUP; /* the other end has gone */
        if ((events & POLLIN) && avail > 0)
            r |= POLLIN;
        if (events & POLLOUT)
            r |= POLLOUT;
        return r;
    }
    case FILE_TYPE_CHAR:
        if ((events & POLLIN) && console_has_key(h))
            r |= POLLIN;
        if (events & POLLOUT)
            r |= POLLOUT;
        return r;
    default: /* a file is always ready */
        return (short)(events & (POLLIN | POLLOUT));
    }
}

int janas_poll(struct pollfd *fds, nfds_t n, int timeout_ms)
{
    /* sockets alone: Winsock's own, which waits without turning */
    int all_sockets = n > 0;
    for (nfds_t i = 0; i < n && all_sockets; i++)
        all_sockets = fds[i].fd < 0 || is_socket(fds[i].fd);
    if (all_sockets) {
        WSAPOLLFD p[64];
        if (n <= 64) {
            for (nfds_t i = 0; i < n; i++)
                p[i] = (WSAPOLLFD){.fd = fds[i].fd < 0
                                             ? INVALID_SOCKET
                                             : (SOCKET)(intptr_t)fds[i].fd,
                                   .events = (short)(fds[i].events & ASKABLE)};
            int r = WSAPoll(p, (ULONG)n, timeout_ms);
            for (nfds_t i = 0; i < n; i++)
                fds[i].revents = p[i].revents;
            return r;
        }
    }
    /* pipes and the console have nothing to wait on with sockets: look,
       and look again a millisecond later */
    ULONGLONG start = GetTickCount64();
    for (;;) {
        int ready = 0;
        for (nfds_t i = 0; i < n; i++) {
            fds[i].revents = fd_ready(fds[i].fd, fds[i].events);
            if (fds[i].revents)
                ready++;
        }
        if (ready || timeout_ms == 0)
            return ready;
        if (timeout_ms > 0 && GetTickCount64() - start >= (ULONGLONG)timeout_ms)
            return 0;
        Sleep(1);
    }
}

#endif /* _WIN32 */
