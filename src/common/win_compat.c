/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * win_compat.c - the POSIX calls MinGW-w64 lacks, for Windows (see
 * win_compat.h). Nothing on other systems.
 */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <ctype.h>
#include <math.h>
#include <stdarg.h>

/* the CRT's own, under the names win_compat.h moved aside */
#undef open
#undef fopen
#undef rename
#undef pread
#undef pwrite
#undef realpath
#undef remove
#undef flock
#undef sysconf
#undef alarm
#undef setenv
#undef unsetenv
#undef getenv
#undef strcasestr
#undef memmem
#undef strndup
#undef aligned_alloc
#undef posix_memalign
#undef free
#undef getline
#undef mkdtemp
#undef open_memstream
#undef fflush
#undef fclose

char *janas_getenv(const char *name)
{
    char *v = getenv(name);
    if ((v && *v) || !name)
        return v;
    static const char *const map[][2] = {{"HOME", "USERPROFILE"},
                                         {"XDG_CONFIG_HOME", "APPDATA"},
                                         {"XDG_CACHE_HOME", "LOCALAPPDATA"},
                                         {"XDG_DATA_HOME", "LOCALAPPDATA"},
                                         {"TMPDIR", "TEMP"}};
    static char kept[5][MAX_PATH * 2];
    for (int i = 0; i < 5; i++) {
        if (strcmp(name, map[i][0]) != 0)
            continue;
        const char *w = getenv(map[i][1]);
        if (!w || !*w)
            return v;
        if (!kept[i][0]) {
            snprintf(kept[i], sizeof kept[i], "%s", w);
            for (char *c = kept[i]; *c; c++)
                if (*c == '\\')
                    *c = '/';
        }
        return kept[i];
    }
    return v;
}

int janas_mkdir(const char *path)
{
    if (_mkdir(path) == 0)
        return 0;
    int e = errno;
    DWORD a = GetFileAttributesA(path);
    errno = a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY)
                ? EEXIST
                : e;
    return -1;
}

int janas_setenv(const char *name, const char *value, int overwrite)
{
    if (!name || !*name || strchr(name, '=')) {
        errno = EINVAL;
        return -1;
    }
    if (!overwrite && getenv(name))
        return 0;
    return _putenv_s(name, value ? value : "") == 0 ? 0 : -1;
}

int janas_unsetenv(const char *name)
{
    if (!name || !*name || strchr(name, '=')) {
        errno = EINVAL;
        return -1;
    }
    /* an empty value removes the variable */
    return _putenv_s(name, "") == 0 ? 0 : -1;
}

char *janas_strcasestr(const char *hay, const char *needle)
{
    size_t n = strlen(needle);
    if (n == 0)
        return (char *)hay;
    for (; *hay; hay++)
        if (_strnicmp(hay, needle, n) == 0)
            return (char *)hay;
    return NULL;
}

void *janas_memmem(const void *hay, size_t hay_len, const void *needle,
                   size_t needle_len)
{
    const unsigned char *h = hay, *n = needle;
    if (needle_len == 0)
        return (void *)hay;
    if (hay_len < needle_len)
        return NULL;
    for (size_t i = 0; i + needle_len <= hay_len; i++)
        if (h[i] == n[0] && memcmp(h + i, n, needle_len) == 0)
            return (void *)(h + i);
    return NULL;
}

char *janas_strndup(const char *s, size_t n)
{
    size_t len = strnlen(s, n);
    char *d = malloc(len + 1);
    if (!d)
        return NULL;
    memcpy(d, s, len);
    d[len] = 0;
    return d;
}

int janas_open(const char *path, int flags, ...)
{
    int mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    return _open(path, flags | _O_BINARY,
                 mode ? (mode & 0200 ? _S_IREAD | _S_IWRITE : _S_IREAD) : 0);
}

FILE *janas_fopen(const char *path, const char *mode)
{
    char m[16];
    size_t n = strlen(mode);
    if (n + 2 > sizeof m || strchr(mode, 'b') || strchr(mode, 't'))
        return fopen(path, mode);
    /* the 'b' after the letter and any '+', before the extensions */
    size_t k = 1;
    while (k < n && mode[k] == '+')
        k++;
    memcpy(m, mode, k);
    m[k] = 'b';
    memcpy(m + k + 1, mode + k, n - k + 1);
    return fopen(path, m);
}

int janas_rename(const char *from, const char *to)
{
    if (MoveFileExA(from, to,
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED)) {
        return 0;
    }
    DWORD e = GetLastError();
    errno = e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND ? ENOENT
            : e == ERROR_ACCESS_DENIED                             ? EACCES
                                                                   : EIO;
    return -1;
}

/* A read at an offset through an OVERLAPPED structure, as POSIX's: the
   handle's position is left where it is not counted on. */
static ssize_t at_offset(int fd, void *buf, size_t n, int64_t off, int write)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    if (n > 0x7fffffffu)
        n = 0x7fffffffu;
    /* an event of the thread's own: on a handle opened for overlapped
       reads (reopen.c) the call may return before the read is done, and
       several threads may read through the same handle */
    static __thread HANDLE ev;
    if (!ev)
        ev = CreateEventA(NULL, TRUE, FALSE, NULL);
    OVERLAPPED o = {.Offset = (DWORD)(uint64_t)off,
                    .OffsetHigh = (DWORD)((uint64_t)off >> 32),
                    .hEvent = ev};
    DWORD got = 0;
    BOOL ok = write ? WriteFile(h, buf, (DWORD)n, &got, &o)
                    : ReadFile(h, buf, (DWORD)n, &got, &o);
    if (!ok && GetLastError() == ERROR_IO_PENDING)
        ok = GetOverlappedResult(h, &o, &got, TRUE);
    if (!ok) {
        DWORD e = GetLastError();
        if (!write && e == ERROR_HANDLE_EOF)
            return 0;
        errno = EIO;
        return -1;
    }
    return (ssize_t)got;
}

ssize_t janas_pread(int fd, void *buf, size_t n, int64_t off)
{
    return at_offset(fd, buf, n, off, 0);
}

ssize_t janas_pwrite(int fd, const void *buf, size_t n, int64_t off)
{
    return at_offset(fd, (void *)buf, n, off, 1);
}

int janas_remove(const char *path)
{
    DWORD a = GetFileAttributesA(path);
    if (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY))
        return _rmdir(path);
    return remove(path);
}

char *janas_realpath(const char *path, char *resolved)
{
    char *r = _fullpath(resolved, path, resolved ? MAX_PATH : 0);
    if (!r)
        return NULL;
    if (GetFileAttributesA(r) == INVALID_FILE_ATTRIBUTES) {
        if (!resolved)
            free(r);
        errno = ENOENT;
        return NULL;
    }
    return r;
}

int janas_flock(int fd, int op)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    OVERLAPPED o = {0};
    if (op & LOCK_UN)
        return UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &o) ? 0 : -1;
    DWORD f = (op & LOCK_EX) ? LOCKFILE_EXCLUSIVE_LOCK : 0;
    if (op & LOCK_NB)
        f |= LOCKFILE_FAIL_IMMEDIATELY;
    if (LockFileEx(h, f, 0, MAXDWORD, MAXDWORD, &o))
        return 0;
    errno = GetLastError() == ERROR_LOCK_VIOLATION ? EWOULDBLOCK : EIO;
    return -1;
}

long janas_sysconf(int name)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    switch (name) {
    case _SC_NPROCESSORS_ONLN:
        return (long)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    case _SC_PAGESIZE:
        return (long)si.dwPageSize;
    case _SC_PHYS_PAGES: {
        MEMORYSTATUSEX m = {.dwLength = sizeof m};
        if (!GlobalMemoryStatusEx(&m))
            return -1;
        /* in pages: a long holds 2^31 of them, 8 TiB of 4 KiB */
        return (long)(m.ullTotalPhys / si.dwPageSize);
    }
    }
    errno = EINVAL;
    return -1;
}

/* the tag before an aligned block: where malloc's block began, and that
   address mixed with a constant, so that the bytes before an ordinary
   block (the heap's own header) are not taken for one */
#define ALIGN_TAG 0x4a414e4153414c47ull
struct align_tag {
    void *base;
    uint64_t check;
};

void *janas_aligned_alloc(size_t align, size_t size)
{
    if (align < sizeof(void *) || (align & (align - 1))) {
        errno = EINVAL;
        return NULL;
    }
    size_t extra = align + sizeof(struct align_tag);
    if (size > SIZE_MAX - extra) {
        errno = ENOMEM;
        return NULL;
    }
    char *base = malloc(size + extra);
    if (!base)
        return NULL;
    uintptr_t a = ((uintptr_t)base + sizeof(struct align_tag) + align - 1) &
                  ~(uintptr_t)(align - 1);
    struct align_tag *t = (struct align_tag *)a - 1;
    t->base = base;
    t->check = ALIGN_TAG ^ (uint64_t)a;
    return (void *)a;
}

int janas_posix_memalign(void **p, size_t align, size_t size)
{
    void *a = janas_aligned_alloc(align, size ? size : 1);
    if (!a)
        return errno;
    *p = a;
    return 0;
}

void janas_free(void *p)
{
    if (!p)
        return;
    struct align_tag *t = (struct align_tag *)p - 1;
    /* A large malloc block can start a fresh page with nothing mapped before
       it (seen on ARM64 Windows: a 4 MiB-aligned block straight from the
       heap's VirtualAlloc). Then the tag would sit on another page: look at
       that page before reading it. */
    if (((uintptr_t)p & 4095) == 0) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery(t, &mbi, sizeof mbi) || mbi.State != MEM_COMMIT ||
            (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) {
            free(p);
            return;
        }
    }
    if (((uintptr_t)p & 15) == 0 &&
        t->check == (ALIGN_TAG ^ (uint64_t)(uintptr_t)p)) {
        void *base = t->base;
        t->check = 0;
        free(base);
        return;
    }
    free(p);
}

ssize_t janas_getline(char **line, size_t *cap, FILE *f)
{
    if (!line || !cap || !f) {
        errno = EINVAL;
        return -1;
    }
    size_t n = 0;
    int c;
    while ((c = getc(f)) != EOF) {
        if (n + 2 > *cap) {
            size_t nc = *cap ? *cap * 2 : 128;
            char *p = realloc(*line, nc);
            if (!p) {
                errno = ENOMEM;
                return -1;
            }
            *line = p;
            *cap = nc;
        }
        (*line)[n++] = (char)c;
        if (c == '\n')
            break;
    }
    if (n == 0)
        return -1;
    (*line)[n] = 0;
    return (ssize_t)n;
}

char *janas_mkdtemp(char *tmpl)
{
    size_t n = strlen(tmpl);
    if (n < 6 || strcmp(tmpl + n - 6, "XXXXXX") != 0) {
        errno = EINVAL;
        return NULL;
    }
    static const char abc[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    for (int tries = 0; tries < 100; tries++) {
        unsigned v = (unsigned)GetTickCount() ^
                     (unsigned)GetCurrentProcessId() * 2654435761u ^
                     (unsigned)tries * 40503u;
        for (int i = 0; i < 6; i++) {
            tmpl[n - 6 + i] = abc[v % 36];
            v = v / 36 + (unsigned)i * 7919u;
        }
        if (_mkdir(tmpl) == 0)
            return tmpl;
        if (errno != EEXIST)
            return NULL;
    }
    return NULL;
}

/* the memory streams open, with where their content goes */
static struct memstream {
    FILE *f;
    char **ptr;
    size_t *size;
} memstreams[16];
static SRWLOCK memstreams_lock = SRWLOCK_INIT;

FILE *janas_open_memstream(char **ptr, size_t *size)
{
    char dir[MAX_PATH], path[MAX_PATH];
    if (!GetTempPathA(sizeof dir, dir) ||
        !GetTempFileNameA(dir, "jns", 0, path))
        return NULL;
    /* T: kept in memory where it can, D: deleted when closed */
    FILE *f = fopen(path, "w+bTD");
    if (!f)
        return NULL;
    AcquireSRWLockExclusive(&memstreams_lock);
    int slot = -1;
    for (int i = 0; i < 16 && slot < 0; i++)
        if (!memstreams[i].f)
            slot = i;
    if (slot >= 0)
        memstreams[slot] = (struct memstream){f, ptr, size};
    ReleaseSRWLockExclusive(&memstreams_lock);
    if (slot < 0) {
        fclose(f);
        errno = EMFILE;
        return NULL;
    }
    *ptr = NULL;
    *size = 0;
    return f;
}

/* The content written so far, into a buffer of its own for the caller. */
static int memstream_publish(struct memstream *m)
{
    if (fflush(m->f) != 0)
        return -1;
    long end = ftell(m->f);
    if (end < 0)
        return -1;
    char *b = malloc((size_t)end + 1);
    if (!b)
        return -1;
    rewind(m->f);
    size_t got = fread(b, 1, (size_t)end, m->f);
    b[got] = 0;
    fseek(m->f, end, SEEK_SET);
    free(*m->ptr);
    *m->ptr = b;
    *m->size = got;
    return 0;
}

static struct memstream *memstream_of(FILE *f)
{
    for (int i = 0; i < 16; i++)
        if (memstreams[i].f == f)
            return &memstreams[i];
    return NULL;
}

int janas_fflush(FILE *f)
{
    if (!f)
        return fflush(NULL);
    AcquireSRWLockExclusive(&memstreams_lock);
    struct memstream *m = memstream_of(f);
    int rc = m ? memstream_publish(m) : fflush(f);
    ReleaseSRWLockExclusive(&memstreams_lock);
    return rc;
}

int janas_fclose(FILE *f)
{
    AcquireSRWLockExclusive(&memstreams_lock);
    struct memstream *m = memstream_of(f);
    int rc = 0;
    if (m) {
        rc = memstream_publish(m);
        m->f = NULL;
    }
    ReleaseSRWLockExclusive(&memstreams_lock);
    return fclose(f) == 0 ? rc : EOF;
}

/*
 * fmaf as the C standard has it, x * y + z rounded once: MinGW's rounds
 * the sum to a double and then again to a float, and now and then lands
 * on the next float, where glibc's is exact. The kernels, built for FMA,
 * never call it (the instruction is inlined there); the scalar code does,
 * and must agree with them to the bit. Defined under the C library's own
 * name, it is the one every file links.
 */
__attribute__((target("fma"))) static float fmaf_fma(float x, float y, float z)
{
    return __builtin_fmaf(x, y, z); /* one vfmadd */
}

/* Without the instruction: the product is exact in a double (24 + 24
   bits), the sum rounded once to a double; a float rounded from that is
   wrong only when the double lies halfway between two floats and was
   itself rounded, and then the exact sum's side decides. */
static float fmaf_soft(float x, float y, float z)
{
    double xy = (double)x * y, r = xy + z;
    uint64_t u;
    memcpy(&u, &r, sizeof u);
    int e = (int)(u >> 52 & 0x7ff);
    if ((u & 0x1fffffff) != 0x10000000 || e == 0x7ff ||
        (r - xy == z && r - z == xy))
        return (float)r;
    /* the error of r, exactly (Knuth's two-sum), moves it off the tie:
       one step of the double toward the exact sum (a step up the bits
       is away from zero) */
    double bv = r - xy, av = r - bv;
    double err = (xy - av) + (z - bv);
    if (err != 0) {
        if ((err > 0) == (r > 0))
            u++;
        else
            u--;
    }
    memcpy(&r, &u, sizeof r);
    return (float)r;
}

float fmaf(float x, float y, float z)
{
    static int has_fma = -1;
    if (has_fma < 0) {
        __builtin_cpu_init();
        has_fma = __builtin_cpu_supports("fma") ? 1 : 0;
    }
    return has_fma ? fmaf_fma(x, y, z) : fmaf_soft(x, y, z);
}

static VOID CALLBACK alarm_fired(PVOID arg, BOOLEAN timer)
{
    (void)arg;
    (void)timer;
    static const char msg[] = "Alarm clock\n";
    _write(2, msg, sizeof msg - 1);
    _exit(142); /* 128 + SIGALRM, as a shell reports it */
}

unsigned janas_alarm(unsigned seconds)
{
    static HANDLE timer;
    if (timer) {
        DeleteTimerQueueTimer(NULL, timer, NULL);
        timer = NULL;
    }
    if (seconds)
        CreateTimerQueueTimer(&timer, NULL, alarm_fired, NULL, seconds * 1000u,
                              0, WT_EXECUTEONLYONCE);
    return 0;
}

#endif /* _WIN32 */
