/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * proc_win.c - see proc.h: Windows, with CreateProcess.
 *
 * The program gets exactly the three handles it is given and no other
 * (PROC_THREAD_ATTRIBUTE_HANDLE_LIST), and is started suspended inside a
 * job object, so that what it starts in turn is in the job too and goes
 * with it: TerminateJobObject is the kill of a process group. A job that
 * is closed takes its processes with it, as a group would not on POSIX:
 * nothing Janas starts outlives Janas.
 *
 * Windows has one command line, not a vector: the arguments are joined
 * and quoted as the C runtime of the program will split them again. A
 * .cmd or .bat (npx is npx.cmd) runs through cmd.exe.
 */
#ifdef _WIN32
#include "common/proc.h"

#include <windows.h>
#include <shellapi.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A growing string. */
struct str {
    char *p;
    size_t n, cap;
    int oom;
};

static void put(struct str *s, const char *t, size_t n)
{
    if (s->oom)
        return;
    if (s->n + n + 2 > s->cap) {
        size_t cap = s->cap ? s->cap * 2 : 256;
        while (cap < s->n + n + 2)
            cap *= 2;
        char *p = realloc(s->p, cap);
        if (!p) {
            s->oom = 1;
            return;
        }
        s->p = p;
        s->cap = cap;
    }
    memcpy(s->p + s->n, t, n);
    s->n += n;
    s->p[s->n] = 0;
}

static void putc_(struct str *s, char c)
{
    put(s, &c, 1);
}

/* One argument quoted as CommandLineToArgvW and the C runtime read it:
   backslashes are literal except before a quote. */
static void quote_as(struct str *s, const char *a, int always)
{
    if (!always && *a && !strpbrk(a, " \t\n\v\"")) {
        put(s, a, strlen(a));
        return;
    }
    putc_(s, '"');
    for (const char *c = a;; c++) {
        size_t bs = 0;
        while (*c == '\\') {
            bs++;
            c++;
        }
        if (!*c) {
            for (size_t i = 0; i < bs * 2; i++)
                putc_(s, '\\');
            break;
        }
        if (*c == '"') {
            for (size_t i = 0; i < bs * 2 + 1; i++)
                putc_(s, '\\');
            putc_(s, '"');
        } else {
            for (size_t i = 0; i < bs; i++)
                putc_(s, '\\');
            putc_(s, *c);
        }
    }
    putc_(s, '"');
}

static void quote(struct str *s, const char *a)
{
    quote_as(s, a, 0);
}

static int ends_with(const char *s, const char *tail)
{
    size_t n = strlen(s), t = strlen(tail);
    return n >= t && _stricmp(s + n - t, tail) == 0;
}

/* The program's file: a path as given (or with .exe), a name looked for
   in PATH as .exe, .cmd, .bat. 0, or -1. */
static int find_program(const char *name, char *out, size_t len)
{
    static const char *const ext[] = {".exe", ".cmd", ".bat"};
    if (strpbrk(name, "/\\:")) {
        DWORD a = GetFileAttributesA(name);
        if (a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY)) {
            snprintf(out, len, "%s", name);
            return 0;
        }
        for (int i = 0; i < 3; i++) {
            snprintf(out, len, "%s%s", name, ext[i]);
            a = GetFileAttributesA(out);
            if (a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY))
                return 0;
        }
        return -1;
    }
    for (int i = 0; i < 3; i++) {
        DWORD n = SearchPathA(NULL, name, ext[i], (DWORD)len, out, NULL);
        if (n > 0 && n < len)
            return 0;
    }
    return -1;
}

/* This process's environment with env laid over it, as a block of
   KEY=VALUE strings that ends with an empty one. Names are compared
   without case, as Windows does. */
static char *env_block(const char *const env[])
{
    struct str b = {0};
    char *cur = GetEnvironmentStringsA();
    if (!cur)
        return NULL;
    for (const char *e = cur; *e; e += strlen(e) + 1) {
        const char *eq = strchr(e + 1, '='); /* "=C:=C:\x" starts with = */
        size_t key = eq ? (size_t)(eq - e) + 1 : strlen(e);
        int replaced = 0;
        for (size_t j = 0; env[j] && !replaced; j++)
            replaced = _strnicmp(env[j], e, key) == 0;
        if (!replaced)
            put(&b, e, strlen(e) + 1);
    }
    FreeEnvironmentStringsA(cur);
    for (size_t j = 0; env[j]; j++)
        put(&b, env[j], strlen(env[j]) + 1);
    put(&b, "", 1);
    if (b.oom) {
        free(b.p);
        return NULL;
    }
    return b.p;
}

/* An inheritable copy of the handle behind fd, or of this process's own
   std handle when fd is -1; NULL when there is none. */
static HANDLE inheritable(int fd, DWORD std)
{
    HANDLE h = fd >= 0 ? (HANDLE)_get_osfhandle(fd) : GetStdHandle(std);
    HANDLE dup = NULL;
    if (h == INVALID_HANDLE_VALUE || h == NULL)
        return NULL;
    if (!DuplicateHandle(GetCurrentProcess(), h, GetCurrentProcess(), &dup, 0,
                         TRUE, DUPLICATE_SAME_ACCESS))
        return NULL;
    return dup;
}

int janas_proc_start(struct janas_proc *p, const char *const argv[],
                     const char *const env[], const char *dir, int fd_in,
                     int fd_out, int fd_err, int flags, char *err,
                     size_t err_len)
{
    memset(p, 0, sizeof(*p));
    if (!argv || !argv[0] || !*argv[0]) {
        snprintf(err, err_len, "no command");
        errno = EINVAL;
        return -1;
    }
    if (dir) {
        DWORD a = GetFileAttributesA(dir);
        if (a == INVALID_FILE_ATTRIBUTES || !(a & FILE_ATTRIBUTE_DIRECTORY)) {
            snprintf(err, err_len, "cannot enter %s", dir);
            errno = ENOTDIR;
            return -1;
        }
    }
    char prog[MAX_PATH * 2];
    if (find_program(argv[0], prog, sizeof prog) != 0) {
        snprintf(err, err_len, "cannot start %s: not found", argv[0]);
        errno = ENOENT;
        return -1;
    }
    int script = ends_with(prog, ".cmd") || ends_with(prog, ".bat");
    /* cmd.exe reads the line before the program does: & | < > ^ ( ) are
       inert only inside quotes, so every argument is quoted; a quote of
       the argument's own would end that, and % (and ! with delayed
       expansion) expands even inside: those are refused, never passed
       (BatBadBut) */
    for (size_t i = 1; script && argv[i]; i++)
        if (strpbrk(argv[i], "\"%!\r\n")) {
            snprintf(err, err_len,
                     "cannot start %s: an argument holds a character "
                     "cmd.exe would act on (\" %% ! or a newline)",
                     argv[0]);
            errno = EINVAL;
            return -1;
        }
    struct str cmd = {0};
    if (script) {
        /* cmd.exe /s /c "the whole line": its first and last quote go */
        put(&cmd, "cmd.exe /d /s /c \"", 18);
        quote_as(&cmd, prog, 1);
    } else {
        quote(&cmd, prog);
    }
    for (size_t i = 1; argv[i]; i++) {
        putc_(&cmd, ' ');
        quote_as(&cmd, argv[i], script);
    }
    if (script)
        putc_(&cmd, '"');
    char *envp = env ? env_block(env) : NULL;
    HANDLE std[3] = {inheritable(fd_in, STD_INPUT_HANDLE),
                     inheritable(fd_out, STD_OUTPUT_HANDLE),
                     inheritable(fd_err, STD_ERROR_HANDLE)};
    HANDLE list[3];
    DWORD n_list = 0;
    for (int i = 0; i < 3; i++)
        if (std[i])
            list[n_list++] = std[i];
    SIZE_T attr_len = 0;
    InitializeProcThreadAttributeList(NULL, 1, 0, &attr_len);
    LPPROC_THREAD_ATTRIBUTE_LIST attrs = malloc(attr_len);
    int rc = -1;
    HANDLE job = NULL;
    PROCESS_INFORMATION pi = {0};
    if (cmd.oom || (env && !envp) || !attrs ||
        !InitializeProcThreadAttributeList(attrs, 1, 0, &attr_len)) {
        snprintf(err, err_len, "out of memory");
        errno = ENOMEM;
        free(attrs);
        attrs = NULL;
        goto out;
    }
    if (n_list > 0)
        UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                  list, n_list * sizeof(HANDLE), NULL, NULL);
    STARTUPINFOEXA si = {.StartupInfo = {.cb = sizeof si,
                                         .dwFlags = STARTF_USESTDHANDLES,
                                         .hStdInput = std[0],
                                         .hStdOutput = std[1],
                                         .hStdError = std[2]},
                         .lpAttributeList = attrs};
    DWORD how = CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT;
    if (flags & JANAS_PROC_DETACHED)
        how |= CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP;
    else if (flags & JANAS_PROC_GROUP)
        how |= CREATE_NEW_PROCESS_GROUP;
    /* a program with no console of its own (Brebus's window) would give
       each console program it starts a window of its own: none */
    if (!GetConsoleWindow())
        how |= CREATE_NO_WINDOW;
    job = CreateJobObjectA(NULL, NULL);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {0};
        li.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li,
                                sizeof li);
    }
    if (!CreateProcessA(NULL, cmd.p, NULL, NULL, TRUE, how, envp, dir,
                        &si.StartupInfo, &pi)) {
        snprintf(err, err_len, "cannot start %s: error %lu", argv[0],
                 (unsigned long)GetLastError());
        errno = ENOEXEC;
        goto out;
    }
    if (job && !AssignProcessToJobObject(job, pi.hProcess)) {
        CloseHandle(job); /* runs on its own: still a program that works */
        job = NULL;
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    p->id = (intptr_t)pi.hProcess;
    p->job = (intptr_t)job;
    p->pid = pi.dwProcessId;
    job = NULL;
    rc = 0;
out:
    if (attrs) {
        DeleteProcThreadAttributeList(attrs);
        free(attrs);
    }
    if (job)
        CloseHandle(job);
    for (int i = 0; i < 3; i++)
        if (std[i])
            CloseHandle(std[i]);
    free(envp);
    free(cmd.p);
    return rc;
}

static void clear(struct janas_proc *p)
{
    CloseHandle((HANDLE)p->id);
    if (p->job)
        CloseHandle((HANDLE)p->job);
    memset(p, 0, sizeof(*p));
}

int janas_proc_wait(struct janas_proc *p, int timeout_ms, int *code)
{
    if (!p->id)
        return -1;
    DWORD r = WaitForSingleObject(
        (HANDLE)p->id, timeout_ms < 0 ? INFINITE : (DWORD)timeout_ms);
    if (r == WAIT_TIMEOUT)
        return 0;
    if (r != WAIT_OBJECT_0)
        return -1;
    DWORD c = 0;
    GetExitCodeProcess((HANDLE)p->id, &c);
    if (code)
        *code = (int)c;
    clear(p);
    return 1;
}

int janas_proc_adopt(struct janas_proc *p, unsigned long pid, int group)
{
    (void)group;
    memset(p, 0, sizeof(*p));
    HANDLE h =
        OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE |
                        PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SET_QUOTA,
                    FALSE, (DWORD)pid);
    if (!h)
        return -1;
    /* a job of this process's: it ends with this one, as the ones started
       here do */
    HANDLE job = CreateJobObjectA(NULL, NULL);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {0};
        li.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li,
                                sizeof li);
        if (!AssignProcessToJobObject(job, h)) {
            CloseHandle(job);
            job = NULL;
        }
    }
    p->id = (intptr_t)h;
    p->job = (intptr_t)job;
    p->pid = pid;
    p->adopted = 1;
    return 0;
}

void janas_proc_kill(struct janas_proc *p, int hard)
{
    if (!p->id)
        return;
    /* Ctrl+Break reaches a group that shares this console; a program
       without one, or one that does not end, is terminated */
    if (!hard && GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, (DWORD)p->pid))
        return;
    if (p->job)
        TerminateJobObject((HANDLE)p->job, 1);
    else
        TerminateProcess((HANDLE)p->id, 1);
}

int janas_pipe(int fd[2])
{
    return _pipe(fd, 65536, _O_BINARY | _O_NOINHERIT);
}

int janas_null_fd(int for_writing)
{
    return _open("NUL", (for_writing ? _O_WRONLY : _O_RDONLY) | _O_BINARY |
                            _O_NOINHERIT);
}

int janas_open_url(const char *url)
{
    /* the default browser, as a link clicked */
    return (INT_PTR)ShellExecuteA(NULL, "open", url, NULL, NULL,
                                  SW_SHOWNORMAL) > 32
               ? 0
               : -1;
}

#endif /* _WIN32 */
