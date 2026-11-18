/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sandbox_win.c - the cage of sandbox.h on Windows, with what Windows has
 * for it:
 *   - an AppContainer, "Janas.Python", made once for the user: a process
 *     in it has no network (no capability is given to it), and reaches
 *     only what is granted to it - the data directory to read and write,
 *     the read-only paths and the program's own folder to read, a folder
 *     of its own for its temporary files; the grants are entries for the
 *     AppContainer's SID in those folders' access lists, kept (granting
 *     again changes nothing);
 *   - a Job Object around it: a memory limit, its first process alone (no
 *     other process started), no access to the user's windows nor to the
 *     clipboard, and every process of it ended when the job's last handle
 *     goes - the caller's, so the cage dies with its caller.
 * The data directory is the program's current folder (on Linux it is
 * /mnt/data). Windows has no limit on a file's size as Linux's
 * RLIMIT_FSIZE: file_bytes and tmp_bytes are not enforced here.
 */
#ifdef _WIN32

#include <errno.h>
#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <windows.h>
#include <aclapi.h>
#include <sddl.h>
#include <userenv.h>

#include "sandbox.h"

#define CONTAINER L"Janas.Python"

/* userenv's calls, looked up when first needed: present from Windows 8 */
typedef HRESULT(WINAPI *create_profile_fn)(PCWSTR, PCWSTR, PCWSTR,
                                           PSID_AND_ATTRIBUTES, DWORD, PSID *);
typedef HRESULT(WINAPI *derive_sid_fn)(PCWSTR, PSID *);

static HMODULE userenv(void)
{
    static HMODULE m;
    if (!m)
        m = LoadLibraryW(L"userenv.dll");
    return m;
}

void janas_sandbox_main(int argc, char **argv)
{
    (void)argc, (void)argv; /* no cage to build in this process here */
}

int janas_sandbox_available(char *why, size_t n)
{
    HMODULE u = userenv();
    if (!u || !GetProcAddress(u, "CreateAppContainerProfile")) {
        snprintf(why, n, "no AppContainers here (Windows 8 or later)");
        return 0;
    }
    return 1;
}

static WCHAR *wide(const char *s)
{
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    WCHAR *w = n > 0 ? malloc((size_t)n * sizeof *w) : NULL;
    if (w)
        MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    return w;
}

void janas_sandbox_forget(void)
{
    typedef HRESULT(WINAPI * delete_profile_fn)(PCWSTR);
    HMODULE u = userenv();
    delete_profile_fn del =
        u ? (delete_profile_fn)(void (*)(void))GetProcAddress(
                u, "DeleteAppContainerProfile")
          : NULL;
    if (del)
        del(CONTAINER);
}

/* The AppContainer's SID, its profile made the first time */
static PSID container_sid(char *why, size_t n)
{
    HMODULE u = userenv();
    create_profile_fn create =
        u ? (create_profile_fn)(void (*)(void))GetProcAddress(
                u, "CreateAppContainerProfile")
          : NULL;
    derive_sid_fn derive =
        u ? (derive_sid_fn)(void (*)(void))GetProcAddress(
                u, "DeriveAppContainerSidFromAppContainerName")
          : NULL;
    PSID sid = NULL;
    if (!create || !derive) {
        snprintf(why, n, "no AppContainers here (Windows 8 or later)");
        return NULL;
    }
    HRESULT hr = create(CONTAINER, L"Janas Python",
                        L"The code a model writes, run by Janas without "
                        L"network and with its workspace alone",
                        NULL, 0, &sid);
    if (hr == HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS))
        hr = derive(CONTAINER, &sid);
    if (FAILED(hr) || !sid) {
        snprintf(why, n, "the AppContainer could not be made (0x%08lx)",
                 (unsigned long)hr);
        return NULL;
    }
    return sid;
}

/* sid granted access to path and, inherited, to what it holds */
static int grant(PSID sid, const char *path, DWORD access, char *why, size_t n)
{
    WCHAR *w = wide(path);
    if (!w)
        return -1;
    PACL old = NULL, acl = NULL;
    PSECURITY_DESCRIPTOR sd = NULL;
    DWORD rc =
        GetNamedSecurityInfoW(w, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
                              NULL, NULL, &old, NULL, &sd);
    if (rc == ERROR_SUCCESS) {
        EXPLICIT_ACCESS_W ea = {
            .grfAccessPermissions = access,
            .grfAccessMode = GRANT_ACCESS,
            .grfInheritance = SUB_CONTAINERS_AND_OBJECTS_INHERIT,
            .Trustee = {.TrusteeForm = TRUSTEE_IS_SID,
                        .TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP,
                        .ptstrName = (LPWSTR)sid}};
        rc = SetEntriesInAclW(1, &ea, old, &acl);
        if (rc == ERROR_SUCCESS)
            rc = SetNamedSecurityInfoW(w, SE_FILE_OBJECT,
                                       DACL_SECURITY_INFORMATION, NULL, NULL,
                                       acl, NULL);
    }
    if (acl)
        LocalFree(acl);
    if (sd)
        LocalFree(sd);
    free(w);
    if (rc != ERROR_SUCCESS) {
        snprintf(why, n, "%s could not be given to the cage (error %lu)", path,
                 (unsigned long)rc);
        return -1;
    }
    return 0;
}

/* an argument quoted as CommandLineToArgvW and the C runtime read it */
static void quote(char *out, size_t cap, size_t *k, const char *a)
{
    int plain = *a && !strpbrk(a, " \t\n\v\"");
    if (plain) {
        *k += (size_t)snprintf(out + *k, cap > *k ? cap - *k : 0, "%s", a);
        return;
    }
    if (*k + 1 < cap)
        out[(*k)++] = '"';
    for (const char *c = a;; c++) {
        size_t bs = 0;
        while (*c == '\\')
            c++, bs++;
        /* backslashes doubled before a quote, and before the closing one */
        size_t times = !*c || *c == '"' ? bs * 2 : bs;
        for (size_t i = 0; i < times && *k + 1 < cap; i++)
            out[(*k)++] = '\\';
        if (!*c)
            break;
        if (*c == '"' && *k + 1 < cap)
            out[(*k)++] = '\\';
        if (*k + 1 < cap)
            out[(*k)++] = *c;
    }
    if (*k + 1 < cap)
        out[(*k)++] = '"';
    out[*k < cap ? *k : cap - 1] = 0;
}

/* envp as a block of wide strings, the whole environment. Windows starts
   an AppContainer's process by pointing its LOCALAPPDATA, TEMP and TMP at
   the container's own folder, and fails with ERROR_ENVVAR_NOT_FOUND (203)
   when the block has no LOCALAPPDATA to point: the user's is added when
   envp has none. */
static WCHAR *env_block(char *const envp[])
{
    static char la[4200];
    const char *u = getenv("LOCALAPPDATA");
    int has = 0;
    for (size_t i = 0; envp && envp[i]; i++)
        has |= !_strnicmp(envp[i], "LOCALAPPDATA=", 13);
    snprintf(la, sizeof la, "LOCALAPPDATA=%s", u ? u : "");
    const char *extra = !has && u ? la : NULL;
    size_t total = 1;
    for (size_t i = 0; envp && envp[i]; i++)
        total += (size_t)MultiByteToWideChar(CP_UTF8, 0, envp[i], -1, NULL, 0);
    if (extra)
        total += (size_t)MultiByteToWideChar(CP_UTF8, 0, extra, -1, NULL, 0);
    WCHAR *b = malloc(total * sizeof *b), *p = b;
    if (!b)
        return NULL;
    for (size_t i = 0; envp && envp[i]; i++)
        p += MultiByteToWideChar(CP_UTF8, 0, envp[i], -1, p,
                                 (int)(total - (size_t)(p - b)));
    if (extra)
        p += MultiByteToWideChar(CP_UTF8, 0, extra, -1, p,
                                 (int)(total - (size_t)(p - b)));
    *p = 0;
    return b;
}

int janas_sandbox_spawn(const struct janas_sandbox_cfg *c, char *const argv[],
                        char *const envp[], struct janas_sandbox *sb, char *why,
                        size_t n)
{
    memset(sb, 0, sizeof *sb);
    sb->in = sb->out = sb->err = -1;
    if (!c || !c->data_dir || !argv || !argv[0]) {
        snprintf(why, n, "no program or no data directory for the cage");
        return -1;
    }
    PSID sid = container_sid(why, n);
    if (!sid)
        return -1;
    /* what it reaches: the data directory, the program's folder, the
       read-only paths, a temporary folder of its own */
    char prog_dir[4096], tmp[4200];
    snprintf(prog_dir, sizeof prog_dir, "%s", argv[0]);
    char *slash = strrchr(prog_dir, '/'), *bs = strrchr(prog_dir, '\\');
    if (bs > slash)
        slash = bs;
    if (slash)
        *slash = 0;
    const char *la = getenv("LOCALAPPDATA");
    snprintf(tmp, sizeof tmp, "%s\\janas\\python\\tmp", la ? la : ".");
    CreateDirectoryA(tmp, NULL); /* there already, most times */
    int bad =
        grant(sid, c->data_dir, GENERIC_ALL, why, n) ||
        grant(sid, tmp, GENERIC_ALL, why, n) ||
        (slash && grant(sid, prog_dir, GENERIC_READ | GENERIC_EXECUTE, why, n));
    for (size_t i = 0; !bad && c->ro && c->ro[i]; i++)
        bad = grant(sid, c->ro[i], GENERIC_READ | GENERIC_EXECUTE, why, n);
    if (bad) {
        FreeSid(sid);
        return -1;
    }
    /* its standard input, output and error: pipes, the cage's ends alone
       handed on */
    HANDLE r[3] = {0}, w[3] = {0};
    SECURITY_ATTRIBUTES inh = {.nLength = sizeof inh, .bInheritHandle = TRUE};
    for (int i = 0; i < 3; i++)
        if (!CreatePipe(&r[i], &w[i], &inh, 0)) {
            snprintf(why, n, "no pipe for the cage");
            goto fail;
        }
    /* the parent's ends stay here */
    SetHandleInformation(w[0], HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(r[1], HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(r[2], HANDLE_FLAG_INHERIT, 0);
    HANDLE job = CreateJobObjectW(NULL, NULL);
    if (!job) {
        snprintf(why, n, "no Job Object for the cage");
        goto fail;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {0};
    li.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PROCESS_MEMORY |
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
        JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
    li.BasicLimitInformation.ActiveProcessLimit = 1;
    li.ProcessMemoryLimit =
        (SIZE_T)(c->mem_bytes ? c->mem_bytes : (uint64_t)2 << 30);
    JOBOBJECT_BASIC_UI_RESTRICTIONS ui = {.UIRestrictionsClass =
                                              JOB_OBJECT_UILIMIT_ALL};
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li,
                                 sizeof li) ||
        !SetInformationJobObject(job, JobObjectBasicUIRestrictions, &ui,
                                 sizeof ui)) {
        snprintf(why, n, "the cage's limits could not be set");
        CloseHandle(job);
        goto fail;
    }
    /* the AppContainer, and the three handles alone inherited */
    SECURITY_CAPABILITIES caps = {.AppContainerSid = sid};
    HANDLE inherit[3] = {r[0], w[1], w[2]};
    SIZE_T alen = 0;
    InitializeProcThreadAttributeList(NULL, 2, 0, &alen);
    LPPROC_THREAD_ATTRIBUTE_LIST attrs = malloc(alen);
    if (!attrs || !InitializeProcThreadAttributeList(attrs, 2, 0, &alen) ||
        !UpdateProcThreadAttribute(attrs, 0,
                                   PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                                   &caps, sizeof caps, NULL, NULL) ||
        !UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                   inherit, sizeof inherit, NULL, NULL)) {
        snprintf(why, n, "the cage's attributes could not be set");
        free(attrs);
        CloseHandle(job);
        goto fail;
    }
    static char line[32768];
    size_t k = 0;
    line[0] = 0;
    for (size_t i = 0; argv[i]; i++) {
        if (i && k + 1 < sizeof line)
            line[k++] = ' ';
        quote(line, sizeof line, &k, argv[i]);
    }
    WCHAR *wline = wide(line), *wdir = wide(c->data_dir),
          *wenv = env_block(envp);
    STARTUPINFOEXW si = {.StartupInfo = {.cb = sizeof si,
                                         .dwFlags = STARTF_USESTDHANDLES,
                                         .hStdInput = r[0],
                                         .hStdOutput = w[1],
                                         .hStdError = w[2]},
                         .lpAttributeList = attrs};
    PROCESS_INFORMATION pi = {0};
    BOOL ok = wline && wdir && wenv &&
              CreateProcessW(NULL, wline, NULL, NULL, TRUE,
                             CREATE_SUSPENDED | CREATE_NO_WINDOW |
                                 EXTENDED_STARTUPINFO_PRESENT |
                                 CREATE_UNICODE_ENVIRONMENT,
                             wenv, wdir, &si.StartupInfo, &pi);
    DWORD err = ok ? 0 : GetLastError();
    DeleteProcThreadAttributeList(attrs);
    free(attrs);
    free(wline), free(wdir), free(wenv);
    if (!ok) {
        snprintf(why, n, "%s could not start in the cage (error %lu)", argv[0],
                 (unsigned long)err);
        CloseHandle(job);
        goto fail;
    }
    /* in its job before it runs a single instruction */
    if (!AssignProcessToJobObject(job, pi.hProcess)) {
        snprintf(why, n, "the cage's limits could not hold it");
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hThread), CloseHandle(pi.hProcess), CloseHandle(job);
        goto fail;
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    CloseHandle(r[0]), CloseHandle(w[1]), CloseHandle(w[2]);
    FreeSid(sid);
    sb->pid = (pid_t)pi.dwProcessId;
    sb->process = (intptr_t)pi.hProcess;
    sb->job = (intptr_t)job;
    sb->in = _open_osfhandle((intptr_t)w[0], _O_WRONLY | _O_BINARY);
    sb->out = _open_osfhandle((intptr_t)r[1], _O_RDONLY | _O_BINARY);
    sb->err = _open_osfhandle((intptr_t)r[2], _O_RDONLY | _O_BINARY);
    return 0;
fail:
    for (int i = 0; i < 3; i++) {
        if (r[i])
            CloseHandle(r[i]);
        if (w[i])
            CloseHandle(w[i]);
    }
    FreeSid(sid);
    return -1;
}

void janas_sandbox_kill(struct janas_sandbox *sb)
{
    if (sb->job) { /* every process of it, at once */
        TerminateJobObject((HANDLE)sb->job, 1);
        CloseHandle((HANDLE)sb->job);
    }
    if (sb->process) {
        WaitForSingleObject((HANDLE)sb->process, 2000);
        CloseHandle((HANDLE)sb->process);
    }
    if (sb->in >= 0)
        close(sb->in);
    if (sb->out >= 0)
        close(sb->out);
    if (sb->err >= 0)
        close(sb->err);
    memset(sb, 0, sizeof *sb);
    sb->in = sb->out = sb->err = -1;
}

#endif
