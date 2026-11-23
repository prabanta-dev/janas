/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-brebus-helper: the one privileged piece of Brebus served
 * to the users of the system (brebus --serve --users system), Linux. It is
 * installed setuid root, and it serves one account alone: the one named
 * in /etc/janas/brebus-helper.conf ("caller = NAME"), a file root's and
 * written by no one else - Brebus's own account, which runs the front.
 * Anyone else is refused, whatever the file's mode lets run it.
 *
 *   auth        reads "NAME\nPASSWORD\n" on its input. A NAME that may use
 *               Brebus - its uid 1000 or more, a member of the group brebus
 *               (a user) or brebus-admin (an administrator) - has its
 *               password checked by PAM (the service "brebus"); no other
 *               name is checked at all, so this is no oracle of other
 *               accounts' passwords. Right: prints "user TICKET" or "admin
 *               TICKET" and exits 0, the ticket kept for ten minutes in
 *               /run/janas-brebus/NAME; else exits 1.
 *   spawn NAME PORT
 *               reads NAME's ticket on its input (spent: one spawn a
 *               login), becomes NAME - its groups, gid, uid, never root
 *               again - in its home, with a clean environment and the
 *               BREBUS_* variables the caller gave, and runs the brebus
 *               beside this program as brebus --child --port PORT. That
 *               brebus, every program beside it and in its services/, its
 *               folder and every folder above it must be root's and
 *               written by no one else; it is run through the descriptor
 *               checked, not by its name again. Every other descriptor is
 *               closed. Its input stays the caller's: the child ends when
 *               Brebus closes it.
 *
 * The environment, the resource limits and the signals the caller gave are
 * not kept. PAM is loaded at run time (libpam.so.0), so building it needs
 * no headers of PAM's.
 */
#include <stdio.h>
#include <string.h>

#ifdef __linux__
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <limits.h>
#include <pwd.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define TICKETS "/run/janas-brebus"
#define CONF "/etc/janas/brebus-helper.conf"
#define TICKET_LIFE 600

/* Linux-PAM's interface, as its headers declare it */
#define PAM_SUCCESS 0
#define PAM_BUF_ERR 5
#define PAM_CONV_ERR 19
#define PAM_PROMPT_ECHO_OFF 1
#define PAM_PROMPT_ECHO_ON 2
#define PAM_RHOST 4
#define PAM_SILENT 0x8000
#define PAM_DISALLOW_NULL_AUTHTOK 0x0001

struct pam_message {
    int msg_style;
    const char *msg;
};
struct pam_response {
    char *resp;
    int resp_retcode;
};
struct pam_conv {
    int (*conv)(int, const struct pam_message **, struct pam_response **,
                void *);
    void *appdata_ptr;
};
typedef struct pam_handle pam_handle_t;

/* the password, to a prompt that hides what is typed; anything that would
   show it (a code asked by another module) is refused */
static int conv(int n, const struct pam_message **msg,
                struct pam_response **resp, void *data)
{
    if (n <= 0 || n > 32)
        return PAM_CONV_ERR;
    struct pam_response *r = calloc((size_t)n, sizeof *r);
    if (!r)
        return PAM_BUF_ERR;
    for (int i = 0; i < n; i++) {
        if (msg[i]->msg_style == PAM_PROMPT_ECHO_ON) {
            for (int k = 0; k < i; k++)
                free(r[k].resp);
            free(r);
            return PAM_CONV_ERR;
        }
        if (msg[i]->msg_style == PAM_PROMPT_ECHO_OFF) {
            r[i].resp = strdup((const char *)data);
            if (!r[i].resp) {
                for (int k = 0; k < i; k++)
                    free(r[k].resp);
                free(r);
                return PAM_BUF_ERR;
            }
        }
    }
    *resp = r;
    return PAM_SUCCESS;
}

/* name: letters, digits, _ and -, not first a digit or -, 32 at most */
static int good_name(const char *s)
{
    size_t n = strlen(s);
    if (!n || n > 32 || (s[0] >= '0' && s[0] <= '9') || s[0] == '-')
        return 0;
    return strspn(s, "abcdefghijklmnopqrstuvwxyz0123456789_-") == n;
}

/* the account, if it may use Brebus: 2 an administrator, 1 a user, 0 no */
static int allowed(const char *name, struct passwd *out, char *buf, size_t bn)
{
    struct passwd *pw = NULL;
    if (!good_name(name) || getpwnam_r(name, out, buf, bn, &pw) != 0 || !pw ||
        pw->pw_uid < 1000 || pw->pw_uid == 65534)
        return 0;
    gid_t gs[256];
    int ng = 256;
    if (getgrouplist(name, pw->pw_gid, gs, &ng) < 0)
        return 0;
    struct group *ga = getgrnam("brebus-admin");
    gid_t admin = ga ? ga->gr_gid : (gid_t)-1;
    struct group *gu = getgrnam("brebus");
    gid_t user = gu ? gu->gr_gid : (gid_t)-1;
    int role = 0;
    for (int i = 0; i < ng; i++) {
        if (gs[i] == admin)
            role = 2;
        else if (gs[i] == user && !role)
            role = 1;
    }
    return role;
}

/* the caller is the account the configuration names: 1, or 0 */
static int caller_ok(void)
{
    int fd = open(CONF, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    struct stat st;
    char buf[512] = "";
    if (fd < 0 || fstat(fd, &st) != 0 || st.st_uid != 0 ||
        (st.st_mode & (S_IWGRP | S_IWOTH)) || !S_ISREG(st.st_mode)) {
        if (fd >= 0)
            close(fd);
        fprintf(stderr,
                "janas-brebus-helper: %s, root's and written by no "
                "one else, must name its caller\n",
                CONF);
        return 0;
    }
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    buf[n > 0 ? n : 0] = 0;
    char name[64] = "";
    for (char *l = strtok(buf, "\n"); l; l = strtok(NULL, "\n"))
        if (sscanf(l, " caller = %63s", name) == 1)
            break;
    struct passwd *pw = good_name(name) ? getpwnam(name) : NULL;
    if (!pw || pw->pw_uid == 0 || pw->pw_uid != getuid()) {
        fprintf(stderr, "janas-brebus-helper: not for this account\n");
        return 0;
    }
    return 1;
}

static void hex(const unsigned char *p, size_t n, char *out)
{
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++)
        out[2 * i] = d[p[i] >> 4], out[2 * i + 1] = d[p[i] & 15];
    out[2 * n] = 0;
}

/* the tickets' folder: root's, 0700, not a link */
static int tickets_dir(void)
{
    if (mkdir(TICKETS, 0700) != 0 && errno != EEXIST)
        return -1;
    struct stat st;
    return lstat(TICKETS, &st) == 0 && S_ISDIR(st.st_mode) && st.st_uid == 0 &&
                   (st.st_mode & 077) == 0
               ? 0
               : -1;
}

static int do_auth(void)
{
    char name[64], pass[512];
    if (!fgets(name, sizeof name, stdin) || !fgets(pass, sizeof pass, stdin))
        return 1;
    name[strcspn(name, "\n")] = 0;
    pass[strcspn(pass, "\n")] = 0;
    struct passwd pw;
    char buf[4096];
    int role = allowed(name, &pw, buf, sizeof buf);
    if (!role) {
        /* not one of Brebus's: no password checked, the time of a wrong
           one */
        explicit_bzero(pass, sizeof pass);
        sleep(2);
        return 1;
    }
    void *lib = dlopen("libpam.so.0", RTLD_NOW);
    int (*start)(const char *, const char *, const struct pam_conv *,
                 pam_handle_t **) =
        lib ? (int (*)(const char *, const char *, const struct pam_conv *,
                       pam_handle_t **))dlsym(lib, "pam_start")
            : NULL;
    int (*authenticate)(pam_handle_t *, int) =
        lib ? (int (*)(pam_handle_t *, int))dlsym(lib, "pam_authenticate")
            : NULL;
    int (*acct)(pam_handle_t *, int) =
        lib ? (int (*)(pam_handle_t *, int))dlsym(lib, "pam_acct_mgmt") : NULL;
    int (*end)(pam_handle_t *, int) =
        lib ? (int (*)(pam_handle_t *, int))dlsym(lib, "pam_end") : NULL;
    int (*set_item)(pam_handle_t *, int, const void *) =
        lib ? (int (*)(pam_handle_t *, int, const void *))dlsym(lib,
                                                                "pam_set_item")
            : NULL;
    if (!start || !authenticate || !acct || !end) {
        fprintf(stderr, "janas-brebus-helper: no PAM (libpam.so.0)\n");
        explicit_bzero(pass, sizeof pass);
        return 1;
    }
    struct pam_conv c = {.conv = conv, .appdata_ptr = pass};
    pam_handle_t *h = NULL;
    int rc = start("brebus", name, &c, &h);
    if (rc == PAM_SUCCESS && set_item)
        set_item(h, PAM_RHOST, "brebus");
    if (rc == PAM_SUCCESS)
        rc = authenticate(h, PAM_SILENT | PAM_DISALLOW_NULL_AUTHTOK);
    if (rc == PAM_SUCCESS)
        rc = acct(h, PAM_SILENT);
    if (h)
        end(h, rc);
    explicit_bzero(pass, sizeof pass);
    if (rc != PAM_SUCCESS)
        return 1;
    /* the ticket a spawn of this user must show, once, within its life */
    unsigned char r[32];
    char t[65], path[128], line[128];
    if (getrandom(r, sizeof r, 0) != (ssize_t)sizeof r || tickets_dir() != 0)
        return 1;
    hex(r, sizeof r, t);
    snprintf(path, sizeof path, "%s/%s", TICKETS, name);
    int fd =
        open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0600);
    int n = snprintf(line, sizeof line, "%s %lld\n", t,
                     (long long)time(NULL) + TICKET_LIFE);
    if (fd < 0 || write(fd, line, (size_t)n) != n || close(fd) != 0)
        return 1;
    printf("%s %s\n", role == 2 ? "admin" : "user", t);
    return 0;
}

/* name's ticket read from the input and spent: 1 when it is the one kept,
   in its life */
static int ticket_ok(const char *name)
{
    char given[80] = "", kept[128] = "", path[128];
    size_t k = 0;
    char ch;
    /* a byte at a time: nothing past the line is taken from the input */
    while (k + 1 < sizeof given && read(STDIN_FILENO, &ch, 1) == 1 &&
           ch != '\n')
        given[k++] = ch;
    given[k] = 0;
    snprintf(path, sizeof path, "%s/%s", TICKETS, name);
    struct stat st;
    int fd =
        tickets_dir() == 0 ? open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC) : -1;
    if (fd < 0)
        return 0;
    ssize_t n = fstat(fd, &st) == 0 && st.st_uid == 0 && S_ISREG(st.st_mode)
                    ? read(fd, kept, sizeof kept - 1)
                    : -1;
    close(fd);
    unlink(path);
    if (n <= 0)
        return 0;
    kept[n] = 0;
    char t[80];
    long long until = 0;
    if (sscanf(kept, "%79s %lld", t, &until) != 2 || strlen(t) != 64 ||
        strlen(given) != 64 || time(NULL) > until)
        return 0;
    unsigned char d = 0;
    for (int i = 0; i < 64; i++)
        d |= (unsigned char)(t[i] ^ given[i]);
    return d == 0;
}

/* a descriptor root's and written by no one else */
static int root_only(int fd)
{
    struct stat st;
    return fstat(fd, &st) == 0 && st.st_uid == 0 &&
           !(st.st_mode & (S_IWGRP | S_IWOTH));
}

/* every file of a folder root's and written by no one else (the programs
   a user's Brebus runs: janas-server, the services...) */
static int files_root_only(int parent, const char *sub)
{
    int fd = sub ? openat(parent, sub,
                          O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                 : dup(parent);
    if (fd < 0)
        return sub ? errno == ENOENT : 0; /* no services/: nothing to run */
    if (!root_only(fd)) {
        close(fd);
        return 0;
    }
    DIR *d = fdopendir(fd);
    if (!d) {
        close(fd);
        return 0;
    }
    int ok = 1;
    struct dirent *e;
    while (ok && (e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        struct stat st;
        ok = fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW) == 0 &&
             st.st_uid == 0 && !(st.st_mode & (S_IWGRP | S_IWOTH));
        if (!ok)
            fprintf(stderr, "janas-brebus-helper: %s/%s is not root's alone\n",
                    sub ? sub : ".", e->d_name);
    }
    closedir(d);
    return ok;
}

/* the folder of this program, each folder from / to it root's and written
   by no one else: its descriptor, or -1 */
static int own_folder(void)
{
    char self[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n <= 0 || self[0] != '/')
        return -1;
    self[n] = 0;
    char *sl = strrchr(self, '/');
    *sl = 0;
    int fd = open("/", O_PATH | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0 || !root_only(fd))
        return -1;
    for (char *p = strtok(self, "/"); p; p = strtok(NULL, "/")) {
        int next = openat(fd, p, O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        close(fd);
        if (next < 0 || !root_only(next)) {
            fprintf(stderr,
                    "janas-brebus-helper: the folders up to it must "
                    "be root's and written by no one else (%s)\n",
                    p);
            if (next >= 0)
                close(next);
            return -1;
        }
        fd = next;
    }
    return fd;
}

static int do_spawn(const char *name, const char *port, char **keep)
{
    struct passwd pw;
    char buf[4096];
    char *end = NULL;
    long p = strtol(port, &end, 10);
    if (!allowed(name, &pw, buf, sizeof buf) || !end || *end || p < 1024 ||
        p > 65535 || !ticket_ok(name)) {
        fprintf(stderr,
                "janas-brebus-helper: no spawn of %s without its "
                "login\n",
                name);
        return 2;
    }
    char ps[8];
    snprintf(ps, sizeof ps, "%ld", p);
    /* the brebus beside this program, and what it runs, root's alone */
    int dir = own_folder();
    if (dir < 0)
        return 1;
    int rd = openat(dir, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int ok =
        rd >= 0 && files_root_only(rd, NULL) && files_root_only(rd, "services");
    if (rd >= 0)
        close(rd);
    int exe = openat(dir, "brebus", O_PATH | O_NOFOLLOW | O_CLOEXEC);
    struct stat st;
    if (!ok || exe < 0 || fstat(exe, &st) != 0 || !S_ISREG(st.st_mode) ||
        !root_only(exe)) {
        fprintf(stderr,
                "janas-brebus-helper: brebus and the programs beside it "
                "must be root's and written by no one else\n");
        return 1;
    }
    close(dir);
    static char env[16][4200];
    char *envp[17];
    int k = 0;
    for (int i = 0; keep[i] && k < 6; i++)
        envp[k] = keep[i], k++;
    snprintf(env[k], sizeof env[k], "BREBUS_ROOT="), envp[k] = env[k], k++;
    snprintf(env[k], sizeof env[k], "BREBUS_STDIN_LIFE=1"), envp[k] = env[k],
                                                            k++;
    snprintf(env[k], sizeof env[k], "HOME=%s", pw.pw_dir), envp[k] = env[k],
                                                           k++;
    snprintf(env[k], sizeof env[k], "USER=%s", pw.pw_name), envp[k] = env[k],
                                                            k++;
    snprintf(env[k], sizeof env[k], "LOGNAME=%s", pw.pw_name), envp[k] = env[k],
                                                               k++;
    snprintf(env[k], sizeof env[k], "PATH=/usr/local/bin:/usr/bin:/bin"),
        envp[k] = env[k], k++;
    snprintf(env[k], sizeof env[k], "LANG=C.UTF-8"), envp[k] = env[k], k++;
    envp[k] = NULL;
    /* the user, for good: groups, gid, uid, and no way back */
    if (initgroups(pw.pw_name, pw.pw_gid) != 0 || setgid(pw.pw_gid) != 0 ||
        setuid(pw.pw_uid) != 0 || setuid(0) == 0 || getuid() == 0 ||
        geteuid() == 0) {
        fprintf(stderr, "janas-brebus-helper: could not become %s\n", name);
        return 1;
    }
    if (chdir(pw.pw_dir) != 0 && chdir("/") != 0)
        return 1;
    umask(077);
    /* nothing of the caller's but its three: the program checked, run
       through its descriptor */
    if (exe > 3)
        syscall(SYS_close_range, 3, (unsigned)exe - 1, 0);
    syscall(SYS_close_range, (unsigned)exe + 1, ~0U, 0);
    char *argv[] = {"brebus", "--child", "--port", ps, NULL};
    syscall(SYS_execveat, exe, "", argv, envp, AT_EMPTY_PATH);
    perror("janas-brebus-helper: brebus");
    return 1;
}

int main(int argc, char **argv)
{
    /* nothing of the caller's: its limits, its signals, its environment
       (but the variables a spawn hands on) */
    struct rlimit z = {0, 0}, inf = {RLIM_INFINITY, RLIM_INFINITY};
    setrlimit(RLIMIT_CORE, &z);
    setrlimit(RLIMIT_FSIZE, &inf);
    sigset_t none;
    sigemptyset(&none);
    sigprocmask(SIG_SETMASK, &none, NULL);
    for (int s = 1; s < NSIG; s++)
        if (s != SIGKILL && s != SIGSTOP)
            signal(s, SIG_DFL);
    static const char *const KEEP[] = {"BREBUS_CHILD_TOKEN", "BREBUS_FRONT",
                                       "BREBUS_FRONT_TLS", "BREBUS_FRONT_CERT",
                                       "JANAS_MODELS"};
    static char kept[6][4200];
    static char *keep[7];
    int nk = 0;
    for (size_t i = 0; i < sizeof KEEP / sizeof *KEEP; i++) {
        const char *v = getenv(KEEP[i]);
        if (v && strlen(v) < 4000 && !strchr(v, '\n'))
            snprintf(kept[nk], sizeof kept[nk], "%s=%s", KEEP[i], v),
                keep[nk] = kept[nk], nk++;
    }
    keep[nk] = NULL;
    clearenv();
    setenv("PATH", "/usr/bin:/bin", 1);
    umask(077);
    if (!caller_ok())
        return 2;
    if (argc == 2 && !strcmp(argv[1], "auth"))
        return do_auth();
    if (argc == 4 && !strcmp(argv[1], "spawn"))
        return do_spawn(argv[2], argv[3], keep);
    fprintf(stderr, "usage: janas-brebus-helper auth | spawn NAME PORT\n"
                    "Brebus's privileged piece (brebus --serve --users "
                    "system): not to be run by hand.\n");
    return 2;
}
#else
int main(void)
{
    fprintf(stderr, "janas-brebus-helper: Linux only\n");
    return 2;
}
#endif
