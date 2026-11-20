/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-brebus-helper: the one privileged piece of Brebus served
 * to the users of the system (brebus --serve --users system), Linux. It is
 * installed setuid root and runnable by Brebus's own account alone (owner
 * root, group that account's, mode 4750), and does two things only:
 *
 *   auth        reads "NAME\nPASSWORD\n" on its input and checks them by
 *               PAM (the service "brebus", /etc/pam.d/brebus), the account
 *               valid, NAME a member of the group brebus (a user) or
 *               brebus-admin (an administrator): prints "user" or "admin"
 *               and exits 0; 1 for a wrong name or password, 2 for an
 *               account not allowed.
 *   spawn NAME PORT
 *               becomes NAME - its groups, its gid, its uid, never root -
 *               in its home, with a clean environment and the BREBUS_*
 *               variables Brebus gave (its token, its port...), and runs
 *               the brebus beside this program as brebus --child --port
 *               PORT. That brebus, and its folder, must be root's and
 *               written by no one else. Its input stays the caller's: the
 *               child ends when Brebus closes it.
 *
 * Accounts of the system (uid below 1000, nobody) are refused whatever
 * their groups. PAM is loaded at run time (libpam.so.0), so building it
 * needs no headers of PAM's.
 */
#include <stdio.h>
#include <string.h>

#ifdef __linux__
#include <dlfcn.h>
#include <grp.h>
#include <limits.h>
#include <pwd.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

/* Linux-PAM's interface, as its headers declare it */
#define PAM_SUCCESS 0
#define PAM_BUF_ERR 5
#define PAM_CONV_ERR 19
#define PAM_PROMPT_ECHO_OFF 1
#define PAM_PROMPT_ECHO_ON 2
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

static int conv(int n, const struct pam_message **msg,
                struct pam_response **resp, void *data)
{
    if (n <= 0 || n > 32)
        return PAM_CONV_ERR;
    struct pam_response *r = calloc((size_t)n, sizeof *r);
    if (!r)
        return PAM_BUF_ERR;
    for (int i = 0; i < n; i++)
        if (msg[i]->msg_style == PAM_PROMPT_ECHO_OFF ||
            msg[i]->msg_style == PAM_PROMPT_ECHO_ON) {
            r[i].resp = strdup((const char *)data);
            if (!r[i].resp) {
                for (int k = 0; k < i; k++)
                    free(r[k].resp);
                free(r);
                return PAM_BUF_ERR;
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
    if (!start || !authenticate || !acct || !end) {
        fprintf(stderr, "janas-brebus-helper: no PAM (libpam.so.0)\n");
        memset(pass, 0, sizeof pass);
        return 1;
    }
    struct pam_conv c = {.conv = conv, .appdata_ptr = pass};
    pam_handle_t *h = NULL;
    /* checked even for a name not allowed: the time does not tell */
    int rc = start("brebus", good_name(name) ? name : "nobody", &c, &h);
    if (rc == PAM_SUCCESS)
        rc = authenticate(h, PAM_SILENT | PAM_DISALLOW_NULL_AUTHTOK);
    if (rc == PAM_SUCCESS)
        rc = acct(h, PAM_SILENT);
    if (h)
        end(h, rc);
    memset(pass, 0, sizeof pass);
    if (rc != PAM_SUCCESS)
        return 1;
    if (!role)
        return 2;
    printf("%s\n", role == 2 ? "admin" : "user");
    return 0;
}

/* a file, and its folder, root's and written by no one else */
static int root_only(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && st.st_uid == 0 &&
           !(st.st_mode & (S_IWGRP | S_IWOTH));
}

static int do_spawn(const char *name, const char *port)
{
    struct passwd pw;
    char buf[4096];
    if (!allowed(name, &pw, buf, sizeof buf)) {
        fprintf(stderr, "janas-brebus-helper: %s may not use Brebus\n", name);
        return 2;
    }
    int p = atoi(port);
    if (p < 1024 || p > 65535)
        return 2;
    /* the brebus beside this program, root's */
    char self[PATH_MAX], exe[PATH_MAX + 16];
    ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n <= 0)
        return 1;
    self[n] = 0;
    char *sl = strrchr(self, '/');
    if (!sl)
        return 1;
    *sl = 0;
    snprintf(exe, sizeof exe, "%s/brebus", self);
    if (!root_only(self) || !root_only(exe)) {
        fprintf(stderr,
                "janas-brebus-helper: %s and its folder must be root's and "
                "written by no one else\n",
                exe);
        return 1;
    }
    /* the variables Brebus gave, and nothing else of its environment */
    static const char *const KEEP[] = {"BREBUS_CHILD_TOKEN", "BREBUS_FRONT",
                                       "BREBUS_FRONT_TLS", "BREBUS_FRONT_CERT",
                                       "JANAS_MODELS"};
    static char env[16][4200];
    char *envp[17];
    int k = 0;
    for (size_t i = 0; i < sizeof KEEP / sizeof *KEEP; i++) {
        const char *v = getenv(KEEP[i]);
        if (v && strlen(v) < 4000 && !strchr(v, '\n'))
            snprintf(env[k], sizeof env[k], "%s=%s", KEEP[i], v),
                envp[k] = env[k], k++;
    }
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
    char *argv[] = {exe, "--child", "--port", (char *)port, NULL};
    execve(exe, argv, envp);
    perror("janas-brebus-helper: brebus");
    return 1;
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "auth"))
        return do_auth();
    if (argc == 4 && !strcmp(argv[1], "spawn"))
        return do_spawn(argv[2], argv[3]);
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
