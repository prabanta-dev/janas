/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sandbox.c - the cage of sandbox.h.
 *
 * The caller runs itself again (/proc/self/exe JANAS_SANDBOX_ARG ...), so
 * the cage is built by a fresh process with a single thread, where every
 * call is safe. That process unshares a user namespace (mapping only its
 * own uid and gid) with mount, PID, network, IPC and UTS namespaces, and
 * forks the cage's first process, PID 1 of the new namespace, which:
 *   - builds a root in memory (tmpfs): /usr and the links or directories
 *     beside it (/bin, /lib, /lib64...) read-only, a few files of /etc the
 *     programs need (the linker's cache, the alternatives, fonts, python's
 *     and matplotlib's settings) read-only, the data directory writable at
 *     /mnt/data, a /tmp in memory of bounded size, a /proc of its own, a
 *     /dev with null, zero, full, random and urandom only;
 *   - moves into it (pivot_root), drops the old root, makes the root
 *     read-only;
 *   - sets its limits (address space, file size, open files, no core);
 *   - forbids new privileges and installs a seccomp filter: no new
 *     processes (fork, vfork, clone without CLONE_THREAD; clone3 answered
 *     ENOSYS so the C library makes its threads with clone), no mounts,
 *     namespaces, tracing, keys, BPF, perf, userfaultfd, io_uring, kernel
 *     modules, reboot or clock changes;
 *   - runs the program with the environment it was given.
 * Both processes die with their parent (PR_SET_PDEATHSIG), and every
 * process of the namespace dies with its PID 1, so the cage never outlives
 * the program that made it. The network namespace has no interface up.
 */
#include "sandbox.h"

#include <stdio.h>
#include <string.h>

#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))

#include <errno.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef environ /* MinGW's stdlib.h: a macro for _environ (issue #40) */
extern char **environ;
#endif

/* ---- the builder (a fresh process: plain calls are safe) ---- */

static void die(const char *what)
{
    dprintf(2, "janas sandbox: %s: %s\n", what, strerror(errno));
    _exit(127);
}

static void write_file(const char *path, const char *text)
{
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0 || write(fd, text, strlen(text)) != (ssize_t)strlen(text))
        die(path);
    close(fd);
}

/* mkdir -p */
static void make_dirs(const char *path)
{
    char p[4096];
    snprintf(p, sizeof p, "%s", path);
    for (char *s = p + 1; *s; s++)
        if (*s == '/') {
            *s = 0;
            if (mkdir(p, 0755) != 0 && errno != EEXIST)
                die(p);
            *s = '/';
        }
    if (mkdir(p, 0755) != 0 && errno != EEXIST)
        die(p);
}

/* An empty file to bind a file onto (its directories made). */
static void make_file(const char *path)
{
    char d[4096];
    snprintf(d, sizeof d, "%s", path);
    char *slash = strrchr(d, '/');
    if (slash && slash != d) {
        *slash = 0;
        make_dirs(d);
    }
    int fd = open(path, O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0)
        die(path);
    close(fd);
}

/* src bound at dst, read-only unless rw; the flags the kernel keeps locked
   in a user namespace (nosuid, nodev, noexec, atime) kept. */
static void bind(const char *src, const char *dst, int rw)
{
    if (mount(src, dst, NULL, MS_BIND | MS_REC, NULL) != 0)
        die(dst);
    struct statvfs v;
    unsigned long fl = MS_BIND | MS_REMOUNT | MS_NOSUID | MS_NODEV;
    if (statvfs(dst, &v) == 0) {
        if (v.f_flag & ST_NOEXEC)
            fl |= MS_NOEXEC;
        if (v.f_flag & ST_NOATIME)
            fl |= MS_NOATIME;
        if (v.f_flag & ST_NODIRATIME)
            fl |= MS_NODIRATIME;
        if (v.f_flag & ST_RELATIME)
            fl |= MS_RELATIME;
    }
    if (!rw)
        fl |= MS_RDONLY;
    if (mount(NULL, dst, NULL, fl, NULL) != 0)
        die(dst);
}

/* Host path p (a directory or a file) bound read-only at root + p, if it
   is there. */
static void bind_host(const char *root, const char *p)
{
    struct stat st;
    if (stat(p, &st) != 0)
        return;
    char dst[4096];
    snprintf(dst, sizeof dst, "%s%s", root, p);
    if (S_ISDIR(st.st_mode))
        make_dirs(dst);
    else
        make_file(dst);
    bind(p, dst, 0);
}

/* /bin, /lib... beside /usr: the same link, or the directory read-only */
static void top_entry(const char *root, const char *name)
{
    char p[256], dst[4096], target[4096];
    snprintf(p, sizeof p, "/%s", name);
    struct stat st;
    if (lstat(p, &st) != 0)
        return;
    snprintf(dst, sizeof dst, "%s/%s", root, name);
    if (S_ISLNK(st.st_mode)) {
        ssize_t k = readlink(p, target, sizeof target - 1);
        if (k < 0)
            die(p);
        target[k] = 0;
        if (symlink(target, dst) != 0)
            die(dst);
    } else if (S_ISDIR(st.st_mode)) {
        make_dirs(dst);
        bind(p, dst, 0);
    }
}

static int pivot_root_(const char *new_root, const char *put_old)
{
    return (int)syscall(SYS_pivot_root, new_root, put_old);
}

/* ---- seccomp ---- */

#if defined(__x86_64__)
#define ARCH AUDIT_ARCH_X86_64
#else
#define ARCH AUDIT_ARCH_AARCH64
#endif

#define DENY(nr)                                                               \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1),                           \
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)

static void seccomp_on(void)
{
    struct sock_filter f[] = {
        /* the architecture this was built for, nothing else */
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, ARCH, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
#if defined(__x86_64__)
        /* no x32 calls */
        BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, 0x40000000, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        DENY(SYS_fork),
        DENY(SYS_vfork),
        DENY(SYS_iopl),
        DENY(SYS_ioperm),
#endif
        /* clone: threads only */
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_clone, 0, 4),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 offsetof(struct seccomp_data, args[0])),
        BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, CLONE_THREAD, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
#ifdef SYS_clone3
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_clone3, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | ENOSYS),
#endif
        DENY(SYS_ptrace),
        DENY(SYS_mount),
        DENY(SYS_umount2),
        DENY(SYS_pivot_root),
        DENY(SYS_chroot),
        DENY(SYS_unshare),
        DENY(SYS_setns),
        DENY(SYS_bpf),
        DENY(SYS_keyctl),
        DENY(SYS_add_key),
        DENY(SYS_request_key),
        DENY(SYS_perf_event_open),
        DENY(SYS_userfaultfd),
        DENY(SYS_init_module),
        DENY(SYS_finit_module),
        DENY(SYS_delete_module),
        DENY(SYS_kexec_load),
#ifdef SYS_kexec_file_load
        DENY(SYS_kexec_file_load),
#endif
        DENY(SYS_reboot),
        DENY(SYS_swapon),
        DENY(SYS_swapoff),
        DENY(SYS_acct),
        DENY(SYS_settimeofday),
        DENY(SYS_clock_settime),
        DENY(SYS_open_by_handle_at),
        DENY(SYS_name_to_handle_at),
        DENY(SYS_quotactl),
        DENY(SYS_personality),
        DENY(SYS_process_vm_readv),
        DENY(SYS_process_vm_writev),
        DENY(SYS_kcmp),
#ifdef SYS_io_uring_setup
        DENY(SYS_io_uring_setup),
        DENY(SYS_io_uring_enter),
        DENY(SYS_io_uring_register),
#endif
#ifdef SYS_fsopen
        DENY(SYS_fsopen),
        DENY(SYS_fsconfig),
        DENY(SYS_fsmount),
        DENY(SYS_fspick),
        DENY(SYS_move_mount),
        DENY(SYS_open_tree),
#endif
#ifdef SYS_mount_setattr
        DENY(SYS_mount_setattr),
#endif
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog prog = {.len = (unsigned short)(sizeof f / sizeof f[0]),
                              .filter = f};
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0)
        die("no_new_privs");
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0)
        die("seccomp");
}

static void limit(int what, uint64_t v)
{
    struct rlimit r = {.rlim_cur = (rlim_t)v, .rlim_max = (rlim_t)v};
    if (setrlimit(what, &r) != 0)
        die("setrlimit");
}

/* argv: JANAS_SANDBOX_ARG data mem file tmp n_ro ro... -- program args */
void janas_sandbox_main(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], JANAS_SANDBOX_ARG) != 0)
        return;
    if (argc < 8)
        _exit(127);
    const char *data = argv[2];
    uint64_t mem = strtoull(argv[3], NULL, 10);
    uint64_t fsz = strtoull(argv[4], NULL, 10);
    uint64_t tmp = strtoull(argv[5], NULL, 10);
    int n_ro = atoi(argv[6]);
    if (n_ro < 0 || 7 + n_ro + 1 >= argc || strcmp(argv[7 + n_ro], "--"))
        _exit(127);
    char **ro = argv + 7;
    char **prog = argv + 8 + n_ro;

    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() == 1)
        _exit(127);
    for (int fd = 3; fd < 1024; fd++)
        close(fd);
    if (access(data, F_OK) != 0)
        die(data);
    if (n_ro > 64)
        _exit(127);
    uid_t uid = getuid();
    gid_t gid = getgid();
    if (unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET |
                CLONE_NEWIPC | CLONE_NEWUTS) != 0)
        die("unshare");
    char map[64];
    write_file("/proc/self/setgroups", "deny");
    snprintf(map, sizeof map, "%u %u 1\n", (unsigned)uid, (unsigned)uid);
    write_file("/proc/self/uid_map", map);
    snprintf(map, sizeof map, "%u %u 1\n", (unsigned)gid, (unsigned)gid);
    write_file("/proc/self/gid_map", map);

    pid_t b = fork();
    if (b < 0)
        die("fork");
    if (b > 0) { /* the first process: waits, and gives its end back */
        int st = 0;
        while (waitpid(b, &st, 0) < 0 && errno == EINTR)
            ;
        _exit(WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st));
    }
    /* PID 1 of the cage */
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0)
        die("pdeathsig");
    if (mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0)
        die("private /");
    /* the sources, opened in this mount namespace (a bind from another is
       refused) before /tmp is covered */
    int data_fd = open(data, O_PATH | O_DIRECTORY | O_CLOEXEC);
    if (data_fd < 0)
        die(data);
    int ro_fd[64];
    for (int i = 0; i < n_ro; i++)
        if ((ro_fd[i] = open(ro[i], O_PATH | O_CLOEXEC)) < 0)
            die(ro[i]);

    const char *root = "/tmp";
    if (mount("tmpfs", root, "tmpfs", MS_NOSUID | MS_NODEV,
              "mode=0755,size=16m") != 0)
        die("root tmpfs");
    char p[4096], src[64];
    snprintf(p, sizeof p, "%s/usr", root);
    make_dirs(p);
    bind("/usr", p, 0);
    static const char *const top[] = {"bin",   "sbin",  "lib",
                                      "lib32", "lib64", "libx32"};
    for (size_t i = 0; i < sizeof top / sizeof top[0]; i++)
        top_entry(root, top[i]);
    static const char *const etc[] = {
        "/etc/ld.so.cache",  "/etc/ld.so.conf", "/etc/ld.so.conf.d",
        "/etc/alternatives", "/etc/localtime",  "/etc/fonts",
        "/etc/matplotlibrc", "/etc/mime.types", "/etc/python3",
        "/etc/python3.11",   "/etc/python3.12", "/etc/python3.13",
        "/etc/python3.14"};
    snprintf(p, sizeof p, "%s/etc", root);
    make_dirs(p);
    for (size_t i = 0; i < sizeof etc / sizeof etc[0]; i++)
        bind_host(root, etc[i]);
    /* the data, writable */
    snprintf(p, sizeof p, "%s/mnt/data", root);
    make_dirs(p);
    snprintf(src, sizeof src, "/proc/self/fd/%d", data_fd);
    bind(src, p, 1);
    /* more paths, read-only where they are */
    for (int i = 0; i < n_ro; i++) {
        struct stat st;
        if (fstat(ro_fd[i], &st) != 0)
            die(ro[i]);
        snprintf(p, sizeof p, "%s%s", root, ro[i]);
        if (S_ISDIR(st.st_mode))
            make_dirs(p);
        else
            make_file(p);
        snprintf(src, sizeof src, "/proc/self/fd/%d", ro_fd[i]);
        bind(src, p, 0);
    }
    /* /tmp in memory */
    char opt[64];
    snprintf(p, sizeof p, "%s/tmp", root);
    make_dirs(p);
    snprintf(opt, sizeof opt, "mode=1777,size=%llu",
             (unsigned long long)(tmp ? tmp : (uint64_t)256 << 20));
    if (mount("tmpfs", p, "tmpfs", MS_NOSUID | MS_NODEV, opt) != 0)
        die("/tmp");
    /* /dev: a few devices */
    snprintf(p, sizeof p, "%s/dev", root);
    make_dirs(p);
    if (mount("tmpfs", p, "tmpfs", MS_NOSUID | MS_NOEXEC, "mode=0755") != 0)
        die("/dev");
    static const char *const dev[] = {"null", "zero", "full", "random",
                                      "urandom"};
    for (size_t i = 0; i < sizeof dev / sizeof dev[0]; i++) {
        char h[64];
        snprintf(h, sizeof h, "/dev/%s", dev[i]);
        snprintf(p, sizeof p, "%s/dev/%s", root, dev[i]);
        make_file(p);
        if (mount(h, p, NULL, MS_BIND, NULL) != 0)
            die(h);
    }
    static const char *const links[][2] = {{"/proc/self/fd", "fd"},
                                           {"/proc/self/fd/0", "stdin"},
                                           {"/proc/self/fd/1", "stdout"},
                                           {"/proc/self/fd/2", "stderr"}};
    for (size_t i = 0; i < sizeof links / sizeof links[0]; i++) {
        snprintf(p, sizeof p, "%s/dev/%s", root, links[i][1]);
        if (symlink(links[i][0], p) != 0)
            die(p);
    }
    snprintf(p, sizeof p, "%s/proc", root);
    make_dirs(p);
    if (mount("proc", p, "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC, NULL) != 0)
        die("/proc");
    /* into the new root */
    snprintf(p, sizeof p, "%s/oldroot", root);
    make_dirs(p);
    if (pivot_root_(root, p) != 0)
        die("pivot_root");
    if (chdir("/") != 0 || umount2("/oldroot", MNT_DETACH) != 0 ||
        rmdir("/oldroot") != 0)
        die("old root");
    if (mount(NULL, "/", NULL, MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV,
              NULL) != 0)
        die("read-only root");
    if (sethostname("sandbox", 7) != 0)
        die("hostname");
    if (chdir("/mnt/data") != 0)
        die("/mnt/data");
    limit(RLIMIT_AS, mem ? mem : (uint64_t)2 << 30);
    limit(RLIMIT_FSIZE, fsz ? fsz : (uint64_t)1 << 30);
    limit(RLIMIT_NOFILE, 256);
    limit(RLIMIT_CORE, 0);
    seccomp_on();
    execve(prog[0], prog, environ);
    die(prog[0]);
}

/* ---- the caller ---- */

int janas_sandbox_spawn(const struct janas_sandbox_cfg *c, char *const argv[],
                        char *const envp[], struct janas_sandbox *sb, char *why,
                        size_t n)
{
    int nro = 0;
    while (c->ro && c->ro[nro])
        nro++;
    int na = 0;
    while (argv[na])
        na++;
    if (nro > 64) {
        snprintf(why, n, "too many read-only paths");
        return -1;
    }
    char num[3][24], cnt[16];
    snprintf(num[0], sizeof num[0], "%llu", (unsigned long long)c->mem_bytes);
    snprintf(num[1], sizeof num[1], "%llu", (unsigned long long)c->file_bytes);
    snprintf(num[2], sizeof num[2], "%llu", (unsigned long long)c->tmp_bytes);
    snprintf(cnt, sizeof cnt, "%d", nro);
    char **args = calloc((size_t)(9 + nro + na), sizeof(char *));
    if (!args) {
        snprintf(why, n, "out of memory");
        return -1;
    }
    int k = 0;
    args[k++] = "janas-sandbox";
    args[k++] = JANAS_SANDBOX_ARG;
    args[k++] = (char *)c->data_dir;
    args[k++] = num[0];
    args[k++] = num[1];
    args[k++] = num[2];
    args[k++] = cnt;
    for (int i = 0; i < nro; i++)
        args[k++] = (char *)c->ro[i];
    args[k++] = "--";
    for (int i = 0; i < na; i++)
        args[k++] = argv[i];
    args[k] = NULL;
    int in[2], out[2], err[2];
    if (pipe2(in, O_CLOEXEC) != 0) {
        free(args);
        snprintf(why, n, "pipe: %s", strerror(errno));
        return -1;
    }
    if (pipe2(out, O_CLOEXEC) != 0) {
        close(in[0]), close(in[1]);
        free(args);
        snprintf(why, n, "pipe: %s", strerror(errno));
        return -1;
    }
    if (pipe2(err, O_CLOEXEC) != 0) {
        close(in[0]), close(in[1]), close(out[0]), close(out[1]);
        free(args);
        snprintf(why, n, "pipe: %s", strerror(errno));
        return -1;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out[1], 1);
    posix_spawn_file_actions_adddup2(&fa, err[1], 2);
    pid_t pid;
    int rc = posix_spawn(&pid, "/proc/self/exe", &fa, NULL, args, envp);
    posix_spawn_file_actions_destroy(&fa);
    free(args);
    close(in[0]);
    close(out[1]);
    close(err[1]);
    if (rc != 0) {
        close(in[1]), close(out[0]), close(err[0]);
        snprintf(why, n, "cannot start the cage: %s", strerror(rc));
        return -1;
    }
    sb->pid = pid;
    sb->in = in[1];
    sb->out = out[0];
    sb->err = err[0];
    return 0;
}

void janas_sandbox_kill(struct janas_sandbox *sb)
{
    if (sb->pid > 0) {
        kill(sb->pid, SIGKILL); /* its PID 1 dies with it, and all the rest */
        while (waitpid(sb->pid, NULL, 0) < 0 && errno == EINTR)
            ;
    }
    if (sb->in >= 0)
        close(sb->in);
    if (sb->out >= 0)
        close(sb->out);
    if (sb->err >= 0)
        close(sb->err);
    sb->pid = 0;
    sb->in = sb->out = sb->err = -1;
}

int janas_sandbox_available(char *why, size_t n)
{
    static int known = -1;
    static char reason[256];
    if (known < 0) {
        /* a cage built for true, read back */
        char dir[] = "/tmp/janas-sandbox-XXXXXX";
        struct janas_sandbox sb;
        char *argv[] = {"/usr/bin/true", NULL};
        char *envp[] = {NULL};
        struct janas_sandbox_cfg c = {.data_dir = dir};
        known = 0;
        if (!mkdtemp(dir)) {
            snprintf(reason, sizeof reason, "no temporary directory");
        } else if (janas_sandbox_spawn(&c, argv, envp, &sb, reason,
                                       sizeof reason) == 0) {
            close(sb.in);
            int st = 0;
            while (waitpid(sb.pid, &st, 0) < 0 && errno == EINTR)
                ;
            char msg[200] = "";
            ssize_t k = read(sb.err, msg, sizeof msg - 1);
            msg[k > 0 ? k : 0] = 0;
            char *nl = strchr(msg, '\n');
            if (nl)
                *nl = 0;
            close(sb.out);
            close(sb.err);
            known = WIFEXITED(st) && WEXITSTATUS(st) == 0;
            if (!known)
                snprintf(reason, sizeof reason, "%s",
                         msg[0] ? msg
                                : "the cage could not be built (are user "
                                  "namespaces allowed?)");
        }
        rmdir(dir);
    }
    if (!known)
        snprintf(why, n, "%s", reason);
    return known;
}

#else /* not Linux */

void janas_sandbox_main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
}

int janas_sandbox_available(char *why, size_t n)
{
    snprintf(why, n, "the cage is built on Linux only, for now");
    return 0;
}

int janas_sandbox_spawn(const struct janas_sandbox_cfg *c, char *const argv[],
                        char *const envp[], struct janas_sandbox *sb, char *why,
                        size_t n)
{
    (void)c;
    (void)argv;
    (void)envp;
    (void)sb;
    snprintf(why, n, "the cage is built on Linux only, for now");
    return -1;
}

void janas_sandbox_kill(struct janas_sandbox *sb)
{
    (void)sb;
}

#endif
