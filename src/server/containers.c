/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * containers.c - OpenAI's Containers API and the Python sessions of
 * code_interpreter (containers.h). A container is a directory beside the
 * files (.containers/<id>): meta.json, files.json (the index of its files:
 * ids, paths, sizes, who wrote them) and data/, which the code sees as
 * /mnt/data. Its session is one python3 in the cage of
 * src/common/sandbox.c, started at the first piece of code and kept: a
 * small driver reads pieces of code (their length, then their bytes),
 * runs them in one namespace as Jupyter does (the value of the last
 * expression shown), saves the figures matplotlib drew into /mnt/data
 * (when the code saved no image itself) and answers with what was
 * printed, the traceback and the figures. A piece
 * that runs past two minutes ends the session (the variables are lost).
 * A container unused for its minutes (20 by default) expires: its session
 * ends and its files go.
 *
 * The code in the cage may leave symbolic links, pipes or anything in
 * /mnt/data: the server never follows a link there - the index takes
 * regular files only, and every file is opened one directory at a time
 * with O_NOFOLLOW under the container's own.
 */
#include "containers.h"

#ifdef _WIN32
/* No cage on Windows yet (src/common/sandbox.c): code is not run, and the
   API says so. */
#include <stdlib.h>
#include <string.h>

static const char OFF[] = "This server cannot run code on Windows yet.";

void srv_cell_free(struct srv_cell *c)
{
    free(c->logs);
    free(c->error);
    free(c->file_id);
    free(c->path);
    memset(c, 0, sizeof(*c));
}

const char *srv_containers_off(void)
{
    return OFF;
}

int srv_container_for(struct srv *s, yyjson_val *spec, char *id, size_t cap,
                      const char **why)
{
    (void)s, (void)spec, (void)id, (void)cap;
    *why = OFF;
    return -1;
}

int srv_container_run(struct srv *s, const char *id, const char *code,
                      struct srv_cell *out)
{
    (void)s, (void)id, (void)code;
    memset(out, 0, sizeof(*out));
    return -1;
}

int srv_container_add_file(struct srv *s, const char *cid, const char *fid,
                           const char **why)
{
    (void)s, (void)cid, (void)fid;
    *why = OFF;
    return -1;
}

int srv_container_file_open(const char *cid, const char *cfid, char *name,
                            size_t cap)
{
    (void)cid, (void)cfid, (void)name, (void)cap;
    return -1;
}

int srv_container_file_name(struct srv *s, const char *cid, const char *fid,
                            char *name, size_t cap)
{
    (void)s, (void)cid, (void)fid, (void)name, (void)cap;
    return -1;
}

int srv_containers_start(struct srv *s)
{
    return s->cfg.code_interpreter ? -1 : 0;
}

void srv_containers_stop(void)
{
}

static int refused(struct srv_req *r)
{
    return srv_reply_error(r, 400, "invalid_request_error", "unsupported_value",
                           "%s", OFF);
}

int srv_containers_create(struct srv_req *r)
{
    return refused(r);
}
int srv_containers_list(struct srv_req *r)
{
    return refused(r);
}
int srv_containers_get(struct srv_req *r)
{
    return refused(r);
}
int srv_containers_delete(struct srv_req *r)
{
    return refused(r);
}
int srv_container_files_create(struct srv_req *r)
{
    return refused(r);
}
int srv_container_files_list(struct srv_req *r)
{
    return refused(r);
}
int srv_container_files_get(struct srv_req *r)
{
    return refused(r);
}
int srv_container_files_delete(struct srv_req *r)
{
    return refused(r);
}
int srv_container_files_content(struct srv_req *r)
{
    return refused(r);
}

#else

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <microhttpd.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "common/pysession.h"
#include "common/sandbox.h"
#include "files_store.h"
#include "llm/json.h"
#include "multipart.h"
#include "store_disk.h"

#define CELL_SECONDS 120
#define DEFAULT_MINUTES 20
#define MAX_FILES 2000

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t reaper_cv = PTHREAD_COND_INITIALIZER;
static pthread_t reaper;
static int running, stopping;
static struct srv *server;
static char work[4200];
static char off_why[300] = "Start the server with --code-interpreter to "
                           "run code.";
static int on;

/* the Python driver: pieces of code in, results out (see above) */
/* ---- paths, kept safe from what the code leaves ---- */

static void path_of(char *out, size_t cap, const char *cid, const char *what)
{
    snprintf(out, cap, "%s/%s/%s", work, cid, what);
}

static const char *or_empty(const char *s)
{
    return s ? s : "";
}

static int id_ok(const char *id, const char *prefix)
{
    return id && !strncmp(id, prefix, strlen(prefix)) && srv_disk_name_ok(id);
}

/* The directory holding a file under the container's data/, by the
   file's relative path, opened one directory at a time without following
   links: its fd with the file's name in *leaf, or -1. */
static int parent_under(const char *cid, const char *rel, const char **leaf)
{
    char base[4600];
    path_of(base, sizeof base, cid, "data");
    int dir = open(base, O_DIRECTORY | O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (dir < 0)
        return -1;
    char part[300];
    const char *p = rel;
    for (;;) {
        const char *slash = strchr(p, '/');
        size_t n = slash ? (size_t)(slash - p) : strlen(p);
        if (!n || n >= sizeof part || (n == 1 && p[0] == '.') ||
            (n == 2 && p[0] == '.' && p[1] == '.')) {
            close(dir);
            return -1;
        }
        if (!slash) {
            *leaf = p;
            return dir;
        }
        memcpy(part, p, n);
        part[n] = 0;
        int next =
            openat(dir, part, O_DIRECTORY | O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        close(dir);
        if (next < 0)
            return -1;
        dir = next;
        p = slash + 1;
    }
}

/* That file opened (O_NOFOLLOW): the fd, or -1. */
static int open_under(const char *cid, const char *rel, int flags)
{
    const char *leaf;
    int dir = parent_under(cid, rel, &leaf);
    if (dir < 0)
        return -1;
    int fd = openat(dir, leaf, flags | O_NOFOLLOW | O_CLOEXEC, 0644);
    close(dir);
    return fd;
}

static int rm_entry(const char *p, const struct stat *st, int flag,
                    struct FTW *f)
{
    (void)st;
    (void)f;
    return flag == FTW_DP ? rmdir(p) : unlink(p);
}

/* a directory and all under it, links removed and never followed */
static void rm_tree(const char *p)
{
    nftw(p, rm_entry, 16, FTW_DEPTH | FTW_PHYS);
}

/* ---- documents ---- */

static yyjson_doc *read_json(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    char *t = NULL;
    long n = fseek(f, 0, SEEK_END) ? -1 : ftell(f);
    if (n >= 0 && !fseek(f, 0, SEEK_SET) && (t = malloc((size_t)n + 1)) &&
        fread(t, 1, (size_t)n, f) != (size_t)n) {
        free(t);
        t = NULL;
    }
    fclose(f);
    yyjson_doc *d = t ? yyjson_read(t, (size_t)n, 0) : NULL;
    free(t);
    return d;
}

static int write_json(const char *path, yyjson_mut_val *v)
{
    size_t n;
    char *t = yyjson_mut_val_write(v, 0, &n);
    if (!t)
        return -1;
    char tmp[4400];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    int ok = f && fwrite(t, 1, n, f) == n;
    if (f && fclose(f) != 0)
        ok = 0;
    free(t);
    if (!ok || rename(tmp, path) != 0) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

static int64_t memory_of(const char *m)
{
    if (!m || !strcmp(m, "1g"))
        return (int64_t)1 << 30;
    if (!strcmp(m, "4g"))
        return (int64_t)4 << 30;
    if (!strcmp(m, "16g"))
        return (int64_t)16 << 30;
    if (!strcmp(m, "64g"))
        return (int64_t)64 << 30;
    return -1;
}

/* the container's meta, running (mu held): its document, or NULL */
static yyjson_doc *meta_of(const char *cid)
{
    if (!id_ok(cid, "cntr_"))
        return NULL;
    char p[4600];
    path_of(p, sizeof p, cid, "meta.json");
    return read_json(p);
}

static int is_running(yyjson_doc *m)
{
    return yyjson_equals_str(yyjson_obj_get(yyjson_doc_get_root(m), "status"),
                             "running");
}

/* last_active_at now (mu held) */
static void touch(const char *cid)
{
    yyjson_doc *m = meta_of(cid);
    if (!m)
        return;
    yyjson_mut_doc *d = yyjson_doc_mut_copy(m, NULL);
    yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
    yyjson_mut_obj_remove_key(o, "last_active_at");
    yyjson_mut_obj_add_int(d, o, "last_active_at", (int64_t)time(NULL));
    char p[4600];
    path_of(p, sizeof p, cid, "meta.json");
    write_json(p, o);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(m);
}

/* ---- the index of a container's files ---- */

struct walk {
    char rel[MAX_FILES][300];
    int64_t bytes[MAX_FILES], mtime[MAX_FILES];
    int n;
};

static void walk_dir(int dir, const char *prefix, struct walk *w, int depth)
{
    DIR *d = fdopendir(dir);
    if (!d) {
        close(dir);
        return;
    }
    struct dirent *e;
    while ((e = readdir(d)) && w->n < MAX_FILES) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        struct stat st;
        if (fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0)
            continue;
        char rel[300];
        if ((size_t)snprintf(rel, sizeof rel, "%s%s", prefix, e->d_name) >=
            sizeof rel)
            continue;
        if (S_ISREG(st.st_mode)) {
            snprintf(w->rel[w->n], sizeof w->rel[0], "%s", rel);
            w->bytes[w->n] = st.st_size;
            w->mtime[w->n] =
                (int64_t)st.st_mtim.tv_sec * 1000000000 + st.st_mtim.tv_nsec;
            w->n++;
        } else if (S_ISDIR(st.st_mode) && depth < 6) {
            int sub = openat(dirfd(d), e->d_name,
                             O_DIRECTORY | O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
            if (sub >= 0) {
                char pre[300];
                if ((size_t)snprintf(pre, sizeof pre, "%s/", rel) >=
                    sizeof pre) {
                    close(sub);
                    continue;
                }
                walk_dir(sub, pre, w, depth + 1);
            }
        }
    }
    closedir(d);
}

/* The index brought up to what is in data/ (mu held): new files get ids
   (source as given), gone ones go. Those new or changed since the index
   was last brought up are listed in changed (ids), images first. 0, or
   -1. */
static int sync_index(const char *cid, const char *source,
                      yyjson_mut_doc *out_d, yyjson_mut_val *changed)
{
    char p[4600];
    path_of(p, sizeof p, cid, "data");
    int dir = open(p, O_DIRECTORY | O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (dir < 0)
        return -1;
    struct walk *w = calloc(1, sizeof(*w));
    if (!w) {
        close(dir);
        return -1;
    }
    walk_dir(dir, "", w, 0);
    path_of(p, sizeof p, cid, "files.json");
    yyjson_doc *old = read_json(p);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *arr = yyjson_mut_arr(d);
    int64_t now = (int64_t)time(NULL);
    for (int i = 0; i < w->n; i++) {
        yyjson_val *prev = NULL, *v;
        size_t k, m;
        yyjson_arr_foreach(yyjson_doc_get_root(old), k, m, v)
        {
            if (yyjson_equals_str(yyjson_obj_get(v, "path"), w->rel[i]))
                prev = v;
        }
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, arr);
        char id[48];
        const char *pid = yyjson_get_str(yyjson_obj_get(prev, "id"));
        if (!pid) {
            srv_new_id(id, sizeof id, "cfile_");
            pid = id;
        }
        yyjson_mut_obj_add_strcpy(d, o, "id", pid);
        yyjson_mut_obj_add_strcpy(d, o, "path", w->rel[i]);
        yyjson_mut_obj_add_int(d, o, "bytes", w->bytes[i]);
        yyjson_mut_obj_add_int(d, o, "mtime", w->mtime[i]);
        yyjson_val *ca = yyjson_obj_get(prev, "created_at");
        yyjson_mut_obj_add_int(d, o, "created_at",
                               ca ? yyjson_get_sint(ca) : now);
        const char *src = yyjson_get_str(yyjson_obj_get(prev, "source"));
        yyjson_mut_obj_add_strcpy(d, o, "source", src ? src : source);
        int changed_now =
            !prev ||
            yyjson_get_sint(yyjson_obj_get(prev, "mtime")) != w->mtime[i] ||
            yyjson_get_sint(yyjson_obj_get(prev, "bytes")) != w->bytes[i];
        if (changed && changed_now) {
            size_t l = strlen(w->rel[i]);
            int image = l > 4 && !strcmp(w->rel[i] + l - 4, ".png");
            yyjson_mut_val *c = yyjson_mut_obj(out_d);
            yyjson_mut_obj_add_strcpy(out_d, c, "id", pid);
            yyjson_mut_obj_add_strcpy(out_d, c, "path", w->rel[i]);
            yyjson_mut_obj_add_bool(out_d, c, "image", image);
            if (image)
                yyjson_mut_arr_prepend(changed, c);
            else
                yyjson_mut_arr_append(changed, c);
        }
    }
    int rc = write_json(p, arr);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(old);
    free(w);
    return rc;
}

/* ---- the sessions ---- */

struct kernel {
    char cid[64];
    struct janas_sandbox sb;
    pthread_mutex_t run; /* one piece of code at a time */
    int refs;            /* the list's, and each piece of code's (mu) */
    struct kernel *next;
};

static struct kernel *kernels;

static void kernel_unref(struct kernel *k)
{
    if (--k->refs > 0)
        return;
    janas_sandbox_kill(&k->sb);
    pthread_mutex_destroy(&k->run);
    free(k);
}

/* the session ended (mu held): out of the list, its process killed (a
   piece of code running sees its end), freed by the last user */
static void kernel_drop(struct kernel *k)
{
    for (struct kernel **p = &kernels; *p; p = &(*p)->next)
        if (*p == k) {
            *p = k->next;
            if (k->sb.pid > 0)
                kill(k->sb.pid, SIGKILL);
            kernel_unref(k);
            return;
        }
}

/* The container's session, started when it has none (mu held): NULL with
   the reason. */
static struct kernel *kernel_of(const char *cid, int64_t mem, char *why,
                                size_t n)
{
    for (struct kernel *k = kernels; k; k = k->next)
        if (!strcmp(k->cid, cid))
            return k;
    struct kernel *k = calloc(1, sizeof(*k));
    if (!k) {
        snprintf(why, n, "out of memory");
        return NULL;
    }
    snprintf(k->cid, sizeof k->cid, "%s", cid);
    pthread_mutex_init(&k->run, NULL);
    char data[4400];
    path_of(data, sizeof data, cid, "data");
    struct janas_sandbox_cfg c = {
        .data_dir = data,
        .mem_bytes = (uint64_t)mem + ((uint64_t)1 << 30), /* + the runtime */
        .file_bytes = (uint64_t)512 << 20,
        .tmp_bytes = (uint64_t)256 << 20};
    const char *py =
        server->cfg.code_python ? server->cfg.code_python : "/usr/bin/python3";
    char *argv[] = {(char *)py, "-I", "-c", (char *)JANAS_PY_DRIVER, NULL};
    char *envp[] = {"PATH=/usr/local/bin:/usr/bin:/bin",
                    "HOME=/tmp",
                    "LANG=C.UTF-8",
                    "MPLBACKEND=Agg",
                    "MPLCONFIGDIR=/tmp/mpl",
                    "OPENBLAS_NUM_THREADS=4",
                    "OMP_NUM_THREADS=4",
                    "PYTHONDONTWRITEBYTECODE=1",
                    NULL};
    if (janas_sandbox_spawn(&c, argv, envp, &k->sb, why, n) != 0) {
        pthread_mutex_destroy(&k->run);
        free(k);
        return NULL;
    }
    k->refs = 1;
    k->next = kernels;
    kernels = k;
    return k;
}

void srv_cell_free(struct srv_cell *c)
{
    free(c->logs);
    free(c->error);
    free(c->file_id);
    free(c->path);
    memset(c, 0, sizeof(*c));
}

int srv_container_run(struct srv *s, const char *id, const char *code,
                      struct srv_cell *out)
{
    (void)s;
    memset(out, 0, sizeof(*out));
    char why[300] = "";
    pthread_mutex_lock(&mu);
    yyjson_doc *m = meta_of(id);
    if (!m || !is_running(m)) {
        yyjson_doc_free(m);
        pthread_mutex_unlock(&mu);
        return -1;
    }
    int64_t mem = memory_of(
        yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(m), "memory_limit")));
    yyjson_doc_free(m);
    touch(id);
    sync_index(id, "user", NULL, NULL); /* what was there before */
    struct kernel *k =
        kernel_of(id, mem > 0 ? mem : (int64_t)1 << 30, why, sizeof why);
    if (k)
        k->refs++;
    pthread_mutex_unlock(&mu);
    if (!k) {
        out->error = strdup(why);
        return 0;
    }
    pthread_mutex_lock(&k->run);
    struct janas_buf raw = {0};
    char *ans =
        janas_py_exchange(&k->sb, code, CELL_SECONDS, &raw, why, sizeof why);
    pthread_mutex_unlock(&k->run);
    pthread_mutex_lock(&mu);
    if (!ans) {         /* the session is over */
        kernel_drop(k); /* if still listed */
        out->error = strdup(why);
    } else {
        yyjson_doc *ad = yyjson_read(ans, strlen(ans), 0);
        yyjson_val *a = yyjson_doc_get_root(ad);
        const char *logs = yyjson_get_str(yyjson_obj_get(a, "logs"));
        const char *err = yyjson_get_str(yyjson_obj_get(a, "error"));
        struct janas_buf lb = {0};
        if (raw.n)
            janas_buf_put(&lb, raw.p, raw.n);
        if (logs)
            janas_buf_puts(&lb, logs);
        janas_buf_put(&lb, "", 1);
        out->logs = lb.oom ? NULL : lb.p;
        if (lb.oom)
            janas_buf_free(&lb);
        if (err)
            out->error = strdup(err);
        yyjson_doc_free(ad);
        free(ans);
    }
    janas_buf_free(&raw);
    /* the files it wrote */
    yyjson_mut_doc *cd = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *changed = yyjson_mut_arr(cd);
    sync_index(id, "assistant", cd, changed);
    touch(id);
    size_t nc = yyjson_mut_arr_size(changed);
    if (nc) {
        out->file_id = calloc(nc, sizeof(*out->file_id));
        out->path = calloc(nc, sizeof(*out->path));
    }
    size_t i, n;
    yyjson_mut_val *c;
    yyjson_mut_arr_foreach(changed, i, n, c)
    {
        if (!out->file_id || !out->path)
            break;
        snprintf(out->file_id[out->n_files], sizeof out->file_id[0], "%s",
                 or_empty(yyjson_mut_get_str(yyjson_mut_obj_get(c, "id"))));
        snprintf(out->path[out->n_files], sizeof out->path[0], "/mnt/data/%s",
                 or_empty(yyjson_mut_get_str(yyjson_mut_obj_get(c, "path"))));
        out->n_files++;
        if (yyjson_mut_get_bool(yyjson_mut_obj_get(c, "image")))
            out->n_images++;
    }
    yyjson_mut_doc_free(cd);
    kernel_unref(k);
    pthread_mutex_unlock(&mu);
    return 0;
}

const char *srv_containers_off(void)
{
    return on ? NULL : off_why;
}

/* ---- containers ---- */

/* A file of the Files API copied into the container's data (mu held):
   NULL, or the reason. */
static const char *copy_file(struct srv *s, const char *cid, const char *fid)
{
    struct srv_file f;
    if (!fid || srv_files_get(s->files, fid, &f) != 0)
        return "A file_id names no file uploaded here.";
    /* its name, never a path */
    char name[256];
    const char *base = strrchr(f.filename, '/');
    snprintf(name, sizeof name, "%s", base ? base + 1 : f.filename);
    if (!name[0] || name[0] == '.')
        snprintf(name, sizeof name, "%s", fid);
    int in = srv_files_open_content(s->files, fid);
    int out =
        in >= 0 ? open_under(cid, name, O_WRONLY | O_CREAT | O_TRUNC) : -1;
    const char *why = NULL;
    if (in < 0 || out < 0)
        why = "The file could not be copied into the container.";
    char buf[65536];
    ssize_t k;
    while (!why && (k = read(in, buf, sizeof buf)) > 0)
        if (write(out, buf, (size_t)k) != k)
            why = "The file could not be copied into the container.";
    if (in >= 0)
        close(in);
    if (out >= 0)
        close(out);
    return why;
}

/* A new container: 0 with its id, -1 with the reason. */
static int make(struct srv *s, const char *name, yyjson_val *file_ids,
                yyjson_val *expires, const char *memory, char *id, size_t cap,
                const char **why)
{
    int64_t minutes = DEFAULT_MINUTES;
    yyjson_val *mn = yyjson_obj_get(expires, "minutes");
    if (expires && !yyjson_is_null(expires)) {
        if (!yyjson_equals_str(yyjson_obj_get(expires, "anchor"),
                               "last_active_at") ||
            !yyjson_is_int(mn) || yyjson_get_sint(mn) < 1 ||
            yyjson_get_sint(mn) > 20) {
            *why = "'expires_after' is {anchor: last_active_at, minutes} "
                   "with minutes from 1 to 20.";
            return -1;
        }
        minutes = yyjson_get_sint(mn);
    }
    if (memory && memory_of(memory) < 0) {
        *why = "'memory_limit' is 1g, 4g, 16g or 64g.";
        return -1;
    }
    if (file_ids &&
        (!yyjson_is_arr(file_ids) || yyjson_arr_size(file_ids) > 100)) {
        *why = "'file_ids' is a list of at most 100 files.";
        return -1;
    }
    srv_new_id(id, cap, "cntr_");
    char p[4600];
    snprintf(p, sizeof p, "%s/%s", work, id);
    pthread_mutex_lock(&mu);
    *why = NULL;
    if (mkdir(p, 0700) != 0) {
        *why = "The container could not be made.";
    } else {
        path_of(p, sizeof p, id, "data");
        if (mkdir(p, 0700) != 0)
            *why = "The container could not be made.";
    }
    size_t i, n;
    yyjson_val *v;
    yyjson_arr_foreach(file_ids, i, n, v)
    {
        if (!*why)
            *why = copy_file(s, id, yyjson_get_str(v));
    }
    if (!*why) {
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *o = yyjson_mut_obj(d);
        int64_t now = (int64_t)time(NULL);
        yyjson_mut_obj_add_strcpy(d, o, "id", id);
        yyjson_mut_obj_add_str(d, o, "object", "container");
        yyjson_mut_obj_add_int(d, o, "created_at", now);
        yyjson_mut_obj_add_str(d, o, "status", "running");
        yyjson_mut_val *ea = yyjson_mut_obj_add_obj(d, o, "expires_after");
        yyjson_mut_obj_add_str(d, ea, "anchor", "last_active_at");
        yyjson_mut_obj_add_int(d, ea, "minutes", minutes);
        yyjson_mut_obj_add_int(d, o, "last_active_at", now);
        yyjson_mut_obj_add_strcpy(d, o, "memory_limit", memory ? memory : "1g");
        yyjson_mut_obj_add_strcpy(d, o, "name", name ? name : "");
        path_of(p, sizeof p, id, "meta.json");
        if (write_json(p, o) != 0 || sync_index(id, "user", NULL, NULL) != 0)
            *why = "The container could not be made.";
        yyjson_mut_doc_free(d);
    }
    if (*why) {
        snprintf(p, sizeof p, "%s/%s", work, id);
        rm_tree(p);
    }
    pthread_mutex_unlock(&mu);
    return *why ? -1 : 0;
}

int srv_container_for(struct srv *s, yyjson_val *spec, char *id, size_t cap,
                      const char **why)
{
    if (yyjson_is_str(spec) && strcmp(yyjson_get_str(spec), "auto")) {
        pthread_mutex_lock(&mu);
        yyjson_doc *m = meta_of(yyjson_get_str(spec));
        int ok = m && is_running(m);
        yyjson_doc_free(m);
        pthread_mutex_unlock(&mu);
        if (!ok) {
            *why = "The container named is not here, or has expired.";
            return -1;
        }
        snprintf(id, cap, "%s", yyjson_get_str(spec));
        return 0;
    }
    if (spec && !yyjson_is_str(spec) &&
        !yyjson_equals_str(yyjson_obj_get(spec, "type"), "auto")) {
        *why = "A code_interpreter's container is an id, or {type: auto}.";
        return -1;
    }
    return make(s, "code_interpreter", yyjson_obj_get(spec, "file_ids"), NULL,
                yyjson_get_str(yyjson_obj_get(spec, "memory_limit")), id, cap,
                why);
}

int srv_container_add_file(struct srv *s, const char *cid, const char *fid,
                           const char **why)
{
    pthread_mutex_lock(&mu);
    yyjson_doc *m = meta_of(cid);
    int ok = m && is_running(m);
    yyjson_doc_free(m);
    *why = ok ? copy_file(s, cid, fid) : "The container is gone.";
    if (!*why)
        sync_index(cid, "user", NULL, NULL);
    pthread_mutex_unlock(&mu);
    return *why ? -1 : 0;
}

int srv_container_file_open(const char *cid, const char *cfid, char *name,
                            size_t cap)
{
    char p[4600];
    int fd = -1;
    pthread_mutex_lock(&mu);
    if (id_ok(cid, "cntr_")) {
        path_of(p, sizeof p, cid, "files.json");
        yyjson_doc *x = read_json(p);
        size_t i, n;
        yyjson_val *v;
        yyjson_arr_foreach(yyjson_doc_get_root(x), i, n, v)
        {
            const char *rel = yyjson_get_str(yyjson_obj_get(v, "path"));
            if (fd < 0 && rel &&
                yyjson_equals_str(yyjson_obj_get(v, "id"), cfid)) {
                fd = open_under(cid, rel, O_RDONLY);
                const char *b = strrchr(rel, '/');
                snprintf(name, cap, "%s", b ? b + 1 : rel);
            }
        }
        yyjson_doc_free(x);
    }
    pthread_mutex_unlock(&mu);
    struct stat st;
    if (fd >= 0 && (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode))) {
        close(fd);
        fd = -1;
    }
    return fd;
}

int srv_container_file_name(struct srv *s, const char *cid, const char *fid,
                            char *name, size_t cap)
{
    (void)s;
    char p[4600];
    int rc = -1;
    pthread_mutex_lock(&mu);
    if (id_ok(cid, "cntr_")) {
        path_of(p, sizeof p, cid, "files.json");
        yyjson_doc *x = read_json(p);
        size_t i, n;
        yyjson_val *v;
        yyjson_arr_foreach(yyjson_doc_get_root(x), i, n, v)
        {
            if (yyjson_equals_str(yyjson_obj_get(v, "id"), fid)) {
                const char *rel = yyjson_get_str(yyjson_obj_get(v, "path"));
                const char *b = rel ? strrchr(rel, '/') : NULL;
                snprintf(name, cap, "%s", b ? b + 1 : rel ? rel : "");
                rc = 0;
            }
        }
        yyjson_doc_free(x);
    }
    pthread_mutex_unlock(&mu);
    return rc;
}

/* ---- expiry ---- */

static void expire(const char *cid)
{
    char p[4600];
    for (struct kernel *k = kernels; k; k = k->next)
        if (!strcmp(k->cid, cid)) {
            kernel_drop(k); /* a piece of code running ends with it */
            break;
        }
    path_of(p, sizeof p, cid, "data");
    rm_tree(p);
    path_of(p, sizeof p, cid, "files.json");
    unlink(p);
    yyjson_doc *m = meta_of(cid);
    if (m) {
        yyjson_mut_doc *d = yyjson_doc_mut_copy(m, NULL);
        yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
        yyjson_mut_obj_remove_key(o, "status");
        yyjson_mut_obj_add_str(d, o, "status", "expired");
        path_of(p, sizeof p, cid, "meta.json");
        write_json(p, o);
        yyjson_mut_doc_free(d);
        yyjson_doc_free(m);
    }
}

static void *reaper_main(void *arg)
{
    (void)arg;
    pthread_mutex_lock(&mu);
    while (!stopping) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 30;
        pthread_cond_timedwait(&reaper_cv, &mu, &ts);
        if (stopping)
            break;
        DIR *d = opendir(work);
        struct dirent *e;
        int64_t now = (int64_t)time(NULL);
        while (d && (e = readdir(d))) {
            if (!id_ok(e->d_name, "cntr_"))
                continue;
            yyjson_doc *m = meta_of(e->d_name);
            yyjson_val *o = yyjson_doc_get_root(m);
            int64_t last = yyjson_get_sint(yyjson_obj_get(o, "last_active_at"));
            int64_t mins = yyjson_get_sint(
                yyjson_obj_get(yyjson_obj_get(o, "expires_after"), "minutes"));
            int busy = 0; /* a piece of code running: not idle */
            for (struct kernel *k = kernels; k; k = k->next)
                busy |= !strcmp(k->cid, e->d_name) && k->refs > 1;
            if (m && is_running(m) && !busy && now > last + mins * 60)
                expire(e->d_name);
            yyjson_doc_free(m);
        }
        if (d)
            closedir(d);
    }
    pthread_mutex_unlock(&mu);
    return NULL;
}

int srv_containers_start(struct srv *s)
{
    server = s;
    snprintf(work, sizeof work, "%s/.containers", srv_files_dir(s->files));
    if (mkdir(work, 0700) != 0 && errno != EEXIST)
        return -1;
    if (!s->cfg.code_interpreter)
        return 0;
    char why[256];
    const char *py =
        s->cfg.code_python ? s->cfg.code_python : "/usr/bin/python3";
    if (!janas_sandbox_available(why, sizeof why)) {
        snprintf(off_why, sizeof off_why,
                 "This server cannot run code here: %s.", why);
        return -1;
    }
    if (access(py, X_OK) != 0) {
        snprintf(off_why, sizeof off_why,
                 "This server cannot run code: there is no %s.", py);
        return -1;
    }
    on = 1;
    stopping = 0;
    if (pthread_create(&reaper, NULL, reaper_main, NULL) == 0)
        running = 1;
    return 0;
}

void srv_containers_stop(void)
{
    pthread_mutex_lock(&mu);
    stopping = 1;
    pthread_cond_broadcast(&reaper_cv);
    pthread_mutex_unlock(&mu);
    if (running)
        pthread_join(reaper, NULL);
    running = 0;
    pthread_mutex_lock(&mu);
    while (kernels)
        kernel_drop(kernels);
    pthread_mutex_unlock(&mu);
}

/* ---- the API ---- */

static int bad(struct srv_req *r, const char *msg)
{
    return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                           "%s", msg);
}

static int not_found(struct srv_req *r, const char *what, const char *id)
{
    return srv_reply_error(r, 404, "invalid_request_error", "not_found",
                           "No %s found with id '%s'.", what, id);
}

static int refused(struct srv_req *r)
{
    return srv_reply_error(r, 400, "invalid_request_error", "unsupported_value",
                           "%s", off_why);
}

static int reply_doc_val(struct srv_req *r, yyjson_val *v)
{
    size_t n;
    char *t = yyjson_val_write(v, 0, &n);
    return srv_reply_json(r, 200, t, t ? n : 0);
}

/* A page of objects (oldest first) as OpenAI's lists answer. */
static int page(struct srv_req *r, yyjson_mut_doc *d, yyjson_mut_val *items)
{
    const char *l = srv_query(r, "limit"), *o = srv_query(r, "order");
    const char *after = srv_query(r, "after");
    long limit = l ? strtol(l, NULL, 10) : 20;
    if (limit < 1 || limit > 100)
        return bad(r, "'limit' is a number from 1 to 100.");
    int desc = !o || !strcmp(o, "desc");
    size_t n = yyjson_mut_arr_size(items);
    yyjson_mut_val *root = yyjson_mut_obj(d), *data = yyjson_mut_arr(d);
    long shown = 0;
    int more = 0, past = !after;
    const char *first = NULL, *last = NULL;
    for (size_t k = 0; k < n; k++) {
        yyjson_mut_val *v = yyjson_mut_arr_get(items, desc ? n - 1 - k : k);
        const char *id = yyjson_mut_get_str(yyjson_mut_obj_get(v, "id"));
        if (!past) {
            past = id && !strcmp(id, after);
            continue;
        }
        if (shown == limit) {
            more = 1;
            break;
        }
        yyjson_mut_arr_append(data, yyjson_mut_val_mut_copy(d, v));
        if (!shown)
            first = id;
        last = id;
        shown++;
    }
    yyjson_mut_obj_add_str(d, root, "object", "list");
    yyjson_mut_obj_add_val(d, root, "data", data);
    if (first) {
        yyjson_mut_obj_add_strcpy(d, root, "first_id", first);
        yyjson_mut_obj_add_strcpy(d, root, "last_id", last);
    } else {
        yyjson_mut_obj_add_null(d, root, "first_id");
        yyjson_mut_obj_add_null(d, root, "last_id");
    }
    yyjson_mut_obj_add_bool(d, root, "has_more", more);
    size_t len;
    char *t = yyjson_mut_val_write(root, 0, &len);
    return srv_reply_json(r, 200, t, t ? len : 0);
}

int srv_containers_create(struct srv_req *r)
{
    if (!on)
        return refused(r);
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *name = yyjson_get_str(yyjson_obj_get(root, "name"));
    char id[48];
    const char *why = NULL;
    if (!yyjson_is_obj(root) || !name)
        why = "A container has a name.";
    else
        make(r->srv, name, yyjson_obj_get(root, "file_ids"),
             yyjson_obj_get(root, "expires_after"),
             yyjson_get_str(yyjson_obj_get(root, "memory_limit")), id,
             sizeof id, &why);
    yyjson_doc_free(doc);
    if (why)
        return bad(r, why);
    pthread_mutex_lock(&mu);
    yyjson_doc *m = meta_of(id);
    pthread_mutex_unlock(&mu);
    int ret = reply_doc_val(r, yyjson_doc_get_root(m));
    yyjson_doc_free(m);
    return ret;
}

int srv_containers_list(struct srv_req *r)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *items = yyjson_mut_arr(d);
    pthread_mutex_lock(&mu);
    DIR *dir = opendir(work);
    struct dirent *e;
    while (dir && (e = readdir(dir))) {
        yyjson_doc *m = id_ok(e->d_name, "cntr_") ? meta_of(e->d_name) : NULL;
        if (m) {
            /* oldest first: by created_at */
            yyjson_mut_val *v = yyjson_val_mut_copy(d, yyjson_doc_get_root(m));
            int64_t at =
                yyjson_mut_get_sint(yyjson_mut_obj_get(v, "created_at"));
            size_t k = 0, n = yyjson_mut_arr_size(items);
            while (k < n &&
                   yyjson_mut_get_sint(yyjson_mut_obj_get(
                       yyjson_mut_arr_get(items, k), "created_at")) <= at)
                k++;
            yyjson_mut_arr_insert(items, v, k);
        }
        yyjson_doc_free(m);
    }
    if (dir)
        closedir(dir);
    pthread_mutex_unlock(&mu);
    int ret = page(r, d, items);
    yyjson_mut_doc_free(d);
    return ret;
}

int srv_containers_get(struct srv_req *r)
{
    pthread_mutex_lock(&mu);
    yyjson_doc *m = meta_of(r->param[0]);
    pthread_mutex_unlock(&mu);
    if (!m)
        return not_found(r, "container", r->param[0]);
    int ret = reply_doc_val(r, yyjson_doc_get_root(m));
    yyjson_doc_free(m);
    return ret;
}

int srv_containers_delete(struct srv_req *r)
{
    pthread_mutex_lock(&mu);
    yyjson_doc *m = meta_of(r->param[0]);
    if (m) {
        expire(r->param[0]);
        char p[4600];
        snprintf(p, sizeof p, "%s/%s", work, r->param[0]);
        rm_tree(p);
    }
    pthread_mutex_unlock(&mu);
    yyjson_doc_free(m);
    if (!m)
        return not_found(r, "container", r->param[0]);
    char out[512];
    snprintf(out, sizeof out,
             "{\"id\":\"%s\",\"object\":\"container.deleted\",\"deleted\":"
             "true}",
             r->param[0]);
    char *t = strdup(out);
    return srv_reply_json(r, 200, t, t ? strlen(t) : 0);
}

/* A container's file as OpenAI writes it. */
static yyjson_mut_val *file_obj(yyjson_mut_doc *d, const char *cid,
                                yyjson_val *v)
{
    yyjson_mut_val *o = yyjson_mut_obj(d);
    char path[320];
    snprintf(path, sizeof path, "/mnt/data/%s",
             or_empty(yyjson_get_str(yyjson_obj_get(v, "path"))));
    yyjson_mut_obj_add_strcpy(d, o, "id",
                              yyjson_get_str(yyjson_obj_get(v, "id")));
    yyjson_mut_obj_add_str(d, o, "object", "container.file");
    yyjson_mut_obj_add_int(d, o, "created_at",
                           yyjson_get_sint(yyjson_obj_get(v, "created_at")));
    yyjson_mut_obj_add_int(d, o, "bytes",
                           yyjson_get_sint(yyjson_obj_get(v, "bytes")));
    yyjson_mut_obj_add_strcpy(d, o, "container_id", cid);
    yyjson_mut_obj_add_strcpy(d, o, "path", path);
    yyjson_mut_obj_add_strcpy(d, o, "source",
                              yyjson_get_str(yyjson_obj_get(v, "source")));
    return o;
}

/* The container's index, up to date (mu held): NULL when it is not there
   or has expired. */
static yyjson_doc *index_of(const char *cid)
{
    yyjson_doc *m = meta_of(cid);
    int ok = m && is_running(m);
    yyjson_doc_free(m);
    if (!ok || sync_index(cid, "assistant", NULL, NULL) != 0)
        return NULL;
    char p[4600];
    path_of(p, sizeof p, cid, "files.json");
    return read_json(p);
}

int srv_container_files_create(struct srv_req *r)
{
    const char *cid = r->param[0];
    if (!on)
        return refused(r);
    const char *ct = r->conn ? MHD_lookup_connection_value(
                                   r->conn, MHD_HEADER_KIND, "Content-Type")
                             : NULL;
    const char *why = NULL;
    char name[256] = "";
    pthread_mutex_lock(&mu);
    yyjson_doc *m = meta_of(cid);
    int ok = m && is_running(m);
    yyjson_doc_free(m);
    if (!ok) {
        pthread_mutex_unlock(&mu);
        return not_found(r, "container", cid);
    }
    if (ct && !strncmp(ct, "multipart/", 10)) {
        struct srv_part parts[4];
        char err[200];
        int n = srv_multipart_parse(ct, r->body, r->n_body, parts, 4, err,
                                    sizeof err);
        const struct srv_part *f =
            n > 0 ? srv_multipart_get(parts, n, "file") : NULL;
        if (!f || !f->has_filename) {
            why = "'file' is the file's bytes, with a name.";
        } else {
            const char *b = strrchr(f->filename, '/');
            snprintf(name, sizeof name, "%s", b ? b + 1 : f->filename);
            int fd = name[0] && name[0] != '.'
                         ? open_under(cid, name, O_WRONLY | O_CREAT | O_TRUNC)
                         : -1;
            if (fd < 0 || write(fd, f->data, f->n) != (ssize_t)f->n)
                why = "The file could not be written into the container.";
            if (fd >= 0)
                close(fd);
        }
    } else {
        yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
        const char *fid =
            yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(doc), "file_id"));
        struct srv_file f;
        if (fid && srv_files_get(r->srv->files, fid, &f) == 0) {
            const char *b = strrchr(f.filename, '/');
            snprintf(name, sizeof name, "%s", b ? b + 1 : f.filename);
        }
        why = copy_file(r->srv, cid, fid);
        yyjson_doc_free(doc);
    }
    if (!why) /* the client's */
        sync_index(cid, "user", NULL, NULL);
    yyjson_doc *x = why ? NULL : index_of(cid);
    pthread_mutex_unlock(&mu);
    if (why)
        return bad(r, why);
    int ret = -1;
    size_t i, n;
    yyjson_val *v;
    yyjson_arr_foreach(yyjson_doc_get_root(x), i, n, v)
    {
        if (ret < 0 && yyjson_equals_str(yyjson_obj_get(v, "path"), name)) {
            yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
            size_t len;
            char *t = yyjson_mut_val_write(file_obj(d, cid, v), 0, &len);
            ret = srv_reply_json(r, 200, t, t ? len : 0);
            yyjson_mut_doc_free(d);
        }
    }
    yyjson_doc_free(x);
    return ret < 0 ? srv_reply_error(r, 500, "server_error", NULL,
                                     "the file is not in the container")
                   : ret;
}

int srv_container_files_list(struct srv_req *r)
{
    pthread_mutex_lock(&mu);
    yyjson_doc *x = index_of(r->param[0]);
    pthread_mutex_unlock(&mu);
    if (!x)
        return not_found(r, "container", r->param[0]);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *items = yyjson_mut_arr(d);
    size_t i, n;
    yyjson_val *v;
    yyjson_arr_foreach(yyjson_doc_get_root(x), i, n, v)
        yyjson_mut_arr_append(items, file_obj(d, r->param[0], v));
    yyjson_doc_free(x);
    int ret = page(r, d, items);
    yyjson_mut_doc_free(d);
    return ret;
}

/* the file's entry in x, or NULL */
static yyjson_val *entry(yyjson_doc *x, const char *fid)
{
    size_t i, n;
    yyjson_val *v;
    yyjson_arr_foreach(yyjson_doc_get_root(x), i, n, v)
    {
        if (yyjson_equals_str(yyjson_obj_get(v, "id"), fid))
            return v;
    }
    return NULL;
}

int srv_container_files_get(struct srv_req *r)
{
    pthread_mutex_lock(&mu);
    yyjson_doc *x = index_of(r->param[0]);
    pthread_mutex_unlock(&mu);
    if (!x)
        return not_found(r, "container", r->param[0]);
    yyjson_val *v = entry(x, r->param[1]);
    int ret;
    if (!v) {
        ret = not_found(r, "container file", r->param[1]);
    } else {
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        size_t len;
        char *t = yyjson_mut_val_write(file_obj(d, r->param[0], v), 0, &len);
        ret = srv_reply_json(r, 200, t, t ? len : 0);
        yyjson_mut_doc_free(d);
    }
    yyjson_doc_free(x);
    return ret;
}

int srv_container_files_delete(struct srv_req *r)
{
    pthread_mutex_lock(&mu);
    yyjson_doc *x = index_of(r->param[0]);
    yyjson_val *v = x ? entry(x, r->param[1]) : NULL;
    int gone = 0;
    if (v) {
        /* removed in its directory reached without links: never what a
           link points to */
        const char *leaf;
        int dir = parent_under(
            r->param[0], yyjson_get_str(yyjson_obj_get(v, "path")), &leaf);
        gone = dir >= 0 && unlinkat(dir, leaf, 0) == 0;
        if (dir >= 0)
            close(dir);
        sync_index(r->param[0], "assistant", NULL, NULL);
    }
    pthread_mutex_unlock(&mu);
    yyjson_doc_free(x);
    if (!x)
        return not_found(r, "container", r->param[0]);
    if (!gone)
        return not_found(r, "container file", r->param[1]);
    char out[512];
    snprintf(out, sizeof out,
             "{\"id\":\"%s\",\"object\":\"container.file.deleted\","
             "\"deleted\":true}",
             r->param[1]);
    char *t = strdup(out);
    return srv_reply_json(r, 200, t, t ? strlen(t) : 0);
}

int srv_container_files_content(struct srv_req *r)
{
    pthread_mutex_lock(&mu);
    yyjson_doc *x = index_of(r->param[0]);
    yyjson_val *v = x ? entry(x, r->param[1]) : NULL;
    int fd = v ? open_under(r->param[0],
                            yyjson_get_str(yyjson_obj_get(v, "path")), O_RDONLY)
               : -1;
    pthread_mutex_unlock(&mu);
    yyjson_doc_free(x);
    struct stat st;
    if (fd >= 0 && (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode))) {
        close(fd);
        fd = -1;
    }
    if (fd < 0)
        return !x ? not_found(r, "container", r->param[0])
                  : not_found(r, "container file", r->param[1]);
    return srv_reply_fd(r, fd, (uint64_t)st.st_size,
                        "application/octet-stream");
}

#endif
