/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * write.c - the tools that change something, each asked of the user by the
 * client (readOnlyHint false): a file written (files_write), a piece of a
 * file replaced (files_edit), a folder made (files_mkdir). A file is
 * written beside itself and renamed over the old one, so that it is never
 * left half written; a file replaced is copied first under
 * ~/.local/share/janas/files-backup/, and the answer says where. Nothing
 * is deleted, and nothing is replaced unless the call says so.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "files.h"

#define WRITE_MAX (16 << 20) /* the largest file written */

/* mkdir -p of an absolute path */
static int make_dirs(const char *path)
{
    char p[PATH_MAX];
    snprintf(p, sizeof p, "%s", path);
    for (char *s = p + 1; *s; s++)
        if (*s == '/') {
            *s = 0;
            if (mkdir(p, 0755) != 0 && errno != EEXIST)
                return -1;
            *s = '/';
        }
    return mkdir(p, 0755) != 0 && errno != EEXIST ? -1 : 0;
}

/* A copy of path kept under the backup folder, its place there into out
   (empty when there was nothing to keep). 0, or -1. */
static int keep_copy(const struct fl_roots *r, const char *path, char *out,
                     size_t cap)
{
    out[0] = 0;
    int in = open(path, O_RDONLY | O_CLOEXEC);
    if (in < 0)
        return errno == ENOENT ? 0 : -1;
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tm);
    size_t h = strlen(r->home);
    const char *rel =
        !strncmp(path, r->home, h) && path[h] == '/' ? path + h + 1 : path + 1;
    if ((size_t)snprintf(out, cap, "%s/.local/share/janas/files-backup/%s/%s",
                         r->home, stamp, rel) >= cap) {
        close(in);
        return -1;
    }
    char dir[PATH_MAX];
    snprintf(dir, sizeof dir, "%s", out);
    *strrchr(dir, '/') = 0;
    int o = make_dirs(dir) == 0
                ? open(out, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600)
                : -1;
    char buf[65536];
    ssize_t k;
    int bad = o < 0;
    while (!bad && (k = read(in, buf, sizeof buf)) > 0)
        bad = write(o, buf, (size_t)k) != k;
    close(in);
    if (o >= 0)
        bad |= close(o) != 0;
    return bad ? -1 : 0;
}

/* n bytes written to path beside it and renamed over it; the mode of the
   file it replaces, or the usual one. 0, or -1 with errno. */
static int put(const char *path, const char *p, size_t n)
{
    char tmp[PATH_MAX];
    if ((size_t)snprintf(tmp, sizeof tmp, "%s.janas-XXXXXX", path) >=
        sizeof tmp) {
        errno = ENAMETOOLONG;
        return -1;
    }
    int fd = mkstemp(tmp);
    if (fd < 0)
        return -1;
    struct stat st;
    mode_t mode = 0644;
    if (stat(path, &st) == 0)
        mode = st.st_mode & 07777;
    else {
        mode_t um = umask(0);
        umask(um);
        mode &= ~um;
    }
    size_t w = 0;
    while (w < n) {
        ssize_t k = write(fd, p + w, n - w);
        if (k <= 0)
            break;
        w += (size_t)k;
    }
    int bad = w != n || fchmod(fd, mode) != 0 || fsync(fd) != 0;
    int e = errno;
    bad |= close(fd) != 0;
    if (!bad && rename(tmp, path) == 0)
        return 0;
    e = errno ? errno : e;
    unlink(tmp);
    errno = e;
    return -1;
}

/* the folder of a path to make, made where it is allowed */
static int make_parent(const struct fl_roots *r, const char *in, char *why,
                       size_t why_len)
{
    char full[PATH_MAX];
    if (fl_expand(r, in, full, sizeof full) != 0) {
        snprintf(why, why_len, "path too long");
        return -1;
    }
    char *slash = strrchr(full, '/');
    if (!slash || slash == full)
        return 0;
    *slash = 0;
    /* each missing part, from the first, judged before it is made */
    for (char *s = full + 1;; s++) {
        int end = *s == 0;
        if (*s == '/' || end) {
            *s = 0;
            char real[PATH_MAX];
            struct stat st;
            if (stat(full, &st) != 0) {
                if (fl_resolve(r, full, 0, real, sizeof real, why, why_len))
                    return -1;
                if (mkdir(real, 0755) != 0 && errno != EEXIST) {
                    snprintf(why, why_len, "%s: %s", real, strerror(errno));
                    return -1;
                }
            }
            if (end)
                break;
            *s = '/';
        }
    }
    return 0;
}

int fl_tool_write(const struct fl_roots *r, const struct janas_json *args,
                  struct janas_buf *b)
{
    const char *in = fl_arg_str(args, "path");
    const struct janas_json *cv = janas_json_get(args, "content");
    const char *content = janas_json_str(cv);
    if (!in || !content)
        return fl_fail(b, "files_write needs a path and the content.");
    size_t n = cv->n;
    if (n > WRITE_MAX)
        return fl_fail(b, "%zu bytes: more than %d MiB.", n, WRITE_MAX >> 20);
    char path[PATH_MAX], why[PATH_MAX + 200], s[PATH_MAX], kept[PATH_MAX];
    /* the folders it goes in are made unless asked not to: the user is
       asked for the whole path anyway */
    const struct janas_json *md = janas_json_get(args, "make_dirs");
    if (!(md && md->type == JANAS_JSON_FALSE) &&
        make_parent(r, in, why, sizeof why) != 0)
        return fl_fail(b, "%s.", why);
    if (fl_resolve(r, in, 0, path, sizeof path, why, sizeof why))
        return fl_fail(b, "%s.", why);
    fl_short(r, path, s, sizeof s);
    struct stat st;
    int exists = stat(path, &st) == 0;
    if (exists && S_ISDIR(st.st_mode))
        return fl_fail(b, "%s is a folder.", s);
    if (exists && !fl_arg_bool(args, "overwrite"))
        return fl_fail(b,
                       "%s is already there: overwrite true replaces it (a "
                       "copy of it is kept), files_edit changes a piece of "
                       "it.",
                       s);
    if (keep_copy(r, path, kept, sizeof kept) != 0)
        return fl_fail(b,
                       "%s: no copy of it could be kept, so it was not "
                       "replaced.",
                       s);
    if (put(path, content, n) != 0)
        return fl_fail(b, "%s: %s.", s, strerror(errno));
    struct janas_buf t = {0};
    janas_buf_printf(&t, "%s %s: %zu bytes.", s,
                     exists ? "replaced" : "written", n);
    if (kept[0]) {
        char k[PATH_MAX];
        fl_short(r, kept, k, sizeof k);
        janas_buf_printf(&t, " The old one is kept in %s.", k);
    }
    return fl_ok(b, &t);
}

int fl_tool_edit(const struct fl_roots *r, const struct janas_json *args,
                 struct janas_buf *b)
{
    const char *in = fl_arg_str(args, "path");
    const struct janas_json *ov = janas_json_get(args, "old_text");
    const struct janas_json *nv = janas_json_get(args, "new_text");
    const char *old = janas_json_str(ov), *rep = janas_json_str(nv);
    if (!in || !old || !*old || !rep)
        return fl_fail(b, "files_edit needs a path, the old_text there is "
                          "and the new_text.");
    size_t on = ov->n, rn = nv->n;
    char path[PATH_MAX], why[PATH_MAX + 200], s[PATH_MAX], kept[PATH_MAX];
    if (fl_resolve(r, in, 1, path, sizeof path, why, sizeof why))
        return fl_fail(b, "%s.", why);
    fl_short(r, path, s, sizeof s);
    FILE *f = fopen(path, "rb");
    struct stat st;
    if (!f || fstat(fileno(f), &st) != 0 || !S_ISREG(st.st_mode) ||
        st.st_size > WRITE_MAX) {
        if (f)
            fclose(f);
        return fl_fail(b, "%s is not a file that can be edited here.", s);
    }
    size_t n = (size_t)st.st_size;
    char *p = malloc(n + 1);
    size_t got = p ? fread(p, 1, n, f) : 0;
    fclose(f);
    if (!p || got != n) {
        free(p);
        return fl_fail(b, "%s could not be read.", s);
    }
    if (!fl_is_text(p, n)) {
        free(p);
        return fl_fail(b, "%s is not text.", s);
    }
    int count = 0;
    for (const char *a = p; (a = memmem(a, (size_t)(p + n - a), old, on));
         a += on)
        count++;
    int all = fl_arg_bool(args, "all");
    if (count == 0 || (count > 1 && !all)) {
        free(p);
        return count ? fl_fail(b,
                               "The old_text is in %s %d times: give more of "
                               "the text around it, or all true to replace "
                               "every one.",
                               s, count)
                     : fl_fail(b,
                               "The old_text is not in %s, as given "
                               "(spaces and line ends count): read the file "
                               "again.",
                               s);
    }
    struct janas_buf o = {0};
    const char *a = p;
    for (const char *hit; (hit = memmem(a, (size_t)(p + n - a), old, on));
         a = hit + on) {
        janas_buf_put(&o, a, (size_t)(hit - a));
        janas_buf_put(&o, rep, rn);
    }
    janas_buf_put(&o, a, (size_t)(p + n - a));
    free(p);
    int bad = o.oom || keep_copy(r, path, kept, sizeof kept) != 0;
    if (!bad && put(path, o.p ? o.p : "", o.n) != 0)
        bad = 2;
    janas_buf_free(&o);
    if (bad)
        return fl_fail(b, "%s was not changed: %s.", s,
                       bad == 2 ? strerror(errno)
                                : "no copy of it could be "
                                  "kept");
    struct janas_buf t = {0};
    char k[PATH_MAX];
    fl_short(r, kept, k, sizeof k);
    janas_buf_printf(&t, "%s: %d %s replaced. The old file is kept in %s.", s,
                     count, count == 1 ? "piece" : "pieces", k);
    return fl_ok(b, &t);
}

int fl_tool_mkdir(const struct fl_roots *r, const struct janas_json *args,
                  struct janas_buf *b)
{
    const char *in = fl_arg_str(args, "path");
    if (!in)
        return fl_fail(b, "files_mkdir needs a path.");
    char path[PATH_MAX], why[PATH_MAX + 200], s[PATH_MAX];
    if (make_parent(r, in, why, sizeof why) != 0 ||
        fl_resolve(r, in, 0, path, sizeof path, why, sizeof why) != 0)
        return fl_fail(b, "%s.", why);
    fl_short(r, path, s, sizeof s);
    struct stat st;
    if (stat(path, &st) == 0)
        return S_ISDIR(st.st_mode) ? fl_fail(b, "%s is already there.", s)
                                   : fl_fail(b, "%s is a file.", s);
    if (mkdir(path, 0755) != 0)
        return fl_fail(b, "%s: %s.", s, strerror(errno));
    struct janas_buf t = {0};
    janas_buf_printf(&t, "%s/ made.", s);
    return fl_ok(b, &t);
}
