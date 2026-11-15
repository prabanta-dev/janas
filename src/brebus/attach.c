/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * attach.c - a document given to the chat: a file of this computer, copied
 * into the workspace's sources/ or read where it is (then added to the
 * folders read only), or a web address, always downloaded into sources/
 * with a note of where and when it came from (<name>.url.txt), so that the
 * conversation can be read again when the page has changed.
 *
 * Its text is read by janas-docs --text, as the index reads it, and handed
 * to the page: whole into the message when it is short, else the page says
 * where it is and the model searches it.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/mcp_http.h"

#define MAX_DOWNLOAD (64u << 20)
#define MAX_TEXT (256u << 10) /* of text handed to the page */

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

static int fail(struct janas_buf *out, const char *why)
{
    out->n = 0;
    janas_buf_puts(out, "{\"error\": ");
    str(out, why);
    janas_buf_puts(out, "}");
    return -1;
}

/* a file's name made safe: letters, digits, . - _ and spaces, else _ */
static void safe_name(const char *in, char *out, size_t cap)
{
    size_t k = 0;
    for (; *in && k + 1 < cap; in++) {
        unsigned char c = (unsigned char)*in;
        int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                 (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_' ||
                 c == ' ' || c >= 0x80;
        out[k++] = ok ? (char)c : '_';
    }
    out[k] = 0;
    while (out[0] == '.') /* never hidden, never .. */
        out[0] = '_';
    if (!out[0])
        snprintf(out, cap, "document");
}

/* dir/name, or dir/stem (2).ext...: a name not taken; the file made (fd) */
static int fresh(const char *dir, const char *name, char *path, size_t cap)
{
    char stem[300], ext[64] = "";
    snprintf(stem, sizeof stem, "%s", name);
    char *dot = strrchr(stem, '.');
    if (dot && dot != stem && strlen(dot) < sizeof ext) {
        snprintf(ext, sizeof ext, "%s", dot);
        *dot = 0;
    }
    for (int i = 1; i < 1000; i++) {
        if (i == 1)
            snprintf(path, cap, "%s/%s%s", dir, stem, ext);
        else
            snprintf(path, cap, "%s/%s (%d)%s", dir, stem, i, ext);
        int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd >= 0 || errno != EEXIST)
            return fd;
    }
    return -1;
}

/* the text of path by janas-docs --text, at most MAX_TEXT bytes */
static int text_of(struct brebus *b, const char *path, struct janas_buf *t,
                   char *why, size_t wn)
{
    char exe[4300];
    snprintf(exe, sizeof exe, "%s/janas-docs%s", b->bin_dir, b->suffix);
    int fd[2];
    if (janas_pipe(fd) != 0) {
        snprintf(why, wn, "no pipe");
        return -1;
    }
    int err[2];
    if (janas_pipe(err) != 0) {
        close(fd[0]);
        close(fd[1]);
        snprintf(why, wn, "no pipe");
        return -1;
    }
    const char *argv[] = {exe, "--text", path, NULL};
    struct janas_proc p;
    int in = janas_null_fd(0);
    int rc =
        janas_proc_start(&p, argv, NULL, NULL, in, fd[1], err[1], 0, why, wn);
    if (in >= 0)
        close(in);
    close(fd[1]);
    close(err[1]);
    if (rc != 0) {
        close(fd[0]);
        close(err[0]);
        return -1;
    }
    char buf[65536];
    ssize_t r;
    int cut = 0;
    while ((r = read(fd[0], buf, sizeof buf)) > 0) {
        size_t room = t->n < MAX_TEXT ? MAX_TEXT - t->n : 0;
        if ((size_t)r > room)
            cut = 1;
        janas_buf_put(t, buf, (size_t)r < room ? (size_t)r : room);
    }
    close(fd[0]);
    char eb[512];
    ssize_t en = read(err[0], eb, sizeof eb - 1);
    close(err[0]);
    int code = -1;
    janas_proc_wait(&p, -1, &code);
    if (code != 0) {
        eb[en > 0 ? en : 0] = 0;
        char *nl = strchr(eb, '\n');
        if (nl)
            *nl = 0;
        snprintf(why, wn, "%s", en > 0 ? eb : "it could not be read");
        return -1;
    }
    return cut; /* 1: cut at MAX_TEXT */
}

/* {"path", "name", "bytes", "text", "cut"} or {"path", ..., "unread"} */
static void answer(struct brebus *b, const char *path, struct janas_buf *out)
{
    struct stat st;
    stat(path, &st);
    struct janas_buf t = {0};
    char why[600];
    int cut = text_of(b, path, &t, why, sizeof why);
    while (t.n && (t.p[t.n - 1] == '\n' || t.p[t.n - 1] == ' '))
        t.n--;
    const char *slash = strrchr(path, '/');
    janas_buf_puts(out, "{\"path\": ");
    str(out, path);
    janas_buf_puts(out, ", \"name\": ");
    str(out, slash ? slash + 1 : path);
    janas_buf_printf(out, ", \"bytes\": %lld", (long long)st.st_size);
    if (cut < 0) {
        janas_buf_puts(out, ", \"unread\": ");
        str(out, why);
    } else {
        janas_buf_puts(out, ", \"text\": ");
        janas_json_write_str(out, t.p ? t.p : "", t.n);
        janas_buf_printf(out, ", \"cut\": %s", cut ? "true" : "false");
    }
    janas_buf_puts(out, "}");
    janas_buf_free(&t);
}

static int copy_file(const char *from, int to_fd)
{
    int in = open(from, O_RDONLY);
    if (in < 0)
        return -1;
    char buf[1 << 16];
    ssize_t r;
    int bad = 0;
    while (!bad && (r = read(in, buf, sizeof buf)) > 0)
        bad = write(to_fd, buf, (size_t)r) != r;
    close(in);
    return bad || r < 0 ? -1 : 0;
}

/* a web address downloaded into sources/: 0 with its path, or -1 */
static int download(const char *url, const char *dir, char *path, size_t cap,
                    char *why, size_t wn)
{
    static char at[8800];
    snprintf(at, sizeof at, "%s", url);
    struct janas_http h;
    struct janas_url u;
    int ok = 0;
    for (int hop = 0; hop < 6 && !ok; hop++) {
        if (janas_url_parse(at, &u, why, wn) != 0)
            return -1;
        const char *hdr[] = {"User-Agent: Brebus (Janas)", "Accept: */*"};
        if (janas_http_request(&h, &u, "GET", hdr, 2, NULL, 0, 30000, NULL, why,
                               wn) != 0)
            return -1;
        int st = h.head.status;
        if (st >= 300 && st < 400 && h.head.location[0]) {
            char next[8800];
            if (h.head.location[0] == '/')
                snprintf(next, sizeof next, "%s://%s%s%s%s",
                         u.tls ? "https" : "http", u.host, u.port[0] ? ":" : "",
                         u.port, h.head.location);
            else
                snprintf(next, sizeof next, "%s", h.head.location);
            snprintf(at, sizeof at, "%s", next);
            janas_http_close(&h);
            continue;
        }
        if (st != 200) {
            janas_http_close(&h);
            snprintf(why, wn, "the address answered %d", st);
            return -1;
        }
        ok = 1;
    }
    if (!ok) {
        snprintf(why, wn, "too many redirects");
        return -1;
    }
    /* its name: the address's last part, with the type's extension */
    char name[300], raw[300] = "";
    const char *p = u.path, *q = strchr(p, '?');
    size_t pl = q ? (size_t)(q - p) : strlen(p);
    const char *last = p + pl;
    while (last > p && last[-1] != '/')
        last--;
    snprintf(raw, sizeof raw, "%.*s", (int)(p + pl - last), last);
    safe_name(raw[0] ? raw : u.host, name, sizeof name);
    /* the extension its type says, where the name does not end with it
       (example.com, a page named by its host) */
    const char *ct = h.head.content_type, *want = NULL;
    if (strstr(ct, "html"))
        want = ".html";
    else if (strstr(ct, "pdf"))
        want = ".pdf";
    else if (strstr(ct, "text/plain"))
        want = ".txt";
    const char *dot = strrchr(name, '.');
    int has =
        dot && (!strcmp(dot, want ? want : "") ||
                (want && !strcmp(want, ".html") &&
                 (!strcasecmp(dot, ".htm") || !strcasecmp(dot, ".xhtml") ||
                  !strcasecmp(dot, ".html"))));
    if (want && !has)
        strncat(name, want, sizeof name - strlen(name) - 1);
    int fd = fresh(dir, name, path, cap);
    if (fd < 0) {
        janas_http_close(&h);
        snprintf(why, wn, "cannot write in sources/");
        return -1;
    }
    struct janas_buf body = {0};
    size_t total = 0;
    int r = 0, bad = 0;
    while (!bad && (r = janas_http_body(&h, &body, 60000)) == 1) {
        total += body.n;
        bad = total > MAX_DOWNLOAD ||
              write(fd, body.p, body.n) != (ssize_t)body.n;
        body.n = 0;
    }
    janas_http_close(&h);
    janas_buf_free(&body);
    bad |= close(fd) != 0 || r < 0;
    if (bad) {
        remove(path);
        snprintf(why, wn,
                 total > MAX_DOWNLOAD ? "larger than 64 MiB: not downloaded"
                                      : "the download broke off");
        return -1;
    }
    /* where and when it came from */
    char note[4700], when[32];
    snprintf(note, sizeof note, "%s.url.txt", path);
    time_t t = time(NULL);
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(when, sizeof when, "%Y-%m-%dT%H:%M:%SZ", &tm);
    FILE *f = fopen(note, "w");
    if (f) {
        fprintf(f, "%s\ndownloaded %s\n", url, when);
        fclose(f);
    }
    return 0;
}

int brebus_attach(struct brebus *b, const char *body, size_t n,
                  struct janas_buf *out)
{
    char ws[4096], err[160], why[600], dir[4200], path[4600];
    if (brebus_ws_path(b, ws, sizeof ws) != 0)
        return fail(out, "no workspace");
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *src = o ? janas_json_str(janas_json_get(o, "source")) : NULL;
    const char *mode = o ? janas_json_str(janas_json_get(o, "mode")) : NULL;
    char source[4096];
    snprintf(source, sizeof source, "%s", src ? src : "");
    int place = mode && !strcmp(mode, "place");
    int info = mode && !strcmp(mode, "info");
    janas_json_free(d);
    snprintf(dir, sizeof dir, "%s/sources", ws);
    if (!strncmp(source, "http://", 7) || !strncmp(source, "https://", 8)) {
        if (info) {
            janas_buf_puts(out, "{\"url\": true}");
            return 0;
        }
        if (janas_mkdirs(dir) != 0)
            return fail(out, "cannot make sources/");
        if (download(source, dir, path, sizeof path, why, sizeof why) != 0)
            return fail(out, why);
        answer(b, path, out);
        return 0;
    }
    char real[4096];
    struct stat st;
    if (source[0] != '/' || !realpath(source, real) || stat(real, &st) != 0 ||
        !S_ISREG(st.st_mode))
        return fail(out, "a file of this computer by its full path, or a "
                         "web address");
    if (info) { /* its size, for the page to ask copy or read in place */
        janas_buf_printf(
            out, "{\"bytes\": %lld, \"inside\": %s}", (long long)st.st_size,
            !strncmp(real, ws, strlen(ws)) && real[strlen(ws)] == '/'
                ? "true"
                : "false");
        return 0;
    }
    if (!strncmp(real, ws, strlen(ws)) && real[strlen(ws)] == '/') {
        answer(b, real, out); /* in the workspace already */
        return 0;
    }
    if (place) {
        /* read where it is: added to the folders read only */
        struct janas_buf cur = {0}, set = {0}, res = {0};
        brebus_folders_get(b, &cur);
        janas_buf_put(&cur, "", 1);
        char *end = strrchr(cur.p, ']');
        if (end) {
            janas_buf_put(&set, cur.p, (size_t)(end - cur.p));
            janas_buf_puts(&set, strstr(cur.p, "[\"") ? ", " : "");
            str(&set, real);
            janas_buf_puts(&set, "]}");
        }
        int rc = end ? brebus_folders_set(b, set.p, set.n, &res) : -1;
        janas_buf_free(&cur);
        janas_buf_free(&set);
        janas_buf_free(&res);
        if (rc != 0)
            return fail(out, "cannot add it to the folders read only");
        answer(b, real, out);
        return 0;
    }
    const char *base = strrchr(real, '/') + 1;
    char name[300];
    safe_name(base, name, sizeof name);
    if (janas_mkdirs(dir) != 0)
        return fail(out, "cannot make sources/");
    int fd = fresh(dir, name, path, sizeof path);
    if (fd < 0)
        return fail(out, "cannot write in sources/");
    int bad = copy_file(real, fd);
    bad |= close(fd) != 0;
    if (bad) {
        remove(path);
        return fail(out, "it could not be copied (disk full?)");
    }
    answer(b, path, out);
    return 0;
}
