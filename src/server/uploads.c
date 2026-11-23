/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * uploads.c - OpenAI's Uploads API: a file larger than one request sent in
 * parts (each at most 64 MiB), joined into a File when the client
 * completes the upload, checked against the size it announced and the MD5
 * it may give. An upload lasts an hour, as at OpenAI. Its parts wait on
 * disk beside the files (.uploads in their directory); what is pending is
 * known only to the server running, so a restart drops it and the parts
 * left over are removed at start.
 */
#include "server.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <microhttpd.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>
#include <yyjson.h>

#include "files.h"
#include "files_store.h"
#include "multipart.h"
#include "request.h"
#include "store_disk.h"

#define UPLOAD_MAX ((uint64_t)8 << 30) /* bytes of an upload, as OpenAI */
#define UPLOAD_LIFE 3600               /* seconds */
#define UPLOADS_KEPT 256               /* uploads remembered at most */

enum { PENDING, COMPLETED, CANCELLED, EXPIRED };
static const char *const STATUS[] = {"pending", "completed", "cancelled",
                                     "expired"};

struct upload {
    char id[48];
    char filename[256];
    char purpose[16];
    uint64_t bytes, got;
    int64_t created_at, expires_at, file_after;
    int status;
    char (*part)[48]; /* the parts' ids, in the order they came */
    uint64_t *part_bytes;
    size_t np;
    struct srv_file file; /* once completed */
};

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static struct upload *ups;
static size_t n_ups;
static char parts_dir[4200];

/* The directory of the parts; what an earlier run left in it goes. */
static int ensure_dir(struct srv *srv)
{
    if (parts_dir[0])
        return 0;
    snprintf(parts_dir, sizeof parts_dir, "%s/.uploads",
             srv_files_dir(srv->files));
    if (mkdir(parts_dir, 0700) != 0 && errno != EEXIST) {
        parts_dir[0] = 0;
        return -1;
    }
    DIR *d = opendir(parts_dir);
    struct dirent *de;
    while (d && (de = readdir(d))) {
        char p[4500];
        if (de->d_name[0] != '.' && snprintf(p, sizeof p, "%s/%s", parts_dir,
                                             de->d_name) < (int)sizeof p)
            unlink(p);
    }
    if (d)
        closedir(d);
    return 0;
}

static void part_path(char *out, size_t cap, const char *id)
{
    snprintf(out, cap, "%s/%s", parts_dir, id);
}

static void drop_parts(struct upload *u)
{
    for (size_t i = 0; i < u->np; i++) {
        char p[4500];
        part_path(p, sizeof p, u->part[i]);
        unlink(p);
    }
    free(u->part);
    free(u->part_bytes);
    u->part = NULL;
    u->part_bytes = NULL;
    u->np = 0;
}

/* Under mu: past its hour, a pending upload expires. */
static void age(struct upload *u)
{
    if (u->status == PENDING && (int64_t)time(NULL) >= u->expires_at) {
        u->status = EXPIRED;
        drop_parts(u);
    }
}

static struct upload *find(const char *id)
{
    for (size_t i = 0; i < n_ups; i++)
        if (strcmp(ups[i].id, id) == 0) {
            age(&ups[i]);
            return &ups[i];
        }
    return NULL;
}

static int reply_upload(struct srv_req *r, const struct upload *u)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "id", u->id);
    yyjson_mut_obj_add_str(d, o, "object", "upload");
    yyjson_mut_obj_add_uint(d, o, "bytes", u->bytes);
    yyjson_mut_obj_add_int(d, o, "created_at", u->created_at);
    yyjson_mut_obj_add_strcpy(d, o, "filename", u->filename);
    yyjson_mut_obj_add_strcpy(d, o, "purpose", u->purpose);
    yyjson_mut_obj_add_str(d, o, "status", STATUS[u->status]);
    yyjson_mut_obj_add_int(d, o, "expires_at", u->expires_at);
    if (u->status == COMPLETED)
        yyjson_mut_obj_add_val(d, o, "file", srv_file_json(d, &u->file));
    else
        yyjson_mut_obj_add_null(d, o, "file");
    yyjson_mut_doc_set_root(d, o);
    size_t n;
    char *out = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, out, n);
}

static int bad(struct srv_req *r, const char *msg)
{
    return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                           "%s", msg);
}

static int no_upload(struct srv_req *r)
{
    return srv_reply_error(r, 404, "invalid_request_error", "not_found",
                           "No such Upload: %s", r->param[0]);
}

static int not_pending(struct srv_req *r, const struct upload *u)
{
    return srv_reply_error(r, 400, "invalid_request_error", "invalid_state",
                           "The upload %s is %s, not pending.", u->id,
                           STATUS[u->status]);
}

int srv_uploads_create(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *filename = yyjson_get_str(yyjson_obj_get(root, "filename"));
    const char *purpose = yyjson_get_str(yyjson_obj_get(root, "purpose"));
    yyjson_val *by = yyjson_obj_get(root, "bytes");
    yyjson_val *ea = yyjson_obj_get(root, "expires_after");
    static const char *const purposes[] = {
        "assistants", "batch", "fine-tune", "vision", "user_data", "evals"};
    int pok = 0;
    for (size_t i = 0; purpose && i < 6; i++)
        pok |= !strcmp(purpose, purposes[i]);
    int64_t file_after = 0;
    const char *why = NULL;
    if (!yyjson_is_obj(root))
        why = "The body is a JSON object: filename, purpose, bytes, "
              "mime_type.";
    else if (!filename || !*filename)
        why = "'filename' is required.";
    else if (!pok)
        why = "'purpose' is assistants, batch, fine-tune, vision, user_data "
              "or evals.";
    else if (!yyjson_is_uint(by) &&
             !(yyjson_is_int(by) && yyjson_get_sint(by) >= 0))
        why = "'bytes' is the file's size, a whole number.";
    else if (yyjson_get_uint(by) > UPLOAD_MAX)
        why = "An upload is at most 8 GiB.";
    else if (!yyjson_get_str(yyjson_obj_get(root, "mime_type")))
        why = "'mime_type' is required.";
    else if (ea) {
        yyjson_val *an = yyjson_obj_get(ea, "anchor"),
                   *se = yyjson_obj_get(ea, "seconds");
        if (!yyjson_equals_str(an, "created_at") || !yyjson_is_int(se) ||
            yyjson_get_sint(se) < 3600 || yyjson_get_sint(se) > 2592000)
            why = "'expires_after' is anchor created_at and seconds from "
                  "3600 to 2592000.";
        else
            file_after = yyjson_get_sint(se);
    }
    if (!why && !ea && !strcmp(purpose, "batch"))
        file_after = 2592000;
    if (why) {
        yyjson_doc_free(doc);
        return bad(r, why);
    }
    struct upload u = {0};
    srv_new_id(u.id, sizeof u.id, "upload_");
    snprintf(u.filename, sizeof u.filename, "%s", filename);
    snprintf(u.purpose, sizeof u.purpose, "%s", purpose);
    u.bytes = yyjson_get_uint(by);
    u.created_at = (int64_t)time(NULL);
    u.expires_at = u.created_at + UPLOAD_LIFE;
    u.file_after = file_after;
    u.status = PENDING;
    yyjson_doc_free(doc);
    pthread_mutex_lock(&mu);
    int ok = ensure_dir(r->srv) == 0;
    if (ok && n_ups == UPLOADS_KEPT) {
        /* the oldest that is no longer pending makes room */
        size_t k = 0;
        while (k < n_ups && (age(&ups[k]), ups[k].status == PENDING))
            k++;
        if (k == n_ups)
            ok = 0;
        else {
            drop_parts(&ups[k]);
            memmove(&ups[k], &ups[k + 1], (n_ups - k - 1) * sizeof(ups[0]));
            n_ups--;
        }
    }
    struct upload *t = ok ? realloc(ups, (n_ups + 1) * sizeof(*t)) : NULL;
    if (t) {
        ups = t;
        ups[n_ups++] = u;
    }
    pthread_mutex_unlock(&mu);
    if (!t)
        return srv_reply_error(r, 429, "invalid_request_error",
                               "too_many_uploads",
                               "%d uploads are pending: complete or cancel "
                               "some.",
                               UPLOADS_KEPT);
    return reply_upload(r, &u);
}

int srv_uploads_part(struct srv_req *r)
{
    struct srv_part parts[4];
    char err[200];
    const char *ct =
        MHD_lookup_connection_value(r->conn, MHD_HEADER_KIND, "Content-Type");
    int n =
        srv_multipart_parse(ct, r->body, r->n_body, parts, 4, err, sizeof err);
    if (n < 0)
        return bad(r, err);
    const struct srv_part *data = srv_multipart_get(parts, n, "data");
    if (!data)
        return bad(r, "'data' is required: the part's bytes.");
    if (data->n > SRV_UPLOAD_PART_MAX)
        return bad(r, "A part is at most 64 MiB.");
    char id[48], path[4500];
    srv_new_id(id, sizeof id, "part_");
    pthread_mutex_lock(&mu);
    struct upload *u = find(r->param[0]);
    int ret = -1;
    if (!u)
        ret = no_upload(r);
    else if (u->status != PENDING)
        ret = not_pending(r, u);
    else if (u->got + data->n > u->bytes)
        ret = bad(r, "The parts would pass the bytes the upload was created "
                     "with.");
    /* the parts of every pending upload, and the disk beside them: within
       what the files kept may take, a margin left free */
    uint64_t pending = 0;
    for (size_t i = 0; ret == -1 && i < n_ups; i++)
        if (ups[i].status == PENDING)
            pending += ups[i].got;
    struct statvfs vs;
    uint64_t avail = statvfs(parts_dir, &vs) == 0
                         ? (uint64_t)vs.f_bavail * (uint64_t)vs.f_frsize
                         : 0;
    if (ret == -1 && pending + data->n > SRV_FILES_MAX)
        ret = srv_reply_error(r, 507, "server_error", NULL,
                              "the pending uploads would pass %llu GiB: "
                              "complete or cancel some",
                              (unsigned long long)(SRV_FILES_MAX >> 30));
    else if (ret == -1 && avail < data->n + SRV_FILES_DISK_FREE)
        ret = srv_reply_error(r, 507, "server_error", NULL,
                              "not enough disk for this part");
    int64_t now = (int64_t)time(NULL);
    if (ret == -1) {
        part_path(path, sizeof path, id);
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        size_t done = 0;
        while (fd >= 0 && done < data->n) {
            ssize_t w = write(fd, data->data + done, data->n - done);
            if (w <= 0)
                break;
            done += (size_t)w;
        }
        void *a = realloc(u->part, (u->np + 1) * sizeof(*u->part));
        if (a)
            u->part = a;
        void *b = realloc(u->part_bytes, (u->np + 1) * sizeof(uint64_t));
        if (b)
            u->part_bytes = b;
        if (fd < 0 || close(fd) != 0 || done != data->n || !a || !b) {
            unlink(path);
            ret = srv_reply_error(r, 500, "server_error", NULL,
                                  "cannot keep the part");
        } else {
            snprintf(u->part[u->np], sizeof u->part[0], "%s", id);
            u->part_bytes[u->np++] = data->n;
            u->got += data->n;
        }
    }
    pthread_mutex_unlock(&mu);
    if (ret != -1)
        return ret;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "id", id);
    yyjson_mut_obj_add_str(d, o, "object", "upload.part");
    yyjson_mut_obj_add_int(d, o, "created_at", now);
    yyjson_mut_obj_add_strcpy(d, o, "upload_id", r->param[0]);
    yyjson_mut_doc_set_root(d, o);
    size_t len;
    char *out = yyjson_mut_write(d, 0, &len);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, out, len);
}

int srv_uploads_complete(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *ids = yyjson_obj_get(root, "part_ids");
    const char *md5 = yyjson_get_str(yyjson_obj_get(root, "md5"));
    if (!yyjson_is_arr(ids)) {
        yyjson_doc_free(doc);
        return bad(r, "'part_ids' is the parts' ids, in order.");
    }
    pthread_mutex_lock(&mu);
    struct upload *u = find(r->param[0]);
    int ret = -1;
    size_t n = yyjson_arr_size(ids);
    char **paths = n ? calloc(n, sizeof(char *)) : NULL;
    if (!u)
        ret = no_upload(r);
    else if (u->status != PENDING)
        ret = not_pending(r, u);
    else if (n == 0 || !paths)
        ret = bad(r, "'part_ids' names no part.");
    /* each id one of this upload's parts, once */
    size_t idx, max;
    yyjson_val *v;
    char *used = u && ret == -1 ? calloc(u->np ? u->np : 1, 1) : NULL;
    if (ret == -1 && !used)
        ret = srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    yyjson_arr_foreach(ids, idx, max, v)
    {
        if (ret != -1)
            break;
        const char *pid = yyjson_get_str(v);
        size_t k = 0;
        while (pid && k < u->np && strcmp(u->part[k], pid) != 0)
            k++;
        if (!pid || k == u->np || used[k]) {
            ret = bad(r, "'part_ids' names a part this upload does not have, "
                         "or one twice.");
            break;
        }
        used[k] = 1;
        paths[idx] = malloc(4500);
        if (!paths[idx]) {
            ret =
                srv_reply_error(r, 500, "server_error", NULL, "out of memory");
            break;
        }
        part_path(paths[idx], 4500, pid);
    }
    free(used);
    /* joined outside the lock: the upload is marked so meanwhile */
    struct upload copy;
    if (ret == -1) {
        u->status = COMPLETED; /* nothing else may touch it now */
        copy = *u;
    }
    pthread_mutex_unlock(&mu);
    if (ret == -1) {
        char err[200];
        struct srv_file f;
        int rc = srv_files_add_parts(r->srv->files, copy.filename, copy.purpose,
                                     (const char *const *)paths, n, copy.bytes,
                                     md5, copy.file_after, &f, err, sizeof err);
        pthread_mutex_lock(&mu);
        u = find(copy.id);
        if (u) {
            if (rc == 0) {
                u->file = f;
                drop_parts(u);
                copy = *u;
            } else {
                u->status = PENDING; /* the client may try again */
            }
        }
        pthread_mutex_unlock(&mu);
        if (rc == 0)
            ret = reply_upload(r, &copy);
        else if (rc == -3 || rc == -2)
            ret = srv_reply_error(
                r, rc == -2 ? 413 : 400, "invalid_request_error",
                rc == -2 ? "files_limit_reached" : "invalid_value", "%s", err);
        else
            ret = srv_reply_error(r, 500, "server_error", NULL, "%s", err);
    }
    for (size_t i = 0; paths && i < n; i++)
        free(paths[i]);
    free(paths);
    yyjson_doc_free(doc);
    return ret;
}

int srv_uploads_cancel(struct srv_req *r)
{
    pthread_mutex_lock(&mu);
    struct upload *u = find(r->param[0]);
    struct upload copy;
    int ret = -1;
    if (!u)
        ret = no_upload(r);
    else if (u->status != PENDING)
        ret = not_pending(r, u);
    else {
        u->status = CANCELLED;
        drop_parts(u);
        copy = *u;
    }
    pthread_mutex_unlock(&mu);
    return ret != -1 ? ret : reply_upload(r, &copy);
}

void srv_uploads_shutdown(void)
{
    pthread_mutex_lock(&mu);
    for (size_t i = 0; i < n_ups; i++)
        drop_parts(&ups[i]);
    free(ups);
    ups = NULL;
    n_ups = 0;
    parts_dir[0] = 0;
    pthread_mutex_unlock(&mu);
}
