/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * connectors.c - desktop extensions for the model of Brebus's chat: MCP
 * Bundles (.mcpb, formerly Claude Desktop's .dxt), a zip with a
 * manifest.json and an MCP server of its own (Node, Python, or a program)
 * that reaches an application of this computer or a service.
 *
 * Looked at (its name, what it is for, the settings it asks the user for,
 * whether its runtime is here), then installed: unpacked into connectors/
 * beside the user's mcp.json, every name of the archive kept inside its
 * folder, and given an entry in mcp.json - its command, arguments and
 * environment with the manifest's ${__dirname}, ${user_config.*}, ${HOME},
 * ${DOCUMENTS}, ${DOWNLOADS}, ${DESKTOP} and ${/} put in - which janas-chat
 * reads too. Removed: its folder and its entry.
 */
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/mcp_tls.h"
#include "common/zip.h"
#include "janas/mcp.h"

#define MAX_BUNDLE ((size_t)512 << 20)
#define MAX_FILE ((size_t)256 << 20)

#ifdef _WIN32
#define SEP "\\"
#define PATH_SEP ';'
#else
#define SEP "/"
#define PATH_SEP ':'
#endif

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

/* mcp.json's path, and the folder beside it (connectors) */
static int paths(char *cfg, size_t cn, char *dir, size_t dn)
{
    int32_t len = 0;
    if (janas_mcp_config_path(cfg, (int32_t)cn, &len) != JANAS_MCP_OK)
        return -1;
    char base[4096];
    snprintf(base, sizeof base, "%s", cfg);
    char *sl = strrchr(base, '/');
#ifdef _WIN32
    char *bs = strrchr(base, '\\');
    if (!sl || (bs && bs > sl))
        sl = bs;
#endif
    if (!sl)
        return -1;
    *sl = 0;
    return snprintf(dir, dn, "%s/connectors", base) < (int)dn ? 0 : -1;
}

static uint8_t *slurp(const char *path, size_t max, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = len > 0 && (size_t)len <= max ? malloc((size_t)len + 1) : NULL;
    size_t r = b ? fread(b, 1, (size_t)len, f) : 0;
    fclose(f);
    if (!b || r != (size_t)len) {
        free(b);
        return NULL;
    }
    b[r] = 0;
    *n = r;
    return b;
}

/* a program on the PATH: its whole path into out, 0; -1 with none */
static int on_path(const char *name, char *out, size_t n)
{
    const char *p = getenv("PATH");
    while (p && *p) {
        const char *e = strchr(p, PATH_SEP);
        size_t l = e ? (size_t)(e - p) : strlen(p);
        if (l) {
#ifdef _WIN32
            snprintf(out, n, "%.*s\\%s.exe", (int)l, p, name);
#else
            snprintf(out, n, "%.*s/%s", (int)l, p, name);
#endif
            if (access(out, X_OK) == 0)
                return 0;
        }
        p = e ? e + 1 : NULL;
    }
    return -1;
}

/* the program a server's command names: node, python (Brebus's own on
   Windows, the system's elsewhere), uv; else the command as it is (a
   program of the bundle, after ${__dirname}) */
static int runtime(const char *cmd, char *out, size_t n, char *why, size_t wn)
{
    if (!strcmp(cmd, "node") || !strcmp(cmd, "npx") || !strcmp(cmd, "uv") ||
        !strcmp(cmd, "uvx")) {
        if (on_path(cmd, out, n) == 0)
            return 0;
        snprintf(why, wn, "it needs %s, which is not on this computer (%s)",
                 cmd,
                 !strcmp(cmd, "node") || !strcmp(cmd, "npx")
                     ? "Node.js: nodejs.org"
                     : "uv: docs.astral.sh/uv");
        return -1;
    }
    if (!strcmp(cmd, "python") || !strcmp(cmd, "python3")) {
#ifdef _WIN32
        const char *la = getenv("LOCALAPPDATA");
        snprintf(out, n, "%s\\janas\\python\\runtime\\python.exe",
                 la ? la : ".");
        if (access(out, F_OK) == 0)
            return 0;
        if (on_path("python", out, n) == 0)
            return 0;
#else
        if (on_path("python3", out, n) == 0)
            return 0;
#endif
        snprintf(why, wn,
                 "it needs Python, which is not on this computer "
                 "(on Windows, Settings, Run code installs it)");
        return -1;
    }
    snprintf(out, n, "%s", cmd);
    return 0;
}

/* ${...} put in: the bundle's folder, the user's settings, the folders */
static void expand(const char *s, const char *dirname,
                   const struct janas_json *uc, const struct janas_json *vals,
                   struct janas_buf *o);

/* a setting's value, its own ${HOME} and the like put in (a default such
   as "${HOME}/Notes"), not other settings */
static void expand_value(const char *s, size_t n, const char *dirname,
                         struct janas_buf *o)
{
    char v[4096];
    snprintf(v, sizeof v, "%.*s", (int)(n < sizeof v ? n : sizeof v - 1), s);
    if (strstr(v, "${user_config."))
        janas_buf_puts(o, v);
    else
        expand(v, dirname, NULL, NULL, o);
}

static void expand(const char *s, const char *dirname,
                   const struct janas_json *uc, const struct janas_json *vals,
                   struct janas_buf *o)
{
    const char *home = getenv("HOME");
#ifdef _WIN32
    const char *up = getenv("USERPROFILE");
    if (up)
        home = up;
#endif
    while (*s) {
        const char *a = strstr(s, "${");
        const char *z = a ? strchr(a, '}') : NULL;
        if (!a || !z) {
            janas_buf_puts(o, s);
            return;
        }
        janas_buf_put(o, s, (size_t)(a - s));
        char k[128];
        snprintf(k, sizeof k, "%.*s", (int)(z - a - 2), a + 2);
        if (!strcmp(k, "__dirname"))
            janas_buf_puts(o, dirname);
        else if (!strcmp(k, "/") || !strcmp(k, "pathSeparator"))
            janas_buf_puts(o, SEP);
        else if (!strcmp(k, "HOME"))
            janas_buf_puts(o, home ? home : "");
        else if (!strcmp(k, "DOCUMENTS") || !strcmp(k, "DOWNLOADS") ||
                 !strcmp(k, "DESKTOP")) {
            janas_buf_printf(o, "%s%s%s", home ? home : "", SEP,
                             !strcmp(k, "DOCUMENTS")   ? "Documents"
                             : !strcmp(k, "DOWNLOADS") ? "Downloads"
                                                       : "Desktop");
        } else if (!strncmp(k, "user_config.", 12)) {
            const char *key = k + 12;
            const struct janas_json *v = janas_json_get(vals, key);
            const struct janas_json *f = janas_json_get(uc, key);
            const struct janas_json *def =
                f ? janas_json_get(f, "default") : NULL;
            const struct janas_json *x = v ? v : def;
            if (x && x->type == JANAS_JSON_STRING)
                expand_value(x->s, x->n, dirname, o);
            else if (x && (x->type == JANAS_JSON_NUMBER))
                janas_buf_put(o, x->s, x->n);
            else if (x && (x->type == JANAS_JSON_TRUE ||
                           x->type == JANAS_JSON_FALSE))
                janas_buf_puts(o,
                               x->type == JANAS_JSON_TRUE ? "true" : "false");
        } else
            janas_buf_put(o, a, (size_t)(z - a + 1)); /* not one of ours */
        s = z + 1;
    }
}

/* a name of the archive that stays inside the folder it is unpacked in */
static int safe_name(const char *n, size_t len)
{
    if (!len || n[0] == '/' || n[0] == '\\' || memchr(n, ':', len))
        return 0;
    for (size_t i = 0; i + 1 < len; i++)
        if (n[i] == '.' && n[i + 1] == '.' &&
            (i == 0 || n[i - 1] == '/' || n[i - 1] == '\\') &&
            (i + 2 == len || n[i + 2] == '/' || n[i + 2] == '\\'))
            return 0;
    return 1;
}

struct unpack {
    char dir[4400];
    int bad;
};

static int put_file(void *ctx, const char *name, size_t name_n,
                    const uint8_t *p, size_t n)
{
    struct unpack *u = ctx;
    if (!safe_name(name, name_n)) {
        u->bad = 1;
        return 1;
    }
    char path[4600];
    snprintf(path, sizeof path, "%s/%.*s", u->dir, (int)name_n, name);
    if (name[name_n - 1] == '/') /* a folder's entry */
        return janas_mkdirs(path) == 0 ? 0 : 1;
    char parent[4600];
    snprintf(parent, sizeof parent, "%s", path);
    char *sl = strrchr(parent, '/');
    if (sl) {
        *sl = 0;
        janas_mkdirs(parent);
    }
    FILE *f = fopen(path, "wb");
    if (!f)
        return 1;
    int bad = fwrite(p, 1, n, f) != n;
    bad |= fclose(f) != 0;
#ifndef _WIN32
    chmod(path, 0755); /* a program of the bundle runs */
#endif
    return bad;
}

/* the manifest of a bundle at path, parsed: NULL with the reason */
static struct janas_json_doc *manifest(const char *path, uint8_t **zip,
                                       size_t *zn, char *why, size_t wn)
{
    *zip = slurp(path, MAX_BUNDLE, zn);
    if (!*zip) {
        snprintf(why, wn, "cannot read %s", path);
        return NULL;
    }
    uint8_t *m = NULL;
    size_t mn = 0;
    int r =
        janas_zip_get(*zip, *zn, "manifest.json", 1 << 20, &m, &mn, why, wn);
    if (r != 0) {
        if (r == 1)
            snprintf(why, wn, "no manifest.json: not an MCP Bundle");
        free(*zip);
        *zip = NULL;
        return NULL;
    }
    struct janas_json_doc *d = janas_json_parse((char *)m, mn, why, wn);
    free(m);
    if (!d || janas_json_root(d)->type != JANAS_JSON_OBJECT) {
        janas_json_free(d);
        free(*zip);
        *zip = NULL;
        snprintf(why, wn, "manifest.json is not a JSON object");
        return NULL;
    }
    return d;
}

static const char *sget(const struct janas_json *o, const char *k)
{
    const char *s = janas_json_str(janas_json_get(o, k));
    return s ? s : "";
}

/* a bundle's name as a folder and an mcp.json entry may carry */
static void safe_id(const char *in, char *out, size_t n)
{
    size_t k = 0;
    for (const char *c = in; *c && k + 1 < n; c++)
        out[k++] =
            (char)((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                           (*c >= '0' && *c <= '9') || *c == '-' || *c == '_' ||
                           *c == '.'
                       ? *c
                       : '-');
    out[k] = 0;
}

int brebus_connector_inspect(struct brebus *b, const char *body, size_t n,
                             struct janas_buf *out)
{
    (void)b;
    char why[300];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, why, sizeof why) : NULL;
    const char *path =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "path")) : NULL;
    uint8_t *zip = NULL;
    size_t zn = 0;
    struct janas_json_doc *m =
        path ? manifest(path, &zip, &zn, why, sizeof why) : NULL;
    if (!path)
        snprintf(why, sizeof why, "no file named");
    if (!m) {
        janas_json_free(d);
        janas_buf_puts(out, "{\"error\": ");
        str(out, why);
        janas_buf_puts(out, "}");
        return -1;
    }
    free(zip);
    const struct janas_json *r = janas_json_root(m);
    const struct janas_json *srv = janas_json_get(r, "server");
    const struct janas_json *mc =
        srv ? janas_json_get(srv, "mcp_config") : NULL;
    char exe[4200], rwhy[300] = "";
    const char *cmd = mc ? sget(mc, "command") : "";
    int ok = *cmd && runtime(cmd, exe, sizeof exe, rwhy, sizeof rwhy) == 0;
    janas_buf_puts(out, "{\"name\": ");
    str(out, sget(r, "name"));
    janas_buf_puts(out, ", \"display_name\": ");
    str(out,
        *sget(r, "display_name") ? sget(r, "display_name") : sget(r, "name"));
    janas_buf_puts(out, ", \"version\": ");
    str(out, sget(r, "version"));
    janas_buf_puts(out, ", \"description\": ");
    str(out, sget(r, "description"));
    janas_buf_puts(out, ", \"server\": ");
    str(out, srv ? sget(srv, "type") : "");
    janas_buf_printf(out, ", \"runs\": %s, \"why\": ", ok ? "true" : "false");
    str(out, *cmd ? rwhy : "its manifest names no command to start");
    /* the settings it asks for, as the manifest gives them */
    janas_buf_puts(out, ", \"user_config\": [");
    const struct janas_json *uc = janas_json_get(r, "user_config");
    int k = 0;
    for (const struct janas_json *f =
             uc && uc->type == JANAS_JSON_OBJECT ? uc->child : NULL;
         f; f = f->next) {
        char key[128];
        snprintf(key, sizeof key, "%.*s", (int)f->key_n, f->key);
        janas_buf_puts(out, k++ ? ", {\"key\": " : "{\"key\": ");
        str(out, key);
        janas_buf_puts(out, ", \"title\": ");
        str(out, *sget(f, "title") ? sget(f, "title") : key);
        janas_buf_puts(out, ", \"description\": ");
        str(out, sget(f, "description"));
        janas_buf_puts(out, ", \"type\": ");
        str(out, sget(f, "type"));
        const struct janas_json *def = janas_json_get(f, "default");
        janas_buf_puts(out, ", \"default\": ");
        if (def)
            janas_json_write(out, def);
        else
            janas_buf_puts(out, "null");
        const struct janas_json *rq = janas_json_get(f, "required"),
                                *se = janas_json_get(f, "sensitive");
        janas_buf_printf(out, ", \"required\": %s, \"sensitive\": %s}",
                         rq && rq->type == JANAS_JSON_TRUE ? "true" : "false",
                         se && se->type == JANAS_JSON_TRUE ? "true" : "false");
    }
    janas_buf_puts(out, "]}");
    janas_json_free(m);
    janas_json_free(d);
    return 0;
}

/* mcp.json with its entry name replaced (entry NULL: removed), the rest as
   it was: 0, or -1 */
static int set_entry(const char *cfg, const char *name, const char *entry)
{
    size_t n = 0;
    uint8_t *t = slurp(cfg, 4 << 20, &n);
    char err[200];
    struct janas_json_doc *d =
        t ? janas_json_parse((char *)t, n, err, sizeof err) : NULL;
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    if (t && (!r || r->type != JANAS_JSON_OBJECT)) {
        free(t);
        janas_json_free(d);
        return -1; /* not ours to overwrite */
    }
    struct janas_buf o = {0};
    janas_buf_puts(&o, "{\n  \"mcpServers\": {");
    int k = 0;
    const struct janas_json *ms = r ? janas_json_get(r, "mcpServers") : NULL;
    for (const struct janas_json *e =
             ms && ms->type == JANAS_JSON_OBJECT ? ms->child : NULL;
         e; e = e->next) {
        if (e->key_n == strlen(name) && !memcmp(e->key, name, e->key_n))
            continue;
        janas_buf_puts(&o, k++ ? ",\n    " : "\n    ");
        janas_json_write_str(&o, e->key, e->key_n);
        janas_buf_puts(&o, ": ");
        janas_json_write(&o, e);
    }
    if (entry) {
        janas_buf_puts(&o, k++ ? ",\n    " : "\n    ");
        str(&o, name);
        janas_buf_puts(&o, ": ");
        janas_buf_puts(&o, entry);
    }
    janas_buf_puts(&o, k ? "\n  }" : "}");
    /* the file's other members, as they were */
    for (const struct janas_json *e = r ? r->child : NULL; e; e = e->next) {
        if (e->key_n == 10 && !memcmp(e->key, "mcpServers", 10))
            continue;
        janas_buf_puts(&o, ",\n  ");
        janas_json_write_str(&o, e->key, e->key_n);
        janas_buf_puts(&o, ": ");
        janas_json_write(&o, e);
    }
    janas_buf_puts(&o, "\n}\n");
    janas_json_free(d);
    free(t);
    char tmp[4200], dir[4200];
    snprintf(dir, sizeof dir, "%s", cfg);
    char *sl = strrchr(dir, '/');
    if (sl) {
        *sl = 0;
        janas_mkdirs(dir);
    }
    snprintf(tmp, sizeof tmp, "%s.new", cfg);
    FILE *f = fopen(tmp, "wb");
    int bad = !f || o.oom;
    if (f) {
        bad |= fwrite(o.p, 1, o.n, f) != o.n;
        bad |= fclose(f) != 0;
    }
    /* it may hold a connector's keys: the user's alone */
    if (!bad)
        janas_tls_private(tmp);
    if (bad || rename(tmp, cfg) != 0) {
        remove(tmp);
        bad = 1;
    }
    janas_buf_free(&o);
    return bad ? -1 : 0;
}

static int remove_tree(const char *dir, int depth)
{
    if (depth > 12)
        return -1;
    DIR *d = opendir(dir);
    if (!d)
        return -1;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        char p[4600];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        if (stat(p, &st) == 0 && S_ISDIR(st.st_mode))
            remove_tree(p, depth + 1);
        else
            remove(p);
    }
    closedir(d);
    return rmdir(dir) == 0 ? 0 : -1;
}

int brebus_connector_install(struct brebus *b, const char *body, size_t n,
                             struct janas_buf *out)
{
    (void)b;
    char why[300] = "", cfg[4096], root[4200], id[128], dir[4400];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, why, sizeof why) : NULL;
    const struct janas_json *rq = d ? janas_json_root(d) : NULL;
    const char *path = rq ? janas_json_str(janas_json_get(rq, "path")) : NULL;
    const struct janas_json *vals = rq ? janas_json_get(rq, "config") : NULL;
    uint8_t *zip = NULL;
    size_t zn = 0;
    struct janas_json_doc *m =
        path ? manifest(path, &zip, &zn, why, sizeof why) : NULL;
    int rc = -1;
    struct janas_buf entry = {0};
    if (!m)
        goto out;
    const struct janas_json *r = janas_json_root(m);
    safe_id(sget(r, "name"), id, sizeof id);
    const struct janas_json *uc = janas_json_get(r, "user_config");
    const struct janas_json *srv = janas_json_get(r, "server");
    const struct janas_json *mc =
        srv ? janas_json_get(srv, "mcp_config") : NULL;
    const char *cmd = mc ? sget(mc, "command") : "";
    if (!id[0] || !*cmd) {
        snprintf(why, sizeof why, "its manifest names no %s",
                 id[0] ? "command to start" : "name");
        goto out;
    }
    /* the settings it needs */
    for (const struct janas_json *f =
             uc && uc->type == JANAS_JSON_OBJECT ? uc->child : NULL;
         f; f = f->next) {
        const struct janas_json *req = janas_json_get(f, "required");
        char key[128];
        snprintf(key, sizeof key, "%.*s", (int)f->key_n, f->key);
        const struct janas_json *v = janas_json_get(vals, key);
        if (req && req->type == JANAS_JSON_TRUE &&
            !janas_json_get(f, "default") &&
            (!v || (v->type == JANAS_JSON_STRING && v->n == 0))) {
            snprintf(why, sizeof why, "it needs %s",
                     *sget(f, "title") ? sget(f, "title") : key);
            goto out;
        }
    }
    if (paths(cfg, sizeof cfg, root, sizeof root) != 0) {
        snprintf(why, sizeof why, "no folder for the connectors");
        goto out;
    }
    /* unpacked beside, put in place of an older version only when whole:
       a bundle refused leaves the one installed as it was */
    char fin[4400];
    snprintf(fin, sizeof fin, "%s/%s", root, id);
    snprintf(dir, sizeof dir, "%s/.%s.new", root, id);
    remove_tree(dir, 0);
    if (janas_mkdirs(dir) != 0) {
        snprintf(why, sizeof why, "cannot make the connector's folder");
        goto out;
    }
    struct unpack u = {.bad = 0};
    snprintf(u.dir, sizeof u.dir, "%s", dir);
    if (janas_zip_each(zip, zn, MAX_FILE, put_file, &u, why, sizeof why) != 0 ||
        u.bad) {
        if (u.bad)
            snprintf(why, sizeof why,
                     "a file of the bundle would land "
                     "outside its folder: not installed");
        remove_tree(dir, 0);
        goto out;
    }
    /* its entry: the command, its arguments and its environment, the
       manifest's ${...} put in */
    char native[4400]; /* where it will be: the entry names it there */
    snprintf(native, sizeof native, "%s", fin);
#ifdef _WIN32
    for (char *c = native; *c; c++)
        if (*c == '/')
            *c = '\\';
#endif
    char exe[4200], rwhy[300];
    if (runtime(cmd, exe, sizeof exe, rwhy, sizeof rwhy) != 0) {
        snprintf(why, sizeof why, "%s", rwhy);
        remove_tree(dir, 0);
        goto out;
    }
    struct janas_buf x = {0};
    expand(exe, native, uc, vals, &x);
    janas_buf_puts(&entry, "{\"command\": ");
    janas_json_write_str(&entry, x.p ? x.p : "", x.n);
    janas_buf_free(&x);
    janas_buf_puts(&entry, ", \"args\": [");
    const struct janas_json *args = janas_json_get(mc, "args");
    int k = 0;
    for (const struct janas_json *a =
             args && args->type == JANAS_JSON_ARRAY ? args->child : NULL;
         a; a = a->next) {
        const char *s = janas_json_str(a);
        if (!s)
            continue;
        struct janas_buf e = {0};
        expand(s, native, uc, vals, &e);
        janas_buf_puts(&entry, k++ ? ", " : "");
        janas_json_write_str(&entry, e.p ? e.p : "", e.n);
        janas_buf_free(&e);
    }
    janas_buf_puts(&entry, "], \"env\": {");
    const struct janas_json *env = janas_json_get(mc, "env");
    k = 0;
    for (const struct janas_json *e =
             env && env->type == JANAS_JSON_OBJECT ? env->child : NULL;
         e; e = e->next) {
        const char *s = janas_json_str(e);
        if (!s)
            continue;
        struct janas_buf v = {0};
        expand(s, native, uc, vals, &v);
        janas_buf_puts(&entry, k++ ? ", " : "");
        janas_json_write_str(&entry, e->key, e->key_n);
        janas_buf_puts(&entry, ": ");
        janas_json_write_str(&entry, v.p ? v.p : "", v.n);
        janas_buf_free(&v);
    }
    janas_buf_puts(&entry, "}}");
    /* the folder in its place: the entry names it */
    {
        char old[4500];
        snprintf(old, sizeof old, "%s/.%s.old", root, id);
        remove_tree(old, 0);
        struct stat st;
        int had = stat(fin, &st) == 0;
        if ((had && rename(fin, old) != 0) || rename(dir, fin) != 0) {
            if (had)
                rename(old, fin);
            snprintf(why, sizeof why, "cannot put the connector in place");
            remove_tree(dir, 0);
            goto out;
        }
        remove_tree(old, 0);
        snprintf(dir, sizeof dir, "%s", fin);
    }
    if (entry.oom || set_entry(cfg, id, entry.p) != 0) {
        snprintf(why, sizeof why, "cannot write mcp.json");
        remove_tree(dir, 0);
        goto out;
    }
    rc = 0;
out:
    free(zip);
    janas_buf_free(&entry);
    janas_json_free(m);
    janas_json_free(d);
    janas_buf_puts(out, rc ? "{\"error\": " : "{\"name\": ");
    str(out, rc ? (why[0] ? why : "not installed") : id);
    janas_buf_puts(out, "}");
    return rc;
}

void brebus_connectors_get(struct brebus *b, struct janas_buf *out)
{
    (void)b;
    char cfg[4096], root[4200];
    janas_buf_puts(out, "{\"installed\": [");
    if (paths(cfg, sizeof cfg, root, sizeof root) == 0) {
        DIR *d = opendir(root);
        struct dirent *e;
        int k = 0;
        while (d && (e = readdir(d))) {
            if (e->d_name[0] == '.')
                continue;
            char p[4600];
            snprintf(p, sizeof p, "%s/%s/manifest.json", root, e->d_name);
            size_t n = 0;
            uint8_t *t = slurp(p, 1 << 20, &n);
            char err[160];
            struct janas_json_doc *m =
                t ? janas_json_parse((char *)t, n, err, sizeof err) : NULL;
            const struct janas_json *r = m ? janas_json_root(m) : NULL;
            janas_buf_puts(out, k++ ? ", {\"name\": " : "{\"name\": ");
            str(out, e->d_name);
            janas_buf_puts(out, ", \"display_name\": ");
            str(out, r && *sget(r, "display_name") ? sget(r, "display_name")
                                                   : e->d_name);
            janas_buf_puts(out, ", \"description\": ");
            str(out, r ? sget(r, "description") : "");
            janas_buf_puts(out, ", \"version\": ");
            str(out, r ? sget(r, "version") : "");
            janas_buf_puts(out, "}");
            janas_json_free(m);
            free(t);
        }
        if (d)
            closedir(d);
    }
    janas_buf_puts(out, "], \"dir\": ");
    str(out, root);
    janas_buf_puts(out, "}");
}

int brebus_connector_remove(struct brebus *b, const char *body, size_t n,
                            struct janas_buf *out)
{
    (void)b;
    char why[200], cfg[4096], root[4200], id[128], dir[4400];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, why, sizeof why) : NULL;
    const char *name =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "name")) : NULL;
    int rc = -1;
    if (name && paths(cfg, sizeof cfg, root, sizeof root) == 0) {
        safe_id(name, id, sizeof id);
        snprintf(dir, sizeof dir, "%s/%s", root, id);
        struct stat st;
        int had = id[0] && stat(dir, &st) == 0;
        if (had)
            remove_tree(dir, 0);
        if (id[0] && (set_entry(cfg, id, NULL) == 0 || had))
            rc = 0;
    }
    janas_json_free(d);
    janas_buf_puts(out, rc ? "{\"error\": \"no such connector\"}" : "{}");
    return rc;
}

/* ---- remote connectors, a click each: servers of well-known services
   that speak MCP over HTTP, sign in by OAuth with a client registering
   itself (mcp_oauth.c), or need no sign-in; read from each maker's own
   documentation on 18 November 2026 ---- */
static const struct remote {
    const char *id, *title, *url, *about;
} REMOTE[] = {
    {"notion", "Notion", "https://mcp.notion.com/mcp",
     "your pages and databases"},
    {"linear", "Linear", "https://mcp.linear.app/mcp",
     "issues, projects and cycles"},
    {"atlassian", "Atlassian", "https://mcp.atlassian.com/v2/mcp",
     "Jira and Confluence"},
    {"sentry", "Sentry", "https://mcp.sentry.dev/mcp",
     "errors and issues of your projects"},
    {"stripe", "Stripe", "https://mcp.stripe.com",
     "payments, customers and the docs"},
    {"huggingface", "Hugging Face", "https://huggingface.co/mcp",
     "models, datasets, papers and Spaces"},
    {"cloudflare", "Cloudflare", "https://mcp.cloudflare.com/mcp",
     "your account's resources"},
    {"cloudflare-docs", "Cloudflare Docs",
     "https://docs.mcp.cloudflare.com/mcp",
     "Cloudflare's documentation (no sign-in)"},
    {"zapier", "Zapier", "https://mcp.zapier.com/api/v1/connect",
     "the actions of the apps you connected there"},
    {"intercom", "Intercom", "https://mcp.intercom.com/mcp",
     "conversations and contacts"},
    {"supabase", "Supabase", "https://mcp.supabase.com/mcp",
     "your projects' databases"},
    {"netlify", "Netlify", "https://mcp.netlify.com/mcp",
     "your sites and their deploys"},
    {"microsoft-learn", "Microsoft Learn",
     "https://learn.microsoft.com/api/mcp",
     "Microsoft's documentation (no sign-in)"},
};

void brebus_connectors_catalog(struct brebus *b, struct janas_buf *out)
{
    (void)b;
    /* which are in mcp.json already, by their name there */
    char cfg[4096], root[4200], err[160];
    size_t n = 0;
    uint8_t *t = paths(cfg, sizeof cfg, root, sizeof root) == 0
                     ? slurp(cfg, 4 << 20, &n)
                     : NULL;
    struct janas_json_doc *d =
        t ? janas_json_parse((char *)t, n, err, sizeof err) : NULL;
    const struct janas_json *ms =
        d ? janas_json_get(janas_json_root(d), "mcpServers") : NULL;
    janas_buf_puts(out, "{\"remote\": [");
    for (size_t i = 0; i < sizeof REMOTE / sizeof *REMOTE; i++) {
        janas_buf_puts(out, i ? ", {\"id\": " : "{\"id\": ");
        str(out, REMOTE[i].id);
        janas_buf_puts(out, ", \"title\": ");
        str(out, REMOTE[i].title);
        janas_buf_puts(out, ", \"url\": ");
        str(out, REMOTE[i].url);
        janas_buf_puts(out, ", \"about\": ");
        str(out, REMOTE[i].about);
        janas_buf_printf(out, ", \"added\": %s}",
                         ms && janas_json_get(ms, REMOTE[i].id) ? "true"
                                                                : "false");
    }
    janas_buf_puts(out, "]}");
    janas_json_free(d);
    free(t);
}

int brebus_connector_add_remote(struct brebus *b, const char *body, size_t n,
                                struct janas_buf *out)
{
    (void)b;
    char why[200], cfg[4096], root[4200];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, why, sizeof why) : NULL;
    const char *id =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "id")) : NULL;
    int rc = -1;
    for (size_t i = 0; id && i < sizeof REMOTE / sizeof *REMOTE; i++)
        if (!strcmp(REMOTE[i].id, id) &&
            paths(cfg, sizeof cfg, root, sizeof root) == 0) {
            struct janas_buf e = {0};
            janas_buf_puts(&e, "{\"url\": ");
            str(&e, REMOTE[i].url);
            janas_buf_puts(&e, "}");
            rc = e.oom ? -1 : set_entry(cfg, REMOTE[i].id, e.p);
            janas_buf_free(&e);
        }
    janas_json_free(d);
    janas_buf_puts(out, rc ? "{\"error\": \"not added\"}" : "{}");
    return rc;
}

int brebus_connector_drop(struct brebus *b, const char *body, size_t n,
                          struct janas_buf *out)
{
    (void)b;
    char why[200], cfg[4096], root[4200];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, why, sizeof why) : NULL;
    const char *id =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "id")) : NULL;
    int rc = -1;
    /* only the catalog's: the user's own entries are theirs to edit */
    for (size_t i = 0; id && i < sizeof REMOTE / sizeof *REMOTE; i++)
        if (!strcmp(REMOTE[i].id, id) &&
            paths(cfg, sizeof cfg, root, sizeof root) == 0)
            rc = set_entry(cfg, id, NULL);
    janas_json_free(d);
    janas_buf_puts(out, rc ? "{\"error\": \"not removed\"}" : "{}");
    return rc;
}
