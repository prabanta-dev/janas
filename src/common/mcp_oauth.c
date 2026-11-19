/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_oauth.c - see mcp_oauth.h. Every answer of the servers is input from
 * outside: each is read as JSON with a limit on its size, and what is taken
 * from it (addresses, tokens) is checked before it is used or kept.
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "common/mcp_http.h"
#include "common/mcp_oauth.h"
#include "common/mcp_tls.h"

#define MAX_ANSWER (256 * 1024)
#define WAIT_MS 20000

/* ---- where the tokens are kept ---- */

static int auth_dir(char *buf, size_t cap)
{
    const char *e = getenv("JANAS_MCP_AUTH_DIR"), *x = getenv("XDG_DATA_HOME"),
               *h = getenv("HOME");
    int n;
    if (e && *e)
        n = snprintf(buf, cap, "%s", e);
    else if (x && *x)
        n = snprintf(buf, cap, "%s/janas/mcp-auth", x);
    else if (h && *h)
        n = snprintf(buf, cap, "%s/.local/share/janas/mcp-auth", h);
    else
        return -1;
    return n > 0 && (size_t)n < cap ? 0 : -1;
}

/* the file of a server: its name with what a file name cannot hold made _ */
static int auth_file(const char *name, char *buf, size_t cap)
{
    char dir[3800], safe[128];
    if (!name || !*name || auth_dir(dir, sizeof dir) != 0)
        return -1;
    size_t k = 0;
    for (const char *c = name; *c && k + 1 < sizeof safe; c++)
        safe[k++] =
            (char)((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                           (*c >= '0' && *c <= '9') || *c == '-' || *c == '_'
                       ? *c
                       : '_');
    safe[k] = 0;
    int n = snprintf(buf, cap, "%s/%s.json", dir, safe);
    return n > 0 && (size_t)n < cap ? 0 : -1;
}

static void mkdirs(const char *dir)
{
    char p[4096];
    snprintf(p, sizeof p, "%s", dir);
    for (char *c = p + 1; *c; c++)
        if (*c == '/') {
            *c = 0;
            mkdir(p, 0700);
            *c = '/';
        }
    mkdir(p, 0700);
}

static struct janas_json_doc *read_json(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    char *b = malloc(MAX_ANSWER);
    size_t n = b ? fread(b, 1, MAX_ANSWER, f) : 0;
    fclose(f);
    char err[160];
    struct janas_json_doc *d =
        n ? janas_json_parse(b, n, err, sizeof err) : NULL;
    free(b);
    return d;
}

/* written whole, readable by the user alone, then put in place */
static int write_file(const char *path, const struct janas_buf *b)
{
    char dir[4096], tmp[4200];
    snprintf(dir, sizeof dir, "%s", path);
    char *sl = strrchr(dir, '/');
    if (sl)
        *sl = 0;
    mkdirs(dir);
    snprintf(tmp, sizeof tmp, "%s.new", path);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0)
        return -1;
    if (janas_tls_private(tmp) != 0) { /* Windows: the user's alone */
        close(fd);
        remove(tmp);
        return -1;
    }
    int bad = b->oom || write(fd, b->p, b->n) != (ssize_t)b->n;
    bad |= close(fd) != 0;
    if (bad || rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

/* what is kept of a server: its fields, as strings and one number */
struct kept {
    char url[2048], client_id[512], client_secret[512], token_ep[2048];
    char access[8192], refresh[4096], scope[1024];
    char state[64], verifier[128], redirect[2048];
    double expires; /* seconds since 1970; 0: never said */
};

static void take(const struct janas_json *o, const char *k, char *dst,
                 size_t cap)
{
    const char *s = janas_json_str(janas_json_get(o, k));
    snprintf(dst, cap, "%s", s && strlen(s) < cap ? s : "");
}

static int kept_read(const char *name, struct kept *k)
{
    char path[4096];
    memset(k, 0, sizeof *k);
    if (auth_file(name, path, sizeof path) != 0)
        return -1;
    struct janas_json_doc *d = read_json(path);
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    if (!o) {
        janas_json_free(d);
        return -1;
    }
    take(o, "url", k->url, sizeof k->url);
    take(o, "client_id", k->client_id, sizeof k->client_id);
    take(o, "client_secret", k->client_secret, sizeof k->client_secret);
    take(o, "token_endpoint", k->token_ep, sizeof k->token_ep);
    take(o, "access_token", k->access, sizeof k->access);
    take(o, "refresh_token", k->refresh, sizeof k->refresh);
    take(o, "scope", k->scope, sizeof k->scope);
    take(o, "state", k->state, sizeof k->state);
    take(o, "verifier", k->verifier, sizeof k->verifier);
    take(o, "redirect_uri", k->redirect, sizeof k->redirect);
    k->expires = janas_json_num(janas_json_get(o, "expires"), 0);
    janas_json_free(d);
    return 0;
}

static void field(struct janas_buf *b, const char *k, const char *v, int *first)
{
    if (!v || !*v)
        return;
    janas_buf_puts(b, *first ? "{\n  " : ",\n  ");
    *first = 0;
    janas_json_write_str(b, k, strlen(k));
    janas_buf_puts(b, ": ");
    janas_json_write_str(b, v, strlen(v));
}

static int kept_write(const char *name, const struct kept *k)
{
    char path[4096];
    if (auth_file(name, path, sizeof path) != 0)
        return -1;
    struct janas_buf b = {0};
    int first = 1;
    field(&b, "url", k->url, &first);
    field(&b, "client_id", k->client_id, &first);
    field(&b, "client_secret", k->client_secret, &first);
    field(&b, "token_endpoint", k->token_ep, &first);
    field(&b, "access_token", k->access, &first);
    field(&b, "refresh_token", k->refresh, &first);
    field(&b, "scope", k->scope, &first);
    field(&b, "state", k->state, &first);
    field(&b, "verifier", k->verifier, &first);
    field(&b, "redirect_uri", k->redirect, &first);
    if (k->expires > 0)
        janas_buf_printf(&b, "%s\"expires\": %.0f", first ? "{\n  " : ",\n  ",
                         k->expires);
    janas_buf_puts(&b, first ? "{}\n" : "\n}\n");
    int r = write_file(path, &b);
    janas_buf_free(&b);
    return r;
}

/* ---- the requests ---- */

static int is_https_or_local(const char *url)
{
    struct janas_url u;
    char err[160];
    if (janas_url_parse(url, &u, err, sizeof err) != 0)
        return 0;
    /* the tokens go only over TLS, or to this computer */
    return u.tls || !strcmp(u.host, "127.0.0.1") ||
           !strcmp(u.host, "localhost") || !strcmp(u.host, "::1");
}

/* method on url, a body of content type ctype or none: the status, the
   answer (MAX_ANSWER at most) and its WWW-Authenticate */
static int fetch(const char *method, const char *url, const char *ctype,
                 const char *body, int *status, struct janas_buf *out,
                 char *www, size_t www_cap, char *err, size_t err_len)
{
    struct janas_url u;
    if (janas_url_parse(url, &u, err, err_len) != 0)
        return -1;
    char ct[128];
    const char *h[2] = {"Accept: application/json", ct};
    size_t nh = 1;
    if (ctype) {
        snprintf(ct, sizeof ct, "Content-Type: %s", ctype);
        nh = 2;
    }
    struct janas_http c;
    if (janas_http_request(&c, &u, method, h, nh, body, body ? strlen(body) : 0,
                           WAIT_MS, NULL, err, err_len) != 0)
        return -1;
    *status = c.head.status;
    if (www)
        snprintf(www, www_cap, "%s", c.head.www_auth);
    int r;
    while ((r = janas_http_body(&c, out, WAIT_MS)) == 1 && out->n < MAX_ANSWER)
        ;
    janas_http_close(&c);
    if (r < 0 || out->oom) {
        snprintf(err, err_len, "%s: the answer did not come whole", url);
        return -1;
    }
    return 0;
}

/* a JSON object from url (GET), or NULL with the reason */
static struct janas_json_doc *get_json(const char *url, char *err,
                                       size_t err_len)
{
    struct janas_buf b = {0};
    int st = 0;
    struct janas_json_doc *d = NULL;
    if (fetch("GET", url, NULL, NULL, &st, &b, NULL, 0, err, err_len) == 0) {
        if (st / 100 != 2)
            snprintf(err, err_len, "%s: HTTP %d", url, st);
        else if (!(d = janas_json_parse(b.p, b.n, err, err_len)) ||
                 janas_json_root(d)->type != JANAS_JSON_OBJECT) {
            janas_json_free(d);
            d = NULL;
            snprintf(err, err_len, "%s: not a JSON object", url);
        }
    }
    janas_buf_free(&b);
    return d;
}

static void urlenc(struct janas_buf *b, const char *s)
{
    for (const unsigned char *c = (const unsigned char *)s; *c; c++)
        if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
            (*c >= '0' && *c <= '9') || *c == '-' || *c == '.' || *c == '_' ||
            *c == '~')
            janas_buf_put(b, (const char *)c, 1);
        else
            janas_buf_printf(b, "%%%02X", *c);
}

static void form(struct janas_buf *b, const char *k, const char *v)
{
    if (!v || !*v)
        return;
    if (b->n)
        janas_buf_puts(b, "&");
    janas_buf_puts(b, k);
    janas_buf_puts(b, "=");
    urlenc(b, v);
}

/* base64url, no padding, of n bytes */
static void b64url(char *out, size_t cap, const unsigned char *p, size_t n)
{
    struct janas_buf b = {0};
    janas_base64(&b, p, n);
    size_t k = 0;
    for (size_t i = 0; i < b.n && b.p[i] != '=' && k + 1 < cap; i++)
        out[k++] = b.p[i] == '+' ? '-' : b.p[i] == '/' ? '_' : b.p[i];
    out[k] = 0;
    janas_buf_free(&b);
}

static int random_text(char *out, size_t cap, size_t bytes)
{
    unsigned char r[64];
    if (bytes > sizeof r || getrandom(r, bytes, 0) != (ssize_t)bytes)
        return -1;
    b64url(out, cap, r, bytes);
    return 0;
}

/* scheme://host[:port] of url */
static void origin_of(const char *url, char *out, size_t cap)
{
    const char *s = strstr(url, "://");
    const char *e = s ? strchr(s + 3, '/') : NULL;
    size_t n = e ? (size_t)(e - url) : strlen(url);
    snprintf(out, cap, "%.*s", (int)(n < cap ? n : cap - 1), url);
}

/* the path of url, without its query: "" for "/" */
static void path_of(const char *url, char *out, size_t cap)
{
    const char *s = strstr(url, "://");
    const char *p = s ? strchr(s + 3, '/') : NULL;
    size_t n = p ? strcspn(p, "?#") : 0;
    while (n && p[n - 1] == '/')
        n--;
    snprintf(out, cap, "%.*s", (int)(n < cap ? n : cap - 1), p ? p : "");
}

/* resource_metadata="..." of a WWW-Authenticate: Bearer ... */
static void metadata_of(const char *www, char *out, size_t cap)
{
    out[0] = 0;
    const char *k = strstr(www, "resource_metadata=\"");
    if (!k)
        return;
    k += strlen("resource_metadata=\"");
    const char *e = strchr(k, '"');
    if (e && (size_t)(e - k) < cap)
        snprintf(out, cap, "%.*s", (int)(e - k), k);
}

/* the endpoints of the authorization server found for the server at url */
struct found {
    char authorize[2048], token[2048], registration[2048], scope[1024];
};

static int discover(const char *url, struct found *f, char *err, size_t err_len)
{
    memset(f, 0, sizeof *f);
    char www[1024] = "", prm[3200], origin[1024], path[2048];
    struct janas_buf b = {0};
    int st = 0;
    /* the server says where its metadata is when it refuses a request */
    const char *init =
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\","
        "\"params\":{\"protocolVersion\":\"2025-06-18\","
        "\"capabilities\":{},\"clientInfo\":{\"name\":"
        "\"janas\",\"version\":\"1\"}}}";
    if (fetch("POST", url, "application/json", init, &st, &b, www, sizeof www,
              err, err_len) != 0) {
        janas_buf_free(&b);
        return -1;
    }
    janas_buf_free(&b);
    metadata_of(www, prm, sizeof prm);
    origin_of(url, origin, sizeof origin);
    path_of(url, path, sizeof path);
    struct janas_json_doc *d = NULL;
    if (prm[0] && is_https_or_local(prm))
        d = get_json(prm, err, err_len);
    if (!d) { /* RFC 9728's place: with the resource's path, then without */
        snprintf(prm, sizeof prm, "%s/.well-known/oauth-protected-resource%s",
                 origin, path);
        d = get_json(prm, err, err_len);
    }
    if (!d && path[0]) {
        snprintf(prm, sizeof prm, "%s/.well-known/oauth-protected-resource",
                 origin);
        d = get_json(prm, err, err_len);
    }
    char issuer[2048] = "";
    if (d) {
        const struct janas_json *as =
            janas_json_get(janas_json_root(d), "authorization_servers");
        const char *s = as && as->type == JANAS_JSON_ARRAY
                            ? janas_json_str(as->child)
                            : NULL;
        if (s && strlen(s) < sizeof issuer)
            snprintf(issuer, sizeof issuer, "%s", s);
        const struct janas_json *sc =
            janas_json_get(janas_json_root(d), "scopes_supported");
        size_t k = 0;
        for (const struct janas_json *e =
                 sc && sc->type == JANAS_JSON_ARRAY ? sc->child : NULL;
             e; e = e->next) {
            const char *v = janas_json_str(e);
            if (v && k + strlen(v) + 2 < sizeof f->scope)
                k += (size_t)snprintf(f->scope + k, sizeof f->scope - k, "%s%s",
                                      k ? " " : "", v);
        }
        janas_json_free(d);
    }
    if (!issuer[0]) /* older servers: the authorization server is the server */
        snprintf(issuer, sizeof issuer, "%s", origin);
    if (!is_https_or_local(issuer)) {
        snprintf(err, err_len, "the authorization server %s is not https",
                 issuer);
        return -1;
    }
    /* RFC 8414: the well-known part before the issuer's path; OpenID's
       after it */
    char io[1024], ip[2048], m[4400];
    origin_of(issuer, io, sizeof io);
    path_of(issuer, ip, sizeof ip);
    const char *tries[] = {"%s/.well-known/oauth-authorization-server%s",
                           "%s/.well-known/openid-configuration%s",
                           "%s%s/.well-known/openid-configuration"};
    d = NULL;
    for (size_t i = 0; i < 3 && !d; i++) {
        snprintf(m, sizeof m, tries[i], io, ip);
        d = get_json(m, err, err_len);
    }
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_root(d);
    take(o, "authorization_endpoint", f->authorize, sizeof f->authorize);
    take(o, "token_endpoint", f->token, sizeof f->token);
    take(o, "registration_endpoint", f->registration, sizeof f->registration);
    janas_json_free(d);
    if (!f->authorize[0] || !f->token[0]) {
        snprintf(err, err_len, "%s names no authorization or token endpoint",
                 m);
        return -1;
    }
    if (!is_https_or_local(f->authorize) || !is_https_or_local(f->token)) {
        snprintf(err, err_len,
                 "the authorization server's endpoints are not "
                 "https");
        return -1;
    }
    return 0;
}

/* this program registered as a client of the authorization server */
static int registration(const struct found *f, const char *redirect,
                        struct kept *k, char *err, size_t err_len)
{
    if (!f->registration[0] || !is_https_or_local(f->registration)) {
        snprintf(err, err_len,
                 "the authorization server takes no "
                 "registration of clients: give a client_id "
                 "in mcp.json");
        return -1;
    }
    struct janas_buf body = {0}, ans = {0};
    janas_buf_puts(&body, "{\"client_name\": \"Janas\", \"redirect_uris\": [");
    janas_json_write_str(&body, redirect, strlen(redirect));
    janas_buf_puts(&body, "], \"grant_types\": [\"authorization_code\", "
                          "\"refresh_token\"], \"response_types\": "
                          "[\"code\"], \"token_endpoint_auth_method\": "
                          "\"none\"}");
    int st = 0, r = -1;
    if (!body.oom && fetch("POST", f->registration, "application/json", body.p,
                           &st, &ans, NULL, 0, err, err_len) == 0) {
        struct janas_json_doc *d =
            st / 100 == 2 ? janas_json_parse(ans.p, ans.n, err, err_len) : NULL;
        const struct janas_json *o = d ? janas_json_root(d) : NULL;
        if (o) {
            take(o, "client_id", k->client_id, sizeof k->client_id);
            take(o, "client_secret", k->client_secret, sizeof k->client_secret);
        }
        janas_json_free(d);
        if (k->client_id[0])
            r = 0;
        else
            snprintf(err, err_len, "the registration of the client: HTTP %d",
                     st);
    }
    janas_buf_free(&body);
    janas_buf_free(&ans);
    return r;
}

int janas_oauth_begin(const char *name, const char *url,
                      const char *redirect_uri, char *auth_url, size_t cap,
                      char *err, size_t err_len)
{
    if (!name || !url || !redirect_uri || !auth_url) {
        snprintf(err, err_len, "invalid argument");
        return -1;
    }
    if (!is_https_or_local(url)) {
        snprintf(err, err_len, "%s is not https: no token is sent to it", url);
        return -1;
    }
    struct found f;
    if (discover(url, &f, err, err_len) != 0)
        return -1;
    struct kept k;
    if (kept_read(name, &k) != 0 || strcmp(k.url, url) ||
        strcmp(k.redirect, redirect_uri) || strcmp(k.token_ep, f.token))
        memset(&k, 0, sizeof k); /* a client of this server, for this place */
    if (!k.client_id[0] && registration(&f, redirect_uri, &k, err, err_len))
        return -1;
    snprintf(k.url, sizeof k.url, "%s", url);
    snprintf(k.token_ep, sizeof k.token_ep, "%s", f.token);
    snprintf(k.redirect, sizeof k.redirect, "%s", redirect_uri);
    snprintf(k.scope, sizeof k.scope, "%s", f.scope);
    unsigned char h[32];
    char challenge[64];
    if (random_text(k.verifier, sizeof k.verifier, 48) != 0 ||
        random_text(k.state, sizeof k.state, 24) != 0 ||
        janas_tls_sha256(k.verifier, strlen(k.verifier), h) != 0) {
        snprintf(err, err_len, "no random bytes");
        return -1;
    }
    b64url(challenge, sizeof challenge, h, 32);
    struct janas_buf q = {0};
    form(&q, "response_type", "code");
    form(&q, "client_id", k.client_id);
    form(&q, "redirect_uri", redirect_uri);
    form(&q, "code_challenge", challenge);
    form(&q, "code_challenge_method", "S256");
    form(&q, "state", k.state);
    form(&q, "resource", url);
    form(&q, "scope", k.scope);
    int n = q.oom
                ? -1
                : snprintf(auth_url, cap, "%s%c%.*s", f.authorize,
                           strchr(f.authorize, '?') ? '&' : '?', (int)q.n, q.p);
    janas_buf_free(&q);
    if (n < 0 || (size_t)n >= cap) {
        snprintf(err, err_len, "the address of the sign-in is too long");
        return -1;
    }
    if (kept_write(name, &k) != 0) {
        snprintf(err, err_len, "cannot keep the sign-in's state");
        return -1;
    }
    return 0;
}

/* the token endpoint asked: a code traded, or a refresh token renewed */
static int token(struct kept *k, const char *grant, const char *code, char *err,
                 size_t err_len)
{
    struct janas_buf body = {0}, ans = {0};
    form(&body, "grant_type", grant);
    if (code) {
        form(&body, "code", code);
        form(&body, "redirect_uri", k->redirect);
        form(&body, "code_verifier", k->verifier);
    } else
        form(&body, "refresh_token", k->refresh);
    form(&body, "client_id", k->client_id);
    form(&body, "client_secret", k->client_secret);
    form(&body, "resource", k->url);
    int st = 0, r = -1;
    if (!body.oom && is_https_or_local(k->token_ep) &&
        fetch("POST", k->token_ep, "application/x-www-form-urlencoded", body.p,
              &st, &ans, NULL, 0, err, err_len) == 0) {
        struct janas_json_doc *d =
            st / 100 == 2 ? janas_json_parse(ans.p, ans.n, err, err_len) : NULL;
        const struct janas_json *o = d ? janas_json_root(d) : NULL;
        const char *at =
            o ? janas_json_str(janas_json_get(o, "access_token")) : NULL;
        int ok =
            at && *at && strlen(at) < sizeof k->access && !strpbrk(at, "\r\n");
        if (ok) {
            snprintf(k->access, sizeof k->access, "%s", at);
            const char *rt = janas_json_str(janas_json_get(o, "refresh_token"));
            if (rt && *rt && strlen(rt) < sizeof k->refresh)
                snprintf(k->refresh, sizeof k->refresh, "%s", rt);
            double in = janas_json_num(janas_json_get(o, "expires_in"), 0);
            k->expires = in > 0 ? (double)time(NULL) + in : 0;
            r = 0;
        } else
            snprintf(err, err_len, "the token endpoint: HTTP %d", st);
        janas_json_free(d);
    }
    janas_buf_free(&body);
    janas_buf_free(&ans);
    return r;
}

int janas_oauth_finish(const char *state, const char *code, char *name,
                       size_t name_cap, char *err, size_t err_len)
{
    char dir[3800];
    if (!state || !*state || !code || !*code || auth_dir(dir, sizeof dir)) {
        snprintf(err, err_len, "no sign-in to finish");
        return -1;
    }
    /* the server whose sign-in waits for this state */
    DIR *dd = opendir(dir);
    struct dirent *de;
    int found = 0;
    struct kept k;
    while (dd && !found && (de = readdir(dd))) {
        size_t l = strlen(de->d_name);
        if (l < 6 || strcmp(de->d_name + l - 5, ".json"))
            continue;
        char nm[256];
        snprintf(nm, sizeof nm, "%.*s", (int)(l - 5), de->d_name);
        if (kept_read(nm, &k) == 0 && k.state[0] && !strcmp(k.state, state)) {
            found = 1;
            snprintf(name, name_cap, "%s", nm);
        }
    }
    if (dd)
        closedir(dd);
    if (!found) {
        snprintf(err, err_len, "this sign-in is unknown or already used");
        return -1;
    }
    int r = token(&k, "authorization_code", code, err, err_len);
    k.state[0] = 0; /* used once */
    k.verifier[0] = 0;
    if (kept_write(name, &k) != 0 && r == 0) {
        snprintf(err, err_len, "cannot keep the token");
        r = -1;
    }
    return r;
}

int janas_oauth_header(const char *name, const char *url, char *out, size_t cap)
{
    struct kept k;
    if (kept_read(name, &k) != 0 || !k.access[0] || strcmp(k.url, url))
        return -1;
    /* renewed a minute before it expires */
    if (k.expires > 0 && (double)time(NULL) + 60 > k.expires) {
        char err[300];
        if (!k.refresh[0] || token(&k, "refresh_token", NULL, err, sizeof err))
            return -1;
        kept_write(name, &k);
    }
    int n = snprintf(out, cap, "Authorization: Bearer %s", k.access);
    return n > 0 && (size_t)n < cap ? 0 : -1;
}

int janas_oauth_signed_in(const char *name)
{
    struct kept k;
    return kept_read(name, &k) == 0 && k.access[0];
}

int janas_oauth_forget(const char *name)
{
    char path[4096];
    if (auth_file(name, path, sizeof path) != 0)
        return -1;
    return remove(path) == 0 ? 0 : -1;
}
