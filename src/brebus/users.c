/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * users.c - the accounts of Brebus served beyond its own window (brebus
 * --serve): users.json in the data folder, readable by its owner alone.
 *
 * A password is kept as PBKDF2-HMAC-SHA256 of it, a salt of 16 random
 * bytes and 600,000 iterations (OWASP's figure for SHA-256 in 2023), never
 * itself. The second factor is TOTP (RFC 6238: HMAC-SHA1, 30 seconds, 6
 * digits, a step either side allowed), its secret 20 random bytes in
 * base32 as authenticator apps take it, with ten recovery codes kept as
 * their SHA-256, each good once. The crypto is PSA's, of the Mbed TLS the
 * MCP client carries, under its lock (janas_tls_lock).
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <psa/crypto.h>

#include "brebus.h"
#include "common/mcp_tls.h"
#include "common/proc.h"

#define ITERATIONS 600000u

static void hex(const uint8_t *p, size_t n, char *out)
{
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++)
        out[2 * i] = d[p[i] >> 4], out[2 * i + 1] = d[p[i] & 15];
    out[2 * n] = 0;
}

static int unhex(const char *s, uint8_t *out, size_t n)
{
    if (!s || strlen(s) != 2 * n)
        return -1;
    for (size_t i = 0; i < 2 * n; i++) {
        char c = s[i];
        int v = c >= '0' && c <= '9'   ? c - '0'
                : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                       : -1;
        if (v < 0)
            return -1;
        if (i & 1)
            out[i / 2] |= (uint8_t)v;
        else
            out[i / 2] = (uint8_t)(v << 4);
    }
    return 0;
}

int brebus_random(void *p, size_t n)
{
    janas_tls_lock();
    int rc = janas_tls_crypto() == 0 && psa_generate_random(p, n) == PSA_SUCCESS
                 ? 0
                 : -1;
    janas_tls_unlock();
    return rc;
}

static int pbkdf2(const char *pw, const uint8_t salt[16], uint32_t iters,
                  uint8_t out[32])
{
    janas_tls_lock();
    int rc = -1;
    if (janas_tls_crypto() == 0) {
        psa_key_derivation_operation_t op = PSA_KEY_DERIVATION_OPERATION_INIT;
        rc =
            psa_key_derivation_setup(
                &op, PSA_ALG_PBKDF2_HMAC(PSA_ALG_SHA_256)) == PSA_SUCCESS &&
                    psa_key_derivation_input_integer(
                        &op, PSA_KEY_DERIVATION_INPUT_COST, iters) ==
                        PSA_SUCCESS &&
                    psa_key_derivation_input_bytes(
                        &op, PSA_KEY_DERIVATION_INPUT_SALT, salt, 16) ==
                        PSA_SUCCESS &&
                    psa_key_derivation_input_bytes(
                        &op, PSA_KEY_DERIVATION_INPUT_PASSWORD,
                        (const uint8_t *)pw, strlen(pw)) == PSA_SUCCESS &&
                    psa_key_derivation_output_bytes(&op, out, 32) == PSA_SUCCESS
                ? 0
                : -1;
        psa_key_derivation_abort(&op);
    }
    janas_tls_unlock();
    return rc;
}

static int sha256(const void *p, size_t n, uint8_t out[32])
{
    size_t len = 0;
    janas_tls_lock();
    int rc = janas_tls_crypto() == 0 &&
                     psa_hash_compute(PSA_ALG_SHA_256, p, n, out, 32, &len) ==
                         PSA_SUCCESS
                 ? 0
                 : -1;
    janas_tls_unlock();
    return rc;
}

/* compared in a time that does not depend on where they differ */
static int same(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint8_t d = 0;
    for (size_t i = 0; i < n; i++)
        d |= a[i] ^ b[i];
    return d == 0;
}

/* ---- base32 (RFC 4648), as authenticator apps take a secret ---- */

static const char B32[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

static void b32(const uint8_t *p, size_t n, char *out)
{
    uint32_t acc = 0;
    int bits = 0;
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        acc = (acc << 8) | p[i];
        bits += 8;
        while (bits >= 5) {
            out[k++] = B32[(acc >> (bits - 5)) & 31];
            bits -= 5;
        }
    }
    if (bits)
        out[k++] = B32[(acc << (5 - bits)) & 31];
    out[k] = 0;
}

static int unb32(const char *s, uint8_t *out, size_t cap)
{
    uint32_t acc = 0;
    int bits = 0;
    size_t k = 0;
    for (; *s; s++) {
        const char *c = strchr(B32, *s >= 'a' && *s <= 'z' ? *s - 32 : *s);
        if (*s == '=' || *s == ' ')
            continue;
        if (!c)
            return -1;
        acc = (acc << 5) | (uint32_t)(c - B32);
        bits += 5;
        if (bits >= 8) {
            if (k >= cap)
                return -1;
            out[k++] = (uint8_t)(acc >> (bits - 8));
            bits -= 8;
        }
    }
    return (int)k;
}

/* the TOTP code of secret (base32) at step t (RFC 6238, RFC 4226) */
static int totp_at(const char *secret, uint64_t t, uint32_t *code)
{
    uint8_t key[64], msg[8], mac[20];
    int kn = unb32(secret, key, sizeof key);
    if (kn <= 0)
        return -1;
    for (int i = 7; i >= 0; i--, t >>= 8)
        msg[i] = (uint8_t)t;
    janas_tls_lock();
    int rc = -1;
    if (janas_tls_crypto() == 0) {
        psa_key_attributes_t at = PSA_KEY_ATTRIBUTES_INIT;
        psa_set_key_type(&at, PSA_KEY_TYPE_HMAC);
        psa_set_key_usage_flags(&at, PSA_KEY_USAGE_SIGN_MESSAGE);
        psa_set_key_algorithm(&at, PSA_ALG_HMAC(PSA_ALG_SHA_1));
        psa_key_id_t id = 0;
        size_t mn = 0;
        if (psa_import_key(&at, key, (size_t)kn, &id) == PSA_SUCCESS) {
            rc = psa_mac_compute(id, PSA_ALG_HMAC(PSA_ALG_SHA_1), msg,
                                 sizeof msg, mac, sizeof mac,
                                 &mn) == PSA_SUCCESS &&
                         mn == 20
                     ? 0
                     : -1;
            psa_destroy_key(id);
        }
    }
    janas_tls_unlock();
    if (rc)
        return -1;
    int off = mac[19] & 15;
    uint32_t v = ((uint32_t)(mac[off] & 0x7f) << 24) |
                 ((uint32_t)mac[off + 1] << 16) |
                 ((uint32_t)mac[off + 2] << 8) | mac[off + 3];
    *code = v % 1000000;
    return 0;
}

int brebus_totp_check(const char *secret, const char *code, time_t now)
{
    if (!code || strlen(code) != 6 || strspn(code, "0123456789") != 6)
        return 0;
    uint32_t want = (uint32_t)atoi(code), got;
    uint64_t t = (uint64_t)now / 30;
    int ok = 0;
    for (int d = -1; d <= 1; d++)
        if (totp_at(secret, t + (uint64_t)(int64_t)d, &got) == 0 && got == want)
            ok = 1;
    return ok;
}

int brebus_totp_code(const char *secret, time_t now, char out[7])
{
    uint32_t c;
    if (totp_at(secret, (uint64_t)now / 30, &c) != 0)
        return -1;
    snprintf(out, 7, "%06u", c);
    return 0;
}

int brebus_totp_secret(char out[33])
{
    uint8_t s[20];
    if (brebus_random(s, sizeof s) != 0)
        return -1;
    b32(s, sizeof s, out);
    return 0;
}

/* ---- the accounts, users.json ---- */

static void users_path(const struct brebus *b, char *out, size_t n)
{
    snprintf(out, n, "%s/users.json", b->data_dir);
}

int brebus_users_load(struct brebus *b)
{
    pthread_mutex_lock(&b->mu);
    b->n_users = 0;
    char p[4200], err[160];
    users_path(b, p, sizeof p);
    FILE *f = fopen(p, "rb");
    struct janas_buf t = {0};
    char buf[16384];
    size_t r;
    while (f && (r = fread(buf, 1, sizeof buf, f)) > 0)
        janas_buf_put(&t, buf, r);
    if (f)
        fclose(f);
    struct janas_json_doc *d =
        t.n ? janas_json_parse(t.p, t.n, err, sizeof err) : NULL;
    const struct janas_json *a =
        d ? janas_json_get(janas_json_root(d), "users") : NULL;
    for (const struct janas_json *e =
             a && a->type == JANAS_JSON_ARRAY ? a->child : NULL;
         e && b->n_users < BREBUS_USERS; e = e->next) {
        struct brebus_user *u = &b->users[b->n_users];
        memset(u, 0, sizeof *u);
        const char *name = janas_json_str(janas_json_get(e, "name"));
        const char *role = janas_json_str(janas_json_get(e, "role"));
        const char *totp = janas_json_str(janas_json_get(e, "totp"));
        const struct janas_json *on = janas_json_get(e, "totp_on");
        if (!name || strlen(name) >= sizeof u->name ||
            unhex(janas_json_str(janas_json_get(e, "salt")), u->salt, 16) ||
            unhex(janas_json_str(janas_json_get(e, "hash")), u->hash, 32))
            continue;
        snprintf(u->name, sizeof u->name, "%s", name);
        u->admin = role && !strcmp(role, "admin");
        u->iters = (uint32_t)janas_json_num(janas_json_get(e, "iters"), 0);
        if (totp && strlen(totp) < sizeof u->totp)
            snprintf(u->totp, sizeof u->totp, "%s", totp);
        u->totp_on = on && on->type == JANAS_JSON_TRUE && u->totp[0];
        const struct janas_json *rc = janas_json_get(e, "recovery");
        for (const struct janas_json *x =
                 rc && rc->type == JANAS_JSON_ARRAY ? rc->child : NULL;
             x && u->n_rec < BREBUS_RECOVERY; x = x->next)
            if (unhex(janas_json_str(x), u->rec[u->n_rec], 32) == 0)
                u->n_rec++;
        b->n_users++;
    }
    janas_json_free(d);
    janas_buf_free(&t);
    pthread_mutex_unlock(&b->mu);
    return 0;
}

/* b->mu held: users.json written again, readable by its owner alone */
static int save_locked(struct brebus *b)
{
    struct janas_buf o = {0};
    janas_buf_puts(&o, "{\"users\": [");
    for (int i = 0; i < b->n_users; i++) {
        const struct brebus_user *u = &b->users[i];
        char s[33], h[65];
        hex(u->salt, 16, s);
        hex(u->hash, 32, h);
        janas_buf_puts(&o, i ? ",\n  {\"name\": " : "\n  {\"name\": ");
        janas_json_write_str(&o, u->name, strlen(u->name));
        janas_buf_printf(&o,
                         ", \"role\": \"%s\", \"salt\": \"%s\", \"hash\": "
                         "\"%s\", \"iters\": %u, \"totp\": ",
                         u->admin ? "admin" : "user", s, h, u->iters);
        janas_json_write_str(&o, u->totp, strlen(u->totp));
        janas_buf_printf(&o, ", \"totp_on\": %s, \"recovery\": [",
                         u->totp_on ? "true" : "false");
        for (int k = 0; k < u->n_rec; k++) {
            char r[65];
            hex(u->rec[k], 32, r);
            janas_buf_printf(&o, "%s\"%s\"", k ? ", " : "", r);
        }
        janas_buf_puts(&o, "]}");
    }
    janas_buf_puts(&o, "\n]}\n");
    char p[4200], tmp[4300];
    users_path(b, p, sizeof p);
    snprintf(tmp, sizeof tmp, "%s.new", p);
    int rc = -1;
#ifdef _WIN32
    FILE *f = o.oom ? NULL : fopen(tmp, "wb");
#else
    int fd = o.oom ? -1 : open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
#endif
    if (f) {
        rc = fwrite(o.p, 1, o.n, f) == o.n ? 0 : -1;
        if (fclose(f) != 0)
            rc = -1;
        if (rc == 0 && rename(tmp, p) != 0)
            rc = -1;
        if (rc)
            remove(tmp);
    }
    janas_buf_free(&o);
    return rc;
}

static struct brebus_user *find_locked(struct brebus *b, const char *name)
{
    for (int i = 0; name && i < b->n_users; i++)
        if (!strcmp(b->users[i].name, name))
            return &b->users[i];
    return NULL;
}

int brebus_user_add(struct brebus *b, const char *name, const char *pw,
                    int admin, char *why, size_t wn)
{
    size_t nl = name ? strlen(name) : 0;
    if (!nl || nl >= sizeof b->users[0].name ||
        strspn(name, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                     "0123456789._-") != nl) {
        snprintf(why, wn, "a name is letters, digits, '.', '_' or '-'");
        return -1;
    }
    if (!pw || strlen(pw) < 12) {
        snprintf(why, wn, "a password is 12 characters or more");
        return -1;
    }
    struct brebus_user u = {.admin = admin, .iters = ITERATIONS};
    snprintf(u.name, sizeof u.name, "%s", name);
    if (brebus_random(u.salt, sizeof u.salt) != 0 ||
        pbkdf2(pw, u.salt, u.iters, u.hash) != 0) {
        snprintf(why, wn, "the password could not be hashed");
        return -1;
    }
    pthread_mutex_lock(&b->mu);
    struct brebus_user *had = find_locked(b, name);
    int rc = 0;
    if (had) {
        /* a new password: the second factor stays */
        memcpy(had->salt, u.salt, sizeof u.salt);
        memcpy(had->hash, u.hash, sizeof u.hash);
        had->iters = u.iters;
        had->admin = admin;
    } else if (b->n_users >= BREBUS_USERS) {
        snprintf(why, wn, "there are %d accounts already", BREBUS_USERS);
        rc = -1;
    } else {
        b->users[b->n_users++] = u;
    }
    if (rc == 0 && save_locked(b) != 0) {
        snprintf(why, wn, "users.json could not be written in %s", b->data_dir);
        rc = -1;
    }
    pthread_mutex_unlock(&b->mu);
    return rc;
}

int brebus_user_check(struct brebus *b, const char *name, const char *pw,
                      struct brebus_user *out)
{
    pthread_mutex_lock(&b->mu);
    struct brebus_user *u = find_locked(b, name);
    struct brebus_user copy = u ? *u : (struct brebus_user){0};
    pthread_mutex_unlock(&b->mu);
    /* hashed even for a name there is not: the time does not tell */
    uint8_t salt[16] = {0}, h[32];
    int ok = pbkdf2(pw ? pw : "", u ? copy.salt : salt,
                    u && copy.iters ? copy.iters : ITERATIONS, h) == 0 &&
             u && copy.iters && same(h, copy.hash, 32);
    if (ok && out)
        *out = copy;
    return ok;
}

int brebus_user_recovery_use(struct brebus *b, const char *name,
                             const char *code)
{
    uint8_t h[32];
    if (!code || sha256(code, strlen(code), h) != 0)
        return 0;
    pthread_mutex_lock(&b->mu);
    struct brebus_user *u = find_locked(b, name);
    int ok = 0;
    for (int k = 0; u && k < u->n_rec && !ok; k++)
        if (same(u->rec[k], h, 32)) {
            /* spent: the last one takes its place */
            memmove(u->rec[k], u->rec[u->n_rec - 1], 32);
            u->n_rec--;
            ok = save_locked(b) == 0;
        }
    pthread_mutex_unlock(&b->mu);
    return ok;
}

int brebus_user_totp_on(struct brebus *b, const char *name, const char *secret,
                        char codes[BREBUS_RECOVERY][12])
{
    pthread_mutex_lock(&b->mu);
    struct brebus_user *u = find_locked(b, name);
    int rc = -1;
    if (u && strlen(secret) < sizeof u->totp) {
        snprintf(u->totp, sizeof u->totp, "%s", secret);
        u->totp_on = 1;
        u->n_rec = 0;
        rc = 0;
        for (int k = 0; k < BREBUS_RECOVERY && rc == 0; k++) {
            /* xxxxx-xxxxx: ten letters and digits a person can type */
            static const char A[] = "abcdefghjkmnpqrstuvwxyz23456789";
            uint8_t r[10];
            rc = brebus_random(r, sizeof r);
            for (int i = 0, j = 0; rc == 0 && i < 10; i++) {
                codes[k][j++] = A[r[i] % (sizeof A - 1)];
                if (i == 4)
                    codes[k][j++] = '-';
            }
            codes[k][11] = 0;
            if (rc == 0 && sha256(codes[k], 11, u->rec[k]) == 0)
                u->n_rec++;
        }
        if (rc == 0)
            rc = save_locked(b);
    }
    pthread_mutex_unlock(&b->mu);
    return rc;
}

int brebus_users_count(struct brebus *b)
{
    pthread_mutex_lock(&b->mu);
    int n = b->n_users;
    pthread_mutex_unlock(&b->mu);
    return n;
}

int brebus_user_system(struct brebus *b, const char *name, const char *pw,
                       struct brebus_user *out)
{
    if (!name || !pw || strlen(name) >= sizeof out->name ||
        strchr(name, '\n') || strchr(pw, '\n') || strlen(pw) > 500)
        return 0;
    char exe[4300], role[16] = "";
    snprintf(exe, sizeof exe, "%s/janas-brebus-helper", b->bin_dir);
    int in[2], outp[2];
    if (janas_pipe(in) != 0)
        return 0;
    if (janas_pipe(outp) != 0) {
        close(in[0]), close(in[1]);
        return 0;
    }
    const char *argv[] = {exe, "auth", NULL};
    struct janas_proc p;
    char err[200];
    int rc = janas_proc_start(&p, argv, NULL, NULL, in[0], outp[1], -1, 0, err,
                              sizeof err);
    close(in[0]);
    close(outp[1]);
    if (rc == 0) {
        char line[600];
        int n = snprintf(line, sizeof line, "%s\n%s\n", name, pw);
        if (write(in[1], line, (size_t)n) != n)
            rc = -1;
        memset(line, 0, sizeof line);
    }
    close(in[1]);
    if (rc == 0) {
        ssize_t r = read(outp[0], role, sizeof role - 1);
        role[r > 0 ? r : 0] = 0;
    }
    close(outp[0]);
    int code = -1;
    if (rc == 0 && janas_proc_wait(&p, 30000, &code) != 1) {
        janas_proc_kill(&p, 1);
        janas_proc_wait(&p, -1, &code);
    }
    if (rc != 0 || code != 0)
        return 0;
    int admin = !strncmp(role, "admin", 5);
    if (!admin && strncmp(role, "user", 4))
        return 0;
    /* the account kept here: its role, its second factor */
    pthread_mutex_lock(&b->mu);
    struct brebus_user *u = find_locked(b, name);
    if (!u && b->n_users < BREBUS_USERS) {
        u = &b->users[b->n_users++];
        memset(u, 0, sizeof *u);
        snprintf(u->name, sizeof u->name, "%s", name);
    }
    int ok = u != NULL;
    if (u) {
        u->admin = admin;
        u->iters = 0; /* no password of Brebus's: the system's alone */
        memset(u->hash, 0, sizeof u->hash);
        save_locked(b);
        *out = *u;
    }
    pthread_mutex_unlock(&b->mu);
    return ok;
}
