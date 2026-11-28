/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * release_sign.c - the project's signature on a release's SHA256SUMS
 * (src/common/release_sig.h), which Brebus checks before it installs an
 * update. Not shipped: it is the maintainer's.
 *
 *   release_sign keygen <key file>    a new key, in a file only its owner
 *                                     reads (never overwritten); prints the
 *                                     public key, for update_key.h
 *   release_sign public <key file>    the public key of a key
 *   release_sign sign <key file> <file>          writes <file>.sig
 *   release_sign verify <public hex> <file> [<sig file>]
 *                                     0 if <file>.sig signs it
 */
#include "common/release_sig.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _WIN32
#define O_NOFOLLOW 0
#endif

static void wipe(void *p, size_t n)
{
    volatile unsigned char *v = p;
    while (n--)
        *v++ = 0;
}

/* a whole file into memory; NULL on error (said) */
static char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "release_sign: %s: %s\n", path, strerror(errno));
        return NULL;
    }
    size_t cap = 1 << 16, len = 0;
    char *p = malloc(cap + 1);
    for (size_t got; p && (got = fread(p + len, 1, cap - len, f)) > 0;) {
        len += got;
        if (len == cap) {
            char *q = realloc(p, 2 * cap + 1);
            if (!q) {
                free(p);
                p = NULL;
                break;
            }
            p = q;
            cap *= 2;
        }
    }
    int bad = ferror(f);
    fclose(f);
    if (!p || bad) {
        fprintf(stderr, "release_sign: %s: could not read it\n", path);
        free(p);
        return NULL;
    }
    p[len] = 0;
    *n = len;
    return p;
}

static int read_key(const char *path, uint8_t priv[JANAS_SIG_PRIV])
{
    size_t n = 0;
    char *s = slurp(path, &n);
    if (!s)
        return -1;
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' '))
        n--;
    int rc = janas_sig_unhex(s, n, priv, JANAS_SIG_PRIV);
    wipe(s, n);
    free(s);
    if (rc != 0)
        fprintf(stderr, "release_sign: %s: not a key (64 hex digits)\n", path);
    return rc;
}

static void print_public(const uint8_t pub[JANAS_SIG_PUB])
{
    char hex[2 * JANAS_SIG_PUB + 1];
    janas_sig_hex(pub, JANAS_SIG_PUB, hex);
    printf("%s\n", hex);
}

static int keygen(const char *path)
{
    uint8_t priv[JANAS_SIG_PRIV], pub[JANAS_SIG_PUB];
    if (janas_sig_keygen(priv, pub) != 0) {
        fprintf(stderr, "release_sign: no key made\n");
        return 1;
    }
    /* never over another key: a lost key is lost for good */
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) {
        fprintf(stderr, "release_sign: %s: %s\n", path, strerror(errno));
        wipe(priv, sizeof priv);
        return 1;
    }
    char hex[2 * JANAS_SIG_PRIV + 2];
    janas_sig_hex(priv, JANAS_SIG_PRIV, hex);
    hex[2 * JANAS_SIG_PRIV] = '\n';
    ssize_t w = write(fd, hex, 2 * JANAS_SIG_PRIV + 1);
    wipe(hex, sizeof hex);
    wipe(priv, sizeof priv);
    if (w != 2 * JANAS_SIG_PRIV + 1 || close(fd) != 0) {
        fprintf(stderr, "release_sign: %s: not written\n", path);
        return 1;
    }
    print_public(pub);
    return 0;
}

static int public_of(const char *path)
{
    uint8_t priv[JANAS_SIG_PRIV], pub[JANAS_SIG_PUB];
    if (read_key(path, priv) != 0)
        return 1;
    int rc = janas_sig_public(priv, pub);
    wipe(priv, sizeof priv);
    if (rc != 0) {
        fprintf(stderr, "release_sign: %s: not a P-256 key\n", path);
        return 1;
    }
    print_public(pub);
    return 0;
}

static int sign(const char *key, const char *path)
{
    uint8_t priv[JANAS_SIG_PRIV];
    if (read_key(key, priv) != 0)
        return 1;
    size_t n = 0;
    char *msg = slurp(path, &n);
    char line[JANAS_SIG_LINE + 1];
    int rc = msg ? janas_sig_sign(priv, msg, n, line, sizeof line) : -1;
    wipe(priv, sizeof priv);
    free(msg);
    if (rc != 0) {
        fprintf(stderr, "release_sign: %s: not signed\n", path);
        return 1;
    }
    char out[4096];
    snprintf(out, sizeof out, "%s.sig", path);
    FILE *f = fopen(out, "wb");
    if (!f || fputs(line, f) < 0 || fclose(f) != 0) {
        fprintf(stderr, "release_sign: %s: not written\n", out);
        return 1;
    }
    printf("%s\n", out);
    return 0;
}

static int verify(const char *pubhex, const char *path, const char *sigpath)
{
    uint8_t pub[JANAS_SIG_PUB];
    if (janas_sig_unhex(pubhex, strlen(pubhex), pub, sizeof pub) != 0) {
        fprintf(stderr, "release_sign: not a public key (130 hex digits)\n");
        return 2;
    }
    char sp[4096];
    if (!sigpath) {
        snprintf(sp, sizeof sp, "%s.sig", path);
        sigpath = sp;
    }
    size_t n = 0, sn = 0;
    char *msg = slurp(path, &n), *sig = msg ? slurp(sigpath, &sn) : NULL;
    int rc = sig ? janas_sig_verify(pub, msg, n, sig, sn) : -1;
    free(msg);
    free(sig);
    printf("%s: %s\n", path, rc == 0 ? "signed" : "NOT signed by this key");
    return rc == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "keygen") == 0)
        return keygen(argv[2]);
    if (argc == 3 && strcmp(argv[1], "public") == 0)
        return public_of(argv[2]);
    if (argc == 4 && strcmp(argv[1], "sign") == 0)
        return sign(argv[2], argv[3]);
    if ((argc == 4 || argc == 5) && strcmp(argv[1], "verify") == 0)
        return verify(argv[2], argv[3], argc == 5 ? argv[4] : NULL);
    fprintf(stderr,
            "usage: release_sign keygen <key file>\n"
            "       release_sign public <key file>\n"
            "       release_sign sign <key file> <file>\n"
            "       release_sign verify <public hex> <file> [<sig file>]\n");
    return 2;
}
