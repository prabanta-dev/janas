/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* The project's signature on a release (release_sig.h), with Mbed TLS's PSA:
   every key a volatile one, destroyed as soon as it has been used. */
#include "common/release_sig.h"

#include <psa/crypto.h>
#include <string.h>

#define SIG_ALG PSA_ALG_ECDSA(PSA_ALG_SHA_256)

/* a buffer that held a private key, cleared so that the compiler keeps it */
static void wipe(void *p, size_t n)
{
    volatile uint8_t *v = p;
    while (n--)
        *v++ = 0;
}

static int start(void)
{
    return psa_crypto_init() == PSA_SUCCESS ? 0 : -1;
}

/* a key of type kind, for usage, from its bytes */
static int import(psa_key_type_t kind, psa_key_usage_t usage, const uint8_t *k,
                  size_t n, psa_key_id_t *id)
{
    psa_key_attributes_t a = PSA_KEY_ATTRIBUTES_INIT;
    psa_set_key_type(&a, kind);
    psa_set_key_bits(&a, 256);
    psa_set_key_usage_flags(&a, usage);
    psa_set_key_algorithm(&a, SIG_ALG);
    int rc = psa_import_key(&a, k, n, id) == PSA_SUCCESS ? 0 : -1;
    psa_reset_key_attributes(&a);
    return rc;
}

int janas_sig_keygen(uint8_t priv[JANAS_SIG_PRIV], uint8_t pub[JANAS_SIG_PUB])
{
    if (start() != 0)
        return -1;
    psa_key_attributes_t a = PSA_KEY_ATTRIBUTES_INIT;
    psa_set_key_type(&a, PSA_KEY_TYPE_ECC_KEY_PAIR(PSA_ECC_FAMILY_SECP_R1));
    psa_set_key_bits(&a, 256);
    psa_set_key_usage_flags(&a, PSA_KEY_USAGE_EXPORT | PSA_KEY_USAGE_SIGN_HASH);
    psa_set_key_algorithm(&a, SIG_ALG);
    psa_key_id_t id = 0;
    size_t n = 0, m = 0;
    int rc =
        psa_generate_key(&a, &id) == PSA_SUCCESS &&
                psa_export_key(id, priv, JANAS_SIG_PRIV, &n) == PSA_SUCCESS &&
                n == JANAS_SIG_PRIV &&
                psa_export_public_key(id, pub, JANAS_SIG_PUB, &m) ==
                    PSA_SUCCESS &&
                m == JANAS_SIG_PUB
            ? 0
            : -1;
    psa_destroy_key(id);
    psa_reset_key_attributes(&a);
    if (rc != 0)
        wipe(priv, JANAS_SIG_PRIV);
    return rc;
}

int janas_sig_public(const uint8_t priv[JANAS_SIG_PRIV],
                     uint8_t pub[JANAS_SIG_PUB])
{
    psa_key_id_t id = 0;
    size_t m = 0;
    if (start() != 0 ||
        import(PSA_KEY_TYPE_ECC_KEY_PAIR(PSA_ECC_FAMILY_SECP_R1),
               PSA_KEY_USAGE_SIGN_HASH, priv, JANAS_SIG_PRIV, &id) != 0)
        return -1;
    int rc = psa_export_public_key(id, pub, JANAS_SIG_PUB, &m) == PSA_SUCCESS &&
                     m == JANAS_SIG_PUB
                 ? 0
                 : -1;
    psa_destroy_key(id);
    return rc;
}

void janas_sig_hex(const uint8_t *p, size_t n, char *out)
{
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = d[p[i] >> 4];
        out[2 * i + 1] = d[p[i] & 15];
    }
    out[2 * n] = 0;
}

static int nibble(char c)
{
    return c >= '0' && c <= '9'   ? c - '0'
           : c >= 'a' && c <= 'f' ? c - 'a' + 10
           : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                  : -1;
}

int janas_sig_unhex(const char *s, size_t len, uint8_t *out, size_t n)
{
    if (len != 2 * n)
        return -1;
    for (size_t i = 0; i < n; i++) {
        int h = nibble(s[2 * i]), l = nibble(s[2 * i + 1]);
        if (h < 0 || l < 0)
            return -1;
        out[i] = (uint8_t)(h << 4 | l);
    }
    return 0;
}

static int digest(const void *msg, size_t n, uint8_t h[32])
{
    size_t got = 0;
    return psa_hash_compute(PSA_ALG_SHA_256, msg, n, h, 32, &got) ==
                       PSA_SUCCESS &&
                   got == 32
               ? 0
               : -1;
}

int janas_sig_sign(const uint8_t priv[JANAS_SIG_PRIV], const void *msg,
                   size_t n, char *line, size_t len)
{
    if (len < JANAS_SIG_LINE + 1 || start() != 0)
        return -1;
    uint8_t h[32], sig[JANAS_SIG_RAW];
    psa_key_id_t id = 0;
    size_t got = 0;
    if (digest(msg, n, h) != 0 ||
        import(PSA_KEY_TYPE_ECC_KEY_PAIR(PSA_ECC_FAMILY_SECP_R1),
               PSA_KEY_USAGE_SIGN_HASH, priv, JANAS_SIG_PRIV, &id) != 0)
        return -1;
    int rc = psa_sign_hash(id, SIG_ALG, h, sizeof h, sig, sizeof sig, &got) ==
                         PSA_SUCCESS &&
                     got == JANAS_SIG_RAW
                 ? 0
                 : -1;
    psa_destroy_key(id);
    if (rc != 0)
        return -1;
    size_t p = sizeof(JANAS_SIG_PREFIX) - 1;
    memcpy(line, JANAS_SIG_PREFIX, p);
    janas_sig_hex(sig, sizeof sig, line + p);
    line[p + 2 * JANAS_SIG_RAW] = '\n';
    line[p + 2 * JANAS_SIG_RAW + 1] = 0;
    return 0;
}

int janas_sig_verify(const uint8_t pub[JANAS_SIG_PUB], const void *msg,
                     size_t n, const char *line, size_t line_len)
{
    while (line_len &&
           (line[line_len - 1] == '\n' || line[line_len - 1] == '\r' ||
            line[line_len - 1] == ' '))
        line_len--;
    while (line_len && (*line == ' ' || *line == '\n' || *line == '\r'))
        line++, line_len--;
    size_t p = sizeof(JANAS_SIG_PREFIX) - 1;
    uint8_t h[32], sig[JANAS_SIG_RAW];
    if (line_len < p || memcmp(line, JANAS_SIG_PREFIX, p) != 0 ||
        janas_sig_unhex(line + p, line_len - p, sig, sizeof sig) != 0 ||
        start() != 0 || digest(msg, n, h) != 0)
        return -1;
    psa_key_id_t id = 0;
    if (import(PSA_KEY_TYPE_ECC_PUBLIC_KEY(PSA_ECC_FAMILY_SECP_R1),
               PSA_KEY_USAGE_VERIFY_HASH, pub, JANAS_SIG_PUB, &id) != 0)
        return -1;
    int rc = psa_verify_hash(id, SIG_ALG, h, sizeof h, sig, sizeof sig) ==
                     PSA_SUCCESS
                 ? 0
                 : -1;
    psa_destroy_key(id);
    return rc;
}
