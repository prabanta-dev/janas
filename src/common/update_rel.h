/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * update_rel.h - what Brebus's updates read of a release (src/brebus/
 * update.c): its version, its files as GitHub's API lists them, their
 * SHA-256 in SHA256SUMS. Text only, no network: tests/test_update_rel.c.
 */
#ifndef JANAS_UPDATE_REL_H
#define JANAS_UPDATE_REL_H

#include <stddef.h>
#include <stdint.h>

/* A release's version: "v2026.11.26" or "v2026.11.26.1" (the v may be
   left out), its parts into part[4] (the fourth 0 when absent). A build
   between releases (git describe's "-3-gabc1234", "-dirty"), "unknown" or
   any other text is no release's: -1. */
int janas_rel_version(const char *s, int part[4]);

/* 1 when a is a release newer than b, which must be a release too;
   0 otherwise (either is not one, or a is not newer) */
int janas_rel_newer(const char *a, const char *b);

struct janas_rel_asset {
    char name[128];
    char url[512];
    uint64_t size;
};

/* GitHub's answer for a release (releases/latest): its tag into tag and
   up to max of its files into a, their count into *n (only those over
   HTTPS, or from http://127.0.0.1: a test's). A draft or a
   pre-release is no release to install: -1, as for a text that is not
   one. 0, or -1 with the reason in err. */
int janas_rel_parse(const char *json, size_t len, char *tag, size_t tag_len,
                    struct janas_rel_asset *a, int max, int *n, char *err,
                    size_t err_len);

/* The SHA-256 that SHA256SUMS (sha256sum's lines: 64 hex digits, two
   spaces or a space and '*', the name) gives for name, as 64 lowercase
   hex digits and a NUL. 0, or -1 when the name is not there, or twice. */
int janas_rel_sum(const char *sums, size_t len, const char *name, char sha[65]);

#endif
