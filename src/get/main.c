/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-get - a model ready for Janas in one command: downloaded from
 * Hugging Face, checked, converted, its prediction file made beside it,
 * every file compared with the fingerprints of MODELS.md.
 *
 *   janas-get list
 *   janas-get <model> [--dir DIR] [--keep] [--no-mtp]
 *   janas-get hf:<owner>/<repo>/<file.gguf> [--dir DIR] [--keep]
 *
 * A download stopped halfway resumes where it stopped; a file already
 * there is checked, not fetched again. The GGUF goes once converted,
 * unless --keep.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>

#include "catalog.h"
#include "common/sysinfo.h"
#include "steps.h"

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-get list\n"
            "       janas-get <model> [--dir DIR] [--keep] [--no-mtp]\n"
            "       janas-get hf:<owner>/<repo>/<file.gguf> [--dir DIR] "
            "[--keep]\n"
            "  --dir DIR   where the models go (default: models)\n"
            "  --keep      keep the downloaded GGUF once converted\n"
            "  --no-mtp    without the model's prediction file\n");
}

static void list(void)
{
    printf("%-22s %8s %7s  %s\n", "model", "download", "memory", "");
    for (int i = 0; i < get_catalog_n; i++) {
        const struct get_model *m = &get_catalog[i];
        printf("%-22s %6.1f GB %4d GB  %s\n", m->name, m->gguf_gb, m->ram_gb,
               m->what);
    }
    printf("\nFingerprints and licences: MODELS.md. Any other GGUF: "
           "janas-get hf:<owner>/<repo>/<file.gguf>\n");
}

static int exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

/* a converter's output compared with MODELS.md; 0 when it matches */
static int check(const char *file, const char *want, const char *who)
{
    int r = get_same(file, want);
    if (r == 1) {
        printf("   %s: fingerprint as in MODELS.md\n", who);
        return 0;
    }
    fprintf(stderr,
            "janas-get: %s wrote %s, and its SHA-256 is not the one of "
            "MODELS.md: please report it (an issue on GitHub)\n",
            who, file);
    return -1;
}

/* gguf2jns from the first part into out */
static int convert(const char *first, const char *out)
{
    char *argv[] = {"gguf2jns", (char *)first, (char *)out, NULL};
    printf("== converting (gguf2jns)\n");
    fflush(stdout);
    return get_run("gguf2jns", argv) == 0 ? 0 : -1;
}

static int from_catalog(const struct get_model *m, const char *dir, int keep,
                        int no_mtp)
{
    char out[4096], first[4096], flat[4200];
    snprintf(out, sizeof(out), "%s/%s", dir, m->out);
    get_local(first, sizeof(first), dir, m->parts[0]);
    if (m->ram_gb &&
        janas_mem_total() < (uint64_t)m->ram_gb * (1ull << 30) * 85 / 100)
        printf("note: %s is meant for machines with %d GB of memory or "
               "more; this one has %.1f GB\n",
               m->name, m->ram_gb, (double)janas_mem_total() / (1ull << 30));
    int have = exists(out) &&
               get_same(out, m->jns_sha ? m->jns_sha : m->flat_sha) == 1;
    if (have) {
        printf("%s: already here, fingerprint checked\n", out);
    } else {
        /* room: two copies at the most - the GGUF and what gguf2jns
           writes, then (the GGUF gone) that and what jns_planes writes */
        struct statvfs vf;
        double need = m->gguf_gb * 2.0 * 1e9;
        if (statvfs(dir, &vf) == 0 &&
            (double)vf.f_bavail * (double)vf.f_frsize < need) {
            fprintf(stderr,
                    "janas-get: %s needs about %.0f GB free in %s while it "
                    "works; there are %.0f\n",
                    m->name, need / 1e9, dir,
                    (double)vf.f_bavail * (double)vf.f_frsize / 1e9);
            return 1;
        }
        printf("== %s: downloading\n", m->name);
        for (int p = 0; p < GET_MAX_PARTS && m->parts[p]; p++)
            if (get_fetch(m->repo, m->parts[p], dir, m->part_sha[p]) != 0)
                return 1;
        snprintf(flat, sizeof(flat), "%s.flat", out);
        const char *conv = m->jns_sha ? flat : out;
        if (convert(first, conv) != 0 ||
            check(conv, m->flat_sha, "gguf2jns") != 0)
            return 1;
        if (!keep)
            for (int p = 0; p < GET_MAX_PARTS && m->parts[p]; p++) {
                char part[4096];
                get_local(part, sizeof(part), dir, m->parts[p]);
                remove(part);
            }
        if (m->jns_sha) {
            char *argv[] = {"jns_planes", flat, out, NULL};
            printf("== cutting the experts into bit planes (jns_planes)\n");
            fflush(stdout);
            if (get_run("jns_planes", argv) != 0 ||
                check(out, m->jns_sha, "jns_planes") != 0)
                return 1;
            remove(flat);
        }
    }
    if (m->extra != EXTRA_NONE && !no_mtp) {
        char xout[4096];
        snprintf(xout, sizeof(xout), "%s/%s", dir, m->extra_out);
        if (exists(xout) && get_same(xout, m->extra_sha) == 1) {
            printf("%s: already here, fingerprint checked\n", xout);
        } else if (m->extra == EXTRA_GGUF) {
            char src[4096];
            printf("== its assistant\n");
            if (get_fetch(m->repo, m->extra_src, dir, m->extra_src_sha) != 0)
                return 1;
            get_local(src, sizeof(src), dir, m->extra_src);
            if (convert(src, xout) != 0 ||
                check(xout, m->extra_sha, "gguf2jns (assistant)") != 0)
                return 1;
            if (!keep)
                remove(src);
        } else {
            char *argv[] = {"hf2jns_mtp", out, (char *)m->extra_src, xout,
                            NULL};
            printf("== its MTP block, from %s (hf2jns_mtp)\n", m->extra_src);
            fflush(stdout);
            if (get_run("hf2jns_mtp", argv) != 0 ||
                check(xout, m->extra_sha, "hf2jns_mtp") != 0)
                return 1;
        }
    }
    printf("\nReady: %s\nChat with it: janas-chat %s\n", out, out);
    return 0;
}

/* hf:<owner>/<repo>/<path>: any GGUF, checked against the SHA-256 Hugging
   Face lists, converted in one step */
static int from_hf(const char *spec, const char *dir, int keep)
{
    char repo[512];
    const char *s = spec + 3, *a = strchr(s, '/'),
               *b = a ? strchr(a + 1, '/') : NULL;
    size_t n = b ? (size_t)(b - s) : 0;
    if (!b || n >= sizeof(repo) || !b[1]) {
        usage();
        return 2;
    }
    memcpy(repo, s, n);
    repo[n] = 0;
    const char *path = b + 1;
    char local[4096], out[4200];
    get_local(local, sizeof(local), dir, path);
    snprintf(out, sizeof(out), "%s", local);
    char *dot = strrchr(out, '.');
    if (dot && strcmp(dot, ".gguf") == 0)
        *dot = 0;
    for (char *c = strrchr(out, '/') ? strrchr(out, '/') + 1 : out; *c; c++)
        if (*c >= 'A' && *c <= 'Z')
            *c = (char)(*c - 'A' + 'a');
    strcat(out, ".jns");
    printf("== %s: downloading (not in the catalog: checked against the "
           "SHA-256 Hugging Face lists)\n",
           path);
    if (get_fetch(repo, path, dir, NULL) != 0 || convert(local, out) != 0)
        return 1;
    if (!keep)
        remove(local);
    printf("\nReady: %s (no fingerprint to compare: not in MODELS.md)\n"
           "Chat with it: janas-chat %s\n",
           out, out);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage();
        return 2;
    }
    const char *dir = "models";
    int keep = 0, no_mtp = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc)
            dir = argv[++i];
        else if (strcmp(argv[i], "--keep") == 0)
            keep = 1;
        else if (strcmp(argv[i], "--no-mtp") == 0)
            no_mtp = 1;
        else {
            usage();
            return 2;
        }
    }
    if (strcmp(argv[1], "list") == 0) {
        list();
        return 0;
    }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        usage();
        return 0;
    }
    mkdir(dir, 0755);
    if (strncmp(argv[1], "hf:", 3) == 0)
        return from_hf(argv[1], dir, keep);
    const struct get_model *m = get_find(argv[1]);
    if (!m) {
        fprintf(stderr, "janas-get: no model called %s (janas-get list)\n",
                argv[1]);
        return 2;
    }
    return from_catalog(m, dir, keep, no_mtp);
}
