/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * update.h - Brebus updates itself (update.c, update_kind.c,
 * update_apply.c). It asks GitHub for the latest release of
 * prabanta-dev/brebus at its start and once a day; a newer one is
 * downloaded, its SHA256SUMS checked against the project's signature
 * (update_key.h, src/common/release_sig.h) and the file against its sum;
 * then, while nobody uses it (or when asked), Brebus ends and a copy of
 * itself (brebus --update-finish) puts the new version in place of the
 * old, starts it, and puts the old back if the new does not start.
 *
 * How it was installed says what is replaced: an AppImage its file, a
 * folder unpacked from a .tar.gz or a .zip the folder, macOS's Brebus.app
 * the bundle, Windows's setup by the new setup run silently (installed
 * per user: no administrator). A system's package (.deb, .rpm, pacman)
 * is the system's to update: Brebus only says that there is a new one.
 */
#ifndef BREBUS_UPDATE_H
#define BREBUS_UPDATE_H

#include <stddef.h>

struct brebus;

enum brebus_up_kind {
    BREBUS_UP_NONE,     /* a build between releases, or not knowable */
    BREBUS_UP_APPIMAGE, /* target: the .AppImage file */
    BREBUS_UP_FOLDER,   /* target: the folder of brebus (.tar.gz, .zip) */
    BREBUS_UP_MACAPP,   /* target: Brebus.app */
    BREBUS_UP_SETUP,    /* target: the folder Windows's setup installed */
    BREBUS_UP_SYSTEM,   /* a package of the system: only told */
};

struct brebus_up_where {
    enum brebus_up_kind kind;
    char target[4096];
    char asset[128]; /* the release's file it is updated from */
    char why[200];   /* why not, for NONE and SYSTEM */
};

/* How this Brebus was installed (update_kind.c): bin_dir is the folder
   of its program, appimage the APPIMAGE variable (NULL: none). */
void brebus_up_where(const char *bin_dir, const char *appimage,
                     struct brebus_up_where *w);

/* The asset of a kind for this processor and system (update_kind.c):
   "" when there is none */
void brebus_up_asset(enum brebus_up_kind kind, char *out, size_t len);

/* The program to start once target is in place (update_kind.c) */
void brebus_up_program(enum brebus_up_kind kind, const char *target, char *out,
                       size_t len);

/* Puts the staged file in place: a copy of brebus runs
   brebus --update-finish with the plan written in data_dir, which waits
   for this Brebus to end; with restart this Brebus is asked to end, and
   the new one started as this one was (update_apply.c). 0, or -1 with the
   reason in err. */
int brebus_up_apply(struct brebus *b, const struct brebus_up_where *w,
                    const char *staged, const char *version, int restart,
                    char *err, size_t err_len);

/* brebus --update-finish <plan>: waits for the old Brebus to end, puts the
   new one in place, starts it, checks that it started (or puts the old
   back). The exit status (update_apply.c). */
int brebus_update_finish(const char *plan);

/* this Brebus's version (BREBUS_UPDATE_VERSION pretends another, for
   tests) (update.c) */
const char *brebus_up_version(void);

/* the file the new Brebus writes once it serves (its version): the
   finisher waits for it */
#define BREBUS_UP_STARTED "started"

#endif
