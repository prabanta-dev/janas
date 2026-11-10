/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * editors.h - the editors of the VS Code family installed here and their
 * extensions (editors.c), through each editor's own command line:
 * --list-extensions says what is installed, --install-extension installs
 * from the editor's gallery (Open VSX for VSCodium, Microsoft's for VS
 * Code), so that only what that gallery offers is ever installed.
 */
#ifndef JANAS_SETUP_EDITORS_H
#define JANAS_SETUP_EDITORS_H

#include <stddef.h>

#define SETUP_EXT_MAX 256

struct setup_editor {
    const char *name;    /* "VSCodium", "VS Code"... */
    char cli[4096];      /* its command: a path, or flatpak */
    const char *pre;     /* flatpak: the application, after "run"; or NULL */
    int open_vsx;        /* its gallery is Open VSX */
    char user_dir[4096]; /* its settings: .../User */
    /* passed on to every call (NULL: the editor's own folders) */
    const char *user_data_dir, *extensions_dir;
    int n_ext;
    char ext[SETUP_EXT_MAX][128]; /* "publisher.name@version", lowercase id */
};

/* The editors found, at most max: their number. */
int setup_editors_find(struct setup_editor *out, int max,
                       const char *user_data_dir, const char *extensions_dir);

/* The version of extension id installed in e ("" when it is not said), or
   NULL when it is not installed. */
const char *setup_editor_has(const struct setup_editor *e, const char *id);

/* Installs extension id in e: 0, or -1 with the reason printed. */
int setup_editor_install(struct setup_editor *e, const char *id);

/* The extensions janas-setup knows. */
struct setup_known {
    const char *id;   /* lowercase, as --list-extensions writes it */
    const char *name; /* as its users know it */
    const char *does; /* a line for the user */
    int configured;   /* janas-setup writes its configuration */
    int on_open_vsx;
    int archived; /* no longer developed: configured, never offered */
};

extern const struct setup_known setup_known_ext[];
extern const int setup_known_n;

#endif
