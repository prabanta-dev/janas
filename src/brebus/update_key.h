/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* The project's public key, ECDSA P-256 (src/common/release_sig.h): a
   release's SHA256SUMS signed by its private key, which never leaves the
   maintainer (tools/release_sign.c), is the only one Brebus installs. */
#ifndef BREBUS_UPDATE_KEY_H
#define BREBUS_UPDATE_KEY_H

#define BREBUS_UPDATE_PUBLIC_KEY                                               \
    "04a13415bdc5989fc99528e922990da02454c0a26f78079a7a357286706e0e89d2"       \
    "6188a8866883daf42c17710d0a9c0ea2ea1285103b41bdf78835593af125266b"

#endif
