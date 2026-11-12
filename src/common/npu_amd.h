/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * npu_amd.h - AMD's NPUs (XDNA), found and named (npu_amd.c): not used by
 * Janas yet, only said.
 */
#ifndef JANAS_COMMON_NPU_AMD_H
#define JANAS_COMMON_NPU_AMD_H

#include <stddef.h>

/* 1 with "AMD NPU, <generation>, <driver or name>" in desc when the
   machine has one of AMD's NPUs; 0 when not. */
int janas_npu_amd_find(char *desc, size_t len);

#endif
