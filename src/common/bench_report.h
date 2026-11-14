/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bench_report.h - janas-bench's reports found in a text: an issue's body
 * or a comment where a tester pasted one (tools/reports_table.c makes
 * docs/reports.md's tables from them). A report is found by the lines
 * janas-bench writes the same way since its first version:
 * "janas-bench <version>, <stamp>", then "system:", "CPU:", "memory:",
 * "GPU:", "model:", and "result:" with the speeds the engine chose; quoted
 * (">") or indented lines too. The names come cleaned as people say them.
 */
#ifndef JANAS_COMMON_BENCH_REPORT_H
#define JANAS_COMMON_BENCH_REPORT_H

#include <stddef.h>

struct janas_bench_row {
    char model[96], cpu[96], gpu[96], ram[32], sys[32], version[32];
    double prefill, decode, mtp_greedy, mtp_sampled; /* tokens a second */
};

/* fn for every report of text[0..len) with a model, a CPU and a reply's
   speed; the reports, or -1 when fn returned -1. */
int janas_bench_report_scan(const char *text, size_t len,
                            int (*fn)(void *ctx,
                                      const struct janas_bench_row *r),
                            void *ctx);

#endif
