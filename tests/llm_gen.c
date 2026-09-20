/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * llm_gen.c - greedy generation with and without speculative decoding.
 *
 * Generates the same continuation twice, first without speculation and then
 * with lookup drafts, checks that the tokens are identical, and reports the
 * speed of both and the acceptance of the drafts.
 *
 * Usage: llm_gen <model.jns> <prompt.int32> <n_prompt> <n_new> <k> [auto]
 *                [mtp=<mtp.jns>] [conf=<min probability of an MTP draft>]
 * With mtp= the drafts come from the model's MTP block instead of lookup.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/sysinfo.h"
#include "llm/generate.h"
#include "llm/model.h"

/* What the planner has learned: per tenth of guessed survival chance, how
   many drafts it tried and how many survived. */
static void report_calibration(const struct janas_gen_stats *s)
{
    int any = 0;
    for (int b = 0; b < 10; b++)
        any |= s->cal_tried[b] != 0;
    if (!any)
        return;
    printf("calibration, guessed chance -> tried, survived:");
    for (int b = 0; b < 10; b++)
        if (s->cal_tried[b])
            printf(" %.1f-%.1f: %u/%u (%.0f%%)", b / 10.0, (b + 1) / 10.0,
                   s->cal_won[b], s->cal_tried[b],
                   100.0 * s->cal_won[b] / s->cal_tried[b]);
    printf("\n");
}

static void report(const char *name, const struct janas_gen_stats *s)
{
    printf("%-8s prefill %u tokens %.2f s (%.1f token/s); decode %u tokens "
           "%.2f s (%.1f token/s), %u passes",
           name, s->prompt_tokens, s->prefill_seconds,
           s->prompt_tokens / s->prefill_seconds, s->new_tokens,
           s->decode_seconds, s->new_tokens / s->decode_seconds, s->passes);
    if (s->new_tokens)
        printf(", %u of %u forced (%.0f%%)", s->forced_tokens, s->new_tokens,
               100.0 * s->forced_tokens / s->new_tokens);
    if (s->sample_seconds > 0)
        printf(", sampling %.2f s (%.0f%%)", s->sample_seconds,
               100 * s->sample_seconds / s->decode_seconds);
    if (s->draft_seconds > 0)
        printf(", drafting %.2f s (%.0f%% of decode)", s->draft_seconds,
               100 * s->draft_seconds / s->decode_seconds);
    if (s->spec_passes)
        printf(", %u verifying: %u of %u drafted accepted (%.0f%%), %.2f "
               "tokens per pass",
               s->spec_passes, s->accepted, s->drafted,
               100.0 * s->accepted / s->drafted,
               (double)s->new_tokens / s->passes);
    if (s->auto_off_passes)
        printf(", %u passes with speculation paused", s->auto_off_passes);
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr,
                "usage: llm_gen <model.jns> <prompt.int32> <n_prompt> "
                "<n_new> <k> [auto] [mtp=<mtp.jns>] [conf=<p>] "
                "[threads=<n>] [cache=<MiB>] [ctx=<n>] [temp=<t>] [seed=<n>] "
                "[draft=<small.jns>]\n");
        return 2;
    }
    int32_t prompt[8192];
    FILE *f = fopen(argv[2], "rb");
    if (!f)
        return 1;
    size_t avail = fread(prompt, sizeof(int32_t), 8192, f);
    fclose(f);
    uint32_t np = (uint32_t)atoi(argv[3]), nn = (uint32_t)atoi(argv[4]);
    if (np == 0 || np > avail || nn == 0)
        return 2;

    int use_auto = 0;
    const char *mtp_path = NULL;
    float min_conf = 0.0f;
    /* the product picks its own thread count; the default here is the one
       this test has always used, so old numbers stay comparable */
    int n_compute = 12, n_ctx = 4096;
    uint64_t cache_mib = 20480;
    float temp = 0.0f; /* greedy, so that the two runs can be compared */
    const char *draft_path = NULL;
    uint64_t seed = 20260922;
    float top_p = 0.8f;
    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "auto") == 0)
            use_auto = 1;
        else if (strncmp(argv[i], "mtp=", 4) == 0)
            mtp_path = argv[i] + 4;
        else if (strncmp(argv[i], "conf=", 5) == 0)
            min_conf = (float)atof(argv[i] + 5);
        else if (strncmp(argv[i], "threads=", 8) == 0)
            n_compute = atoi(argv[i] + 8);
        else if (strncmp(argv[i], "cache=", 6) == 0)
            cache_mib = (uint64_t)atol(argv[i] + 6);
        else if (strncmp(argv[i], "ctx=", 4) == 0)
            n_ctx = atoi(argv[i] + 4);
        else if (strncmp(argv[i], "temp=", 5) == 0)
            temp = (float)atof(argv[i] + 5);
        else if (strncmp(argv[i], "draft=", 6) == 0)
            draft_path = argv[i] + 6;
        else if (strncmp(argv[i], "seed=", 5) == 0)
            seed = strtoull(argv[i] + 5, NULL, 10);
        else if (strncmp(argv[i], "top_p=", 6) == 0)
            top_p = (float)atof(argv[i] + 6);
        else if (strncmp(argv[i], "seed=", 5) == 0)
            seed = (uint64_t)strtoull(argv[i] + 5, NULL, 10);
    }
    int icpus[8] = {12, 13, 14, 15, 16, 17, 18, 19};
    struct janas_cpu_layout lay;
    int ccpus[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const int *compute = ccpus;
    if (n_compute != 12 && janas_cpu_layout(NULL, 0, &lay) == 0) {
        if (n_compute < 1 || n_compute > lay.n)
            n_compute = lay.n;
        compute = lay.cpus;
    }
    struct janas_llm_options mo = {.cache_bytes = cache_mib << 20,
                                   .n_ctx = (uint32_t)n_ctx,
                                   .compute_cpus = compute,
                                   .n_compute = n_compute,
                                   .io_cpus = icpus,
                                   .n_io = 8,
                                   .mtp_path = mtp_path};
    char err[256];
    struct janas_llm_model *m =
        janas_llm_model_load(argv[1], &mo, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "%s: %s\n", argv[1], err);
        return 1;
    }
    /* the drafting model shares the target's pool: two of them would take
       the cores from each other */
    struct janas_llm_model *draft = NULL;
    if (draft_path) {
        struct janas_llm_options dmo = mo;
        dmo.mtp_path = NULL;
        dmo.cache_bytes = (uint64_t)2048 << 20;
        dmo.shared_compute = janas_llm_model_compute(m);
        draft = janas_llm_model_load(draft_path, &dmo, err, sizeof(err));
        if (!draft) {
            fprintf(stderr, "%s: %s\n", draft_path, err);
            return 1;
        }
    }
    int32_t *a = malloc(nn * sizeof(int32_t)),
            *b = malloc(nn * sizeof(int32_t));
    /* the same seed on both runs: at a temperature above zero the two
       still draw the same tokens unless the drafts change the outcome */
    struct janas_gen_options off = {.spec_mode = JANAS_SPEC_OFF,
                                    .spec_k = 1,
                                    .eos = -1,
                                    .temperature = temp,
                                    /* the product's own sampling */
                                    .top_k = temp > 0 ? 20 : 0,
                                    .top_p = temp > 0 ? top_p : 0.0f,
                                    .seed = seed};
    int has_mtp = janas_llm_model_has_mtp(m); /* a file, or in the model */
    struct janas_gen_options on = {.spec_mode = draft     ? JANAS_SPEC_DRAFT
                                                : has_mtp ? JANAS_SPEC_MTP
                                                          : JANAS_SPEC_LOOKUP,
                                   .draft = draft,
                                   .spec_k = atoi(argv[5]),
                                   .spec_auto = use_auto,
                                   .eos = -1,
                                   .spec_min_conf = min_conf,
                                   .temperature = temp,
                                   .top_k = temp > 0 ? 20 : 0,
                                   .top_p = temp > 0 ? top_p : 0.0f,
                                   .seed = seed};
    struct janas_gen_stats sa, sb;
    /* warm-up run so both measured runs see the same expert cache */
    int na = janas_llm_generate(m, prompt, np, nn, &off, a, &sa);
    na = janas_llm_generate(m, prompt, np, nn, &off, a, &sa);
    int nb = janas_llm_generate(m, prompt, np, nn, &on, b, &sb);
    if (na < 0 || nb < 0) {
        fprintf(stderr, "generation failed\n");
        return 1;
    }
    printf("first tokens:");
    for (int i = 0; i < 12 && i < na; i++)
        printf(" %d", a[i]);
    printf("\n");
    printf("first tokens:");
    for (int i = 0; i < 12 && i < na; i++)
        printf(" %d", a[i]);
    printf("\n");
    report("normal", &sa);
    report(draft     ? "draft"
           : has_mtp ? (on.spec_auto ? "mtp+a" : "mtp")
                     : (on.spec_auto ? "lookup+a" : "lookup"),
           &sb);
    report_calibration(&sb);
    int same = na == nb && memcmp(a, b, (size_t)na * sizeof(int32_t)) == 0;
    printf("tokens identical: %s (%d and %d); speed-up %.2fx\n",
           same ? "yes" : "NO", na, nb,
           (sb.new_tokens / sb.decode_seconds) /
               (sa.new_tokens / sa.decode_seconds));
    free(a);
    free(b);
    char tune[512];
    janas_llm_model_tuning(m, tune, sizeof(tune));
    printf("tuning: %s\n", tune);
    janas_llm_model_free(m);
    return same ? 0 : 1;
}
