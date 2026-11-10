/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gate.h - the entrance gate (job/docs/progetto_nucleus_cancello.md): from
 * a model's residual after one layer, for the last token of a request,
 * linear readers decide whether the model goes on, hands the request to a
 * Specialist (and of which domain), or does not know what it is asked
 * about.
 *
 * A gate file, little-endian: "JANASGT1", then the 32-bit words version
 * (1), d_model, layer, outcomes, domains; then, for each of the outcomes'
 * readers (in the order of enum janas_gate_outcome, at least the first
 * three) and then each domain's, a 32-byte name (NUL-padded) and d_model + 1
 * floats: the weights and the bias of a score w.x + b. The highest outcome
 * score is the decision; the highest domain score names the domain.
 */
#ifndef JANAS_LLM_GATE_H
#define JANAS_LLM_GATE_H

#include <stddef.h>
#include <stdint.h>

enum janas_gate_outcome {
    JANAS_GATE_GO_ON,      /* the model answers */
    JANAS_GATE_SPECIALIST, /* a Specialist of the domain answers */
    JANAS_GATE_UNKNOWN,    /* about something the model does not know */
    JANAS_GATE_CORRECT,    /* a false premise (not reliable yet) */
    JANAS_GATE_TYPO,       /* a misspelt name, understood */
    JANAS_GATE_OUTCOMES
};

struct janas_gate;

/* The gate in buf (len bytes), for a model of width d_model and n_layer
   layers; NULL with err set when it is not a valid gate for that model. */
struct janas_gate *janas_gate_parse(const void *buf, size_t len,
                                    uint32_t d_model, uint32_t n_layer,
                                    char *err, size_t err_len);
/* The same from a file. */
struct janas_gate *janas_gate_load(const char *path, uint32_t d_model,
                                   uint32_t n_layer, char *err, size_t err_len);
void janas_gate_free(struct janas_gate *g);

uint32_t janas_gate_layer(const struct janas_gate *g);
uint32_t janas_gate_domains(const struct janas_gate *g);
const char *janas_gate_domain_name(const struct janas_gate *g, uint32_t i);
const char *janas_gate_outcome_name(int outcome);

/* The decision on x (d_model floats): the outcome, the domain into
   *domain (when not NULL), each outcome's score into scores (when not
   NULL, JANAS_GATE_OUTCOMES of them, the ones the gate lacks -inf). */
int janas_gate_decide(const struct janas_gate *g, const float *x,
                      uint32_t *domain, double *scores);

#endif
