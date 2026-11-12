/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * serve.h - janas-setup serve (serve.c): the servers of the saved plan.
 */
#ifndef JANAS_SETUP_SERVE_H
#define JANAS_SETUP_SERVE_H

/* Runs them until Ctrl+C or until one stops: the exit status. With monitor,
   each serves its page of the engine at work too (janas-server --monitor,
   /monitor). */
int setup_serve(int monitor);

#endif
