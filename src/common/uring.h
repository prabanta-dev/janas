/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * uring.h - asynchronous reads with io_uring, through the system calls.
 *
 * The point is not bandwidth - at the block sizes of an expert, io_uring and
 * pread on a pool of threads read just as fast (bozza Fase 0) - but the cost
 * of starting a read: here the caller queues the reads itself and the kernel
 * runs them while it computes, so no thread has to be woken to begin them.
 *
 * A ring belongs to one thread at a time: queue with janas_uring_read(), hand
 * the batch to the kernel with janas_uring_submit(), and collect it with
 * janas_uring_wait(). A read that returns fewer bytes than asked is submitted
 * again for what is missing, so a request that completes has read all of its
 * bytes. janas_uring_create() returns NULL where io_uring is not available (an
 * old kernel, a sandbox that forbids it): the caller keeps its own way of
 * reading.
 */
#ifndef JANAS_COMMON_URING_H
#define JANAS_COMMON_URING_H

#include <stdint.h>

struct janas_uring;

/* A ring that holds up to entries reads in flight. NULL if unavailable. */
struct janas_uring *janas_uring_create(unsigned entries);
void janas_uring_destroy(struct janas_uring *u);

/*
 * Queues one read of bytes from offset of fd into dst. Returns 0, or -1 if
 * the ring is full: wait for what is in flight, then queue again.
 */
int janas_uring_read(struct janas_uring *u, int fd, void *dst, uint64_t bytes,
                     uint64_t offset);

/* Hands the queued reads to the kernel without waiting. Returns 0 or -1. */
int janas_uring_submit(struct janas_uring *u);

/* Waits for every read in flight. Returns 0, or -1 if one of them failed. */
int janas_uring_wait(struct janas_uring *u);

/* Reads queued or submitted and not yet complete. */
unsigned janas_uring_inflight(const struct janas_uring *u);


#endif
