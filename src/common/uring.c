/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * uring.c - asynchronous reads with io_uring (see uring.h).
 *
 * liburing is not used: the whole of it, for reads, is three system calls and
 * two shared rings, and one dependency less is worth the hundred lines. The
 * operation is IORING_OP_READV rather than IORING_OP_READ because it is there
 * in every kernel that has io_uring at all; with blocks of a megabyte and
 * more the one iovec costs nothing measurable.
 */
#define _GNU_SOURCE
#include "common/uring.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sys/syscall.h>
#if defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter) &&            \
    defined(__has_include)
#if __has_include(<linux/io_uring.h>)
#define JANAS_HAVE_URING 1
#endif
#endif
#endif

#ifndef JANAS_HAVE_URING

struct janas_uring *janas_uring_create(unsigned entries)
{
    (void)entries;
    return NULL;
}

void janas_uring_destroy(struct janas_uring *u)
{
    (void)u;
}

int janas_uring_read(struct janas_uring *u, int fd, void *dst, uint64_t bytes,
                     uint64_t offset)
{
    (void)u, (void)fd, (void)dst, (void)bytes, (void)offset;
    return -1;
}

int janas_uring_submit(struct janas_uring *u)
{
    (void)u;
    return -1;
}

int janas_uring_wait(struct janas_uring *u)
{
    (void)u;
    return -1;
}

unsigned janas_uring_inflight(const struct janas_uring *u)
{
    (void)u;
    return 0;
}


#else

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

struct request {
    struct iovec iov; /* read by the kernel at submission: it must live here */
    int fd;
    uint8_t *dst;
    uint64_t offset, bytes, done;
};

struct janas_uring {
    int fd;
    unsigned entries;

    void *sq_map, *cq_map, *sqe_map;
    size_t sq_size, cq_size, sqe_size;

    unsigned *sq_head, *sq_tail, *sq_mask, *sq_array;
    unsigned *cq_head, *cq_tail, *cq_mask;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;

    struct request *req;
    unsigned n_req;     /* slots taken by this batch */
    unsigned pending;   /* of them, not yet complete */
    unsigned to_submit; /* queued, not yet handed to the kernel */
};

static int uring_setup(unsigned entries, struct io_uring_params *p)
{
    return (int)syscall(__NR_io_uring_setup, entries, p);
}

static int uring_enter(int fd, unsigned to_submit, unsigned min_complete,
                       unsigned flags)
{
    return (int)syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags,
                        (void *)NULL, (size_t)0);
}

void janas_uring_destroy(struct janas_uring *u)
{
    if (!u)
        return;
    if (u->sqe_map)
        munmap(u->sqe_map, u->sqe_size);
    if (u->cq_map && u->cq_map != u->sq_map)
        munmap(u->cq_map, u->cq_size);
    if (u->sq_map)
        munmap(u->sq_map, u->sq_size);
    if (u->fd >= 0)
        close(u->fd);
    free(u->req);
    free(u);
}

struct janas_uring *janas_uring_create(unsigned entries)
{
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    int fd = uring_setup(entries, &p);
    if (fd < 0)
        return NULL;
    struct janas_uring *u = calloc(1, sizeof(*u));
    if (!u) {
        close(fd);
        return NULL;
    }
    u->fd = fd;
    u->entries = p.sq_entries;
    u->sq_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    u->cq_size = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) { /* one mapping for both */
        if (u->cq_size > u->sq_size)
            u->sq_size = u->cq_size;
        u->cq_size = u->sq_size;
    }
    u->sq_map = mmap(NULL, u->sq_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (u->sq_map == MAP_FAILED) {
        u->sq_map = NULL;
        janas_uring_destroy(u);
        return NULL;
    }
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
        u->cq_map = u->sq_map;
    } else {
        u->cq_map = mmap(NULL, u->cq_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (u->cq_map == MAP_FAILED) {
            u->cq_map = NULL;
            janas_uring_destroy(u);
            return NULL;
        }
    }
    u->sqe_size = p.sq_entries * sizeof(struct io_uring_sqe);
    u->sqe_map = mmap(NULL, u->sqe_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (u->sqe_map == MAP_FAILED) {
        u->sqe_map = NULL;
        janas_uring_destroy(u);
        return NULL;
    }
    uint8_t *sq = u->sq_map, *cq = u->cq_map;
    u->sq_head = (unsigned *)(sq + p.sq_off.head);
    u->sq_tail = (unsigned *)(sq + p.sq_off.tail);
    u->sq_mask = (unsigned *)(sq + p.sq_off.ring_mask);
    u->sq_array = (unsigned *)(sq + p.sq_off.array);
    u->cq_head = (unsigned *)(cq + p.cq_off.head);
    u->cq_tail = (unsigned *)(cq + p.cq_off.tail);
    u->cq_mask = (unsigned *)(cq + p.cq_off.ring_mask);
    u->sqes = u->sqe_map;
    u->cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);
    u->req = calloc(u->entries, sizeof(*u->req));
    if (!u->req) {
        janas_uring_destroy(u);
        return NULL;
    }
    return u;
}

/* Puts the missing part of request i in the submission queue. 0 if full. */
static int queue_sqe(struct janas_uring *u, unsigned i)
{
    struct request *r = &u->req[i];
    unsigned tail = *u->sq_tail;
    unsigned head = __atomic_load_n(u->sq_head, __ATOMIC_ACQUIRE);
    if (tail - head >= u->entries)
        return 0;
    unsigned index = tail & *u->sq_mask;
    struct io_uring_sqe *sqe = &u->sqes[index];
    memset(sqe, 0, sizeof(*sqe));
    r->iov.iov_base = r->dst + r->done;
    r->iov.iov_len = (size_t)(r->bytes - r->done);
    sqe->opcode = IORING_OP_READV;
    /* Without this the kernel runs the read inside io_uring_enter() whenever
       it can, which for an O_DIRECT block of a megabyte and more means the
       whole read: measured on 20 Sep 2026, 10.5 ms a call for fourteen
       experts, 2.3 GB/s, all of it on the thread that was meant to go on
       computing. IOSQE_ASYNC hands it to the kernel's own workers instead,
       which is the whole point of submitting it. */
    sqe->flags = IOSQE_ASYNC;
    sqe->fd = r->fd;
    sqe->off = r->offset + r->done;
    sqe->addr = (uint64_t)(uintptr_t)&r->iov;
    sqe->len = 1;
    sqe->user_data = i;
    u->sq_array[index] = index;
    __atomic_store_n(u->sq_tail, tail + 1, __ATOMIC_RELEASE);
    u->to_submit++;
    return 1;
}

int janas_uring_read(struct janas_uring *u, int fd, void *dst, uint64_t bytes,
                     uint64_t offset)
{
    if (u->n_req >= u->entries)
        return -1;
    u->req[u->n_req] = (struct request){.iov = {.iov_base = NULL, .iov_len = 0},
                                        .fd = fd,
                                        .dst = dst,
                                        .offset = offset,
                                        .bytes = bytes,
                                        .done = 0};
    if (!queue_sqe(u, u->n_req))
        return -1;
    u->n_req++;
    u->pending++;
    return 0;
}

int janas_uring_submit(struct janas_uring *u)
{
    while (u->to_submit) {
        int n = uring_enter(u->fd, u->to_submit, 0, 0);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (n == 0) /* the kernel took nothing: leave it to the wait */
            break;
        u->to_submit -= (unsigned)n;
    }
    return 0;
}

/* Takes every completion the kernel has published. -1 if a read failed. */
static int reap(struct janas_uring *u)
{
    int rc = 0;
    unsigned head = *u->cq_head;
    unsigned tail = __atomic_load_n(u->cq_tail, __ATOMIC_ACQUIRE);
    for (; head != tail; head++) {
        struct io_uring_cqe *e = &u->cqes[head & *u->cq_mask];
        unsigned i = (unsigned)e->user_data;
        int res = e->res;
        int again = 0;
        if (res > 0) {
            u->req[i].done += (uint64_t)res;
            again = u->req[i].done < u->req[i].bytes; /* a short read */
        } else {
            again = res == -EINTR || res == -EAGAIN;
            if (!again)
                rc = -1;
        }
        if (again) {
            if (!queue_sqe(u, i)) { /* cannot happen: a slot just freed */
                rc = -1;
                u->pending--;
            }
        } else {
            u->pending--;
        }
    }
    __atomic_store_n(u->cq_head, head, __ATOMIC_RELEASE);
    return rc;
}

int janas_uring_wait(struct janas_uring *u)
{
    int rc = 0;
    /* on a failure the wait goes on all the same: until the kernel has
       answered, the buffers of the other reads are still its own */
    while (u->pending) {
        int n = uring_enter(u->fd, u->to_submit, 1, IORING_ENTER_GETEVENTS);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            rc = -1;
            break;
        }
        u->to_submit -= (unsigned)n < u->to_submit ? (unsigned)n : u->to_submit;
        if (reap(u) != 0)
            rc = -1;
    }
    u->n_req = 0;
    u->pending = 0;
    u->to_submit = 0;
    return rc;
}

unsigned janas_uring_inflight(const struct janas_uring *u)
{
    return u->pending;
}


#endif
