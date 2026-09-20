/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_keep.c - conversations kept computed besides the current one
 * (janas_llm_chat_keep in include/janas/llm.h).
 *
 * The model holds one sequence. A caller that switches between several -
 * an HTTP server, whose clients also ask on the side for titles, tags and
 * suggestions - had the current one replaced at every switch and read
 * again from the start when it came back: six thousand tokens for a chat
 * of Open WebUI, at every turn. Here a conversation about to be cut is
 * copied out (janas_llm_session_save), and the one that shares most of the
 * next request is copied back in its place.
 */
#include <stdlib.h>
#include <string.h>

#include "common/sysinfo.h"
#include "llm/chat.h"

/* a conversation is kept when it would lose at least this many computed
   tokens, and one kept takes the current one's place when it spares that
   many more. Losing only the last reply does not count: a request that
   holds all of the last prompt (n replies to it, a reply asked again) is
   not another conversation */
#define KEEP_MIN 64

/* The tokens a sequence of hn, n_kv computed, shares with ids, as
   load_ids keeps them: the last of ids must be read for its logits, so it
   is left out when it was computed. */
static uint32_t shared(const int32_t *h, uint32_t hn, uint32_t n_kv,
                       const int32_t *ids, size_t n_ids)
{
    uint32_t same = 0;
    while (same < hn && same < n_ids && h[same] == ids[same])
        same++;
    if (same == n_ids && same > 0 && n_kv >= same)
        same--;
    return same;
}

/*
 * How many tokens of ids need not be computed again, starting from a
 * sequence of hn tokens whose first n_kv are computed: what
 * janas_llm_session_truncate keeps of their common prefix. A recurrent
 * state cannot go back: all of it is kept, or nothing.
 */
uint32_t janas_api_keep_reuse(const int32_t *h, uint32_t hn, uint32_t n_kv,
                              const int32_t *ids, size_t n_ids, int recurrent)
{
    uint32_t same = shared(h, hn, n_kv, ids, n_ids);
    uint32_t r = same > 0 ? same - 1 : 0;
    if (r >= n_kv)
        return n_kv;
    return recurrent ? 0 : r;
}

static void drop(janas_llm_chat *c, int32_t i)
{
    c->kept_bytes -= janas_llm_saved_bytes(c->kept[i].sv);
    janas_llm_saved_free(c->kept[i].sv);
    c->kept[i] = c->kept[--c->n_kept];
}

/* The least recently used go until the rest fits. */
static void evict(janas_llm_chat *c)
{
    while (c->n_kept > 0 &&
           (c->n_kept > c->keep_max || c->kept_bytes > c->keep_bytes)) {
        int32_t old = 0;
        for (int32_t i = 1; i < c->n_kept; i++)
            if (c->kept[i].used < c->kept[old].used)
                old = i;
        /* out of the memory, onto the disk if there is one */
        janas_api_disk_put(c, c->kept[old].sv, c->kept[old].prefix);
        drop(c, old);
    }
}

static void add(janas_llm_chat *c, struct janas_llm_saved *sv, int prefix)
{
    c->kept[c->n_kept].sv = sv;
    c->kept[c->n_kept].used = ++c->keep_clock;
    c->kept[c->n_kept].prefix = prefix;
    c->n_kept++;
    c->kept_bytes += janas_llm_saved_bytes(sv);
}

/* A copy of the current conversation among the kept ones; those it
   contains whole are no longer needed, but for the system messages kept
   on a recurrent model: another conversation will start from them. */
static void keep_current(janas_llm_chat *c, int prefix)
{
    struct janas_llm_saved *sv = janas_llm_session_save(c->s);
    if (!sv)
        return;
    const int32_t *t;
    uint32_t n_kv, n = janas_llm_saved_tokens(sv, &t, &n_kv);
    for (int32_t i = c->n_kept - 1; i >= 0; i--) {
        const int32_t *u;
        uint32_t u_kv, un = janas_llm_saved_tokens(c->kept[i].sv, &u, &u_kv);
        if ((!c->kept[i].prefix || u_kv == n_kv) && u_kv <= n_kv && un <= n &&
            memcmp(u, t, u_kv * sizeof(*t)) == 0)
            drop(c, i);
    }
    add(c, sv, prefix);
}

void janas_api_keep_pick(janas_llm_chat *c)
{
    if (c->keep_max <= 0)
        return;
    int rec = janas_llm_model_recurrent(c->llm->m);
    const int32_t *h;
    uint32_t hn = janas_llm_session_tokens(c->s, &h);
    uint32_t cur_kv = janas_llm_session_computed(c->s);
    uint32_t r_cur = janas_api_keep_reuse(h, hn, cur_kv, c->ids, c->n_ids, rec);
    int32_t best = -1;
    uint32_t r_best = r_cur;
    for (int32_t i = 0; i < c->n_kept; i++) {
        const int32_t *t;
        uint32_t n_kv, n = janas_llm_saved_tokens(c->kept[i].sv, &t, &n_kv);
        uint32_t r = janas_api_keep_reuse(t, n, n_kv, c->ids, c->n_ids, rec);
        if (r > r_best) {
            best = i;
            r_best = r;
        }
    }
    /* a file that spares more is read into memory, and chosen there */
    uint32_t r_disk;
    int32_t on_disk = janas_api_disk_best(c, r_best, &r_disk);
    if (on_disk >= 0 && r_disk - r_cur >= KEEP_MIN) {
        int prefix = 0;
        struct janas_llm_saved *sv = janas_api_disk_take(c, on_disk, &prefix);
        if (sv) {
            add(c, sv, prefix);
            best = c->n_kept - 1;
            r_best = r_disk;
        }
    }
    /* what the two share, not what can be reused of it: a recurrent state
       cannot go back into its last reply, but losing that reply alone is
       still not losing a conversation */
    uint32_t upto = cur_kv < c->loaded ? cur_kv : c->loaded;
    int lose = shared(h, hn, cur_kv, c->ids, c->n_ids) + 1 < upto &&
               cur_kv - r_cur >= KEEP_MIN;
    if (best >= 0 && r_best - r_cur >= KEEP_MIN) {
        /*
         * Copied in, and kept: the request may use only its start (a new
         * chat with another's system message and tools), and the rest is
         * still that other conversation. When the request continues it
         * instead, the old copy goes once the longer one is kept, as one
         * it holds whole. Taken out of the list meanwhile: keeping the
         * current one drops what it holds whole.
         */
        struct janas_llm_saved *sv = c->kept[best].sv;
        int prefix = c->kept[best].prefix;
        c->kept_bytes -= janas_llm_saved_bytes(sv);
        c->kept[best] = c->kept[--c->n_kept];
        if (lose)
            keep_current(c, 0);
        janas_llm_session_restore(c->s, sv); /* failed: the current stays */
        add(c, sv, prefix);
    } else if (lose) {
        keep_current(c, 0);
    }
    evict(c);
}

/*
 * A recurrent state is reused all of it or not at all, so a new
 * conversation with the system message of another (the same tools, the
 * same instructions: a client's every chat) could reuse nothing of it. The
 * system message is computed alone first and kept, marked so that the
 * conversations that grow from it do not drop it.
 */
uint32_t janas_api_keep_system(janas_llm_chat *c, uint32_t from)
{
    uint32_t sys = c->sys_tokens;
    if (c->keep_max <= 0 || !janas_llm_model_recurrent(c->llm->m) ||
        sys < KEEP_MIN || from >= sys || sys >= c->n_ids ||
        janas_llm_session_append(c->s, c->ids + from, sys - from) != 0)
        return from;
    if (janas_llm_session_prefill(c->s) == 0) {
        keep_current(c, 1);
        evict(c);
    }
    return sys;
}

/*
 * A recurrent state cannot go back from a reply to its prompt: a second
 * reply to the same prompt (n of them, or a reply asked again) read it all
 * again. So the prompt is computed now, before the reply, and kept like
 * another conversation; the next load that holds all of it finds it there.
 * A copy of what was kept before and this one holds whole goes with it.
 * The prompt goes in turn when its conversation is kept (keep_current): a
 * reply asked again right away finds it, one asked again after other
 * conversations reads it again - the price of not filling the room with a
 * prompt for every conversation (a client like Open WebUI makes four
 * requests a turn).
 */
void janas_api_keep_prompt(janas_llm_chat *c)
{
    if (c->keep_max <= 0 || !janas_llm_model_recurrent(c->llm->m) ||
        janas_llm_session_prefill(c->s) != 0)
        return; /* a failed pass fails again, and says so, in the reply */
    keep_current(c, 0);
    evict(c);
}

void janas_api_keep_free(janas_llm_chat *c)
{
    /* with a disk, what is kept goes there, and the conversation too */
    if (c->disk_dir) {
        struct janas_llm_saved *cur = janas_llm_session_save(c->s);
        if (cur) {
            janas_api_disk_put(c, cur, 0);
            janas_llm_saved_free(cur);
        }
        for (int32_t i = 0; i < c->n_kept; i++)
            janas_api_disk_put(c, c->kept[i].sv, c->kept[i].prefix);
    }
    while (c->n_kept > 0)
        drop(c, c->n_kept - 1);
    free(c->kept);
    c->kept = NULL;
    janas_api_disk_close(c);
}

/*
 * Half of the memory left free by the expert cache and by the memory kept
 * for other programs (janas_llm_params.reserve_bytes), at least
 * 256 MiB. Fixed at the call: measured at every request, the free memory
 * would shrink with the copies themselves. The cache's memory is counted
 * as taken although it fills while the model runs: at the start it still
 * looks free, and the copies would end up in the margin, pushing the
 * machine into swap.
 */
static uint64_t auto_bytes(const janas_llm *llm)
{
    uint64_t avail = janas_mem_available(), floor = (uint64_t)256 << 20;
    uint64_t taken = llm->cache_bytes + llm->reserve_bytes;
    return avail > taken + 2 * floor ? (avail - taken) / 2 : floor;
}

int32_t janas_llm_chat_keep(janas_llm_chat *c, int32_t n, uint64_t bytes)
{
    if (!c || n < 0 || n > 1024)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    /* room for n, and for the current one and the one restored while
       they change places, before the evicting */
    struct kept *k = realloc(c->kept, (size_t)(n + 3) * sizeof(*k));
    if (!k)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    c->kept = k;
    c->keep_max = n;
    c->keep_bytes = bytes ? bytes : auto_bytes(c->llm);
    evict(c);
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_kept(const janas_llm_chat *c, int32_t *n,
                            uint64_t *bytes)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    if (n)
        *n = c->n_kept;
    if (bytes)
        *bytes = c->kept_bytes;
    return JANAS_LLM_OK;
}
