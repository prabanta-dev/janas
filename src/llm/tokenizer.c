/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tokenizer.c - byte-level BPE (see tokenizer.h).
 */
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unicode_data.h"

struct merge {
    uint64_t pair; /* (left id + 1) << 32 | right id: 0 marks a free slot */
    uint32_t rank, result;
};

struct janas_tokenizer {
    uint32_t n;
    char *pool;     /* token texts, back to back */
    uint32_t *off;  /* n + 1 offsets into pool */
    uint32_t *vmap; /* text -> id + 1, open addressing; 0: free */
    size_t vcap;
    struct merge *mmap;
    size_t mcap;
    int32_t byte_id[256]; /* token of the byte-level character of a byte */
    uint8_t cp_byte[512]; /* byte-level code point -> byte */
    int32_t *special;     /* ids of the special tokens, longest first */
    uint32_t n_special;
    int marks; /* words take combining marks too (the qwen35 expression) */
    int spm;   /* SentencePiece style (gemma4): raw UTF-8 characters, spaces
                  written as U+2581, unknown characters as <0xXX> bytes */
};

static uint64_t hash_bytes(const char *s, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++)
        h = (h ^ (uint8_t)s[i]) * 0x100000001b3ull;
    return h;
}

static uint64_t hash_u64(uint64_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return x;
}

static int32_t lookup(const struct janas_tokenizer *t, const char *s, size_t n)
{
    for (size_t i = hash_bytes(s, n) & (t->vcap - 1);;
         i = (i + 1) & (t->vcap - 1)) {
        uint32_t v = t->vmap[i];
        if (!v)
            return -1;
        uint32_t id = v - 1, len = t->off[id + 1] - t->off[id];
        if (len == n && memcmp(t->pool + t->off[id], s, n) == 0)
            return (int32_t)id;
    }
}

static const struct merge *find_merge(const struct janas_tokenizer *t,
                                      int32_t a, int32_t b)
{
    uint64_t key = ((uint64_t)(uint32_t)a + 1) << 32 | (uint32_t)b;
    for (size_t i = hash_u64(key) & (t->mcap - 1);;
         i = (i + 1) & (t->mcap - 1)) {
        if (t->mmap[i].pair == key)
            return &t->mmap[i];
        if (!t->mmap[i].pair)
            return NULL;
    }
}

static size_t pow2_above(size_t n)
{
    size_t c = 16;
    while (c < 2 * n)
        c <<= 1;
    return c;
}

/* UTF-8 of code point cp into out (at most 4 bytes); returns the length. */
static int put_utf8(uint32_t cp, char *out)
{
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | cp >> 6);
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | cp >> 12);
        out[1] = (char)(0x80 | (cp >> 6 & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | cp >> 18);
    out[1] = (char)(0x80 | (cp >> 12 & 0x3F));
    out[2] = (char)(0x80 | (cp >> 6 & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* The length of the UTF-8 sequence led by byte c, as its lead byte says,
   at most n (a stray continuation byte counts as one). */
static size_t lead_len(unsigned char c, size_t n)
{
    size_t k = c < 0xC0 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    return k < n ? k : n;
}

/* Decodes one code point at s (n bytes left); *len its length. Invalid
   sequences decode to 0xFFFD, one byte long. */
static uint32_t get_utf8(const unsigned char *s, size_t n, int *len)
{
    uint32_t c = s[0];
    int k = c < 0x80         ? 1
            : (c >> 5) == 6  ? 2
            : (c >> 4) == 14 ? 3
            : (c >> 3) == 30 ? 4
                             : 0;
    if (k == 0 || (size_t)k > n) {
        *len = 1;
        return 0xFFFD;
    }
    uint32_t cp = k == 1 ? c : c & (0x7F >> k);
    for (int i = 1; i < k; i++) {
        if ((s[i] & 0xC0) != 0x80) {
            *len = 1;
            return 0xFFFD;
        }
        cp = cp << 6 | (s[i] & 0x3F);
    }
    *len = k;
    return cp;
}

static int in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0])
            hi = mid;
        else if (cp > r[mid][1])
            lo = mid + 1;
        else
            return 1;
    }
    return 0;
}

#define NR(a) (sizeof(a) / sizeof(a[0]))

static int is_letter(uint32_t c)
{
    return in_ranges(janas_uni_letter, NR(janas_uni_letter), c);
}

static int is_number(uint32_t c)
{
    return in_ranges(janas_uni_number, NR(janas_uni_number), c);
}

static int is_mark(uint32_t c)
{
    return in_ranges(janas_uni_mark, NR(janas_uni_mark), c);
}

static int is_space(uint32_t c)
{
    return in_ranges(janas_uni_space, NR(janas_uni_space), c);
}

static int is_newline(uint32_t c)
{
    return c == '\r' || c == '\n';
}

struct janas_tokenizer *janas_tokenizer_create(const struct janas_jns *j,
                                               char *err, size_t err_len)
{
    uint64_t n, nm, nt;
    const uint8_t *toks, *merges;
    const int32_t *types;
    if (err_len)
        err[0] = 0;
    if (janas_jns_meta_strings(j, "tokenizer.ggml.tokens", &n, &toks) != 0 ||
        janas_jns_meta_strings(j, "tokenizer.ggml.merges", &nm, &merges) != 0 ||
        janas_jns_meta_ints(j, "tokenizer.ggml.token_type", &nt, &types) != 0 ||
        nt != n || n == 0 || n > (1u << 24)) {
        snprintf(err, err_len, "no BPE tokenizer in the model file");
        return NULL;
    }
    char pre[32] = "", model[32] = "";
    janas_jns_meta_str(j, "tokenizer.ggml.pre", pre, sizeof(pre));
    janas_jns_meta_str(j, "tokenizer.ggml.model", model, sizeof(model));
    struct janas_tokenizer *t = calloc(1, sizeof(*t));
    if (!t) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    t->n = (uint32_t)n;
    t->marks = strcmp(pre, "qwen35") == 0;
    t->spm = strcmp(model, "gemma4") == 0;
    uint64_t total = 0;
    const uint8_t *p = toks;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t len;
        memcpy(&len, p, 8);
        total += len;
        p += 8 + len;
    }
    t->pool = malloc(total + 1);
    t->off = malloc((n + 1) * sizeof(uint32_t));
    t->vcap = pow2_above(n);
    t->vmap = calloc(t->vcap, sizeof(uint32_t));
    t->mcap = pow2_above(nm);
    t->mmap = calloc(t->mcap, sizeof(struct merge));
    t->special = malloc(n * sizeof(int32_t));
    if (!t->pool || !t->off || !t->vmap || !t->mmap || !t->special)
        goto fail;
    p = toks;
    uint32_t pos = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t len;
        memcpy(&len, p, 8);
        memcpy(t->pool + pos, p + 8, len);
        t->off[i] = pos;
        pos += (uint32_t)len;
        p += 8 + len;
    }
    t->off[n] = pos;
    for (uint32_t i = 0; i < t->n; i++) {
        const char *s = t->pool + t->off[i];
        size_t len = t->off[i + 1] - t->off[i];
        if (lookup(t, s, len) >= 0)
            continue; /* a duplicate text keeps the lower id */
        size_t k = hash_bytes(s, len) & (t->vcap - 1);
        while (t->vmap[k])
            k = (k + 1) & (t->vcap - 1);
        t->vmap[k] = i + 1;
        int32_t type;
        memcpy(&type, types + i, 4);
        /* not a normal token nor a byte: control or user-defined */
        if (type != 1 && type != 6)
            t->special[t->n_special++] = (int32_t)i;
    }
    /* special tokens, longest first, so that none hides a longer one */
    for (uint32_t a = 1; a < t->n_special; a++)
        for (uint32_t b = a; b > 0; b--) {
            int32_t x = t->special[b - 1], y = t->special[b];
            if (t->off[x + 1] - t->off[x] >= t->off[y + 1] - t->off[y])
                break;
            t->special[b - 1] = y;
            t->special[b] = x;
        }
    /* merges "A B" in rank order */
    p = merges;
    for (uint64_t r = 0; r < nm; r++) {
        uint64_t len;
        memcpy(&len, p, 8);
        const char *s = (const char *)p + 8;
        p += 8 + len;
        const char *sp = memchr(s, ' ', len);
        if (!sp)
            continue;
        size_t la = (size_t)(sp - s), lb = len - la - 1;
        int32_t a = lookup(t, s, la), b = lookup(t, sp + 1, lb);
        char buf[512];
        if (a < 0 || b < 0 || la + lb > sizeof(buf))
            continue;
        memcpy(buf, s, la);
        memcpy(buf + la, sp + 1, lb);
        int32_t res = lookup(t, buf, la + lb);
        if (res < 0)
            continue;
        uint64_t key = ((uint64_t)(uint32_t)a + 1) << 32 | (uint32_t)b;
        size_t k = hash_u64(key) & (t->mcap - 1);
        while (t->mmap[k].pair && t->mmap[k].pair != key)
            k = (k + 1) & (t->mcap - 1);
        if (t->mmap[k].pair)
            continue; /* the first (lowest) rank wins */
        t->mmap[k] = (struct merge){key, (uint32_t)r, (uint32_t)res};
    }
    /* SentencePiece style: each byte has a token <0xXX> */
    for (int b = 0; t->spm && b < 256; b++) {
        char u[8];
        snprintf(u, sizeof(u), "<0x%02X>", b);
        t->byte_id[b] = lookup(t, u, 6);
        if (t->byte_id[b] < 0) {
            snprintf(err, err_len, "byte %d has no token", b);
            goto fail;
        }
    }
    /* the byte-level alphabet: printable bytes stand for themselves, the
       others for code points from 256 on, in byte order */
    int next = 0;
    for (int b = 0; !t->spm && b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                        (b >= 174 && b <= 255);
        uint32_t cp = printable ? (uint32_t)b : 256u + (uint32_t)next++;
        char u[4];
        int l = put_utf8(cp, u);
        t->byte_id[b] = lookup(t, u, (size_t)l);
        t->cp_byte[cp] = (uint8_t)b;
        if (t->byte_id[b] < 0) {
            snprintf(err, err_len, "byte %d has no token", b);
            goto fail;
        }
    }
    return t;

fail:
    if (err_len && err[0] == 0)
        snprintf(err, err_len, "out of memory");
    janas_tokenizer_destroy(t);
    return NULL;
}

void janas_tokenizer_destroy(struct janas_tokenizer *t)
{
    if (!t)
        return;
    free(t->pool);
    free(t->off);
    free(t->vmap);
    free(t->mmap);
    free(t->special);
    free(t);
}

uint32_t janas_tokenizer_n_vocab(const struct janas_tokenizer *t)
{
    return t->n;
}

uint32_t janas_tokenizer_specials(const struct janas_tokenizer *t,
                                  const int32_t **ids)
{
    *ids = t->special;
    return t->n_special;
}

int32_t janas_tokenizer_find(const struct janas_tokenizer *t, const char *s)
{
    return lookup(t, s, strlen(s));
}

/*
 * The end of the pre-token starting at code point i of cp[0 .. n), by the
 * model's expression, alternative by alternative, first match wins:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
 *   | ' '?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * With marks (Qwen3.5 and later), \p{L} in the word runs and in the
 * punctuation alternative's exclusion becomes [\p{L}\p{M}].
 */
static size_t pretoken_end(const uint32_t *cp, size_t n, size_t i, int marks)
{
#define WORD(c) (is_letter(c) || (marks && is_mark(c)))
    uint32_t c = cp[i];
    if (c == '\'' && i + 1 < n) {
        uint32_t a = cp[i + 1] | 0x20, b = i + 2 < n ? cp[i + 2] | 0x20 : 0;
        if (cp[i + 1] < 0x80) {
            if (a == 's' || a == 't')
                return i + 2;
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e'))
                return i + 3;
            if (a == 'm')
                return i + 2;
            if (a == 'l' && b == 'l')
                return i + 3;
            if (a == 'd')
                return i + 2;
        }
    }
    size_t k = i;
    if (!is_letter(c) && !is_newline(c) && !is_number(c) && i + 1 < n &&
        WORD(cp[i + 1]))
        k = i + 1;
    if (WORD(cp[k])) {
        while (k < n && WORD(cp[k]))
            k++;
        return k;
    }
    if (is_number(c))
        return i + 1;
    k = i + (c == ' ');
    if (k < n && !is_space(cp[k]) && !WORD(cp[k]) && !is_number(cp[k])) {
        while (k < n && !is_space(cp[k]) && !WORD(cp[k]) && !is_number(cp[k]))
            k++;
        while (k < n && is_newline(cp[k]))
            k++;
        return k;
    }
    if (is_space(c)) {
        size_t e = i, last_nl = 0;
        while (e < n && is_space(cp[e])) {
            if (is_newline(cp[e]))
                last_nl = e + 1;
            e++;
        }
        if (last_nl)
            return last_nl;
        if (e == n)
            return e;
        if (e - i >= 2)
            return e - 1; /* leave the last space to the next word */
        return e;
    }
    return i + 1;
}

struct cand {
    uint32_t rank, pos; /* a mergeable pair, by its left symbol */
};

static int cand_less(struct cand a, struct cand b)
{
    return a.rank < b.rank || (a.rank == b.rank && a.pos < b.pos);
}

static void heap_push(struct cand *h, size_t *n, struct cand c)
{
    size_t i = (*n)++;
    while (i > 0 && cand_less(c, h[(i - 1) / 2])) {
        h[i] = h[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    h[i] = c;
}

static struct cand heap_pop(struct cand *h, size_t *n)
{
    struct cand top = h[0], last = h[--*n];
    size_t i = 0;
    for (;;) {
        size_t c = 2 * i + 1;
        if (c >= *n)
            break;
        if (c + 1 < *n && cand_less(h[c + 1], h[c]))
            c++;
        if (!cand_less(h[c], last))
            break;
        h[i] = h[c];
        i = c;
    }
    h[i] = last;
    return top;
}

static void push_pair(const struct janas_tokenizer *t, const int32_t *sym,
                      int32_t left, int32_t right, struct cand *h, size_t *n)
{
    if (left < 0 || right < 0 || sym[left] < 0 || sym[right] < 0)
        return;
    const struct merge *m = find_merge(t, sym[left], sym[right]);
    if (m)
        heap_push(h, n, (struct cand){m->rank, (uint32_t)left});
}

/*
 * BPE on one pre-token; appends the ids. The symbols are its bytes, or in
 * SentencePiece style its characters, and a character with no token (-2)
 * goes out as its bytes' <0xXX> tokens. The pair of lowest
 * rank merges first, the leftmost among equal ones; symbols live in a linked
 * list and the candidate pairs in a heap, so a long word costs n log n.
 * Heap entries go stale when a neighbour merges: they are checked on pop.
 */
static void bpe(const struct janas_tokenizer *t, const unsigned char *s,
                size_t len, int32_t *out, size_t max, long *count)
{
    enum { SMALL = 128 };
    int32_t small_link[4 * SMALL];
    struct cand small_heap[3 * SMALL];
    int32_t *link = small_link;
    struct cand *h = small_heap;
    if (len > SMALL) {
        link = malloc(4 * len * sizeof(int32_t));
        h = malloc(3 * len * sizeof(struct cand));
        if (!link || !h) {
            free(link);
            free(h);
            return;
        }
    }
    int32_t *sym = link, *prev = link + len, *next = link + 2 * len;
    int32_t *at = link + 3 * len;
    size_t ns = 0;
    for (size_t b = 0; b < len; ns++) {
        size_t l = t->spm ? lead_len(s[b], len - b) : 1;
        sym[ns] = t->spm ? lookup(t, (const char *)s + b, l) : t->byte_id[s[b]];
        if (sym[ns] < 0)
            sym[ns] = -2;
        at[ns] = (int32_t)b;
        b += l;
    }
    for (size_t i = 0; i < ns; i++) {
        prev[i] = (int32_t)i - 1;
        next[i] = i + 1 < ns ? (int32_t)i + 1 : -1;
    }
    size_t nh = 0;
    for (size_t i = 0; i + 1 < ns; i++)
        push_pair(t, sym, (int32_t)i, (int32_t)i + 1, h, &nh);
    while (nh > 0) {
        struct cand c = heap_pop(h, &nh);
        int32_t l = (int32_t)c.pos, r = next[l];
        if (sym[l] < 0 || r < 0)
            continue;
        const struct merge *m = find_merge(t, sym[l], sym[r]);
        if (!m || m->rank != c.rank)
            continue;
        sym[l] = (int32_t)m->result;
        sym[r] = -1;
        next[l] = next[r];
        if (next[r] >= 0)
            prev[next[r]] = l;
        push_pair(t, sym, prev[l], l, h, &nh);
        push_pair(t, sym, l, next[l], h, &nh);
    }
    for (int32_t i = ns ? 0 : -1; i >= 0; i = next[i]) {
        if (sym[i] == -2) {
            size_t b = (size_t)at[i], l = lead_len(s[b], len - b);
            for (size_t k = 0; k < l; k++, (*count)++)
                if ((size_t)*count < max)
                    out[*count] = t->byte_id[s[b + k]];
            continue;
        }
        if ((size_t)*count < max)
            out[*count] = sym[i];
        (*count)++;
    }
    if (link != small_link) {
        free(link);
        free(h);
    }
}

/*
 * Plain text, SentencePiece style: spaces become U+2581, the text is cut
 * only between runs of newlines and the rest; a run of newlines that is a
 * token goes as it is, anything else through BPE.
 */
static int encode_spm(const struct janas_tokenizer *t, const char *text,
                      size_t len, int32_t *out, size_t max, long *count)
{
    char *e = malloc(3 * len + 1);
    if (!e)
        return -1;
    size_t n = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] == ' ') {
            memcpy(e + n, "\xE2\x96\x81", 3);
            n += 3;
        } else {
            e[n++] = text[i];
        }
    }
    for (size_t i = 0; i < n;) {
        int nl = e[i] == '\n';
        size_t k = i;
        while (k < n && (e[k] == '\n') == nl)
            k++;
        int32_t id = nl ? lookup(t, e + i, k - i) : -1;
        if (id >= 0) {
            if ((size_t)*count < max)
                out[*count] = id;
            (*count)++;
        } else {
            bpe(t, (const unsigned char *)e + i, k - i, out, max, count);
        }
        i = k;
    }
    free(e);
    return 0;
}

/* Plain text (no special tokens): pre-tokens, then BPE on each. */
static int encode_plain(const struct janas_tokenizer *t, const char *text,
                        size_t len, int32_t *out, size_t max, long *count)
{
    if (t->spm)
        return encode_spm(t, text, len, out, max, count);
    uint32_t *cp = malloc((len + 1) * sizeof(uint32_t));
    size_t *at = malloc((len + 1) * sizeof(size_t));
    if (!cp || !at) {
        free(cp);
        free(at);
        return -1;
    }
    size_t n = 0;
    for (size_t b = 0; b < len;) {
        int l;
        cp[n] = get_utf8((const unsigned char *)text + b, len - b, &l);
        at[n++] = b;
        b += (size_t)l;
    }
    at[n] = len;
    for (size_t i = 0; i < n;) {
        size_t e = pretoken_end(cp, n, i, t->marks);
        bpe(t, (const unsigned char *)text + at[i], at[e] - at[i], out, max,
            count);
        i = e;
    }
    free(cp);
    free(at);
    return 0;
}

long janas_tokenizer_encode(const struct janas_tokenizer *t, const char *text,
                            size_t len, int special, int32_t *out, size_t max)
{
    long count = 0;
    size_t start = 0;
    for (size_t i = 0; special && i < len; i++) {
        for (uint32_t k = 0; k < t->n_special; k++) {
            int32_t id = t->special[k];
            size_t l = t->off[id + 1] - t->off[id];
            if (l == 0 || l > len - i ||
                memcmp(text + i, t->pool + t->off[id], l) != 0)
                continue;
            if (encode_plain(t, text + start, i - start, out, max, &count))
                return -1;
            if ((size_t)count < max)
                out[count] = id;
            count++;
            i += l - 1;
            start = i + 1;
            break;
        }
    }
    if (encode_plain(t, text + start, len - start, out, max, &count))
        return -1;
    return count;
}

size_t janas_tokenizer_decode(const struct janas_tokenizer *t, int32_t id,
                              char *out, size_t max)
{
    if (id < 0 || (uint32_t)id >= t->n)
        return 0;
    const unsigned char *s = (const unsigned char *)t->pool + t->off[id];
    size_t len = t->off[id + 1] - t->off[id], w = 0;
    int is_special = 0;
    for (uint32_t k = 0; k < t->n_special && !is_special; k++)
        is_special = t->special[k] == id;
    if (is_special) {
        for (size_t i = 0; i < len; i++, w++)
            if (w < max)
                out[w] = (char)s[i];
        return w;
    }
    if (t->spm) {
        if (len == 6 && memcmp(s, "<0x", 3) == 0) {
            int b = (int)strtol((const char *)s + 3, NULL, 16);
            if (b >= 0 && b < 256 && t->byte_id[b] == id) {
                if (max)
                    out[0] = (char)b;
                return 1;
            }
        }
        for (size_t i = 0; i < len;) {
            int sp = i + 3 <= len && memcmp(s + i, "\xE2\x96\x81", 3) == 0;
            if (w < max)
                out[w] = sp ? ' ' : (char)s[i];
            w++;
            i += sp ? 3 : 1;
        }
        return w;
    }
    for (size_t i = 0; i < len;) {
        int l;
        uint32_t cp = get_utf8(s + i, len - i, &l);
        if (w < max)
            out[w] = cp < 512 ? (char)t->cp_byte[cp] : '?';
        w++;
        i += (size_t)l;
    }
    return w;
}
