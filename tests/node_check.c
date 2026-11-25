/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * node_check.c - the module for Node.js against the library, on a model:
 *
 *     node_check model.jns [seconds]
 *
 * First the library itself, here: a message and its reply, then the whole
 * conversation loaded with a second message, greedy, on the CPU alone
 * (JANAS_GPU=0, JANAS_NPU=0, for the same numbers in both processes). Then
 * node, through janas.node, says the same and much more: the replies must
 * be the same text, byte for byte, streamed and read whole, after a reset
 * and in a new chat; and leaving the loop, an AbortSignal (fired, and
 * aborted before), two replies asked at once, a prompt of several blocks
 * (progress pieces), reasoning on and off, the errors, closing twice, and a
 * model left open when node leaves. Not one of the tests run by build.sh:
 * it needs a model (Qwen3-0.6B takes about a minute).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "janas/llm.h"

#ifndef __linux__
int main(void)
{
    printf("node_check: the module is built on Linux only\n");
    return 0;
}
#else
#include "node_run.h"

static const char p1[] = "Write one sentence about the sea.";
static const char p2[] = "And one about the mountains.";

static const char script[] =
    "'use strict';\n"
    "const assert = require('node:assert/strict');\n"
    "const [addon, path, P1, P2] = process.argv.slice(1);\n"
    "const janas = require(addon);\n"
    "const hex = (s) => Buffer.from(s, 'utf8').toString('hex');\n"
    "const say = (k, v) => process.stdout.write(k + ' ' + v + '\\n');\n"
    "const pieces = async (r) => { const p = []; for await (const x of r) "
    "p.push(x); return p; };\n"
    "const answer = (p) => p.filter((x) => x.type === 'text' && "
    "!x.reasoning).map((x) => x.text).join('');\n"
    "const greedy = { temperature: 0, thinking: false, maxReply: 48 };\n"
    "(async () => {\n"
    "  const m = await janas.open(path, { context: 4096 });\n"
    "  assert.equal(m.status().open, true);\n"
    "  say('model', JSON.stringify(m.name) + ' reasons ' + m.reasons);\n"
    "  const chat = await m.chat(greedy);\n"
    "  await assert.rejects(m.chat(), { code: 'EBUSY' });\n"
    /* the reply streamed, then the conversation loaded */
    "  const r1 = chat.send(P1);\n"
    "  const t1 = answer(await pieces(r1));\n"
    "  say('t1', hex(t1));\n"
    "  const s1 = await r1.stats();\n"
    "  assert.ok(s1.outputTokens > 0 && ['stop', 'length'].includes("
    "s1.finish), JSON.stringify(s1));\n"
    "  const r2 = chat.load([{ role: 'user', content: P1 },\n"
    "    { role: 'assistant', content: t1 }, { role: 'user', content: P2 "
    "}]);\n"
    "  say('t2', hex(await r2.text()));\n"
    "  const s2 = await r2.stats();\n"
    "  say('cached', s2.cachedTokens + ' of ' + s2.promptTokens);\n"
    "  assert.ok(s2.cachedTokens > 0, 'nothing of the prompt reused');\n"
    /* the same message after a reset, read whole */
    "  await chat.reset();\n"
    "  say('t1again', hex(await chat.send(P1).text()));\n"
    /* leaving the loop, an AbortSignal fired, one aborted before */
    "  await chat.reset();\n"
    "  const r4 = chat.send('Count from one to two hundred, one number a "
    "line.');\n"
    "  let k = 0;\n"
    "  for await (const x of r4) if (x.type === 'text' && ++k === 3) break;\n"
    "  assert.equal((await r4.stats()).finish, 'stopped');\n"
    "  const ac = new AbortController();\n"
    "  const r5 = chat.send('Count from one to two hundred.', { signal: "
    "ac.signal });\n"
    "  for await (const x of r5) if (x.type === 'text') ac.abort();\n"
    "  assert.equal((await r5.stats()).finish, 'stopped');\n"
    "  const ac2 = new AbortController();\n"
    "  ac2.abort();\n"
    "  assert.equal((await pieces(chat.send('Hello', { signal: ac2.signal "
    "}))).length, 0);\n"
    /* two at once: the second waits */
    "  await chat.reset();\n"
    "  const [a, b] = await Promise.all([chat.send('Say yes.').text(), "
    "chat.send('Say no.').text()]);\n"
    "  assert.ok(a.length > 0 && b.length > 0);\n"
    /* a prompt of several blocks */
    "  await chat.reset();\n"
    "  const long = 'The quick brown fox jumps over the lazy dog. '"
    ".repeat(80) + 'How many times did the fox jump?';\n"
    "  const p7 = await pieces(chat.send(long));\n"
    "  say('progress', p7.filter((x) => x.type === 'progress').length);\n"
    "  assert.ok(p7.some((x) => x.type === 'progress'), 'no progress "
    "piece');\n"
    /* reasoning */
    "  if (m.reasons) {\n"
    "    await chat.reset();\n"
    "    await chat.set({ thinking: true, maxReply: 64 });\n"
    "    const p8 = await pieces(chat.send('What is 17 times 23?'));\n"
    "    assert.ok(p8.some((x) => x.type === 'text' && x.reasoning), 'no "
    "reasoning');\n"
    "    await chat.set(greedy);\n"
    "  }\n"
    /* errors */
    "  assert.throws(() => chat.load([]), TypeError);\n"
    "  assert.throws(() => chat.load([{ role: 'robot', content: 'x' }]), "
    "/message 0/);\n"
    "  assert.throws(() => chat.send(7), TypeError);\n"
    "  assert.throws(() => chat.set({ temperature: -1 }), TypeError);\n"
    "  assert.throws(() => chat.send('x', { signal: {} }), "
    "/AbortSignal/);\n"
    /* the chat closed, another */
    "  await chat.close();\n"
    "  assert.throws(() => chat.send('x'), { code: 'ECLOSED' });\n"
    "  const chat2 = await m.chat(greedy);\n"
    "  say('t1chat2', hex(await chat2.send(P1).text()));\n"
    /* closed twice, then nothing */
    "  await Promise.all([m.close(), m.close()]);\n"
    "  assert.equal(m.status().open, false);\n"
    "  assert.throws(() => chat2.send('x'), { code: 'ECLOSED' });\n"
    "  await assert.rejects(m.chat(), { code: 'ECLOSED' });\n"
    /* one left open: node leaves all the same, closing it */
    "  const m2 = await janas.open(path, { context: 1024 });\n"
    "  await (await m2.chat({ ...greedy, maxReply: 8 })).send('Hi').text();\n"
    "  say('done', 1);\n"
    "})().catch((e) => { console.error(e); process.exitCode = 1; });\n";

/* the reply's answer, its reasoning left out */
static char *reply_text(janas_llm_chat *c, int32_t rc)
{
    size_t len = 0, cap = 4096;
    char *s = malloc(cap), buf[4096];
    int32_t n;
    while (s && rc == JANAS_LLM_OK &&
           (rc = janas_llm_chat_next(c, buf, sizeof(buf), &n)) ==
               JANAS_LLM_OK) {
        if (n <= 0 || janas_llm_chat_thinking(c))
            continue;
        if (len + (size_t)n + 1 > cap) {
            char *b = realloc(s, cap *= 2);
            if (!b) {
                free(s);
                return NULL;
            }
            s = b;
        }
        memcpy(s + len, buf, (size_t)n);
        len += (size_t)n;
    }
    if (!s || rc < 0) {
        fprintf(stderr, "node_check: %s\n", janas_llm_last_error());
        free(s);
        return NULL;
    }
    s[len] = 0;
    return s;
}

static char *to_hex(const char *s)
{
    size_t n = strlen(s);
    char *h = malloc(2 * n + 1);
    for (size_t i = 0; h && i < n; i++)
        sprintf(h + 2 * i, "%02x", (unsigned char)s[i]);
    if (h)
        h[2 * n] = 0;
    return h;
}

/* the line "key value" of node's output: its value is want */
static int same(const char *out, const char *key, const char *want)
{
    size_t kl = strlen(key);
    for (const char *line = out; *line;) {
        size_t n = strcspn(line, "\n");
        if (n > kl && !strncmp(line, key, kl) && line[kl] == ' ')
            return n - kl - 1 == strlen(want) &&
                   !strncmp(line + kl + 1, want, n - kl - 1);
        line += n + (line[n] == '\n');
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: node_check model.jns [seconds]\n");
        return 2;
    }
    int timeout_s = argc > 2 ? atoi(argv[2]) : 600;
    char node[4096], addon[4096];
    if (!node_find(node, sizeof(node)) || !node_addon(addon, sizeof(addon))) {
        fprintf(stderr, "node_check: no node in the PATH, or no janas.node "
                        "beside this program\n");
        return 2;
    }
    /* the same numbers in both processes: the CPU's */
    setenv("JANAS_GPU", "0", 1);
    setenv("JANAS_NPU", "0", 1);

    struct janas_llm_params p;
    janas_llm_params_default(&p);
    p.n_ctx = 4096;
    janas_llm *llm;
    if (janas_llm_open(argv[1], &p, &llm) != JANAS_LLM_OK) {
        fprintf(stderr, "node_check: %s\n", janas_llm_last_error());
        return 2;
    }
    struct janas_llm_chat_params cp;
    janas_llm_chat_params_default(&cp);
    cp.temperature = 0;
    cp.thinking = 0;
    cp.max_reply = 48;
    janas_llm_chat *c;
    char *t1 = NULL, *t2 = NULL;
    if (janas_llm_chat_create(llm, &cp, &c) == JANAS_LLM_OK) {
        t1 = reply_text(c, janas_llm_chat_send(c, p1, -1));
        if (t1) {
            const int32_t roles[] = {JANAS_LLM_ROLE_USER,
                                     JANAS_LLM_ROLE_ASSISTANT,
                                     JANAS_LLM_ROLE_USER};
            const char *texts[] = {p1, t1, p2};
            t2 = reply_text(c, janas_llm_chat_load(c, 3, roles, texts, NULL));
        }
        janas_llm_chat_destroy(c);
    }
    janas_llm_close(llm);
    if (!t1 || !t2) {
        fprintf(stderr, "node_check: the library's replies failed\n");
        return 1;
    }
    printf("library: \"%s\" / \"%s\"\n", t1, t2);

    char *out = malloc(1 << 20);
    const char *flags[] = {"--unhandled-rejections=strict", NULL};
    const char *args[] = {addon, argv[1], p1, p2, NULL};
    int st =
        out ? node_run(node, flags, script, args, NULL, timeout_s, out, 1 << 20)
            : -1;
    char *h1 = to_hex(t1), *h2 = to_hex(t2);
    struct {
        const char *key, *want, *what;
    } cmp[] = {
        {"t1", h1, "the reply streamed"},
        {"t2", h2, "the conversation loaded, read whole"},
        {"t1again", h1, "the same message after a reset"},
        {"t1chat2", h1, "the same message in a new chat"},
    };
    int failures = st != 0;
    printf("node: exit %d%s\n", st,
           out && strstr(out, "\ndone 1") ? "" : ", did not finish");
    if (!out || !strstr(out, "\ndone 1"))
        failures++;
    for (size_t i = 0; out && i < sizeof(cmp) / sizeof(cmp[0]); i++) {
        int ok = h1 && h2 && same(out, cmp[i].key, cmp[i].want);
        printf("%-5s %s\n", ok ? "ok" : "FAIL", cmp[i].what);
        failures += !ok;
    }
    if (out) {
        const char *keys[] = {"model", "cached", "progress"};
        for (size_t i = 0; i < 3; i++) {
            char k[32];
            snprintf(k, sizeof(k), "%s ", keys[i]);
            const char *q = strstr(out, k);
            if (q)
                printf("node: %.*s\n", (int)strcspn(q, "\n"), q);
        }
    }
    printf("node_check: %s\n", failures ? "FAIL" : "ok");
    free(out);
    free(h1);
    free(h2);
    free(t1);
    free(t2);
    return failures ? 1 : 0;
}
#endif
