/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_node.c - the module for Node.js (janas.node) without a model (a
 * speech model neither): node
 * loads it; its classes refuse new, and their methods any other object;
 * wrong arguments and options throw TypeErrors that name them; a file that
 * is not there rejects with JANAS_EOPEN, several at once too, each engine
 * thread started and ended; and node then exits by itself, the module
 * keeping nothing alive. Skipped where node is not in the PATH.
 *
 * Where Node's headers are installed, src/node/napi.h is checked against
 * them as well: its declarations follow Node's here, and the compiler stops
 * on any that differs.
 */
#if !defined(_WIN32) && defined(__has_include)
#if __has_include(<node/node_api.h>)
#include <node/node_api.h>
#define JANAS_NAPI_NODE_TYPES
#include "node/napi.h"
#endif
#endif

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
int main(void)
{
    printf("test_node: skipped, the module is built on Linux only\n");
    return 0;
}
#else
#include "node_run.h"

static const char script[] =
    "'use strict';\n"
    "const assert = require('node:assert/strict');\n"
    "const janas = require(process.argv[1]);\n"
    "let checks = 0;\n"
    "const check = (f) => { f(); checks++; };\n"
    "check(() => {\n"
    "  assert.equal(typeof janas.open, 'function');\n"
    "  assert.equal(janas.abiVersion(), 1);\n"
    "  assert.equal(typeof janas.version, 'string');\n"
    "  for (const c of ['Model', 'Chat', 'Reply', 'Speech'])\n"
    "    assert.equal(typeof janas[c], 'function');\n"
    "  assert.equal(typeof janas.Reply.prototype[Symbol.asyncIterator], "
    "'function');\n"
    "});\n"
    "check(() => {\n"
    "  assert.throws(() => new janas.Model(), /comes from open\\(\\)/);\n"
    "  assert.throws(() => new janas.Chat(), TypeError);\n"
    "  assert.throws(() => new janas.Reply(), TypeError);\n"
    "  assert.throws(() => new janas.Speech(), /comes from openSpeech/);\n"
    "  assert.throws(() => janas.Speech.prototype.transcribe.call({}), "
    "TypeError);\n"
    "  assert.throws(() => janas.Chat.prototype.image.call({}), TypeError);\n"
    "  assert.throws(() => janas.Model.prototype.chat.call({}), TypeError);\n"
    "  assert.throws(() => janas.Reply.prototype.next.call("
    "janas.Chat.prototype), TypeError);\n"
    "});\n"
    "check(() => {\n"
    "  assert.throws(() => janas.open(), TypeError);\n"
    "  assert.throws(() => janas.open(42), TypeError);\n"
    "  assert.throws(() => janas.open('m.jns', 3), /must be an object/);\n"
    "  assert.throws(() => janas.open('m.jns', { cache: 4 }), "
    "/no option 'cache'/);\n"
    "  assert.throws(() => janas.open('m.jns', { context: 'big' }), "
    "/context must be a number/);\n"
    "  assert.throws(() => janas.open('m.jns', { context: 1.5 }), "
    "/whole number/);\n"
    "  assert.throws(() => janas.open('m.jns', { mode: 'fast' }), "
    "/mode cannot be 'fast'/);\n"
    "  assert.throws(() => janas.open('m.jns', { expertBits: 3 }), "
    "/expertBits/);\n"
    "  assert.throws(() => janas.open('m.jns', { mtp: 5 }), TypeError);\n"
    "  assert.throws(() => janas.openSpeech(), TypeError);\n"
    "  assert.throws(() => janas.openSpeech('w.jns', { vad: 'x' }), "
    "/no option 'vad'/);\n"
    "  assert.throws(() => janas.openSpeech('w.jns', { threads: -1 }), "
    "/threads/);\n"
    "});\n"
    "(async () => {\n"
    "  for (let i = 0; i < 3; i++)\n"
    "    await assert.rejects(janas.open('/nonexistent/janas-test.jns'),\n"
    "      (e) => e instanceof Error && e.code === 'EOPEN');\n"
    "  checks++;\n"
    "  await assert.rejects(janas.openSpeech('/nonexistent/janas-test.jns'),\n"
    "    { code: 'EOPEN' });\n"
    "  checks++;\n"
    "  await Promise.all([0, 1, 2, 3].map(() => assert.rejects(\n"
    "    janas.open('/nonexistent/janas-test.jns', { context: 1024 }),\n"
    "    { code: 'EOPEN' })));\n"
    "  checks++;\n"
    "  console.log('checks ' + checks);\n"
    "})().catch((e) => { console.error(e); process.exitCode = 1; });\n";

int main(void)
{
    char node[4096], addon[4096], out[4096];
    if (!node_find(node, sizeof(node))) {
        printf("test_node: skipped, no node in the PATH\n");
        return 0;
    }
    if (!node_addon(addon, sizeof(addon))) {
        printf("test_node: FAIL, no janas.node beside the test\n");
        return 1;
    }
    const char *args[] = {addon, NULL};
    int st = node_run(node, NULL, script, args, NULL, 120, out, sizeof(out));
    int ok = st == 0 && strstr(out, "checks 6") != NULL;
    printf("test_node: %s (%s, exit %d: %s)\n", ok ? "ok" : "FAIL", addon, st,
           out[0] ? strtok(out, "\n") : "no output");
    return ok ? 0 : 1;
}
#endif
