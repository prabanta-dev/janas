// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// logic.test.ts - the parts of the extension that need no editor: the
// versions side by side and going back, the release's tags and digests,
// the catalog read and advised, janas-setup's plan read back. node --test.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import { advise, parseCatalog, parseInstalled } from "../src/models";
import { parsePlan } from "../src/plan";
import { checkSums, compareTags, digestOf, packageName } from "../src/release";
import { binDir, inUse, janasRoot, previous, prune, rollback, setCurrent } from "../src/versions";

test("Janas's folder, as janas_models_dir has it", () => {
  assert.equal(janasRoot({ XDG_DATA_HOME: "/d" }, "linux"), path.join("/d", "janas"));
  assert.equal(janasRoot({ LOCALAPPDATA: "C:/L" }, "win32"), path.join("C:/L", "janas"));
  assert.equal(packageName("linux", "x64"), "janas-linux-x86_64.tar.gz");
  assert.equal(packageName("win32", "x64"), "janas-windows-x86_64.zip");
  assert.equal(packageName("darwin", "arm64"), null);
});

test("versions side by side, going back and forth", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "janas-ext-"));
  try {
    for (const t of ["v1", "v2", "v3"]) fs.mkdirSync(binDir(root, t), { recursive: true });
    assert.equal(rollback(root), null);
    setCurrent(root, "v1");
    setCurrent(root, "v2");
    assert.equal(inUse(root), "v2");
    assert.equal(previous(root), "v1");
    assert.equal(rollback(root), "v1");
    assert.equal(inUse(root), "v1");
    assert.equal(previous(root), "v2");
    assert.equal(rollback(root), "v2");
    setCurrent(root, "v3");
    assert.deepEqual(prune(root), ["v1"]);
    assert.ok(fs.existsSync(binDir(root, "v2")) && fs.existsSync(binDir(root, "v3")));
    // the previous one gone: nothing to go back to
    fs.rmSync(binDir(root, "v2"), { recursive: true });
    assert.equal(rollback(root), null);
    assert.equal(inUse(root), "v3");
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("tags in order", () => {
  assert.equal(compareTags("v2026.10.21", "v2026.10.21.1"), -1);
  assert.equal(compareTags("v2026.10.29", "v2026.10.21.1"), 1);
  assert.equal(compareTags("v2026.11.7", "v2026.11.7"), 0);
  assert.equal(compareTags("v2026.11.10", "v2026.11.9"), 1);
});

test("digests, published and inside the package", async () => {
  const h = "a".repeat(64);
  assert.equal(digestOf(`${h}  janas-linux-x86_64.tar.gz\n`, "janas-linux-x86_64.tar.gz"), h);
  assert.equal(digestOf(`${h}  other.zip\n`, "janas-linux-x86_64.tar.gz"), null);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "janas-ext-"));
  try {
    fs.mkdirSync(path.join(dir, "node"));
    fs.writeFileSync(path.join(dir, "janas-get"), "x");
    fs.writeFileSync(path.join(dir, "node", "janas.node"), "y");
    const d = (s: string) => createHash("sha256").update(s).digest("hex");
    fs.writeFileSync(path.join(dir, "SHA256SUMS"), `${d("x")}  ./janas-get\n${d("y")}  ./node/janas.node\n`);
    assert.deepEqual(await checkSums(dir), []);
    fs.writeFileSync(path.join(dir, "janas-get"), "changed");
    assert.deepEqual(await checkSums(dir), ["janas-get"]);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

const LIST = `model                  download  memory
qwen3-0.6b                0.4 GB    2 GB  Qwen3-0.6B, dense - also the draft of Qwen3-4B
qwen3-4b                  2.5 GB    8 GB  Qwen3-4B, dense - the quick start
qwen3.5-0.8b              0.5 GB    2 GB  Qwen3.5-0.8B, dense - a test model of the Qwen3.5 architecture, not for chatting
qwen3-coder-30b-a3b      18.6 GB   32 GB  Qwen3-Coder-30B-A3B-Instruct, mixture of experts, for code and completion in the editor
qwen3-coder-next         48.4 GB   32 GB  Qwen3-Coder-Next, mixture of experts, for code
qwen3-embedding-0.6b      0.6 GB    2 GB  Qwen3-Embedding-0.6B, for janas-server --embedding-model

Downloaded converted from Hugging Face (prabanta-dev); --from-gguf converts here.`;

test("the catalog, advised for the memory", () => {
  const c = parseCatalog(LIST);
  assert.equal(c.length, 6);
  // 32 GB: the largest coder that fits, coders first
  let a = advise(c, 32);
  assert.equal(a[0]?.model.name, "qwen3-coder-next");
  assert.ok(a[0]?.recommended);
  assert.equal(a[1]?.model.name, "qwen3-coder-30b-a3b");
  assert.ok(!a.some((x) => /embedding|3\.5-0\.8b/.test(x.model.name)));
  // 16 GB: no coder fits, the largest general one; the coders said slow
  a = advise(c, 16);
  assert.equal(a[0]?.model.name, "qwen3-4b");
  assert.match(a[0]?.note ?? "", /no model for code fits/);
  assert.match(a.find((x) => x.model.name === "qwen3-coder-30b-a3b")?.note ?? "", /slowly/);
});

test("janas-setup's lists and plan read back", () => {
  const m = parseInstalled(`Memory: 33 GB
Models in /m:
   qwen3-0.6b-q4km                       0.4 GB  general
   qwen3-coder-30b-a3b-q4km             18.6 GB  made for code, completes at the cursor
Editors:
   VSCodium (/usr/bin/codium, extensions from Open VSX)
      no extension Janas knows (0 others)`);
  assert.deepEqual(m, [
    { name: "qwen3-0.6b-q4km", gb: 0.4, coder: false, completes: false },
    { name: "qwen3-coder-30b-a3b-q4km", gb: 18.6, coder: true, completes: true },
  ]);
  assert.deepEqual(
    parsePlan("# Written by janas-setup\nctx=32768\ncode=/m/a.jns\ncode_name=a\ncode_port=11435\n"),
    { codeName: "a", codePort: 11435, completeName: null, completePort: null },
  );
  assert.deepEqual(
    parsePlan("code_name=a\ncode_port=11435\ncomplete=/m/b.jns\ncomplete_name=b\ncomplete_port=11436\n"),
    { codeName: "a", codePort: 11435, completeName: "b", completePort: 11436 },
  );
  assert.equal(parsePlan("# nothing\n"), null);
});
