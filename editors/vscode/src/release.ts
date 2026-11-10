// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// release.ts - Janas's releases on GitHub (prabanta-dev/janas): the
// latest one, its package for this system downloaded, checked against the
// SHA-256 published beside it and against the SHA256SUMS inside it, and
// unpacked as a version of its own (versions.ts).

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { binDir, setCurrent, versionDir } from "./versions";

export const REPO = "prabanta-dev/janas";

export interface Release {
  tag: string;
  page: string;
  assets: { name: string; url: string; size: number }[];
}

/** The package of this system, or null where Janas has none. */
export function packageName(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): string | null {
  if (arch !== "x64") return null;
  if (platform === "linux") return "janas-linux-x86_64.tar.gz";
  if (platform === "win32") return "janas-windows-x86_64.zip";
  return null;
}

/** -1, 0 or 1 as tag a is older, the same or newer than b
    (v2026.10.21 < v2026.10.21.1 < v2026.10.29). */
export function compareTags(a: string, b: string): number {
  const pa = a.replace(/^v/, "").split(".").map(Number),
    pb = b.replace(/^v/, "").split(".").map(Number);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i] ?? 0, y = pb[i] ?? 0;
    if (x !== y) return x < y ? -1 : 1;
  }
  return 0;
}

export async function latest(): Promise<Release> {
  const r = await fetch(`https://api.github.com/repos/${REPO}/releases/latest`, {
    headers: { Accept: "application/vnd.github+json" },
    signal: AbortSignal.timeout(20000),
  });
  if (!r.ok) throw new Error(`GitHub answered ${r.status}`);
  const j = (await r.json()) as {
    tag_name: string;
    html_url: string;
    assets: { name: string; browser_download_url: string; size: number }[];
  };
  return {
    tag: j.tag_name,
    page: j.html_url,
    assets: j.assets.map((a) => ({
      name: a.name,
      url: a.browser_download_url,
      size: a.size,
    })),
  };
}

export async function sha256(file: string): Promise<string> {
  const h = createHash("sha256");
  await pipeline(fs.createReadStream(file), h);
  return h.digest("hex");
}

/** The digest a SHA256SUMS-style text gives name, or null. */
export function digestOf(text: string, name: string): string | null {
  for (const line of text.split(/\r?\n/)) {
    const m = /^([0-9a-f]{64})\s+\*?(?:\.\/)?(.+)$/.exec(line.trim());
    if (m && m[2] === name) return m[1] ?? null;
  }
  return null;
}

async function download(
  url: string,
  file: string,
  progress?: (done: number, total: number) => void,
): Promise<void> {
  const r = await fetch(url, { signal: AbortSignal.timeout(30 * 60000) });
  if (!r.ok || !r.body) throw new Error(`${url}: ${r.status}`);
  const total = Number(r.headers.get("content-length") ?? 0);
  let done = 0;
  const body = Readable.fromWeb(r.body as import("node:stream/web").ReadableStream);
  body.on("data", (c: Buffer) => {
    done += c.length;
    progress?.(done, total);
  });
  const part = file + ".part";
  await pipeline(body, fs.createWriteStream(part));
  fs.renameSync(part, file);
}

function run(cmd: string, args: string[]): Promise<void> {
  return new Promise((ok, fail) => {
    const p = spawn(cmd, args, { stdio: "ignore", windowsHide: true });
    p.on("error", fail);
    p.on("exit", (code) =>
      code === 0 ? ok() : fail(new Error(`${cmd} ${args[0]}: ${code}`)),
    );
  });
}

/** Every file of an unpacked package against its SHA256SUMS: the names
    that differ or are missing. */
export async function checkSums(dir: string): Promise<string[]> {
  const sums = fs.readFileSync(path.join(dir, "SHA256SUMS"), "utf8"),
    bad: string[] = [];
  for (const line of sums.split(/\r?\n/)) {
    const m = /^([0-9a-f]{64})\s+\*?(?:\.\/)?(.+)$/.exec(line.trim());
    if (!m || !m[1] || !m[2]) continue;
    const f = path.join(dir, m[2]);
    if (!fs.existsSync(f) || (await sha256(f)) !== m[1]) bad.push(m[2]);
  }
  return bad;
}

/** rel's package downloaded, checked and unpacked under root, then made
    the version in use; the folder of its programs. */
export async function install(
  root: string,
  rel: Release,
  say: (s: string) => void,
  progress?: (done: number, total: number) => void,
): Promise<string> {
  const name = packageName();
  if (!name) throw new Error("Janas has no package for this system yet");
  const pkg = rel.assets.find((a) => a.name === name),
    sum = rel.assets.find((a) => a.name === name + ".sha256");
  if (!pkg || !sum)
    throw new Error(`${rel.tag} has no ${name} with its .sha256`);
  const dl = path.join(root, "downloads");
  fs.mkdirSync(dl, { recursive: true });
  const file = path.join(dl, `${rel.tag}-${name}`);
  say(`downloading ${pkg.url}`);
  await download(pkg.url, file, progress);
  const r = await fetch(sum.url, { signal: AbortSignal.timeout(60000) });
  const want = digestOf(await r.text(), name), got = await sha256(file);
  if (!want || want !== got) {
    fs.rmSync(file, { force: true });
    throw new Error(
      `${name}: SHA-256 ${got}, the release says ${want ?? "nothing"}: not installed`,
    );
  }
  say(`SHA-256 as published: ${got}`);

  const dir = versionDir(root, rel.tag), tmp = dir + ".part";
  fs.rmSync(tmp, { recursive: true, force: true });
  fs.mkdirSync(tmp, { recursive: true });
  // tar unpacks a .zip too on Windows 10 and 11 (bsdtar)
  await run("tar", [name.endsWith(".zip") ? "-xf" : "-xzf", file, "-C", tmp]);
  const bad = await checkSums(path.join(tmp, "janas"));
  if (bad.length) {
    fs.rmSync(tmp, { recursive: true, force: true });
    throw new Error(`files not as in SHA256SUMS: ${bad.join(", ")}`);
  }
  say("every file as in the package's SHA256SUMS");
  fs.rmSync(dir, { recursive: true, force: true });
  fs.renameSync(tmp, dir);
  fs.rmSync(file, { force: true });
  setCurrent(root, rel.tag);
  return binDir(root, rel.tag);
}
