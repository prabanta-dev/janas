// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// versions.ts - the versions of Janas this extension installed, side by
// side: <root>/versions/<tag>/janas/ holds each, <root>/current names the
// one in use and <root>/previous the one before it, so that going back is
// changing a name. <root> is Janas's own folder, the one its models are in
// (janas_models_dir: ~/.local/share/janas, %LOCALAPPDATA%\janas): Janas
// works on its own too, the extension only points at it.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

export function janasRoot(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): string {
  if (platform === "win32")
    return path.join(
      env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"),
      "janas",
    );
  return path.join(
    env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"),
    "janas",
  );
}

export function modelsDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.JANAS_MODELS || path.join(janasRoot(env), "models");
}

/** A program of Janas in dir, with .exe on Windows. */
export function exe(
  dir: string,
  name: string,
  platform: NodeJS.Platform = process.platform,
): string {
  return path.join(dir, platform === "win32" ? name + ".exe" : name);
}

export function versionDir(root: string, tag: string): string {
  return path.join(root, "versions", tag);
}

/** Where the programs of a version are: the package's own folder. */
export function binDir(root: string, tag: string): string {
  return path.join(versionDir(root, tag), "janas");
}

function readName(file: string): string | null {
  try {
    const t = fs.readFileSync(file, "utf8").trim();
    return t || null;
  } catch {
    return null;
  }
}

function writeName(file: string, tag: string | null): void {
  if (tag === null) {
    fs.rmSync(file, { force: true });
    return;
  }
  const tmp = file + ".tmp";
  fs.writeFileSync(tmp, tag + "\n");
  fs.renameSync(tmp, file);
}

export function inUse(root: string): string | null {
  return readName(path.join(root, "current"));
}

export function previous(root: string): string | null {
  return readName(path.join(root, "previous"));
}

/** The version in use becomes tag; the one before is kept as previous. */
export function setCurrent(root: string, tag: string): void {
  fs.mkdirSync(root, { recursive: true });
  const was = inUse(root);
  if (was && was !== tag) writeName(path.join(root, "previous"), was);
  writeName(path.join(root, "current"), tag);
}

/** The previous version back in use, the current one kept as previous:
    its tag, or null when there is none to go back to. */
export function rollback(root: string): string | null {
  const back = previous(root), now = inUse(root);
  if (!back || !fs.existsSync(binDir(root, back))) return null;
  writeName(path.join(root, "current"), back);
  writeName(path.join(root, "previous"), now);
  return back;
}

/** The versions neither in use nor previous removed, and what is left of
    a download stopped halfway; the names removed. */
export function prune(root: string): string[] {
  const dir = path.join(root, "versions"),
    keep = new Set([inUse(root), previous(root)]);
  let names: string[];
  try {
    names = fs.readdirSync(dir);
  } catch {
    return [];
  }
  const gone: string[] = [];
  for (const n of names)
    if (!keep.has(n)) {
      fs.rmSync(path.join(dir, n), { recursive: true, force: true });
      gone.push(n);
    }
  return gone;
}
