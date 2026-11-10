// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// plan.ts - what janas-setup editor chose, as it saved it (editor.conf in
// its folder, src/setup/plan.c), and whether its servers answer.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

export interface Plan {
  codeName: string;
  codePort: number;
  completeName: string | null;
  completePort: number | null;
}

export function confDir(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): string {
  if (platform === "win32")
    return path.join(
      env.APPDATA || path.join(os.homedir(), "AppData", "Roaming"),
      "janas",
    );
  return path.join(
    env.XDG_CONFIG_HOME || path.join(os.homedir(), ".config"),
    "janas",
  );
}

/** editor.conf's key=value lines; null when janas-setup has not run. */
export function parsePlan(text: string): Plan | null {
  const kv = new Map<string, string>();
  for (const line of text.split(/\r?\n/)) {
    const i = line.indexOf("=");
    if (i > 0) kv.set(line.slice(0, i).trim(), line.slice(i + 1).trim());
  }
  const codeName = kv.get("code_name"), codePort = Number(kv.get("code_port"));
  if (!codeName || !codePort) return null;
  const completeName = kv.get("complete_name") || null;
  return {
    codeName,
    codePort,
    completeName,
    completePort: completeName ? Number(kv.get("complete_port")) || null : null,
  };
}

export function readPlan(dir: string = confDir()): Plan | null {
  try {
    return parsePlan(fs.readFileSync(path.join(dir, "editor.conf"), "utf8"));
  } catch {
    return null;
  }
}

/** Whether a Janas server answers on port. */
export async function answers(port: number): Promise<boolean> {
  try {
    const r = await fetch(`http://127.0.0.1:${port}/api/version`, {
      signal: AbortSignal.timeout(1500),
    });
    return r.ok;
  } catch {
    return false;
  }
}
