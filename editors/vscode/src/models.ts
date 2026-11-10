// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// models.ts - what janas-get and janas-setup say about the models, read
// from their output, and the advice for this machine's memory: the user
// is guided, never held to it.

export interface CatalogModel {
  name: string;
  downloadGb: number;
  memoryGb: number;
  what: string;
}

/** janas-get list: "name  N.N GB  M GB  what". */
export function parseCatalog(text: string): CatalogModel[] {
  const out: CatalogModel[] = [];
  for (const line of text.split(/\r?\n/)) {
    const m = /^(\S+)\s+([\d.]+) GB\s+(\d+) GB\s+(.*)$/.exec(line);
    if (m && m[1] && m[2] && m[3])
      out.push({
        name: m[1],
        downloadGb: Number(m[2]),
        memoryGb: Number(m[3]),
        what: (m[4] ?? "").trim(),
      });
  }
  return out;
}

export interface InstalledModel {
  name: string;
  gb: number;
  coder: boolean;
  completes: boolean;
}

/** The models of janas-setup editor --list, between "Models in" and
    "Editors:". */
export function parseInstalled(text: string): InstalledModel[] {
  const out: InstalledModel[] = [];
  let inside = false;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("Models in")) {
      inside = true;
      continue;
    }
    if (!line.startsWith(" ")) inside = false;
    const m = inside ? /^\s+(\S+)\s+([\d.]+) GB\s+(.*)$/.exec(line) : null;
    if (m && m[1] && m[2])
      out.push({
        name: m[1],
        gb: Number(m[2]),
        coder: /made for code/.test(m[3] ?? ""),
        completes: /completes/.test(m[3] ?? ""),
      });
  }
  return out;
}

export interface Advice {
  model: CatalogModel;
  recommended: boolean;
  fits: boolean;
  note: string;
}

const forCode = (m: CatalogModel) => /for code/i.test(m.what);
const notChat = (m: CatalogModel) =>
  /not for chatting|embedding-model/i.test(m.what);

/** The catalog for a machine of memGb: the models for code first, the one
    advised marked, what does not fit said. */
export function advise(models: CatalogModel[], memGb: number): Advice[] {
  const usable = models.filter((m) => !notChat(m));
  const fit = (m: CatalogModel) => m.memoryGb <= memGb;
  const pick =
    usable.filter((m) => forCode(m) && fit(m)).sort((a, b) => b.downloadGb - a.downloadGb)[0] ??
    usable.filter(fit).sort((a, b) => b.downloadGb - a.downloadGb)[0];
  return usable
    .map((m) => ({
      model: m,
      recommended: m === pick,
      fits: fit(m),
      note:
        m === pick
          ? forCode(m)
            ? `made for code, and this machine's ${memGb} GB of memory are enough`
            : `no model for code fits in ${memGb} GB of memory: this is the largest general one that does`
          : fit(m)
            ? `wants ${m.memoryGb} GB of memory`
            : `wants ${m.memoryGb} GB of memory, this machine has ${memGb}: it would run, reading from the disk, slowly`,
    }))
    .sort(
      (a, b) =>
        Number(b.recommended) - Number(a.recommended) ||
        Number(forCode(b.model)) - Number(forCode(a.model)) ||
        Number(b.fits) - Number(a.fits) ||
        b.model.downloadGb - a.model.downloadGb,
    );
}
