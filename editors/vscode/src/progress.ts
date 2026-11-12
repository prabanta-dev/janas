// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// progress.ts - what janas-get writes while it downloads, read: its latest
// "0.27 of 0.40 GB (67%), 25 MB/s", and the lines worth a log.

export interface Step {
  done: number;
  size: number;
  pct: number;
  mbs: number;
}

/** The last progress step in a piece of janas-get's output, or null. */
export function lastStep(text: string): Step | null {
  const all = [...text.matchAll(/([\d.]+) of ([\d.]+) GB \((\d+)%\), (\d+) MB\/s/g)];
  const m = all[all.length - 1];
  return m ? { done: Number(m[1]), size: Number(m[2]), pct: Number(m[3]), mbs: Number(m[4]) } : null;
}

/** Its lines, without the steps of the progress (one a fraction of a
    second, which would fill the log). */
export function logLines(text: string): string[] {
  return text.split(/\r|\n/).filter((l) => l.trim() && !/ GB \(\d+%\), /.test(l));
}
