// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
//
// extension.ts - Janas in VS Code and VSCodium: Janas installed where it
// lives on its own (versions.ts), a model downloaded with janas-get as
// this machine's memory advises (models.ts), the editor's extensions set
// up by janas-setup editor, its servers started with janas-setup serve
// and kept while the editor is open; a new version said once a day,
// installed only when asked, the one before kept to go back to.

import { spawn, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as vscode from "vscode";
import { advise, parseCatalog, parseInstalled } from "./models";
import { answers, readPlan } from "./plan";
import { compareTags, install, latest, packageName } from "./release";
import { binDir, exe, inUse, janasRoot, previous, prune, rollback } from "./versions";

const KNOWN = [
  {
    id: "continue.continue",
    label: "Continue",
    does: "completion as you type, chat, edits and an agent - one extension for all, recommended",
  },
  {
    id: "ggml-org.llama-vscode",
    label: "llama.vscode",
    does: "completion as you type, alone",
  },
  {
    id: "saoudrizwan.claude-dev",
    label: "Cline",
    does: "an agent that works step by step, asking before each change",
  },
];

let log: vscode.OutputChannel;
let status: vscode.StatusBarItem;
let servers: ChildProcess | null = null;
let ctx: vscode.ExtensionContext;

/** The folder of Janas's programs: the user's own, or the version in use. */
function janasDir(): string | null {
  const own = vscode.workspace.getConfiguration("janas").get<string>("path");
  if (own) return fs.existsSync(exe(own, "janas-setup")) ? own : null;
  const tag = inUse(janasRoot());
  const dir = tag ? binDir(janasRoot(), tag) : null;
  return dir && fs.existsSync(exe(dir, "janas-setup")) ? dir : null;
}

/** A program of Janas run to its end, what it writes in the log: its exit
    code and its output. */
function run(dir: string, prog: string, args: string[]): Promise<{ code: number; out: string }> {
  return new Promise((ok) => {
    log.appendLine(`$ ${prog} ${args.join(" ")}`);
    const p = spawn(exe(dir, prog), args, { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    const take = (b: Buffer) => {
      out += b.toString();
      log.append(b.toString());
    };
    p.stdout.on("data", take);
    p.stderr.on("data", take);
    p.on("error", (e) => {
      log.appendLine(String(e));
      ok({ code: -1, out });
    });
    p.on("exit", (code) => ok({ code: code ?? -1, out }));
  });
}

function editorName(): string {
  return /codium/i.test(vscode.env.appName) ? "vscodium" : "vscode";
}

/** This editor's folders, for janas-setup editor: the user data three
    levels over globalStorage, the extensions' folder the one an extension
    Janas knows is in, or this one's - not while this one is run from its
    sources, which are elsewhere (janas-setup would install there). */
function editorArgs(): string[] {
  const udd = path.resolve(ctx.globalStorageUri.fsPath, "..", "..", "..");
  const other = KNOWN.map((k) => vscode.extensions.getExtension(k.id)).find((x) => x);
  const xd = other
    ? path.dirname(other.extensionPath)
    : ctx.extensionMode === vscode.ExtensionMode.Production
      ? path.dirname(ctx.extensionPath)
      : null;
  return ["--editor", editorName(), "--user-data-dir", udd, ...(xd ? ["--extensions-dir", xd] : [])];
}

async function installJanas(): Promise<string | null> {
  if (!packageName()) {
    vscode.window.showErrorMessage(
      "Janas has no package for this system yet (Linux and Windows on x86-64). It can be built from its sources: github.com/prabanta-dev/janas.",
    );
    return null;
  }
  let rel;
  try {
    rel = await latest();
  } catch (e) {
    vscode.window.showErrorMessage(`Janas: cannot reach GitHub (${e})`);
    return null;
  }
  const pkg = rel.assets.find((a) => a.name === packageName());
  const yes = await vscode.window.showInformationMessage(
    `Janas is not installed. Install ${rel.tag} (${((pkg?.size ?? 0) / 1e6).toFixed(0)} MB from github.com/${"prabanta-dev/janas"}) in ${janasRoot()}? It works outside the editor too.`,
    "Install",
    "Not now",
  );
  if (yes !== "Install") return null;
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: `Janas ${rel.tag}`, cancellable: false },
    async (p) => {
      try {
        let last = 0;
        const dir = await install(janasRoot(), rel, (s) => log.appendLine(s), (done, total) => {
          const pc = total ? Math.floor((100 * done) / total) : 0;
          if (pc > last) {
            p.report({ increment: pc - last, message: `${(done / 1e6).toFixed(0)} MB` });
            last = pc;
          }
        });
        vscode.window.showInformationMessage(`Janas ${rel.tag} installed in ${dir}.`);
        return dir;
      } catch (e) {
        log.appendLine(String(e));
        vscode.window.showErrorMessage(`Janas was not installed: ${e}`);
        return null;
      }
    },
  );
}

/** A model of the catalog downloaded in a terminal, which shows its
    progress; whether it ended well. */
async function getModel(dir: string): Promise<boolean> {
  const list = await run(dir, "janas-get", ["list"]);
  const mem = Math.round(os.totalmem() / 1e9);
  const items = advise(parseCatalog(list.out), mem).map((a) => ({
    label: (a.recommended ? "$(star-full) " : a.fits ? "" : "$(warning) ") + a.model.name,
    description: `${a.model.downloadGb} GB`,
    detail: `${a.model.what} - ${a.note}`,
    name: a.model.name,
  }));
  const pick = await vscode.window.showQuickPick(items, {
    title: `A model for this machine (${mem} GB of memory)`,
    placeHolder: "The one marked with a star is advised; any can be chosen",
    matchOnDetail: true,
  });
  if (!pick) return false;
  const term = vscode.window.createTerminal({
    name: `Janas: ${pick.name}`,
    shellPath: exe(dir, "janas-get"),
    shellArgs: [pick.name],
  });
  term.show();
  return new Promise((ok) => {
    const sub = vscode.window.onDidCloseTerminal((t) => {
      if (t !== term) return;
      sub.dispose();
      ok(t.exitStatus?.code === 0);
    });
  });
}

/** One of the extensions Janas sets up, installed when none is. */
async function ensureExtension(): Promise<boolean> {
  if (KNOWN.some((k) => vscode.extensions.getExtension(k.id))) return true;
  const pick = await vscode.window.showQuickPick(
    KNOWN.map((k) => ({ label: k.label, detail: k.does, id: k.id })),
    { title: "What would you like Janas to do in this editor?" },
  );
  if (!pick) return false;
  await vscode.commands.executeCommand("workbench.extensions.installExtension", pick.id);
  return true;
}

async function setUp(models?: { code: string; complete: string | null }): Promise<void> {
  const dir = janasDir() ?? (await installJanas());
  if (!dir) return;
  let installed = parseInstalled((await run(dir, "janas-setup", ["editor", "--list"])).out);
  if (!installed.length) {
    if (!(await getModel(dir))) return;
    installed = parseInstalled((await run(dir, "janas-setup", ["editor", "--list"])).out);
    if (!installed.length) return;
  }
  if (!(await ensureExtension())) return;
  const args = ["editor", "--yes", ...editorArgs()];
  if (models) args.push("--code", models.code, "--complete", models.complete ?? "none");
  const r = await run(dir, "janas-setup", args);
  if (r.code !== 0) {
    log.show();
    vscode.window.showErrorMessage("janas-setup could not set up the editor: the log says why.");
    return;
  }
  // what janas-setup said of the models, to the user
  const why = r.out.split(/\r?\n/).filter((l) => l.startsWith("For ")).join(" ");
  vscode.window.showInformationMessage(`Janas is set up. ${why}`, "Show the log").then((a) => {
    if (a) log.show();
  });
  await stop();
  await start();
}

async function chooseModels(): Promise<void> {
  const dir = janasDir();
  if (!dir) return setUp();
  const installed = parseInstalled((await run(dir, "janas-setup", ["editor", "--list"])).out);
  const mem = os.totalmem() / 1e9;
  const label = (m: { name: string; gb: number; coder: boolean; completes: boolean }) => ({
    label: m.name,
    description: `${m.gb} GB`,
    detail:
      (m.coder ? "made for code" : "general") +
      (m.completes ? ", completes at the cursor" : "") +
      (m.gb > mem - 6 ? ` - more than this machine's ${mem.toFixed(0)} GB hold beside the system: slower` : ""),
    name: m.name,
  });
  const code = await vscode.window.showQuickPick(installed.map(label), {
    title: "The model for code: chat, edits, the agent",
  });
  if (!code) return;
  const none = { label: "None", description: "", detail: "no completion as you type", name: "" };
  const comp = await vscode.window.showQuickPick(
    [...installed.filter((m) => m.completes).map(label), none],
    { title: "The model for completion as you type (the same one keeps one model in memory)" },
  );
  if (!comp) return;
  if (comp.name && comp.name !== code.name) {
    const a = installed.find((m) => m.name === code.name)?.gb ?? 0,
      b = installed.find((m) => m.name === comp.name)?.gb ?? 0;
    if (a + b > mem - 6) {
      const go = await vscode.window.showWarningMessage(
        `The two models together are ${(a + b).toFixed(1)} GB, more than this machine's ${mem.toFixed(0)} GB of memory hold beside the system: Janas will read from the disk what does not fit, and answer more slowly.`,
        "Use them anyway",
        "Cancel",
      );
      if (go !== "Use them anyway") return;
    }
  }
  await setUp({ code: code.name, complete: comp.name || null });
}

function show(text: string, tip: string): void {
  status.text = text;
  status.tooltip = tip;
  status.show();
}

async function start(): Promise<void> {
  const dir = janasDir(), plan = readPlan();
  if (!dir || !plan) {
    show("$(circle-slash) Janas", "Janas is not set up: click to set it up");
    status.command = "janas.setUp";
    return;
  }
  status.command = "janas.menu";
  const names = plan.completeName && plan.completeName !== plan.codeName
    ? `${plan.codeName}, ${plan.completeName}`
    : plan.codeName;
  if (await answers(plan.codePort)) {
    show("$(check) Janas", `${names} on port ${plan.codePort} (started elsewhere)`);
    return;
  }
  show("$(sync~spin) Janas", `starting ${names}`);
  log.appendLine(`$ janas-setup serve`);
  const p = spawn(exe(dir, "janas-setup"), ["serve"], { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
  servers = p;
  p.stdout?.on("data", (b: Buffer) => log.append(b.toString()));
  p.stderr?.on("data", (b: Buffer) => log.append(b.toString()));
  p.on("exit", (code) => {
    if (servers === p) servers = null;
    log.appendLine(`janas-setup serve ended (${code})`);
    show("$(circle-slash) Janas", "the servers are stopped: click for Janas's commands");
  });
  // a model takes a few seconds to open, a large one a minute
  for (let i = 0; i < 120 && servers === p; i++) {
    if (await answers(plan.codePort)) {
      show("$(check) Janas", `${names} on port ${plan.codePort}`);
      return;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function stop(): Promise<void> {
  const p = servers;
  if (!p) return;
  servers = null;
  await new Promise<void>((ok) => {
    const t = setTimeout(() => {
      p.kill("SIGKILL");
      ok();
    }, 10000);
    p.on("exit", () => {
      clearTimeout(t);
      ok();
    });
    p.kill("SIGINT");
  });
}

async function update(asked: boolean): Promise<void> {
  if (vscode.workspace.getConfiguration("janas").get<string>("path")) {
    if (asked)
      vscode.window.showInformationMessage("Janas is the one you installed yourself (janas.path): this extension does not update it.");
    return;
  }
  const root = janasRoot(), now = inUse(root);
  let rel;
  try {
    rel = await latest();
  } catch (e) {
    if (asked) vscode.window.showErrorMessage(`Janas: cannot reach GitHub (${e})`);
    return;
  }
  await ctx.globalState.update("janas.lastCheck", Date.now());
  if (!now) return;
  if (compareTags(rel.tag, now) <= 0) {
    if (asked) vscode.window.showInformationMessage(`Janas ${now} is the latest.`);
    return;
  }
  const a = await vscode.window.showInformationMessage(
    `Janas ${rel.tag} is out (you have ${now}).`,
    "Update",
    "What changed",
    "Not now",
  );
  if (a === "What changed") {
    vscode.env.openExternal(vscode.Uri.parse(rel.page));
    return;
  }
  if (a !== "Update") return;
  await stop();
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: `Janas ${rel.tag}` },
    async () => {
      try {
        await install(root, rel, (s) => log.appendLine(s));
        prune(root);
        vscode.window
          .showInformationMessage(
            `Janas ${rel.tag} installed; ${now} is kept. If Janas is slower or answers worse than before, go back to it.`,
            `Go back to ${now}`,
          )
          .then((b) => b && back());
      } catch (e) {
        vscode.window.showErrorMessage(`Janas was not updated, ${now} stays: ${e}`);
      }
    },
  );
  await start();
}

async function back(): Promise<void> {
  const root = janasRoot(), now = inUse(root), was = previous(root);
  if (!was) {
    vscode.window.showInformationMessage("There is no version of Janas to go back to.");
    return;
  }
  await stop();
  const tag = rollback(root);
  await start();
  if (tag)
    vscode.window.showInformationMessage(`Janas ${tag} is back in use; ${now} is kept, to return to it the same way.`);
  else vscode.window.showErrorMessage(`Janas ${was} is no longer installed.`);
}

async function remove(): Promise<void> {
  const dir = janasDir();
  if (!dir) return;
  await stop();
  await run(dir, "janas-setup", ["editor", "--undo"]);
  show("$(circle-slash) Janas", "Janas is not set up: click to set it up");
  status.command = "janas.setUp";
  vscode.window.showInformationMessage("Janas is out of the extensions' settings, every file as it was. Janas itself and its models stay.");
}

async function menu(): Promise<void> {
  const items: { label: string; command: string }[] = [
    { label: servers ? "$(debug-stop) Stop the servers" : "$(play) Start the servers", command: servers ? "janas.stop" : "janas.start" },
    { label: "$(settings) Choose the models", command: "janas.chooseModels" },
    { label: "$(cloud-download) Download a model", command: "janas.getModel" },
    { label: "$(sync) Look for a new version", command: "janas.update" },
    { label: "$(history) Go back to the previous version", command: "janas.rollback" },
    { label: "$(output) Show the log", command: "janas.showLog" },
  ];
  const pick = await vscode.window.showQuickPick(items, { title: "Janas" });
  if (pick) vscode.commands.executeCommand(pick.command);
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  ctx = context;
  log = vscode.window.createOutputChannel("Janas");
  status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  const cmd = (id: string, f: () => unknown) =>
    context.subscriptions.push(vscode.commands.registerCommand(id, f));
  cmd("janas.setUp", () => setUp());
  cmd("janas.getModel", async () => {
    const dir = janasDir() ?? (await installJanas());
    if (dir && (await getModel(dir)))
      vscode.window.showInformationMessage("The model is here: Janas: Choose the models gives it to the editor.");
  });
  cmd("janas.chooseModels", chooseModels);
  cmd("janas.start", start);
  cmd("janas.stop", async () => {
    await stop();
    show("$(circle-slash) Janas", "the servers are stopped: click for Janas's commands");
  });
  cmd("janas.update", () => update(true));
  cmd("janas.rollback", back);
  cmd("janas.remove", remove);
  cmd("janas.showLog", () => log.show());
  cmd("janas.menu", menu);
  context.subscriptions.push(log, status, { dispose: () => void stop() });

  const conf = vscode.workspace.getConfiguration("janas");
  if (!janasDir() || !readPlan()) {
    show("$(circle-slash) Janas", "Janas is not set up: click to set it up");
    status.command = "janas.setUp";
    if (!context.globalState.get("janas.offered")) {
      await context.globalState.update("janas.offered", true);
      vscode.window
        .showInformationMessage("Code with a model of this machine, nothing leaving it? Janas can set it up.", "Set up Janas")
        .then((a) => a && setUp());
    }
  } else if (conf.get<boolean>("startServers")) await start();
  const last = context.globalState.get<number>("janas.lastCheck") ?? 0;
  if (conf.get<boolean>("checkForUpdates") && Date.now() - last > 24 * 3600e3) void update(false);
}

export async function deactivate(): Promise<void> {
  await stop();
}
