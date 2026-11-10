# Janas for VS Code and VSCodium

Code with a model of your own machine, and nothing leaves it. This extension installs [Janas](https://github.com/prabanta-dev/janas), helps you pick a model for your machine's memory, sets up the extension you code with to use it (Continue, llama.vscode or Cline), and keeps Janas's servers running while the editor is open.

## What it does

- **Janas, installed where it lives on its own.** The package for your system is downloaded from Janas's GitHub releases, checked against the SHA-256 published beside it and against the `SHA256SUMS` inside it, and unpacked in Janas's own folder: `~/.local/share/janas` on Linux, `%LOCALAPPDATA%\janas` on Windows. The extension only points at it, so `janas-chat`, `janas-server` and the rest work from a terminal too. If you installed Janas yourself, give its folder in the setting `janas.path`.
- **A model for your machine.** The models Janas knows are listed with the memory each wants. The models made for code come first, and the one marked with a star is the largest of them your memory holds. Any model can still be chosen; one that wants more memory than the machine has is marked, because it runs reading from the disk, slowly. The download runs in a terminal, so you can watch its progress.
- **Your extension, set up.** If none of the extensions Janas knows is installed, it asks what you want to do and installs one: completion, chat, edits and an agent (Continue); completion alone (llama.vscode); or an agent that works step by step (Cline). Then `janas-setup editor` gives the extension one model for code and one for completion at the cursor, or the same model when memory holds only one, and says why. Janas is added beside your own settings and never replaces them. **Janas: Take Janas away from the extensions' settings** puts every file back as it was.
- **The servers.** `janas-setup serve` starts one server for each model, from port 11435, while the editor is open. The status bar shows whether they are up; click it for Janas's commands.
- **New versions, and going back.** Once a day the extension asks GitHub which version is the latest. If a newer one is out, it says so and installs it only if you ask. The version you had is kept beside the new one: if Janas is slower or answers worse after an update, **Janas: Go back to the previous version** puts it back. Only those two versions are kept.

## Commands

| Command | What it does |
|---|---|
| Janas: Set up Janas in this editor | the whole way: Janas, a model, an extension, the settings, the servers |
| Janas: Download a model | a model of the catalog, chosen for your memory |
| Janas: Choose the models for code and completion | among the models you have, instead of the proposed ones |
| Janas: Start the servers / Stop the servers | |
| Janas: Look for a new version | |
| Janas: Go back to the previous version | |
| Janas: Take Janas away from the extensions' settings | the extensions' files back as they were; Janas and its models stay |
| Janas: Show the log | what Janas's programs wrote |

## Settings

- `janas.path`: the folder of a Janas you installed yourself (where `janas-setup` is). This extension does not update it.
- `janas.startServers` (on): start the servers when the editor starts.
- `janas.checkForUpdates` (on): ask GitHub once a day for the latest version.

## What leaves the machine

Only what you ask for: Janas's package from github.com/prabanta-dev/janas, a model from huggingface.co/prabanta-dev, and once a day a request to GitHub's API for the latest release (`janas.checkForUpdates` turns it off). Your code goes only to the servers on your own machine.

## What has been run

7 November 2026, on Linux, VSCodium 1.135 with Continue 2.0.0 and Qwen3-0.6B. The extension set Janas up from its command (`janas-setup editor` wrote Continue's file and the plan), started the server, stopped and restarted it, and took Janas away again, every file removed. The installation was run on its own against the release v2026.11.07: the package downloaded, its SHA-256 as published, every file as in its `SHA256SUMS`, unpacked as a version of its own. That release does not have `janas-setup` yet, so the extension needs the next one. It has not been run on Windows yet.

## Licence

GPL-3.0-or-later, as Janas.
