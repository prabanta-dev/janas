# Janas-LLM in the editor and with coding agents

`janas-server` answers the APIs the editors' extensions and the coding agents speak - OpenAI's, Ollama's and Anthropic's ([serving.md](serving.md)) - so that they work with a model of this machine, and nothing leaves it. This page says how each one tried was set up, and what it did. For the completion of code at the cursor alone, see [code completion](code-completion.md).

The task given to each was the same, from [issue #52](https://github.com/prabanta-dev/janas/issues/52): a folder holds a small Gambas3 project (`ref/Notes`: `.project`, `.src/FMain.class`, `.src/FMain.form`, a list of notes saved in `~/.notes.txt`); read it, and write in `new/Todo` a to-do list in the same structure and style, with a Done button that removes the selected item, saved in `~/.todo.txt`. The model was Qwen3-Coder-30B-A3B-Instruct (Q4_K_M), on an Intel Core Ultra 9 185H with 30 GB of memory, 2-3 November 2026.

## Setting it up in one command: janas-setup

`janas-setup editor` does what the rest of this page does by hand. It finds VS Code and VSCodium (in PATH, where their installers put them on Windows, or as a Flatpak), the extensions they have, and the models installed (`janas-get`'s folder); it proposes a model for code (chat, edits, the agent) and one for completion at the cursor, from the machine's memory and what each model can do - a model made for code that fills in the middle and has no recurrent layers completes, and when memory holds only one model the same one does both - and says why; then it sets up the extensions it finds:

- **Continue**: a file of Janas's own, `~/.continue/models/janas.yaml`, which Continue reads beside the user's `config.yaml` (not when it still reads the old `config.json`: janas-setup says so).
- **llama.vscode**: its endpoints in the editor's `settings.json`.
- **Cline**: its Ollama provider in `~/.cline/data/settings/providers.json`, where its version keeps the providers there and none of the user's is set; otherwise the values to fill in.
- **VS Code's own chat** (VS Code, not VSCodium): a provider of the Custom Endpoint kind in `chatLanguageModels.json`.
- **Roo Code**: a file to import from its settings (an import at every start would make Janas its profile in use every time); Roo Code's repository was archived in May 2026, so it is set up where it is installed, never offered.

Where no extension it knows is installed it asks what to do - completion, chat and an agent (Continue), completion alone (llama.vscode), an agent step by step (Cline) - and installs it from the editor's own gallery, Open VSX for VSCodium. Janas is added beside what is there: a value it replaces is said and written down, and `janas-setup editor --undo` puts every file back as it was, byte for byte, comments and commas included. `--list` only says what is there; `--code` and `--complete` name the models instead of the ones proposed. `janas-setup serve` then starts one `janas-server` for each model, from port 11435 (Ollama's 11434 is left to it), until Ctrl+C; with `--monitor` each serves its page of the engine at work too ([docs/monitor.md](monitor.md)), as the extension does.

```sh
janas-setup editor --list
janas-setup editor
janas-setup serve
```

What has been run: on Linux and under Wine, with VSCodium, Continue, llama.vscode and Cline installed from Open VSX into folders of their own, the settings written, written again, and put back to the same bytes; `serve` with Qwen3-0.6B answering Ollama's `/api/tags` and `/api/chat`, and stopping its server when stopped. VS Code's chat and Roo Code were tried with a script standing for VS Code. In VSCodium 1.135 with Continue 2.0.0, its `config.yaml` naming no model, Continue took Qwen3-0.6B from Janas's file and sent its request to Janas's server (7 November 2026). Not yet run: a model for code behind it.

`--editor vscode` or `--editor vscodium` sets up that editor only, where both are installed. The extension for VS Code and VSCodium in [`editors/vscode/`](../editors/vscode/README.md) does all this from the editor: it installs Janas, helps choose a model for the machine's memory, runs `janas-setup`, keeps the servers running while the editor is open, says when a new version is out and goes back to the one before if asked.

## The server

```sh
janas-server models/jns/qwen3-coder-30b-a3b-q4km.jns --port 11434 --name qwen3-coder --ctx 32768
```

Port 11434 is Ollama's: the clients that know a local model only as Ollama's find it there by themselves. `--name` is the name the clients see and ask for. The agents' prompts are long - 12,679 tokens for Claude Code, 13,247 for Roo Code, most of them their tools - so the context must hold them and the work: 32768 tokens did, 65536 for Claude Code.

**The first request is the slow one.** The prompt of an agent is read once, then kept computed: on this laptop Claude Code's 12,679 tokens took 266 s the first time, and the requests after it read only what they add (tens to hundreds of tokens, a few seconds). Whatever changes the start of the prompt has it read again: Roo Code's switch from one mode to another changes its system prompt, and cost the same four and a half minutes.

## Continue (VSCodium, VS Code)

[Continue](https://www.continue.dev) 2.0.0, from Open VSX, with its `ollama` provider, in `~/.continue/config.yaml`:

```yaml
name: Janas
version: 1.0.0
schema: v1
models:
  - name: Janas
    provider: ollama
    model: qwen3-coder
    apiBase: http://127.0.0.1:11434
    roles: [chat, edit, apply, autocomplete]
    capabilities: [tool_use]
    autocompleteOptions:
      modelTimeout: 20000
```

Continue gives a completion at the cursor 150 ms (`modelTimeout`), then keeps what has come and closes the connection: with a model that takes seconds over a prompt, raise it, or the proposal is cut after a few tokens. Continue writes its own prompt for filling in, `<|fim_prefix|>` and the other markers included, and sends it raw: the server reads them as the model's tokens.

What it did: "write a docstring" on a C function, shown as a diff and accepted; a completion at the cursor in an empty function body, the loop, the count and the return. Its command-line agent (`cn` 1.5.47, the same provider and file) did the Gambas3 task in nine requests: the three files read, the three written, in the style of the others.

## Cline (VSCodium, VS Code)

[Cline](https://cline.bot) 4.1.26, with its Ollama provider: base URL `http://127.0.0.1:11434`, model `qwen3-coder`, context 32768. It speaks Ollama's `/api/chat`, streamed.

What it did: the Gambas3 task in six requests, the three files written in the style of the others.

## Roo Code (VSCodium, VS Code)

[Roo Code](https://roocode.com) 3.54.0, with its Ollama provider: base URL `http://127.0.0.1:11434`, model `qwen3-coder`. It speaks `/api/chat`.

In VSCodium 1.135 it stops before its first request with "Could not find ripgrep binary": it looks for `rg` under `node_modules.asar.unpacked/@vscode/ripgrep/bin/`, and this VSCodium has it under `@vscode/ripgrep-universal/bin/linux-x64/`. A link, made as root, lets it start (an update of VSCodium may remove it):

```sh
cd /usr/share/codium/resources/app/node_modules.asar.unpacked/@vscode
sudo mkdir -p ripgrep/bin
sudo ln -s ../../ripgrep-universal/bin/linux-x64/rg ripgrep/bin/rg
```

Start it in its Code mode for a task that writes: it began in Architect mode, asked to switch, and the switch had its 13,247-token prompt read again.

What it did: the Gambas3 task in twenty requests, the switch of mode approved, the three files written in the style of the others.

## Claude Code

[Claude Code](https://code.claude.com) 2.1.311, through Anthropic's Messages API:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:11434 ANTHROPIC_AUTH_TOKEN=janas \
ANTHROPIC_MODEL=qwen3-coder ANTHROPIC_SMALL_FAST_MODEL=qwen3-coder \
claude --model qwen3-coder
```

It warns that it does not know the model and assumes a window of 200,000 tokens: the server's is `--ctx`.

What it did (`claude -p`, the tools to read, search and write files allowed): the Gambas3 task in 25 requests, the three files written. Its class file is less like the others: `File.Exists("~/.todo.txt")`, the tilde as text, where Notes builds the path with `User.Home &/ ".todo.txt"`. Once, of four tries, the model wrote a tool call without the token that opens it, and Claude Code got it as text: Claude Code asks for no temperature, and Anthropic's default is 1.

## aider

[aider](https://aider.chat) 0.86.2, through Ollama's API:

```sh
OLLAMA_API_BASE=http://127.0.0.1:11434 aider --model ollama_chat/qwen3-coder
```

What it did (its "whole" edit format, the three files of Notes given as read-only and the three of Todo as files to fill): the three files written with the right content, but two of them under a wrong path - the model wrote `.new/Todo/.src/FMain.class`, a dot before the folder's name, and aider made that path.

## Open WebUI

[Open WebUI](https://openwebui.com) with `OLLAMA_BASE_URL=http://127.0.0.1:11434`: the model shows among Ollama's, and a chat goes through `/api/chat`, whole and streamed; the speeds it shows come from the times the server gives in Ollama's fields.

## VS Code's own chat

VS Code finds local models with its Ollama provider: `/api/version`, `/api/tags`, `/api/show` (whose capabilities say tools, and vision where the model has its encoder), then `/v1/chat/completions`. That sequence was replayed with `curl` and the `openai` SDK, a streamed chat with a tool included; VS Code itself, which needs the Copilot extension for it, has not been tried yet.

## Your files outside the editor

In the editor the agents read and write the files of the folder you open. Outside it, `janas-chat` starts [`janas-files`](../src/services/files/README.md) by itself: the model lists, reads and searches your home and the folders you list, and writes, edits or makes a folder only when you say yes to each call; the private files - keys, passwords, the browsers' data - are never reached. Any MCP client can use it too. In `janas-chat` it read the Gambas3 project, explained it, and wrote the Todo project, eleven calls confirmed. Linux only for now.
