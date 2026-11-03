# Janas-LLM in the editor and with coding agents

`janas-server` answers the APIs the editors' extensions and the coding agents speak - OpenAI's, Ollama's and Anthropic's ([serving.md](serving.md)) - so that they work with a model of this machine, and nothing leaves it. This page says how each one tried was set up, and what it did. For the completion of code at the cursor alone, see [code completion](code-completion.md).

The task given to each was the same, from [issue #52](https://github.com/prabanta-dev/janas/issues/52): a folder holds a small Gambas3 project (`ref/Notes`: `.project`, `.src/FMain.class`, `.src/FMain.form`, a list of notes saved in `~/.notes.txt`); read it, and write in `new/Todo` a to-do list in the same structure and style, with a Done button that removes the selected item, saved in `~/.todo.txt`. The model was Qwen3-Coder-30B-A3B-Instruct (Q4_K_M), on an Intel Core Ultra 9 185H with 30 GB of memory, 2-3 November 2026.

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
