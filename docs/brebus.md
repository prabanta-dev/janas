# Brebus

Brebus is Janas's app: the models of this machine, a chat with them and the server for other programs, in a window of its own. *Brebus*, in Sardinian, are the magic words of the *janas*. It is at its beginning, Linux only, and on the `develop` branch.

```sh
bin/x86_64-linux/brebus                    # its window
bin/x86_64-linux/brebus --install-desktop  # in the desktop's menu, with its icon
bin/x86_64-linux/brebus --browser          # in the browser instead
bin/x86_64-linux/brebus --no-open          # neither: http://127.0.0.1:41414/
bin/x86_64-linux/brebus --port 41500
```

The window is WebKitGTK's (`libwebkit2gtk-4.1`, on Debian and Ubuntu the package `libwebkit2gtk-4.1-0`), loaded when Brebus starts: nothing is needed to build it, and where it is missing, or there is no display, Brebus opens its page in the browser. Links to other sites open in the browser. Brebus ends when its window is closed (or with Ctrl+C), and the model's server ends with it.

Its icon is the Prabanta logo, drawn for each size and system ([src/brebus/icons](../src/brebus/icons/README.md)). When the window opens, Brebus writes its icons in `~/.local/share/icons/hicolor` and, where there is none, `brebus.desktop` in `~/.local/share/applications`: the desktop's menu has Brebus, and the task bar shows its logo (on Wayland it comes from there only). `--install-desktop` writes them without opening it, and the `JANAS_MODELS` given then is kept for the menu's start.

## The workspace

Brebus works in a folder, as Claude Code does: started from a terminal, the folder it is started in, or the one given with `--workspace DIR`; started from the desktop's menu, the page asks for one: a recent one, a path, or a new one, `Documents/Brebus/<date>` (then `<date> (2)`…). The home itself and `/` are refused, too wide for a model's files. The button under the logo shows it and changes it.

| In the workspace | |
|---|---|
| `.brebus/chats/` | its conversations, a JSON file each |
| `exports/` | the conversations exported in Markdown |
| `sources/` | the files attached to the chat by copy, and the web pages downloaded, each with `<name>.url.txt` (its address and when) |
| `.brebus/workspace.json` | the folders and files added to be read only |
| `.brebus/data/`, `.brebus/config/` | the index of its documents, and its list |

**Read only, beside it**: folders and files of yours (documents, code, pictures, audio; video is not read by Janas yet) that the model may search and read but never write, added and removed in the workspace's window.

## Documents and tools in the chat

Brebus starts, for the workspace, [`janas-docs`](../src/services/docs/README.md), which indexes the workspace and the folders added to it (PDF, Word, Excel, PowerPoint, LibreOffice's, RTF, web pages, EPUB, text and Markdown), and [`janas-files`](../src/services/files/README.md), which reaches the workspace to read and write and the folders added only to read, not the rest of the home.

- **Search documents** (on by default): before each question, Brebus searches the workspace's documents for it and gives the model the passages found, each with its document, as LM Studio's documents do; it works with models that cannot call tools. The passages are shown, folded, under the question.
- **Attach** (the `+` in the field): a file of this computer by its path, or a web address. A file is copied into `sources/` or read where it is (then added to the folders read only); a copy is offered up to 100 MB. A web address is always downloaded into `sources/`. A document of up to 24,000 characters goes into the message whole; a longer one is named, for the model to search or read.
- **Tools** (on by default): the model may search and read the documents (`docs_search`, `docs_read`), list, read and search the files, and write in the workspace (`files_write`, `files_edit`, `files_mkdir`). A tool that writes runs only once you allow it, its arguments shown; the calls and their results are shown, folded, in the conversation.

The index searches by words; by meaning too when `janas-docs` has an embedding model (`.brebus/config/janas/docs.conf`, `embedding_model =`).

The recent ones are listed in `~/.local/share/janas/brebus/workspaces.json`; the conversations the first Brebus kept in that folder move, once, to the first workspace opened.

## What it has

| Section | |
|---|---|
| Chat | the workspace's conversations on the left, renamed, deleted or exported in Markdown to its `exports/`; a system message of each one's own; the reply as it is written, in Markdown, the model's reasoning folded above it, its tokens and how fast they came; Enter sends, Shift+Enter starts a new line, and while a reply comes the button in the field stops it, as Esc does |
| Discover | the models of `janas-get`, how much each one downloads and asks of the memory, and whether it is meant for a machine of this memory; *Get* downloads one, with how far it is, and *Stop* keeps what came so far for the next *Get*; any other model of Hugging Face by its file's address or as `hf:owner/repo/file.gguf` (a GGUF converted here, a JNS as it is) |
| My models | the models in Janas's models folder (`janas-get`'s, `$JANAS_MODELS`), with their size and context; *Settings* keeps for each one its context, reasoning on or off and how many conversations at once, and *Load* starts the model's server with them |
| Server | the model loaded, and its addresses for other programs: OpenAI's API at `/v1`, Ollama's and Anthropic's beside it, on the local network too where it is open to it; the engine monitor in a window of its own; the settings: reachable from the local network, the key other programs must send (a new one made on request), the web pages of other sites allowed to call it |
| Machine | the processor, its features, the memory and the power |

## How it works

`brebus` serves its page and a small API of its own (`/brebus/...`) on `127.0.0.1` only, and opens no model itself. *Get* runs the `janas-get` beside it, one download at a time, into the models folder: the file is checked against its SHA-256 as from the terminal, and the page follows its steps. *Load* starts a `janas-server` beside it, on the next port, with the model and the server's own choices (several conversations at once where the memory holds them, see [serving](serving.md)). The page's chat goes to Brebus, which passes `/v1/...` and `/api/...` on to that server; other programs can use the server's address directly. One model at a time: loading another one ends the first. The key reaches `janas-server` in its environment, never on its command line, which every user of the machine can read; Brebus's own page needs none, as Brebus adds it to what it passes on. The server's output goes to `~/.local/share/janas/brebus/server.log`, the models' settings to `settings.json` beside it.

A page of another site could make the browser send requests to the loopback address: Brebus answers only requests for its own address, and refuses those of any other page (`403`).

## What has been tried

On the development laptop, with Qwen3-0.6B: the page in Chromium (headless), the model loaded from *My models*, two turns of a chat with their reasoning, the conversation still there after the page was opened again, the model unloaded, no `janas-server` left; Qwen3-0.6B downloaded from *Discover* into an empty folder and then in *My models*, Qwen3.5-2B stopped at 0.27 GB and taken up from there; settings saved and the server started with them (`--parallel 2 --ctx 8192 --think off`), a reply in Markdown, a conversation renamed and deleted, and those the first Brebus kept in the browser moved to their files. Its window on a virtual screen (Xvfb) with the models of the development laptop. With Qwen3-4B, from the page in Chromium: a note attached and put whole into the message, quoted by the model; "which river is in Lazio, and how long" answered "the Tevere, 405 km" from an Excel sheet in a folder read only, its document named; a file written by the model with `files_write` once allowed. `tests/test_brebus` starts `brebus` with no model and checks its page, its API, another site's page refused, a model that is not there refused (to load and to download), workspaces (none at first, the home and `/` refused, a new one in `Documents/Brebus`, one opened), conversations and settings saved, listed, read and deleted, with bad names and what is not a JSON object refused, exports never written over a file, the tools of the workspace (a write where it may only be read refused, one in the workspace done), a folder read only added (a path that is not there refused), a file attached by copy with its text, `/v1` answered 503 with no model loaded, and Brebus ending on SIGTERM.
