# Brebus

Brebus is Janas's app: the models of this machine, a chat with them and the server for other programs, in a window of its own. *Brebus*, in Sardinian, are the magic words of the *janas*. It is on Linux and Windows; on macOS it is not yet. In the binary packages it is `./brebus` (Linux) and `brebus.exe` (Windows), beside `janas-chat`.

```sh
bin/x86_64-linux/brebus                    # its window
bin/x86_64-linux/brebus --install-desktop  # in the desktop's menu, with its icon
bin/x86_64-linux/brebus --browser          # in the browser instead
bin/x86_64-linux/brebus --no-open          # neither: http://127.0.0.1:41414/
bin/x86_64-linux/brebus --port 41500
```

On Linux the window is WebKitGTK's (`libwebkit2gtk-4.1`, on Debian and Ubuntu the package `libwebkit2gtk-4.1-0`), loaded when Brebus starts: nothing is needed to build it, and where it is missing, or there is no display, Brebus opens its page in the browser. On Windows it is WebView2, the engine of Edge that Windows 10 and 11 carry, through `WebView2Loader.dll` beside `brebus.exe` (Microsoft's, under its BSD licence, `LICENSE-WebView2.txt`); started from Explorer or the Start menu, Brebus asks for its folder instead of taking the program's, keeps its data in `%LOCALAPPDATA%\janas\brebus`, and starts at login, when asked, through the user's Run key. The Python sandbox is Linux's only. Links to other sites open in the browser. Brebus ends when its window is closed (or with Ctrl+C), and the model's server ends with it.

Its icon is the Prabanta logo, drawn for each size and system ([src/brebus/icons](../src/brebus/icons/README.md)). When the window opens, Brebus writes its icons in `~/.local/share/icons/hicolor` and, where there is none, `brebus.desktop` in `~/.local/share/applications`: the desktop's menu has Brebus, and the task bar shows its logo (on Wayland it comes from there only). `--install-desktop` writes them without opening it, and the `JANAS_MODELS` given then is kept for the menu's start.

## The workspace

Brebus works in a folder, as Claude Code does: started from a terminal, the folder it is started in, or the one given with `--workspace DIR`; started from the desktop's menu, the page asks for one: a recent one, a path, or a new one, `Documents/Brebus/<date>` (then `<date> (2)`…). The home itself and `/` are refused, too wide for a model's files. The button under the logo shows it and changes it; in Brebus's window, *Browse…* opens the system's own dialog to choose a folder (GTK's, the desktop portal's on KDE and GNOME), there and to add a folder read only or attach a file; in the browser, the path is typed.

The box over the conversations searches those of every recent workspace, by their titles and their messages: each found with its workspace and the words around what was found; one of another workspace opens it.

| In the workspace | |
|---|---|
| `.brebus/chats/` | its conversations, a JSON file each |
| `exports/` | the conversations exported in Markdown |
| `artifact/` | what the model made to keep (pages, drawings, documents, programs), `artifact/<title>/v1.html`, `v2.html`... |
| `sources/` | the files attached to the chat by copy, and the web pages downloaded, each with `<name>.url.txt` (its address and when) |
| `.brebus/workspace.json` | the folders and files added to be read only |
| `.brebus/data/`, `.brebus/config/` | the index of its documents, and its list |

**Read only, beside it**: folders and files of yours that the model may search and read but never write, added and removed in the workspace's window. Their documents and code are indexed and read; their pictures and recordings are not (yet): attach one to the chat for the model to see or hear it. Video is not read by Janas.

## Documents and tools in the chat

Brebus starts, for the workspace, [`janas-docs`](../src/services/docs/README.md), which indexes the workspace and the folders added to it (PDF, Word, Excel, PowerPoint, LibreOffice's, RTF, web pages, EPUB, text and Markdown), and [`janas-files`](../src/services/files/README.md), which reaches the workspace to read and write and the folders added only to read, not the rest of the home.

- **Search documents** (on by default): before each question, Brebus searches the workspace's documents for it and gives the model the passages found, each with its document, as LM Studio's documents do; it works with models that cannot call tools. The passages are shown, folded, under the question.
- **Attach** (the `+` in the field): a file of this computer by its path, or a web address. A file is copied into `sources/` or read where it is (then added to the folders read only); a copy is offered up to 100 MB. A web address is always downloaded into `sources/`. A document of up to 24,000 characters goes into the message whole; a longer one is named, for the model to search or read.
- **Pictures and recordings**: a picture attached (PNG, JPEG, WebP, GIF) goes to the model as a picture when it sees pictures (a `.vision.jns` beside it: Gemma 4, Qwen3.5; *My models* says "sees pictures"), else the model is told it cannot; a recording (WAV, MP3, FLAC, OGG, M4A) is transcribed by the model's server with the best Whisper among your models (large-v3-turbo first), and the transcript goes into the message.
- **Tools** (on by default): the model may search and read the documents (`docs_search`, `docs_read`), list, read and search the files, and write in the workspace (`files_write`, `files_edit`, `files_mkdir`). A tool that writes runs only once you allow it, its arguments shown; the calls and their results are shown, folded, in the conversation.

- **Run code** (off by default): the model may run Python in a cage of Janas's own (`src/common/sandbox.c`): no network, no other process, limits on memory, files and time (2 minutes a piece), the workspace as its `/mnt/data` and its current folder, to read and write, the folders read only at their own paths, nothing else of the computer. Variables stay from one piece of code to the next; each piece runs only once you allow it, its code shown. Linux only.

The documents are searched by their words, and by their meaning too with an embedding model of your models (the first, or the one chosen in the workspace's window; *words only* to do without): the index is made again when it changes.

The dialogs are Brebus's own: the choice between copying a file and reading it where it is can be remembered, for files up to 100 MB and over.

The recent ones are listed in `~/.local/share/janas/brebus/workspaces.json`; the conversations the first Brebus kept in that folder move, once, to the first workspace opened.

## Artifacts

With *Artifacts* on (by default), the model is asked to put what it makes to see or keep between `<artifact type="..." title="...">` and `</artifact>`, and the reply shows a card in its place; the artifact opens in a pane beside the chat, its left edge dragged to widen it, ⤢ to put it over the chat. Its kinds are those of Claude's artifacts:

| Kind | Shown as |
|---|---|
| `html` | the page, its scripts run |
| `react` | a React component exported as default, its JSX read by Babel, Tailwind's classes, the imports of `react`, `recharts`, `lucide-react`, `d3`, `lodash`, `mathjs`, `papaparse`, `xlsx`, `chart.js`, `three`, `tone`, and shadcn/ui's components from `@/components/ui/...` (written for Brebus with the same names and props) |
| `mermaid` | the diagram |
| `svg` | the drawing |
| `markdown` | as the replies are, formulas between `$$` with KaTeX |
| `code` | the program, its syntax in colours |

*Preview* and *Code* switch between the artifact and its text; *Download* saves it as a file; *Open* hands the saved one to the system's program; the versions, each saved as `artifact/<title>/v1.html`, `v2.html`..., are paged through. An error of the artifact shows above it, with *Ask the model to fix it*, which sends the error to the model.

An artifact runs in a frame with no origin of its own and no network: it cannot reach Brebus, the page around it, the computer or the Internet. The libraries it may use - React 18.3.1, Recharts 2.13.3, lucide-react 0.460.0, D3 7.9.0, lodash 4.17.21, Papa Parse 5.4.1, math.js 13.2.2, Chart.js 4.4.6, Three.js r128, Plotly 2.35.2, SheetJS 0.18.5, Mermaid 11.4.1, KaTeX 0.16.11, highlight.js 11.10.0, Tone.js 15.0.4, mammoth 1.8.0, Tailwind 4.1.10, Babel 7.26.4 - come from Brebus: each file is downloaded once from jsdelivr's copy of npm when first needed, checked against the SHA-256 Brebus carries, and kept in `~/.local/share/janas/brebus/lib`; a page that asks for one of them from jsdelivr, unpkg or cdnjs gets Brebus's copy. They are not part of Janas: each keeps its own licence (MIT, BSD, ISC or Apache 2.0). Other addresses stay out, and the pane says so.

Inside a page or a component, as in Claude's: `await window.claude.complete(prompt)` asks the model loaded and gives its answer as a string; `window.storage.get(key)`, `set(key, value)`, `delete(key)` and `list(prefix)` keep the artifact's data between visits, in `artifact/<title>/storage.json` (20 MB at most); `window.fs.readFile(name)` reads a file attached to the conversation, `{encoding: "utf8"}` as text. Code and formulas in the replies are shown with the same libraries.

## Services and your MCP servers

Beside the workspace's documents and files, the model may use Janas's services found beside Brebus (the web, Wikipedia, maps, weather, prices, flights, earthquakes, git, GitHub, this computer) and the MCP servers of your `~/.config/janas/mcp.json`, in the format of Cursor and LM Studio. As in `janas-chat`, their tools are not all put in the prompt: the model gets a line for each and opens one when a question needs it, its tools then given to it; a service starts only when it is opened. Each can be turned off in the workspace's window. A tool of Janas's that says it changes something asks you first; a tool of your servers asks you unless it says it only reads.

## What it has

| Section | |
|---|---|
| Chat | the workspace's conversations in the side bar, under the sections (the bar hidden to a strip of icons with « or Ctrl+B, and by itself when an artifact opens on a narrow window), in folders, searched with those of every recent workspace, renamed, deleted, exported in Markdown to `exports/` or as a PDF (the system's print dialog); each one's settings: a system message, temperature, top-p, an answer in JSON (any object, or by a schema), saved as presets; the reply as it is written, in Markdown, the model's reasoning folded above it, its tokens and how fast they came; *Regenerate* writes the last answer again, the earlier ones paged through; *Edit* changes a question and goes on from it; *New window* puts a chat in a window of its own, beside the other; Enter sends, Shift+Enter starts a new line, and while a reply comes the button in the field stops it, as Esc does |
| Discover | the models of `janas-get`, how much each one downloads and asks of the memory, and whether it is meant for a machine of this memory; *Get* downloads one, with how far it is, and *Stop* keeps what came so far for the next *Get*; Hugging Face searched by words (its models with GGUF files, the most downloaded first, each one's files with their sizes), or a file by its address or as `hf:owner/repo/file.gguf` (a GGUF converted here, a JNS as it is; a GGUF split in parts not yet) |
| My models | the models in Janas's models folder (`janas-get`'s, `$JANAS_MODELS`, or the one chosen here), with their size and context, and whether they see pictures; *Settings* keeps for each one its context, reasoning on or off and how many conversations at once, and *Load* starts the model's server with them |
| Server | the model loaded, and its addresses for other programs: OpenAI's API at `/v1`, Ollama's and Anthropic's beside it, on the local network too where it is open to it; the engine monitor in a window of its own; the settings: reachable from the local network, the key other programs must send (a new one made on request), the web pages of other sites allowed to call it, the model unloaded after a quiet while (loaded again at the next message; counted on Brebus's own chat) |
| Machine | the processor, its features, the memory and the power; Brebus's server started at login, without a window (opened from the menu, Brebus finds it and only opens its window) |

## How it works

`brebus` serves its page and a small API of its own (`/brebus/...`) on `127.0.0.1` only, and opens no model itself. *Get* runs the `janas-get` beside it, one download at a time, into the models folder: the file is checked against its SHA-256 as from the terminal, and the page follows its steps. *Load* starts a `janas-server` beside it, on the next port, with the model and the server's own choices (several conversations at once where the memory holds them, see [serving](serving.md)). The page's chat goes to Brebus, which passes `/v1/...` and `/api/...` on to that server; other programs can use the server's address directly. One model at a time: loading another one ends the first. The key reaches `janas-server` in its environment, never on its command line, which every user of the machine can read; Brebus's own page needs none, as Brebus adds it to what it passes on. The server's output goes to `~/.local/share/janas/brebus/server.log`, the models' settings to `settings.json` beside it.

A page of another site could make the browser send requests to the loopback address: Brebus answers only requests for its own address, and refuses those of any other page (`403`).

## What has been tried

On the development laptop, with Qwen3-0.6B: the page in Chromium (headless), the model loaded from *My models*, two turns of a chat with their reasoning, the conversation still there after the page was opened again, the model unloaded, no `janas-server` left; Qwen3-0.6B downloaded from *Discover* into an empty folder and then in *My models*, Qwen3.5-2B stopped at 0.27 GB and taken up from there; settings saved and the server started with them (`--parallel 2 --ctx 8192 --think off`), a reply in Markdown, a conversation renamed and deleted, and those the first Brebus kept in the browser moved to their files. Its window on a virtual screen (Xvfb) with the models of the development laptop. With Qwen3-4B, from the page in Chromium: a note attached and put whole into the message, quoted by the model; "which river is in Lazio, and how long" answered "the Tevere, 405 km" from an Excel sheet in a folder read only, its document named; a file written by the model with `files_write` once allowed; with Qwen3-4B and Run code, the lines of a file of the workspace counted by Python in the cage, as `wc -l` counts them (149). With Qwen3-30B-A3B (16 November 2026): the weather of Cagliari asked, the weather service opened by the model and its data answered; an HTML page with a bar chart made as an artifact, saved and shown in the pane; an answer written again (two kept, paged through), a question edited, an answer in JSON by a schema (`{"river": "Po", "km": 657}`), the model unloaded after a minute without a message and loaded again by the next one; a page's attempts to read Brebus, the network and the page around it, all refused. With Gemma 4 E2B, the Prabanta logo attached and described ("a dark brown square with black dots arranged radially"); an Italian recording attached, transcribed by Whisper large-v3-turbo and summarised. The workspace's 60 passages given their vectors by Qwen3-Embedding-0.6B. `tests/test_brebus` starts `brebus` with no model and checks its page, its API, another site's page refused, a model that is not there refused (to load and to download), workspaces (none at first, the home and `/` refused, a new one in `Documents/Brebus`, one opened), conversations and settings saved, listed, read and deleted, with bad names and what is not a JSON object refused, exports never written over a file, the tools of the workspace (a write where it may only be read refused, one in the workspace done), a folder read only added (a path that is not there refused), a file attached by copy with its text, python in the cage writing in the workspace and refused in the folder read only, `/v1` answered 503 with no model loaded, and Brebus ending on SIGTERM.
