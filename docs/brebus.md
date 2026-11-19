# Brebus

Brebus is Janas's app: the models of your computer, a chat with them and a server for other programs, in a window of its own. Nothing leaves the computer unless you ask it to: the models run on it, your chats and documents stay on it. *Brebus*, in Sardinian, are the magic words of the *janas*. It runs on Linux and Windows; macOS is not supported yet.

![Brebus: a chat with Qwen3-30B-A3B, which made a page with a doughnut chart of a monthly budget, open beside the chat as an artifact](img/brebus.png)

## Get it

| | |
|---|---|
| Linux x86-64 (Ubuntu 22.04, Debian 12 and later) | the release's `janas-linux-x86_64.tar.gz`: unpack it and start `./brebus` |
| Windows 10 and 11 x86-64 | the release's `janas-windows-x86_64.zip`: unpack it and start `brebus.exe` |

Releases: [github.com/prabanta-dev/janas/releases](https://github.com/prabanta-dev/janas/releases). Brebus carries Janas-LLM and its tools beside it (`janas-server`, `janas-get`, the converters, the services); nothing else needs installing.

- **Linux:** the window is WebKitGTK's (`libwebkit2gtk-4.1`; on Debian and Ubuntu the package `libwebkit2gtk-4.1-0`), loaded when Brebus starts. Where it is missing, or there is no display, Brebus opens its page in the browser.
- **Windows:** the window is WebView2, the engine of Edge that Windows 10 and 11 carry, through `WebView2Loader.dll` beside `brebus.exe` (Microsoft's, under its BSD licence, `LICENSE-WebView2.txt`).

From a terminal:

```sh
./brebus                    # its window
./brebus --install-desktop  # Linux: in the desktop's menu, with its icon
./brebus --browser          # in the browser instead
./brebus --no-open          # neither: http://127.0.0.1:41414/
./brebus --port 41500       # another port
./brebus --workspace DIR    # open that folder as a project
```

## The first start

The window opens maximized the first time; afterwards it opens where it was left, at the size it had (on Windows, also on the screen it was on, or on the main one if that screen is gone; on Linux with Wayland the desktop places it, and only the size and maximized state are kept). On Windows its frame follows the system's light or dark theme.

Brebus opens on a new chat. With no model on the computer, the model button under the field says *Choose a model*: *Other models…* in its menu, or *Discover* in the side bar, lists the models Janas knows, with how much each one downloads and how much memory it is meant for. *Get* downloads one, checked against its published fingerprint; then it is chosen under the field and loads by itself. From then on Brebus loads, at each start, the model of the chat it opens, or the one used last; with none remembered, the largest installed model that leaves half the memory free.

At each start Brebus opens the chat used last, in its project or among the free chats; if that chat is gone or archived, the newest free chat, or a new one.

## The window

![A project's chat: the side bar with the free chats and the project "Trip to Sardinia" with its chats, the reply as an artifact beside it](img/brebus-project.png)

**The side bar**, from the top:

- the magnifier (Ctrl+K) opens *Find a chat*; the panel icon (Ctrl+B) hides the bar to a strip of icons, and its right edge, dragged, widens it;
- *New chat* (Ctrl+Shift+O): a chat of no project;
- *Find a chat*: the chats whose title or messages hold the words typed, in every project;
- **Chats**: the free chats, the newest first; a **Projects** group, each project with its chats; **Archived**, folded. The arrow beside each group's name folds it; under the pointer, `+` makes a new chat (or, beside *Projects*, a new project), and `⋯` beside a chat opens its menu without opening it: *Open*, *Rename*, *Move to…*, *Duplicate*, *Export*, *Archive* or *Restore*, *Delete*;
- at the bottom, the pages: *Discover*, *My models*, *Server*, *Settings*, *Info*.

**The chat's bar**, over the conversation: its title opens what is done to the chat (*Rename*, *Move to…*, *Duplicate*, *Export*, *In a new window*, *Archive*, *Delete*); the project it belongs to is named beside it; `⋯` at the right lists the chat's artifacts and the pictures its replies showed, the files attached to it, and *Instructions and answer*.

**Under the field**, from the left: `+` attaches a file or a web address; the speaker reads the replies aloud (grey and crossed off: off; orange: on); then, at the right, the ring of the model's memory, the model that answers (its menu chooses another, or *Other models…*), the engine's monitor and the model's settings (the gear).

## Chats and projects

A **chat** belongs to no project: its conversation is kept in Brebus's data folder, and the model's tools see only that folder.

A **project** is a folder of yours, for work that goes on: the model searches and reads its documents, writes there what it makes for you (artifacts, exports, files), and its chats are kept in it (`.brebus/chats/`). `+` beside *Projects* opens one: a recent folder, a folder you choose (*Browse…* opens the system's own dialog), or a new one, `Documents/Brebus/<date>`. The home folder itself and `/` are refused, too wide for a model's files. `⋯` beside a project opens its settings: the folders to read only beside it, the services, the model that searches its documents by meaning.

A chat moves between the free chats and the projects (*Move to…*). An **archived** chat is kept to read: it no longer takes messages, and *Restore it* brings it back.

**Find a chat** (the magnifier): words, and where to look for them - titles and messages, your messages only, or titles only; every project, the free chats or one project; any time, today, the last week, month or year; one model; archived chats too or not. The results are in a list; one opens its chat.

![Find a chat: words and filters, the results in a list](img/brebus-find.png)

| In a project's folder | |
|---|---|
| `.brebus/chats/` | its chats, a JSON file each |
| `artifact/` | what the model made to keep, `artifact/<title>/v1.html`, `v2.html`... |
| `exports/` | the chats exported in Markdown |
| `sources/` | the files attached by copy, and the web pages downloaded, each with `<name>.url.txt` (its address and when) |
| `.brebus/workspace.json` | the folders and files added to read only |
| `.brebus/data/`, `.brebus/config/` | the index of its documents, and its list |

## A conversation

- **The reply** is shown as it is written, in Markdown, with formulas (between `$$`, `\(` `\)`, or single dollars when they are plainly a formula: `$5 and $10` stays money) code in colours, and diagrams: a ```mermaid block is drawn where it is, as a picture (Brebus's copy of Mermaid, strict, so it runs nothing). The model's reasoning is shown folded above it, and opens while it is being written too; how many tokens came and how fast is shown under it.
- **Each message's actions**, small, under the pointer: on yours, copy, and *change this message and ask again from it*; on a reply, copy, *a new chat with the conversation up to this reply* (a branch, "Branch of …", to go on another way while the first stays as it is), read aloud, and on the last one *write this reply again* (the earlier ones kept, paged through with ‹ ›). *Duplicate*, in the chat's menu, copies it whole.
- **The ring under the field** fills as the chat takes the model's memory (its context): grey, orange from 70%, red from 90%. Under the pointer it shows the tokens used, the room, what is left and the model. When it is full, the earliest messages are forgotten; a new chat starts afresh. The room is set in the model's settings.
- **The field**: Enter sends, Shift+Enter starts a new line; while a reply comes, the button stops it, as Esc does. With the field empty, the button is a microphone (below, *Voice*).
- **Instructions and answer** (`⋯` in the chat's bar): what the model should be or do in this chat (its role, the language, how long its answers are); precise or creative answers; under *Advanced*, the temperature, top-p, and an answer in JSON (any object, or by a schema). Instructions can be kept under a name, to use again.
- **Export**: Markdown or plain text in the project's `exports/` (or Brebus's downloads), or a PDF through the system's print dialog.

## Documents and tools

For a project, Brebus starts [`janas-docs`](../src/services/docs/README.md), which indexes the project and the folders added to it (PDF, Word, Excel, PowerPoint, LibreOffice's, RTF, web pages, EPUB, text and Markdown), and [`janas-files`](../src/services/files/README.md), which reaches the project to read and write and the folders added only to read, not the rest of the home. Settings turns each of these on or off:

- **Search documents** (on): before each question, Brebus searches the project's documents for it and gives the model the passages found, each with its document. It works with models that cannot call tools. The passages are shown, folded, under the question. The documents are searched by their words, and by their meaning too with an embedding model of yours (the first, or the one chosen in the project's settings).
- **Tools** (on): the model may search and read the documents, list, read and search the files, and write in the project; a tool that writes runs only once you allow it, its arguments shown. Janas's services found beside Brebus (the web, Wikipedia, maps, weather, prices, flights, earthquakes, git, GitHub, this computer) and the MCP servers of your `~/.config/janas/mcp.json` (the format of Cursor and LM Studio) are offered as in `janas-chat`: a line for each, opened when a question needs it. A server of yours over HTTP that asks you to sign in (OAuth, as MCP's authorization says) has *Sign in* beside it in the project's settings: its own sign-in page opens in the browser, and the token it gives is kept for you alone (`~/.local/share/janas/mcp-auth`), renewed when it expires, and sent by `janas-chat` too; *Sign out* forgets it.
- **Attach** (`+` under the field): a file of this computer, or a web address. A file is copied into `sources/` or read where it is; a web address is downloaded into `sources/`. A document of up to 24,000 characters goes into the message whole; a longer one is named, for the model to search or read. A picture goes to the model as a picture when it sees pictures (Gemma 4, Qwen3.5: *My models* says "sees pictures"); a recording is transcribed by Whisper and the transcript goes into the message.

**Connectors**, in *Settings*: services of the web that speak MCP, added with a click - Notion, Linear, Atlassian, Sentry, Stripe, Hugging Face, Cloudflare, Zapier, Intercom, Supabase, Netlify and Microsoft Learn's and Cloudflare's documentation, each read from its maker's own documentation - then signed in to from a project's settings, Services (OAuth, above); and desktop extensions, the MCP Bundles (`.mcpb`) of Claude Desktop: Brebus reads the bundle's manifest, says whether its runtime is on this computer (Node.js, Python), asks for the settings it declares, unpacks it into `connectors/` beside `mcp.json`, every file kept inside its folder, and writes its entry in `mcp.json`, which janas-chat reads too. A bundle refused leaves the version installed as it was; *Remove* takes away its folder and its entry, the other entries as they were. Its program runs on this computer as its maker wrote it, not in a cage.

**Skills**, in *Settings*: instructions for particular tasks, which the model reads when a task needs them, in the format of Claude's skills - a folder with a `SKILL.md` (its name and description in a frontmatter, then the instructions) and the files it names. A skill made for Claude works here as it is. Those for every chat live in `skills/` beside `mcp.json` (`~/.config/janas/skills`, `%APPDATA%\janas\skills`), a project's own in its `.brebus/skills/`. The model's list of tools names each skill with its description, and the model reads one with `janas__skill` - its `SKILL.md`, or a file of its folder, never outside it - before the task; the scripts a skill names run with *Run code*, in its cage. *Settings* turns each on or off, adds one by copying its folder, and removes one of every chat's.

## Run code

With *Run code* on (Settings; off by default), the model may run Python in a cage of Janas's own; each piece of code runs only once you allow it, its code shown. In the cage: no network, no other process, limits on memory and time (2 minutes a piece), the project's folder to read and write (the folders added to it, to read), nothing else of the computer. Variables stay from one piece to the next; a figure drawn with matplotlib is saved in the project, and opens beside the chat when the reply names it. With *Run code* off, the model is told it cannot run code, and not to report results of code it did not run.

- **Linux**: the system's Python 3, in namespaces with seccomp (`src/common/sandbox.c`).
- **Windows**: an AppContainer with no network and the project's folder alone, inside a Job Object (memory, a single process, no access to your windows or clipboard). Python is not part of Brebus: the first time *Run code* is turned on, Brebus says what it would download - Python 3.14 from python.org and numpy and matplotlib with what they need from PyPI, about 45 MB - and does so only on *Install*, each file checked by its SHA-256. *Settings* removes it.

## Artifacts

With *Artifacts* on (by default), what the model makes to see or keep opens in a pane beside the chat, and the reply shows a card in its place. The pane is widened by its left edge, or put over the chat with ⤢. Its kinds are those of Claude's artifacts:

| Kind | Shown as |
|---|---|
| `html` | the page, its scripts run |
| `react` | a React component exported as default, its JSX read by Babel, Tailwind's classes, the imports of `react`, `recharts`, `lucide-react`, `d3`, `lodash`, `mathjs`, `papaparse`, `xlsx`, `chart.js`, `three`, `tone`, and shadcn/ui's components from `@/components/ui/...` (written for Brebus with the same names and props) |
| `mermaid` | the diagram |
| `svg` | the drawing |
| `markdown` | as the replies are, formulas with KaTeX |
| `code` | the program, its syntax in colours |

*Preview* and *Code* switch between the artifact and its text; *Copy*, *Download*, *Open* (the saved file, with the system's own program); the versions, each saved as `artifact/<title>/v1.html`, `v2.html`..., are paged through. A click on its title renames it, in the chat and in its folder. One the model left without a title takes the one it carries (a page's title, a document's first heading, a program's first comment), or `Untitled-001`, `Untitled-002`... An error of the artifact shows above it, with *Ask the model to fix it*.

A picture a reply names from the project (`![a chart](chart.png)`, a file the model's code wrote) is a card too, and opens in the pane.

An artifact runs in a frame with no origin of its own and no network: it cannot reach Brebus, the page around it, the computer or the Internet. The libraries it may use - React 18.3.1, Recharts 2.13.3, lucide-react 0.460.0, D3 7.9.0, lodash 4.17.21, Papa Parse 5.4.1, math.js 13.2.2, Chart.js 4.4.6, Three.js r128, Plotly 2.35.2, SheetJS 0.18.5, Mermaid 11.4.1, KaTeX 0.16.11, highlight.js 11.10.0, Tone.js 15.0.4, mammoth 1.8.0, Tailwind 4.1.10, Babel 7.26.4 - come from Brebus: each file is downloaded once from jsdelivr's copy of npm when first needed, checked against the SHA-256 Brebus carries, and kept in Brebus's data folder; a page that asks for one of them from jsdelivr, unpkg or cdnjs gets Brebus's copy. They are not part of Janas: each keeps its own licence (MIT, BSD, ISC or Apache 2.0). Inside a page or a component, as in Claude's: `await window.claude.complete(prompt)` asks the model loaded; `window.storage` keeps the artifact's data between visits (in `artifact/<title>/storage.json`, 20 MB at most); `window.fs.readFile(name)` reads a file attached to the chat.

## Voice

- **Dictation**: with the field empty, its button is a microphone. Speak, click again, and what you said is written in the field by OpenAI's Whisper, run on this computer by Janas: nothing you say leaves it. The first time, Brebus offers to get the speech model: whisper-small (0.49 GB), from [`prabanta-dev/whisper-small-JNS`](https://huggingface.co/prabanta-dev/whisper-small-JNS), checked against its fingerprint. *Settings*, *Voice* has it and the more accurate large-v3-turbo (1.63 GB), each got or removed there. A speech model is used from the next time the chat's model loads. Windows asks once for the microphone.
- **Reading aloud**: the speaker under the field reads the replies with the computer's own voices; *Settings*, *Voice* chooses one. On Linux the voices come from speech-dispatcher (with espeak-ng, for example); Janas's own voices will come with its speech synthesis.

## Discover

![Discover: a model's files on Hugging Face, each with how far it is from running here](img/brebus-discover.png)

- **The catalog**: the models Janas knows, converted and checked, with how much each downloads and the memory it is meant for ("fits", or "needs more": it still runs, slower, reading its parts from the disk). *Get* downloads one into the models' folder, checked against its published fingerprint.
- **One field to search or get**: words search Hugging Face for models with GGUF files, the most downloaded first; an address (`https://huggingface.co/…` or `hf:owner/repo/file.gguf`) gets that file. *Clear*, or emptying the field, brings the catalog back into view.
- **A model's files**, each with its size and the free space its conversion takes (about twice a GGUF's size), and **how far it is from running here**: *Found* (listed); *Check* reads its first megabyte and says *Likely to run* (its kind of model is one Janas runs; only the conversion and the first load prove it) or *Not supported yet*; *Not available* (a gated model, which wants its licence accepted on its page); *Converted here*, *Loaded here*. A GGUF is converted here after its download; a GGUF split in parts, or a vision encoder alone, is not offered.
- **Several downloads at once** (four at most), each with its progress and *Stop*; *Get* again goes on from where it stopped. A download starts only when the disk has room for it beside those going on, or Brebus says how much is needed and how much there is.

## My models

The models in the models' folder (`janas-get`'s, `$JANAS_MODELS`, or the one chosen with *Change…*), with their kind, size and context, and whether they see pictures. *Load* starts one (the loaded one reads *Loaded*); *Settings* says how it runs - how much of the chat it keeps in mind, whether it thinks before answering, and under *Advanced* where it reads your message, checks its guesses and writes the reply (by default chosen by Janas, which tries the processor, the GPU and the NPU this computer has and keeps the fastest; one of them, or the NPU and the GPU together, to try a device or leave one free), how it reaches the graphics card (by default Vulkan, which every maker's card has, or where there is none the maker's own: CUDA for NVIDIA, HIP for AMD; or one of them chosen) and how many chats it answers at the same time - used the next time it loads; *Info* shows what Janas chose for the model loaded; *Remove* deletes it from the disk, with its prediction head and its vision encoder when no other model needs them.

## Server

The model loaded in Brebus can answer other programs too - an editor, a script, another chat app - through OpenAI's API at `/v1` and Ollama's and Anthropic's beside it, on this computer. *Server* shows the address for programs, the key, and the engine's monitor; its settings: free the memory after so many minutes without a message (the model loads again with the next one), let other computers of the network use it (with a key, so that only those who know it may), encrypt it (HTTPS: Brebus makes a certificate for this computer, its names and addresses, kept in its data folder's `tls/` with the key readable by you alone, renewed before it expires; *Server* shows its SHA-256 fingerprint, for a program to compare before trusting it), and, under *Advanced*, the key and the web pages of other sites allowed to call it. See [serving](serving.md).

## Settings and Info

*Settings*: Search documents, Tools, Run code and Artifacts, each with what it does; *Voice*; *When the computer starts* (Brebus in the background at login, without its window, the model loaded for other programs); Python for Run code on Windows. *Info*: Brebus's version, its makers and licence, its source, this computer (its processor, instructions, memory now free, power, the model loaded) and the engine's monitor.

## Where Brebus keeps things

| | Linux | Windows |
|---|---|---|
| Brebus's data: settings, the free chats, the window's place, the artifacts' libraries, the server's log | `~/.local/share/janas/brebus` | `%LOCALAPPDATA%\janas\brebus` |
| The models | `~/.local/share/janas/models` | `%LOCALAPPDATA%\janas\models` |
| A project's chats and files | in the project's folder | in the project's folder |
| Python for Run code | the system's | `%LOCALAPPDATA%\janas\python` |

## How it works

`brebus` serves its page and a small API of its own (`/brebus/...`) on `127.0.0.1` only, and opens no model itself. *Get* runs the `janas-get` beside it into the models' folder. *Load* starts a `janas-server` beside it, on the next port, with the model and its settings. The page's chat goes to Brebus, which passes `/v1/...` and `/api/...` on to that server; other programs can use the server's address directly. One model at a time: loading another one ends the first. The key reaches `janas-server` in its environment, never on its command line, which every user of the machine can read. Brebus answers only requests for its own address, and refuses those of another site's page (`403`). It ends when its window is closed (or with Ctrl+C), and the model's server ends with it.

Its icon is the Prabanta logo, drawn for each size and system ([src/brebus/icons](../src/brebus/icons/README.md)). On Linux, when the window opens, Brebus writes its icons in `~/.local/share/icons/hicolor` and, where there is none, `brebus.desktop` in `~/.local/share/applications`.

## What has been tried

- **Windows 11** (a virtual machine, November 2026):
  - the window, a project in Documents, Qwen3-0.6B downloaded and answering (at temperature 0 the same text to the byte as on Linux), documents found, a system file outside the project refused, the artifacts' libraries;
  - chats and projects: moves, archives, the side bar's menus, *Find a chat* with its filters;
  - whisper-small got from the microphone's offer, and a 7-second Italian recording written word for word in 4.3 s;
  - a reply read aloud with the voice chosen;
  - Python downloaded on *Install*, a matplotlib figure drawn and saved; the network, the user's folders and the models refused; MemoryError at the limit; a loop stopped at 120 s; nothing left running when Brebus closes.

  Every check of the page's controls went through the page itself, with no errors in its console.
- **Linux**: earlier, with Qwen3-4B and Qwen3-30B-A3B:
  - a note attached and quoted;
  - "which river is in Lazio, and how long" answered "the Tevere, 405 km" from an Excel sheet in a folder read only, its document named;
  - a file written by the model once allowed;
  - the lines of a file counted by Python in the cage as `wc -l` counts them;
  - the weather of Cagliari from the weather service;
  - an HTML page with a chart made as an artifact;
  - an answer written again, a question edited, an answer in JSON by a schema;
  - the model unloaded after a quiet minute and loaded again by the next message;
  - a page's attempts to read Brebus, the network and the page around it, all refused;
  - with Gemma 4 E2B, a picture described and an Italian recording transcribed and summarised.
- **`tests/test_brebus`** starts `brebus` with no model and checks:
  - its page, its API, another site's page refused;
  - the free chats at the start; projects listed, a chat moved and refused outside Brebus's projects; the search's filters;
  - chats and settings saved, listed, read and deleted;
  - the project's tools, a folder read only, a file attached;
  - Python in the cage;
  - a model that is not there neither loaded, nor downloaded, nor removed;
  - Brebus ending on SIGTERM.
