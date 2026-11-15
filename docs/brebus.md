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

## What it has

| Section | |
|---|---|
| Chat | conversations on the left, a file each in `~/.local/share/janas/brebus/chats`, renamed, deleted or exported in Markdown to the downloads folder; a system message of each one's own; the reply as it is written, in Markdown, the model's reasoning folded above it, its tokens and how fast they came |
| Discover | the models of `janas-get`, how much each one downloads and asks of the memory, and whether it is meant for a machine of this memory; *Get* downloads one, with how far it is, and *Stop* keeps what came so far for the next *Get* |
| My models | the models in Janas's models folder (`janas-get`'s, `$JANAS_MODELS`), with their size and context; *Settings* keeps for each one its context, reasoning on or off and how many conversations at once, and *Load* starts the model's server with them |
| Server | the model loaded, and its addresses for other programs: OpenAI's API at `/v1`, Ollama's and Anthropic's beside it |
| Machine | the processor, its features, the memory and the power |

## How it works

`brebus` serves its page and a small API of its own (`/brebus/...`) on `127.0.0.1` only, and opens no model itself. *Get* runs the `janas-get` beside it, one download at a time, into the models folder: the file is checked against its SHA-256 as from the terminal, and the page follows its steps. *Load* starts a `janas-server` beside it, on the next port, with the model and the server's own choices (several conversations at once where the memory holds them, see [serving](serving.md)). The page's chat goes to Brebus, which passes `/v1/...` and `/api/...` on to that server; other programs can use the server's address directly. One model at a time: loading another one ends the first. The server's output goes to `~/.local/share/janas/brebus/server.log`, the models' settings to `settings.json` beside it.

A page of another site could make the browser send requests to the loopback address: Brebus answers only requests for its own address, and refuses those of any other page (`403`).

## What has been tried

On the development laptop, with Qwen3-0.6B: the page in Chromium (headless), the model loaded from *My models*, two turns of a chat with their reasoning, the conversation still there after the page was opened again, the model unloaded, no `janas-server` left; Qwen3-0.6B downloaded from *Discover* into an empty folder and then in *My models*, Qwen3.5-2B stopped at 0.27 GB and taken up from there; settings saved and the server started with them (`--parallel 2 --ctx 8192 --think off`), a reply in Markdown, a conversation renamed and deleted, and those the first Brebus kept in the browser moved to their files. Its window on a virtual screen (Xvfb) with the models of the development laptop. `tests/test_brebus` starts `brebus` with no model and checks its page, its API, another site's page refused, a model that is not there refused (to load and to download), conversations and settings saved, listed, read and deleted, with bad names and what is not a JSON object refused, exports never written over a file, `/v1` answered 503 with no model loaded, and Brebus ending on SIGTERM.
