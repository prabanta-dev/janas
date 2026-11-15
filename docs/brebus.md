# Brebus

Brebus is Janas's app: the models of this machine, a chat with them and the server for other programs, in one page of the browser. *Brebus*, in Sardinian, are the magic words of the *janas*. It is at its beginning, Linux only, and on the `develop` branch.

```sh
bin/x86_64-linux/brebus            # http://127.0.0.1:41414/, opened in the browser
bin/x86_64-linux/brebus --no-open  # the address only
bin/x86_64-linux/brebus --port 41500
```

It runs until Ctrl+C, and the model's server ends with it.

## What it has

| Section | |
|---|---|
| Chat | conversations on the left, kept in the browser; the reply as it is written, the model's reasoning folded above it, and how fast it came |
| Discover | the models of `janas-get`, how much each one downloads and asks of the memory, and whether it is meant for a machine of this memory; downloading them from the page comes next, `janas-get <name>` today |
| My models | the models in Janas's models folder (`janas-get`'s, `$JANAS_MODELS`), with their size and context; *Load* starts the model's server |
| Server | the model loaded, and its addresses for other programs: OpenAI's API at `/v1`, Ollama's and Anthropic's beside it |
| Machine | the processor, its features, the memory and the power |

## How it works

`brebus` serves its page and a small API of its own (`/brebus/...`) on `127.0.0.1` only, and opens no model itself: *Load* starts a `janas-server` beside it, on the next port, with the model and the server's own choices (several conversations at once where the memory holds them, see [serving](serving.md)). The page's chat goes to Brebus, which passes `/v1/...` and `/api/...` on to that server; other programs can use the server's address directly. One model at a time: loading another one ends the first. The server's output goes to `~/.local/share/janas/brebus/server.log`.

A page of another site could make the browser send requests to the loopback address: Brebus answers only requests for its own address, and refuses those of any other page (`403`).

## What has been tried

On the development laptop, with Qwen3-0.6B: the page in Chromium (headless), the model loaded from *My models*, two turns of a chat with their reasoning, the conversation still there after the page was opened again, the model unloaded, no `janas-server` left. `tests/test_brebus` starts `brebus` with no model and checks its page, its API, another site's page refused, a model that is not there refused, `/v1` answered 503 with no model loaded, and Brebus ending on SIGTERM.
