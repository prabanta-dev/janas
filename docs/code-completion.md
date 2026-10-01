# Code completion in the editor

`janas-server` fills in code at the cursor of an editor, as the
completion extensions of VS Code and VSCodium ask for it: the code before
the cursor and the code after it go to the model, which writes what goes
between. It runs on the processor, with a model for code; nothing leaves
the machine.

## The model

A model trained to fill in the middle is needed, with the
`<|fim_prefix|>`, `<|fim_suffix|>` and `<|fim_middle|>` tokens of the Qwen
coders. Every Qwen3-Coder is, says its
[README](https://github.com/QwenLM/Qwen3-Coder). The one tried is
**Qwen3-Coder-30B-A3B-Instruct**, Q4_K_M (unsloth's GGUF, converted with
`gguf2jns`): 18.6 GB, a mixture of experts with 3 billion parameters
active a token, which on a 32 GB machine fits in memory whole
(`--cache 18`).

Qwen3-Coder-Next fills in too, but its recurrent layers cannot go back to
a point of a sequence: every keystroke would read the window again from
its start.

## Starting the server

```sh
janas-server models/jns/qwen3-coder-30b-a3b-q4km.jns --cache 18
```

It listens on `http://127.0.0.1:8080`. `--verbose` writes on stderr, for
each request, where its window was cut and why.

## The two routes

- **`POST /v1/completions` with a `suffix`**, OpenAI's: the `prompt` is
  the code before the cursor, the `suffix` the code after it.
- **`POST /infill`**, llama.cpp's, which the extensions made for
  llama.cpp's server ask: `input_prefix` (the lines before the cursor),
  `prompt` (the line up to it), `input_suffix` (what follows), and the
  answer as llama.cpp gives it (`content`, `timings`). It reads
  `n_predict` (0: the prompt only, read so that the next request finds it
  computed), `n_indent` (a new line indented less ends the text) and
  `t_max_predict_ms` (after a new line, past that time the text ends), as
  llama.cpp does. The code of other files (`input_extra`) is not used: it
  comes first, changes often, and every change would have the whole
  prompt read again. The most likely token is always taken, so the same
  place gives the same proposal.

## The window

An editor asks at every keystroke. The server keeps computed what the
previous request shares from its start, so what matters is that the
start of the text sent does not move; and the extensions move it, sending
"the last N lines before the cursor". So, for requests that fill in:

- the code before the cursor is read up to `--fim-prefix` tokens (2048);
  over that it is cut at a line to three quarters of it, which leaves
  room for many lines before the next cut;
- the start of each window is remembered, for a few files, and a later
  request holding it is cut there;
- a request whose text starts inside the head of a window before, past
  its start, comes from an editor that slides its window: its start is put
  a quarter of the way in, so that the lines it slides next keep it in;
- the code after the cursor, read again at every request, is cut after
  `--fim-suffix` lines (8);
- the text written is at most `--fim-max` tokens (256), whatever the
  request asks: the extensions ask for thousands, and the requests of the
  next keystrokes would wait for them.

Each cut costs one reading of the whole window; the keystrokes after it
read only what they changed and the lines after the cursor.

## In VSCodium (and VS Code): llama.vscode

[llama.vscode](https://github.com/ggml-org/llama.vscode), by the authors of
llama.cpp, is on Open VSX, where VSCodium finds it. In the settings:

```json
{
  "llama-vscode.endpoint": "http://127.0.0.1:8080",
  "llama-vscode.max_parallel_completions": 1,
  "llama-vscode.ring_n_chunks": 0,
  "llama-vscode.rag_enabled": false,
  "editor.quickSuggestions": {"other": "off", "comments": "off", "strings": "off"},
  "editor.inlineSuggest.suppressSuggestions": true
}
```

One proposal is written (`max_parallel_completions`), the chunks of other
files would not be used (`ring_n_chunks`), and the editor's own list of
suggestions is kept from opening by itself: while it is open, a proposal
that does not extend its selected entry is not shown. Its welcome page
offers to install llama.cpp: it is not needed.

Tab takes the proposal, Shift+Tab its first line, Ctrl+Right its first
word; Ctrl+L asks for one.

## Continue

[Continue](https://www.continue.dev) works too, with three things to know:
the provider must be one that sends the suffix (`siliconflow`, whose
requests go to `{apiBase}completions` with `prompt` and `suffix`: the
`openai` provider builds the prompt itself and sends it as raw text), its
default `modelTimeout` of 150 ms cuts a proposal after ten pieces, and the
snippets of other files it puts in front of the prefix change at every
request. In `~/.continue/config.yaml`:

```yaml
name: Janas
version: 1.0.0
schema: v1
models:
  - name: Janas Coder
    provider: siliconflow
    model: qwen3-coder-30b-a3b-q4km
    apiBase: http://127.0.0.1:8080/v1/   # the final slash is needed
    apiKey: none
    roles:
      - autocomplete
    autocompleteOptions:
      maxPromptTokens: 2048
      prefixPercentage: 0.85
      maxSuffixPercentage: 0.1
      debounceDelay: 500
      modelTimeout: 20000
      multilineCompletions: never
      useCache: false
      useRecentlyEdited: false
      useRecentlyOpened: false
      useImports: false
      experimental_includeRecentlyVisitedRanges: false
      experimental_includeRecentlyEditedRanges: false
      experimental_includeDiff: false
```

With Continue some proposals the server wrote were not shown, and the
editor was not seen to ask again after some keystrokes; why was not found
(Continue filters, caches and reuses proposals before showing them).
llama.vscode's behaviour was easier to follow.

## What has been run

1 October 2026, Qwen3-Coder-30B-A3B-Instruct Q4_K_M on an Intel Core
Ultra 9 185H with every expert in memory (`--cache 18`), VSCodium 1.135
with llama.vscode 0.0.67 and Continue 2.0.0.

The window, with requests as an editor makes them on the 1,445 lines of
`src/chat/main.c`, the cursor going down a line a request, time to the
first token:

| Requests | first request | the ones after |
|---|---|---|
| before the window: the 110 lines before the cursor, slid a line (1,330 tokens) | 17 s | 17 s: the whole window read again |
| before the window: the file from its start to the cursor (5,300 tokens) | 81 s | 7 s |
| now: the 200 lines before the cursor, slid a line (cut to 1,527 tokens) | 19 s | 1.5-2.2 s |
| now: the file from its start to the cursor (cut to 1,566 tokens) | 20 s | 1.6-1.8 s |

In VSCodium with llama.vscode, writing in `src/flights/schedule.c`: after
the first request of a window, each request found 680-800 of its 750-870
tokens computed and was answered in one to two seconds (about 90 tokens
read, the lines after the cursor and what had changed, and 2-30
written).

What it proposed:

- under the comment "how many answers are kept and still fresh", the body
  of a new function counting the answers kept, with the names of the file
  (`kept[]`, `N_KEPT`); with `time_t now = time(NULL);` written first, it
  added the test on their age (`now - kept[i].at < KEEP_S`) by itself;
- for a line taken out of `local_time` (`tm.tm_mon -= 1;`), the right line
  first (`tm.tm_mon--;`), and then lines of its own: a handling of time
  zones the file does not have. With 8, 16, 32 or 64 lines after the
  cursor given to it the first line was always right and the rest always
  invented: the model does not see that only one line was missing.
  Shift+Tab takes the first line only.

The first request of a new window reads all of it, at 65-80 tokens a
second: 10-20 s for the 750-1,500 tokens of a window.
