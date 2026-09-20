# Janas

[![Status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)]()
[![Platform: Linux x86-64](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)]()
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Maurizio%20Cammalleri-0077B5?logo=linkedin)](https://www.linkedin.com/in/maurizio-cammalleri-80a89a11/)
[![Substack](https://img.shields.io/badge/Substack-Maurizio%20Cammalleri-FF6719?logo=substack)](https://cammalleri.substack.com/)

```
  ▄███████████▄    
▄███████████████▄  
███ ▄ █████ ▄ ███ ▄   J A N A S
███▄▄▄█████▄▄▄████ 
▀█████▄▀▀▀▄█████▀ ▀   a family of tools for artificial intelligence, in C
  █████████████    
 █ █ █ █ █ █ █ █   
▀ ▀ ▀ ▀ ▀ ▀ ▀ ▀    
```

Janas is a family of tools for artificial intelligence, written in C with no
dependencies beyond the C library. Its first component, **Janas-LLM**, runs
large mixture-of-experts language models on ordinary computers, with or without
a GPU: the weights stay on disk, the experts each token needs are streamed from
an NVMe SSD, and the ones that keep coming back stay in memory. An 80-billion
parameter model runs, and answers, on a laptop with 32 GB of RAM.

---

## ⚠️ Read this first

**This is an experiment, not a product.**

| | |
|---|---|
| **Stage** | Early development. Formats, file layouts, the public API and the command line **change without notice**, and have changed several times a week. |
| **Stability** | **Not verified.** The engine is deliberately hard on the machine: it fills the free memory with its expert cache and reads the disk at full speed with `O_DIRECT`. On a machine with less headroom than it reckons, the desktop can be pushed into swap and become slow or unresponsive. Using a GPU has, once, caused a driver reset that took the desktop with it. |
| **Data** | It writes only in `~/.cache/janas` (what it learns about your machine) and where you tell it to put a converted model. It does not touch the model files it reads. |
| **Hardware** | Developed and run on **one** machine (below). On anything else it is untested: it may be slower, it may refuse the model, it may misbehave. |

If something goes wrong, `Ctrl-C` stops a reply and `Ctrl-D` leaves the chat;
`--cache <GiB>` puts a hard limit on the memory it takes, and `JANAS_GPU=0`
keeps it off the GPU entirely.

## The machine it grew on

Everything in this repository was written and measured on:

- **Linux Debian 13**, x86-64
- **Intel Core Ultra 9 185H** (6 performance cores, 8 efficiency, 2 low-power; AVX2, FMA, F16C, AVX-VNNI; integrated Arc GPU)
- **32 GB** of RAM
- **1 TB NVMe SSD** (about 5.7 GB/s reading)

Everything the engine does is chosen from what it measures, so another machine
will make other choices — that part is meant to travel. What has never been
tried is the rest: other CPUs (especially without AVX-VNNI, or AMD), discrete
GPUs, slower disks, more or less memory. **If you run it somewhere else, the
report from `janas-bench` is the single most useful thing you can send back**
(see [Helping it run on more hardware](#helping-it-run-on-more-hardware)).

## What works now

- **Models:** Qwen3.5 and Qwen3.6 (`qwen35moe`, e.g. Qwen3.6-35B-A3B), Qwen3-Next
  (`qwen3next`, e.g. Qwen3-Next-80B-A3B-Instruct and Qwen3-Coder-Next), Qwen3
  MoE (`qwen3moe`, e.g. Qwen3-30B-A3B), the **dense** Qwen3.5 models
  (`qwen35`) and the **dense** Qwen3 models (`qwen3`), from the usual
  **Q4_K_M** GGUF files, or **Q3_K**, **IQ3_S**, **IQ4_NL**, **IQ4_XS**,
  **Q4_0**, **Q4_1** and **Q5_1** ones, and unsloth's "UD" mixes of them
  (checked on Qwen3.5-0.8B and 9B against llama.cpp). A dense
  feed-forward is a mixture with one expert and no router, so the same path
  serves both and there is no second engine to keep in step. Of the dense
  ones **Qwen3.5-0.8B, 2B and 9B, Qwen3-4B and Qwen3-0.6B are the ones that
  have been run here**; the Qwen3.5 three were checked against llama.cpp on
  the same files.
  **Nearly everything here was measured on Qwen3-Next-80B-A3B-Instruct**: it is
  the one the engine was shaped around, and the one whose numbers are quoted
  below. The others have been converted, checked against a reference and used,
  but far less.
- **In progress, not yet usable: Gemma 4** (`gemma4`). Gemma-4-12B-it
  converts, its tokenizer gives llama.cpp's tokens on a mixed text, and
  its forward pass runs; but the chat does not know Gemma's format yet, and
  with the engine's eight-bit KV cache its logits drift from llama.cpp's
  (the most likely token agrees at 175 of 200 positions; llama.cpp itself
  agrees at 169 when its own cache goes from f16 to eight bits). A finer
  cache for Gemma comes first.
- **Reasoning models:** the reasoning is shown apart, in grey, and can be turned off.
- **`janas-chat`:** a terminal chat with a line editor of its own, history,
  colours, a status line and a context that slides instead of ending.
- **`libjanas_llm`:** a C library ([`include/janas/llm.h`](include/janas/llm.h))
  and a FreeBASIC binding ([`include/janas/llm.bi`](include/janas/llm.bi)).
- **Tools from MCP servers** in `janas-chat`: the servers configured as the
  other clients configure them, each call shown and confirmed before it runs
  ([below](#tools-from-mcp-servers)). The client is a library of its own,
  **`libjanas_mcp`** ([`include/janas/mcp.h`](include/janas/mcp.h),
  [`include/janas/mcp.bi`](include/janas/mcp.bi)).
- **`janas-mcp`:** the other way round, the model offered as tools to MCP
  clients - a question to it, the embedding of a text
  ([below](#janas-as-an-mcp-server)).
- **`janas-server`:** the model behind an HTTP API that follows OpenAI's, so
  that clients written for it can use a model running on this machine:
  chat and completions with tools, JSON output held to a schema and
  log-probabilities, the Responses API with its conversations, embeddings
  and moderations, and a chat page of its own to try it from a browser
  ([below](#using-it-over-http)).
- **`janas-bench`:** measures the machine, fills its profile and writes a report.
- **Faster replies** from drafts the model verifies, so the text is exactly the
  one it would have written: from the model's own multi-token prediction block
  where it has one, from the conversation where it does not, or from a small
  model of the same family given with `--draft`.
- **Optional GPU** (Vulkan 1.3, integrated or discrete), used only where the
  engine measures that it helps. Results are identical with and without it.

On the machine above, with nothing else running, `janas-bench` reports this:

| Model | reading a prompt | writing a reply | with drafts, in a chat |
|---|---:|---:|---:|
| Qwen3.5-2B (dense, 1.3 GB) | 215 tok/s | 49 tok/s | **75 tok/s** |
| Qwen3-4B (dense, 2.5 GB) | 93 tok/s | 28 tok/s | see `--draft` below |
| Qwen3.5-9B (dense, 5.7 GB) | 46 tok/s | 13 tok/s | **22 tok/s** |
| Qwen3-30B-A3B (18.6 GB) | 112 tok/s | 33 tok/s | — |
| Qwen3.6-35B-A3B (22.3 GB) | 73 tok/s | 23 tok/s | **36 tok/s** |
| Qwen3-Next-80B-A3B (48.4 GB) | 65 tok/s | 23 tok/s | **36 tok/s** |

Every figure is the **mean of three rounds**, not a best round, of the
configuration the engine itself chose for each kind of pass - the one a chat
runs: all the usable cores (performance and efficiency cores, not the
low-power ones), with the integrated GPU only where it helped, which was
reading prompts on the two largest mixtures. The rounds run in alternating
order so that the machine warming up and the cache filling weigh on each
setting alike. Expect a few per cent either way between runs, and rather
less than these on a first run, while the cache is still filling.

The last column is the model's own prediction block guessing the next tokens
and the model checking them, which is the same text at a higher rate — the
whole of it is explained under [`--mtp`](#using-the-chat). It is measured on
a chat reply: an ordinary question in English ("explain how bread is made at
home"), 64 tokens of the answer, sampled as the chat samples. **What the
block is worth depends on the text**: there the drafts survived 78-89% of
the time; on Italian prose Qwen3.5-2B kept about a third of them, and
gained next to nothing. Qwen3-30B-A3B has no such block, nor have the dense
Qwen3; the dense Qwen3.5 have one in their original checkpoint, which
`hf2jns_mtp` converts ([MODELS.md](MODELS.md#the-fingerprints)).

**This column was wrong until 26 September 2026**, and higher: it measured
the block continuing a paragraph that repeated itself, which a model copies
and the block guesses almost every time - 107 tok/s for Qwen3.5-2B, 26 for
Qwen3.5-9B. The MoE figures of the time (29 and 30) happened to be lower
than today's; the ChangeLog has the details.

For a point of reference, llama.cpp (build `2b18470` of 18 September 2026)
on the same Qwen3.5-9B file and the same machine, measured the same night
with `llama-bench -p 256 -n 32` at 6, 12 and 20 threads, did best at 20:
**32 tok/s** reading the prompt and **9.0 tok/s** writing. The two tools do
not measure in exactly the same way (`llama-bench` writes without a prompt
before it; `janas-bench` after 256 tokens of one), and it is one model on
one machine.

**Try it on an idle machine first.** The expert cache takes the memory that is
free when the model is opened, and what is left of the model is read from disk
while it answers, so everything else running takes its share. It degrades
gently rather than breaking: measured with a browser and an editor open, the
same model kept about three quarters of its speed and waited five times longer
on the disk. But the first thing you see should be the machine's
real speed, and any measurement you mean to send to others has to be taken with
the machine to itself.

## Documentation

- [INSTALL.md](INSTALL.md) — what the build needs, how to prepare a model, and the first run
- [ChangeLog.md](ChangeLog.md) — what changed, newest first
- [MODELS.md](MODELS.md) — what a converted model's licence is, and what may be redistributed
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to help, and the sign-off
- [LICENSE](LICENSE) — GNU GPL, version 3 or later

## Getting started

```sh
sudo apt install build-essential          # Debian/Ubuntu
sudo apt install libvulkan-dev glslang-tools vulkan-tools   # optional, GPU

./build.sh                                # everything into bin/x86_64-linux/
./build.sh release test                   # and run the tests
```

## Getting a model

Janas reads its own format, `.jns`, converted from a **Q4_K_M** GGUF (Q3_K,
IQ3_S, IQ4_NL, IQ4_XS, Q4_0, Q4_1 and Q5_1 convert too, and so do unsloth's
"UD" files made of them; IQ2 and IQ1 types, not yet). Keep both
in `models/` next to the sources — the tools take a path, so anywhere works, but
that is where these examples put them, and where the project keeps its own.

| Model | GGUF converted here | GGUF | `.jns` | Memory it likes |
|---|---|---|---|---|
| **Qwen3-Next-80B-A3B-Instruct** — most of this engine was measured on it | [Qwen](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF) | 48.4 GB | 48.4 GB | 32 GB |
| **Qwen3.6-35B-A3B** — the easiest to start with | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) | 22.3 GB | 22.3 GB | 16-32 GB |
| **Qwen3-Coder-Next** | [Qwen](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) | 48.4 GB | 48.4 GB | 32 GB |
| **Qwen3-30B-A3B** | [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF) | 18.6 GB | 18.6 GB | 16 GB |
| **Qwen3.5-9B** — dense; its prediction block from the checkpoint ([MODELS.md](MODELS.md#the-fingerprints)) | [unsloth](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | 5.7 GB | 5.7 GB | 16 GB |
| **Qwen3.5-2B** — dense, small and quick; a prediction block too | [unsloth](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | 1.3 GB | 1.3 GB | 8 GB |
| **Qwen3-4B** — dense, and the smallest here that answers well | [Qwen](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | 2.5 GB | 2.5 GB | 8 GB |
| **Qwen3-0.6B** — not to talk to: to draft for a dense one (`--draft`) | [unsloth](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) | 0.40 GB | 0.40 GB | with the model above |
| **Qwen3-Embedding-0.6B** — not to talk to: for embeddings (`janas-server --embedding-model`); published in Q8_0 and converted as it is | [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) | 0.64 GB | 0.64 GB | beside the chat model |

Qwen3 has other dense models - 1.7B, 8B, 14B, 32B - which declare the same
architecture as the 4B and so should convert and run the same way. **None of
them has been tried here**, and until one is, that is a reading of the file
and not a claim. (Qwen3-30B-A3B is not one of them: the number is close but it
is a mixture of experts, with an architecture of its own.)

Any other Q4_K_M of the same models converts and runs just as well, including
the "dynamic" ones that give different tensors different types: the engine
reads Q4_K, Q5_K, Q6_K and Q8_0, and refuses a file carrying anything else
rather than guessing. The four above are the files the numbers in this
repository were measured on, and the ones [MODELS.md](MODELS.md) gives
fingerprints for, so that a converted file can be checked against the one
measured here without downloading anything.

All four models are published under the Apache 2.0 licence by Alibaba Cloud,
and quantized into GGUF by them or by the people above. Their licence is
theirs, not this project's: [MODELS.md](MODELS.md) says what that means if you
pass a converted file on to somebody else.

**Download.** Nothing needs installing — one file, resumable:

```sh
mkdir -p models/gguf && cd models/gguf
curl -L -C - -O \
  https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
cd ../..
```

With `pip install huggingface_hub` you can instead fetch a whole quantization,
which is handier when it is split into several files, as Qwen3-Coder-Next is:

```sh
huggingface-cli download Qwen/Qwen3-Coder-Next-GGUF \
    --include "Qwen3-Coder-Next-Q4_K_M/*" --local-dir models/gguf
```

**Convert**, with the converter the build made (nothing else to install).
With a model split into several files, give the first one and the rest is
found:

```sh
bin/x86_64-linux/gguf2jns models/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    models/qwen3.6-35b-a3b-flat.jns
```

It says what it is doing, and refuses a model it does not know with a clear
message rather than half a conversion.

**Cut the experts into bit planes.** Worth it on any machine that cannot hold
all the experts in memory, which is the case this engine is built for: the
down matrix is stored as three planes of two bits, so the engine can read four
bits of it, or two, instead of six, and decide which when the model is opened.
Nothing is lost — all three planes give the weights the file came with, bit for
bit. This is the form the models here are kept in:

```sh
bin/x86_64-linux/jns_planes models/qwen3.6-35b-a3b-flat.jns \
    models/qwen3.6-35b-a3b.jns
rm models/qwen3.6-35b-a3b-flat.jns
```

Afterwards:

```sh
bin/x86_64-linux/jns_check models/qwen3.6-35b-a3b.jns            # what is inside
bin/x86_64-linux/jns_check models/qwen3.6-35b-a3b.jns --verify   # and the checksums
```

Both steps are deterministic, so the file can be checked against the
fingerprints in [MODELS.md](MODELS.md).

Room needed: three copies of the model at the worst moment — the GGUF, the
converted file and the one with the planes — so about 145 GB for the largest.
The flat file goes as soon as the planes are written, and the GGUF can go too,
though keeping it saves the download the day a new version of the converter is
worth running.

**The multi-token prediction block** (Qwen3-Next and the dense Qwen3.5;
optional, faster replies). Its own model carries it, GGUF files leave it
out, so it comes from the original checkpoint. `hf2jns_mtp` can read just
the block's tensors from Hugging Face, with range requests, instead of the
shards that hold them:

```sh
bin/x86_64-linux/hf2jns_mtp models/qwen3-next.jns \
    hf:Qwen/Qwen3-Next-80B-A3B-Instruct models/qwen3-next-mtp.jns
```

For the dense Qwen3.5-9B that is 487 MB instead of 14 GB of shards, in under
a minute. It also takes the shards themselves, when you have them on disk
(`hf2jns_mtp <model.jns> <shard>... <out.jns>`). Qwen3.6-35B-A3B needs none of
this: its GGUF keeps the block.

**And chat:**

```sh
bin/x86_64-linux/janas-chat models/qwen3-next.jns \
    --mtp models/qwen3-next-mtp.jns --stats
```

The first start reads the model's resident weights (a couple of GB) and then
fills the expert cache in the background, while you read and type. The second
reply of a session is the one that shows the machine's real speed.

[INSTALL.md](INSTALL.md) has the details, including the other distributions and
what to do when the GPU is not found.

## Using the chat

```
janas-chat <model.jns> [options]
```

| Option | What it does |
|---|---|
| `--mtp <file>` | the model's multi-token prediction block: faster replies, same text |
| `--draft <file>` | a small model of the same family to guess the next tokens: faster replies, same text (see below) |
| `--ctx <tokens>` | context length (default 16384, never more than the model was trained for) |
| `--cache <GiB>` | memory for streamed experts (default: what the machine can spare) |
| `--reserve <GiB>` | when the cache is automatic, the memory left to other programs (default: a fifth of the machine's); more keeps a busy desktop out of swap, at the price of a smaller cache |
| `--bits <2\|4\|6>` | bits per weight of the experts' down matrix (default: the memory decides) |
| `--attention <exact\|fast>` | how the attention scores are computed (default: fast above 16384 tokens of context) |
| `--no-preload` | do not fill the expert cache at start from this machine's profile |
| `--no-recap` | when the context fills, drop the oldest turns without summing them up |
| `--system <text>` | the system message (`""` for none; the default gives the assistant its name, Janas, names the model it thinks with, and says who wrote the engine) |
| `--temp`, `--top-k`, `--top-p`, `--min-p`, `--seed` | sampling |
| `--max <tokens>` | longest reply (default: no limit — a model can fill the whole context) |
| `--no-spec` | no speculative decoding (slower, and the same text: see below) |
| `--no-markdown` | print the model's marks instead of reading them |
| `--no-gpu` | never give the GPU work, whatever the mode says |
| `--mode auto\|eco\|max` | power mode: `eco` never uses the GPU for speed alone |
| `--think on\|off` | reasoning before replying, for models that do it |
| `--stats` | a line of counters after every reply: speeds, time to the first token, threads, memory |
| `--progress <s>` | a line on stderr every `s` seconds while a long prompt is read (tokens read, rate, time left as an estimate), and one after each reply; on a terminal the status line shows the same while it reads |

**On `--mtp`, `--draft` and `--no-spec`.** Speculative decoding is not a trade of
quality for speed: it changes how many tokens come out of one pass, never
which ones. The small multi-token prediction block guesses the next few
tokens, the model then runs one pass over the guesses, and a guess is kept
only where the token the model itself sampled is the very same one - at the
first disagreement the guess is thrown away and the model's own token stands.
So no wrong guess can survive into the text, and with the same seed the reply
is the one you would have got without any of it - checked on every run of the
tests, 256 tokens with drafts and 256 without, from one seed, identical. On
the development laptop `--mtp` takes Qwen3-Next-80B-A3B from 23 tok/s to 36
and Qwen3.5-9B from 13 to 22 on an English explanation; how much it gives
depends on the text, and on Italian prose a small model gains little.
`--no-spec` and `/spec off` turn it off; they buy nothing but time, and are
there to measure with.

Where a model has no prediction block the guesses are copied from the
conversation instead, which costs nothing and is right about a quarter of the
time - enough to be worth it on a dense model and barely enough on a mixture.
**`--draft` puts a small model of the same family in their place**: it has to
share the vocabulary, it borrows the big model's threads, and it predicts
where copying only repeats, so it is right about three quarters of the time.
Whether that pays depends on what a second token in a pass costs, which is
about 6% on a dense model and about 28% on a mixture, where two tokens route
to different experts. So it is for dense models: Qwen3-0.6B drafting Qwen3-4B
gives **1.14x** - 23.0 to 26.3 tok/s, measured at the chat's own
temperature, which is why the pair is lower than the table above - and on
Qwen3-30B-A3B it would not pay for itself.

**On sampling.** The defaults are the ones Qwen recommends for its own models:
temperature 0.7, top-k 20, top-p 0.8. They are narrower than they look. With
top-p 0.8, on Qwen3-4B, **196 of 256 tokens had a single candidate left after
the cut** - nothing to draw, so the reply is the one the model would have given
greedily, and two questions that mean the same thing get the same words.
Starting with `--top-p 0.95` brings that to 142 of 256. It is not a fault and
not a setting this project chose: it is what those numbers do to a model that
is sure of itself.

**On `--attention`.** At a long context most of a token goes into attention,
and most of that into the scores: one dot product per position, per head. The
`fast` way reads the query as sixteen-bit integers with a scale of its own, so
the dot product becomes an exact integer sum — the machine does sixteen of
those products at a time instead of eight, and the scores come out a fifth to
a quarter quicker. What it costs is the query's quantization, a relative
3e-5, sixty times less than the eight-bit keys and values already carry; on
four hundred tokens against llama.cpp the most likely token agreed 394 times
out of 400 on C source against 393 the exact way, and 377 against 382 on
Italian prose. The engine asks for it by itself only when you asked for more
than 16384 tokens of context, where the gain is real and the conversation
will get there; `--attention exact` and `--attention fast` settle it either
way, and so does `JANAS_ATTN=float|int16`.

In the chat, a line starting with `/` is a command, and it turns grey-blue as
soon as it names one the chat really has:

| Command | |
|---|---|
| `/help` | the commands, on a page of its own |
| `/about` | what this is, who wrote it, and what it is running on |
| `/stats [on\|off]` | the counters after each reply |
| `/context` | how full the context is, and with what: the last prompt by its parts (system message and tools, conversation, last message) |
| `/reset` | a new conversation, and a clean window |
| `/system [text\|off]` | show or change the system message |
| `/temp <t>`, `/spec`, `/think` | sampling, drafts, reasoning |
| `/mode [auto\|eco\|max]` | power mode, and what the engine chose |
| `/experts [n]` | experts per token: fewer is faster and a little less accurate |
| `/markdown [on\|off]` | read the model's marks, or print them |
| `/gpu [on\|off]` | whether the GPU may be given work |
| `/save-config`, `/load-config` | keep these settings for the next start, or read them back; both ask first where something would be lost |
| `/del-config` | forget the saved settings: the next start uses the chat's own |
| `/quit` | leave (also `Ctrl-D`) |

`/help` and `/about` open a page over the conversation: the text is laid out
to the window however narrow it is, the arrows and **Page Up** and **Page
Down** move in it, and **ESC** closes it - no other key does, so a page cannot
be shut by the one you pressed looking for a way down. `/about` gives the
version, the author, the licence and the repository, and then the model, the
machine and the engine as they are at that moment: the three things a report
of a problem needs, which were until now to be gathered by hand.

What a command changes lasts for that conversation and no longer: a knob
turned to try something does not quietly become the way things are.
`/save-config` writes the settings to `~/.config/janas/chat.conf` and
`/load-config` reads them back, losing whatever is set at the time, which it
asks about first, and `/del-config` throws the file away so that the next
start uses the chat's own settings again. The file
holds the words of the command line, one option to a line and the rest of the
line its value, so the same reader takes both and you can open it and edit it;
it is read before the command line, which therefore wins. The system message
is not kept in it, and the context, the cache, the bits and the attention are
read at the next start, since they are settled when the model is opened.

A model marks its replies up - `**bold**`, `*italic*`, backticks, fenced
blocks, headings, bullets - and the chat reads those marks rather than
printing them: bold is bold, a block of code is told from the prose by its
colour, a dash becomes a bullet. An underscore counts as italic only alone and
at the edge of a word, so `_this_` and `**_this_**` are marks while
`snake_case` and `__init__`, which a conversation about code is full of, stay
as they are; a star inside a word is not a mark either, so `2*3` is a product.
`--no-markdown` and `/markdown off` print the marks as they come, and a pipe
always gets them: a script may want them.

The box the message is written in takes the rows the message needs, up to a
third of the window, and goes back to one when it has been sent: a long
question is read while it is written instead of scrolling away sideways.

The conversation is kept as it is written, so **Page Up** and **Page Down**
look back over it and **Ctrl-Home** and **Ctrl-End** go to its beginning and
its end. The line being written stays where it is while you read, and sending
brings the end back. A terminal cannot do this by itself here: the chat pins
its footer with a scrolling region, and the lines that leave the top of one
never reach the terminal's own history.

The arrows move in the message - up and down between its lines while it has
more than one - and bring back earlier prompts from its first and last line,
kept between sessions; `Home`, `End`, `Delete`, `Ctrl-A/E/K/U/W` and `Ctrl-L` do what they do
everywhere; `Tab` completes a command; a backslash and `Enter` add a line to
the message rather than sending it, the backslash itself is not part of what
the model reads, and the box grows to hold what is written; `Ctrl-C` stops a
reply.

Piped in or out, the chat writes plain text with no drawing at all, so it can be
scripted:

```sh
printf 'Explain mixture of experts in one paragraph.\n/quit\n' \
    | bin/x86_64-linux/janas-chat qwen3-next.jns --temp 0 --max 200
```

### Tools from MCP servers

The chat starts the [Model Context Protocol](https://modelcontextprotocol.io)
servers listed in `~/.config/janas/mcp.json` (or in the file `--mcp-config`
names), in the format the other clients use, so a configuration can be
copied from one of them:

```json
{"mcpServers": {
    "files": {"command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-filesystem",
                       "/home/me/notes"]}}}
```

Their tools are given to the model as `server__tool`. When the model calls
one, the chat shows the call with its arguments and waits for a key: `y` runs
it, `n` tells the model it was not allowed, `a` runs it and every later call
of the same tool. `/mcp` lists the servers and their tools; `/mcp auto`, or
`--mcp-auto` at the start, runs the calls without asking - which a pipe
needs, since nobody is there to answer, and without which a piped chat runs
none. `--no-mcp` starts no server. A tool runs with your rights, and what it
answers is text somebody else wrote that the model then reads: configure
only servers you trust, and read the calls before saying yes. For the same
reason the instructions a server gives for its tools stay out of the
model's system message unless `--mcp-instructions` asks for them; then
they follow it, marked as the servers' and not the user's.

Servers are spoken to over their standard input and output (the stdio
transport), in either era of the protocol: the stateless revision 2026-07-28,
and the handshake of 2025-11-25 and before, which most servers still speak;
which one a server speaks is found out when it starts. What a server writes
to its standard error goes to `~/.cache/janas/mcp/<name>.log`. A server
that listens at an address is named with `url` (and `headers`, for an
`Authorization` say) instead of `command`, and spoken to over Streamable
HTTP, again in either era; `https://` too, the server's certificate checked
against the system's certificate authorities (or `SSL_CERT_FILE`) and its
name.
It has been run against the test server in
[`tests/mcp_fake_server.c`](tests/mcp_fake_server.c), in both eras, over
stdio and over HTTP, and against two reference servers over stdio, both of
the 2025-11-25 era: `@modelcontextprotocol/server-filesystem` 0.2.0 (npx,
14 tools) and `mcp-server-git` 1.30.0 (uvx, 12 tools), together, with
Qwen3-4B choosing among their 26 tools - `git_log` for the last commit of a
repository, `list_directory` then `read_text_file` to find a date in a
folder. Their descriptions take about 3,700 tokens of the system message,
read once and kept computed. Over HTTPS it has reached DeepWiki
(`https://mcp.deepwiki.com/mcp`, 2025-11-25 era, a session per client). A small model does not always find its way:
asked without a hint, Qwen3-4B tried to read the folder itself as a file,
got the server's error, and stopped there.

### Watching a long prompt

A long prompt is most of what a reply costs: on a laptop CPU, 260,000 tokens
of Qwen3-Next-80B-A3B take hours to read and seconds to answer. So the
reading tells how it goes. The library reads a prompt a block of 256 tokens
at a time, and `janas_llm_chat_next` returns an empty piece after each
block, so a program has control between them; `janas_llm_chat_stats` says
the stage (reading, writing, done), the tokens read of those to read, the
time to the first token, the time spent building the prompt, the processor
time of the reading, the threads the engine chose, the memory of the
context and the process's peak, how the prompt is made up (system
message and tools, the conversation before, the last message), and an
estimate of the time left. `max_input`
in the chat's parameters refuses a longer prompt with its numbers, and never
cuts it.

Any program on the library, its own code unchanged, can have the same from
the environment: `JANAS_PROGRESS=30` writes a line on stderr every thirty
seconds while a prompt is read and one at the end of each reply, and
`JANAS_METRICS=file` appends JSON lines - events `open`, `prompt`,
`progress` (every `JANAS_METRICS_EVERY` seconds, 10 by default), `first_token`,
`reply` and `refused` - with counters and times only, never the text of a
prompt or a reply:

```json
{"event":"progress","time":"2026-09-25T08:44:52.027Z","pid":3948047,"reply":1,"stage":"input","input_done":1536,"input_tokens":3044,"elapsed_seconds":3.8,"tokens_per_second":362.75,"tokens_per_second_avg":405.32,"eta_seconds_estimate":4}
```

The time left is an estimate that follows how the reading slows: a token
costs more the more context it follows, attention having more to look at,
so the engine fits that cost as a line in the token's position over the
blocks read so far and adds it up to the end of the prompt, never below
what the latest speed says. On a 260,954-token prompt of
Qwen3-Next-80B-A3B the elapsed time followed that shape to R² 0.99999,
where an estimate from the current rate promised 2.5 times too little at
32K ([issue #9](https://github.com/prabanta-dev/janas/issues/9)); the new
estimate has not yet been checked on a run that long.
`janas-chat` shows the progress in its status line and takes `--progress`;
`janas-server` takes `--metrics`, `--progress`, `--max-input` and
`--warn-input`.

## Using it from a program

The library is shaped for foreign-function interfaces from the start: opaque
handles, fixed-width types, nothing passed by value that is not a number,
parameter blocks that carry their own size, error codes as `int32`, and reply
text returned on request as whole UTF-8 characters. It is declared for **C**
and for **FreeBASIC**.

### From C

[`include/janas/llm.h`](include/janas/llm.h).

```c
janas_llm *llm;
janas_llm_chat *chat;
struct janas_llm_params p;
janas_llm_params_default(&p);
janas_llm_open("qwen3-next.jns", &p, &llm);
janas_llm_chat_create(llm, NULL, &chat);
janas_llm_chat_send(chat, "Hello! Introduce yourself.", -1);

char piece[256];
int32_t len;
while (janas_llm_chat_next(chat, piece, sizeof(piece), &len) == JANAS_LLM_OK)
    fwrite(piece, 1, (size_t)len, stdout);
```

```sh
cc yours.c -Iinclude -Lbin/x86_64-linux -ljanas_llm -o yours
```

A program that keeps the conversation itself - an HTTP client does, and sends
it whole every time - hands it over with `janas_llm_chat_load`: the messages
with their roles, the last one the user's. Whatever the new conversation
shares with the one already computed is not read again, so the same messages
plus a new one cost only the new one. `janas_llm_chat_prompt` continues raw
text, with no chat format around it, and `janas_llm_default_system` gives the
system message Janas's own programs use. `janas_llm_chat_stats` says why a
reply ended and how much of its prompt was already computed.
`janas_llm_chat_keep` keeps a few conversations computed besides the current
one, for a program that switches between them: the one that shares most of
the next request is copied back instead of being read again.

### From FreeBASIC and BASIC MODERN

[`include/janas/llm.bi`](include/janas/llm.bi) declares the same library for
FreeBASIC, and so for **BASIC MODERN**, the dialect implemented by Prabanta —
the platform Janas is meant to sit inside. The same conversation as above:

```basic
#include once "janas/llm.bi"

dim as janas_llm ptr llm
dim as janas_llm_chat ptr chat
dim as janas_llm_params p
janas_llm_params_default(@p)
janas_llm_open("qwen3-next.jns", @p, @llm)
janas_llm_chat_create(llm, NULL, @chat)
janas_llm_chat_send(chat, "Hello! Introduce yourself.", -1)

dim as zstring * 256 piece
dim as long n
do
    '' the piece is not closed by a NUL: one byte is left for it
    dim as long r = janas_llm_chat_next(chat, @piece, sizeof(piece) - 1, @n)
    if r <> JANAS_LLM_OK then exit do
    piece[n] = 0
    print piece;
loop
```

[`tools/chat.bas`](tools/chat.bas) is a whole chat in a hundred lines, sampling
and counters included:

```sh
fbc tools/chat.bas -i include -p bin/x86_64-linux -x janas-chat-fb
LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-fb qwen3-next.jns
echo "Hello! Introduce yourself." | LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-fb qwen3-next.jns
```

Two things to know. FreeBASIC does not tell names apart by their case, so the
constant the C header calls `JANAS_LLM_ABI_VERSION` is `JANAS_LLM_ABI` there —
it would otherwise collide with the function `janas_llm_abi_version()`. And its
`line input` reads the terminal, not the standard input: to work in a pipe too,
`chat.bas` asks `isatty` and reads a pipe through the `CONS` device.

### MCP servers from a program

[`include/janas/mcp.h`](include/janas/mcp.h) is the MCP client the chat uses,
as a library of its own, `libjanas_mcp`, which needs nothing of
`libjanas_llm`; [`include/janas/mcp.bi`](include/janas/mcp.bi) declares it
for FreeBASIC. It fits the model's side of the tools: `janas_mcp_tools` gives
a server's tools as the JSON `janas_llm_chat_tools` takes, the calls the model
writes go to `janas_mcp_call`, and what `janas_mcp_result` returns goes back
with `janas_llm_chat_send_results`.

```c
janas_mcp *files;
janas_mcp_open(NULL, "files", 0, &files); /* from ~/.config/janas/mcp.json */
char tools[65536];
int32_t len;
janas_mcp_tools(files, "files__", -1, tools, sizeof(tools), &len);
janas_llm_chat_tools(chat, tools, len);
/* ... a reply that ends with calls: for each, name without "files__" */
janas_mcp_call(files, name + 7, -1, args, -1, 60);
janas_mcp_result(files, answer, sizeof(answer), &len);
/* ... then janas_llm_chat_send_results(chat, n, answers, NULL) */
```

```sh
cc yours.c -Iinclude -Lbin/x86_64-linux -ljanas_llm -ljanas_mcp -o yours
```

## Janas as an MCP server

`janas-mcp` offers the model running here to the clients of the Model
Context Protocol - Claude Code, editors, other agents - as tools: `generate`
asks it a question (a prompt, and optionally a system message, a longest
answer and a temperature) and returns its answer, and `embed` returns the
vector of a text when an embedding model is given with `--embedding-model`.
The client starts it and speaks to it over its standard input and output;
it is configured like any other stdio server:

```json
{"mcpServers": {"janas": {"command": "/path/to/janas-mcp",
                          "args": ["/path/to/qwen3-4b.jns"]}}}
```

It answers both eras of the protocol: a request that names revision
2026-07-28 in its `_meta` is served statelessly, and a client that opens with
`initialize` gets the handshake of the revision it asks for (2025-11-25,
2025-06-18, 2025-03-26 or 2024-11-05). A ping is answered while a generation
runs, and `notifications/cancelled` stops it. Every call starts a
conversation of its own; the reasoning of a model that reasons is never
returned, only the answer. `janas-mcp --help` lists the options (`--ctx`,
`--temp`, `--max`, `--think`, `--timeout` and the memory ones).

What has been run: a probe script speaking each era to it with Qwen3-0.6B
and Qwen3-Embedding-0.6B, `janas-chat` with Qwen3-4B calling its two tools
through `libjanas_mcp`, and Claude Code (2.1.282, revision 2026-07-28),
which listed both tools and called them - a translation, a summary, an
embedding - with Qwen3-4B and Qwen3-Embedding-0.6B behind them. To add it
there: `claude mcp add janas -- /path/to/janas-mcp /path/to/model.jns`.

## Using it over HTTP

```sh
bin/x86_64-linux/janas-server qwen3-next.jns
```

`janas-server` puts the model behind an HTTP API that follows
[OpenAI's](https://github.com/openai/openai-openapi), on
`http://127.0.0.1:8080/v1`, so that a client written for that API can use a
model running on this machine. It takes the model options `janas-chat` takes
(`--ctx`, `--cache`, `--reserve`, `--mtp`, `--draft`, `--mode`, `--no-gpu`) and these:

| Option | |
|---|---|
| `--host ADDR` | the address to listen on; `127.0.0.1` (the default) is this machine only, `0.0.0.0` opens it to the network |
| `--port N` | 8080 by default |
| `--api-key KEY` | ask every client for this key, as `Authorization: Bearer KEY`; `JANAS_API_KEY` sets it too |
| `--name ID` | the model's name for clients; the file's name without `.jns` by default |
| `--think on\|off` | reasoning before replying, for models that do it, when a request does not say |
| `--test` | serve a chat page for trying it out, on `/test` (below) |
| `--max-input N` | the longest input a request may have, in tokens, below the context: a longer one gets a 400 (`input_limit_exceeded`) whose error says `input_tokens`, `max_input_tokens`, `excess_tokens` and the parts, and nothing is ever cut to fit |
| `--warn-input N` | longer inputs are taken, and said on stderr with their parts |
| `--metrics FILE`, `--progress S` | JSON lines of what every reply costs, and progress lines while a long prompt is read (see [Watching a long prompt](#watching-a-long-prompt)); `--verbose` prints progress every 30 s |
| `--no-mcp` | refuse tools of type `mcp` in `/v1/responses`: no connection leaves the server on a request's word (below) |
| `--keep N` | conversations kept computed besides the one in use (8; 0: none), so that clients taking turns, or a client's requests on the side for titles and tags, do not have their conversations read again from the start |
| `--keep-disk GIB` | disk for them below the memory, in `~/.cache/janas`, so that they outlive the server: a long system message and tools are read once (8; 0: none; never more than a quarter of the space left). Written only when a conversation leaves the memory and when the server stops |
| `--keep-memory GIB` | the memory for them; by default half of what the expert cache and the margin left to other programs leave free at start, at least 256 MiB |
| `--store DIR` | where stored completions, responses and conversations are kept, so that they outlive the server (`~/.local/share/janas/server`) |
| `--no-store` | keep them in memory only |
| `--queue N` | requests allowed to wait (16) |
| `--max-body MIB` | the largest request body (32) |
| `--connections N` | open connections at most (64) |
| `--verbose` | a line per request on the standard error |

What it answers:

| Request | |
|---|---|
| `GET /v1/models`, `GET /v1/models/{model}` | the models it runs, with a `description` of how they run here; `DELETE` answers that a model read from a file is not deleted through the API |
| `POST /v1/chat/completions` | a conversation, the reply whole or streamed (`"stream": true`, server-sent events), with tools, JSON output, log-probabilities and more than one reply (below) |
| `GET`, `POST`, `DELETE /v1/chat/completions/{id}`, `GET /v1/chat/completions`, `.../messages` | the completions a client asked to keep (`"store": true`): read, listed with their `metadata`, changed, deleted |
| `POST /v1/completions` | raw text to continue, no chat format around it; several prompts, token ids, `echo`, `logprobs`, `best_of`, `suffix` |
| `POST /v1/responses` and the six operations under it | OpenAI's newer API: items in and out, `previous_response_id`, conversations, streamed events, `background` and `cancel`, `input_items`, `input_tokens`, `compact` |
| `/v1/conversations` and its seven operations | conversations the server keeps, and their items |
| `POST /v1/embeddings` | with `--embedding-model FILE`: the vectors of texts (below) |
| `POST /v1/moderations` | asked of the chat model (below) |
| `GET /health` | `{"status":"ok"}` |

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"messages": [{"role": "user", "content": "What is the capital of Sardinia?"}]}'
```

A client that lets you set the address of an OpenAI-compatible API should
need nothing else: the base URL is `http://127.0.0.1:8080/v1`, and the key
whatever you gave `--api-key` (anything, if you gave none). It has been tried
here with `curl`, its own test page, the `openai` Python SDK and Open WebUI
0.11.4 (driven through its own API: streamed chat, titles, tags, follow-up
suggestions, and a document indexed with Janas's embeddings and asked
about), and by a user with AnythingLLM.

**Models that reason, and the limits a client sets.** Qwen3 and its
successors reason before they answer, unless told not to. OpenAI's
`max_completion_tokens` (and `max_output_tokens` in the Responses API)
counts the reasoning too, and Janas takes it so. The older `max_tokens` of
a chat, which OpenAI does not take at all for the models that reason, is
taken for the answer alone: a client that sends it means the answer, and
counted with the reasoning a small limit was spent before a word of it was
written (Open WebUI asks for a chat's title with 1000, and got none).

**With Open WebUI**, start the server with `--think off`: the titles, tags
and suggestions it asks for after every reply are then quick, and a model
that reasons can still be asked to (`reasoning_effort`). Its built-in tools,
when they are on, add about 5,000 tokens to every prompt; the server reads
them once and keeps them computed, but the first reply of a conversation
waits for them.

**The conversation is the client's.** OpenAI's API keeps no state: every
request carries the whole conversation. `janas-server` keeps the one it
computed last, and reads again only what a request adds to it: a client that
sends the same messages plus a new one pays for the new one alone, and
`usage.prompt_tokens_details.cached_tokens` says how much was already
computed. The replies it wrote are remembered with their reasoning, so the
answer a client sends back - without the reasoning, as clients do - still
counts as the same turn. A conversation that differs earlier is read again
from where it differs, and from the start on models with a recurrent state
(Qwen3-Next, Qwen3.5 and 3.6).

**Reasoning** comes apart from the answer, in `reasoning_content`, whole or
streamed. `"reasoning_effort": "none"` turns it off for a request, any other
effort turns it on, and `"chat_template_kwargs": {"enable_thinking": false}`
works too; `--think` decides for requests that say nothing.

**Beyond OpenAI's fields**, which clients ignore: `top_k` and `min_p` in a
request, and in every reply's `usage` a `janas` object with the context used
and its size and the seconds spent reading the prompt and writing the reply.

**Tools.** `tools`, `tool_choice` (`none`, `auto`, `required`, a function)
and `parallel_tool_calls`, and the older `functions`. The tools are written
into the system message the way the model's chat template writes them, and a
call the model opens is held to a grammar of the functions and their
parameters until it closes it, so its arguments are always a JSON object
valid against the function's schema. The calls come back in `tool_calls`,
whole in one delta of the stream, and the reply ends with
`"finish_reason": "tool_calls"`; sent back with the tools' answers, they are
found again among the replies the server remembers, so the next turn reads
only the answers. Qwen models write calls in one of two ways, and the
template says which: as JSON (Qwen3, Qwen3-Next) or as XML (Qwen3.5 and 3.6,
Qwen3-Coder). **Both have been run here**: the JSON way on Qwen3-4B and
Qwen3-0.6B, the XML way on Qwen3.6-35B-A3B (two calls at once, their answers
sent back, 422 of 471 prompt tokens reused on the next turn) and on
Qwen3-Coder-Next, which describes the tools in a way of its own (two calls
at once with valid arguments, their answers used in the reply, 496 of 562
prompt tokens reused).

**MCP servers in the Responses API.** A tool of type `mcp` (`server_label`,
`server_url`, `headers`, `authorization`, `allowed_tools`,
`require_approval`) has the server reach that MCP server over HTTP or
HTTPS, as OpenAI's does. The response begins with an `mcp_list_tools` item
(once per conversation), the model sees the tools as
`server_label__tool`, and a call it writes either becomes an
`mcp_approval_request` that ends the response - `require_approval` is
`always` by default, as at OpenAI - and runs when the next request answers
it with an `mcp_approval_response`, or, where no approval is needed, runs at
once as an `mcp_call` item with its output, and the model goes on in the
same response (up to eight rounds). Streamed, the calls come as
`response.mcp_call.*` events. `allowed_tools` and the approval filters take
names or `{tool_names, read_only}`; `tool_choice` may name an MCP tool.
Connectors and tunnels are not served. A request makes the server open
connections to the address it names: `--no-mcp` turns that off for a
server others can reach. Tried with Qwen3-4B against the test server of
`tests/mcp_fake_server.c`: a call run at once, one approved on the next
request, and the same streamed; and through the `openai` SDK against
DeepWiki over HTTPS, as OpenAI's own examples do: its tool listed, called,
and the model answering from what it returned.

**JSON output.** `response_format` `json_object` or `json_schema` (and
`text.format` in the Responses API): the answer is held to a grammar token by
token, so it is valid JSON when the model ends it (not when `max_tokens` cuts
it). A schema is held to its types, the properties of an object in the
schema's order with the required ones always there and no others, array
items and bounds up to 32, string lengths up to 64, `enum`, `const`,
`anyOf`, `oneOf`, `$ref` within the schema and `nullable`; `pattern`,
`format` and numeric bounds are not enforced. The reasoning, where there is
one, stays free.

**Log-probabilities.** `logprobs` and `top_logprobs` (up to 20) in chat,
`logprobs` in completions, `top_logprobs` in responses. They are the
model's own: where a grammar forced a token the model found unlikely, its
log-probability says so.

**More than one reply** with `n` (and `best_of` in completions, which keeps
the best by mean log-probability); `presence_penalty`, `frequency_penalty`
and `logit_bias` as OpenAI defines them. The replies of `n` are written one
after the other on the same prompt.

**Embeddings** come from a second, small model opened with
`--embedding-model`, such as Qwen3-Embedding-0.6B: texts or token ids, one or
many, `dimensions` to cut the vector shorter, floats or `base64`. Against
llama.cpp on the same file the vectors of the same text agree to a cosine of
0.98 to 0.996, and the similarities between texts come out alike.

**Moderations** are asked of the chat model: at temperature 0 it answers
OpenAI's thirteen categories with a JSON object held to a schema, and each
score is the chance it gave `true` against `false` in that place. It is a
general model following instructions, not a classifier trained for the task.

**Kept on disk.** Stored completions, responses and conversations are
written, one file each, to `~/.local/share/janas/server` (or `--store DIR`),
and read back when the server starts again; the oldest go past a thousand of
each. The directory and its files are readable by their owner alone, since
they hold what the clients said, and one server at a time uses a directory.
`--no-store` keeps them in memory only, until the server stops. A response
running in the background when the server is killed is found as failed at
the next start. Images and sound in messages are refused with a 400 that
says so.

**Every other operation of OpenAI's API** - there are 345 in the version it
follows (2.3.0) - has a route that answers 501 and says why: not written
yet, needs a model of another kind (sound, images, video), trains models, or
belongs to the administration of OpenAI's service. The table is generated
from the specification by `tools/openapi_routes.py`, which also counts what is
done.

**One request at a time.** The model holds one sequence, so requests wait in
line and run in the order they came; when the line is full the answer is 429.
While a streamed request waits, the stream says where it stands in the line
(`: queue N`, a comment clients skip) and the test page shows it.
A client that closes the connection while a reply streams stops the reply.

**On the network.** It listens on this machine alone unless `--host` says
otherwise, and then it warns if there is no key. There is no HTTPS yet: across
a network you do not trust, put it behind something that adds it.

**The test page.** With `--test`, `http://127.0.0.1:8080/test` is
*Janas-Chat Web*: a chat in the browser that talks to the API like any client,
with the reasoning in grey, a line under each reply with the context used and
the speed (last, lowest, highest and mean of the conversation), and the
settings - system message, temperature, reasoning, key - behind *Settings*.
It reads the marks a model writes as `janas-chat` does, and also quotes,
tables, struck text and links; a link asks before it opens, because links an
AI writes can be wrong or unsafe. *Markdown sample* shows a text of ours with
every mark it knows, without asking the model. The page is inside the
program, loads nothing from anywhere else and may talk only to this server.

## Measuring your machine

```sh
bin/x86_64-linux/janas-bench qwen3-next.jns
```

Close what you can before running it: it measures the machine you give it, and
a browser with thirty tabs is part of the machine. It tries the configurations
the machine offers — how many threads, which cores,
with and without the GPU — on real passes of the model, keeps the fastest,
writes them into `~/.cache/janas` so the next start begins from them, and leaves
a report in the current directory. On the development laptop that report says,
among other things, that bringing the efficiency cores in makes decoding about
a tenth faster than the performance cores alone - and that for blocks of
several tokens, which is what a prompt and a draft check are, the engine often
wants fewer threads than for one.

## Tools

| | |
|---|---|
| `tools/gguf2jns` | GGUF → the `.jns` format the engine reads |
| `tools/hf2jns_mtp` | the multi-token prediction block, from the original checkpoint: its shards, or just the block's tensors from Hugging Face (`hf:<owner>/<repo>`) |
| `tools/jns_planes` | rewrites a model with the experts' down matrix in bit planes, so a machine short of memory can read part of it |
| `jns_check` | checks a model file, with `--verify` every expert's checksum |
| `tools/openapi_routes.py` | `janas-server`'s route table, from OpenAI's specification |
| `mcp_fake_server` | an MCP server over stdio in either era of the protocol, for trying a client without anybody else's server |
| `llm_eval` | compares the engine's logits against a reference dump |
| `bench_gemm`, `bench_attn`, `bench_long`, ... | the pieces measured on their own |

They are built into `bin/x86_64-linux/` along with everything else.

## Helping it run on more hardware

This is where help is worth most. The engine decides everything from what it
measures, but it has only ever measured **one** machine, so the decisions are
tuned to one shape of CPU, one disk and one GPU.

**The easy way: one command.** From a fresh clone, on Linux, with a C
compiler and `curl` and nothing else:

```sh
git clone https://github.com/prabanta-dev/janas && cd janas
./tools/janas-try.sh
```

It builds Janas and runs its tests, downloads a model from Hugging Face
(checking its SHA-256), converts it (checking the result against the
fingerprints of [MODELS.md](MODELS.md)), checks that the model's answer at
temperature 0 is the very text every other machine gets, measures the speed
with `janas-bench`, tries `janas-server`, and writes a report with no host
name, user name or paths in it. It asks before every long step and shows
the report before anything is sent. **Run it with the machine to itself**
(close the browser, builds, other models): before measuring it checks how
idle the CPU is and, if it is busy, offers to wait; the report says how idle
it was, in general terms only (a load average and a percentage, never which
programs ran). If you agree, it opens the issue with
`gh`, or gives you a link to a filled
[test report form](https://github.com/prabanta-dev/janas/issues/new?template=04-test-report.yml).
Three levels: `quick` (Qwen3-4B, 2.5 GB to download), `medium`
(Qwen3.6-35B-A3B, 22 GB) and `full` (Qwen3-Next-80B-A3B with its MTP block,
about 52 GB); it offers those your machine can hold, and deletes at the end
only what it downloaded, if you say so. `--help` lists the options.

**What to send, by hand.** Run the benchmark and attach the report it writes (a
`janas-bench-<date>.txt` in the current directory):

```sh
bin/x86_64-linux/janas-bench <model.jns>
```

The report already carries the CPU's name, the core layout, the GPU if there is
one, and the speed of every configuration tried. Add:

- the **distribution** and kernel (`uname -a`), and the **RAM** (`free -g`);
- the **disk** the model sits on (NVMe? SATA? over USB?);
- the **model** you used and its file size;
- what you saw: numbers that look wrong, a reply that made no sense, a crash, a
  machine that went unresponsive — with what you were doing at the time.

The [hardware report form](https://github.com/prabanta-dev/janas/issues/new?template=01-hardware-report.yml)
asks for exactly that, in that order. There are two more forms, for
[something that went wrong](https://github.com/prabanta-dev/janas/issues/new?template=02-something-went-wrong.yml)
and for [a model that will not convert](https://github.com/prabanta-dev/janas/issues/new?template=03-a-model-will-not-work.yml).

**What is most wanted, in order.**

1. **CPUs without AVX-VNNI**, and **AMD** — the arithmetic has a path for them
   and it is the least exercised of all.
2. **Discrete GPUs** (NVIDIA, AMD, Intel Arc). The GPU support is written and
   correct, bit for bit, but on an integrated GPU it gains nothing: whether it
   pays on a real card is unknown. **Careful:** a GPU driver reset takes the
   desktop with it. Start with `janas-bench`, not with a long chat.
3. **Machines with 16 GB or less.** The engine is supposed to trade quality for
   memory on its own — fewer bits per weight, fewer experts per token — and the
   thresholds for that were measured on 32 GB.
4. **Slower disks.** Everything assumes an NVMe SSD; on a SATA disk the engine
   should still work and simply wait more, but nobody has watched it do so.
5. **Other models** of the supported families. A file that will not convert, or
   converts and then answers nonsense, is a useful bug.

**What not to bother with yet:** Windows and macOS (not supported), and models
outside the Qwen3 MoE families (they are refused with a message).

Patches are welcome too, but a measurement from a machine nobody here can buy is
worth more than most patches.

## Licence

Janas is free software under the **GNU General Public License, version 3 or
later** — see [LICENSE](LICENSE). Every source file carries its SPDX line.

`janas-server`, and only it, has two libraries of others compiled in, their
sources unchanged in [`src/third_party/`](src/third_party) with their
licences and where they came from: **GNU libmicrohttpd** 1.0.10 (LGPL 2.1 or
later, or the eCos licence without HTTPS, as it is built here) and **yyjson**
0.13.0 (MIT). `libjanas_mcp`, and the programs with it, have **Mbed TLS**
4.2.0 with TF-PSA-Crypto (Apache-2.0 or GPL-2.0-or-later), for `https://`
MCP servers. `libjanas_llm` has none.

**The models are not covered by it.** A converted `.jns` file is a derivative of
the model, not of Janas, and keeps the licence its authors gave it;
[MODELS.md](MODELS.md) says what this project will and will not redistribute,
and what to check before you pass a converted model on.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md). The short version: a measurement from a
machine nobody here owns is worth more than most patches, and every commit needs
a `Signed-off-by` line (`git commit -s`).
