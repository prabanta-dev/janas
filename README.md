# Janas

[![Status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)]() [![Platform: Linux x86-64](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey.svg)]() [![LinkedIn](https://img.shields.io/badge/LinkedIn-Maurizio%20Cammalleri-0077B5?logo=linkedin)](https://www.linkedin.com/in/maurizio-cammalleri-80a89a11/) [![Substack](https://img.shields.io/badge/Substack-Maurizio%20Cammalleri-FF6719?logo=substack)](https://cammalleri.substack.com/)

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

Janas is a family of tools for artificial intelligence, written in C with no dependencies beyond the C library. Its first component, **Janas-LLM**, runs large mixture-of-experts language models on ordinary computers, with or without a GPU: the weights stay on disk, the experts each token needs are streamed from an NVMe SSD, and the ones that keep coming back stay in memory. An 80-billion parameter model runs, and answers, on a laptop with 32 GB of RAM.

Designed and written by **Maurizio "camauri" Cammalleri**.

---

## ⚠️ Read this first

**This is an experiment, not a product.**

| | |
|---|---|
| **Stage** | Early development. Formats, file layouts, the public API and the command line **change without notice**, and have changed several times a week. |
| **Stability** | **Not verified.** The engine is deliberately hard on the machine: it fills the free memory with its expert cache and reads the disk at full speed with `O_DIRECT`. On a machine with less headroom than it reckons, the desktop can be pushed into swap and become slow or unresponsive. Using a GPU has, once, caused a driver reset that took the desktop with it. |
| **Data** | It writes only in `~/.cache/janas` (what it learns about your machine, and the system messages it has read, so as not to read them again) and where you tell it to put a converted model. It does not touch the model files it reads. |
| **Hardware** | Developed and run on **one** machine (below). On anything else it is untested: it may be slower, it may refuse the model, it may misbehave. |

If something goes wrong, `Ctrl-C` stops a reply and `Ctrl-D` leaves the chat; `--cache <GiB>` puts a hard limit on the memory it takes, and `JANAS_GPU=0` keeps it off the GPU entirely.

## The machine it grew on

Everything in this repository was written and measured on:

- **Linux Debian 13**, x86-64
- **Intel Core Ultra 9 185H** (6 performance cores, 8 efficiency, 2 low-power; AVX2, FMA, F16C, AVX-VNNI; integrated Arc GPU)
- **32 GB** of DDR5 RAM
- **1 TB NVMe SSD** (about 5.7 GB/s reading)

Everything the engine does is chosen from what it measures, so another machine will make other choices — that part is meant to travel. What has never been tried is the rest: other CPUs (especially without AVX-VNNI, or AMD), discrete GPUs, slower disks, more or less memory. **If you run it somewhere else, the report from `janas-bench` is the single most useful thing you can send back** (see [Helping it run on more hardware](#helping-it-run-on-more-hardware)).

## What works now

- **Models:** Qwen3.5 and Qwen3.6 (`qwen35moe`, e.g. Qwen3.6-35B-A3B), Qwen3-Next (`qwen3next`, e.g. Qwen3-Next-80B-A3B-Instruct and Qwen3-Coder-Next), Qwen3 MoE (`qwen3moe`, e.g. Qwen3-30B-A3B), the **dense** Qwen3.5 models (`qwen35`) and the **dense** Qwen3 models (`qwen3`), from the usual **Q4_K_M** GGUF files, or **Q3_K**, **IQ3_S**, **IQ4_NL**, **IQ4_XS**, **Q4_0**, **Q4_1** and **Q5_1** ones, and unsloth's "UD" mixes of them (checked on Qwen3.5-0.8B and 9B against llama.cpp). A dense feed-forward is a mixture with one expert and no router, so the same path serves both and there is no second engine to keep in step. Of the dense ones **Qwen3.5-0.8B, 2B and 9B, Qwen3-4B and Qwen3-0.6B are the ones that have been run here**; the Qwen3.5 three were checked against llama.cpp on the same files. **Nearly everything here was measured on Qwen3-Next-80B-A3B-Instruct**: it is the one the engine was shaped around, and the one whose numbers are quoted below. The others have been converted, checked against a reference and used, but far less.
- **Gemma 4** (`gemma4`): **Gemma-4-12B-it** (dense) and **Gemma-4-26B-A4B-it** (a mixture of 128 experts beside a dense feed-forward) convert and chat, with their reasoning on or off (`--think`); the prompts are, token for token, the ones the model's own chat template writes, and a full context slides as it does for the other models. Its sliding-window layers keep only the last positions they can see, so a long context costs mostly the full layers: for the 12B about 16 KB a token and half a gigabyte fixed, instead of 175 KB a token. Keys and values are kept at sixteen bits (Gemma is far more sensitive to them than Qwen). Tools work in Gemma's own notation - declarations, calls and the tools' answers inside the model's turn, token for token as its template writes them. The small **Gemma-4-E2B-it** and **E4B-it** run too, with chat and tools: their per-layer embeddings - a table as large as half of E2B's file - are read from the file a row at a time and never loaded, so E2B ran here in 1.5 GB of memory (a 3.1 GB file) and E4B in 3.1 GB (5.0 GB). Each of the four has an **assistant**, a small drafting model of its own that reads the main model's keys and values, taken as its `--mtp` file: on an Italian chat reply of 110 tokens, greedy, it took E2B from 34.7 tokens/s to 51.0, E4B from 18.3 to 26.3, the 12B from 8.5 to 12.9 and the 26B from 16.9 to 24.5 on the development laptop, the same tokens with and without it. Against llama.cpp on 200 tokens of prose the most likely token agrees at 160 positions on the 12B (mean logit difference 0.84) and at 160 on the 26B (0.97); on another 200-token text at 191 on E2B (0.45) and 195 on E4B (0.28). Against an exact float64 computation over the same weights (a NumPy forward pass written to check them), on the prose: the 12B agrees at 167 positions (llama.cpp 173), the 26B at 156 (llama.cpp 160), with mean errors of 0.75 and 0.87 (llama.cpp 0.67 and 0.90). The larger two turn small numerical differences into large ones, and llama.cpp is not a steady reference for them: against itself with an eight-bit cache it agrees at 153 positions on the 12B, and with a 32-bit cache and no flash attention, at 112. The reference has to be computed in batches of at most 32 tokens: with larger ones llama.cpp's logits for the early positions change (it looks ahead), for a reason not yet found.
- **Reasoning models:** the reasoning is shown apart, in grey, and can be turned off.
- **`janas-chat`:** a terminal chat with a line editor of its own, history, colours, a status line and a context that slides instead of ending.
- **`libjanas_llm`:** a C library ([`include/janas/llm.h`](include/janas/llm.h)) and a FreeBASIC binding ([`include/janas/llm.bi`](include/janas/llm.bi)).
- **Tools from MCP servers** in `janas-chat`: the servers configured as the other clients configure them, each call shown and confirmed before it runs ([below](#tools-from-mcp-servers)). The client is a library of its own, **`libjanas_mcp`** ([`include/janas/mcp.h`](include/janas/mcp.h), [`include/janas/mcp.bi`](include/janas/mcp.bi)).
- **`janas-mcp`:** the other way round, the model offered as tools to MCP clients - a question to it, the embedding of a text ([below](#janas-as-an-mcp-server)).
- **`janas-flights`:** where a flight is and what flies over a sea or a place, as MCP tools, from the ADS-B receivers of the community, and today's schedules with a free AviationStack key: the first of Janas's services.
- **`janas-weather`:** the weather now, with what the nearest station measured, the forecast, the sea and its waves, and the warnings in force (in Italy the Civil Protection's levels), from free sources with no key.
- **`janas-maps`:** roads and places: a route by car, bike or on foot, step by step in your language, journeys by public transport, what is near (fuel, a pharmacy, a cash machine...), where an address is; OpenStreetMap's data through free services with no key.
- **`janas-prices`:** prices and finance, for information: currencies at the ECB's rates, crypto-assets, shares (with a free key of yours), inflation, central banks' rates, electricity where it is open data, the cheapest fuel near a place in Italy, France, Spain and Austria.
- **`janas-github`:** a project on GitHub: its issues and pull requests (the new ones since you last asked), its latest releases, its recent commits, what it is; it only reads, with the gh command's token when you have one.
- **`janas-git`:** the git repositories of your computer: what has changed, the commits, the branches, a diff; and a commit, pull, push or switch of branch, each shown whole and asked of you first; nothing that throws work away.
- **`janas-quakes`:** the earthquakes: the latest near a place or where you are, the strongest in the world, one in detail, from INGV (Italy and its seas) and the USGS.
- **`janas-system`:** this computer: how it is (memory, processor, disks, battery, temperatures), what fills a directory, the programs running, the errors of its logs, the updates, the network; it only reads, and none of its tools reaches the network.
- **`janas-wiki`:** Wikipedia when you ask for it: its search, a page shown to you as it is (the introduction, then the section asked for) while the model reads only a note of it, and the pages about what lies around a place.
- **The services, each with its page** - what it answers, its sources, its keys, how to use it in Claude Code and other clients: [src/services](src/services/README.md).
- **`janas-server`:** the model behind an HTTP API that follows OpenAI's, so that clients written for it can use a model running on this machine: chat and completions with tools, JSON output held to a schema and log-probabilities, the Responses API with its conversations, embeddings and moderations, a chat page of its own to try it from a browser ([below](#using-it-over-http)), and code completion for VSCodium and VS Code ([docs/code-completion.md](docs/code-completion.md)).
- **`janas-bench`:** measures the machine, fills its profile and writes a report.
- **Faster replies** from drafts the model verifies, so the text is exactly the one it would have written: from the model's own multi-token prediction block where it has one, from the conversation where it does not, or from a small model of the same family given with `--draft`.
- **Optional GPU** (Vulkan 1.3, integrated or discrete), used only where the engine measures that it helps, for every quantized type the CPU reads. A reply's tokens are identical with and without it; a prompt is read with the fast prompt (next).
- **The fast prompt**, on by default since 10 October 2026: a prompt's blocks go to the GPU's tiled products (Q4_K, Q5_K, Q6_K, Q8_0, the experts of a mixture in one launch per layer, f32 and bf16 weights), and on the dense Qwen3 and Qwen3.5 models the whole block - every layer, attention, Qwen3.5's recurrent layers - goes to the GPU in one submission while the CPU rests: on a laptop the CPU working beside the GPU takes the power the GPU would use. They round the sums group by group, so a prompt is not read to the bit as the CPU reads it; against the references the most likely token agrees as often as the exact way (within 2-3 of 256 on every model measured, the mean difference of the scores within 2%). On the development laptop Qwen3-4B reads the table's prompt (below) at 220 tok/s with it and at 85 without the GPU, and every model compared reads faster than llama.cpp does on the same GPU. `JANAS_GPU_FAST_PROMPT=0` reads them the exact way.
- **Experts read ahead** when the expert cache is too small for the model: the next layer's router is applied early and the experts it picks that are not in memory are read from disk while the current layer computes. It turns itself on only when the cache misses many experts a token (Qwen3-Next-80B with a 4 GiB cache: +6%) and changes no result. A long prompt is read the same way, block by block: the next layer's router is applied to the whole block and the experts most of its tokens will want are read while the current layer computes; on after a block that missed many experts. Qwen3-Next-80B reading 4,096 tokens: 36.8 to 42.0 tokens/s with a 4 GiB cache, 48.5 to 50.0 with 16 GiB (the 256 tokens written after it 1.4% slower, the cache holding other experts); Qwen3-30B-A3B, whose experts all fit, even. `JANAS_PREFETCH_PREFILL=0` turns it off.
- **Light experts not in memory are left out** when the expert cache holds less than a fifth of the experts: an expert the router weighs below 0.8/k of a token (0.95/k under a tenth), and that would have to be read from disk, is not read and not computed, and the token's other experts take its share — the cache-conditional routing of Skliar et al. (TMLR 2025). Until 8 October 2026 the engine dropped the lightest experts whether in memory or not (six of ten, or eight). Qwen3-Next-80B, four bits, 400 tokens against llama.cpp, most likely token agreeing and tokens/s: with a 2 GiB cache all ten 393 at 11.0, six 383 at 20.9, now 380 at 26.4; with 4 GiB 393 at 20.7, six 383 at 27.6, now 384 at 27.7, with the mean logit difference 0.36 against 0.53; on Italian prose with 8 GiB 382 at 18.7, eight 361 at 21.4, now 372 at 23.7. On Qwen3-30B-A3B and Qwen3.6-35B-A3B with caches of 9% and 18% of their experts (1.5-3.5 GiB), on code and Italian prose, it was faster than the old way every time, by 9-37% (Qwen3-30B-A3B with 3 GiB: 28.7 tokens/s against 24.4, 381 tokens agreeing of 400 against 377), with a mean logit difference as low or lower in seven of the eight; it agreed on fewer most likely tokens in two (Italian, 64 tokens: 47 against 54 on the 30B with 3 GiB). `/experts n` (or `0`) asks for a number of experts and turns it off.
- **Every weight type near the memory's speed**, and the same results to the bit on every path (AVX2, AVX-VNNI, GPU) for a reply's tokens (and for a prompt with `JANAS_GPU_FAST_PROMPT=0`). `tests/bench_kernels` measures each type's product on its own: writing a token, on 12 threads of the development laptop, Q8_0 84 GB/s, Q6_K 73, Q4_K 69, the other 4- and 5-bit types 59-68, Q3_K 61 (48 until 9 October 2026, when its unpacking went through the stack), IQ3_S 25, against the 81 GB/s the memory gives; and a token of Qwen3-Next-80B-A3B, Qwen3.6-35B-A3B, Qwen3-30B-A3B or a 27B, phase by phase, reads its weights at 59-62 GB/s on average. Until 6 October 2026 the Q3_K, IQ4_XS and IQ3_S products were six to thirteen times slower than that (their kernels converted a scale by calling a function), which made unsloth's UD files, which mix those types in, write at half their speed: Qwen3.8-27B UD-Q4_K_XL went from 1.7 tokens/s to 3.6.

On the machine above, with nothing else running, `janas-bench` reports this (measured on 10 October 2026, with the fast prompt the engine uses by default since that day):

| Model | reading a prompt | writing a reply | with drafts, in a chat |
|---|---:|---:|---:|
| Qwen3.5-2B (dense, 1.3 GB) | 395 tok/s | 50 tok/s | **67 tok/s** |
| Qwen3-4B (dense, 2.5 GB) | 220 tok/s | 28 tok/s | see `--draft` below |
| Qwen3.5-9B (dense, 5.7 GB) | 103 tok/s | 14 tok/s | **24 tok/s** |
| Qwen3-30B-A3B (18.6 GB) | 133 tok/s | 34 tok/s | — |
| Qwen3.6-35B-A3B (22.3 GB) | 86 tok/s | 25 tok/s | **34 tok/s** |
| Qwen3-Next-80B-A3B (48.4 GB) | 64 tok/s | 22 tok/s | **37 tok/s** |
| Gemma-4-E2B-it (3.1 GB) | 211 tok/s | 38 tok/s | **52 tok/s** |
| Gemma-4-E4B-it (5.0 GB) | 133 tok/s | 20 tok/s | **32 tok/s** |
| Gemma-4-12B-it (dense, 7.1 GB) | 71 tok/s | 10 tok/s | **14 tok/s** |
| Gemma-4-26B-A4B-it (18.3 GB) | 90 tok/s | 18 tok/s | **26 tok/s** |

Every figure is the **mean of three rounds**, not a best round, of the configuration the engine itself chose for each kind of pass - the one a chat runs: a prompt of 256 tokens, then 64 written. Here the integrated GPU reads every prompt (the fast prompt, below), and the threads the engine settled on were the performance cores' 12 for a prompt and for a single token on nearly every model, all the usable cores (performance and efficiency, not the low-power ones) for a few tokens at once. On another machine the engine may settle on other threads: the first passes of each kind try every core, one thread per performance core and the performance cores' threads, keep the fastest, and later try the others now and then, so that a machine that warms up can change the verdict (on an i9-14900HX every thread replied at less than half the speed of the performance cores' 16). The rounds run in alternating order so that the machine warming up and the cache filling weigh on each setting alike. Expect a few per cent either way between runs, and rather less than these on a first run, while the cache is still filling. Without the GPU (`JANAS_GPU=0`) Qwen3-4B reads the same prompt at 85 tok/s and writes at 27.

The last column is a draft guessing the next tokens and the model checking them, which is the same text at a higher rate — the whole of it is explained under [`--mtp`](#using-the-chat): the Qwen3.5 and Qwen3.6 models' and Qwen3-Next's own prediction block, Gemma 4's assistant (a small drafting model of its own). It is measured on a chat reply: an ordinary question in English ("explain how bread is made at home"), 64 tokens of the answer, sampled as the chat samples. **What the draft is worth depends on the text**: there the drafts survived 74-89% of the time on the Qwen models and Gemma's two larger ones, 63-72% on Gemma's E2B and E4B; on Italian prose Qwen3.5-2B kept about a third of them, and gained next to nothing. One round of 64 tokens is noisy here, a tenth either way between runs. Qwen3-30B-A3B has no such block, nor have the dense Qwen3; the dense Qwen3.5 have one in their original checkpoint, which `hf2jns_mtp` converts ([MODELS.md](MODELS.md#the-fingerprints)).

**This column was wrong until 26 September 2026**, and higher: it measured the block continuing a paragraph that repeated itself, which a model copies and the block guesses almost every time - 107 tok/s for Qwen3.5-2B, 26 for Qwen3.5-9B. The MoE figures of the time (29 and 30) happened to be lower than today's; the ChangeLog has the details.

For a point of reference, llama.cpp (build `2b18470` of 18 September 2026) with its Vulkan backend on the same integrated GPU, everything on it (`llama-bench -p 1024 -n 16 -ngl 99 -t 12`), against `janas-bench --prompt 1024 --gen 16` on the same files the same days (9 and 10 October 2026; llama.cpp's best configuration, Janas's own choice):

| Model | reading 1,024 tokens: Janas | llama.cpp | writing: Janas | llama.cpp |
|---|---:|---:|---:|---:|
| Qwen3.5-2B | 441 tok/s | 323 tok/s | 52 tok/s | 28 tok/s |
| Qwen3-4B | 193 tok/s | 164 tok/s | 27 tok/s | 18 tok/s |
| Qwen3.5-9B | 111 tok/s | 78 tok/s | 14 tok/s | 8.7 tok/s |
| Qwen3-30B-A3B | 135 tok/s | 122 tok/s | 32 tok/s | 15 tok/s |
| Gemma-4-E2B-it | 191 tok/s | 150 tok/s | 38 tok/s | 19 tok/s |
| Gemma-4-E4B-it | 122 tok/s | 109 tok/s | 20 tok/s | 15 tok/s |
| Gemma-4-12B-it | 65 tok/s | 46 tok/s | 9.9 tok/s | 7.1 tok/s |
| Gemma-4-26B-A4B-it | 84 tok/s | 76 tok/s | 18 tok/s | 12 tok/s |

Janas's figures are one round each; the writing column is without drafts on both sides. The two tools do not measure in exactly the same way (`llama-bench` writes without a prompt before it; `janas-bench` after the 1,024 tokens), and it is one machine. llama.cpp's own CPU backend did worse on this laptop than its Vulkan one at reading (Qwen3-4B 70 tok/s on 20 threads) and as well at writing.

**Try it on an idle machine first.** The expert cache takes the memory that is free when the model is opened, and what is left of the model is read from disk while it answers, so everything else running takes its share. It degrades gently rather than breaking: measured with a browser and an editor open, the same model kept about three quarters of its speed and waited five times longer on the disk. But the first thing you see should be the machine's real speed, and any measurement you mean to send to others has to be taken with the machine to itself.

## Documentation

- [INSTALL.md](INSTALL.md) — what the build needs, how to prepare a model, and the first run
- [ChangeLog.md](ChangeLog.md) — what changed, newest first
- [MODELS.md](MODELS.md) — what a converted model's licence is, and what may be redistributed
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to help, and the sign-off
- [src/services](src/services/README.md) — Janas's services, each with its page, in `janas-chat` and in other MCP clients
- [docs/code-completion.md](docs/code-completion.md) — code completion in VSCodium and VS Code, through `janas-server`
- [AUTHORS](AUTHORS) — who wrote it, and what it owes to others
- [LICENSE](LICENSE) — GNU GPL, version 3 or later

## Getting started

```sh
sudo apt install build-essential          # Debian/Ubuntu
sudo apt install libvulkan-dev glslang-tools vulkan-tools   # optional, GPU
sudo pacman -S vulkan-headers glslang vulkan-tools           # the same, Arch

./build.sh                                # everything into bin/x86_64-linux/
./build.sh release test                   # and run the tests
```

## Getting a model

**The short way: `janas-get`.** One command downloads a model from Hugging Face, checks the SHA-256 of every file against [MODELS.md](MODELS.md), converts it and makes its prediction file beside it (an MTP block, Gemma 4's assistant, or for Qwen3-4B the small Qwen3-0.6B that guesses its tokens), which the chat then finds by itself. The models go to `~/.local/share/janas/models`, or wherever `JANAS_MODELS` or `--dir` says:

```sh
bin/x86_64-linux/janas-get list                  # the models it knows
bin/x86_64-linux/janas-get gemma-4-e4b           # into ~/.local/share/janas/models
bin/x86_64-linux/janas-chat ~/.local/share/janas/models/gemma-4-e4b-it-q4km.jns
```

A download stopped halfway resumes where it stopped, a file already there is checked rather than fetched again, and the GGUF goes once converted (`--keep` keeps it). Any other GGUF: `janas-get hf:<owner>/<repo>/<file.gguf>`, checked against the SHA-256 Hugging Face lists for it. The memory column of `janas-get list` is the smallest machine each model is meant for. The rest of this section is the same by hand.

Janas reads its own format, `.jns`, converted from a **Q4_K_M** GGUF (Q3_K, IQ3_S, IQ4_NL, IQ4_XS, Q4_0, Q4_1 and Q5_1 convert too, and so do unsloth's "UD" files made of them; IQ2 and IQ1 types, not yet). Keep both in `models/` next to the sources — the tools take a path, so anywhere works, but that is where these examples put them, and where the project keeps its own.

| Model | GGUF converted here | GGUF | `.jns` | Memory it likes |
|---|---|---|---|---|
| **Qwen3-Next-80B-A3B-Instruct** — most of this engine was measured on it | [Qwen](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF) | 48.4 GB | 48.4 GB | 32 GB |
| **Qwen3.6-35B-A3B** — the easiest to start with | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) | 22.3 GB | 22.3 GB | 16-32 GB |
| **Qwen3-Coder-Next** | [Qwen](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) | 48.4 GB | 48.4 GB | 32 GB |
| **Qwen3-Coder-30B-A3B-Instruct** — for code completion in the editor ([docs/code-completion.md](docs/code-completion.md)) | [unsloth](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 18.6 GB | 18.6 GB | 32 GB, to keep it whole in memory |
| **Qwen3-30B-A3B** | [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF) | 18.6 GB | 18.6 GB | 16 GB |
| **Qwen3-30B-A3B with an importance matrix** (`qwen3-30b-a3b-imatrix`) — bartowski's Q4_K_M: the nearest to Qwen's own Q8_0 of the three here (400 tokens: code 388, Italian 367, against 376 and 335 for the Q4_K_M above), 8-10% slower held in memory: it keeps more matrices in Q6_K | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF) | 18.6 GB | 18.6 GB | 16 GB |
| **Qwen3-30B-A3B in three bits** (`qwen3-30b-a3b-q3`) — unsloth's dynamic UD-Q3_K_XL: against Qwen's own Q8_0 it agrees more often than the Q4_K_M above (400 tokens: code 386 against 376, Italian 360 against 335), with 27% fewer bytes of experts; with the expert cache held at 2 GiB on this 32 GB machine, 19.8 tokens/s against 13.6; held whole in memory, as fast (32.0-32.2 tokens/s against 32.1-32.2) | [unsloth](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) | 13.8 GB | 13.8 GB | 8-16 GB |
| **Qwen3.5-9B** — dense; its prediction block from the checkpoint ([MODELS.md](MODELS.md#the-fingerprints)) | [unsloth](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | 5.7 GB | 5.7 GB | 16 GB |
| **Qwen3.5-2B** — dense, small and quick; a prediction block too | [unsloth](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | 1.3 GB | 1.3 GB | 8 GB |
| **Qwen3-4B** — dense, and the smallest here that answers well | [Qwen](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | 2.5 GB | 2.5 GB | 8 GB |
| **Qwen3-4B with an importance matrix** (`qwen3-4b-imatrix`) — bartowski's Q4_K_M, the same weight types and size as Qwen's: against Qwen's own Q8_0 (400 tokens) code 381 against 364, Italian 351 against 329, at the same speed | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF) | 2.5 GB | 2.5 GB | 8 GB |
| **Qwen3-0.6B** — not to talk to: to draft for a dense one (`--draft`) | [unsloth](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) | 0.40 GB | 0.40 GB | with the model above |
| **Qwen3-Embedding-0.6B** — not to talk to: for embeddings (`janas-server --embedding-model`); published in Q8_0 and converted as it is | [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) | 0.64 GB | 0.64 GB | beside the chat model |

Qwen3 has other dense models - 1.7B, 8B, 14B, 32B - which declare the same architecture as the 4B and so should convert and run the same way. **None of them has been tried here**, and until one is, that is a reading of the file and not a claim. (Qwen3-30B-A3B is not one of them: the number is close but it is a mixture of experts, with an architecture of its own.)

Any other Q4_K_M of the same models converts and runs just as well, including the "dynamic" ones that give different tensors different types: the engine reads Q4_K, Q5_K, Q6_K and Q8_0, and refuses a file carrying anything else rather than guessing. The four above are the files the numbers in this repository were measured on, and the ones [MODELS.md](MODELS.md) gives fingerprints for, so that a converted file can be checked against the one measured here without downloading anything.

All four models are published under the Apache 2.0 licence by Alibaba Cloud, and quantized into GGUF by them or by the people above. Their licence is theirs, not this project's: [MODELS.md](MODELS.md) says what that means if you pass a converted file on to somebody else.

**Download.** Nothing needs installing — one file, resumable:

```sh
mkdir -p models/gguf && cd models/gguf
curl -L -C - -O \
  https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
cd ../..
```

With `pip install huggingface_hub` you can instead fetch a whole quantization, which is handier when it is split into several files, as Qwen3-Coder-Next is:

```sh
huggingface-cli download Qwen/Qwen3-Coder-Next-GGUF \
    --include "Qwen3-Coder-Next-Q4_K_M/*" --local-dir models/gguf
```

**Convert**, with the converter the build made (nothing else to install). With a model split into several files, give the first one and the rest is found:

```sh
bin/x86_64-linux/gguf2jns models/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    models/qwen3.6-35b-a3b-flat.jns
```

It says what it is doing, and refuses a model it does not know with a clear message rather than half a conversion.

**Cut the experts into bit planes.** Worth it on any machine that cannot hold all the experts in memory, which is the case this engine is built for: the down matrix is stored as three planes of two bits, so the engine can read four bits of it, or two, instead of six, and decide which when the model is opened. Nothing is lost — all three planes give the weights the file came with, bit for bit. This is the form the models here are kept in:

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

Both steps are deterministic, so the file can be checked against the fingerprints in [MODELS.md](MODELS.md).

Room needed: three copies of the model at the worst moment — the GGUF, the converted file and the one with the planes — so about 145 GB for the largest. The flat file goes as soon as the planes are written, and the GGUF can go too, though keeping it saves the download the day a new version of the converter is worth running.

**The multi-token prediction block** (Qwen3-Next and the dense Qwen3.5; optional, faster replies). Its own model carries it, GGUF files leave it out, so it comes from the original checkpoint. `hf2jns_mtp` can read just the block's tensors from Hugging Face, with range requests, instead of the shards that hold them:

```sh
bin/x86_64-linux/hf2jns_mtp models/qwen3-next.jns \
    hf:Qwen/Qwen3-Next-80B-A3B-Instruct models/qwen3-next-mtp.jns
```

For the dense Qwen3.5-9B that is 487 MB instead of 14 GB of shards, in under a minute. It also takes the shards themselves, when you have them on disk (`hf2jns_mtp <model.jns> <shard>... <out.jns>`). Qwen3.6-35B-A3B needs none of this: its GGUF keeps the block.

**Gemma 4** comes from unsloth's GGUF files, converted in one step, and so does its assistant (the `mtp-gemma-4-*.gguf` file beside the model, under 0.5 GB), which serves as the multi-token prediction file:

```sh
huggingface-cli download unsloth/gemma-4-E4B-it-GGUF \
    --include "gemma-4-E4B-it-Q4_K_M.gguf" "mtp-gemma-4-E4B-it.gguf" \
    --local-dir models/gguf
bin/x86_64-linux/gguf2jns models/gguf/gemma-4-E4B-it-Q4_K_M.gguf \
    models/gemma-4-e4b.jns
bin/x86_64-linux/gguf2jns models/gguf/mtp-gemma-4-E4B-it.gguf \
    models/gemma-4-e4b-mtp.jns
```

The same for E2B, the 12B (`gemma-4-12b-it-Q4_K_M.gguf`) and the 26B (`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`); the fingerprints are in [MODELS.md](MODELS.md#the-fingerprints). E2B's per-layer table, half of its file, is read from the file a token at a time and never loaded: E2B ran here in 1.5 GB of memory, E4B in 3.1 GB.

**And chat:**

```sh
bin/x86_64-linux/janas-chat models/qwen3-next.jns --stats
bin/x86_64-linux/janas-chat models/gemma-4-e4b.jns --stats
```

The chat finds the multi-token prediction file by itself: of the `.jns` files beside the model, the one whose header fits it (an MTP block of the same architecture and width, or Gemma 4's assistant for a model that wide and with its vocabulary), and says which it took. `--mtp <file>` names one; `--no-mtp` goes without. `janas-server` does the same, and programs can ask for it with `janas_llm_find_mtp()`.

The first start reads the model's resident weights (a couple of GB) and then fills the expert cache in the background, while you read and type. The second reply of a session is the one that shows the machine's real speed.

[INSTALL.md](INSTALL.md) has the details, including the other distributions and what to do when the GPU is not found.

## Using the chat

```
janas-chat [model] [options]
```

The model is a `.jns` file, or the name of a model of the catalog (`janas-get list` shows them): `janas-chat qwen3-30b-a3b` looks for it in `~/.local/share/janas/models` (or `JANAS_MODELS`, or `--models`), and if it is not there yet offers to download and convert it, with its prediction file or draft, through `janas-get`. With no model at all the chat lists the catalog, marks what is already there, and asks which one. Qwen3.5-0.8B is left out of that list: it is a test model of the Qwen3.5 architecture, kept for the engine's checks (it is converted in eight weight types), not for chatting; asked on 8 October 2026 for twenty actions (the weather, aircraft overhead, Wikipedia), it called the right tool 7 times out of 15 and made facts up, where Qwen3.5-2B, the small model for such work, called it 14 times. It still opens by name, with a word of warning. The first download proposes the folder and takes another if you type one; janas-chat then finds the models there only when told (`--models` or `JANAS_MODELS`).

The settings come in layers, each over the one before: the model's profile in the catalog, the measure of the model on this machine, the chat's own file (`~/.config/janas/chat.conf`), the model's file (`~/.config/janas/models/<name>.conf`), the command line. `/save-config` writes the model's file (`/save-config global` the chat's), `/del-config` deletes it, and the catalog itself is never written. A model of the catalog has two profiles: `quality`, the default, where the answers are the ones the file gives; and `fast` (`--profile fast`), which adds the levers measured to make that model faster at a small, measured cost to its answers (see below). A model whose catalog entry names a draft gets it by itself: Qwen3-4B guesses with Qwen3-0.6B.

**The fast profiles, as measured.** Every lever that trades a little accuracy for speed — `--head 4`, `--attention fast`, `--bits 4` where the experts are in planes — was tried on every model of the catalog, alone and in every combination, on 8 October 2026 on the development laptop; a lever is in a model's profile when it adds 2% or more. Speed: 128 tokens after a prompt of 1,900 (`tests/bench_long`, two runs each); for the three models larger than their 16 GiB cache, a fixed text read token by token instead (`tests/llm_eval`), because each lever changes the text a greedy reply takes and with it the experts read from disk. Agreement: the most likely token against llama.cpp, Italian prose / C source, as in the file → with the profile.

| Model | Fast profile | Faster by | Agreement |
|---|---|---|---|
| Qwen3-0.6B | `--head 4 --attention fast` | 14.3% | 345 → 331 / 377 → 368 of 400 |
| Qwen3-4B | `--head 4 --attention fast` | 6.8% | 379 → 369 / 397 → 388 |
| Qwen3.5-0.8B | `--attention fast` | 2.0% | 387 → 386 / 386 → 387 |
| Qwen3.5-2B | `--head 4` | 10.2% | 376 → 352 / 388 → 375 |
| Qwen3.5-9B | `--head 4` | 5.1% | 383 → 372 / 393 → 391 |
| Qwen3-30B-A3B | `--head 4 --attention fast --bits 4` | 10.0% | 57 → 55 of 64 / 392 → 385 |
| Qwen3-4B with an importance matrix | `--head 4 --attention fast` (9 October) | 5.7% | 351 → 350 / 381 → 376, against Qwen's Q8_0 |
| Qwen3-30B-A3B with an importance matrix | `--head 4` (9 October; `--bits 4` added 1.6%, `--attention fast` 0.6%) | 3.2% | 367 → 354 / 388 → 387, against Qwen's Q8_0 |
| Qwen3-30B-A3B in three bits | `--head 4` (measured on 9 October; `--attention fast` added 1.3%) | 3.8% | 360 → 348 / 386 → 382, against Qwen's Q8_0 |
| Qwen3.6-35B-A3B | `--head 4 --bits 4` | 6.5% | 60 → 55 of 64 / 196 → 194 of 200 |
| Qwen3-Next-80B-A3B | `--head 4 --bits 4` | 8.6% | 386 → 372 / 397 → 392 |
| Qwen3-Coder-Next | `--head 4 --bits 4` | 10.5% | — / 193 → 196 of 200 |

Qwen3.5-0.8B does not take `--head 4`: with it, it answered "3" to "2+3". Gemma 4 has no fast profile: its keys are sixteen-bit, where `--attention fast` has nothing to do, its heads are not six-bit and its experts are not in planes. `--attention fast` pays where attention is a larger share of a token at 1,900 positions (Qwen3-0.6B +7.5%, Qwen3-30B-A3B +4.0%); on the hybrid Qwen3.5, Qwen3.6 and Qwen3-Next it adds 2% or less.

**The measure on this machine.** What makes a model fastest depends on the machine too, so the first time a model starts in a terminal the chat offers to measure it: the model opened each way that can differ — drafts from its prediction file or draft model, from the conversation, or none; in the `fast` profile also the output head as in the file — a reply to each of three prompts (prose, code, a list) to warm it, then the same three measured at temperature 0, so every way writes the same text; drafting and not drafting are measured prompt by prompt, taking turns at going first, since a mixture of experts is still warming while it is measured (until 8 October 2026 the drafts went first and warmed only on the prose, and Qwen3-30B-A3B looked 10% slower with them: with the order swapped, 9% faster). A way other than the one the model starts with has to be at least 3% faster to be kept (a couple of per cent is the noise of three replies). The result goes to `~/.cache/janas/tune-<name>-<profile>.conf`, with the speeds as a comment, and is used for the levers you did not set yourself; a "no" is kept too, so the question comes once. `--tune` measures again, `--no-tune` neither asks nor uses it. On the development laptop: Qwen3-4B 33.1 tokens/s with Qwen3-0.6B's drafts, 29.2 without drafts, 29.5 with drafts from the conversation; Qwen3.5-2B 77.4 with its MTP block, 41.9 without drafts, 60.6 from the conversation, and with the head at four bits 77.4 against 75.6 at six; Qwen3-0.6B within 1% either way, so it starts as it would have.

| Option | What it does |
|---|---|
| `--profile <quality\|fast>` | the catalog's profile for the model (default `quality`) |
| `--models <dir>` | where the models are (default: `JANAS_MODELS`, or `~/.local/share/janas/models`) |
| `--tune`, `--no-tune` | measure again how the model runs fastest on this machine; or neither ask nor use that measure |
| `--mtp <file>` | the model's multi-token prediction file (an MTP block, or Gemma 4's assistant): faster replies, same text; without it, the one beside the model that fits it |
| `--no-mtp` | no prediction file, not even one found beside the model |
| `--draft <file>` | a small model of the same family to guess the next tokens: faster replies, same text (see below) |
| `--ctx <tokens>` | context length (default 16384, never more than the model was trained for) |
| `--cache <GiB>` | memory for streamed experts (default: what the machine can spare) |
| `--reserve <GiB>` | when the cache is automatic, the memory left to other programs (default: a fifth of the machine's); more keeps a busy desktop out of swap, at the price of a smaller cache |
| `--bits <2\|4\|6>` | bits per weight of the experts' down matrix (default: the memory decides) |
| `--attention <exact\|fast>` | how the attention scores are computed (default: fast above 16384 tokens of context) |
| `--head <4\|6>` | bits per weight of the output head: `4` requantizes a six-bit head at start, a few per cent faster and a little less exact (default: `6`, as in the file; see below) |
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

**On `--mtp`, `--draft` and `--no-spec`.** Speculative decoding is not a trade of quality for speed: it changes how many tokens come out of one pass, never which ones. The small multi-token prediction block guesses the next few tokens, the model then runs one pass over the guesses, and a guess is kept only where the token the model itself sampled is the very same one - at the first disagreement the guess is thrown away and the model's own token stands. So no wrong guess can survive into the text, and with the same seed the reply is the one you would have got without any of it - checked on every run of the tests, 256 tokens with drafts and 256 without, from one seed, identical, and in `janas-chat` itself over a conversation of three replies cut at 600 tokens, with and without drafts. How many guesses a pass checks is decided from what passes have cost on this machine, and a pass with a guess is still tried now and then when the engine has stopped guessing, so that a moment of a busy disk does not switch the guesses off for the rest of the session (until 1 October 2026 it could: the same chat of Qwen3-Next-80B then ran at 22 tok/s one time and 26 the next). On the development laptop `--mtp` takes Qwen3-Next-80B-A3B from 23 tok/s to 36 and Qwen3.5-9B from 13 to 22 on an English explanation, and Gemma 4's assistants take its four models about one and a half times faster on an Italian chat reply (E4B from 18 tok/s to 26, the 12B from 8.5 to 12.9); how much it gives depends on the text, and on Italian prose a small Qwen gains little. It gives nothing either where the expert cache is much smaller than the model: two tokens in a pass call more experts than one, and the disk is already the limit (Qwen3-Next-80B with a 4 GiB cache: a run whose guesses had stopped after the first reply wrote at 12.75 tok/s, the five that kept them at 12.05-12.55). `--no-spec` and `/spec off` turn it off; they buy nothing but time, and are there to measure with.

Where a model has no prediction block the guesses are copied from the conversation instead, which costs nothing and is right about a quarter of the time - enough to be worth it on a dense model and barely enough on a mixture. **`--draft` puts a small model of the same family in their place**: it has to share the vocabulary, it borrows the big model's threads, and it predicts where copying only repeats, so it is right about three quarters of the time. Whether that pays depends on what a second token in a pass costs, which is about 6% on a dense model and about 28% on a mixture, where two tokens route to different experts. So it is for dense models: Qwen3-0.6B drafting Qwen3-4B gives **1.14x** - 23.0 to 26.3 tok/s, measured at the chat's own temperature, which is why the pair is lower than the table above - and on Qwen3-30B-A3B it would not pay for itself.

**On sampling.** The defaults are the ones Qwen recommends for its own models: temperature 0.7, top-k 20, top-p 0.8. They are narrower than they look. With top-p 0.8, on Qwen3-4B, **196 of 256 tokens had a single candidate left after the cut** - nothing to draw, so the reply is the one the model would have given greedily, and two questions that mean the same thing get the same words. Starting with `--top-p 0.95` brings that to 142 of 256. It is not a fault and not a setting this project chose: it is what those numbers do to a model that is sure of itself.

**On `--attention`.** At a long context most of a token goes into attention, and most of that into the scores: one dot product per position, per head. The `fast` way reads the query as sixteen-bit integers with a scale of its own, so the dot product becomes an exact integer sum — the machine does sixteen of those products at a time instead of eight, and the scores come out a fifth to a quarter quicker. What it costs is the query's quantization, a relative 3e-5, sixty times less than the eight-bit keys and values already carry; on four hundred tokens against llama.cpp the most likely token agreed 394 times out of 400 on C source against 393 the exact way, and 377 against 382 on Italian prose. Since 7 October 2026 the `fast` way also weighs the values with sixteen-bit integers, a further 6-11% off a layer's attention (`tests/bench_attn`, Qwen3-30B-A3B's geometry, 2,048 to 32,000 positions); on Qwen3-30B-A3B, 400 tokens of C against llama.cpp, the most likely token agreed 390 times out of 400 against 388 with the scores alone and 392 the exact way, with the same mean logit difference (0.2865, 0.2893, 0.2871); the figures before this sentence are for the scores alone. The engine asks for it by itself only when you asked for more than 16384 tokens of context, where the gain is real and the conversation will get there; `--attention exact` and `--attention fast` settle it either way, and so does `JANAS_ATTN=float|int16`.

**On `--head`.** The output head - the matrix that turns the model's state into a score for every word of the vocabulary - is the largest one read whole for every token: on Qwen3.5-9B a seventh of a token's time. Most files keep it at six bits per weight (Q6_K); `--head 4` requantizes it to four (Q4_K) when the model opens, a second or two, and leaves everything else as it is. Measured on 7 October 2026, 128 tokens after a prompt of 1,900: Qwen3.5-9B writes 13.35-13.36 tokens/s where 12.76-12.77 (+4.6%, the head 7.8 ms a token where 11.0), Qwen3-Next-80B-A3B with a 16 GiB cache 25.0-25.3 where 24.4-24.5 (+2.8%); prompts are read at the same speed. The price is in the answers: against llama.cpp on 400 tokens, the most likely token agreed 372 times where 383 on Qwen3.5-9B (Italian prose) and 391 where 393 (C source), 372 where 382 and 392 where 393 on Qwen3-Next-80B-A3B, 387 where 392 on Qwen3-30B-A3B (C source), and the mean difference of the scores grows by a quarter to two thirds. Requantizing the other six-bit matrices as well (attention, the shared experts) cost more agreement for next to no speed, so the option stops at the head. The files of Qwen3.5-9B, Qwen3.8-27B, Qwen3-30B-A3B, Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B and Qwen3-Next-80B-A3B have a six-bit head of their own; those of Qwen3-0.6B, Qwen3-4B, Qwen3.5-0.8B and Qwen3.5-2B use their six-bit token embeddings as the head, and there the table is requantized for both uses (the input embeddings then read at four bits, as the larger models' files already keep them): three runs each with nothing else running, Qwen3-0.6B 107.0-107.7 tokens/s where 100.6-101.9 (+6%), Qwen3.5-0.8B 113.5-114.7 where 104.2-105.8 (+9%), Qwen3.5-2B 54.4-54.9 where 50.0-50.4 (+9%), Qwen3-4B 24.0-24.1 where 23.2-23.3 (+3.4%). Here the answers move more: 352 where 376 on Qwen3.5-2B (prose), 363 where 387 on Qwen3.5-0.8B, 366 where 379 on Qwen3-4B, 327 where 345 on Qwen3-0.6B. Gemma 4's embeddings are not six-bit and are left alone. `JANAS_HEAD_BITS=4` does the same from the environment.

In the chat, a line starting with `/` is a command, and it turns grey-blue as soon as it names one the chat really has:

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
| `/experts [n]` | experts per token: fewer is faster and a little less accurate; any number, `0` included, turns off the leaving out of light experts not in memory |
| `/markdown [on\|off]` | read the model's marks, or print them |
| `/gpu [on\|off]` | whether the GPU may be given work |
| `/save-config`, `/load-config` | keep these settings for the next start, or read them back; both ask first where something would be lost |
| `/del-config` | forget the saved settings: the next start uses the chat's own |
| `/quit` | leave (also `Ctrl-D`) |

`/help` and `/about` open a page over the conversation: the text is laid out to the window however narrow it is, the arrows and **Page Up** and **Page Down** move in it, and **ESC** closes it - no other key does, so a page cannot be shut by the one you pressed looking for a way down. `/about` gives the version, the author, the licence and the repository, and then the model, the machine and the engine as they are at that moment: the three things a report of a problem needs, which were until now to be gathered by hand.

What a command changes lasts for that conversation and no longer: a knob turned to try something does not quietly become the way things are. `/save-config` writes the settings to `~/.config/janas/chat.conf` and `/load-config` reads them back, losing whatever is set at the time, which it asks about first, and `/del-config` throws the file away so that the next start uses the chat's own settings again. The file holds the words of the command line, one option to a line and the rest of the line its value, so the same reader takes both and you can open it and edit it; it is read before the command line, which therefore wins. The system message is not kept in it, and the context, the cache, the bits and the attention are read at the next start, since they are settled when the model is opened. A word the chat does not know is refused, on the command line and in the file alike, with the option nearest to it: `--tmep` ("did you mean --temp?"), a value that is not one - `--ctx 32k`, `--temp 0,7`, `--think of`, `--mode ecco` - rather than taken for something else. `janas-server`, `janas-mcp` and `janas-bench` do the same with theirs, and each program tells at its start a variable of the environment named like Janas's and not one of them (`JANAS_EXPERT_BIT` for `JANAS_EXPERT_BITS`), which would otherwise do nothing in silence.

A model marks its replies up - `**bold**`, `*italic*`, backticks, fenced blocks, headings, bullets - and the chat reads those marks rather than printing them: bold is bold, a block of code is told from the prose by its colour, a dash becomes a bullet. An underscore counts as italic only alone and at the edge of a word, so `_this_` and `**_this_**` are marks while `snake_case` and `__init__`, which a conversation about code is full of, stay as they are; a star inside a word is not a mark either, so `2*3` is a product. `--no-markdown` and `/markdown off` print the marks as they come, and a pipe always gets them: a script may want them.

The box the message is written in takes the rows the message needs, up to a third of the window, and goes back to one when it has been sent: a long question is read while it is written instead of scrolling away sideways.

The conversation is kept as it is written, so **Page Up** and **Page Down** look back over it and **Ctrl-Home** and **Ctrl-End** go to its beginning and its end. The line being written stays where it is while you read, and sending brings the end back. A terminal cannot do this by itself here: the chat pins its footer with a scrolling region, and the lines that leave the top of one never reach the terminal's own history.

The arrows move in the message - up and down between its lines while it has more than one - and bring back earlier prompts from its first and last line, kept between sessions; `Home`, `End`, `Delete`, `Ctrl-A/E/K/U/W` and `Ctrl-L` do what they do everywhere; `Tab` completes a command; a backslash and `Enter` add a line to the message rather than sending it, the backslash itself is not part of what the model reads, and the box grows to hold what is written; `Ctrl-C` stops a reply.

Piped in or out, the chat writes plain text with no drawing at all, so it can be scripted:

```sh
printf 'Explain mixture of experts in one paragraph.\n/quit\n' \
    | bin/x86_64-linux/janas-chat qwen3-next.jns --temp 0 --max 200
```

### Janas's services

The chat starts by itself the services of Janas it finds beside it - `janas-flights`, `janas-git`, `janas-github`, `janas-maps`, `janas-prices`, `janas-quakes`, `janas-system`, `janas-weather` and `janas-wiki` - and the model uses their tools when a question needs them, without being told: ask "which flights are over the Tyrrhenian Sea now?", "what's the weather like?", "how do I get to Palermo by train?", "are there new issues on Janas?", "what has changed in this repository?", "where is diesel cheapest near here?", "what is filling my disk?", "was there an earthquake?" or "show me Wikipedia's page on Zeffirelli" (with no place, where you are: `JANAS_LOCATION`, or a guess from the internet connection). `--no-services` starts none. How they work, their keys, and how to use them in Claude Code and other clients: [src/services](src/services/README.md).

**The services' tools are given when they are needed.** Written into the system message, the tools of three services made it four thousand tokens, over a minute of a large model before the first message, and every service more made it longer. The system message now holds a catalog, a line a service, and the model opens a service when a question needs it: that service's instructions and tools come back as the answer of the call, its tools can be called from then on, and the next reply is held to be a call (a small model told to call one made the answer up instead). With Qwen3.6-35B-A3B the system message went from 4,057 tokens to 616, and opening the weather costs about a thousand tokens, once in a conversation, when the first question about the weather comes. `--all-tools` writes them all into the system message as before.

**The system message is read ahead.** It used to be four thousand tokens with the services' tools, and the first reply waited for all of it. `janas-chat` now reads it as soon as it starts, while you write, and says so: a line in the conversation when it begins, and "reading the tools ahead" with how far it is in the status line, where it is the last thing to be dropped on a narrow window; a message sent before the end waits for it, with "waiting for the tools" and the share, and Ctrl-C leaves the rest to the reply. Once read it is kept in `~/.cache/janas` (2 GiB at most for these files), and the next chat on the same model finds it there: "System message and tools ready". Changing the system message or the tools reads them again.

### Tools from MCP servers

The chat also starts the [Model Context Protocol](https://modelcontextprotocol.io) servers listed in `~/.config/janas/mcp.json` (or in the file `--mcp-config` names), in the format the other clients use, so a configuration can be copied from one of them:

```json
{"mcpServers": {
    "files": {"command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-filesystem",
                       "/home/me/notes"]}}}
```

A key a letter or two from one of those Janas reads (`arg` for `args`, `evn` for `env`) is refused with the one meant, and `disabled` is `true` or `false`, not the text "true"; the keys of the other clients (`autoApprove`, `cwd`, `envFile`...) are left to them. Their tools are given to the model as `server__tool`. When the model calls one, the chat shows the call with its arguments and waits for a key: `y` runs it, `n` tells the model it was not allowed, `a` runs it and every later call of the same tool. `/mcp` lists the servers and their tools; `/mcp auto`, or `--mcp-auto` at the start, runs the calls without asking - which a pipe needs, since nobody is there to answer, and without which a piped chat runs none. `--no-mcp` starts no server. A tool runs with your rights, and what it answers is text somebody else wrote that the model then reads: configure only servers you trust, and read the calls before saying yes. For the same reason the instructions a server gives for its tools stay out of the model's system message unless `--mcp-instructions` asks for them; then they follow it, marked as the servers' and not the user's.

Servers are spoken to over their standard input and output (the stdio transport), in either era of the protocol: the stateless revision 2026-07-28, and the handshake of 2025-11-25 and before, which most servers still speak; which one a server speaks is found out when it starts. What a server writes to its standard error goes to `~/.cache/janas/mcp/<name>.log`. A server that listens at an address is named with `url` (and `headers`, for an `Authorization` say) instead of `command`, and spoken to over Streamable HTTP, again in either era; `https://` too, the server's certificate checked against the system's certificate authorities (or `SSL_CERT_FILE`) and its name. It has been run against the test server in [`tests/mcp_fake_server.c`](tests/mcp_fake_server.c), in both eras, over stdio and over HTTP, and against two reference servers over stdio, both of the 2025-11-25 era: `@modelcontextprotocol/server-filesystem` 0.2.0 (npx, 14 tools) and `mcp-server-git` 1.30.0 (uvx, 12 tools), together, with Qwen3-4B choosing among their 26 tools - `git_log` for the last commit of a repository, `list_directory` then `read_text_file` to find a date in a folder. Their descriptions take about 3,700 tokens of the system message, read once and kept computed. Over HTTPS it has reached DeepWiki (`https://mcp.deepwiki.com/mcp`, 2025-11-25 era, a session per client). A small model does not always find its way: asked without a hint, Qwen3-4B tried to read the folder itself as a file, got the server's error, and stopped there.

### Watching a long prompt

A long prompt is most of what a reply costs: on a laptop CPU, 260,000 tokens of Qwen3-Next-80B-A3B take hours to read and seconds to answer. So the reading tells how it goes. The library reads a prompt a block of 256 tokens at a time, and `janas_llm_chat_next` returns an empty piece after each block, so a program has control between them; `janas_llm_chat_stats` says the stage (reading, writing, done), the tokens read of those to read, the time to the first token, the time spent building the prompt, the processor time of the reading, the threads the engine chose, the memory of the context and the process's peak, how the prompt is made up (system message and tools, the conversation before, the last message), and an estimate of the time left. `max_input` in the chat's parameters refuses a longer prompt with its numbers, and never cuts it.

Any program on the library, its own code unchanged, can have the same from the environment: `JANAS_PROGRESS=30` writes a line on stderr every thirty seconds while a prompt is read and one at the end of each reply, and `JANAS_METRICS=file` appends JSON lines - events `open`, `prompt`, `progress` (every `JANAS_METRICS_EVERY` seconds, 10 by default), `first_token`, `reply` and `refused` - with counters and times only, never the text of a prompt or a reply:

```json
{"event":"progress","time":"2026-09-25T08:44:52.027Z","pid":3948047,"reply":1,"stage":"input","input_done":1536,"input_tokens":3044,"elapsed_seconds":3.8,"tokens_per_second":362.75,"tokens_per_second_avg":405.32,"eta_seconds_estimate":4}
```

The time left is an estimate that follows how the reading slows: a token costs more the more context it follows, attention having more to look at, so the engine fits that cost as a line in the token's position over the blocks read so far and adds it up to the end of the prompt, never below what the latest speed says. On a 260,954-token prompt of Qwen3-Next-80B-A3B the elapsed time followed that shape to R² 0.99999, where an estimate from the current rate promised 2.5 times too little at 32K ([issue #9](https://github.com/prabanta-dev/janas/issues/9)); the new estimate has not yet been checked on a run that long. `janas-chat` shows the progress in its status line and takes `--progress`; `janas-server` takes `--metrics`, `--progress`, `--max-input` and `--warn-input`.

## Filling in code

`janas-server` fills in code at an editor's cursor, with a model for code such as Qwen3-Coder-30B-A3B: OpenAI's `/v1/completions` with a `suffix`, and llama.cpp's `/infill`, which llama.vscode asks. The window sent at every keystroke is kept from moving, so that a keystroke reads only what it changed: 1.5-2.2 s a proposal on a laptop, where a window slid a line was read again in 17 s. How to set it up in VSCodium and VS Code, with llama.vscode or Continue, and what was measured: [docs/code-completion.md](docs/code-completion.md).

## Using it from a program

The library is shaped for foreign-function interfaces from the start: opaque handles, fixed-width types, nothing passed by value that is not a number, parameter blocks that carry their own size, error codes as `int32`, and reply text returned on request as whole UTF-8 characters. It is declared for **C** and for **FreeBASIC**.

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

A program that keeps the conversation itself - an HTTP client does, and sends it whole every time - hands it over with `janas_llm_chat_load`: the messages with their roles, the last one the user's. Whatever the new conversation shares with the one already computed is not read again, so the same messages plus a new one cost only the new one. `janas_llm_chat_prompt` continues raw text, with no chat format around it, and `janas_llm_default_system` gives the system message Janas's own programs use. `janas_llm_chat_stats` says why a reply ended and how much of its prompt was already computed. `janas_llm_chat_keep` keeps a few conversations computed besides the current one, for a program that switches between them: the one that shares most of the next request is copied back instead of being read again. `janas_llm_chat_prepare` reads the system message and the tools before the first message, a block at a time, so that a program can read them while its user writes; with `janas_llm_chat_keep_disk` a system message read once on a model is found on the disk by the next chat.

### From FreeBASIC and BASIC MODERN

[`include/janas/llm.bi`](include/janas/llm.bi) declares the same library for FreeBASIC, and so for **BASIC MODERN**, the dialect implemented by Prabanta — the platform Janas is meant to sit inside. The same conversation as above:

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

[`tools/chat.bas`](tools/chat.bas) is a whole chat in a hundred lines, sampling and counters included:

```sh
fbc tools/chat.bas -i include -p bin/x86_64-linux -x janas-chat-fb
LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-fb qwen3-next.jns
echo "Hello! Introduce yourself." | LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-fb qwen3-next.jns
```

Two things to know. FreeBASIC does not tell names apart by their case, so the constant the C header calls `JANAS_LLM_ABI_VERSION` is `JANAS_LLM_ABI` there — it would otherwise collide with the function `janas_llm_abi_version()`. And its `line input` reads the terminal, not the standard input: to work in a pipe too, `chat.bas` asks `isatty` and reads a pipe through the `CONS` device.

### MCP servers from a program

[`include/janas/mcp.h`](include/janas/mcp.h) is the MCP client the chat uses, as a library of its own, `libjanas_mcp`, which needs nothing of `libjanas_llm`; [`include/janas/mcp.bi`](include/janas/mcp.bi) declares it for FreeBASIC. It fits the model's side of the tools: `janas_mcp_tools` gives a server's tools as the JSON `janas_llm_chat_tools` takes, the calls the model writes go to `janas_mcp_call`, and what `janas_mcp_result` returns goes back with `janas_llm_chat_send_results`.

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

`janas-mcp` offers the model running here to the clients of the Model Context Protocol - Claude Code, editors, other agents - as tools: `generate` asks it a question (a prompt, and optionally a system message, a longest answer and a temperature) and returns its answer, and `embed` returns the vector of a text when an embedding model is given with `--embedding-model`. The client starts it and speaks to it over its standard input and output; it is configured like any other stdio server:

```json
{"mcpServers": {"janas": {"command": "/path/to/janas-mcp",
                          "args": ["/path/to/qwen3-4b.jns"]}}}
```

It answers both eras of the protocol: a request that names revision 2026-07-28 in its `_meta` is served statelessly, and a client that opens with `initialize` gets the handshake of the revision it asks for (2025-11-25, 2025-06-18, 2025-03-26 or 2024-11-05). A ping is answered while a generation runs, and `notifications/cancelled` stops it. Every call starts a conversation of its own; the reasoning of a model that reasons is never returned, only the answer. `janas-mcp --help` lists the options (`--ctx`, `--temp`, `--max`, `--think`, `--timeout` and the memory ones).

What has been run: a probe script speaking each era to it with Qwen3-0.6B and Qwen3-Embedding-0.6B, `janas-chat` with Qwen3-4B calling its two tools through `libjanas_mcp`, and Claude Code (2.1.282, revision 2026-07-28), which listed both tools and called them - a translation, a summary, an embedding - with Qwen3-4B and Qwen3-Embedding-0.6B behind them. To add it there: `claude mcp add janas -- /path/to/janas-mcp /path/to/model.jns`.

## Using it over HTTP

```sh
bin/x86_64-linux/janas-server qwen3-next.jns
```

`janas-server` puts the model behind an HTTP API that follows [OpenAI's](https://github.com/openai/openai-openapi), on `http://127.0.0.1:8080/v1`, so that a client written for that API can use a model running on this machine. It takes the model options `janas-chat` takes (`--ctx`, `--cache`, `--reserve`, `--mtp`, `--draft`, `--mode`, `--no-gpu`) and these:

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
| `--fim-prefix N`, `--fim-suffix N`, `--fim-max N` | completions with a `suffix`, an editor filling in code at its cursor: the tokens before the cursor read at most (2048; 0: all) and the lines after it (8; 0: all); `--fim-max N` the tokens written there at most, whatever the request asks (256; 0: as asked), since an editor asks for thousands and the requests of the next keystrokes wait for them. Over the budget the text before the cursor is cut at a line to three quarters of it, and the start of the window is kept across keystrokes, also when the editor slides its window, so that a keystroke reads only what it changed (see [Filling in code](#filling-in-code)) |
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
| `POST /infill` | llama.cpp's route for filling in code, which editors' extensions made for llama.cpp ask (llama.vscode): `input_prefix`, `prompt`, `input_suffix`, `n_predict` (0: the prompt only, read for the next request), `n_indent`, `t_max_predict_ms`, the answer as llama.cpp's (`content`, `timings`); `input_extra` is not used, and the most likely token is always taken |
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

A client that lets you set the address of an OpenAI-compatible API should need nothing else: the base URL is `http://127.0.0.1:8080/v1`, and the key whatever you gave `--api-key` (anything, if you gave none). It has been tried here with `curl`, its own test page, the `openai` Python SDK and Open WebUI 0.11.4 (driven through its own API: streamed chat, titles, tags, follow-up suggestions, and a document indexed with Janas's embeddings and asked about), and by a user with AnythingLLM.

**Models that reason, and the limits a client sets.** Qwen3 and its successors reason before they answer, unless told not to. OpenAI's `max_completion_tokens` (and `max_output_tokens` in the Responses API) counts the reasoning too, and Janas takes it so. The older `max_tokens` of a chat, which OpenAI does not take at all for the models that reason, is taken for the answer alone: a client that sends it means the answer, and counted with the reasoning a small limit was spent before a word of it was written (Open WebUI asks for a chat's title with 1000, and got none).

**With Open WebUI**, start the server with `--think off`: the titles, tags and suggestions it asks for after every reply are then quick, and a model that reasons can still be asked to (`reasoning_effort`). Its built-in tools, when they are on, add about 5,000 tokens to every prompt; the server reads them once and keeps them computed, but the first reply of a conversation waits for them.

**The conversation is the client's.** OpenAI's API keeps no state: every request carries the whole conversation. `janas-server` keeps the one it computed last, and reads again only what a request adds to it: a client that sends the same messages plus a new one pays for the new one alone, and `usage.prompt_tokens_details.cached_tokens` says how much was already computed. The replies it wrote are remembered with their reasoning, so the answer a client sends back - without the reasoning, as clients do - still counts as the same turn. A conversation that differs earlier is read again from where it differs, and from the start on models with a recurrent state (Qwen3-Next, Qwen3.5 and 3.6).

**Reasoning** comes apart from the answer, in `reasoning_content`, whole or streamed. `"reasoning_effort": "none"` turns it off for a request, any other effort turns it on, and `"chat_template_kwargs": {"enable_thinking": false}` works too; `--think` decides for requests that say nothing.

**Beyond OpenAI's fields**, which clients ignore: `top_k` and `min_p` in a request, and in every reply's `usage` a `janas` object with the context used and its size and the seconds spent reading the prompt and writing the reply.

**Tools.** `tools`, `tool_choice` (`none`, `auto`, `required`, a function) and `parallel_tool_calls`, and the older `functions`. The tools are written into the system message the way the model's chat template writes them, and a call the model opens is held to a grammar of the functions and their parameters until it closes it, so its arguments are always a JSON object valid against the function's schema. The calls come back in `tool_calls`, whole in one delta of the stream, and the reply ends with `"finish_reason": "tool_calls"`; sent back with the tools' answers, they are found again among the replies the server remembers, so the next turn reads only the answers. Qwen models write calls in one of two ways, and the template says which: as JSON (Qwen3, Qwen3-Next) or as XML (Qwen3.5 and 3.6, Qwen3-Coder). **Both have been run here**: the JSON way on Qwen3-4B and Qwen3-0.6B, the XML way on Qwen3.6-35B-A3B (two calls at once, their answers sent back, 422 of 471 prompt tokens reused on the next turn) and on Qwen3-Coder-Next, which describes the tools in a way of its own (two calls at once with valid arguments, their answers used in the reply, 496 of 562 prompt tokens reused).

**MCP servers in the Responses API.** A tool of type `mcp` (`server_label`, `server_url`, `headers`, `authorization`, `allowed_tools`, `require_approval`) has the server reach that MCP server over HTTP or HTTPS, as OpenAI's does. The response begins with an `mcp_list_tools` item (once per conversation), the model sees the tools as `server_label__tool`, and a call it writes either becomes an `mcp_approval_request` that ends the response - `require_approval` is `always` by default, as at OpenAI - and runs when the next request answers it with an `mcp_approval_response`, or, where no approval is needed, runs at once as an `mcp_call` item with its output, and the model goes on in the same response (up to eight rounds). Streamed, the calls come as `response.mcp_call.*` events. `allowed_tools` and the approval filters take names or `{tool_names, read_only}`; `tool_choice` may name an MCP tool. Connectors and tunnels are not served. A request makes the server open connections to the address it names: `--no-mcp` turns that off for a server others can reach. Tried with Qwen3-4B against the test server of `tests/mcp_fake_server.c`: a call run at once, one approved on the next request, and the same streamed; and through the `openai` SDK against DeepWiki over HTTPS, as OpenAI's own examples do: its tool listed, called, and the model answering from what it returned.

**JSON output.** `response_format` `json_object` or `json_schema` (and `text.format` in the Responses API): the answer is held to a grammar token by token, so it is valid JSON when the model ends it (not when `max_tokens` cuts it). A schema is held to its types, the properties of an object in the schema's order with the required ones always there and no others, array items and bounds up to 32, string lengths up to 64, `enum`, `const`, `anyOf`, `oneOf`, `$ref` within the schema and `nullable`; `pattern`, `format` and numeric bounds are not enforced. The reasoning, where there is one, stays free.

**Log-probabilities.** `logprobs` and `top_logprobs` (up to 20) in chat, `logprobs` in completions, `top_logprobs` in responses. They are the model's own: where a grammar forced a token the model found unlikely, its log-probability says so.

**More than one reply** with `n` (and `best_of` in completions, which keeps the best by mean log-probability); `presence_penalty`, `frequency_penalty` and `logit_bias` as OpenAI defines them. The replies of `n` are written one after the other on the same prompt.

**Embeddings** come from a second, small model opened with `--embedding-model`, such as Qwen3-Embedding-0.6B: texts or token ids, one or many, `dimensions` to cut the vector shorter, floats or `base64`. Against llama.cpp on the same file the vectors of the same text agree to a cosine of 0.98 to 0.996, and the similarities between texts come out alike.

**Moderations** are asked of the chat model: at temperature 0 it answers OpenAI's thirteen categories with a JSON object held to a schema, and each score is the chance it gave `true` against `false` in that place. It is a general model following instructions, not a classifier trained for the task.

**Kept on disk.** Stored completions, responses and conversations are written, one file each, to `~/.local/share/janas/server` (or `--store DIR`), and read back when the server starts again; the oldest go past a thousand of each. The directory and its files are readable by their owner alone, since they hold what the clients said, and one server at a time uses a directory. `--no-store` keeps them in memory only, until the server stops. A response running in the background when the server is killed is found as failed at the next start. Images and sound in messages are refused with a 400 that says so.

**Every other operation of OpenAI's API** - there are 345 in the version it follows (2.3.0) - has a route that answers 501 and says why: not written yet, needs a model of another kind (sound, images, video), trains models, or belongs to the administration of OpenAI's service. The table is generated from the specification by `tools/openapi_routes`, which also counts what is done.

**One request at a time.** The model holds one sequence, so requests wait in line and run in the order they came; when the line is full the answer is 429. While a streamed request waits, the stream says where it stands in the line (`: queue N`, a comment clients skip) and the test page shows it. A client that closes the connection while a reply streams stops the reply.

**On the network.** It listens on this machine alone unless `--host` says otherwise, and then it warns if there is no key. There is no HTTPS yet: across a network you do not trust, put it behind something that adds it.

**The test page.** With `--test`, `http://127.0.0.1:8080/test` is *Janas-Chat Web*: a chat in the browser that talks to the API like any client, with the reasoning in grey, a line under each reply with the context used and the speed (last, lowest, highest and mean of the conversation), and the settings - system message, temperature, reasoning, key - behind *Settings*. It reads the marks a model writes as `janas-chat` does, and also quotes, tables, struck text and links; a link asks before it opens, because links an AI writes can be wrong or unsafe. *Markdown sample* shows a text of ours with every mark it knows, without asking the model. The page is inside the program, loads nothing from anywhere else and may talk only to this server.

## Measuring your machine

```sh
bin/x86_64-linux/janas-bench qwen3-next.jns
```

Close what you can before running it: it measures the machine you give it, and a browser with thirty tabs is part of the machine. It tries the configurations the machine offers — how many threads, which cores, with and without the GPU — on real passes of the model, keeps the fastest, writes them into `~/.cache/janas` so the next start begins from them, and leaves a report in the current directory. On the development laptop that report says, among other things, that bringing the efficiency cores in makes decoding about a tenth faster than the performance cores alone - and that for blocks of several tokens, which is what a prompt and a draft check are, the engine often wants fewer threads than for one.

## Tools

| | |
|---|---|
| `tools/gguf2jns` | GGUF → the `.jns` format the engine reads |
| `tools/hf2jns_mtp` | the multi-token prediction block, from the original checkpoint: its shards, or just the block's tensors from Hugging Face (`hf:<owner>/<repo>`) |
| `tools/jns_planes` | rewrites a model with the experts' down matrix in bit planes, so a machine short of memory can read part of it |
| `jns_check` | checks a model file, with `--verify` every expert's checksum |
| `tools/openapi_routes` | `janas-server`'s route table, from OpenAI's specification |
| `tools/gen_unicode` | the tokenizer's Unicode tables, from the Unicode Character Database |
| `tools/gen_geo` | the services' tables of airports, cities and seas, from OurAirports, GeoNames and Natural Earth |
| `mcp_fake_server` | an MCP server over stdio in either era of the protocol, for trying a client without anybody else's server |
| `llm_eval` | compares the engine's logits against a reference dump |
| `bench_kernels`, `bench_gemm`, `bench_attn`, `bench_long`, ... | the pieces measured on their own |

They are built into `bin/x86_64-linux/` along with everything else.

## Helping it run on more hardware

This is where help is worth most. The engine decides everything from what it measures, but it has only ever measured **one** machine, so the decisions are tuned to one shape of CPU, one disk and one GPU.

**The easy way: one command.** From a fresh clone, on Linux, with a C compiler and `curl` and nothing else:

```sh
git clone https://github.com/prabanta-dev/janas && cd janas
./tools/janas-try.sh
```

It builds Janas and runs its tests, downloads a model from Hugging Face (checking its SHA-256), converts it (checking the result against the fingerprints of [MODELS.md](MODELS.md)), checks that the model's answer at temperature 0 is the very text every other machine gets, measures the speed with `janas-bench`, tries `janas-server`, and writes a report with no host name, user name or paths in it. It asks before every long step and shows the report before anything is sent. **Run it with the machine to itself** (close the browser, builds, other models): before measuring it checks how idle the CPU is and, if it is busy, offers to wait; the report says how idle it was, in general terms only (a load average and a percentage, never which programs ran). If you agree, it opens the issue with `gh`, or gives you a link to a filled [test report form](https://github.com/prabanta-dev/janas/issues/new?template=04-test-report.yml). Four levels: `quick` (Qwen3-4B, 2.5 GB to download), `medium` (Qwen3.6-35B-A3B, 22 GB), `full` (Qwen3-Next-80B-A3B with its MTP block, about 52 GB) and `gemma` (Gemma-4-E4B with its assistant, 5.1 GB, runs in about 3 GB of memory); it offers those your machine can hold, and deletes at the end only what it downloaded, if you say so. `--help` lists the options.

**What to send, by hand.** Run the benchmark and attach the report it writes (a `janas-bench-<date>.txt` in the current directory):

```sh
bin/x86_64-linux/janas-bench <model.jns>
```

The report already carries the CPU's name, the core layout, the GPU if there is one, and the speed of every configuration tried. Add:

- the **distribution** and kernel (`uname -a`), and the **RAM** (`free -g`);
- the **disk** the model sits on (NVMe? SATA? over USB?);
- the **model** you used and its file size;
- what you saw: numbers that look wrong, a reply that made no sense, a crash, a machine that went unresponsive — with what you were doing at the time.

The [hardware report form](https://github.com/prabanta-dev/janas/issues/new?template=01-hardware-report.yml) asks for exactly that, in that order. There are two more forms, for [something that went wrong](https://github.com/prabanta-dev/janas/issues/new?template=02-something-went-wrong.yml) and for [a model that will not convert](https://github.com/prabanta-dev/janas/issues/new?template=03-a-model-will-not-work.yml).

**What is most wanted, in order.**

1. **CPUs without AVX-VNNI**, and **AMD** — the arithmetic has a path for them and it is the least exercised of all.
2. **Discrete GPUs** (NVIDIA, AMD, Intel Arc). The GPU support is correct - bit for bit on a reply's tokens, and on a prompt as close to the references as the CPU - and it pays where it was measured: on the development laptop's integrated GPU, with the fast prompt, Qwen3-4B reads a prompt 2.6 times faster than without the GPU (85 to 220 tokens/s; before the fast prompt, 83 to 107), and on an RTX 5080 (issue #15) Qwen3-4B read 13% faster and, with the GPU on single tokens too, replied 11-16% faster. Every other card is unknown, and the fast prompt has not run on a discrete one yet (`JANAS_GPU_FAST_PROMPT=0` turns it off if it misbehaves). **Careful:** a GPU driver reset takes the desktop with it. Start with `janas-bench`, not with a long chat.
3. **Machines with 16 GB or less.** The engine is supposed to trade quality for memory on its own — fewer bits per weight, light experts not in memory left out — and the thresholds for that were measured on 32 GB.
4. **Slower disks.** Everything assumes an NVMe SSD; on a SATA disk the engine should still work and simply wait more, but nobody has watched it do so.
5. **Other models** of the supported families. A file that will not convert, or converts and then answers nonsense, is a useful bug.

**What not to bother with yet:** Windows and macOS (not supported), and models outside the Qwen3 MoE families (they are refused with a message).

Patches are welcome too, but a measurement from a machine nobody here can buy is worth more than most patches.

## Credits

Janas is designed and written by **Maurizio "camauri" Cammalleri** ([LinkedIn](https://www.linkedin.com/in/maurizio-cammalleri-80a89a11/), [Substack](https://cammalleri.substack.com/)).

It stands on the work of others, and says so:

- **The models** it runs are their authors': Qwen3, Qwen3.5, Qwen3.6 and Qwen3-Next by the Qwen team of Alibaba Cloud, Gemma 4 by Google; the GGUF quantizations by Qwen, bartowski and unsloth ([MODELS.md](MODELS.md) has every source).
- **llama.cpp and ggml** are the reference the engine is checked against (logits, tokens, speed); none of their code is in Janas. The one thing taken from ggml is data that defines a format: the IQ3_S grid, in `src/llm/iq3s_grid.h`, under ggml's MIT licence.
- **The Unicode Character Database** gives the tokenizer its tables (`tools/gen_unicode`), and **OpenAI's OpenAPI specification** gives `janas-server` its route table (`tools/openapi_routes`).
- **GNU libmicrohttpd, yyjson and Mbed TLS** are compiled into the programs named below.
- **`janas-flights`** reads the positions of aircraft from [adsb.lol](https://adsb.lol) (data under the Open Database License) and [adsb.fi](https://adsb.fi), routes from [adsbdb.com](https://www.adsbdb.com) and, with a key of the user's, schedules from [AviationStack](https://aviationstack.com); its tables of places are built from [OurAirports](https://ourairports.com) (public domain), [GeoNames](https://www.geonames.org) (CC BY 4.0) and [Natural Earth](https://www.naturalearthdata.com) (public domain).
- **`janas-quakes`** reads the earthquakes from [INGV](https://terremoti.ingv.it) (Istituto Nazionale di Geofisica e Vulcanologia, CC BY 4.0) for Italy and its seas and from the [USGS](https://earthquake.usgs.gov) (public domain) elsewhere, and places from [Nominatim](https://nominatim.org) (© OpenStreetMap contributors, ODbL).
- **`janas-weather`** reads forecasts and the sea from [Open-Meteo](https://open-meteo.com) (CC BY 4.0) and [MET Norway](https://api.met.no) (CC BY 4.0), the stations' reports from [aviationweather.gov](https://aviationweather.gov) (NOAA), warnings from [MeteoAlarm](https://meteoalarm.org) (EUMETNET) and the Italian alert levels from the [Dipartimento della Protezione Civile](https://github.com/pcm-dpc/DPC-Bollettini-Criticita-Idrogeologica-Idraulica) (CC BY 4.0). With no place named, the services ask [GeoJS](https://www.geojs.io) or [ipwho.is](https://ipwho.is) where the internet connection is.
- **`janas-maps`** reads [OpenStreetMap](https://www.openstreetmap.org)'s data (© OpenStreetMap contributors, ODbL) through [Nominatim](https://nominatim.org), [Photon](https://photon.komoot.io), [Valhalla](https://github.com/valhalla/valhalla) and [OSRM](https://project-osrm.org) on the servers of [FOSSGIS](https://fossgis.de), and the [Overpass API](https://overpass-api.de) (with [Private.coffee's instance](https://overpass.private.coffee) when it is busy), and journeys by public transport from [Transitous](https://transitous.org), over the operators' open timetables.
- **`janas-prices`** reads the [European Central Bank](https://www.ecb.europa.eu) (through [Frankfurter](https://frankfurter.dev)), [CoinGecko](https://www.coingecko.com/en/api) (data provided by CoinGecko) and [Coinbase](https://www.coinbase.com), [Alpha Vantage](https://www.alphavantage.co) with the user's key, [Eurostat](https://ec.europa.eu/eurostat) and the [World Bank](https://data.worldbank.org) (CC BY 4.0), the [New York Fed](https://www.newyorkfed.org), the [Bank of England](https://www.bankofengland.co.uk) and the [Swiss National Bank](https://data.snb.ch), [energy-charts.info](https://www.energy-charts.info) (Fraunhofer ISE; the zones under CC BY 4.0, from the Bundesnetzagentur's SMARD), and the fuel prices of Italy's [MIMIT](https://www.mimit.gov.it) (IODL 2.0), France's [prix-carburants](https://data.economie.gouv.fr), Spain's [ministry](https://energia.serviciosmin.gob.es) and Austria's [E-Control](https://www.e-control.at); places through [Nominatim](https://nominatim.org).
- **`janas-github`** reads [GitHub](https://github.com)'s REST API; **`janas-git`** runs [git](https://git-scm.com).
- **`janas-wiki`** reads [Wikipedia](https://www.wikipedia.org) through its API; the text is Wikipedia's, under CC BY-SA 4.0, and every answer gives the page's address.
- **The people who tried it on their own machines** and reported what they saw, in the issues of this repository: their reports changed the engine more than once (the threads it picks, the prompt's progress, the GPU's diagnosis).

## Licence

Copyright © 2026 Maurizio "camauri" Cammalleri.

Janas is free software under the **GNU General Public License, version 3 or later** — see [LICENSE](LICENSE). Every source file carries its SPDX line.

`janas-server`, and only it, has two libraries of others compiled in, their sources unchanged in [`src/third_party/`](src/third_party) with their licences and where they came from: **GNU libmicrohttpd** 1.0.10 (LGPL 2.1 or later, or the eCos licence without HTTPS, as it is built here) and **yyjson** 0.13.0 (MIT). `libjanas_mcp`, and the programs with it, have **Mbed TLS** 4.2.0 with TF-PSA-Crypto (Apache-2.0 or GPL-2.0-or-later), for `https://` MCP servers. `libjanas_llm` has none.

**The models are not covered by it.** A converted `.jns` file is a derivative of the model, not of Janas, and keeps the licence its authors gave it; [MODELS.md](MODELS.md) says what this project will and will not redistribute, and what to check before you pass a converted model on.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md). The short version: a measurement from a machine nobody here owns is worth more than most patches, and every commit needs a `Signed-off-by` line (`git commit -s`).
