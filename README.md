# Janas

[![Status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)]() [![Platform: Linux | Windows x86-64](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20x86--64-lightgrey.svg)]() [![LinkedIn](https://img.shields.io/badge/LinkedIn-Maurizio%20Cammalleri-0077B5?logo=linkedin)](https://www.linkedin.com/in/maurizio-cammalleri-80a89a11/) [![Substack](https://img.shields.io/badge/Substack-Maurizio%20Cammalleri-FF6719?logo=substack)](https://cammalleri.substack.com/)

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

> **Download Janas only from here.** Janas is published only at [github.com/prabanta-dev/janas](https://github.com/prabanta-dev/janas), its [releases](https://github.com/prabanta-dev/janas/releases) and its models at [huggingface.co/prabanta-dev](https://huggingface.co/prabanta-dev). Other sites using its name are not the project's: one, `joaniberian9588.github.io`, offers a Windows download that is malware, with no part of Janas in it (reported to GitHub on 6 November 2026). Do not run it; if you did, treat that computer as compromised.

**Qwen3-Next-80B, a 48 GB model, on a laptop with 32 GB of memory: 26 tok/s writing a reply, where llama.cpp manages 3.0, and 68 tok/s reading a prompt, against 18.** On the same laptop (Debian 13 Linux, Intel Core Ultra 9 185H with its integrated Arc GPU and its NPU, 32 GB, an NVMe SSD; the whole machine [below](#the-machine-it-grew-on)) Janas-LLM read a prompt and wrote a reply faster than llama.cpp on every model compared, under the same conditions wherever possible - the same laptop, the same files, the same day, each engine in its best configuration (21 October 2026):

| Model | reading 1,024 tokens: Janas-LLM | how | llama.cpp | writing: Janas-LLM | how | llama.cpp |
|---|---:|---|---:|---:|---|---:|
| Qwen3.8-Flash-Next (66.5 GB, 177B, more than the memory) | 33.0 tok/s | GPU, 12 threads | 4.82 tok/s‡ | 10.95 tok/s | CPU, 12 threads | 3.12 tok/s‡ |
| Qwen3-Next-80B-A3B (48.4 GB, more than the memory) | 68.4 tok/s | GPU, 20 threads | 17.7 tok/s | 26.4 tok/s | CPU, 20 threads | 3.0 tok/s |
| Qwen3-Coder-Next (48.4 GB, more than the memory) | 71.4 tok/s | GPU, 12 threads | 14.0 tok/s | 26.5 tok/s | CPU, 20 threads | 2.9 tok/s |
| Qwen3.6-35B-A3B | 136 tok/s | GPU, 12 threads | 81.8 tok/s | 28.0 tok/s | CPU, 20 threads | 12.1 tok/s |
| Qwen3-30B-A3B | 133 tok/s | GPU, 6 threads | 122 tok/s† | 33.3 tok/s | CPU, 20 threads | 15.0 tok/s† |
| Qwen3-30B-A3B, bartowski's Q4_K_M (importance matrix) | 132 tok/s | GPU, 6 threads | 125 tok/s | 32.3 tok/s | CPU, 20 threads | 15.9 tok/s |
| Qwen3-30B-A3B, unsloth's UD-Q3_K_XL (3 bits) | 120 tok/s | GPU, 12 threads | 114 tok/s | 35.4 tok/s | CPU, 20 threads | 11.3 tok/s |
| Gemma-4-26B-A4B-it | 85.0 tok/s | GPU, 6 threads | 78.0 tok/s | 18.9 tok/s | CPU, 20 threads | 12.6 tok/s |
| Qwen3.8-27B (dense, 17.6 GB) | 31.4 tok/s | GPU, 6 threads | 25.5 tok/s | 4.4 tok/s | CPU, 20 threads | 2.1 tok/s |
| Gemma-4-12B-it | 65.7 tok/s | GPU, 12 threads | 47.6 tok/s | 10.0 tok/s | CPU, 20 threads | 7.4 tok/s |
| Qwen3.5-9B | 122 tok/s | GPU, 6 threads | 80.3 tok/s | 14.5 tok/s | CPU, 20 threads | 8.7 tok/s |
| Gemma-4-E4B-it | 245 tok/s | NPU, 6 threads | 110 tok/s | 19.8 tok/s | CPU, 12 threads | 15.3 tok/s |
| Qwen3-4B | 333 tok/s | NPU, 6 threads | 172 tok/s | 27.9 tok/s | CPU, 20 threads | 17.9 tok/s |
| Qwen3-4B, bartowski's Q4_K_M (importance matrix) | 333 tok/s | NPU, 6 threads | 173 tok/s | 27.8 tok/s | CPU, 20 threads | 18.2 tok/s |
| Gemma-4-E2B-it | 508 tok/s | NPU, 6 threads | 150 tok/s | 37.0 tok/s | CPU, 12 threads | 18.9 tok/s |
| Qwen3.5-2B | 729 tok/s | NPU, 6 threads | 314 tok/s | 53.3 tok/s | CPU, 20 threads | 27.5 tok/s |
| Qwen3.5-0.8B | 1,403 tok/s | NPU, 6 threads | 514 tok/s | 110 tok/s | CPU, 12 threads | 31.1 tok/s |
| Qwen3-0.6B | 1,421 tok/s | NPU, 6 threads | 710 tok/s | 133 tok/s | CPU, 20 threads | 46.0 tok/s |

Janas-LLM's figures are the best of the configurations `janas-bench` measures, and "how" names it: the CPU's threads, with the GPU reading a prompt's blocks (its fast prompt) or the NPU reading them; a reply is written by the CPU. The configuration the engine chose by itself, as a chat runs, was within 1.5% of the best reading and within 2.6% writing, on every model. llama.cpp ran with its Vulkan backend on the same GPU, the API Janas-LLM uses, for reading and for writing; on the two models larger than the memory it wrote with its CPU backend, faster there than with the GPU, as Janas-LLM writes. Where the conditions could not be the same, the table says so: Janas-LLM writes the other models' replies on the CPU, where llama.cpp's writing on the CPU was not measured; and where Janas-LLM reads on the NPU, llama.cpp's figure is from the GPU, its OpenVINO backend, the one that uses the NPU, having stopped on Gemma 4, on the Qwen3.5 models and on every prompt of 768 tokens and more (the two compared on the same NPU: [Intel's NPU](#intels-npu)). Its SYCL backend, Intel's own for Arc GPUs, was not used: Janas-LLM reaches every GPU through Vulkan. † llama.cpp's figures of 11 October: the file it read is no longer on this machine. ‡ Qwen3.8-Flash-Next, 23 October: llama.cpp on its CPU backend, the faster way it has here - its Vulkan backend cannot run the file on 32 GB (it copies 35.9 GB of weights into pinned host buffers; with `no_host`, 63 GB into ordinary memory) - and Janas-LLM through `tests/bench_long`, not `janas-bench`: the same 1,024 tokens of Italian prose and 200 written, 12 threads on CPUs 0-11, a 12 GiB expert cache; one run of llama.cpp, the median of three of Janas-LLM reading with the GPU and the mean of two writing. The drafts, which bring a reply sooner (37 tok/s from Qwen3-Next-80B-A3B in `janas-bench` on 10 October, the expert cache warm; 31-35 in the chat on 21 October), and how every figure was taken: [How the figures were measured](#how-the-figures-were-measured). On another machine they will be other ones: `janas-bench` measures yours.

**New: Intel's NPU reads the prompts.** On the same laptop's NPU Janas-LLM reads 1,024 tokens of Qwen3.5-2B at 729 tok/s, of Gemma-4-E2B at 508 and of Qwen3-4B at 333, 1.5 to 2.6 times as fast as without it, with answers as close to llama.cpp's as the CPU's. It takes the NPU by itself where it measures faster, and three options choose by hand: [Intel's NPU](#intels-npu), [Choosing what runs where](#choosing-what-runs-where).

**Long contexts: at 8,192 tokens Janas-LLM writes a reply at 18.2 tok/s, where llama.cpp manages 4.9, and reads the prompt at 74 tok/s, against 20 - both keeping keys and values at eight bits.** Qwen3-4B Q4_K_M on the same laptop, the CPU alone on both sides (12 threads), the same file, 26-27 October 2026; a prompt read from the start, then 64 tokens written after it:

| Context | reading the prompt: Janas-LLM | llama.cpp | writing after it: Janas-LLM | llama.cpp |
|---|---|---|---|---|
| 1,024 tokens | 118 tok/s | 54 | 27.6 tok/s | 12.1 |
| 4,096 | 96 | 32 | 22.0 | 7.5 |
| 8,192 | 74 | 20 | 18.2 | 4.9 |
| 16,384 | 56 | under 9.1 | 15.0 | - |
| 32,000 | 35 | - | 10.7 | - |

The longer the context, the wider the gap: 2.3 times as fast writing at 1,024 tokens, 3.7 times at 8,192. Both slow down as the context grows, and on both the attention is what grows: at 32,000 tokens it takes 59 of the 93 ms Janas-LLM spends on a token written, and the 32,000 tokens of prompt take 15 minutes to read. Janas-LLM's peak memory went from 2.6 GiB at 1,024 tokens to 4.9 at 32,000. The same conditions on both sides: keys and values at eight bits (llama.cpp's q8_0 cache, with its flash attention, which q8_0 values need). llama.cpp had not finished reading 16,384 tokens after 30 minutes, so its writing there was not measured, nor anything at 32,000. With its default sixteen-bit cache llama.cpp read faster (67, 55, 44 and 31 tok/s at 1,024 to 16,384 tokens) and wrote about as fast (12.5, 7.7, 4.6 and 2.2). How it was measured: [below](#how-the-figures-were-measured).

**New: Gemma 4 whole on the GPU, Llama 3, and `janas-nsm`.** Gemma 4's four models - 12B, the 26B mixture, E2B and E4B - now run every layer of a token on the GPU, with the GPU's norms and quantizations giving the CPU's bits: Gemma-4-12B agrees with llama.cpp at 367 of 400 most likely tokens on the GPU, at 362 on the CPU. Llama 3 models run too, a LoRA adapter merged into one by `tools/hf2gguf`: DeepNSM-1B (Llama-3.2-1B with the adapter of [arXiv 2505.11764](https://arxiv.org/abs/2505.11764)) writes its answers at 57.8 tok/s on the laptop's CPU, where llama.cpp writes 37.9 to 39.8 on the same file. `janas-nsm` asks it for a word's explication in the Natural Semantic Metalanguage - Goddard and Wierzbicka's sixty-odd semantic primes, the words every language has - and counts it as the paper does; it can also restrict the model's output to those words alone. A research tool, and a sign of how far this project means to go: [docs/tools.md](docs/tools.md#janas-nsm).

**Try it** on Linux x86-64 (Ubuntu 22.04, Debian 12 and later):

```sh
curl -LO https://github.com/prabanta-dev/janas/releases/latest/download/janas-linux-x86_64.tar.gz
tar xf janas-linux-x86_64.tar.gz && cd janas
./janas-chat                     # the models it knows: pick one, it is downloaded and checked
```

**Or on Windows** 10 (1903 or later) and 11, x86-64: download [janas-windows-x86_64.zip](https://github.com/prabanta-dev/janas/releases/latest/download/janas-windows-x86_64.zip), unpack it, and in a terminal in its `janas` folder run `.\janas-chat`. On the same laptop under Windows 11 (12 October 2026) Janas-LLM wrote as fast as on Linux (Qwen3-4B 25.6 tok/s, Qwen3-30B-A3B 34.2) and read a prompt as fast with the mixture (Qwen3-30B-A3B 132 tok/s) and 1.8 times as fast with the dense model (Qwen3-4B 346 tok/s, against 193 on Linux that week), whose whole blocks go to the GPU, through Intel's own Windows driver there.

The models come already converted from [Hugging Face](https://huggingface.co/prabanta-dev); building from the sources: [Getting started](#getting-started).

Janas is a family of tools for artificial intelligence, written in C with no dependencies beyond the C library. Its first component, **Janas-LLM**, runs large mixture-of-experts language models on ordinary computers, on the CPU alone or with a GPU, and with the NPU of Intel's Core Ultra where there is one: the weights stay on disk, the experts each token needs are streamed from an NVMe SSD, and the ones that keep coming back stay in memory. An 80-billion parameter model runs, and answers, on a laptop with 32 GB of RAM.

Designed and written by **Maurizio "camauri" Cammalleri**.

---

## ⚠️ Read this first

**This is an experiment, not a product.**

| | |
|---|---|
| **Stage** | Early development. The public API and the command line **change without notice**, and have changed several times a week. The models' files are the exception: the `.jns` files published on Hugging Face (format version 2) will be read by every later version of Janas-LLM. |
| **Stability** | **Not verified.** The engine is deliberately hard on the machine: it fills the free memory with its expert cache and reads the disk at full speed with `O_DIRECT`. On a machine with less headroom than it reckons, the desktop can be pushed into swap and become slow or unresponsive. Using a GPU has, once, caused a driver reset that took the desktop with it. |
| **Data** | It writes only in `~/.cache/janas` (what it learns about your machine, and the system messages it has read, so as not to read them again) and where you tell it to put a converted model. It does not touch the model files it reads. |
| **Hardware** | Developed and run on **one** machine (below). On anything else it is untested: it may be slower, it may refuse the model, it may misbehave. |

If something goes wrong, `Ctrl-C` stops a reply and `Ctrl-D` leaves the chat; `--cache <GiB>` puts a hard limit on the memory it takes, and `JANAS_GPU=0` and `JANAS_NPU=0` keep it off the GPU and off the NPU entirely.

## Intel's NPU

New on 21 October 2026: the NPU of Intel's Core Ultra processors reads the prompts. On the development laptop's (Meteor Lake, "Intel AI Boost"), with `janas-bench --prompt 1024 --gen 16 --rounds 2`, the fastest configuration without the NPU (the GPU's fast prompt included) against the fastest with it:

| Model | reading 1,024 tokens: without the NPU | with it | the NPU made ready in |
|---|---:|---:|---:|
| Qwen3-0.6B | 974 tok/s | **1,421 tok/s** | 7.5 s |
| Qwen3.5-0.8B | 950 tok/s | **1,403 tok/s** | 7.9 s |
| Qwen3.5-2B | 477 tok/s | **729 tok/s** | 17 s |
| Gemma-4-E2B-it | 196 tok/s | **508 tok/s** | 24 s |
| Qwen3-4B | 201 tok/s | **333 tok/s** | 53 s |
| Qwen3-4B, bartowski's Q4_K_M (importance matrix) | 202 tok/s | **333 tok/s** | 46 s |
| Gemma-4-E4B-it | 129 tok/s | **245 tok/s** | 48 s |

On all of them the NPU takes the layers whole (Gemma-4-E4B 41 of its 42). A reply is written as fast as before, by the CPU: its tokens come one at a time, each reading the weights from memory, and the NPU reads the same memory.

**Against llama.cpp on the same NPU.** llama.cpp's OpenVINO backend (build `2b18470`, OpenVINO 2026.4, `GGML_OPENVINO_DEVICE=NPU`) reads and writes on the NPU. From 768 tokens of prompt on it stopped on a segmentation fault, so the two were compared on 512, Janas-LLM with `--prompt-on npu`; the Qwen3.5 models and Gemma 4's did not run on it ("failed to decode prompt batch"):

| Model | reading 512 tokens: Janas-LLM | llama.cpp | writing: Janas-LLM (CPU) | llama.cpp (NPU) |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1,542 tok/s | 419 tok/s | 136 tok/s | 12.0 tok/s |
| Qwen3-4B | 364 tok/s | 120 tok/s | 26.6 tok/s | 5.0 tok/s |
| Qwen3-4B, bartowski's Q4_K_M (importance matrix) | 369 tok/s | 121 tok/s | 28.5 tok/s | 4.9 tok/s |
| Qwen3.5-0.8B | 1,325 tok/s | did not run | 111 tok/s | - |
| Qwen3.5-2B | 722 tok/s | did not run | 54.0 tok/s | - |
| Gemma-4-E2B-it | 532 tok/s | did not run | 37.1 tok/s | - |
| Gemma-4-E4B-it | 236 tok/s | did not run | 19.8 tok/s | - |
| Qwen3.5-9B | 64.7 tok/s | did not run | 13.7 tok/s | - |
| Gemma-4-12B-it | 35.2 tok/s | did not run | 10.2 tok/s | - |

**How it works.** The NPU runs graphs its driver compiles. Janas-LLM writes them itself, in OpenVINO's IR format, and opens Intel's Level Zero driver when a model opens: nothing is linked against it, and without it nothing changes. The weights the NPU takes are copied in eight bits, with a scale for each row, and computed in sixteen-bit floats, the norms in 32 bits. A dense model whose layers fit in the NPU's memory, nine in ten at least (about 3.7 GB), goes to it layer by layer, whole, the layers left over to the CPU: each block of the prompt in two halves, the NPU computing a layer on one half while the CPU computes the attention, or Qwen3.5's recurrence, on the other. A larger model, or a mixture of experts, gives it the products of the weights that stay in memory - attention, recurrent layers, shared experts - and leaves the routed experts to the CPU and the GPU. It takes passes of 64 tokens and more: prompts, not a reply's tokens nor the checks of the drafts.

**When the engine uses it.** It measures the NPU among its configurations and keeps it where it reads faster: on this laptop, the dense models whose layers go to it whole. Where it takes products, on the mixtures and the larger dense models, the GPU's fast prompt read faster here (Qwen3-30B-A3B 133 tok/s, against 90-110 with the NPU), and the engine kept the GPU. Against llama.cpp's scores on two texts, Italian prose and code, the most likely token agreed at as many positions with the NPU as with the CPU, give or take 7 of 512, on the eight models checked (Qwen3.5-0.8B and 2B, Qwen3-4B, Gemma-4-E2B and E4B, Qwen3-30B-A3B, Qwen3.6-35B-A3B, Qwen3-Next-80B-A3B). On Gemma 4's larger models, which go product by product, it agreed less (Gemma-4-26B-A4B: 363 against 398): there the NPU reads only when asked, with `--prompt-on npu`. Where the expert cache needs the memory the NPU's eight-bit copies would take - the mixtures whose experts do not all fit in memory, and Qwen3.8-27B, which wrote at 1.7 tok/s instead of 4.4 with them - the NPU is not opened unless `--prompt-on` names it.

**The first prompt waits** while the NPU is made ready, once a session: the weights it takes copied and its graphs compiled, the times in the table above. The chat says how far it has got on its status line ("preparing the NPU 12/25, 8 s") and how long it took; `janas-bench` makes it ready before measuring.

**What it needs.** Linux on an Intel Core Ultra: only the first generation's NPU (Meteor Lake) has been tried, not Lunar Lake's or Arrow Lake's. Intel's NPU driver, the `intel-level-zero-npu` and `intel-driver-compiler-npu` packages of [linux-npu-driver](https://github.com/intel/linux-npu-driver) (1.38.0 here), a Level Zero loader (`libze1`, 1.32.0 here), and your user in the `render` group. `janas-bench` names the NPU on its `NPU:` line, or says why there is none to use; `JANAS_NPU=0` keeps the engine off it, and `JANAS_NPU_LOADER` names a loader outside the library path. The Windows build has no NPU path yet.

## Choosing what runs where

The engine chooses for each kind of pass how many threads to run and whether the GPU and the NPU work beside the CPU, by measuring them on your machine, and keeps what it learns in `~/.cache/janas`. Three options choose by hand, the same in `janas-chat`, `janas-server` and `janas-bench`:

| Option | the passes it sets |
|---|---|
| `--prompt-on <devices>` | reading a prompt: passes of 17 tokens and more, the NPU taking those of 64 and more |
| `--drafts-on <devices>` | checking drafts: passes of 2 to 16 tokens |
| `--reply-on <devices>` | writing a reply: a token at a time |

`<devices>` is `auto` (the default: the engine measures and chooses), `cpu`, `gpu` or `npu`, two joined by `+` to work in the same pass (`npu+gpu`), or a list apart by commas, among which the engine measures and keeps the fastest (`npu,gpu`). The CPU takes part in every pass. The NPU reads prompts only: asked for drafts or a reply it is left out, with a warning, and so is a device the machine does not have. Without `gpu` in `--prompt-on` the GPU's fast prompt, which otherwise reads every prompt's blocks, is off too: `--prompt-on npu` is the NPU and the CPU alone. `--reply-on gpu` gives an integrated GPU a reply's single tokens, which it does not take by itself. A choice holds over `--mode`; `/save-config` keeps it in `janas-chat`, and `JANAS_PROMPT_ON`, `JANAS_DRAFTS_ON` and `JANAS_REPLY_ON` are the same as variables. A device kept out entirely: `--no-gpu` or `JANAS_GPU=0`, and `JANAS_NPU=0`.

```sh
./janas-chat qwen3-4b --prompt-on npu          # prompts on the NPU and the CPU, the rest as the engine measures
./janas-bench qwen3.5-2b --prompt-on npu,gpu   # both measured for the prompts, the faster kept
```

## The machine it grew on

Everything in this repository was written and measured on:

- **Linux Debian 13** (trixie), x86-64, kernel 6.12.111 (6.12.107 for the figures taken before 21 October); GPU driver Mesa 26.1.6 (Vulkan); NPU driver Intel's linux-npu-driver 1.38.0; built with GCC 14.2
- **Intel Core Ultra 9 185H** (6 performance cores, 8 efficiency, 2 low-power; AVX2, FMA, F16C, AVX-VNNI; integrated Arc GPU; NPU)
- **32 GB** of DDR5 RAM
- **1 TB NVMe SSD** (about 5.7 GB/s reading)

The Windows build was run on the same laptop under Windows 11 (build 26300), with Intel's own graphics driver: its tests pass there, and how fast it was is under **Or on Windows**, above.

Everything the engine does is chosen from what it measures, so another machine will make other choices — that part is meant to travel. What has never been tried is the rest: other CPUs (especially without AVX-VNNI, or AMD), discrete GPUs, slower disks, more or less memory. **If you run it somewhere else, the report from `janas-bench` is the single most useful thing you can send back** (see [Helping it run on more hardware](docs/hardware.md#helping-it-run-on-more-hardware)).

<a id="what-works-now"></a>

## How the figures were measured

What Janas-LLM runs today, model by model and feature by feature: [docs/features.md](docs/features.md).

The table at the top, on 21 October 2026: `janas-bench --prompt 1024 --gen 16 --rounds 2 --no-mtp --settle 300`, which measures each configuration the engine may choose (6, 12 and 20 threads, with the GPU in the shorter passes or not, with the NPU or not), the best of each column taken, against llama.cpp (build `2b18470` of 18 September 2026) with its Vulkan backend on the same integrated GPU, everything on it (`llama-bench -p 1024 -n 16 -r 2 -t 12 -ngl 99`), on the same files the same day. The writing columns are without drafts on both sides. The two tools do not measure in quite the same way (`llama-bench` writes without a prompt before it, `janas-bench` after the 1,024 tokens), and it is one machine. The conditions are the same wherever the two engines allow it, and where they are not, it is said.

The long contexts, on 26 October 2026: Janas-LLM through `tests/bench_long` (`JANAS_GPU=0`, 12 threads on CPUs 0-11, a 4 GiB expert cache), a prompt of n tokens read in blocks from position 0, then 64 written greedily one at a time; llama.cpp (build `2b18470`, CPU backend, `-t 12 -ngl 0 -fa 1 -ctk q8_0 -ctv q8_0`; its defaults for the sixteen-bit figures) with `llama-bench -p n -n 0` for reading and `-p 0 -n 64 -d n` for writing after n tokens, each run stopped after 30 minutes; the same Qwen3-4B Q4_K_M file (Qwen's own GGUF), the same tokens (English Wikipedia articles, C sources and Italian prose, mixed), one run each, ten seconds apart, the machine otherwise idle.

Qwen3-Next-80B-A3B and Qwen3-Coder-Next are larger than the laptop's memory. llama.cpp with its usual build ran out of memory as soon as the GPU took part, even with the experts left on the CPU (`-ngl 99 -ncmoe 48`): its CPU backend copies the weights into memory of its own to repack them, and the Vulkan backend copies those it leaves to the CPU into buffers the system cannot reclaim. It ran with a build without repacking (`GGML_CPU_REPACK=OFF`) and `--no-host 1 -lm mmap`, which leave the experts in the file, read through the page cache: 17.7 tok/s reading with the GPU, 3.0 writing on the CPU alone (1.8 with the GPU; its CPU backend read at 12.6). Janas-LLM ran them with no option of its own, the experts it uses most in a cache of 21 GiB and the rest read from the disk.

**The drafts** are tokens guessed ahead that the model checks, all at once: the same text, sooner. They come from the model's own prediction block (the Qwen3.5 and Qwen3.6 models, Qwen3-Next) or from Gemma 4's assistant, a small drafting model of its own; the whole of it is under [`--mtp`](docs/chat.md). Measured on 10 October 2026 by `janas-bench` on a chat reply - an ordinary question in English ("explain how bread is made at home"), 64 tokens of the answer, sampled as the chat samples with one fixed seed, after a prompt of 256 tokens had warmed the expert cache, in the configuration the engine chose (on Qwen3-Next-80B-A3B 20 threads for every pass) - the reply came at 37 tok/s from Qwen3-Next-80B-A3B, 34 from Qwen3.6-35B-A3B, 26 from Gemma-4-26B-A4B, 14 from Gemma-4-12B, 24 from Qwen3.5-9B, 32 from Gemma-4-E4B, 52 from Gemma-4-E2B and 67 from Qwen3.5-2B. In the chat, on 21 October, the same question brought Qwen3-Next-80B-A3B's reply at 31-35 tok/s with the cache warm (31.4 the median of five seeds, its single tokens on the 12 threads the engine chose there) and at 16-20 tok/s as the first reply of a session, its experts still coming from the disk. What the drafts are worth depends on the text: there 74-89% of them were kept on the Qwen models and on Gemma's two larger ones, 63-72% on E2B and E4B, while on Italian prose Qwen3.5-2B kept about a third and gained next to nothing; and one round of 64 tokens is noisy, a tenth either way. Qwen3-30B-A3B and the dense Qwen3 have no such block (Qwen3-4B can take Qwen3-0.6B as its draft, `--draft`); the dense Qwen3.5 have one in their original checkpoint, which `hf2jns_mtp` converts ([MODELS.md](MODELS.md#the-fingerprints)).

**The threads.** On the first passes of each kind the engine tries every core, one thread per performance core and the performance cores' threads, keeps the fastest, and tries the others now and then, so that a machine that warms up can change its verdict. Here, on 21 October 2026, it settled on all 20 usable threads for a reply's tokens on 12 of the 17 files and on the performance cores' 12 on the other five, and on 6 to 20 threads for a prompt, beside the GPU or the NPU; on an i9-14900HX every thread replied at less than half the speed of the performance cores' 16.

**An idle machine first.** The expert cache takes the memory that is free when the model is opened, and what is left of the model is read from disk while it answers, so everything else running takes its share. It degrades gently rather than breaking: with a browser and an editor open, the same model kept about three quarters of its speed and waited five times longer on the disk. But the first thing you see should be the machine's real speed, and any measurement you mean to send to others has to be taken with the machine to itself.

## Getting started

```sh
sudo apt install build-essential          # Debian/Ubuntu
sudo apt install libvulkan-dev glslang-tools vulkan-tools   # optional, GPU (glslang 15.2 or later: INSTALL.md)
sudo pacman -S vulkan-headers glslang vulkan-tools           # the same, Arch

./build.sh                                # everything into bin/x86_64-linux/
./build.sh release test                   # and run the tests
```

## Documentation

<a id="getting-a-model"></a>
<a id="using-the-chat"></a>
<a id="janass-services"></a>
<a id="tools-from-mcp-servers"></a>
<a id="watching-a-long-prompt"></a>
<a id="filling-in-code"></a>
<a id="using-it-from-a-program"></a>
<a id="from-c"></a>
<a id="from-freebasic-and-basic-modern"></a>
<a id="mcp-servers-from-a-program"></a>
<a id="janas-as-an-mcp-server"></a>
<a id="using-it-over-http"></a>
<a id="measuring-your-machine"></a>
<a id="tools"></a>
<a id="helping-it-run-on-more-hardware"></a>

- [INSTALL.md](INSTALL.md) — what the build needs, how to prepare a model, and the first run
- [ChangeLog.md](ChangeLog.md) — what changed, newest first
- [MODELS.md](MODELS.md) — what a converted model's licence is, and what may be redistributed
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to help, and why code from contributors waits for now
- [src/services](src/services/README.md) — Janas's services, each with its page, in `janas-chat` and in other MCP clients
- [docs/features.md](docs/features.md) — what Janas-LLM runs today, model by model and feature by feature
- [docs/models.md](docs/models.md) — getting a model: `janas-get`, or a GGUF file converted, and which files were checked
- [docs/chat.md](docs/chat.md) — `janas-chat`: its options and commands, the drafts, Janas's services, tools from MCP servers, a long prompt watched as it is read
- [docs/library.md](docs/library.md) — using it from a program, in C, in FreeBASIC and BASIC MODERN, or in Node.js and TypeScript, and MCP servers from a program
- [examples/node-chat](examples/node-chat/README.md) — a chat in TypeScript on the module for Node.js: the model inside the server's process, its replies streamed to the browser
- [examples/fb-chat](examples/fb-chat/README.md) — a chat in FreeBASIC, procedure by procedure and with Janas's classes
- [docs/serving.md](docs/serving.md) — the model as tools for Claude Code, editors and other MCP clients (`janas-mcp`), and over HTTP, an API that follows OpenAI's, Ollama's and Anthropic's (Claude Code), with a chat page to try it from a browser (`janas-server`)
- [docs/editor.md](docs/editor.md) — Continue, Cline, Roo Code, Claude Code, aider and Open WebUI with a model of this machine, each set up and tried on the same task; your files outside the editor (`janas-files`)
- [docs/code-completion.md](docs/code-completion.md) — code filled in at an editor's cursor, in VSCodium and VS Code through `janas-server`: 1.5-2.2 s a proposal on a laptop, where a window slid a line was read again in 17 s
- [docs/tools.md](docs/tools.md) — the converters, checkers and benchmarks built beside the programs
- [docs/hardware.md](docs/hardware.md) — measuring your machine with `janas-bench`, and helping it run on more hardware: the machines wanted most, the one command that tries Janas and writes a report, and what to send
- [docs/reports.md](docs/reports.md) — what testers measured on their own machines
- [docs/janet-lr-study.md](docs/janet-lr-study.md) — Janet, a new feed-forward block: a first comparison with SwiGLU, GeGLU and ReLU² across learning rates
- [AUTHORS](AUTHORS) — who wrote it, and what it owes to others
- [LICENSE](LICENSE) — GNU GPL, version 3 or later

## Credits

Janas is designed and written by **Maurizio "camauri" Cammalleri** ([LinkedIn](https://www.linkedin.com/in/maurizio-cammalleri-80a89a11/), [Substack](https://cammalleri.substack.com/)).

It stands on the work of others, and says so:

- **The models** it runs are their authors': Qwen3, Qwen3.5, Qwen3.6 and Qwen3-Next by the Qwen team of Alibaba Cloud, Gemma 4 by Google, Llama 3.2 by Meta (under the Llama 3.2 Community License); DeepNSM's adapter by the authors of [arXiv 2505.11764](https://arxiv.org/abs/2505.11764) (CC BY-NC-SA: research, not commercial use); the GGUF quantizations by Qwen, bartowski and unsloth ([MODELS.md](MODELS.md) has every source).
- **llama.cpp and ggml** are the reference the engine is checked against (logits, tokens, speed); none of their code is in Janas. The one thing taken from ggml is data that defines a format: the IQ3_S grid, in `src/llm/iq3s_grid.h`, under ggml's MIT licence.
- **The Unicode Character Database** gives the tokenizer its tables (`tools/gen_unicode`), and **OpenAI's OpenAPI specification** gives `janas-server` its route table (`tools/openapi_routes`).
- **GNU libmicrohttpd, yyjson and Mbed TLS** are compiled into the programs named below.
- **Intel's NPU driver** ([linux-npu-driver](https://github.com/intel/linux-npu-driver)) compiles and runs the NPU's graphs, opened when a model opens if it is installed; Janas-LLM writes the graphs itself in OpenVINO's IR format, and declares the Level Zero functions it calls from the oneAPI Level Zero specification (`src/common/npu_ze.h`): no code of OpenVINO's or of Level Zero's is in Janas.
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

[CONTRIBUTING.md](CONTRIBUTING.md). The short version: a measurement from a machine nobody here owns is worth more than most patches, and code from contributors cannot be merged until the non-profit foundation that will hold Janas's copyright is set up.
