# Installing Janas

Janas-LLM runs large mixture-of-experts language models on ordinary computers: the model's weights stay on disk and the experts each token needs are streamed from an NVMe SSD, with the useful ones kept in memory. This file explains how to build it, prepare a model, chat with it and measure your machine.

## Requirements

- **Linux on x86-64.** Windows is not supported yet.
- **CPU:** any x86-64 runs it; AVX2 with FMA and F16C (Intel since 2013, AMD since 2015) is needed for good speed, and AVX-VNNI is used when present.
- **Memory:** at least 16 GB; 32 GB or more for large models (Qwen3-Next-80B-A3B: about 2 GB of resident weights, plus as much expert cache as the memory allows).
- **Disk:** the model on an **NVMe SSD** (the experts are read directly from it while generating).
- **GPU (optional):** an integrated or discrete GPU with a **Vulkan 1.3** driver. The engine works fully without one; it measures whether the GPU helps on your machine and uses it only then.

## Building

On Debian or Ubuntu:

```sh
sudo apt install build-essential
# optional, for GPU support:
sudo apt install libvulkan-dev glslang-tools vulkan-tools
```

On Fedora: `sudo dnf install gcc make` and, for the GPU, `vulkan-loader-devel glslang vulkan-tools`. On Arch: `sudo pacman -S base-devel` and `vulkan-headers vulkan-icd-loader glslang vulkan-tools`.

The GPU also needs its Vulkan driver: Mesa (`mesa-vulkan-drivers` on Debian/Ubuntu) for Intel and AMD, the proprietary driver for NVIDIA. `vulkaninfo --summary` should list your GPU with Vulkan 1.3 or later. When the engine takes none, `janas-bench` and the model's description say why: a build without GPU support (`glslangValidator` was missing: install it and build again), no Vulkan loader, a GPU whose driver lacks something (named), or the GPU turned off.

Then, in the source directory:

```sh
./build.sh
```

Everything goes to `bin/x86_64-linux/`: `janas-chat`, `janas-server`, `janas-mcp`, `janas-bench`, the libraries `libjanas_llm.so` (C API in `include/janas/llm.h`) and `libjanas_mcp.so` (the MCP client, `include/janas/mcp.h`) and the conversion tools. Without `glslangValidator` the build still succeeds, without GPU support. `./build.sh release test` also runs the tests, and `./build.sh --help` lists the other variants — a debug build, and the three sanitizers `asan`, `ubsan` and `tsan` — with what each one catches.

## Preparing a model

Janas reads its own file format (`.jns`), converted from a GGUF file. Supported now: **Qwen3.5 and Qwen3.6** (`qwen35moe`, e.g. Qwen3.6-35B-A3B), **Qwen3-Next** (`qwen3next`, e.g. Qwen3-Next-80B-A3B-Instruct and Qwen3-Coder-Next) and **Qwen3 MoE** (`qwen3moe`, e.g. Qwen3-30B-A3B), in the Q4_K, Q5_K, Q6_K and Q8_0 formats (so the usual **Q4_K_M** files).

Keep models in `models/` next to the sources: the tools take a path, so
anywhere works, but that is where the examples here and in the
[README](README.md#getting-a-model) put them. The README also lists, for each
model, which Hugging Face repository to take the GGUF from and how large it is.

1. Download a Q4_K_M GGUF of the model from Hugging Face, into `models/gguf/`.
2. Convert it with the converter the build made (with a model split into
   several files, give the first):

   ```sh
   bin/x86_64-linux/gguf2jns models/gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
       models/qwen3-next.jns
   ```

   You need room for both while it runs: about twice the model.

3. Optional, for Qwen3-Next: its multi-token prediction block makes replies faster (drafts of the next tokens, verified by the model: the text is the same). GGUF files leave that block out, so take it from the original checkpoint, [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), whose last shard holds it:

   ```sh
   bin/x86_64-linux/hf2jns_mtp models/qwen3-next.jns \
       models/hf/model-00041-of-00041.safetensors models/qwen3-next-mtp.jns
   ```

Qwen3.5 and Qwen3.6 keep their prediction block inside the model file: it is converted and used with no further step.

Keep the `.jns` files on the NVMe SSD: the experts are read from there while the
model answers, and a slower disk is felt at every token.

## Chatting

```sh
bin/x86_64-linux/janas-chat qwen3-next.jns --mtp qwen3-next-mtp.jns
```

Type a message and press Enter; a backslash and Enter add a line to it instead of sending it, and the backslash is not part of what the model reads. Commands start with `/`: `/help` lists them, `/stats` shows the speed of every reply, `/mode` the power mode (`auto`, `eco`, `max`) and what the engine chose, `/think` turns reasoning on or off for models that reason (their reasoning is printed in grey), `/experts n` sets how many experts each token uses (fewer: faster replies, a little less accurate). Ctrl-C stops a reply, Ctrl-D quits. `janas-chat` with no arguments lists the options (context length, memory for the expert cache, sampling).

The engine tunes itself while it runs: it tries the configurations your machine offers (how many threads, whether the GPU takes part) on real work and keeps the fastest. The first minutes of use are the learning phase; the choices are kept in `~/.cache/janas/`, per machine and model. Afterwards it still tries an alternative now and then, to follow the machine's state (heat, power source), and ever more rarely while the alternatives keep losing.

## Measuring your machine

```sh
bin/x86_64-linux/janas-bench qwen3-next.jns --mtp qwen3-next-mtp.jns
```

It takes a few minutes: it measures prompt processing and generation speed for every configuration, fills the tuning profile (so chatting starts tuned) and writes a report, `janas-bench-<date>.txt`, in the current directory. The report lists your processor, memory, GPU, the speeds and the choices: send it along with any feedback.

## Troubleshooting

- `JANAS_GPU=0` turns the GPU off; `JANAS_GPU_DEVICE=integrated` or `discrete` picks one when there are both.
- `JANAS_KERNELS=avx2` uses the AVX2 kernels even where AVX-VNNI is available.
- `JANAS_TUNE=0` turns self-tuning off (fixed defaults); delete `~/.cache/janas/tuning.txt` to start tuning from scratch.
- The expert cache takes the free memory, leaving a fifth of the machine's memory to the rest of the system; `--cache <GiB>` in `janas-chat` sets it.
