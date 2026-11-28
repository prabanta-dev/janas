# Janas for macOS on Apple Silicon

> **Download Janas only from github.com/prabanta-dev/janas** and its releases; its models are at huggingface.co/prabanta-dev. Other sites using its name are not the project's.

Janas-LLM, the inference engine of Janas, runs large language models on ordinary computers, mixtures of experts larger than the memory among them: the weights stay on disk, the experts each token needs are read from the SSD and the ones that keep coming back stay in memory. It is written in C and needs nothing but the system's C library.

**Janas for macOS uses the CPU (its NEON and dot product kernels) and Apple's GPU through Metal**, built and tested on GitHub's Macs: there every product computed by Metal matched the CPU's bit for bit, and `janas-bench` ran a model with the GPU allowed (`JANAS_GPU=1`). Brebus has a window of its own on macOS, built on GitHub's Macs but not yet tried on a real Mac's screen. Its speed on a Mac has not been measured yet: `./janas-bench` measures yours, and its report is welcome (<https://github.com/prabanta-dev/janas/issues>).

## Open it the first time

The programs are not signed by Apple yet, so macOS stops them the first time. Either:

- in a terminal, in the folder you unpacked: `xattr -dr com.apple.quarantine .` - then they start as any other program; or
- start one, then System Settings, Privacy & Security, *Open Anyway*, for each program.

## Start

```sh
./janas-get list                 # the models it knows, their size, the memory they want
./janas-get qwen3-4b             # downloads it, checked, into ~/.local/share/janas/models
./janas-chat qwen3-4b            # and talk to it
./janas-bench qwen3-4b           # how fast this machine runs it: a report to send back
./brebus                         # or all of it in a window: models, a chat, the server
```

`./janas-chat` with no model shows the list and downloads the one you choose, after asking. The models come already converted from Hugging Face ([prabanta-dev](https://huggingface.co/prabanta-dev)), each file checked against the fingerprints of `MODELS.md`.

## What is here

The same programs as for Linux and Windows (`docs/` says what each does): `brebus`, `janas-chat`, `janas-server`, `janas-get`, `janas-setup`, `janas-bench`, `janas-mcp`, the converters (`gguf2jns`, `hf2jns_*`, `silero2jns`, `wespeaker2jns`), `jns_check`, `jns_planes`; Janas's services in `services/`; `libjanas_llm.dylib` and `libjanas_mcp.dylib` with their C headers and FreeBASIC bindings in `lib/` and `include/`. Not yet on macOS: the module for Node.js, the sign-in of other users of the computer to Brebus, and the cage of the model's Python.

## Needs

- macOS 14 or later on Apple Silicon (M1 and later).
- An SSD for the models larger than the memory: every Mac with Apple Silicon has one.

## Licence

Janas is under the GNU General Public License, version 3 or later: `LICENSE`. The models are not part of it and keep their own licences: `MODELS.md`.
