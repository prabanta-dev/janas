# Getting a model

**The short way: `janas-get`.** One command downloads a model already converted, from the project's own repositories on Hugging Face ([prabanta-dev](https://huggingface.co/prabanta-dev)), with its prediction file (an MTP block, Gemma 4's assistant, or for Qwen3-4B the small Qwen3-0.6B that guesses its tokens), which the chat then finds by itself, and checks the SHA-256 of every file against [MODELS.md](../MODELS.md). The models go to `~/.local/share/janas/models`, or wherever `JANAS_MODELS` or `--dir` says:

```sh
bin/x86_64-linux/janas-get list                  # the models it knows
bin/x86_64-linux/janas-get gemma-4-e4b           # into ~/.local/share/janas/models
bin/x86_64-linux/janas-chat ~/.local/share/janas/models/gemma-4-e4b-it-q4km.jns
```

A download stopped halfway resumes where it stopped, and a file already there is checked rather than fetched again. `--from-gguf` takes the other way: the original GGUF the catalog names, downloaded and converted on your machine to the very same fingerprints (the GGUF goes once converted, `--keep` keeps it); `janas-get` takes it by itself where a converted file is not on Hugging Face. Any other file: `janas-get hf:<owner>/<repo>/<file.jns|file.gguf>`, checked against the SHA-256 Hugging Face lists for it. The memory column of `janas-get list` is the smallest machine each model is meant for. The rest of this section is the same by hand.

Janas-LLM reads its own format, `.jns`, converted from a **Q4_K_M** GGUF (Q3_K, IQ3_S, IQ4_NL, IQ4_XS, Q4_0, Q4_1 and Q5_1 convert too, and so do unsloth's "UD" files made of them and the "MXFP4_MOE" files, whose experts are MXFP4; IQ2 and IQ1 types, not yet). Keep both in `models/` next to the sources — the tools take a path, so anywhere works, but that is where these examples put them, and where the project keeps its own.

| Model | GGUF converted here | GGUF | `.jns` | Memory it likes |
|---|---|---|---|---|
| **Qwen3-Next-80B-A3B-Instruct** — most of this engine was measured on it | [Qwen](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF) | 48.4 GB | 48.4 GB | 32 GB |
| **Qwen3.6-35B-A3B** — the easiest to start with | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) | 22.3 GB | 22.3 GB | 16-32 GB |
| **Qwen3-Coder-Next** | [Qwen](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) | 48.4 GB | 48.4 GB | 32 GB |
| **Qwen3-Coder-30B-A3B-Instruct** — for code completion in the editor ([docs/code-completion.md](code-completion.md)) | [unsloth](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 18.6 GB | 18.6 GB | 32 GB, to keep it whole in memory |
| **Qwen3-30B-A3B** | [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF) | 18.6 GB | 18.6 GB | 16 GB |
| **Qwen3-30B-A3B with an importance matrix** (`qwen3-30b-a3b-imatrix`) — bartowski's Q4_K_M: the nearest to Qwen's own Q8_0 of the three here (400 tokens: code 388, Italian 367, against 376 and 335 for the Q4_K_M above), 8-10% slower held in memory: it keeps more matrices in Q6_K | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF) | 18.6 GB | 18.6 GB | 16 GB |
| **Qwen3-30B-A3B in three bits** (`qwen3-30b-a3b-q3`) — unsloth's dynamic UD-Q3_K_XL: against Qwen's own Q8_0 it agrees more often than the Q4_K_M above (400 tokens: code 386 against 376, Italian 360 against 335), with 27% fewer bytes of experts; with the expert cache held at 2 GiB on this 32 GB machine, 19.8 tokens/s against 13.6; held whole in memory, as fast (32.0-32.2 tokens/s against 32.1-32.2) | [unsloth](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) | 13.8 GB | 13.8 GB | 8-16 GB |
| **Qwen3.5-9B** — dense; its prediction block from the checkpoint ([MODELS.md](../MODELS.md#the-fingerprints)) | [unsloth](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | 5.7 GB | 5.7 GB | 16 GB |
| **Qwen3.5-2B** — dense, small and quick; a prediction block too | [unsloth](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | 1.3 GB | 1.3 GB | 8 GB |
| **Qwen3-4B** — dense, and the smallest here that answers well | [Qwen](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | 2.5 GB | 2.5 GB | 8 GB |
| **Qwen3-4B with an importance matrix** (`qwen3-4b-imatrix`) — bartowski's Q4_K_M, the same weight types and size as Qwen's: against Qwen's own Q8_0 (400 tokens) code 381 against 364, Italian 351 against 329, at the same speed | [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF) | 2.5 GB | 2.5 GB | 8 GB |
| **Qwen3-0.6B** — not to talk to: to draft for a dense one (`--draft`) | [unsloth](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) | 0.40 GB | 0.40 GB | with the model above |
| **Qwen3-Embedding-0.6B** — not to talk to: for embeddings (`janas-server --embedding-model`); published in Q8_0 and converted as it is | [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) | 0.64 GB | 0.64 GB | beside the chat model |

Qwen3 has other dense models - 1.7B, 8B, 14B, 32B - which declare the same architecture as the 4B and so should convert and run the same way. **None of them has been tried here**, and until one is, that is a reading of the file and not a claim. (Qwen3-30B-A3B is not one of them: the number is close but it is a mixture of experts, with an architecture of its own.)

Any other Q4_K_M of the same models converts and runs just as well, including the "dynamic" ones that give different tensors different types: the engine reads Q4_K, Q5_K, Q6_K and Q8_0, and refuses a file carrying anything else rather than guessing. The four above are the files the numbers in this repository were measured on, and the ones [MODELS.md](../MODELS.md) gives fingerprints for, so that a converted file can be checked against the one measured here without downloading anything.

All four models are published under the Apache 2.0 licence by Alibaba Cloud, and quantized into GGUF by them or by the people above. Their licence is theirs, not this project's: [MODELS.md](../MODELS.md) says what that means if you pass a converted file on to somebody else.

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

Both steps are deterministic, so the file can be checked against the fingerprints in [MODELS.md](../MODELS.md).

Room needed: three copies of the model at the worst moment — the GGUF, the converted file and the one with the planes — so about 145 GB for the largest. The flat file goes as soon as the planes are written, and the GGUF can go too, though keeping it saves the download the day a new version of the converter is worth running.

**The vision encoder** (Qwen3.5 and Gemma 4: E2B, E4B, 12B and 26B-A4B; pictures in janas-chat with `/image`, in janas-server's messages, `janas_llm_chat_image` in the library). Like the MTP block it comes from the original checkpoint, which `hf2jns_vision` reads from the directory a download from Hugging Face makes (`config.json`, `preprocessor_config.json` or `processor_config.json`, and the safetensors; of Gemma's checkpoint, which holds the whole model, only the vision tower is read):

```
bin/x86_64-linux/hf2jns_vision models/hf/Qwen/Qwen3.5-0.8B \
    models/jns/qwen3.5-0.8b.vision.jns
bin/x86_64-linux/hf2jns_vision models/hf/google/gemma-4-E2B-it \
    models/jns/gemma-4-e2b-it.vision.jns
```

The engine finds it beside the model: the `.vision.jns` whose name is the longest beginning of the model's (`qwen3.5-0.8b.vision.jns` serves `qwen3.5-0.8b-q4km.jns` and every other quantization of it). For Qwen3.5-0.8B it is 202 MB, its matrices kept in bfloat16 as the checkpoint has them. Its output against llama.cpp's (an mmproj made from the same checkpoint): cosine 0.999997 on average. Gemma 4 E2B's is 338 MB; a picture is made at most 280 tokens, as its processor does, and its output against llama.cpp's is at a cosine of 0.999999 on average, 0.99999 at worst, on pictures of 256 and 252 tokens. E4B is converted the same way and answers the same questions about the same pictures. The 26B-A4B too (1.1 GB, 27 blocks; cosine 0.999998 on average against llama.cpp): in its language model a picture's tokens see one another, the later ones too, so the engine, finding its encoder beside it, reads prompts in blocks that hold a picture whole (320 tokens). The 12B's has no blocks - a patch of 48x48 pixels normed, through one dense layer, plus its positions - 100 MB, at a cosine of 1.000000 against llama.cpp's, its pictures attended both ways as the 26B's.

**Speech to text: Whisper** (`janas-server --speech-model`, `janas_llm_speech_*` in the library). OpenAI's models from their Hugging Face checkpoints (`openai/whisper-small`, `openai/whisper-large-v3-turbo` and the others: `config.json`, `generation_config.json`, `preprocessor_config.json`, `tokenizer.json` and the safetensors), converted by `hf2jns_whisper`:

```sh
bin/x86_64-linux/hf2jns_whisper models/hf/openai/whisper-small \
    models/jns/whisper-small.jns
```

The matrices are kept in half precision, as OpenAI released them (the checkpoints' floats are their exact copies), laid out as the engine reads them; whisper-small is 489 MB. Against whisper.cpp on the same weights and the same sound, the logits of every token of a transcription agree to a cosine of 1.000000 on average, 0.999996 at worst (whisper.cpp without its half-precision flash attention, which alone takes it to 0.99996), and the spectrograms are the same. On John F. Kennedy's 11 seconds, whisper-large-v3 (whose 128 mel filters `hf2jns_whisper` computes as librosa does, to the bit of whisper.cpp's), large-v3-turbo and small.en at a cosine of 0.999991, 0.999933 and 0.999990 on average, the most likely token the same at all of their 29, 30 and 28 positions; the tokenizer splits text as Hugging Face's does. The transcription goes window by window as OpenAI's does (language found, timestamps, temperatures raised where the text loops or is unlikely, silence skipped). A recording has no length limit: one whose decoded sound would take more than a quarter of the memory free is not kept whole but decoded again from its bytes for each pass over it (the spectrogram's largest value, the voice activity, the windows, the speakers), a few seconds of it in memory at a time, and gives the same result (`JANAS_SPEECH_STREAM=1` reads every recording so, `=0` none).

**Voice activity detection: Silero VAD** (chunking a recording, `chunking_strategy` in janas-server, `chunking` in the library), from its ONNX model (`onnx-community/silero-vad`, MIT, 2.2 MB), converted by `silero2jns` into a file beside the speech model, where it is found by itself:

```sh
bin/x86_64-linux/silero2jns models/hf/onnx-community/silero-vad/onnx/model.onnx \
    models/jns/silero-vad.jns
```

The weights of its 16 kHz model in single precision as the ONNX file holds them (read by Janas's own protobuf reader); its probability of speech in each 32 ms agrees with onnxruntime's on the same file to 1.7e-6 over 46 seconds of speech.

**Speaker embeddings: WeSpeaker ResNet34-LM** (who speaks when: `diarized_json` in janas-server, `diarize` in the library), from its ONNX model (`Wespeaker/wespeaker-voxceleb-resnet34-LM`, by the WeSpeaker project, trained on VoxCeleb2, CC-BY-4.0: credit to WeSpeaker, https://github.com/wenet-e2e/wespeaker), converted by `wespeaker2jns` into a file beside the speech model:

```sh
bin/x86_64-linux/wespeaker2jns models/hf/Wespeaker/wespeaker-voxceleb-resnet34-LM/voxceleb_resnet34_LM.onnx \
    models/jns/wespeaker-resnet34.jns
```

Its convolutions in half precision (13 MB); on the same filter banks its embedding agrees with onnxruntime's to a cosine of 0.9999993, and Janas's filter banks with Kaldi's (kaldi-native-fbank) to 1.6e-4.

**The multi-token prediction block** (Qwen3-Next and the dense Qwen3.5; optional, faster replies). Its own model carries it, GGUF files leave it out, so it comes from the original checkpoint. `hf2jns_mtp` can read just the block's tensors from Hugging Face, with range requests, instead of the shards that hold them:

```sh
bin/x86_64-linux/hf2jns_mtp models/qwen3-next.jns \
    hf:Qwen/Qwen3-Next-80B-A3B-Instruct models/qwen3-next-mtp.jns
```

For the dense Qwen3.5-9B that is 487 MB instead of 14 GB of shards, in under a minute. It also takes the shards themselves, when you have them on disk (`hf2jns_mtp <model.jns> <shard>... <out.jns>`). Qwen3.6-35B-A3B needs none of this: its GGUF keeps the block.

Qwen3.8-Flash-Next's block comes in a GGUF file of its own, which unsloth publishes beside its quantizations of the model ([`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), folder `MTP`). `hf2jns_mtp` copies the block as it is, in its own types, and leaves out the file's token embeddings and output head - the engine takes the model's own:

```sh
bin/x86_64-linux/hf2jns_mtp models/qwen3.8-flash-next.jns \
    mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf models/qwen3.8-flash-next-mtp.jns
```

From the Q4_K_M file (2.8 GB) that is 1.9 GB, the same bytes on every run. In the chat, on one reply of 64 tokens to a question on bread, 72.5% of its drafts were kept greedy and 80.5% sampled as the chat samples, and the reply was the one written without them.

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

The same for E2B, the 12B (`gemma-4-12b-it-Q4_K_M.gguf`) and the 26B (`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`); the fingerprints are in [MODELS.md](../MODELS.md#the-fingerprints). E2B's per-layer table, half of its file, is read from the file a token at a time and never loaded: E2B ran here in 1.5 GB of memory, E4B in 3.1 GB.

**And chat:**

```sh
bin/x86_64-linux/janas-chat models/qwen3-next.jns --stats
bin/x86_64-linux/janas-chat models/gemma-4-e4b.jns --stats
```

The chat finds the multi-token prediction file by itself: of the `.jns` files beside the model, the one whose header fits it (an MTP block of the same architecture and width, or Gemma 4's assistant for a model that wide and with its vocabulary), and says which it took. `--mtp <file>` names one; `--no-mtp` goes without. `janas-server` does the same, and programs can ask for it with `janas_llm_find_mtp()`.

The first start reads the model's resident weights (a couple of GB) and then fills the expert cache in the background, while you read and type. The second reply of a session is the one that shows the machine's real speed.

[INSTALL.md](../INSTALL.md) has the details, including the other distributions and what to do when the GPU is not found.
