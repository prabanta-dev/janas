# Models, and what may be done with them

Janas reads models; it does not carry them. This file says what the licence of a converted model is, and what this project will and will not redistribute.

## The engine's licence does not reach the weights

`tools/gguf2jns` turns a GGUF file into the `.jns` file the engine reads. The tool is under the GPL; **the file it writes is not**. A converted model is a derivative of the model, not of Janas, and it keeps the licence its author gave it. Running GPL software over someone's data does not make that data GPL, and nothing in this project claims otherwise.

The other way round matters too: a model's licence may forbid things the GPL would allow. The model decides what may be done with the model.

## What conversion changes, and what it does not

Converting a GGUF into `.jns` does not retrain, re-quantize or prune anything. It rewrites the same numbers in another order:

- one slot per (layer, expert), so the experts a token needs can be read from disk without reading anything else;
- the experts laid out by co-activation, so the reads fall near each other;
- the model's metadata carried over byte for byte;
- a resident tensor in MXFP4 (unsloth's Qwen3-Coder-Next files have their shared experts so) written as Q8_0, which holds each of its weights exactly: the same values in another encoding, at twice the bytes; routed experts in MXFP4 (the "MXFP4_MOE" files) stay as they are, the engine reads MXFP4;
- optionally (`tools/jns_planes`) the experts' down matrix cut into three planes of two bits, which is a reordering of the same bits: with all three planes the weights are the ones the file came with, to the last bit.

One thing changes when a model is opened, in memory and not in the file: an output head in BF16 or F16 (the "MXFP4_MOE" files keep theirs in BF16) is read as Q8_0, rewritten once as llama.cpp's quantization makes Q8_0: about half the bytes, and the head is read whole at every token. That is a re-quantization, and the model's description says so. Its cost, measured on noctrex's Qwen3.6-35B-A3B-MXFP4_MOE against llama.cpp on the same file, which keeps the head in BF16: 199 of 200 most likely tokens agree, mean logit difference 0.105, closer than a Q4_K_M file to its own llama.cpp, where nothing is re-quantized (196, 0.146).

Anyone redistributing a converted file has to say this, because the licences that allow redistribution (Apache 2.0 among them) require that a modified work state that it was modified.

## Converting twice gives the same file

`tools/gguf2jns` and `tools/jns_planes` are deterministic. The layout of the file follows from the sizes in the source, nothing is timed, drawn at random or read from the environment, and the weights are copied, never recomputed (an MXFP4 tensor's re-encoded as Q8_0, exactly). The same GGUF, converted on another machine, gives the same bytes. The converter was a Python script until 25 September 2026; the C one that took its place writes the very same bytes, and the fingerprints below hold for both.

That is what this project publishes about its models: the recipe, and the fingerprint of what the recipe produces — not forty gigabytes. Convert your own copy, compare, and you know you are running the file the numbers in this repository were measured on.

Not a claim, a measurement: four of the Qwen models below were converted again from their GGUF and compared, byte for byte, with the files this project runs, and so were the eight Gemma 4 files. All of them matched.

### The recipe

```sh
huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF \
    --include "Qwen3-30B-A3B-Q4_K_M.gguf" --local-dir models/gguf
bin/x86_64-linux/gguf2jns models/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
    models/qwen3-30b-a3b-q4km-flat.jns
bin/x86_64-linux/jns_planes models/qwen3-30b-a3b-q4km-flat.jns \
    models/qwen3-30b-a3b-q4km.jns
sha256sum models/qwen3-30b-a3b-q4km.jns
```

On the development laptop (a mobile Core Ultra, the models on NVMe) the two steps cost 47 s and 104 s for the 18.6 GB model, 153 s and 194 s for the 48.4 GB one, with the Python converter: less than the download either way. The C one took 51 s for the 18.6 GB model on a busy machine - the disk sets the pace, not the language.

### The fingerprints

Every GGUF below was checked against the SHA-256 that Hugging Face publishes for it, so the chain from the model's own repository to the converted file is unbroken. The `-flat.jns` line is what `gguf2jns` writes (JNS version 2), the line after it what `jns_planes` writes from that (version 3), which is the form these models are kept in here.

**Qwen3-Next-80B-A3B-Instruct**, from [`Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF):

```
d103b2733ec1012a52d01edda66b7e5c24ae50508c9f99f5297ea459ef3c061a  Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
fa9a50dd910de4e78064ecff17aef43286694277eea3a89a57e9921ca2bfd2da  qwen3-next-80b-a3b-q4km-flat.jns   (gguf2jns)
fa3255c73106391e9cb5a97efd30f77a928fb4518d40060f795a8cff6656daed  qwen3-next-80b-a3b-q4km.jns   (jns_planes)
```

**Qwen3.6-35B-A3B**, from [`bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF):

```
b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772  Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
31772037300f332d712f4ef8dd5e7251c7cd76cf6b9dab33b089394b7c079fb2  qwen3.6-35b-a3b-q4km-flat.jns   (gguf2jns)
8feaeb66d93593b9876b564ca3878f2a9252e46b8d4e2d06caa4e19d43dc6464  qwen3.6-35b-a3b-q4km.jns   (jns_planes)
```

**Qwen3-Coder-Next**, from [`Qwen/Qwen3-Coder-Next-GGUF`](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF):

```
6bcfc9f9c37901eeb92172e2ab871224dab36a453d263bcb2547f737409534da  Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf
817def0691ee9d08bf3dc4444be7aed29c9e52091e8fa9d97901ce7e7f6f01d3  Qwen3-Coder-Next-Q4_K_M-00002-of-00004.gguf
23aa634d47dca9b4ca3ea249384e6f01951b24c83cdc076f37f6f43d6c99883f  Qwen3-Coder-Next-Q4_K_M-00003-of-00004.gguf
249c768cc5f130dc731567d6edcbdacc48e14dec9e02c5dbe2b2185d2c5bdb2b  Qwen3-Coder-Next-Q4_K_M-00004-of-00004.gguf
5912d964247002473eff1811058b4ffadf7b9bb47ae336c486cdf2964de68382  qwen3-coder-next-q4km-flat.jns   (gguf2jns)
0276b0697a05075fec8ab076a757205537804ff41dd7e9dadace7c6993763563  qwen3-coder-next-q4km.jns   (jns_planes)
```

**Qwen3-Coder-30B-A3B-Instruct**, for code completion, from [`unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF):

```
fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad  Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
bee7d25509e911a195cbf05a157e69bb05a62d2241cca2c4fdcfef269dcb2bdd  qwen3-coder-30b-a3b-q4km.jns   (gguf2jns)
```

**Qwen3-0.6B**, from [`unsloth/Qwen3-0.6B-GGUF`](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) (the Qwen team's own repository no longer has a Q4_K_M), dense, and the model that guesses Qwen3-4B's tokens (`--draft`). Converted in one step; two conversions give the same bytes.

```
ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a  Qwen3-0.6B-Q4_K_M.gguf
9174da760919b16b9f3a4ae8738c2d53605412f40bb2c0fe68d60174bf8a4fb5  qwen3-0.6b-q4km.jns   (gguf2jns)
```

**Qwen3-4B**, from [`Qwen/Qwen3-4B-GGUF`](https://huggingface.co/Qwen/Qwen3-4B-GGUF). This one is dense: it has no experts to route, so its feed-forward becomes the single slot of each layer, and half its layers keep a `down` matrix in Q4_K, which has no planes to cut.

```
7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5  Qwen3-4B-Q4_K_M.gguf
e8ed44bdd3c612ce3ec1204a4b9db671115efab4a2ada384c70cf6492d430291  qwen3-4b-q4km-flat.jns   (gguf2jns)
1cbd8ebfdf14aee05777f5278e655382f040cdedc21d282e93806fc8cf279fc5  qwen3-4b-q4km.jns   (jns_planes)
```

**Qwen3-4B with an importance matrix** (`qwen3-4b-imatrix`), from [`bartowski/Qwen_Qwen3-4B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF): the same weight types as Qwen's Q4_K_M, quantized with an importance matrix.

```
fbe1d5edd4ce802ae3ae7c7e4ab7d09789d697fdac1fc7929f8df4ca3c41bae3  Qwen_Qwen3-4B-Q4_K_M.gguf
aa4f0bc6033a206ba5ac51dcdd799299d369019d5614dc08a6baa0bda6123098  qwen3-4b-bq4km-flat.jns   (gguf2jns)
134ae0afbbd3e45b3cdbfd36fd205fb98c91df62a129541048e29370a42b835e  qwen3-4b-bq4km.jns   (jns_planes)
```

**Qwen3.5-0.8B, Qwen3.5-2B and Qwen3.5-9B**, dense, from [`unsloth/Qwen3.5-0.8B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF), [`unsloth/Qwen3.5-2B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) and [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), GGUF conversions of the Qwen team's checkpoints. Converted in one step, as they are kept here: `jns_planes` could cut planes from their Q6_K `down` matrices, but a model this size has no need to read fewer bits.

```
bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517  Qwen3.5-0.8B-Q4_K_M.gguf
75a3b62c4a10cad87a7ce64bf89485cfcc564fa949f2c5af8c7051615dc26124  qwen3.5-0.8b-q4km.jns   (gguf2jns)
aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223  Qwen3.5-2B-Q4_K_M.gguf
286118ea3ad1903890a593582e22bed82859bf7c38b0ff9ed043050e5bd572a6  qwen3.5-2b-q4km.jns   (gguf2jns)
03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8  Qwen3.5-9B-Q4_K_M.gguf
e40ec6c58ef2390c7a4ce8c9b7b6b384ed4e55c9980f646d1c98e83e2c018115  qwen3.5-9b-q4km.jns   (gguf2jns)
```

**Qwen3.8-27B**, dense, from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at revision `4ca72078`, unsloth's UD-Q4_K_XL mix (Q4_K, Q5_K, Q6_K and Q8_0 with IQ4_XS, IQ3_S, IQ4_NL and Q3_K); its multi-token prediction block is in the GGUF, and so in the converted file. Converted in one step:

```
3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e  Qwen3.8-27B-UD-Q4_K_XL.gguf
138d1db3af7e48d3bd36a428f7a81dfd8813434521aced84459c546e4830ab5b  qwen3.8-27b-ud-q4kxl.jns   (gguf2jns)
```

**Qwen3.8-Flash-Next**, a mixture of 512 experts, from [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) at revision `ed59f920`, ISTA's GSQ-RCO quantization in Q2_0 (folder `Q2_0`), a quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at `de4b8e4d`. Two parts, the first given to `gguf2jns`, which reads the second beside it; converted in one step. Its multi-token prediction block is a GGUF file of its own, from [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) at `766911a6` (folder `MTP`), copied by `hf2jns_mtp` with the converted model ([how](docs/models.md)); two runs of it give the same bytes. Under the Qwen Community License 1.0, not Apache 2.0 ([below](#what-this-project-redistributes)).

```
69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
b5b4dbe4e3ae01bffc41acefa0d1e62afdfe4b5b9bc24c2a3fab591a41f9087f  qwen3.8-flash-next-gsq-q20.jns   (gguf2jns)
8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231  mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
d9c9dd94ef2e52391613db98c2e34c574341c48ab8752454696534d0f69a29ca  qwen3.8-flash-next-mtp-q4km.jns   (hf2jns_mtp)
```

**Qwen3-30B-A3B**, from [`Qwen/Qwen3-30B-A3B-GGUF`](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF):

```
0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5  Qwen3-30B-A3B-Q4_K_M.gguf
6906de51f2923b3be3a83eb040eaaeb090cba3389e9217648abe08bd678376e9  qwen3-30b-a3b-q4km-flat.jns   (gguf2jns)
3fa0185b7eecd43e222d420f68aae28a86b3b90392fd7e4052b5cd337318e895  qwen3-30b-a3b-q4km.jns   (jns_planes)
```

**Qwen3-30B-A3B with an importance matrix** (`qwen3-30b-a3b-imatrix`), from [`bartowski/Qwen_Qwen3-30B-A3B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF), bartowski's Q4_K_M.

```
a015794bfb1d69cb03dbb86b185fb2b9b339f757df5f8f9dd9ebdab8f6ed5d32  Qwen_Qwen3-30B-A3B-Q4_K_M.gguf
c2089d293f10d2e0fdabbb7ea6f17d6834f3027480de987fb6dd718e301eaa04  qwen3-30b-a3b-bq4km-flat.jns   (gguf2jns)
bee2aa5a9b5a33f19d9c153d2a0a7e8ac493a670b2d0084c195ce977cf3c83ea  qwen3-30b-a3b-bq4km.jns   (jns_planes)
```

**Qwen3-30B-A3B in three bits** (`qwen3-30b-a3b-q3`), from [`unsloth/Qwen3-30B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF), unsloth's dynamic UD-Q3_K_XL. Converted in one step: its down matrices are Q3_K and Q4_K, so there are no planes to cut.

```
f1b8111d3e49a0238fcaa77ec9ee24eb9805266b5e85bf8e54c12dedb02cf4ff  Qwen3-30B-A3B-UD-Q3_K_XL.gguf
7a77e62a3d6b1df321f455472e0b970a09eb3534ec76982a6e169087d4897656  qwen3-30b-a3b-udq3kxl.jns   (gguf2jns)
```

**Qwen3-Embedding-0.6B**, from [`Qwen/Qwen3-Embedding-0.6B-GGUF`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF), for `janas-server --embedding-model`. It is published in Q8_0 and converted as it is, in one step: its matrices are not Q6_K, so there are no planes to cut. Two conversions give the same bytes.

```
06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439  Qwen3-Embedding-0.6B-Q8_0.gguf
1e8eb8cbeba7a1a657354d259c5730bb79f6087160a9ca8da901bd08bdbe943f  qwen3-embedding-0.6b-q8.jns   (gguf2jns)
```

**Whisper small and large-v3-turbo**, speech to text, from [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) (Apache-2.0) and [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT), for `janas-server --speech-model`. There is no GGUF of them: `hf2jns_whisper` converts OpenAI's safetensors, their matrices kept in half precision as released, so only the converted files are published, in [`prabanta-dev/whisper-small-JNS`](https://huggingface.co/prabanta-dev/whisper-small-JNS) and [`prabanta-dev/whisper-large-v3-turbo-JNS`](https://huggingface.co/prabanta-dev/whisper-large-v3-turbo-JNS) (`janas-get whisper-small`). Converted on 29-30 October 2026 and again on 18 November 2026: the same bytes.

```
78401176c4abda8124188d429ac777ed79bd7e63b262d4f1f0075066b42b316a  whisper-small.jns            (hf2jns_whisper)
b2396b2c413af93a471b3f14ff6f0076630cc27f2e8a181cc982ac0d23d91b34  whisper-large-v3-turbo.jns   (hf2jns_whisper)
```

**Gemma 4: E2B, E4B, 12B and 26B-A4B, and their assistants**, from [`unsloth/gemma-4-E2B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF), [`unsloth/gemma-4-E4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF), [`unsloth/gemma-4-12b-it-GGUF`](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) and [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF). Each converted in one step, as it is kept here, and each converted twice on 27 September 2026: the same bytes. The assistant, Gemma's drafting model (the `mtp-` file, given to the chat with `--mtp`), converts with `gguf2jns` like the model. The 26B's feed-forwards are widened with zeros to a multiple of 256, which is why its file is larger than its GGUF.

**Thinking Machines Inkling-2-Mini**, from [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (revision `8078d149286eec7193795413b9e0643da6d11fa1`; the SHA-256 of the GGUF as published there), converted on 24 November 2026 by `gguf2jns` into an 18.3 GB file. Its experts stay NVFP4 as unsloth wrote them, each with its scale for gate and up and one for down; its small projections in BF16 are read as F32 rows. Its licence is Apache 2.0, with Thinking Machines' [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) (both on [the model's page](https://huggingface.co/thinkingmachines/Inkling-2-Mini)): the converted file is published in [`prabanta-dev/Inkling-2-Mini-JNS`](https://huggingface.co/prabanta-dev/Inkling-2-Mini-JNS) with the licence and that policy named beside it (`janas-get inkling-2-mini`).

```
78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae  Inkling-2-Mini-UD-Q4_K_M.gguf
82456e0707c6e02b0a331a014de2fb53e0a6b57dae611116b81207ff9ac11a07  inkling-2-mini-ud-q4km.jns   (gguf2jns)
```

**NVIDIA Nemotron-3.5-Bolt-74B-A9B (24GB)**, from [`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) (SHA-256 of the GGUF as published there, 91d63363...99af75), converted on 28 October 2026 by `gguf2jns` into a 22.0 GB file, and again on 14 November 2026: the same bytes. Its layers without experts (Mamba-2 and attention) get an empty slot, its experts no gate, and the table of experts is as wide as the layer with the most (496), the others' extra slots absent. Its licence is NVIDIA's OpenMDW-1.1 (on the model's page), which lets anyone pass the model on with a copy of the licence and NVIDIA's notices of origin: the converted file is published so, in [`prabanta-dev/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-JNS`](https://huggingface.co/prabanta-dev/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-JNS), with the licence and NVIDIA's model card beside it (`janas-get nemotron-3.5-bolt-74b`).

```
91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75  NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
5c67c321b7e4dd2397d990bb12cd4ae5cd08db2edd75dbbc71b7aeea139b54c1  nemotron-3.5-bolt-74b-24gb.jns   (gguf2jns)
```

**DeepSeek-V4-Flash** (284B parameters, 13B active), from DeepSeek's own checkpoint [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (MIT), converted on 19 November 2026 by `hf2jns_ds4`, which reads the checkpoint from Hugging Face by ranges and checks each shard against the SHA-256 the Hub lists. There is no GGUF of the original weights: the conversion is the only way. It changes the weights in these ways, all of them stated:
- **The routed experts are the same weights to the bit.** They are FP4 (E2M1) with a power-of-two scale per 32, which is MXFP4: the same codes and scale bytes, the nibbles reordered.
- **The dense matrices are re-quantized.** These are FP8 E4M3 with a scale per 128 x 128 block, and they become Q8_0 (blocks of 32), from each weight's exact value, as llama.cpp's converter does: a re-quantization.
- **So are the token embeddings and the output head**, BF16 written as Q8_0.
- **The other tensors are kept or widened:** the router becomes F32, the norms F32, and the hash layers' token-to-expert tables I32 (from I64); the other BF16 and F32 tensors are kept.

The multi-token prediction block is a file of its own (`hf2jns_ds4 --mtp`), its `e_proj` and `h_proj` side by side as one Q8_0 matrix. Both files are published in [`prabanta-dev/DeepSeek-V4-Flash-JNS`](https://huggingface.co/prabanta-dev/DeepSeek-V4-Flash-JNS) with DeepSeek's licence (`janas-get deepseek-v4-flash`). They have been converted once so far.

```
37b0d0b9a74820d2f69886b14e663b363605ab0edb2a2249072ee7230223d87a  deepseek-v4-flash.jns       (hf2jns_ds4)
8d46e88068b98e520b6ccc0f463d8b4cf53597dd16f72d8a5f665cfd92ec765c  deepseek-v4-flash-mtp.jns   (hf2jns_ds4 --mtp)
```

```
740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8  gemma-4-E2B-it-Q4_K_M.gguf
1f18e08878df41f377d563b38048d41841c7973ddea845cd9672fcf23cbd864b  gemma-4-e2b-it-q4km.jns   (gguf2jns)
9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919  mtp-gemma-4-E2B-it.gguf
79df1f3c817542b3867a0940b1a5b7330d478bddb4102ec90ff817334877cd62  gemma-4-e2b-it-mtp.jns   (gguf2jns)
85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87  gemma-4-E4B-it-Q4_K_M.gguf
7ec828e9bdc1ada2eda907e93b87072f3180f2e8cba5ccedcd360d1a53858675  gemma-4-e4b-it-q4km.jns   (gguf2jns)
b6a723115efa510d3b3215db1e26790dae84cd08c2134a764f3d194f1f0c3376  mtp-gemma-4-E4B-it.gguf
7c6fdb74ba1eb11f9d47b15097a09cce0142aea380a41f1986ba7b8ec4785b53  gemma-4-e4b-it-mtp.jns   (gguf2jns)
0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42  gemma-4-12b-it-Q4_K_M.gguf
cae09e04598891dea6e9cd93fb496fbde08d822ac35de4ff45dbfab1ea25c4dd  gemma-4-12b-it-q4km.jns   (gguf2jns)
145db9094bc0f85f1701e255a2ed216dcc9800fc8bc8631ad00905b456bd451b  mtp-gemma-4-12b-it.gguf
35e640f20dd74e862df9c1d949ad4edca9f58beef0cb29e99cccc6c99297826f  gemma-4-12b-it-mtp.jns   (gguf2jns)
f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f  gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
35c3845bbc21de52bf630d3c7455b4bb5379c34487b5925e020ec2e573931666  gemma-4-26b-a4b-it-udq4km.jns   (gguf2jns)
6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b  mtp-gemma-4-26B-A4B-it.gguf
6ba24ee91fe6fe904b29919acbb654a66d9caa0aba67db1df9f921a76e5bcedf  gemma-4-26b-it-mtp.jns   (gguf2jns)
```

The multi-token prediction block is not converted from a GGUF — GGUF files leave it out — but from the original checkpoint's last shard, quantized to Q4_K by `tools/hf2jns_mtp`, which also copies the metadata of the converted model handed to it:

```sh
bin/x86_64-linux/hf2jns_mtp models/qwen3-next-80b-a3b-q4km.jns \
    models/hf/model-00041-of-00041.safetensors \
    models/qwen3-next-80b-a3b-mtp-q4k.jns
```

Two runs of it give the same bytes, and it takes 75 s. **The shard**, from [`Qwen/Qwen3-Next-80B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), and what comes out:

```
1f3ac4d828f7e08dd14eb4dc6f282139ae1fa0d894e43f247da2539d2ef43826  model-00041-of-00041.safetensors
5c9bc4cb6f739193fac5e4daa30c922f73255acf41443ecfc0123ef084584559  qwen3-next-80b-a3b-mtp-q4k.jns   (hf2jns_mtp)
```

Until 25 September 2026 this line read `0dfd70d4…`: the fingerprint of a file whose levels `hf2jns_mtp` left at zero (issue #6). The engine now reads such a file as it was meant, and `jns_check` says what it is; the tool writes the levels, which is the file above.

**The dense Qwen3.5 models** keep their block in the checkpoint too, spread over several shards (three of the four of Qwen3.5-9B, 14 GB). `hf2jns_mtp` reads just the block's tensors from Hugging Face, with range requests - 487 MB for the 9B, under a minute - and deletes them once converted. Name the checkpoint's revision, so that the fingerprint below holds:

```sh
bin/x86_64-linux/hf2jns_mtp models/qwen3.5-9b-q4km.jns \
    hf:Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    models/qwen3.5-9b-mtp-q4k.jns
```

From [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) at `c2022362`, and [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) at `15852e8c` (`hf:Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`), each with the converted model above:

```
67f31f57dbc78ca8c02ef5e3aa137c5df3836a3decda20b6ea057d004089c361  qwen3.5-9b-mtp-q4k.jns   (hf2jns_mtp)
19c2d7eba1800acb10a06d7b38709ec92776e852cb2f25032ddd8977d17f2b05  qwen3.5-2b-mtp-q4k.jns   (hf2jns_mtp)
```

The same tensors from shards on disk give the same bytes. A private or gated repository wants `HF_TOKEN` in the environment; it is sent to huggingface.co only.

### If a fingerprint does not match

- **Check the GGUF first.** Another quantization of the same model is another file, and so is the same quantization from somebody else's repository. A `.jns` fingerprint says nothing on its own: it belongs to the GGUF above it.
- **The tools move.** A change to the layout changes every fingerprint here, and the ChangeLog says when that happened. These are the tools as this repository publishes them.
- **The file checks itself.** A `.jns` carries a checksum for every expert and for its metadata, so a damaged copy is caught without any of this:

  ```sh
  bin/x86_64-linux/jns_check model.jns --verify
  ```

## What this project redistributes

**The converter, always. The weights, only when their licence plainly allows it.** In order:

1. **Models under Apache 2.0, MIT or a similar licence.** Every model Janas supports but two is under **Apache 2.0** or **MIT** (DeepSeek-V4-Flash) and none is gated: Qwen3-Next (including Qwen3-Coder-Next), Qwen3.5, Qwen3.6 and Qwen3.8-27B, Qwen3 MoE, Gemma 4 and Thinking Machines' Inkling-2-Mini (whose use Thinking Machines' Model Acceptable Use Policy governs besides: whoever uses it, or offers it to others in a product, answers for keeping to it) — checked on their pages, not in their files. NVIDIA's **Nemotron 3.5 Bolt** is under NVIDIA's **OpenMDW-1.1**, which allows anything with the model as long as a copy of the licence and NVIDIA's notices go with it: its converted file carries both. The other is **Qwen3.8-Flash-Next**, under the **Qwen Community License 1.0** (on [its page](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)): use, copies, changes and redistribution are allowed with its notice, but a product above 100 million monthly users or US$ 20 million a month must show the model's name, and a business offering models as a service or an AI work assistant needs a separate licence from Qwen for commercial use. ISTA-DASLab's GGUF of it says Apache 2.0 in its card; the model's page is the licence. Its converted files are published with that notice: [`prabanta-dev/Qwen3.8-Flash-Next-JNS`](https://huggingface.co/prabanta-dev/Qwen3.8-Flash-Next-JNS) carries the Qwen Community License as the model's repository gives it, and `janas-get qwen3.8-flash-next --from-gguf` converts ISTA's GGUF on your machine instead. Converted files of all of them may be published, and when they are, they carry:
   - a link to the original model and to its licence;
   - the original `LICENSE` and, where there is one, the `NOTICE`;
   - a statement of what was changed (the list above) and which version of the tools did it;
   - the SHA-256 of every file;
   - no name, logo or wording that suggests the model's authors endorse this. Apache 2.0 grants no trademark rights: "converted from Qwen3-Next-80B-A3B, by Alibaba Cloud" is a fact; naming a repository after the model is not.

2. **Models whose file says nothing about a licence.** Nothing is published until the original page has been read. This is not a formality: of the four models converted here, two had lost the `general.license` field on the way, through somebody's re-quantization, although the models themselves are Apache 2.0. **The metadata is not the licence**; the model's page is.

   A GGUF that went through a third party has two hands on it: the model's authors and whoever quantized it. Apache 2.0 lets the second redistribute and lets this project redistribute in turn, but the attribution is owed to both. Where the chain is long — a checkpoint quantized to four bits, brought back to half precision and quantized again — the file is better left alone, not for the licence but because nobody can honestly say what is in it.

3. **Models under a community or research licence** (Llama, Gemma up to Gemma 3, and the like - Gemma 4 is Apache 2.0, on Google's page): **nothing is published, ever**. Their terms travel with the weights, and some of them forbid redistribution outright or attach conditions this project will not carry for its users. Convert them yourself, from the copy you downloaded under the terms you accepted.

Today there is nothing to fetch, and that is on purpose: the recipe above rebuilds every one of these files exactly, so publishing them would only move bytes that anyone can make. Should that change, converted models would live in the project's model repositories on Hugging Face, not in this one: a file of forty gigabytes has no business in a source tree, and GitHub will not take it — 100 MB a file, 2 GB with Git LFS, 2 GB an asset in a release.

## If you convert a model yourself

You are the one redistributing it, if you pass it on. Check the licence of the model **and** of the GGUF you converted (they may be different hands), keep the notices, say what you changed. The metadata Janas carries over can tell you where to start looking:

```sh
bin/x86_64-linux/jns_check model.jns
```
