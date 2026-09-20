# Models, and what may be done with them

Janas reads models; it does not carry them. This file says what the licence of a
converted model is, and what this project will and will not redistribute.

## The engine's licence does not reach the weights

`tools/gguf2jns` turns a GGUF file into the `.jns` file the engine reads. The
tool is under the GPL; **the file it writes is not**. A converted model is a
derivative of the model, not of Janas, and it keeps the licence its author gave
it. Running GPL software over someone's data does not make that data GPL, and
nothing in this project claims otherwise.

The other way round matters too: a model's licence may forbid things the GPL
would allow. The model decides what may be done with the model.

## What conversion changes, and what it does not

Converting a GGUF into `.jns` does not retrain, re-quantize or prune anything.
It rewrites the same numbers in another order:

- one slot per (layer, expert), so the experts a token needs can be read from
  disk without reading anything else;
- the experts laid out by co-activation, so the reads fall near each other;
- the model's metadata carried over byte for byte;
- optionally (`tools/jns_planes`) the experts' down matrix cut into three
  planes of two bits, which is a reordering of the same bits: with all three
  planes the weights are the ones the file came with, to the last bit.

Anyone redistributing a converted file has to say this, because the licences
that allow redistribution (Apache 2.0 among them) require that a modified work
state that it was modified.

## Converting twice gives the same file

`tools/gguf2jns` and `tools/jns_planes` are deterministic. The layout of the
file follows from the sizes in the source, nothing is timed, drawn at random or
read from the environment, and the weights are copied, never recomputed. The
same GGUF, converted on another machine, gives the same bytes. The
converter was a Python script until 25 September 2026; the C one that took
its place writes the very same bytes, and the fingerprints below hold for
both.

That is what this project publishes about its models: the recipe, and the
fingerprint of what the recipe produces — not forty gigabytes. Convert your own
copy, compare, and you know you are running the file the numbers in this
repository were measured on.

Not a claim, a measurement: each of the four models below was converted again
from its GGUF and compared, byte for byte, with the file this project runs.
All four matched.

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

On the development laptop (a mobile Core Ultra, the models on NVMe) the two
steps cost 47 s and 104 s for the 18.6 GB model, 153 s and 194 s for the
48.4 GB one, with the Python converter: less than the download either way.
The C one took 51 s for the 18.6 GB model on a busy machine - the disk sets
the pace, not the language.

### The fingerprints

Every GGUF below was checked against the SHA-256 that Hugging Face publishes
for it, so the chain from the model's own repository to the converted file is
unbroken. The `-flat.jns` line is what `gguf2jns` writes (JNS version 2),
the line after it what `jns_planes` writes from that (version 3), which is the
form these models are kept in here.

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

**Qwen3-4B**, from [`Qwen/Qwen3-4B-GGUF`](https://huggingface.co/Qwen/Qwen3-4B-GGUF).
This one is dense: it has no experts to route, so its feed-forward becomes the
single slot of each layer, and half its layers keep a `down` matrix in Q4_K,
which has no planes to cut.

```
7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5  Qwen3-4B-Q4_K_M.gguf
e8ed44bdd3c612ce3ec1204a4b9db671115efab4a2ada384c70cf6492d430291  qwen3-4b-q4km-flat.jns   (gguf2jns)
1cbd8ebfdf14aee05777f5278e655382f040cdedc21d282e93806fc8cf279fc5  qwen3-4b-q4km.jns   (jns_planes)
```

**Qwen3.5-0.8B, Qwen3.5-2B and Qwen3.5-9B**, dense, from
[`unsloth/Qwen3.5-0.8B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF),
[`unsloth/Qwen3.5-2B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
and [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF),
GGUF conversions of the Qwen team's checkpoints. Converted in one
step, as they are kept here: `jns_planes` could cut planes from their Q6_K
`down` matrices, but a model this size has no need to read fewer bits.

```
bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517  Qwen3.5-0.8B-Q4_K_M.gguf
75a3b62c4a10cad87a7ce64bf89485cfcc564fa949f2c5af8c7051615dc26124  qwen3.5-0.8b-q4km.jns   (gguf2jns)
aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223  Qwen3.5-2B-Q4_K_M.gguf
286118ea3ad1903890a593582e22bed82859bf7c38b0ff9ed043050e5bd572a6  qwen3.5-2b-q4km.jns   (gguf2jns)
03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8  Qwen3.5-9B-Q4_K_M.gguf
e40ec6c58ef2390c7a4ce8c9b7b6b384ed4e55c9980f646d1c98e83e2c018115  qwen3.5-9b-q4km.jns   (gguf2jns)
```

**Qwen3-30B-A3B**, from [`Qwen/Qwen3-30B-A3B-GGUF`](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF):

```
0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5  Qwen3-30B-A3B-Q4_K_M.gguf
6906de51f2923b3be3a83eb040eaaeb090cba3389e9217648abe08bd678376e9  qwen3-30b-a3b-q4km-flat.jns   (gguf2jns)
3fa0185b7eecd43e222d420f68aae28a86b3b90392fd7e4052b5cd337318e895  qwen3-30b-a3b-q4km.jns   (jns_planes)
```

**Qwen3-Embedding-0.6B**, from [`Qwen/Qwen3-Embedding-0.6B-GGUF`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF),
for `janas-server --embedding-model`. It is published in Q8_0 and converted
as it is, in one step: its matrices are not Q6_K, so there are no planes to
cut. Two conversions give the same bytes.

```
06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439  Qwen3-Embedding-0.6B-Q8_0.gguf
1e8eb8cbeba7a1a657354d259c5730bb79f6087160a9ca8da901bd08bdbe943f  qwen3-embedding-0.6b-q8.jns   (gguf2jns)
```

The multi-token prediction block is not converted from a GGUF — GGUF files
leave it out — but from the original checkpoint's last shard, quantized to
Q4_K by `tools/hf2jns_mtp`, which also copies the metadata of the converted
model handed to it:

```sh
bin/x86_64-linux/hf2jns_mtp models/qwen3-next-80b-a3b-q4km.jns \
    models/hf/model-00041-of-00041.safetensors \
    models/qwen3-next-80b-a3b-mtp-q4k.jns
```

Two runs of it give the same bytes, and it takes 75 s. **The shard**, from
[`Qwen/Qwen3-Next-80B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct),
and what comes out:

```
1f3ac4d828f7e08dd14eb4dc6f282139ae1fa0d894e43f247da2539d2ef43826  model-00041-of-00041.safetensors
5c9bc4cb6f739193fac5e4daa30c922f73255acf41443ecfc0123ef084584559  qwen3-next-80b-a3b-mtp-q4k.jns   (hf2jns_mtp)
```

Until 25 September 2026 this line read `0dfd70d4…`: the fingerprint of a
file whose levels `hf2jns_mtp` left at zero (issue #6). The engine now reads
such a file as it was meant, and `jns_check` says what it is; the tool
writes the levels, which is the file above.

**The dense Qwen3.5 models** keep their block in the checkpoint too, spread
over several shards (three of the four of Qwen3.5-9B, 14 GB). `hf2jns_mtp`
reads just the block's tensors from Hugging Face, with range requests - 487
MB for the 9B, under a minute - and deletes them once converted. Name the
checkpoint's revision, so that the fingerprint below holds:

```sh
bin/x86_64-linux/hf2jns_mtp models/qwen3.5-9b-q4km.jns \
    hf:Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    models/qwen3.5-9b-mtp-q4k.jns
```

From [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) at
`c2022362`, and [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) at
`15852e8c` (`hf:Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`),
each with the converted model above:

```
67f31f57dbc78ca8c02ef5e3aa137c5df3836a3decda20b6ea057d004089c361  qwen3.5-9b-mtp-q4k.jns   (hf2jns_mtp)
19c2d7eba1800acb10a06d7b38709ec92776e852cb2f25032ddd8977d17f2b05  qwen3.5-2b-mtp-q4k.jns   (hf2jns_mtp)
```

The same tensors from shards on disk give the same bytes. A private or gated
repository wants `HF_TOKEN` in the environment; it is sent to huggingface.co
only.

### If a fingerprint does not match

- **Check the GGUF first.** Another quantization of the same model is another
  file, and so is the same quantization from somebody else's repository. A
  `.jns` fingerprint says nothing on its own: it belongs to the GGUF above it.
- **The tools move.** A change to the layout changes every fingerprint here,
  and the ChangeLog says when that happened. These are the tools as this
  repository publishes them.
- **The file checks itself.** A `.jns` carries a checksum for every expert and
  for its metadata, so a damaged copy is caught without any of this:

  ```sh
  bin/x86_64-linux/jns_check model.jns --verify
  ```

## What this project redistributes

**The converter, always. The weights, only when their licence plainly allows
it.** In order:

1. **Models under Apache 2.0, MIT or a similar licence.** Every model Janas
   supports today is under **Apache 2.0** and none is gated: Qwen3-Next
   (including Qwen3-Coder-Next), Qwen3.5 and Qwen3.6, and Qwen3 MoE — checked
   on their pages, not in their files. Converted files may be
   published, and when they are, they carry:
   - a link to the original model and to its licence;
   - the original `LICENSE` and, where there is one, the `NOTICE`;
   - a statement of what was changed (the list above) and which version of the
     tools did it;
   - the SHA-256 of every file;
   - no name, logo or wording that suggests the model's authors endorse this.
     Apache 2.0 grants no trademark rights: "converted from Qwen3-Next-80B-A3B,
     by Alibaba Cloud" is a fact; naming a repository after the model is not.

2. **Models whose file says nothing about a licence.** Nothing is published
   until the original page has been read. This is not a formality: of the four
   models converted here, two had lost the `general.license` field on the way,
   through somebody's re-quantization, although the models themselves are
   Apache 2.0. **The metadata is not the licence**; the model's page is.

   A GGUF that went through a third party has two hands on it: the model's
   authors and whoever quantized it. Apache 2.0 lets the second redistribute
   and lets this project redistribute in turn, but the attribution is owed to
   both. Where the chain is long — a checkpoint quantized to four bits,
   brought back to half precision and quantized again — the file is better
   left alone, not for the licence but because nobody can honestly say what is
   in it.

3. **Models under a community or research licence** (Llama, Gemma up to
   Gemma 3, and the like - Gemma 4 is Apache 2.0, on Google's page):
   **nothing is published, ever**. Their terms travel with the weights, and
   some of them forbid redistribution outright or attach conditions this project
   will not carry for its users. Convert them yourself, from the copy you
   downloaded under the terms you accepted.

Today there is nothing to fetch, and that is on purpose: the recipe above
rebuilds every one of these files exactly, so publishing them would only move
bytes that anyone can make. Should that change, converted models would live in
the project's model repositories on Hugging Face, not in this one: a file of
forty gigabytes has no business in a source tree, and GitHub will not take it —
100 MB a file, 2 GB with Git LFS, 2 GB an asset in a release.

## If you convert a model yourself

You are the one redistributing it, if you pass it on. Check the licence of the
model **and** of the GGUF you converted (they may be different hands), keep the
notices, say what you changed. The metadata Janas carries over can tell you
where to start looking:

```sh
bin/x86_64-linux/jns_check model.jns
```
