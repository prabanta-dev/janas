# ChangeLog

> Curated, user-facing summary of completed work, newest first.

## [2026-09-26] - Gemma 4, first part (not yet usable)

- **Gemma 4 (`gemma4`) runs its forward pass**, on Gemma-4-12B-it: layers
  with a sliding window of 1,024 positions and every sixth a full one, each
  kind with its own head size (256 and 512), KV heads and RoPE (the full
  layers with Google's per-pair frequency factors); V taken from K where a
  layer has no V projection, and normalized; norms after attention and
  after the feed-forward, GELU, a scale per layer, soft-capped logits. The
  attention takes a sliding window and a score scale of its own, heads up
  to 512, and the KV cache is laid out per layer.
- **Gemma's tokenizer**: BPE on UTF-8 characters with SentencePiece's
  spaces and byte tokens, in the same tokenizer as Qwen's. On a mixed text
  (Italian, English, code, CJK, emoji) it gives llama.cpp's 128 tokens.
- **Not yet usable**: the chat does not know Gemma's format, and with the
  eight-bit KV cache the logits drift from llama.cpp's (the most likely
  token agrees at 175 of 200 positions, mean difference 1.36; llama.cpp
  agrees with itself at 169 when its cache goes from f16 to eight bits).
  Every Qwen model gives the same logits as before, to the bit.

## [2026-09-26] - Long prompts: the expert cache, the tuner, the ETA (issue #9)

- **The expert cache no longer thrashes on large prefill blocks.** A pass
  asks for experts layer after layer, and when one pass asks for more
  than the cache holds, least-recently-used eviction throws out exactly the
  expert wanted again soonest: on a 262K prompt of Qwen3-Next-80B, blocks
  of 256 read 2.17x the bytes of blocks of 64 (issue #9). In that case the
  victim is now the expert wanted again latest - the current layer's
  others, then the layer before, and back from there. Measured on the 80B
  (4,096 tokens, 12 GB of cache, four bits, blocks of 256): 238 GB read at
  38.2 token/s before, 172 GB at 45.8 now; blocks of 64 are unchanged.
- **The tuner tries alternatives ever more rarely while they lose**: at a
  fixed 128 passes it made one slow block every 32,768 tokens of a long
  prompt. The gap now doubles up to 1,024 passes and returns to 128 as soon
  as a try beats the choice.
- **A prompt's time left follows the cost of a token as the context
  grows** (`input_eta_seconds`, new at the end of `janas_llm_chat_stats`
  and in `llm.bi`): a line in the position, fitted on the blocks read and
  integrated to the end of the prompt, instead of the latest speed, which
  promised 2.5 times too little at 32K of a 262K prompt.

## [2026-09-26] - IQ3_S weights

- **IQ3_S weights**, the last type unsloth's "UD" files of the Qwen
  family were missing: 256 weights in eight groups, each weight a level of
  a 512-entry grid, with its sign and a scale per group. A group's eight
  grid entries come in one gather, the signs go to the activations, and
  the integer sum is exact, one fma per 256 weights; AVX2 and AVX-VNNI
  give the reference's bits. The grid is ggml's, copied as data
  (`src/llm/iq3s_grid.h`, MIT, as ggml is) since it defines the format.
  The weights match ggml's to the bit on random blocks; Qwen3.5-0.8B
  requantized to IQ3_S agrees with llama.cpp at a mean logit difference of
  0.123, the disagreements near-ties.

## [2026-09-26] - Q3_K weights, and unsloth's UD files of Qwen3.5-9B

- **Q3_K weights**: 256 three-bit values in sixteen groups with 6-bit
  scales. The values are summed unsigned (0-7) and the -4 of each is taken
  from the activations' block sums, so the integer sum of a block is exact
  and one fma follows, as for Q4_K; AVX2 and AVX-VNNI give the reference's
  bits. The weights match ggml's to the bit on random blocks; Qwen3.5-0.8B
  in Q3_K_M agrees with llama.cpp at a mean logit difference of 0.115.
- **A DeltaNet's alpha and beta in F16 or BF16** are taken too: unsloth's
  "UD" files of Qwen3.5-9B keep them so. **Qwen3.5-9B UD-Q4_K_XL** - Q4_K,
  Q5_K, Q6_K, Q8_0, IQ4_XS and F16 in one file - runs and agrees with
  llama.cpp at 0.110, as its Q4_K_M does.

## [2026-09-26] - Q4_0, Q4_1 and Q5_1 weights

- **Models quantized in Q4_0, Q4_1 and Q5_1 convert and run** - the types of
  Google's QAT Gemma files and of many older GGUFs. Q4_0 is IQ4_NL's layout
  with evenly spaced levels, so it runs IQ4_NL's kernel with another table;
  Q4_1 and Q5_1 (weights d x q + m, Q5_1 with a fifth bit) share a kernel
  that takes the minimum's share from the activations' block sums, exactly.
  On one to eight vectors, AVX2 and AVX-VNNI give the scalar reference's
  bits. Checked against ggml on 4,096 random blocks of each (the weights
  identical to the bit), and on Qwen3.5-0.8B against llama.cpp: the Q4_0
  file (which holds Q4_1 tensors too) at a mean logit difference of 0.109,
  a Q5_1 one requantized from the Q8_0 at 0.114, the disagreements
  near-ties.
- A DeltaNet layer's alpha and beta are taken in any type Janas unpacks
  (they came in Q5_1 in the requantized file), not only f32 and Q8_0.

## [2026-09-26] - IQ4_NL and IQ4_XS weights

- **Models quantized in IQ4_NL and IQ4_XS convert and run**: the two 4-bit
  types whose sixteen levels are spaced unevenly, closer near zero, that
  many of the most downloaded GGUF files use. Their kernels are Q8_0's with
  a table lookup in front (one `pshufb` turns 32 indices into 32 weights),
  on one to eight vectors, AVX2 and AVX-VNNI giving the same bits as the
  scalar reference; IQ4_XS sums its eight scaled groups exactly and takes
  one fma per 256 weights. Checked against ggml on 4,096 random blocks of
  each (the unpacked weights identical to the bit) and on Qwen3.5-0.8B in
  both types against llama.cpp: mean logit difference 0.11, as the Q4_K_M
  file of the same model, and where the most likely token differs the two
  were within 0.46 of each other in llama.cpp's own logits.
- Not yet: files that mix in IQ3_S or Q3_K (unsloth's "UD"
  quantizations), and the speed of these kernels, which has not been worked
  on - reading a prompt, Qwen3.5-0.8B went at 230 tok/s in IQ4_NL and 155 in
  IQ4_XS against 289 in Q4_K_M.

## [2026-09-26] - `janas-bench` measures the drafts as a chat meets them

- **The "with MTP" figures were too high, and this is why.** `janas-bench`
  measured the prediction block by continuing its own prompt, one paragraph
  repeated, greedily - and a model continuing prose copies the paragraphs
  it has seen, which the block guesses almost every time. On Qwen3.5-2B it
  kept 95% of its drafts there and reported 107 tok/s; a real chat with the
  same model kept 34-68% and wrote at 43-62. Every "with drafts" figure in
  the README until today (Qwen3-Next-80B-A3B 30, Qwen3.6-35B-A3B 29,
  Qwen3.5-9B 26, Qwen3.5-2B 107) came from that measurement.
- **Now the block is measured on a chat reply**: an ordinary question in the
  model's chat format, the first 64 tokens of the answer, once greedy and
  once sampled as the chat samples (temperature 0.7, top-k 20, top-p 0.8,
  fixed seed), each with the share of drafts kept. The prompt for the
  reading speed is prose that does not repeat itself; three rounds by
  default, as the README always said (it was two).
- What the block is worth depends on the text: on Qwen3.5-2B, 68% of the
  drafts kept on an English explanation, 34% on Italian prose, where the
  engine drafts less and the gain all but goes.
- **The chat runs on the cores `janas-bench` measures**: every usable core,
  the performance cores with their threads and the efficiency cores, never
  the low-power island. The tuner used to try one thread per performance
  core and their threads alone as well, and after a stretch of long context
  a chat could settle on 12 threads where the bench had reported 20: the
  same machine, two speeds. Measured on six models, from Qwen3.5-2B to
  Qwen3-Next-80B-A3B, all the cores were the fastest or level in every kind
  of pass, and the others lost 5-20% writing; so the threads are fixed now,
  and the tuner only decides whether the GPU helps. `JANAS_TUNE_THREADS=1`
  brings the other counts back, to experiment with.
- **`janas-bench` reports what the engine will run**: its result line is the
  configuration the engine chose for each kind of pass, as a chat or the
  server will use it, not the fastest figure of each column.

## [2026-09-26] - Qwen3.5 dense models

- **Measured** on the development laptop, idle, `janas-bench`: Qwen3.5-9B
  reads a prompt at 46 tok/s and writes at 13, **22 with the block** in a
  chat; Qwen3.5-2B at 215, 49 and **75**. (First published as 26 and 107,
  with the block measured on repeated text: see the entry above.)
  llama.cpp on the same 9B file and machine (`llama-bench`, build of 18
  September) did best at 32 and 9.0.
  README and MODELS.md list the three dense Qwen3.5 models, their
  fingerprints and how to get their block (`hf:` with the checkpoint's
  revision).
- **The dense Qwen3.5 models run** (`qwen35`: Qwen3.5, 3.6 and 3.8 without
  experts, half of the GGUF downloads on Hugging Face): the layers of
  Qwen3.6-35B-A3B with a dense feed-forward, one row of the architecture
  table. The DeltaNet's alpha and beta come in Q8_0 in these files and are
  made f32 once, exactly, at load. Checked against llama.cpp on the same
  files, 400 tokens of Italian prose and 400 of Pascal: Qwen3.5-0.8B agrees
  on the most likely token at 387 and 386 positions, Qwen3.5-2B at 376 and
  388, mean logit difference 0.10-0.13, and where they differ the two
  candidates were within 0.53 of each other in llama.cpp's own logits.
  Chat and tool calls (XML) work through `janas-chat` and `janas-server`.
- **`hf2jns_mtp` converts the MTP block of a dense Qwen3.5** too, its
  feed-forward as the one slot, and takes the block from several shards
  (`hf2jns_mtp <main.jns> <shard>... <out.jns>`): Qwen3.5-9B spreads it
  over three. The number and shape of the experts come from the model and
  the checkpoint instead of Qwen3-Next's written in; its block converts to
  the same file as before, to the byte. On Qwen3.5-9B the drafts are
  accepted at 53% (prose) and 68% (code), the text the same as without.
- **`hf2jns_mtp <main.jns> hf:<owner>/<repo> <out.jns>` reads the MTP
  block straight from Hugging Face**, with range requests: only the
  `mtp.*` tensors, 487 MB of Qwen3.5-9B's 14 GB of shards, in under a
  minute, with no other program (Janas's own HTTPS client, the one of its
  MCP transport, now following redirects). The file it writes is the same,
  to the byte, as from the shards on disk. A server that ignores a range
  stops it, never a whole file downloaded. Qwen3.5-2B's block too: drafts
  accepted at 78% and 64%. Neighbouring tensors are read in one request (up
  to 256 MB): Qwen3-Next's block, 1,500 tensors, came in 27 minutes one
  connection each and comes in 5 now, to the same bytes.
- The memory counted for the context asks the architecture table which
  layers have keys and values, instead of two names written in.

## [2026-09-26] - Architectures as data

- **What sets one architecture apart from another is a row of a table**
  (`src/llm/arch.c`): recurrent layers and their layout, a gated query,
  a shared expert, which norm comes before the feed-forward, how value
  heads share key heads, MTP layers in the file. The flags `next` and
  `q35` are gone; a new family is a row plus the code of what it has that
  no other had. The logits are the same bits as before on Qwen3-4B,
  Qwen3-30B-A3B, Qwen3.6-35B-A3B and Qwen3-Next-80B-A3B, token by token
  and in blocks, and so are the MTP drafts of the 35B and the 80B.
- **`model.c` divided by what it does**: it was 2,597 lines; the passes
  stay in it, the attention, DeltaNet and feed-forward layers, opening and
  closing, and the threads with the experts' profile each have a file of
  their own (`model_attn.c`, `model_rec.c`, `model_ffn.c`, `model_load.c`,
  `model_tune.c`). Code moved, not changed: the same bits again on the four
  models.
- **`build.sh` makes its archives anew** every time: `ar` only adds and
  replaces, so the object of a source since removed or renamed stayed in
  `libjanas.a` and was linked twice (going back to an older commit failed
  to link).

## [2026-09-26] - Qwen3-Coder's tool calls, run

- **Qwen3-Coder-Next calls tools through `janas-server`**, in the XML way
  its template describes: asked for the weather and a list of files, it
  made both calls at once with valid arguments, and used their answers in
  the reply; the second turn read 66 new tokens and reused 496. The README
  no longer calls this way untried.

## [2026-09-25] - Builds with clang

- **`janas-try.sh` no longer skips a 16 GB machine**: it compared the
  memory the kernel reports with 16 GiB exactly, and a machine sold with
  16 GB shows less (the kernel and an integrated GPU keep some), so every
  one of them lost the medium and full levels, and an 8 GB one the quick
  level. 85% of the memory a level asks for now counts as enough.
- **`CC=clang ./build.sh` works** (clang 19, every variant under `-Werror`):
  two float conversions of `RAND_MAX` in the benchmarks, a string
  concatenation clang took for a missing comma in `test_grammar`, and the
  `die` of the two converters marked as not returning. The test suite
  passes, and the 4B's answer at temperature 0 is the same text built with
  gcc or clang, with AVX-VNNI or AVX2 kernels.

## [2026-09-25] - `tools/janas-try.sh`: try Janas and report, in one command

- **One command to try Janas on a machine**: it builds and tests, downloads
  a model (quick 4B, medium 35B-A3B, full 80B-A3B with MTP), checks every
  file against its SHA-256 and every conversion against MODELS.md,
  checks the answer at temperature 0 against the text every machine should
  write (the same with any thread count, AVX2 or AVX-VNNI, GPU or not),
  measures with `janas-bench`, tries `janas-server`, and writes a report
  with no names or paths. It asks before each long step and before sending;
  the issue goes through `gh`, or a link fills the new **test report**
  form. Nothing to install, no root.
- **It measures only a machine left to itself**, or says it did not: before
  the speed test it samples how idle the CPU is for five seconds and, when
  something else is running, explains that the numbers would not be the
  machine's and offers to wait, measure all the same or skip. The report
  says how idle the CPU was and the load average - general figures only,
  never which programs ran.
- MODELS.md: the fingerprint of the MTP block is the one of the file
  `hf2jns_mtp` writes since issue #6.

## [2026-09-25] - The converter is C: nothing but the build to run Janas

- **`gguf2jns` is now a C program**, built with everything else, in place
  of the Python script: trying Janas needs a compiler and nothing more. It
  writes the very same bytes - the SHA-256 fingerprints in MODELS.md,
  taken from the Python one, come out identical for Qwen3-4B, Qwen3-30B-A3B
  and Qwen3-Embedding-0.6B.
- Its GGUF reader (`src/llm/gguf.c`) takes nothing on trust: lengths and
  counts checked against the bytes there are, every tensor's data against
  the size of the file; `test_gguf` fuzzes it.

## [2026-09-25] - `max_tokens` is the answer's; Open WebUI tried

- **In janas-server's chat completions, `max_tokens` limits the answer
  alone**, the reasoning not counted; `max_completion_tokens` (and the
  Responses API's `max_output_tokens`) still counts both, as OpenAI defines
  it. Open WebUI asks for a chat's title with `max_tokens` 1000, and with a
  model that reasons at length the reasoning used it all: no title. The
  library has `max_answer` for it, beside `max_reply`.
- **Open WebUI 0.11.4 tried against janas-server** through its API:
  streamed chat, titles, tags, follow-up suggestions, a document indexed
  with Janas's embeddings and answered from; the conversations kept
  computed across its side requests. The README says how to run them
  together (`--think off`).

## [2026-09-25] - The MTP block of Qwen3-Next loads again (issue #6)

- **`hf2jns_mtp` wrote a file the engine refused** since model files gained
  levels (version 3, bit planes): its layer's levels were left at zero, and
  loading it with `--mtp` failed with `layer 0: bad levels`. It writes them
  now, and a file it wrote before is read as what it meant (a slot with no
  planes), so nothing has to be converted again.
- The error of an MTP file now names that file, not the model's.
- `jns_check` says when a file has such levels, and that converting it
  again is optional.

## [2026-09-25] - MCP: a cancel between rounds, the servers' instructions

- **janas-server**: a `/responses/{id}/cancel` arriving between two MCP
  rounds of a background response now stops it - the next round is no
  longer started, and the tool call running is stopped too. Checked under
  ThreadSanitizer.
- **janas-mcp works under Claude Code** (2.1.282, which speaks revision
  2026-07-28): its discover and tools/list results now carry the caching
  hints that revision requires (`ttlMs`, `cacheScope`), without which Claude
  Code refused the tool list. generate and embed tried from it.
- `JANAS_METRICS` naming a file that cannot be opened is now said on stderr.
- **Tried against servers of others**: the reference filesystem and git
  servers over stdio (npx, uvx), DeepWiki over HTTPS from the client and
  from janas-server's Responses API through the `openai` SDK.
- **janas-chat `--mcp-instructions`**: what the MCP servers say of their
  tools goes after the system message, marked as theirs (off by default).
  `janas_mcp_instructions` in libjanas_mcp gives it to any program.

## [2026-09-25] - A long prompt says how it goes, and what it costs

Asked for in issue #5, after a 262,144-token run that read its prompt for
fifteen hours without a word.

- **The prompt is read a block at a time**: `janas_llm_chat_next` returns an
  empty piece after each block of 256 tokens, so a program has control
  while a long prompt is read - to show progress, or to stop (janas-server
  now stops a cancelled request in the middle of its prompt). Same speed as
  before, measured.
- **`janas_llm_chat_stats` says more**, appended to the struct: the stage,
  the tokens read so far, time to the first token, time building the
  prompt, processor time of the reading, the threads chosen, the memory of
  the context and the process's peak, and the prompt by its parts (system
  message and tools, conversation, last message).
- **`max_input`** in the chat's parameters, and `--max-input` in
  janas-server: a longer prompt is refused with its numbers
  (`JANAS_LLM_ELIMIT`; a 400 `input_limit_exceeded` with `input_tokens`,
  `max_input_tokens`, `excess_tokens`), never cut. `--warn-input` takes it
  and says so.
- **`JANAS_PROGRESS` and `JANAS_METRICS`** for any program on the library:
  progress lines on stderr, and JSON lines of events (open, prompt,
  progress, first token, reply, refused), counters only, never text.
  janas-chat shows the progress in its status line, has `/context` and
  `--progress`; janas-server has `--metrics` and `--progress`.
- Documented: the prompt is read in `janas_llm_chat_next`, not when it is
  sent or loaded; `context_used` holds the token that ended the reply,
  hence prompt + output + 1.

## [2026-09-25] - MCP servers in janas-server's Responses API

- **Tools of type `mcp` in `POST /v1/responses`**, as OpenAI's API has
  them: the server reaches the MCP server a request names (`server_url`,
  `headers`, `authorization`) through libjanas_mcp, starts the output with
  an `mcp_list_tools` item, and gives the tools to the model as
  `server_label__tool`. A call needing approval (`require_approval`,
  `always` by default) ends the response with an `mcp_approval_request`
  and runs when an `mcp_approval_response` answers it; one that needs none
  runs at once as an `mcp_call` item and the model goes on in the same
  response, a round at a time. `allowed_tools`, the approval filters,
  `tool_choice` of type `mcp`, streamed `response.mcp_*` events, and the
  items read back from the history. `--no-mcp` refuses them.

## [2026-09-25] - https:// MCP servers

- **TLS for the MCP client**, on **Mbed TLS 4.2.0** compiled in
  (`src/third_party/mbedtls`, unmodified, with TF-PSA-Crypto): `https://`
  servers from libjanas_mcp and janas-chat, TLS 1.2 and 1.3, the server's
  certificate checked against the system's certificate authorities
  (`SSL_CERT_FILE` and `SSL_CERT_DIR` override them) and against the host
  name. Mbed TLS's symbols stay hidden inside libjanas_mcp; libjanas_llm
  has none of it.

## [2026-09-25] - MCP servers over HTTP

- **libjanas_mcp reaches servers over Streamable HTTP** (`janas_mcp_open_url`,
  or `url` and `headers` in the configuration, so janas-chat uses them too):
  every message a POST, the answer a JSON object or an event stream, the
  body's fields mirrored into `MCP-Protocol-Version`, `Mcp-Method`,
  `Mcp-Name` and the `Mcp-Param-*` headers a tool's schema asks for with
  `x-mcp-header` (base64 where a header cannot carry the value; tools whose
  marks break the rules are left out). Both eras: a modern server found by
  its answer to `server/discover`, a legacy one by its refusal, then
  `initialize` and its session, ended with DELETE. `https://` needs TLS,
  which is not built in yet.

## [2026-09-25] - Janas as an MCP server

- **`janas-mcp`**: the model offered to MCP clients (Claude Code, editors,
  agents) over stdio, as the tools `generate` (a question to the model, its
  answer back) and `embed` (with `--embedding-model`). Both protocol eras:
  requests of revision 2026-07-28 served statelessly, and the `initialize`
  handshake for revisions 2025-11-25 back to 2024-11-05. Pings are answered
  during a generation, and `notifications/cancelled` stops it.

## [2026-09-25] - Tools from MCP servers in the chat

- **janas-chat gives the model the tools of MCP servers**: those listed in
  `~/.config/janas/mcp.json` (or `--mcp-config FILE`), in the `mcpServers`
  format of the other clients, started as child processes and spoken to
  over stdio, in both eras of the protocol - the stateless revision
  2026-07-28, found by probing with `server/discover`, and the handshake of
  2025-11-25 for the servers that predate it. Every call is shown and waits
  for yes, no or always-this-tool; `/mcp auto` and `--mcp-auto` run them
  without asking, `/mcp` lists servers and tools, `--no-mcp` starts none.
- **`libjanas_mcp`**, the client as a library of its own
  (`include/janas/mcp.h`, and `mcp.bi` for FreeBASIC), with nothing of the
  engine in it.
- **`janas_llm_chat_send_results`** in `libjanas_llm`: what the tools
  answered to the calls of the last reply, sent in the conversation the
  library keeps, the model's reply following as after a message.
- `tests/mcp_fake_server` and `test_mcp`: the client against a server of
  each era, and the readers of JSON-RPC lines and configuration files fuzzed.

## [2026-09-24] - The memory left to other programs can be chosen

- **`--reserve GIB`** in janas-chat, janas-server and janas-bench, and
  `janas_llm_params.reserve_bytes` in the library: when the expert cache
  takes the free memory, how much of it stays free for the rest of the
  machine. The default is unchanged, a fifth of the memory and at least 2
  GiB. More keeps a busy desktop out of swap, at the price of a smaller
  cache on a model larger than the memory: on the 32 GB test machine,
  Qwen3-Next-80B-A3B's cache went from 20.6 to 18.8 GiB with `--reserve 8`.
  The conversations kept computed size their automatic ceiling on it too.

## [2026-09-24] - Prompts read in blocks of 256 tokens

- **A prompt is now read 256 tokens at a pass instead of 64**, so that a
  mixture-of-experts model reads each expert once for four times as many
  tokens, and gives it enough of them to run at full speed. Attention and
  the recurrent log still work 64 tokens at a time inside the block, and
  the logits are bit for bit those of reading token by token (checked on
  Qwen3-4B, Qwen3-30B-A3B and Qwen3-Next-80B-A3B). Reading a 1024-token
  prompt: Qwen3-Next-80B-A3B 68.4 → 75.1 tokens/s (+9.9%), Qwen3-30B-A3B
  92.4 → 97.5 (+5.5%), the dense Qwen3-4B unchanged (87.9 → 87.3). Replies
  are as fast as before. `JANAS_PREFILL_BLOCK` (64 to 4096) sets another
  size; larger blocks gained nothing more here, and 1024 was slower on
  Qwen3-4B and on Qwen3-Next-80B-A3B.

## [2026-09-24] - The chat's cache share moves while it waits; attention's memory counted

- **The share of the expert cache filled, in janas-chat's footer, was drawn
  only at the start and after each reply**: it showed whatever it was at
  that moment - 60% at one start, 0% at the next - and kept it. The footer
  is now drawn again while the editor waits for a key, when the share
  changes, and the share goes when the filling is over; on
  Qwen3-Next-80B-A3B it climbed from 0% to 98% in nine seconds and went,
  with the cursor never leaving the line being written. A filling
  overtaken by the conversation's own requests (they filled the cache
  first) now ends: `janas_llm_preload` then gives a total equal to what it
  loaded, instead of a share stuck below 100%.
- **The partial results of attention grow with the context** like the
  keys and values, and were left out when the memory was shared out:
  1.08 GB at 262,144 tokens of Qwen3-Next-80B-A3B, taken from the margin
  left to other programs. They are counted now (`janas_attn_part_bytes`).

## [2026-09-23] - Conversations kept computed, and a store that outlives the server

- **Conversations kept computed.** The model holds one sequence, so a server
  whose clients take turns - or one client that asks on the side for titles,
  tags and suggestions, as Open WebUI does four times a turn - had each
  conversation read again from the start when it came back. Now the last
  few are copied out of the model and back in: `janas_llm_chat_keep` in the
  library, `--keep N` (8) and `--keep-memory GIB` in `janas-server`. The
  copy is of the keys and values of every attention layer (the prediction
  block's included) and of the recurrent state of Qwen3-Next and Qwen3.5/3.6.
  On Qwen3-4B, two conversations taking turns reuse 133 of 160 and 119 of
  143 tokens at their second turn, against 3; on Qwen3.6-35B-A3B, 133 and
  119 against 0; the replies are the same text either way.
- **A second reply to the same prompt** on models with a recurrent state
  (`n` above 1, a reply asked again) read the whole prompt again, since that
  state cannot go back. The prompt is now kept before the reply: 143 of 143
  tokens found computed on Qwen3.6-35B-A3B. The system message is kept on
  its own too, so that a new conversation with the same one starts from it.
- **On disk, below the memory**: `janas_llm_chat_keep_disk`, `--keep-disk
  GIB` (8, never more than a quarter of the free space), in
  `~/.cache/janas`. A copy of at least 1024 tokens is written when it leaves
  the memory and when the server stops, never at every turn; a long system
  message with its tools is then read once, also across restarts. A new
  chat found 2012 of 2013 tokens computed on Qwen3-4B (a 153 MiB file) and
  2103 of 2103 on Qwen3.6-35B-A3B.
- **The store outlives the server.** Stored completions, responses and
  conversations are written one file each to `~/.local/share/janas/server`
  (`--store DIR`; `--no-store` keeps them in memory only), readable by
  their owner alone, one server at a time per directory. A response left
  running in the background by a killed server is found as failed.
- **Fixed: the memory a long context takes was counted four times** on
  Qwen3-Next and Qwen3.5/3.6, whose files give one count of key-value
  heads for all layers: every layer was counted, not only the attention
  ones. Opened at 262,144 tokens, Qwen3-Next-80B-A3B set 13.1 GB aside for
  3.3 GB of keys and values, left its expert cache 6.7 GiB, and fell back
  to four bits and eight experts; now 13.5 GiB and all ten experts. Reading
  15,904 tokens went from 22.1 to 30.4 tokens/s and the reply from 14.2 to
  16.9, with the three facts hidden in the text found either way.

## [2026-09-23] - It builds with GCC 13 again

- **Ubuntu 24.04's compiler stopped the build** (issue #4): GCC 13 drops a
  `#pragma GCC unroll` on a loop whose condition holds a `?:` and warns,
  and `-Werror` made the warning an error. The bound of the two loops is now
  a constant of its own; with GCC 14 the machine code is the same, byte for
  byte. Checked with GCC 13.3.0 from Debian's packages, also with the options
  Ubuntu's GCC turns on by default.

## [2026-09-23] - The first ring of OpenAI's API, whole

Every operation of the API that a local text model can serve now answers:
27 of the specification's 345, against 4.

- **Tools** in chat and in responses: the tools written into the system
  message as the model's template writes them, and every call held to a
  grammar of the functions and their parameters from the moment the model
  opens it, so its arguments are always valid JSON for the schema;
  `tool_choice` none, auto, required or a function, parallel calls, the
  calls of earlier turns and the tools' answers in the history. Tried on
  Qwen3-4B: two calls at once, their answers sent back, the reply using
  them, the second turn reusing 304 of its 351 tokens; the XML way of
  Qwen3.5 and 3.6 on Qwen3.6-35B-A3B the same. Qwen3-Coder's description of
  the tools is written from its template and not yet run.
- **JSON output** held to a grammar token by token: a JSON object, or JSON
  valid against a JSON Schema (types, properties in order, required ones,
  items and bounds, enum, const, anyOf, $ref with recursion).
- **Log-probabilities** with up to 20 alternatives; `n`, `best_of`,
  presence and frequency penalties, `logit_bias`; several prompts, token
  ids, `echo` and `suffix` in completions.
- **Stored chat completions**: kept on request, listed with metadata
  filters and pages, read, changed, deleted, their messages listed.
- **The Responses API** with its conversations: items in and out,
  `previous_response_id`, streamed events, responses in the background and
  their cancelling, input items, token counts, compaction into a summary;
  the eight operations of `/conversations`.
- **Embeddings** from a second model (`--embedding-model`, e.g.
  Qwen3-Embedding-0.6B): against llama.cpp on the same file, a cosine of
  0.977 to 0.996 between the vectors of the same text.
- **Moderations** asked of the chat model, with the scores taken from its
  log-probabilities: a general model with instructions, not a trained
  classifier.
- **In the library**, for every program: a sampler of its own with a filter
  of the tokens that may come next, grammars with the model's special tokens
  as symbols, JSON Schema as a grammar, a small JSON reader, conversations
  built message by message with their tool calls, token counts, fill in the
  middle, embeddings. The text of a reply is unchanged byte for byte where
  none of it is asked for.
- **While a request waits**, its stream says where it stands in the line,
  and the test page shows it.

## [2026-09-23] - janas-server: the model behind OpenAI's API

Asked for in issue #3: an HTTP API that clients written for OpenAI's can use,
so that a model running on this machine can sit behind a chat front end, an
editor or a script without anything written for Janas.

- **`janas-server`** answers `GET /v1/models`, `POST /v1/chat/completions`
  and `POST /v1/completions`, whole or streamed as server-sent events, with
  the reasoning apart in `reasoning_content` and the counts in `usage`. It
  listens on this machine only unless told otherwise, asks for a key when
  given one, and warns when it is opened to the network without one. Requests
  run one at a time, in the order they came; a client that goes away stops
  its reply.
- **The whole of OpenAI's API is in its route table**, generated from the
  specification (2.3.0, 345 operations) by `tools/openapi_routes.py`: what is
  not written yet answers 501 and says why, and what a request asks that is
  not written yet (tools, JSON output, log probabilities) is refused with a
  400 rather than ignored.
- **A conversation is not read twice.** OpenAI's API sends the whole
  conversation with every request; the server reads again only what is new.
  The replies it wrote are remembered with their reasoning, so the answer a
  client sends back without it still matches. Checked: three turns through
  the server and through `janas-chat` give the same words, and the second
  turn reads 18 tokens of 260 or 19 of 200.
- **`--test` serves Janas-Chat Web**, a chat page inside the program: the
  reasoning in grey, the context and the speed under each reply, and the
  marks a model writes read as `janas-chat` reads them, plus quotes, tables,
  struck text, six levels of heading and links that ask before they open.
  *Markdown sample* shows every mark it knows without asking the model.
- **Two libraries of others, in `janas-server` only**: GNU libmicrohttpd
  1.0.10 (signature checked) and yyjson 0.13.0, unchanged in
  `src/third_party/`. `build.sh` compiles them apart and never into
  `libjanas_llm`.
- **In the library**: `janas_llm_chat_load` takes a conversation whole,
  `janas_llm_chat_prompt` continues raw text, `janas_llm_default_system` is
  the system message both programs now share, and the reply statistics say
  why a reply ended and how much of its prompt was already computed. The
  FreeBASIC binding follows, checked with `fbc`.
- **In `janas-chat`**: the title is *Janas-Chat*; a mark at the very end of a
  reply (`**bold**`, `` `code` ``) was printed instead of read, and now is
  read; an underscore alone at the edge of a word is italic, so `**_this_**`
  works while `snake_case` and `__init__` stay as they are; and the system
  message asks the model not to introduce itself unless asked, which on
  Qwen3-4B turned the whole presentation it gave to "Ciao!" into "Ciao! 😊".
- **Checked**: the unit tests in release, asan and tsan, a test that sends
  twenty thousand mangled requests to the parser, the engine's logits
  unchanged byte for byte, and the server under asan and tsan. The only races
  tsan reports are two inside libmicrohttpd, formal and harmless in the mode
  it runs in.

## [2026-09-23] - Tests: two failures on every machine but one

Reported in issue #2 from a Ryzen AI 9 HX 370: `test_expert_cache` could not
write its file and `test_tuner` found its saved choices missing. Both had the
same cause, a habit of the development machine leaking into the tests: they
wrote under `$HOME/tmp`, a directory that exists there and almost nowhere
else.

- **`test_expert_cache`** writes its file in `$TMPDIR`, else in `/var/tmp` -
  a disk, where O_DIRECT works, which it does not on the tmpfs `/tmp` often is.
  If it cannot, it says so and says what to set.
- **`test_tuner`** makes a scratch directory of its own with `mkdtemp` and
  removes it when done.
- **The engine made only the last two levels of its cache directory.** With
  `XDG_CACHE_HOME` pointing somewhere that did not exist yet, the tuner and
  the expert profile were silently not saved. Every missing level is made
  now.

Checked the way it failed: with a `HOME` that has no `tmp` and no `TMPDIR`,
the old binaries print exactly the two failures of the report and the new ones
pass.

## [2026-09-22] - Janas-LLM: reading a prompt, 8 to 12 per cent faster

Measured before touching anything: while a block of 64 prompt tokens went
through the model, nineteen of twenty threads spent **9% of the time** on
Qwen3-4B and **13%** on Qwen3-30B-A3B waiting for the calling thread, which
did per-token work alone - the norms and the quantization of the
activations, the normalization and rotation of each head of q and k with the
eight-bit copy into the attention cache, the router's choice of experts, and
the sum of the experts' outputs.

- **All of that now runs on the pool**, token by token or head by head. Each
  token takes the same operations in the same order as before, including the
  order in which it adds up its experts, so nothing changes in the result:
  the logits of Qwen3-4B, Qwen3-30B-A3B and Qwen3-Next-80B-A3B are identical
  byte for byte to the previous version's, over 256 tokens, both in blocks
  and one token at a time.
- **Prefill, 1024 tokens, cache warm, 20 threads**, old and new alternated:
  Qwen3-4B 82.4 to 88.5 tok/s (+7.6%), Qwen3-30B-A3B 82.7 to 92.6 (+12.1%),
  Qwen3-Next-80B-A3B 58.3 to 64.1 (+9.9%). The time the calling thread works
  alone falls to 0.3% and 0.9%.
- **Decoding**: the heads of q and k are spread over the pool there too, which
  saves about a third of a millisecond a token: Qwen3-30B-A3B 28.56 to 28.92
  tok/s over three alternated rounds, Qwen3-4B unchanged within the noise.

## [2026-09-22] - Janas-LLM: three ways the chat laid text out wrong

All three found by using it, and all three in the same place: what the chat
knows about where a line may break.

- **A code block inside a list never closed.** A model asked for examples put a Python block inside a numbered list, so it indented the fence that ends it. The markdown machine closed a block only on a backtick in the very first column, so the fence came out as text and everything after it stayed inside the block - where nothing else is a mark, which is why the rest of the reply arrived with its stars and its bullets as written. Spaces no longer end the start of a line inside a block, which is the rule the rest of the machine already used.
- **And the indent of that fence stayed on the screen.** A fence that opens is swallowed whole, its line with it, but the spaces before it had already been printed and stayed behind on a line whose end had gone: the first line of code then came out with its own indent added to the fence's. The spaces a line begins with are now held until it is known whether a fence follows them.
- **A list item that wraps now lines up under its own text**, not under its mark, which is what a reader expects and what this looked wrong without. A line that begins with a bullet or with a number and a dot has its continuations pushed right by the columns its mark took; everything else is untouched, which is what separates this from the hanging indent on all text that was tried earlier in the month and taken out again. Below about eight columns of room it is dropped. Checked at 74, 45, 34 and 20 columns.
- **The box you type in broke words in half.** It wrapped by counting columns while the conversation has wrapped by word since September. It uses the same rule now, and so does the line repeated in the conversation after it is sent. `term_box_rows` had a copy of the counting and now asks the layout, so the two cannot disagree about where a word goes. One thing the change brought with it and that had to be caught: a cursor just after the space a break leaves at the end of a row would have been drawn one column outside the box - seen at sixteen columns, position fourteen, column seventeen of sixteen. It rests on the last column instead. Checked at every position of a line at 74, 40, 24, 16 and 12 columns.

## [2026-09-22] - Janas-LLM: the engine was slower with its own speculation on

Four things, found by pulling one thread: what a draft is worth depends on
what a second token in a pass costs, and nobody had ever measured that. It is
**6% on a dense model**, where the weights are read once for both tokens, and
**28% on a mixture**, where two tokens route to different experts. Against
that, drafts copied from the conversation survive 11% of the time on ordinary
prose and 64% on a page of tables, and at the temperature a chat actually uses
they survive 25%. The engine asked for four of them regardless.

- **The drafts copied from the context now have a planner**, the one the prediction block already used, with the chance of a draft in each position measured per position rather than guessed and halved every 192 drafts so it follows the text. `spec_k` goes back to being a ceiling on what may be asked for. Before this, on ordinary prose at the chat's own temperature, speculation made the engine **slower than not speculating at all**: 0.97x on Qwen3-4B and **0.78x on Qwen3-30B-A3B**, with the safety net parking the drafts for 137 passes out of 256. After: 24.3 and 26.3 token/s against 21.4 and 22.0, so 13 and 20 per cent recovered.
- **The sampler sorted thirty-four thousand tokens to use twenty.** Choosing a token collected every logit within thirty times the temperature of the largest - about 34000 of a vocabulary of 151936 - sorted all of them, then kept the twenty `top_k` asks for. It cost 2.4 to 2.9 ms a token on the calling thread while nineteen others waited, five to six per cent of a token of Qwen3-4B, and more with a larger vocabulary (Qwen3.6 has 248320). The twenty largest are now found in one pass with a heap of twenty. Paired against a binary differing only in this: sampling 0.51 s to 0.09 s, decoding 22.3 to 23.7 token/s over three rounds.
- **`--draft` puts a small model in place of the copying.** A model of the same family predicts where copying only repeats: Qwen3-0.6B drafting Qwen3-4B is accepted 78% of the time and gives **1.14x**, 23.0 to 26.3 token/s. It shares the big model's compute pool, because two pools of twenty threads on twenty cores take the cores from each other - a pass of the drafting model cost 22.1 ms beside the target's pool and 7.2 ms alone. On a mixture it does not pay, and their answer stays the prediction block.
- **None of this changes a word of the output.** A draft is never a token: the engine samples from the big model and keeps the draft only where the token it sampled is the very same one. `llm_gen` checks it on every run, 256 tokens with drafts and 256 without from one seed, identical.
- **What was measured and refused:** the planner of the prediction block is not conservative, whatever it looked like. Fixed numbers of drafts were tried against it on Qwen3-Next-80B - 26.2, 25.1, 23.6 and 21.0 token/s at one, two, three and four - and the planner gives **29.1**. It wins by drafting on 79 passes out of 145 rather than by drafting more: its acceptance is 91% where a fixed single draft gets 72%. At four drafts speculation becomes a loss even there.

## [2026-09-22] - Janas-LLM: the 35B's expert phase was never slow

This project's notes have carried, for two sessions, a figure described as the
largest thing left open: the routed-expert phase of Qwen3.6-35B running at 40
GB/s where the same phase of Qwen3-30B runs at 62.4, with the difference
unexplained after several candidates had been measured and dropped. There was
nothing to explain. The three models run that phase at **65.3, 65.9 and 63.4
GB/s** - within four per cent of each other - and the 40 was an accounting
mistake in three parts.

- **The phase must not be measured whole.** It holds three things with three different rules for counting bytes: the cache, which is a wait on the disk and is 11.9% of the phase on the 35B in decoding and 25.8% on Qwen3-Next-80B; the `gate` and `up` matrices, plain Q4_K read in full every time; and `down`, which is six-bit planes whose byte count depends on a level the engine picks from how much memory there is. The honest number comes from `gate`+`up` alone, where nothing has to be guessed, and `JANAS_EXPTRACE=1` separates them.
- **The shared expert was not being counted.** `qwen35moe` and `qwen3next` have one, used by every token in every layer on top of the `k` routed ones; `qwen3moe`, which is the 30B, does not. Comparing the two while counting only routed experts takes away from the 35B bytes it really reads.
- **And the 35B's shared expert is Q8_0**, where Qwen3-Next's is Q4_K: 1.114 MB against 0.590 for the same 2048x512 matrix, 1.89 times the bytes. It carries **89.1 MB a token**, 19% of all the `gate`+`up` bytes, for one expert out of nine. That comes from the GGUF it was converted from, not from any choice here.
- A fourth, smaller: the 35B declares 41 layers and runs 40. The last one is the multi-token-prediction block, which sits in the file and does not run when generating normally.

| Model | layers | k | routed matrix | shared | bytes/token | ms/token | GB/s |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 48 | 8 | 0.885 MB | — | 679.5 MB | 10.41 | **65.3** |
| Qwen3.6-35B-A3B | 40 | 8 | 0.590 MB | 2.228 MB | 466.6 MB | 7.08 | **65.9** |
| Qwen3-Next-80B-A3B | 48 | 10 | 0.590 MB | 1.180 MB | 622.8 MB | 9.83 | **63.4** |

- **Checked by trying to break it.** An arithmetic that comes out right can come out right by accident, so the count was asked to predict something it had not been fitted to: with `JANAS_EXPERTS=6` on the 35B it says `gate`+`up` should take 5.65 ms. Measured, 5.60. Nine parts in a thousand.
- **What is left, and it is real:** requantizing the 35B's shared expert to Q4_K would save 42 MB a token, 9% of the `gate`+`up` bytes, about **1.7% of a token**. It is not an obvious call - the shared expert is the one every token crosses in every layer, so it is the most sensitive of all to losing bits, which is very likely why whoever quantized the source left it at eight. The quality has to be measured first, against the oracles that already exist. Converter work, offline, with no line of the engine to change.

## [2026-09-22] - Janas-LLM: the pool was not the bottleneck, and now there is a yardstick

This project's own notes opened the session with a plan: 21.2% of decoding
sits in the thread pool's machinery, the engine makes some 340 dispatches a
token and each one is a barrier, so fusing the three dispatches of the expert
phase into one would take away two thirds of them. The premise does not
survive being measured. What came out of measuring it is worth more than the
plan was.

- **`JANAS_POOLTRACE=1`** is new and stays: one line per call site - the name comes free from a macro around `janas_pool_run` - with its runs, the threads they used, their wall time, and how much of that thread time was spent at the end of a run waiting for the slowest. What the table deliberately does not hold is the time *between* runs, the caller working alone; that is the difference between the wall of the pass and the sum of the column, and it is printed beside it. Off, it costs one variable read per dispatch.
- **The barriers cost 2.4% of the thread time, not 21%.** On Qwen3-30B-A3B decoding 256 tokens, the matrix products - 89% of the time inside the pool - waste 1.4%. Work is claimed on demand there, and it shows.
- **What the profile was really showing is the caller.** The caller alone is 11.0% of decoding on Qwen3-30B and **2.9%** on Qwen3-4B, whose experts all fit in RAM and which reads nothing from disk. The threads spinning under the pool's name in the profile are waiting for the caller, and the caller is waiting for the NVMe. The lever is the cache, not the barrier.
- **So the expert phase will not be fused.** It would attack 96 barriers out of 392, worth well under one per cent, against a concurrent rewrite of `run_experts` with per-expert dependencies. The two sites that do waste their threads - the activation and the attention merge, which divide the work by index rather than on demand - come to 73 ms out of 8272 together. The number does not justify the change, and saying so is the result.
- **The chunks of a matrix product now shrink towards the end of a run.** They were a fixed `rows/(threads*4)`, about four per thread, so whoever claimed the last one kept everyone else waiting for as long as it took - and this machine's cores come in three speeds. A chunk is now `1/(2*threads)` of what is left to claim, between the same sixteen and two hundred and fifty-six rows: long at the start where the prefetcher earns its keep, short at the end where nobody should wait.

| Model, prefill at 20 threads | before | after |
|---|---|---|
| Qwen3-4B (dense) | 79.7 | **86.8** (+8.9%) |
| Qwen3-30B-A3B | 84.4-84.9 | **88.0-88.7** (+4.4%) |
| Qwen3-Next-80B-A3B | 56.1 | **59.0** (+5.2%) |

  Decoding gains about one per cent at six and twelve threads and nothing at twenty; prefill gains most because there each row carries 256 vectors, so a chunk costs 256 times more and so does its tail. On `bench_long` the spread across three runs fell from 0.14 to 0.02 token/s - the runs became repeatable, which is worth as much as the per cent.
- **The barrier was fixed anyway, on its own evidence.** It ended on one counter that every thread decremented, which at twenty threads is twenty transfers of a single cache line from core to core, in series, and the last thing to happen before the caller can go on. Each thread now has a line of its own; the mutex and the broadcast are skipped when nobody is asleep, which between two dispatches a few microseconds apart is always. An empty run at twenty threads went from **2.75 us to 0.53**. On the engine it is worth nothing measurable - those transfers were hiding under the caller's own share of the work - and that is precisely what makes it certain that dispatching is not where the time goes.
- **A caution, since this session produced one wrong number before the right one.** A first version of the probe reported "41.9% idle at the barrier" by adding the I/O pool's runs, which have eight threads and wait on the disk, to the compute pool's, and multiplying the sum by twenty. The true figure is 2.4%. Thread time has to be counted run by run, with the threads that run actually used.

## [2026-09-22] - Janas-LLM: the expert phase taken apart

The entry below put the headroom at two per cent and said the engine was
finished as an implementation. That was argued from the engine measured
against itself, which settles nothing: it says there is no waste between a
kernel and the phase around it, not that the kernel is good. Taken apart
properly, the figure is about seven per cent, and it is in one place.

- **`JANAS_EXPTRACE=1`** is new and stays: it splits the routed-expert phase into its parts - the two grouped products, the activation and its quantization, the final accumulate, the setup, and the cache - and prints them whenever the per-phase times are read, resetting as it goes. That last detail is the point of it. Prefill and decoding are different machines: decoding is **95.5% matrix product** and its cache costs 2.7%, prefill is 67.4% and 29.5%, because prefill really does read from disk. Averaged together they say something true of neither, which is exactly the mistake this entry was written to correct.
- **There is nothing to take in the glue.** In decoding the whole of the setup, the activation, the accumulate and the cache come to 4.5% of the phase, and the measured wait on the disk is 1% of a token. The 9% quoted earlier was prefill mixed in.
- **The ceiling is the machine, not our probe** - checked a second time, since it had already been wrong once today. Four independent load chains or sixteen make no difference, and software prefetch makes it worse: the hardware prefetcher is already saturating. 81 GB/s, and eight extra cores past twelve buy one per cent, so what saturates is the bus rather than the cores. Against a theoretical 102 to 119 GB/s for this machine's memory, that is 68 to 79%, and no kernel crosses it. At one vector the work *is* the read: a better kernel can only read fewer bytes.
- **Scattered reading costs nothing.** Large matrices drawn at random from eight gigabytes run at 77.6 GB/s against 78.4 for one contiguous matrix. The contiguity lever, taken literally, is not there.
- **What is left sits in the expert product itself:** 64.5 GB/s in the engine, against 70.2 for the same geometry in a probe and 78.4 for a contiguous read. Six of those points are the model's own shape - a single expert matrix of Qwen3-30B is 0.9 MB, and the size curve puts it there. **Eight points are between the engine and its own probe at the same geometry**, which is the first properly paired comparison of the day with a difference that survives it. The engine keeps its down matrix in six-bit planes where the probe uses four, and groups pairs through strides rather than separate matrices. The cause is not identified.
- **Headroom, third figure of the day.** This project published 1.32x this morning, then 1.02x, now about **1.07x**. The first two were extrapolated from a yardstick taken somewhere else; this one is built on the phase itself, in decoding, separated from prefill. It is not promised to be the last.

## [2026-09-22] - Janas-LLM: how much is left, and the answer is very little

The entry below this one closed by saying the engine reaches the metal on one
phase and asking what the rest could be worth. Measured, the answer is two
per cent, and the number published earlier today - a third more speed - does
not survive. What follows is how that was settled, including the two guesses
that were wrong.

- **There are two yardsticks, not one.** A bare Q4_K product over one contiguous matrix runs at **78.4 GB/s**, 97% of the machine - measured on 1.2 GB, where the level-3 cache is 1% of it and cannot flatter the figure. The same product over expert-sized blocks drawn at random from eight gigabytes runs at **68.4**. Reading scattered costs 12%, and that is the structural price of being a mixture engine, not a fault in the code.
- **How wrong a yardstick can be:** the same command on a 151 MB matrix reported 84.3 GB/s, above the machine's own read bandwidth, because a sixth of that matrix never leaves the cache between passes. 151 MB reads 84.3, 302 reads 80.8, 604 reads 79.2, 1208 reads 78.4. A benchmark that re-reads its data measures the cache unless it is far larger than one.
- **Matrix size is the whole story, and it is bigger than reported this morning.** Earlier today this was measured on twelve threads and over too narrow a range, and came out as 9%. On twenty threads, from one matrix of 0.9 MB up to one of 56.6, the throughput goes **44.3, 55.7, 62.5, 65.4, 66.1, 67.1, 68.4**: a spread of 54%. A single expert of Qwen3-30B is 0.9 MB, at the bottom of that curve. The engine does not send them one at a time - it groups the eight a token needs into one product, which is what lifts 44.3 to 64.6. The recovery is already taken.
- **Every phase is at 97 to 100% of what its own shape allows**: output 77.4 against 78.4, the attention output 65.1 against about 66, the experts 62.4 against 64.6, the attention projections 57.7 against 58. There is no implementation left to win on this machine.
- **Two ideas measured and dropped.** Laying the experts a token needs side by side in the cache, instead of wherever replacement put them, is worth **nothing** - a slot holds gate, up and down back to back, so neighbouring slots still leave the gate matrices 2.65 MB apart, and neighbouring turns out not to matter at that size anyway. And the bit-plane format, which the kernels said costs 22% of the arithmetic, costs **nothing** on the engine: that 22% is measured where the matrix sits in the level-2 cache and the arithmetic is the limit, while in the engine it hides under the wait for memory. Both were expected to pay and neither does.
- **What is left that is free:** fusing the three attention projections into one matrix in the converter. Three matrices of 4.7, 1.2 and 1.2 MB become one of 7.1, which the curve puts at 65.4 GB/s against 58. On Qwen3-30B that is 0.59 ms of 32.4: **two per cent**. The same bits, offline, with nothing in the engine to change.
- Everything beyond that costs bytes - fewer bits for each weight, which is quality - or different hardware.

## [2026-09-22] - Janas-LLM: the speed of the thing, measured again

All of it re-measured on an idle machine with the expert cache left to size
itself, which is what the engine does when nobody tells it otherwise. The
numbers published before were taken with the cache pinned to 20 GiB by hand -
a setting the product never chooses on its own.

| Model | cache | prefill | decode | with MTP |
|---|---|---|---|---|
| Qwen3-4B (dense) | 1.6 GiB | 81.1 | **25.7** | — |
| Qwen3-30B-A3B | 17.5 GiB | 84.7-86.6 | **31.0-31.4** | — |
| Qwen3.6-35B-A3B | 20.5 GiB | 72.2 | **23.9** | **29.1** |
| Qwen3-Next-80B-A3B | 19.8 GiB | 63.4 | **24.5** | **34.2** |

- Qwen3-30B gains 4 to 6%: the cache cap fixed this session gave it 17.5 GiB where it used to get 16.3. Qwen3.6-35B gains 7% in decoding. Qwen3-Next-80B is where it was.
- **Letting the cache size itself is as good as pinning it, or better.** Checked in alternation rather than back to back, so that drift could not pass for an effect: on Qwen3.6-35B the automatic size gives 23.7 token/s against 23.1 and 23.2 for the old cap, on Qwen3-Next-80B 24.6 against 24.5 twice, with the speculative figure better in both.
- **A first measurement on a fresh machine reads 7 to 15% low, and it is not the engine.** `janas-bench` fills the engine's tuning profile while it measures, so its runs are not independent of each other: from an empty profile Qwen3-Next-80B reports 58.5 / 22.6 / 29.9, and after a few runs the same binary on the same weights reports 63.4 / 24.5 / 34.2. Twice in an hour that rise was mistaken here for a regression that had been introduced, and twice a paired run said otherwise. Three runs, and take the numbers when they stop climbing.

## [2026-09-22] - Janas-LLM: the ceiling we measured against was wrong

We have been quoting a number that does not hold, in this file and in the
project's own notes, since the first week. It is corrected here in full,
with how it was found and what it changes.

- **The claim.** This machine's memory reads at 83.9 GB/s, the engine's kernels reach 84% of that, and the missing 16% is the cost of unpacking 4-bit weights.
- **What was wrong.** The probe behind that number gave every thread an equal share of the work and waited for all of them. On a CPU with six performance cores, eight efficiency cores and two low-power ones, equal shares on unequal cores measure the slowest core, not the memory. It reported 81.4 GB/s on twelve threads, then a collapse to 69.4 on sixteen - a collapse that does not exist.
- **How it surfaced.** The bare Q4_K product, on a matrix far past the caches, reached 77.6 GB/s on twenty threads while the same machine's "pure read" was measuring 73.4. A kernel that also unpacks and multiplies cannot outrun a loop that only reads. One of the two was wrong, and it was not the kernel.
- **The correct figure.** With the work handed out dynamically - the way the engine's own thread pool has always done it - the read bandwidth is **80.5 GB/s and flat from twelve threads to twenty-two**. The new probe sits beside the old one, which now carries a warning rather than being deleted.
- **What it changes.** The kernels are not at 84% of the machine. On twenty threads they are at **97%**: 77.6 GB/s of 79.6, unpacking included. The missing 16% was never the dequantization - it was the four cores the old measurement could not use. The conclusion it had been used to support, that hand-written assembly has nothing left to win, still stands, and now for the right reason.
- **Where the engine actually is.** On Qwen3-30B-A3B the output projection - one large matrix, read end to end - runs at **77.4 GB/s, 97% of the machine and 100% of the bare kernel**. The engine can reach the metal. A whole token averages 59.3 GB/s, because the other phases do not: the attention projections run at 57.7 and, on Qwen3.6-35B, the routed experts at 40.0. So the gap is not spread thinly across the engine - it sits in named phases. (This entry first went on to say that if every phase ran as fast as the output phase a token would take 24.8 ms and 30.5 token/s would become 40.3. That was wrong, and the entry above it says why: the other phases read a different shape and cannot get there.)
- **Two things we looked for and did not find.** The size of the expert matrices is not the cause: across a twenty-four-fold change in matrix size, at constant bytes and constant dispatches, the throughput moves by 9%. Nor are the cache misses: a *smaller* expert cache is *faster* - on Qwen3-30B, 14 GiB gives 28.0 token/s where 20 GiB gives 27.2, with the hit rate unchanged at 97.7%. Both were announced here as explanations before being measured, and both were wrong.
- **The bit-plane format costs 22%.** `q6kp3` holds the same bits as plain Q6_K, reordered into three planes so that a machine short of memory can read a prefix of them. At the real expert sizes that reordering costs 57.7 GMAC/s against 73.8. Every model in `MODELS.md` is kept in plane form, so on a machine with memory to spare that is 22% paid for an option not taken.
- The engine is unchanged by any of this. What changed is what we know about it, and what we will say about it.

## [2026-09-22] - Janas-LLM: dense models, and where the speed comes from

- Dense models run. A dense feed-forward is a mixture with one expert and no router: with a single expert the softmax of any one logit is exactly 1, so the routed path serves it as it is. `gguf2jns.py` writes the layer's feed-forward as its only slot when the source has no expert count, and the engine allocates a zero router for it - 5 KB a layer, 0.05 ms of the 42 a token costs. Nothing in the forward pass had to learn a second shape.
- **The automatic expert cache was one slot short of the layers, and on a dense model that is every token.** The cache's slots are all as big as the largest, but the size it was allowed to take was the expert region, which is smaller wherever the layers differ - Qwen3-4B keeps `down` in Q6_K in half its layers and Q4_K in the other half, so 1.63 GB of region bought 33 slots for 36 layers. A mixture survives that; a dense model asks for every layer at every token, in order, which is the one pattern LRU cannot serve: **0% hits, 3.2 token/s**. The cache is now allowed the room its slots actually take: 36 slots, 100% hits, **26.1 token/s**. Mixtures gain from it too - Qwen3-30B can hold all 6144 experts where it used to hold 93.5% of them.
- The reasoning markers are no longer printed. The opening `<think>` is written into the prompt and never generated by the model, so letting the closing one through put half a pair on the screen; the colour of the reasoning already says where it begins and ends.
- The dense Qwen3 ties its output head to the embedding table, so `output.weight` is no longer required: where it is absent the table is the head.
- Qwen3-4B against llama.cpp on the same weights, machine idle: **24.9 token/s decoding against 17.9, 77.3 prefilling against 66.2**. Both read the same 2.49 GB a token; Janas moves them at 62 GB/s, llama.cpp at 44.6 GB/s. (The first version of this line put those 62 GB/s at 74% of an 83.9 GB/s ceiling. That ceiling was wrong - see the entry below.)
- That margin is not the streaming. On a 4B nothing is read from disk - the expert cache never misses - and the ratio is the same 1.39x it has always been. On Qwen3-Next-80B, where 45 GiB have to pass through 30 GiB of memory, it becomes **2.38x** (24.5 against 10.3), and **3.19x** with the multi-token block (32.8). The engine's own kernels win the first factor; streaming wins the second.
- Correctness of the dense path against llama.cpp: 397/400 on C source, 379/400 on Italian, and blocks of 64 bit-identical to decoding one token at a time.
- A 4B dense model decodes slower than a 30B mixture (24.9 against 29.7), because dense means every parameter is active: 2.49 GB a token against 1.82. The project's premise, measured rather than argued.

## [2026-09-21] - janas-chat: /about, and pages that can be read

- `/about` says what this is and what it is running on: the version, the author, the licence and the repository beside the mascot, then the model, the machine and the engine as they are at that moment. Nothing in it is new - it was all there, in the line at the start, in the terminal's own idea of the processor, in what the engine was told to do - but it was to be gathered by hand from three places, and a report of a problem needs exactly those three.
- `/help` and `/about` are pages now rather than a screenful printed once. The text is laid out to the window however narrow it is, the arrows and Page Up and Page Down move in it, and **ESC** closes it - no other key does, since a page that shuts at the first key touched shuts while you are looking for a way down it. Before this, in a window narrow enough to want a big font, the help was a text the terminal broke wherever it liked, with no way to reach the end of it.
- The help itself is a command to a line, in the colour of the program, with what it does under it and a blank line between one and the next. Two columns were a column of wrapped fragments as soon as the window narrowed; this reads the same at any width, and `/about` is built the same way, a section name and its text.
- The model was named in three places at once - the banner, the bar and `/about` - so the banner no longer names it. The bar says it and goes on saying it while the banner scrolls away, and `/about` has the whole of it.
- The default system message says who wrote the engine: asked what it is, the assistant answers Janas, the model it thinks with, and Janas-LLM written in C by Maurizio "camauri" Cammalleri.
- A space that a line break carried to the head of the next row is dropped now, and one that hung over the end of a row is no longer part of it - it was being copied out with the text. The nothing after the last newline of a page is no longer a blank row of its own.

## [2026-09-21] - janas-chat: a narrow window keeps what matters

- A big font is a narrow window, and the chat was written for a wide one. The line at the foot was built whole and cut at the edge, so the first thing lost was whatever came last in it - the speed, which is the one number a reader watches while a reply arrives. It is made of parts now, each saying how long it is to be kept: the cache goes first, then the bits and the experts, the name of the model, the context, the state, the average, and the speed last of all, still there in twenty columns. A part is in or out, never half printed, since a number cut in two is worse than none.
- The speed was also written twice while a reply came - once in the state, once in its own field, both out of the same statistics - and the average had no unit beside it. Fifteen columns back, and a number that says what it is.
- Beside the mascot in the banner there are nineteen columns less than the window has, so in a narrow one the name of the model and the line about the commands ran off the edge and came back at the first column, under the drawing. They are laid out now, and what does not fit goes on under the line above, from the column where `Janas-LLM` begins; the banner grows past the mascot when the words need it, and below twelve columns of room the mascot goes altogether, a drawing that costs the words their room not being worth it. The layout is the rule the conversation already uses, not a second one written for the banner.
- Both of them used to count bytes where they should count columns, so a name with an accent in it was cut short and then padded past the edge.

## [2026-09-21] - build: `./build.sh --help`, and a clean that stays in its variant

- The script had no help at all, and the part nobody can guess is the variants: a name like `ubsan` says nothing about signed overflow or a shift as wide as the type, and a sanitizer that is built but never run reports nothing whatever, since it speaks at run time. `./build.sh --help` now says for each one what it catches, what it costs and when it is the one to reach for, with the actions, where the objects and the programs land, and the three environment variables (`CC`, `AR`, `JANAS_CFLAGS`).
- `clean` on its own leaves a tree that does not build, which is seldom what the hand that typed it wanted. `clean-build` and `clean-test` pull down and put back in one go - the first for a changed compiler or a doubt about a stale object, the second as the honest answer before a commit. `clean` stays, because freeing the disk is a reason of its own.
- And `clean` now removes what its variant made and nothing else. Release has no suffix to match, so it had been sweeping away the programs of `asan`, `ubsan` and `tsan` as well, leaving their objects behind: the next sanitizer run found nothing to run and rebuilt in silence. An unknown action is an error now rather than a quiet `build`, and a variant typed where the action goes is told where it belongs.

## [2026-09-21] - llm: the model line keeps what it says about the experts

- The line that describes an open model was built in ninety-six bytes and kept in forty-eight. With the down matrix at three bits and the number of experts chosen by the tuner the sentence is sixty-two characters, so what was cut was the end: how many experts a token is given, and that it was the engine that decided, not the reader. Both buffers now fit what can be written into them; `janas_llm_describe` still cuts to the buffer the caller hands it, so nothing changes for a program using the library.

## [2026-09-21] - janas-chat: settings last for the conversation, unless they are saved

- What a command changes has always lasted only as long as the chat is open, which is right - a knob turned to try something should not quietly become the way things are - but there was then no way to keep a setting one had settled on. `/save-config` writes them to `~/.config/janas/chat.conf` and `/load-config` reads them back, losing whatever is set at the time - which it asks about first, since what it throws away was typed on purpose - and `/del-config` throws the file away, so that the next start uses the chat's own settings again. That one asks first too.
- The file holds the words of the command line, one option to a line and the rest of the line its value, so one reader takes both and a setting cannot mean one thing typed and another saved. It is read before the command line, which therefore wins, and nothing is written to it unless it is asked for. The system message is not kept, and the context, the cache, the bits, the attention and the MTP block are read at the next start, since they are settled when the model is opened - `/load-config` says so when it has read one of those.

## [2026-09-21] - janas-chat: the GPU can be told no, and the line says which it is

- `--no-gpu` at the start and `/gpu [on|off]` during a conversation say whether the GPU may be given work at all, without going through `--mode eco`, which decides other things besides. The engine gained `janas_llm_set_gpu` for it, and the FreeBASIC binding has it too.
- The line at the start is built when it is asked for rather than when the model was opened, so it tells the truth as the engine changes its mind: whether the GPU is given the long passes is something the tuner decides pass by pass, and it was being reported once, at the only moment nothing had been decided yet.

## [2026-09-21] - janas-chat: the line at the start says what the GPU does, not that it is there

- "GPU available" told the reader nothing they could act on: the engine opens the GPU unless it is in eco, and the tuner then decides pass by pass whether to give it work - on this laptop it takes the block passes of one model and none of another. The line now says which it is - "GPU present (active, on blocks)", "GPU present (not used)", "GPU not present" - and says it in all three cases, silence being the one answer that left the reader guessing. When it is given work it names which passes it is given, since that is the part worth knowing; `/mode` has the same thing class by class, with the times behind the choice. `/mode` has the whole of it, class by class, and is live as the tuner changes its mind.

## [2026-09-21] - janas-chat: the reply is laid out, not left to the terminal

- The chat printed the reply and let the terminal break the line at its edge, which it does wherever the edge falls: "per" came out as "p" with "er" on the row below. It lays the text out itself now, breaking after the last space that fits, so a word is never cut - unless it is longer than the row and has nowhere to break, where it is cut as it must be. A space at the end of a row is allowed to hang over the edge, or a word that fits exactly would be pushed down by the space in front of it.
- A window that is not full fills from the top, as a terminal does: drawing it instead of printing into it had put the banner of a conversation just begun at the foot of the screen, with the empty room above it.
- One rule does it, and it is the one that was already there for reading the conversation back and for re-wrapping it when the window narrows: what streams and what is read again are laid out by the same code, so they cannot disagree. The window is drawn rather than printed into, which costs a few kilobytes a token and buys a line that breaks where a line should.

## [2026-09-21] - janas-chat: the assistant is called Janas, and says which model it thinks with

- The default system message told the model it *was* the model - "You are Qwen3 Next 80B A3B Instruct, running through Janas" - so asked who it was it gave the file's name and nothing else. The assistant has a name of its own now, Janas, and the model is what it thinks with: asked who or what it is, it is told to give both. Left to itself a model answers out of its training, and they very often claim to be somebody else's assistant, which is why the message exists at all.

## [2026-09-21] - janas-chat: the bar keeps its own colour, and says the average

- The bar at the foot of the window came out grey while the model was writing its reasoning and white the rest of the time, because saving the cursor saves the colours with it and the bar was drawing its own dimming on top of whatever was underneath. It sets the colours it wants from nothing now, and so do the rules around the box.
- `token/s` is `tok/s` there and in the counters, which is what `janas-bench` has always written, and the bar says the average of the whole conversation beside the speed of the last reply: one long answer and one short one are two different numbers, and the second on its own says little about how the machine is doing.
- A mark that closed at the end of a line - and `**` at the end of a line is where a model puts most of them - was printed instead of read: the newline was looked at before the two stars being held for it, and it flushed them as text. What is held is decided first now, the newline included.

## [2026-09-21] - build: harder compiler settings buy nothing here, measured

- `build.sh` takes `JANAS_CFLAGS` for trying a compiler out - nothing is added when it is unset, so the released build is the one it always was - and uses whatever `AR` names, since link-time optimization needs an archiver that reads its own objects. It was written to answer a question with a number rather than an opinion: is `-O2` leaving anything on the table?
- It is not. Four builds - `-O2`, `-O3`, `-O2 -flto`, `-O3 -flto` - on the naked Q4_K product over a matrix past the caches: 69.4, 69.2, 68.9, 68.9 GB/s. On the engine itself with Qwen3-Next-80B-A3B: 24.10, 24.14, 24.00, 24.16 token/s. Nothing outside the noise, and it is not a surprise: what the time goes into is hand-written AVX2 and AVX-VNNI, which a compiler translates almost one for one, and the rest waits for memory. The settings would have to make the machine read fewer bytes, and no setting can.
- Trying them did find something, though: with the whole program in front of it the compiler could no longer see that two variables in the chat are only read where the call that fills them has succeeded. They are initialized now - it costs nothing and the reasoning is written beside them.

## [2026-09-21] - janas-chat: a message of several lines is written as one

- A backslash at the end of a line and Enter used to send that line and wait for the next, so a message of three lines was three trips through the editor, each with its own rule drawn under it - and the backslash and the newline it stood for both went to the model, which had to make sense of them. Now the backslash asks for a line and is gone once it has been given, the way it would be in an editor: the message stays in the box, the box grows a row, and Enter on a line that does not end in one sends the whole thing. What the model reads is the message, with its newlines and without a backslash in sight.
- The box and the line repeated in the conversation break a row wherever a newline is, and the cursor is followed by its byte rather than by its column, because a newline takes no columns and two places either side of one would otherwise be the same place.
- Up and down move through the message while there is one to move through, keeping the column as an editor keeps it, and bring back earlier prompts once the cursor is off its first or last row. They follow the rows on the screen, so they walk a long line that has wrapped as well as one the writer broke.

## [2026-09-21] - janas-chat: the marks a model writes are read, not printed

- Models mark their replies up and the marks were printed as they came: `**` around a word, three backticks around a block, hashes before a heading. Printed they are noise; dropped, the reply loses what they meant - a block of code is worth telling from prose. They are read now and turned into what the terminal can do: bold is bold, emphasis is emphasis, code takes a colour of its own, a dash or a star at the start of a line becomes a bullet, and the language written after a fence is not printed.
- The text arrives a piece at a time and a mark can be cut in half by the end of a piece, so the few bytes that might yet be a mark are held back until what follows says what they are; nothing is lost, the end of the reply flushes them. Checked on the same text fed one byte at a time, four at a time and seven: the same output all three ways.
- Underscores are left alone on purpose: in a conversation about code `snake_case` is everywhere, and every renderer that treats `_` as emphasis turns half of it into slanted nonsense. A star opens emphasis only when something follows it and closes it only when something came before, and never inside a word, so `2*3` stays a product. `--no-markdown` and `/markdown off` print the marks as they come; a pipe always gets them, because a script may want them.

## [2026-09-21] - janas-chat: the box grows with the message

- A message longer than the window scrolled sideways inside a box one row high, so what had been written was out of sight while it was being written. The box now takes the rows the message needs - up to a third of the window, so that a long question never leaves the conversation without room - and goes back to one when the message has been sent. Past the cap it shows the rows the cursor is on, which are the ones being written.
- Growing it moves the scrolling region, and the place the conversation writes from can end up under it, which is how a footer of this kind gets written over. Each time the region moves, the window above is drawn again at its new size and that place is put back where the window now ends.

## [2026-09-21] - janas-chat: the conversation can be read again

- A terminal that pins a footer does it with a scrolling region, and the lines that leave the top of a region never reach the terminal's own history: they are gone, and Page Up found nothing to show. The chat keeps them itself now - as the lines it wrote, escape sequences and all, not broken into screen rows - and draws a window on what it kept. **Page Up** and **Page Down** move it, **Ctrl-Home** and **Ctrl-End** go to the beginning and the end. The line being written stays where it is while you read, and sending brings the end back, because asking is wanting to see the answer.
- Lines are wrapped when they are drawn, not when they are written, so making the window narrower re-wraps the conversation instead of ruining it, and a row that starts in the middle of a coloured line is drawn with the colour it inherits. A reply that arrives while you are reading further up waits below instead of throwing you to the bottom.
- The keys a terminal sends were read three bytes at a time, which was enough for the arrows and Home but not for Ctrl-Home, which sends five: the rest of the sequence used to be typed into the line. The chat reads the whole of a sequence now, whatever its length.

## [2026-09-21] - janas-chat: a column is not a character

- The chat measured its lines by counting the lead bytes of the UTF-8, which counts characters, not the room they take: a tick or an ideogram fills two columns and a combining accent none. A model puts ticks and arrows in its replies all the time, so the line the user wrote came back wrapped a column short of where it should, and on Chinese or Japanese a line of eighteen characters was drawn twice as wide as the window. Widths now come from `wcwidth()`, and the chat asks the system for the user's own locale, without which that function knows nothing past ASCII. Where `wchar_t` is not Unicode the old count is kept: right for everything but the wide and the combining.
- Checked on ticks, ideograms, combining accents and the box-drawing characters the chat itself uses, and on a window of twenty columns, where a run of ideograms now stops at nine to the line.

## [2026-09-21] - janas-chat: a blank line between one block and the next

- The line the user wrote, the reply, the counters, what a command answers: each ran straight into the one after it, and the counters in particular read as though they belonged to the last paragraph of the reply. Every block now asks for the space it needs before printing (`term_gap`), so there is exactly one blank line between any two and none at the top. One place decides, which is how the two the editor and the counters were each printing became one.
- A reply that ends on a newline of its own no longer gets a second one, and none of this happens on a pipe: a script reads the lines, and the spacing is not for it to guess at.

## [2026-09-21] - Janas-LLM: turning the drafts off and on again brings them back

- `/spec off` did not turn the drafts off: it turned the model's multi-token prediction block off. The session read "this conversation uses the block" and "I want drafts now" as one flag, so switching the drafts off stopped the block from following the tokens, and switching them on again found a block whose cache had holes and a conversation past its beginning - the one case the code refuses, for a good reason - so nothing drafted again until the conversation was reset. Seen on the development laptop: 25.3 token/s before the switch, 20.6 after, and not one draft in between. The two are now two things. The block keeps noting the tokens whether drafts are wanted or not, which is one pass of one layer and no vocabulary head, and the drafts come and go as asked.
- The planner that decides how many tokens to draft then refused to draft anyway, about two times in three. It weighs the cost of a plain pass against the cost of a pass over a block of drafts, and it had timed the blocks before the pause and the plain pass after it - but a conversation is not where it was half an hour earlier, because the expert cache warms as it goes: on this laptop a plain pass fell from 73.7 to 43.1 ms over five turns. It now forgets those timings when the drafts come back and measures them again. Three runs of the same sequence, all three the same: the first reply after `/spec on` runs at 32 token/s with every draft accepted, against 16.7 with the drafts off. Read that 32 for what it is - the best turn there is, the one whose expert cache two turns on the same subject have just warmed. Over the two replies that follow the switch it averages 27, and a conversation that moves from one subject to another gives 25 to 27: that is the figure to expect, and it is the same with the sampling greedy or at a temperature of 0.7, which costs 9% on one reply and gives 12% back on the next.
- Both are the shape of fault this file had twice already: a decision taken from a measurement that has aged, which then stops the measurement being taken again. The drafts' own switch-off was the first (fixed earlier today), and it is worth saying plainly that none of the three was ever caught by a test, because none of them makes anything wrong - only slow.

## [2026-09-21] - Janas-LLM: the drafts are not switched off on one unlucky pass

- The engine drafts tokens ahead and keeps the drafts the model itself would have written, and it stops drafting where it judges the drafts are not paying for themselves. That judgement was made on one measurement, and on a comparison of unlike things: the cost of a plain pass is learned only on the passes that make no draft, which are the first of a reply, where the context is shortest and the expert cache warmest, while the cost of a speculative pass is learned all the way through a reply. Read as it was, the comparison leaned towards switching off - and then switched off for 32 passes, then 64, then 128, up to 512, longer than most replies, with the way back needing the single speculative pass at the end of that count to go well. Seen on 21 Sep 2026 on the development laptop: after one such judgement a reply of 1301 tokens came out at 19.0 token/s where the drafts were giving 25.6, and no later reply in that session ever drafted again.
- Now the drafts must lose by more than 5%, must lose three judgements running, and must be judged against a plain cost measured within the last sixty-four passes - without one that fresh there is nothing honest to compare them with, and they stay on. The longest they go off for is 128 passes instead of 512. Every one of those conditions makes switching off rarer than it was, so the worst this can do is what the engine did before the judgement ever ran, which is never switch off at all.
- The counter that would have shown this - passes run without drafting although drafts were asked for - existed inside the engine and reached nobody. It is now the last field of `janas_llm_chat_stats` (the struct carries its own size, so callers built against the older header are unaffected), it is in the FreeBASIC binding, and `janas-chat --stats` prints it: "drafts off N passes (the engine judged)". A reply that is slower than it should be now says who decided.

## [2026-09-21] - tests: a test that deadlocks fails instead of waiting

- A test that hangs is worse than one that fails: the suite stops and says nothing. A deadlock in the thread pool, introduced and caught the same afternoon, held it for twenty-two minutes before anyone looked. Every test now sets an alarm of ten minutes, so it dies, the runner counts a failure and the suite goes on.

## [2026-09-21] - Janas-LLM: a run wakes only the threads it has work for

- The thread pool can be told to use only the first n of its threads, which is how the engine tries one configuration against another. Every run still broadcast on the single condition variable they all waited on, so the threads with no work woke, queued for the mutex, found the run was not for them and went back to sleep - some three hundred times a token. A pool of twenty running twelve cost 6.72 us a run where a pool of twelve cost 1.82: the engine paid 17% for threads it had decided not to use. Those left out now wait apart, on a second condition variable broadcast only when the count grows again, and the cost is gone: 1.82 against 1.81.
- On the development laptop with the twenty threads the engine uses, Qwen3-Next-80B-A3B decodes at 23.77 token/s against 22.72, and steadily: three paired runs spread 0.4 token/s where before they spread 3. Over a longer reply, 512 tokens, 24.05 against 20.94. With twelve threads, where no thread is ever left out, nothing changes either way.

## [2026-09-21] - Janas-LLM: the engine learns again

- The engine tries every configuration it can use on the passes it runs anyway and keeps the fastest, and it does not learn while the expert cache is still filling in the background, because a pass that waits for the disk says nothing about the configuration. It asked "is it filling?" by comparing the experts the filling had loaded with the number it meant to load - and that comparison is never equal: the filling only reads while nobody is asking, and it stops as soon as the cache is full, so the experts the caller fetched for itself fill slots the filling then never counts. Measured on 21 Sep 2026: in a session that generates without pause the filling loaded **nothing at all**, 0 experts of 5596, and the engine believed it was still settling from the first token to the last. So it never learned, in any session; and `janas-bench`, whose whole purpose is to leave the machine tuned, threw away every measurement it took - two full runs left the saved profile identical to the digit.
- The question is now whether the filling has read anything in the last quarter of a second, which is what it was always meant to ask. The saved profile becomes truthful: for Qwen3-Next-80B-A3B it now holds 46.4, 41.2 and 40.5 ms a token for the three thread counts, against a true 41, where before it held 54.3 and 65.9. Profiles are keyed v2 instead of v1, since what earlier versions wrote there was measured under the old rule and cannot be compared with what is written now.

## [2026-09-21] - Janas-LLM: the activation is quantized eight lanes at a time

- Before every product the engine turns the vector it multiplies into Q8_K, and that one step had stayed scalar: three passes over each block of 256 values, one `lrintf` at a time. It runs on the thread that drives the phase while the other eleven wait for it, so on an activation of 4096 values it cost 6.83 us, and a layer has several. Eight lanes at a time it costs 1.00 us. The answer is the same to the bit - the largest absolute value is a maximum, which no order changes, the scale and its reciprocal stay the two scalar operations they were, and the vector conversion rounds to nearest with ties to even, which is what `lrintf` and `nearbyintf` do - so every logit that follows is unchanged. `test_quant` checks it on zeros, on values that land exactly halfway, on tiny and on enormous ones, for both the plain and the deterministic quantizer, and fails if one byte differs.
- On the development laptop, paired runs, machine idle: Qwen3-30B-A3B decode 27.70 to 28.53 token/s, Qwen3.6-35B-A3B 25.55 to 26.08, Qwen3-Next-80B-A3B 23.60 to 24.41. In the phases where it was worst, the projection after attention goes from 4.28 to 3.76 ms a token, which is 89% of what its own kernel does against 79% before.
- `bench_long` takes the number of compute threads as an optional last argument. Without it, the twelve threads on CPUs 0-11 every measure of this file has used so far; with it, the CPUs the engine itself picks. It was written to find out whether the benchmarks had been measuring a configuration the product does not use: they had not, twelve is also what the engine chooses, and it is the fastest of the three tried (12, 16 and 20 give 23.72, 23.64 and 22.52 token/s).

## [2026-09-21] - Janas-LLM: the experts' reads start without waking anybody

- Reading an expert the cache does not hold was handed to a coordinator thread, which woke the I/O pool: the hand-over alone measured 271 us a time, 2.43 ms of a 46 ms token, because a thread woken by a busy one lands on its CPU and takes its turn. The engine now queues those reads itself, with io_uring - its system calls, not liburing: for reads the whole of it is three calls and two shared rings - so the kernel performs them while the engine computes on the experts it already has, and nobody is woken at all. Where there is no io_uring, an old kernel or a sandbox that forbids it, the old way is still there and still whole.
- Neither way is better than the other, and the engine picks one for each batch of reads. The ring costs nothing to start, but the kernel makes its readers one at a time, as each blocks, so a short queue is read almost one read at a time; the pool has its eight threads already there, and costs the wake-up. So: the ring for a batch of at most four reads, which is a reply being written one token at a time, where that wake-up is the whole cost; the pool for what is in between; and the ring again past four times the I/O threads, a queue deep enough that the kernel's readers are by then more numerous than the pool's and free to run on any core.
- Qwen3-Next-80B-A3B on the development laptop, three paired runs each, machine idle. With an expert cache of 20 GiB, which holds three quarters of what a prompt touches: decode 23.42 to 23.76 token/s, and the prefill unchanged, 32.73 to 32.90. With 8 GiB, the machine short of memory this engine is for: decode 19.66 to 20.38, prefill 26.07 to 27.73. The expert phase of a decode step gives up 0.61 ms of 46 in the first case and 1.37 ms of 54 in the second, and the time waited for the disk falls by a sixth to a half.

## [2026-09-20] - Janas-LLM: the Q6_K kernels build their scales once

- The Q6_K kernels, plain and in bit planes, rebuilt the scale vector of every eighth of a block with two broadcasts and an insert - a third of their instructions went there, and at the low bit levels, where there is least else to do, it showed most. They now take the block's sixteen scales once, spread both halves over both lanes, and pick each pair with one shuffle. The weights, the sums and the results are what they were, to the last bit: against llama.cpp the logits agree exactly as before (393 of 400, mean 0.1714), and on plain AVX2 too.
- On a 122880 x 2048 matrix with twelve threads: the two-bit plane product 1.43x quicker, four-bit 1.12x, six-bit and plain Q6_K 1.06x. In the engine, where the experts' down matrix and the output head are the ones that use them, a decode step goes from 22.2 to 22.5 token/s with the experts at four bits and from 22.1 to 22.7 at two - the level a machine short of memory falls back to, which is the one this engine is for.

## [2026-09-20] - Janas-LLM: one undefined shift out of the reference

- The scalar reference of the bit-plane dot product shifted a signed sum left, which C leaves undefined when that sum is negative, as it often is. It multiplies by the power of two instead - the same numbers, the same instruction once compiled, and `./build.sh ubsan test` now finishes clean. The vector kernels never had the problem: `vpslld` shifts a bit pattern and says so.

## [2026-09-20] - Janas-LLM: a Q8_0 token embedding is read as Q8_0

- A model whose `token_embd.weight` is Q8_0 - unsloth's "dynamic" quantizations among them - loaded, passed its checksums and then answered nonsense: the embedding was unpacked as Q4_K whatever its type really was, which made the numbers hundreds of times too large and the first layer produce NaN. Every row the engine has to unpack whole now goes through one place, `janas_dequantize`, which reaches the right unpacker for each of the four types a model file may carry and refuses anything else instead of guessing. Found and diagnosed on an Intel i7-1365U, with the two-line remedy already worked out (issue #1).
- `test_quant` now checks that unpacking a row of every supported type gives what that type's own unpacker gives, and that a row read at its own offset in a table gives the same numbers: put the old behaviour back and both checks fail.

## [2026-09-20] - Janas-LLM: attention scores as exact integer sums, when the context is long

- The attention scores can now be computed two ways, and the engine keeps both. The exact one reads the query as it is and sums in floating point, as before. The fast one reads it as sixteen-bit integers with a scale of its own per token and head: the dot product is then an *exact* integer sum, which AVX-VNNI does sixteen products at a time (`vpdpwssd`) and plain AVX2 in two instructions (`vpmaddwd`, `vpaddd`) instead of eight floating-point ones. On the development laptop that is 1.22x off a decode step at 16384 positions and 1.26x off a prefill block, and the scores kernel alone runs at 71 GMAC/s against 39 on one core.
- Every way of computing them agrees to the last bit: scalar, AVX2 and AVX-VNNI sum the same integers, and the eight query heads of a group are reduced together by a tree of adds without changing a single result. A block of tokens still gives what the same tokens give one at a time. The test prints a checksum of every output it produced, so the two instruction sets can be compared by running it twice.
- What the fast way costs is the query's quantization, a relative 3e-5 - against the 4e-3 the eight-bit keys and values already carry. Measured on four hundred tokens against llama.cpp: the most likely token agrees 394 times of 400 on C source (393 the exact way) and 377 of 400 on Italian prose (382). The engine therefore asks for it by itself only above 16384 tokens of context, where attention is the greater part of a token and a conversation will reach the far end; `janas-chat --attention exact|fast`, `janas_llm_params.attn_scores` and `JANAS_ATTN=float|int16` settle it either way, and `janas_llm_attn_scores` says what is in use.

## [2026-09-20] - Janas: three forms, so a report carries what it needs

- Three issue forms in `.github/ISSUE_TEMPLATE`. The hardware report asks for the file `janas-bench` writes and then for the four things that file cannot know - distribution and kernel, memory free at the time, the kind of disk the model sits on, which model - because a measurement missing any of them cannot be compared with another machine. The other two are for something that went wrong, which asks for the banner Janas prints before anything else, and for a model that will not convert, which asks for the GGUF's own SHA-256 so that the file in question is beyond doubt.
- The README and CONTRIBUTING now link the forms where they ask for a report.

## [2026-09-20] - Janas-LLM: the models are a recipe and a fingerprint, not a download

- `MODELS.md` now carries the whole chain for every model this project runs: the Hugging Face repository each GGUF came from, its SHA-256 as Hugging Face itself publishes it, the two commands that convert it, and the SHA-256 of both files that come out. Nothing is uploaded, and nothing needs to be: `tools/gguf2jns.py` and `tools/jns_planes` are deterministic — the layout follows from the sizes in the source, nothing is timed, drawn at random or read from the environment, and the weights are copied, never recomputed — so a fingerprint is worth as much as the file itself.
- Measured, not assumed. Each of the four models was converted again from its GGUF and compared byte for byte with the file the numbers in this repository were measured on; all four matched. The two steps cost 47 s and 104 s for the 18.6 GB model, 153 s and 194 s for the 48.4 GB one, less than the download either way. The multi-token prediction block, which comes from the original checkpoint instead, gives the same bytes on two runs.
- The README pointed at the wrong repositories: three of the four GGUFs come from Qwen's own, not from ggml-org's or unsloth's. It also left out the bit-planes step, although that is the form the models are kept in here, and understated the room a conversion needs: three copies of the model at the worst moment, about 145 GB for the largest.

## [2026-09-20] - Janas-LLM: the FreeBASIC example reads a pipe, and prints what it was given

- `tools/chat.bas` now works in a pipe as well as at the terminal. FreeBASIC's `line input` reads the console, not the standard input: given a pipe it waits for a terminal that will never answer. The example asks `isatty` and, when the input is redirected, reads it through the `CONS` device; at the terminal nothing changes, `line input` keeps its line editing.
- It also prints the reply as the library hands it over. `janas_llm_chat_next` returns the length of the piece and does not close it with a NUL, so the example leaves room for one and writes it: before, a short piece carried the tail of the one before it, and the reply came out as `Okay,kay the user iser asking`. The two lines shown in the README had the same fault.
- The counters fit their format again: four hundred tokens in a field of two digits printed `%400`.

## [2026-09-20] - Janas-LLM: long context, and a library for FreeBASIC

- The KV cache is kept in eight bits with a scale of its own per position and head, instead of half precision: half the memory, so twice the context in the same machine, and at a long context - where attention is bound by reading it - half the work. Nothing is lost by it: against llama.cpp on Qwen3-Next the most likely token now agrees in 397 positions of 400 on C source where half precision gave 394, and 386 of 400 on Italian prose where it gave 383, since a scale that follows one head's numbers describes a narrow range better than a fixed exponent. A decode step at 65,536 positions costs 3.6 ms a layer.
- The context stops where the model was trained (262,144 tokens for Qwen3-Next, 40,960 for Qwen3-30B) and is paid for out of the expert cache: asking for 131,072 tokens on the development laptop leaves 15.2 GiB of cache instead of 20.5, and runs.
- `include/janas/llm.bi` declares the library for FreeBASIC, and so for BASIC MODERN, the dialect Prabanta implements; `tools/chat.bas` is a chat in sixty lines that uses it.

## [2026-09-20] - Janas-LLM: a chat that does not stop, and a window of its own

- The context slides instead of ending. A conversation that fills it now forgets its oldest exchanges - whole turns, never half a message - and keeps going at the same speed; the system message, or the first few tokens when there is none, stays where it is, since attention leans on those positions. In a hybrid model the recurrent state is left untouched: it has seen everything, and that is what carries the older text once the window has passed it. Twelve turns in a window of 1024 tokens run through without stopping, and the model still answers what the cat in its story was called.
- `janas-chat` takes the terminal: a banner, the conversation scrolling under it, and a footer that stays at the bottom with the model, the bits and experts in use, the context filled, the speed of the last reply and the state of the expert cache. Lines are edited with the arrows, Home, End and the usual control keys, and earlier prompts come back, kept between sessions. Colours follow `NO_COLOR` and `TERM`. Piped in or out, it stays the plain thing it was.
- How many bits the experts' down matrix is read with, and how many experts a token uses, now follow the memory the machine has: every bit when the experts all fit in the cache, four otherwise; all the experts unless the cache holds less than a fifth of the model. On a machine with a cache of 6% of the experts, that is 21.9 token/s instead of 15.0.

## [2026-09-20] - Janas-LLM: the expert cache is filled before the first reply

- The engine remembers which experts this machine uses with a model (a small file in `~/.cache/janas`, added to at every session) and fills the cache with them when the model is opened, instead of letting the first replies find them one token at a time. With Qwen3-Next-80B-A3B on the development laptop, preloading 11,360 experts costs 3.9 seconds once: the second reply of a chat - the one that used to finish filling the cache - goes from 17.9 to 24.5 token/s, its prompt from 23.5 to 46.7 token/s, and it reads 3.0 GB from disk instead of 14.3, waiting 0.35 seconds instead of 3.35. The first short reply goes from 10.3 to 15.3 token/s.
- `janas-chat --no-preload`, `janas_llm_params.no_preload` and `JANAS_WARM=0` turn it off; the startup line says how many experts were preloaded and how long it took.

## [2026-09-20] - Janas-LLM: experts' down matrix in bit planes

- A model file can now hold the down matrix of every routed expert as three planes of two bits instead of one block of six, which takes exactly the same room: the engine then reads as much of each expert as the machine can afford - all of it (the weights as they were quantized, bit for bit), two planes, or one - and the choice is made when the model is opened, from the same file. `tools/jns_planes` rewrites an existing model file into the new form without going back to the original GGUF; files in the old form keep working.
- With Qwen3-Next-80B-A3B on the development laptop, four bits per weight instead of six: 21.5 token/s instead of 19.6 with the most likely token agreeing with llama.cpp in 389 positions of 400 against 394, and in a real chat a quarter to a half fewer bytes read from disk (Italian prose 8,969 -> 3,572 MB, C source 19,895 -> 13,386 MB) with the waits for them cut by half. Two bits: 21.9 token/s and 381 of 400.
- `janas-chat --bits 2|4|6`, `janas_llm_params.expert_bits` for the library, `JANAS_EXPERT_BITS` for measurements. Six bits, the whole matrix, stays the default.

## [2026-09-20] - Janas-LLM: experts per token as a dial

- How many experts a token uses is now a choice, not a constant of the model: `janas_llm_set_experts`, `janas_llm_experts`, `/experts n` in `janas-chat` and `JANAS_EXPERTS` for measurements. The experts left out are those the router weights least, so fewer of them means fewer bytes read per token: with Qwen3-Next-80B-A3B, eight of its ten give 21.9 token/s instead of 20.6 on the development laptop, and the most likely token still agrees with llama.cpp in 387 positions of 400 against 394 for all ten.
- In a real chat the dial pays more than the bare arithmetic suggests, because a smaller working set fits the expert cache better. Same question, warm cache, temperature zero, Qwen3-Next-80B-A3B on the development laptop: with all ten experts 23.6 token/s out and 30.5 in, 16,615 MB read from disk; with six, 26.4 out (+12%) and 41.9 in (+37%), 9,284 MB (-44%). The hidden cost is the multi-token draft, which predicts the whole model: its drafts are accepted 69.3% of the time instead of 77.0%.

## [2026-09-20] - Janas-LLM: Qwen3.5 and Qwen3.6, reasoning models

- Qwen3.5 and Qwen3.6 mixture-of-experts models (`qwen35moe`, e.g. Qwen3.6-35B-A3B): the same layers as Qwen3-Next with separate DeltaNet projections, a different key-head sharing (settled against llama.cpp's graph with a new per-layer trace) and the multi-token prediction block inside the model file. Top-1 agreement with llama.cpp 197/200 on C source; the internal MTP accepts 87% of its drafts (32.3 token/s decoding, 76 prefill on the development laptop).
- Q5_K and Q8_0 weights, used by recent GGUF files, with the same exact per-block arithmetic as the other formats.
- The qwen35 pre-tokenizer (word runs take combining marks): identical to llama.cpp's tokenizer on every corpus.
- Reasoning models: the reasoning part of a reply is marked (`janas_llm_chat_thinking`), can be turned off (`thinking`, `/think`) and is printed in grey by `janas-chat`.
- `tools/gguf2jns.py` reads models split into several GGUF files, so Qwen3-Coder-Next (80 billion parameters, the same architecture as Qwen3-Next) runs with no change: 194/200 top-1 agreement with llama.cpp.

## [2026-09-19] - Janas-LLM: self-tuning engine and janas-bench

- The engine tunes itself on the passes it runs: for one token, 2-16 tokens and blocks it tries the configurations the machine offers (one thread per performance core, all their threads, the efficiency cores too; with and without the GPU), measures them net of disk waits, keeps the fastest and checks the others again now and then. Choices are remembered per machine and model in `~/.cache/janas/tuning.txt`.
- Power modes: `eco` never uses the GPU for speed alone, `max` takes the fastest configuration, `auto` (default) follows the power source; `janas_llm_set_mode`, `janas_llm_tuning`, `/mode` in `janas-chat`.
- `janas-bench`: measures prefill and decoding speed (also with MTP drafts) for every configuration, fills the tuning profile and writes a report to share. On the development laptop the efficiency cores speed decoding up by 16%.

## [2026-09-19] - Janas-LLM: shared CPU/GPU arithmetic, optional GPU, adaptive threads

- Q4_K and Q6_K products compute exact integer sums per block and one fma per block and vector; the scalar reference, the AVX2 and AVX-VNNI kernels and the GPU shader perform the same float operations and agree bit for bit (tests check equality). Faster on blocks (12288x2048, 8 vectors: 0.68 -> 0.62 ms).
- Optional integrated GPU (`src/llm/gpu.c`, off by default, `JANAS_GPU=1`): Vulkan loaded at run time, the shader built into the library, the resident weights shared with no copy, products split by rows with an adaptive share, a keep-alive against the GPU's sleep states. Results identical to the CPU's. On the development laptop it brings no net speed-up (CPU and iGPU share the package's power and memory); `tests/bench_power` measures time and energy per product (the iGPU does 2-2.6x the work per joule).
- Adaptive threads: one thread per physical core for passes over blocks, all threads for single tokens; the thread pool runs on a chosen number of threads. Qwen3-Next prefill 56.9 -> 62.5 token/s.
- The chat's automatic expert cache leaves a fifth of the memory free.

## [2026-09-19] - Janas-LLM: tokenizer, sessions, C library and chat

- Own byte-level BPE tokenizer (`src/llm/tokenizer.c`), built from the model file's metadata: the pre-tokenizer's expression as a hand-written matcher over generated Unicode tables (`tools/gen_unicode.py`), merges by rank with a heap (n log n on long words). Identical token ids to the reference on every corpus and a stress text, exact round trip; `tests/tok_check.c`.
- Sessions (`janas_llm_session_*`): a token sequence that grows by appended turns and by tokens generated on request, each computed once, with the KV cache and the recurrent state carried across turns. Sampling (temperature, top-k, top-p, min-p, seed) draws one random number per token, so the text is the same with and without speculation. `tests/llm_session.c`.
- Public C API `include/janas/llm.h` and `libjanas_llm.so` (only the API is exported), shaped for foreign-function interfaces: opaque handles, fixed-width types, int32 error codes, size-prefixed parameter structs, reply text returned on request as whole UTF-8 characters. Chat format and stop tokens from the model's metadata (ChatML); expert cache size and CPU split chosen from the machine.
- `janas-chat`: terminal chat with streaming output, slash commands (`/help`, `/stats`, `/reset`, `/system`, `/temp`, `/spec`, `/quit`) and a per-reply report of input speed, output speed and total time.
- The per-reply report also gives the share of routed experts served from the cache, the bytes read from disk and the time waited for them. `janas-chat` starts with a system message naming the model (from its metadata) and the engine, since models left without one often claim to be another model.

## [2026-09-19] - Janas-LLM: half-precision router and recurrent state

- The router weights are converted to half precision in place at load (no extra memory) and multiplied with an F16C kernel: 3.5 -> 2.1 ms per token on Qwen3-Next, 0.8 -> 0.6 on Qwen3-30B-A3B; agreement with llama.cpp unchanged within noise.
- The Gated DeltaNet state is stored in half precision (computed in float, rounded once per stored row, `janas_gdn_step_h`): half the memory traffic, 2.6-3.0 -> 2.0-2.1 ms per token. Its effect on the logits is within the model's numerical noise; a more precise int16 format with row scales measured the same and ran slower.

## [2026-09-19] - Janas-LLM: MTP drafts for Qwen3-Next

- `tools/hf2jns_mtp.c`: converts the multi-token prediction block, absent from GGUF conversions, from the original safetensors checkpoint into a one-layer JNS file (routed experts as expert slots, the rest resident); Q4_K with Janas's own quantizer (`janas_q4k_quantize`: per sub-block affine fit searched over shrunken ranges and refit by least squares, block factors refit against the 6-bit scales). The checkpoint's conventions were checked on the weights (zero-centred norms, row order).
- `janas_llm_mtp_forward` and the hidden states of the main pass: the block's input order and hidden-state choice were settled on data (`tests/mtp_eval.c`): it agrees with the main model's next choice 88.9% of the time.
- Generation with `JANAS_SPEC_MTP`: after each pass the MTP rows of the accepted tokens keep its KV cache complete, then up to k drafts are chained on its own hidden state. On Qwen3-Next: 1.17x with one draft (89% accepted), 3.12 tokens per pass with three; output identical to plain decoding.
- `build.sh` also builds the C tools in `tools/`.
- Drafts are chosen on a reduced LM head: the 32,768 lowest token ids plus every token seen in the context (95.5% coverage measured), 21% of the full head's reads (`janas_llm_mtp_draft`, `janas_llm_mtp_note_tokens`).
- Draft planner (`spec_auto` with MTP): measured pass and chain-step costs and survival odds calibrated on earlier verifications choose, pass by pass, the number of drafts that maximizes expected tokens per second; the chain stops as soon as the last draft would not be verified. Without any tuning it lands near the best fixed threshold of each text: 1.28x on C source, 1.12x on Italian prose, 1.10x on BASIC. A fixed threshold (`spec_min_conf`) is also available: 1.34x on C source at 0.8.

## [2026-09-19] - Janas-LLM: block products and verification passes

- Quantized dot products over 1 to 8 vectors from one kernel body (`src/llm/quant_dotn.h`), compiled for AVX2 and for AVX-VNNI (`vpdpwssd`): the unpacked weights and one accumulator per vector stay in registers. 12288x2048 Q4_K over 16 vectors: 168 -> 229 GMAC/s on AVX2, 297 GMAC/s with VNNI. Every variant gives bit-identical results (integer sums are exact); `JANAS_KERNELS=avx2` forces the AVX2 path.
- Gated DeltaNet rework: the recurrent state is double-buffered as a base and a tip, so a call that continues the previous one needs no replay; the recurrence is split by state rows (128 work items instead of 32 heads); vectorized convolution with the L2 norms, and gated RMS norm with a vectorized SiLU (`src/llm/vmath.h`). 3.7 -> 2.6 ms per decoded token, 15.2 -> 6.5 ms per 5-token block.
- The router reads each expert's weight row once per block.
- A 5-token verification pass on Qwen3-Next costs 2.47 single passes (was 2.87); lookup speculation now pays: 1.11x with k = 2. Prefill 54 token/s.
- Phase timing gains `ssm` (DeltaNet convolution and recurrence).

## [2026-09-19] - Janas-LLM: Qwen3-Next

- `src/llm/model.c` runs qwen3next (Qwen3-Next-80B-A3B): Gated DeltaNet layers (causal convolution, L2-normalized keys and queries, gated RMS norm) alternating with gated full attention (partial NeoX RoPE on 64 of 256 dimensions), routed experts plus a sigmoid-gated shared expert. The KV cache is allocated only for the attention layers.
- `src/llm/deltanet.c`: the DeltaNet step in a single pass over each state row (decay, key product, update, query product fused); `tests/test_deltanet.c` checks it against the recurrence in double precision.
- The shared expert runs as one more pair in the same dispatch as the routed experts already in RAM.
- Logits against llama.cpp: 393/400 top-1 agreement on a C source prompt, mean absolute difference 0.133; on 64 Italian tokens 0.094, below llama.cpp's own difference between its two kernel paths (0.101). Blocks are bit-identical to token-by-token decoding.
- 16.1 token/s at steady state with a 20 GiB expert cache.
- Rollback of the recurrent state for speculative decoding, at no extra memory traffic: the state stays at the start of the last call and each call replays the accepted tokens of a per-token input log in place before running its own tokens on a cached copy of each head. Speculative and plain generation produce identical tokens.
- `src/llm/attention.c`: attention by GQA group over fixed chunks of 256 positions, merged with an online softmax, so every key and value row is read once for all the query heads of its group and a long context uses every thread. Kernels on 8, 4, 2 or 1 query vectors with the accumulators in registers, a vectorized exp, dynamic work claiming. Chunks start at absolute positions, keeping blocks bit-identical to token-by-token decoding; `tests/test_attention.c` checks both properties, `tests/bench_attn.c` times one layer.
- The KV cache is stored in half precision (the representation llama.cpp uses too): attention at 8,000 positions went from 1.13 to 0.375 ms per layer, and the cache takes half the memory.
- The I/O threads sleep between requests instead of spinning (`janas_pool_set_spin`), leaving the package power budget to the compute cores.
- Qwen3-Next after an 8,000-token prompt within 24 GB (peak resident memory 21.9 GiB, 20 GiB of expert cache): 17.6 token/s decoding, 27.8 token/s prefill. At 400 tokens: 19.2 token/s; Qwen3-30B-A3B: 24.4 token/s.
- Fix: the Q4_K/Q6_K kernels are selected before `main`, removing a data race when the first product ran on pool threads.

## [2026-09-19] - Janas-LLM: speculative decoding

- Forward pass over blocks of up to 64 tokens (prefill, speculative verification), bit-identical to token-by-token decoding; four-vector Q4_K/Q6_K kernels.
- `src/llm/generate.c`: greedy generation with optional speculative decoding. Options `spec_mode` (off, lookup), `spec_k`, `spec_auto`; lookup drafts copy the continuation of the longest recent n-gram match. The output is identical with and without speculation.
- `spec_auto` measures tokens per verification pass against the cost of a plain pass and pauses speculation when it does not pay, retrying with exponential back-off.
- Measured on Qwen3-30B-A3B (C source prompt, 150 tokens): drafts accepted 65% (k = 4, 3.57 tokens per pass) but a 5-token verification costs 4.1 plain passes, so speculation is 0.87x; `spec_auto` holds it at 0.96x. It will pay once block products are compute-efficient (register-tiled GEMM).

## [2026-09-19] - Janas-LLM: first complete forward pass

- `src/llm/model.c`: decode for the qwen3moe architecture (Qwen3-30B-A3B): RMS norms, per-head q/k norms, NeoX RoPE, grouped-query attention over a float32 KV cache, softmax router with normalized top-k weights, routed experts streamed through the expert cache with compute overlapping the reads.
- Decode at 21.4 token/s on Qwen3-30B-A3B (positions 200-400, 12 compute threads): AVX2 float dot/axpy for attention and router, router spread over the thread pool, RoPE angles computed once per token. Per-phase timing in `janas_llm_model_phases`.
- `tests/llm_eval.c` compares the logits with a llama.cpp dump: on 64 positions of Qwen3-30B-A3B the most likely token agrees 59 times, with a mean absolute logit difference of 0.155; llama.cpp against itself with a different kernel path agrees 60 times with 0.140.

## [2026-09-19] - Janas-LLM: model file format

- JNS, the Janas-LLM model file (`src/llm/jns.h`): each routed expert is one contiguous, 4 KiB-aligned slot (gate, up, down) read with a single O_DIRECT request; an offset table leaves the on-disk order of the experts free; checksums on header, tables and every expert.
- Expert cache (`src/llm/expert_cache.c`): fixed slots in one aligned arena, LRU replacement that never evicts the experts of the current request, warm start from prompt frequencies, misses read in parallel with O_DIRECT by an I/O thread pool; `tests/test_expert_cache.c` checks contents and hit counts against a reference model.
- The expert cache reads misses in the background (`janas_expert_cache_begin`/`finish`), so the experts already in RAM are computed while the missing ones load: +12-17% on Qwen3-Next with a 20 GiB cache, +34% with 10 GiB. No prediction involved.
- `tests/bench_stream.c`: the Phase 1 test bench. Replays a router trace against a JNS file through the expert cache, computing gate, up and down with the real weights; reports coverage, NVMe bytes, I/O wait and compute time per token.
- JNS version 2: a metadata region carries the key/value section of the source GGUF byte for byte (hyperparameters, tokenizer), validated entry by entry and covered by the fuzz test.
- `tools/gguf2jns.py`: pure-Python GGUF to JNS converter, streaming, no dependencies.
- Reader with overflow-checked validation of every offset and size; `tests/test_jns.c` fuzzes it with byte flips and field mutations; `tests/jns_check.c` validates a file and verifies every expert through O_DIRECT.

## [2026-09-19] - Janas-LLM: first kernel

- `build.sh`: release, debug, asan, ubsan and tsan variants; `./build.sh <variant> test` runs the tests.
- Q4_K and Q8_K block formats, byte-compatible with GGUF k-quants; portable scalar dot product and an AVX2 kernel selected at run time.
- Thread pool (`src/common/pool.c`) and a multithreaded Q4_K matrix-vector product with dynamic row distribution.
- `tests/test_quant.c` (kernels against the scalar reference) and `tests/bench_matvec.c` (weight bandwidth).
- Q6_K format with scalar reference and AVX2 kernel (the down projection of GGUF Q4_K_M experts is often Q6_K); dequantization matches ggml bit for bit on real OLMoE tensors. Grouped products take a quantization type per task.
- Grouped matvec: several independent products (the experts of a MoE layer) in one pool dispatch; `tests/bench_moe_layer.c` measures the routed-expert part of decode layers.
- Q4_K matvec at 63.6 GB/s of weights on 12 threads (Core Ultra 9 185H, 81 GB/s read bandwidth): SIMD unpacking of scales and mins, adaptive row chunks so each thread streams long contiguous runs.

## [2026-09-18] - Project start

- Repository created with the base layout: `src/`, `include/`, documentation files.
