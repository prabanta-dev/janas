'' SPDX-License-Identifier: GPL-3.0-or-later
'' Copyright (C) 2026 Maurizio Cammalleri
'' llm.bi - Janas-LLM for FreeBASIC, and so for BASIC MODERN, the dialect
'' Prabanta implements.
''
'' The same library the C header include/janas/llm.h describes: handles are
'' opaque, nothing is passed by value that is not a number, every parameter
'' block starts with its own size, and text comes back on request as whole
'' UTF-8 characters. Keep this file and llm.h in step: a parameter block
'' carries its own size and the library refuses one it does not know.
''
'' Build:   fbc yours.bas -l janas_llm
'' Run:     the library must be found (LD_LIBRARY_PATH, or install it)

#pragma once
#inclib "janas_llm"

extern "c"

'' The C header calls this JANAS_LLM_ABI_VERSION; here it drops the last
'' word, because FreeBASIC does not tell names apart by their case and the
'' function janas_llm_abi_version() would be the same identifier.
const JANAS_LLM_ABI = 1

'' Results: 0 is success, DONE says a reply is over, the rest are failures.
const JANAS_LLM_OK      =  0
const JANAS_LLM_DONE    =  1
const JANAS_LLM_EINVAL  = -1   '' invalid argument
const JANAS_LLM_ENOMEM  = -2   '' out of memory
const JANAS_LLM_EOPEN   = -3   '' model file missing, unreadable, invalid
const JANAS_LLM_EMODEL  = -4   '' model not supported
const JANAS_LLM_EFULL   = -5   '' the context is full
const JANAS_LLM_EBUSY   = -6   '' the model already has a chat
const JANAS_LLM_ESMALL  = -7   '' buffer too small
const JANAS_LLM_EFAIL   = -8   '' internal failure
const JANAS_LLM_ELIMIT  = -9   '' the prompt is longer than max_input

'' Power modes: auto follows the power source, eco never uses the GPU for
'' speed alone, max takes the fastest configuration.
const JANAS_LLM_MODE_AUTO = 0
const JANAS_LLM_MODE_ECO  = 1
const JANAS_LLM_MODE_MAX  = 2

'' how the attention scores are computed: the query as it is, or to sixteen
'' bits with an exact integer sum (about twice as fast, an error of 3e-5)
const JANAS_LLM_ATTN_AUTO  = 0
const JANAS_LLM_ATTN_EXACT = 1
const JANAS_LLM_ATTN_FAST  = 2

type janas_llm as any       '' an open model
type janas_llm_chat as any  '' a conversation

type janas_llm_params
    size        as ulong     '' sizeof(janas_llm_params)
    n_ctx       as ulong     '' context in tokens (0: 16384)
    cache_bytes as ulongint  '' RAM for streamed experts (0: automatic)
    mtp_path    as const zstring ptr '' multi-token prediction block, or NULL
    draft_path  as const zstring ptr '' a small model of the same family to
                                      '' guess the next tokens, or NULL
    mode        as long      '' JANAS_LLM_MODE_*
    no_preload  as long      '' 1: do not fill the expert cache at open
    expert_bits as long      '' 2, 4 or 6; 0: the engine chooses
    attn_scores as long      '' JANAS_LLM_ATTN_*; 0: the engine chooses
    reserve_bytes as ulongint '' memory left to other programs (0: a fifth)
end type

type janas_llm_chat_params
    size        as ulong
    temperature as single    '' 0: always the most likely token
    top_k       as long
    top_p       as single
    min_p       as single
    seed        as ulongint  '' 0: a new one each chat
    speculate   as long      '' 1: draft ahead when it pays
    max_reply   as long      '' tokens per reply; 0: no limit
    thinking    as long      '' 1 on, 0 off, -1 the model's own default
    recap       as long      '' 1: sum up the oldest turns before they go
    presence_penalty  as single '' OpenAI's penalties (default 0)
    frequency_penalty as single
    logprobs     as long     '' 1: keep every token's log-probability
    top_logprobs as long     '' and that many alternatives, 0..20
    max_input    as ulong    '' the longest prompt, in tokens; above it
                             '' JANAS_LLM_ELIMIT, never a cut (0: none)
    max_answer   as long     '' the longest answer, reasoning not counted
                             '' (0: none); max_reply counts it too
end type

type janas_llm_chat_stats
    size           as ulong
    input_tokens   as ulong
    output_tokens  as ulong
    context_used   as ulong
    context_size   as ulong
    drafted        as ulong
    accepted       as ulong
    input_seconds  as double
    output_seconds as double
    total_seconds  as double
    experts_used   as ulongint
    experts_read   as ulongint
    bytes_read     as ulongint
    io_wait_seconds as double
    '' passes run without drafting although drafts were asked for: the
    '' engine judged they were not paying, and the reply was slower for it
    auto_off_passes as ulong
    finish          as long  '' why the reply ended: JANAS_LLM_FINISH_*
    prompt_tokens   as ulong '' the whole prompt of the reply
    cached_tokens   as ulong '' of which already computed
    '' what the reply is doing (JANAS_LLM_STAGE_*), and of input_tokens how
    '' many are read: the progress of a long prompt, between the empty
    '' pieces janas_llm_chat_next returns while it reads it
    stage           as long
    input_done      as ulong
    first_token_seconds as double '' from the send to the first token
    prepare_seconds     as double '' building the prompt before reading it
    input_cpu_seconds   as double '' processor time of the reading
    input_threads   as long  '' compute threads of the reading and of the
    output_threads  as long  '' writing, as the engine chose them
    context_bytes   as ulongint '' keys and values of context_used tokens
    peak_rss_bytes  as ulongint '' the most memory the process held
    '' the prompt by source: system (tools included, prompt_tools of it),
    '' the conversation before the last message, the last message
    prompt_system   as ulong
    prompt_tools    as ulong
    prompt_history  as ulong
    prompt_last     as ulong
    '' seconds left to read the prompt (stage INPUT), an estimate; -1:
    '' not yet known
    input_eta_seconds as double
end type

const JANAS_LLM_STAGE_IDLE = 0
const JANAS_LLM_STAGE_INPUT = 1
const JANAS_LLM_STAGE_OUTPUT = 2
const JANAS_LLM_STAGE_DONE = 3

const JANAS_LLM_FINISH_STOP = 1    '' the model ended it
const JANAS_LLM_FINISH_LENGTH = 2  '' max_reply tokens
const JANAS_LLM_FINISH_CONTEXT = 3 '' the context is full
const JANAS_LLM_FINISH_ERROR = 4
const JANAS_LLM_FINISH_TOOLS = 5   '' the model ended it after calling tools

'' tool_choice, the form of the answer, and the parts of a reply
const JANAS_LLM_TOOLS_AUTO = 0
const JANAS_LLM_TOOLS_NONE = 1
const JANAS_LLM_TOOLS_REQUIRED = 2
const JANAS_LLM_TOOLS_FUNCTION = 3
const JANAS_LLM_FORMAT_TEXT = 0
const JANAS_LLM_FORMAT_JSON = 1
const JANAS_LLM_FORMAT_SCHEMA = 2
const JANAS_LLM_PART_ANSWER = 0
const JANAS_LLM_PART_REASONING = 1
const JANAS_LLM_PART_CALL = 2
const JANAS_LLM_PART_MARK = 3

'' Roles of the messages of janas_llm_chat_load.
const JANAS_LLM_ROLE_SYSTEM = 0
const JANAS_LLM_ROLE_USER = 1
const JANAS_LLM_ROLE_ASSISTANT = 2
const JANAS_LLM_ROLE_TOOL = 3

'' The library's own ABI version, to check against the constant above, and
'' the last failure on this thread as text (never NULL).
declare function janas_llm_abi_version() as long
declare function janas_llm_last_error() as const zstring ptr

declare sub janas_llm_params_default(byval p as janas_llm_params ptr)
declare function janas_llm_open(byval path as const zstring ptr, _
                                byval p as const janas_llm_params ptr, _
                                byval out as janas_llm ptr ptr) as long
declare sub janas_llm_close(byval llm as janas_llm ptr)

'' Text out of the library: at most cap bytes, NUL-terminated, with *len the
'' full length (JANAS_LLM_ESMALL when it did not fit).
'' Whether the GPU may be given work at all (0 or 1).
declare function janas_llm_set_gpu(byval llm as janas_llm ptr, _
                                   byval on_ as long) as long

declare function janas_llm_name(byval llm as const janas_llm ptr, _
                                byval buf as zstring ptr, byval cap as long, _
                                byval length as long ptr) as long
declare function janas_llm_default_system(byval llm as const janas_llm ptr, _
                                          byval buf as zstring ptr, _
                                          byval cap as long, _
                                          byval length as long ptr) as long
declare function janas_llm_describe(byval llm as const janas_llm ptr, _
                                    byval buf as zstring ptr, _
                                    byval cap as long, _
                                    byval length as long ptr) as long
declare function janas_llm_tuning(byval llm as const janas_llm ptr, _
                                  byval buf as zstring ptr, byval cap as long, _
                                  byval length as long ptr) as long
declare function janas_llm_set_mode(byval llm as janas_llm ptr, _
                                    byval mode as long) as long

'' How many experts a token uses, of the count the model was trained with,
'' and how many bits its down matrix is read with (6, 4 or 2).
declare function janas_llm_experts(byval llm as const janas_llm ptr, _
                                   byval used as long ptr, _
                                   byval most as long ptr) as long
declare function janas_llm_set_experts(byval llm as janas_llm ptr, _
                                       byval n as long) as long
declare function janas_llm_expert_bits(byval llm as const janas_llm ptr) as long
declare function janas_llm_attn_scores(byval llm as const janas_llm ptr) as long
declare function janas_llm_preload(byval llm as const janas_llm ptr, _
                                   byval done as longint ptr, _
                                   byval total as longint ptr) as long

'' Embeddings (models such as Qwen3-Embedding): the vector's length (0: the
'' model gives none), and the vector of a text, of unit length.
declare function janas_llm_embed_dim(byval llm as const janas_llm ptr) as long
declare function janas_llm_embed(byval llm as janas_llm ptr, _
                                 byval text as const zstring ptr, _
                                 byval length as long, byval dim as long, _
                                 byval out as single ptr, byval n as long ptr, _
                                 byval tokens as long ptr) as long

'' Tokens: the ids of a text, and the bytes of one token.
declare function janas_llm_tokenize(byval llm as const janas_llm ptr, _
                                    byval text as const zstring ptr, _
                                    byval length as long, _
                                    byval special as long, _
                                    byval ids as long ptr, byval max as long, _
                                    byval n as long ptr) as long
declare function janas_llm_token_text(byval llm as const janas_llm ptr, _
                                      byval id as long, _
                                      byval buf as zstring ptr, _
                                      byval cap as long, _
                                      byval length as long ptr) as long

declare sub janas_llm_chat_params_default(byval p as janas_llm_chat_params ptr)
declare function janas_llm_chat_create(byval llm as janas_llm ptr, _
                                       byval p as const janas_llm_chat_params ptr, _
                                       byval out as janas_llm_chat ptr ptr) as long
declare sub janas_llm_chat_destroy(byval c as janas_llm_chat ptr)
declare function janas_llm_chat_set_params(byval c as janas_llm_chat ptr, _
                                           byval p as const janas_llm_chat_params ptr) as long
declare function janas_llm_chat_system(byval c as janas_llm_chat ptr, _
                                       byval text as const zstring ptr, _
                                       byval length as long) as long
declare function janas_llm_chat_send(byval c as janas_llm_chat ptr, _
                                     byval text as const zstring ptr, _
                                     byval length as long) as long
'' What the tools answered to the calls of the last reply, in their order:
'' n texts (lens may be NULL: then they are NUL-terminated); the model's
'' reply to them follows as after janas_llm_chat_send.
declare function janas_llm_chat_send_results(byval c as janas_llm_chat ptr, _
                                             byval n as long, _
                                             byval texts as const zstring ptr ptr, _
                                             byval lens as const long ptr) as long
'' The reply as it is written: whole UTF-8 characters, JANAS_LLM_DONE at the
'' end. Call it until DONE, or leave it and send again to give up the reply.
declare function janas_llm_chat_next(byval c as janas_llm_chat ptr, _
                                     byval buf as zstring ptr, _
                                     byval cap as long, _
                                     byval length as long ptr) as long
declare function janas_llm_chat_thinking(byval c as const janas_llm_chat ptr) as long
'' A whole conversation in place of the current one (the last message the
'' user's or a tool's); what it shares with the computed one is not read
'' again. lens may be NULL: then the texts are NUL-terminated.
declare function janas_llm_chat_load(byval c as janas_llm_chat ptr, _
                                     byval n as long, _
                                     byval roles as const long ptr, _
                                     byval texts as const zstring ptr ptr, _
                                     byval lens as const long ptr) as long
'' Text to fill in between prefix and suffix (models with FIM tokens).
declare function janas_llm_chat_infill(byval c as janas_llm_chat ptr, _
                                       byval prefix as const zstring ptr, _
                                       byval length as long, _
                                       byval suffix as const zstring ptr, _
                                       byval suffix_len as long) as long
'' Raw text to continue, with no chat format around it.
declare function janas_llm_chat_prompt(byval c as janas_llm_chat ptr, _
                                       byval text as const zstring ptr, _
                                       byval length as long) as long

'' The same, message by message, with the tool calls of an assistant's
'' message (name, and the arguments as a JSON object).
declare function janas_llm_chat_begin(byval c as janas_llm_chat ptr) as long
declare function janas_llm_chat_add(byval c as janas_llm_chat ptr, _
                                    byval role as long, _
                                    byval text as const zstring ptr, _
                                    byval length as long) as long
declare function janas_llm_chat_add_call(byval c as janas_llm_chat ptr, _
                                         byval name_ as const zstring ptr, _
                                         byval name_len as long, _
                                         byval args as const zstring ptr, _
                                         byval args_len as long) as long
declare function janas_llm_chat_run(byval c as janas_llm_chat ptr) as long
'' How many tokens the conversation built would be (nothing computed).
declare function janas_llm_chat_count(byval c as janas_llm_chat ptr, _
                                      byval n as long ptr) as long

'' Conversations kept computed besides the current one: at most n, in at
'' most bytes (0: automatic, see llm.h); n of 0 keeps none.
declare function janas_llm_chat_keep(byval c as janas_llm_chat ptr, _
                                     byval n as long, _
                                     byval bytes as ulongint) as long
declare function janas_llm_chat_kept(byval c as const janas_llm_chat ptr, _
                                     byval n as long ptr, _
                                     byval bytes as ulongint ptr) as long
'' And on disk, below the memory, so that they outlive the program: at most
'' bytes (0: off, the default).
declare function janas_llm_chat_keep_disk(byval c as janas_llm_chat ptr, _
                                          byval bytes as ulongint) as long
declare function janas_llm_chat_kept_disk(byval c as const janas_llm_chat ptr, _
                                          byval n as long ptr, _
                                          byval bytes as ulongint ptr) as long

'' Tools as OpenAI's JSON array (NULL: none), whether the model calls them,
'' and the form of the answer (text, a JSON object, or a JSON Schema).
declare function janas_llm_chat_tools(byval c as janas_llm_chat ptr, _
                                      byval json as const zstring ptr, _
                                      byval length as long) as long
declare function janas_llm_chat_tool_choice(byval c as janas_llm_chat ptr, _
                                            byval choice as long, _
                                            byval name_ as const zstring ptr, _
                                            byval length as long, _
                                            byval parallel as long) as long
declare function janas_llm_chat_format(byval c as janas_llm_chat ptr, _
                                       byval kind as long, _
                                       byval schema as const zstring ptr, _
                                       byval length as long) as long
'' The tool calls of the reply, and call i: name and arguments.
declare function janas_llm_chat_calls(byval c as const janas_llm_chat ptr) as long
declare function janas_llm_chat_call(byval c as const janas_llm_chat ptr, _
                                     byval i as long, _
                                     byval name_ as zstring ptr, _
                                     byval name_cap as long, _
                                     byval name_len as long ptr, _
                                     byval args as zstring ptr, _
                                     byval args_cap as long, _
                                     byval args_len as long ptr) as long
declare function janas_llm_chat_logit_bias(byval c as janas_llm_chat ptr, _
                                           byval n as long, _
                                           byval ids as const long ptr, _
                                           byval bias as const single ptr) as long
'' The tokens of the reply: id, part (JANAS_LLM_PART_*), log-probability,
'' bytes; and the most likely alternatives in each one's place.
declare function janas_llm_chat_tokens(byval c as const janas_llm_chat ptr) as long
declare function janas_llm_chat_token(byval c as const janas_llm_chat ptr, _
                                      byval i as long, byval id as long ptr, _
                                      byval part as long ptr, _
                                      byval logprob as single ptr, _
                                      byval buf as zstring ptr, _
                                      byval cap as long, _
                                      byval length as long ptr) as long
declare function janas_llm_chat_token_top(byval c as const janas_llm_chat ptr, _
                                          byval i as long, byval max as long, _
                                          byval ids as long ptr, _
                                          byval logprobs as single ptr, _
                                          byval n as long ptr) as long

declare function janas_llm_reasons(byval llm as const janas_llm ptr) as long
declare function janas_llm_chat_reset(byval c as janas_llm_chat ptr) as long
declare function janas_llm_chat_stats(byval c as const janas_llm_chat ptr, _
                                      byval s as janas_llm_chat_stats ptr) as long

end extern

'' The parameter blocks carry their own size, and every call checks it, so a
'' header out of step with the library is refused instead of misread. As
'' built on 25 Sep 2026 the sizes are 56, 72 and 176 bytes; a program can see
'' for itself with sizeof() and with janas_llm_abi_version().
