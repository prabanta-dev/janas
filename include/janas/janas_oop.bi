'' SPDX-License-Identifier: GPL-3.0-or-later
'' Copyright (C) 2026 Maurizio Cammalleri
'' janas_oop.bi - Janas for FreeBASIC, with objects: a model, its
'' conversation, the options of both, a reply's counters, a conversation kept
'' by the program, a handler that receives a reply as it is written, and an
'' MCP server's tools - classes in the namespace Janas, on the libraries that
'' janas.bi declares procedure by procedure.
''
''     #include once "janas/janas_oop.bi"
''     dim as Janas.Model model = Janas.Model("qwen3-4b.jns")
''     if model.IsOpen = false then print model.ErrorText : end 1
''     dim as Janas.Chat chat = Janas.Chat(model)
''     print chat.Ask("Hello! Introduce yourself.")
''
'' The objects close what they hold when they go out of scope, and cannot be
'' copied (a copy would close it twice). A Chat lives on its Model: closing
'' the Model closes the Chat too, which then answers that it is closed.
'' Nothing raises an error: a method that can fail returns False (or an empty
'' text), and the object says why in ErrorCode (JANAS_LLM_E..., as llm.bi
'' names them) and ErrorText, until its next call. The options are the
'' library's own parameter blocks, so their fields have the C names; FreeBASIC
'' does not mind their case: options.Temperature = 0.2.
''
'' The bodies of the methods are in this file too: a program of one module
'' includes it and is done. In a program of several modules, every module but
'' one defines JANAS_DECLARE_ONLY before including it, or the linker finds
'' the methods twice.
''
'' Build:   fbc yours.bas -i <janas>/include -p <janas>/bin/x86_64-linux
'' Run:     the libraries must be found (LD_LIBRARY_PATH, or install them)

#pragma once
#include once "janas/janas.bi"

namespace Janas

enum PowerMode explicit
    Auto = JANAS_LLM_MODE_AUTO '' max on mains power, eco on battery
    Eco = JANAS_LLM_MODE_ECO   '' no GPU where it would only buy speed
    Max = JANAS_LLM_MODE_MAX   '' whatever is fastest
end enum

enum Quality explicit '' the lowest the engine goes on its own
    Low = JANAS_LLM_QUALITY_LOW       '' fastest where memory is short
    Medium = JANAS_LLM_QUALITY_MEDIUM '' experts at four bits, none left out
    High = JANAS_LLM_QUALITY_HIGH     '' nothing lowered
end enum

enum AttentionMode explicit
    Auto = JANAS_LLM_ATTN_AUTO
    Exact = JANAS_LLM_ATTN_EXACT '' the query as it is, summed in floating point
    Fast = JANAS_LLM_ATTN_FAST   '' sixteen-bit integers, an exact integer sum
end enum

'' reasoning before the reply, on the models that do it
enum ThinkMode explicit
    ModelDefault = -1
    Off = 0
    Think = 1
end enum

enum MessageRole explicit
    System = JANAS_LLM_ROLE_SYSTEM
    User = JANAS_LLM_ROLE_USER
    Assistant = JANAS_LLM_ROLE_ASSISTANT
    Tool = JANAS_LLM_ROLE_TOOL '' what a tool the model called answered
end enum

enum FinishReason explicit
    Running = 0
    Stop = JANAS_LLM_FINISH_STOP       '' the model ended it
    Length = JANAS_LLM_FINISH_LENGTH   '' max_reply tokens
    Context = JANAS_LLM_FINISH_CONTEXT '' the context is full
    Failure = JANAS_LLM_FINISH_ERROR
    Tools = JANAS_LLM_FINISH_TOOLS     '' it ended after calling tools
end enum

enum ReplyStage explicit
    Idle = JANAS_LLM_STAGE_IDLE
    Reading = JANAS_LLM_STAGE_INPUT    '' the prompt
    Writing = JANAS_LLM_STAGE_OUTPUT   '' the reply
    Done = JANAS_LLM_STAGE_DONE
end enum

enum TokenPart explicit
    Answer = JANAS_LLM_PART_ANSWER
    Reasoning = JANAS_LLM_PART_REASONING
    InCall = JANAS_LLM_PART_CALL '' inside a tool call
    Mark = JANAS_LLM_PART_MARK   '' reasoning or a call opened or closed
end enum

'' whether the model calls a tool
enum ToolMode explicit
    Auto = JANAS_LLM_TOOLS_AUTO
    None = JANAS_LLM_TOOLS_NONE
    Required = JANAS_LLM_TOOLS_REQUIRED
    Named = JANAS_LLM_TOOLS_FUNCTION '' the one named
end enum

'' the form of the answer
enum AnswerForm explicit
    Text = JANAS_LLM_FORMAT_TEXT
    Json = JANAS_LLM_FORMAT_JSON
    Schema = JANAS_LLM_FORMAT_SCHEMA '' valid against a JSON Schema
end enum

'' How a model opens: janas_llm_params, filled with the library's defaults,
'' and the files as strings. Set the fields by their C names
'' (options.n_ctx = 8192, options.Mode = Janas.PowerMode.Eco).
type OpenOptions extends janas_llm_params
    Mtp as string         '' a multi-token prediction block's file; "": none
    FindMtp as boolean    '' look for one beside the model (when Mtp is "")
    Draft as string       '' a small model of the same family to guess ahead
    Gpu as boolean = true '' false: never give the GPU work
    declare constructor()
    '' memory for the expert cache, and left to other programs, in GiB
    declare property CacheGiB(byval gib as double)
    declare property ReserveGiB(byval gib as double)
end type

'' A conversation's parameters: janas_llm_chat_params with the library's
'' defaults (options.Temperature = 0, options.max_reply = 400,
'' options.Thinking = Janas.ThinkMode.Off).
type ChatOptions extends janas_llm_chat_params
    declare constructor()
end type

'' What a reply did: janas_llm_chat_stats, its fields by their C names
'' (output_tokens, cached_tokens...), and what follows from them.
type ReplyStats extends janas_llm_chat_stats
    declare constructor()
    declare const function TokensPerSecond() as double       '' writing the reply
    declare const function PromptTokensPerSecond() as double '' reading the prompt
    declare const function Reason() as FinishReason
    declare const function FinishText() as string
end type

'' A conversation as a program keeps it, for Chat.Load: the messages in
'' order, only the first the system's, the last the user's or a tool's.
type Conversation
    declare sub Add(byval role as MessageRole, byref text as const string)
    declare sub Clear()
    declare const property Count() as long
    declare const function RoleOf(byval i as long) as MessageRole
    declare const function TextOf(byval i as long) as string
    roles(any) as long
    texts(any) as string
    n as long
end type

'' Receives a reply as it is written, from Chat.ReadReply: extend it and
'' override what you need. OnPiece returning False stops the reply there.
type ReplyHandler extends object
    declare virtual function OnPiece(byref text as const string, _
                                     byval reasoning as boolean) as boolean
    '' while a long prompt is read, after each block of it
    declare virtual sub OnProgress(byval readTokens as ulong, _
                                   byval totalTokens as ulong)
    declare virtual sub OnEnd(byref st as const ReplyStats)
    declare virtual destructor()
end type

'' A ReplyHandler that prints the answer as it comes, the reasoning left out
'' unless asked for, and nothing else.
type Printer extends ReplyHandler
    ShowReasoning as boolean
    declare function OnPiece(byref text as const string, _
                             byval reasoning as boolean) as boolean override
end type

'' The handles a Model and its Chat share: each of them lets the other know
'' when it closes its own. For the classes of this file.
type Handles
    llm as janas_llm ptr
    chat as janas_llm_chat ptr
    refs as long
end type

'' A model file (.jns) opened: the weights stay on disk, the experts are read
'' from it and the most used kept in memory.
type Model
public:
    declare constructor()
    declare constructor(byref modelFile as const string)
    declare constructor(byref modelFile as const string, _
                        byref settings as const OpenOptions)
    declare destructor()
    declare function Open(byref modelFile as const string) as boolean
    declare function Open(byref modelFile as const string, _
                          byref settings as const OpenOptions) as boolean
    '' closes the model, and its chat
    declare sub Close()
    declare const property IsOpen() as boolean
    declare const property ErrorCode() as long
    declare const property ErrorText() as string
    declare const property Handle() as janas_llm ptr '' for the procedures of llm.bi
    declare const property Path() as string
    declare const property Name() as string '' from its metadata ("Qwen3 4B")
    declare const property Description() as string
    '' the system message Janas's programs give it
    declare const property DefaultSystem() as string
    declare const property Tuning() as string
    declare const property Reasons() as boolean
    declare property Mode(byval power as PowerMode)
    declare property Gpu(byval allowed as boolean)
    '' experts a token uses (Experts = 0: the model's own count again)
    declare const property Experts() as long
    declare property Experts(byval count_ as long)
    declare const property ExpertsMost() as long
    declare const property ExpertBits() as long
    declare const property Attention() as AttentionMode
    declare const property EmbedDim() as long '' 0: the model gives no embeddings
    declare function Embed(byref text as const string, vector() as single, _
                           byval dims as long = 0) as boolean
    declare function Tokenize(byref text as const string, ids() as long, _
                              byval special as boolean = false) as boolean
    declare function TokenText(byval id as long) as string
    declare function Memory(byref resident as ulongint, _
                            byref cache as ulongint, byref kv as ulongint) _
                            as boolean
    '' the expert cache filling from this machine's profile
    declare function Preload(byref loaded as longint, _
                             byref total as longint) as boolean
    '' the NPU being made ready: True while it is
    declare function NpuPreparing(byref graphs as long, byref total as long, _
                                  byref seconds as double) as boolean
    declare const property Shared_() as Handles ptr '' for Chat
    declare static function AbiVersion() as long
    declare static function FindMtp(byref modelFile as const string) as string
    '' the memory a model would take, before opening it
    declare static function Estimate(byref modelFile as const string, _
                                     byref settings as const OpenOptions, _
                                     byref fixed as ulongint, _
                                     byref cacheAll as ulongint) as boolean
private:
    declare constructor(byref other as const Model) '' no copies
    declare operator let(byref other as const Model)
    declare function Fail(byval rc as long) as boolean
    declare function Done_() as boolean
    hs as Handles ptr
    lastCode as long
    lastText as string
    opened as string
end type

'' A model's conversation (one at a time per model). Send a message, then
'' read the reply: NextPiece piece by piece, Reply whole, or ReadReply with
'' a handler. Load gives it a whole conversation kept by the program.
type Chat
public:
    declare constructor(byref owner as Model)
    declare constructor(byref owner as Model, _
                        byref settings as const ChatOptions)
    declare destructor()
    declare sub Close()
    declare const property IsOpen() as boolean
    declare const property ErrorCode() as long
    declare const property ErrorText() as string
    declare const property Handle() as janas_llm_chat ptr
    declare const property Options() as ChatOptions
    declare function SetOptions(byref settings as const ChatOptions) as boolean
    '' one parameter at a time, from the next reply on
    declare property Temperature(byval t as single)
    declare property MaxReply(byval limit as long)
    declare property Thinking(byval think as ThinkMode)
    '' the system message, before the first message (or after Reset)
    declare property System(byref text as const string)
    '' reads the system message and the tools before the first message: 1
    '' while more is left, 0 when read, negative on failure
    declare function Prepare(byref readTokens as long, _
                             byref totalTokens as long) as long
    declare function Send(byref text as const string) as boolean
    declare function Load(byref conv as const Conversation) as boolean
    declare function Prompt(byref text as const string) as boolean
    declare function Infill(byref prefix as const string, _
                            byref suffix as const string) as boolean
    declare function SendResults(results() as string) as boolean
    '' the reply: True with the next piece (empty while a long prompt is
    '' read), False when it is over (Finished) or failed (ErrorCode)
    declare function NextPiece(byref piece as string) as boolean
    declare const property Reasoning() as boolean '' the last piece was
    declare const property Finished() as boolean
    '' the rest of the reply, its answer: the reasoning left out
    declare function Reply() as string
    declare function ReadReply(byref handler as ReplyHandler) as boolean
    declare function Ask(byref text as const string) as string
    declare const function Stats() as ReplyStats
    declare function Reset() as boolean
    '' tools: OpenAI's JSON array; whether the model calls them; the calls
    '' of the reply and what they answered (SendResults, in their order)
    declare function Tools(byref json as const string) as boolean
    declare function ToolsAdd(byref json as const string) as boolean
    declare function ToolChoice(byval choice as ToolMode, _
                                byref toolName as const string = "", _
                                byval parallel as boolean = true) as boolean
    declare function Format(byval kind as AnswerForm, _
                            byref schema as const string = "") as boolean
    declare const property Calls() as long
    declare function ToolCall(byval i as long, byref toolName as string, _
                              byref args as string) as boolean
    declare function LogitBias(ids() as long, bias() as single) as boolean
    '' the tokens the replies may have (an empty array: all of them)
    declare function Vocab(ids() as long) as boolean
    '' the tokens of the reply (options.logprobs = 1 for their
    '' log-probabilities, options.top_logprobs for the alternatives)
    declare const property Tokens() as long
    declare function Token(byval i as long, byref id as long, _
                           byref part as TokenPart, byref logprob as single, _
                           byref bytes as string) as boolean
    declare function TokenTop(byval i as long, ids() as long, _
                              logprobs() as single, _
                              byval most as long = 20) as boolean
    '' conversations kept computed besides the current one, and on disk
    declare function Keep(byval conversations as long, _
                          byval maxBytes as ulongint = 0) as boolean
    declare function Kept(byref conversations as long, _
                          byref usedBytes as ulongint) as boolean
    declare function KeepDisk(byval maxBytes as ulongint) as boolean
    declare function KeptDisk(byref files as long, _
                              byref usedBytes as ulongint) as boolean
    '' a request on the side, the conversation put back as it was
    declare function Aside(byref systemText as const string, _
                           byref text as const string, _
                           byval maxTokens as long = 256) as string
    '' a conversation message by message, with the tool calls of the
    '' assistant's messages; Submit loads it as Load does
    declare function Begin() as boolean
    declare function Add(byval role as MessageRole, _
                         byref text as const string) as boolean
    declare function AddCall(byref toolName as const string, _
                             byref args as const string) as boolean
    declare function Submit() as boolean
    declare function Count() as long '' its tokens, nothing computed
    '' the conversation as it was said: its messages, each with its reasoning
    '' and its calls (a JSON array); its title
    declare function Messages() as long
    declare function Message(byval i as long, byref role as MessageRole, _
                             byref text as string) as boolean
    declare function MessageReasoning(byval i as long) as string
    declare function MessageCalls(byval i as long) as string
    declare function Title() as string
    declare function SetTitle(byref text as const string) as boolean
    '' conversations in files: saved as they go in a folder ("": the
    '' default), or not; written, taken up, exported for people
    declare function AutoSave(byref folder as const string = "") as boolean
    declare function NoAutoSave() as boolean
    declare function SavedTo() as string
    declare function SaveTo(byref path as const string) as boolean
    declare function RestoreFrom(byref path as const string) as boolean
    declare function ExportTo(byref path as const string, _
                              byval plainText as boolean = false) as boolean
private:
    declare function MessagePart(byval i as long, byval what as long, _
                          byref role as long) as string
    declare constructor(byref other as const Chat) '' no copies
    declare operator let(byref other as const Chat)
    declare sub Make(byref owner as Model, byref settings as const ChatOptions)
    declare function Fail(byval rc as long) as boolean
    declare function Closed() as boolean
    declare function Done_() as boolean
    declare function Started(byval rc as long) as boolean
    hs as Handles ptr
    current as ChatOptions
    lastCode as long
    lastText as string
    lastReasoning as boolean
    over as boolean
end type

'' An MCP server's tools: a server started from a configuration file (by
'' default ~/.config/janas/mcp.json) or a command line, or reached at an
'' address. Its tools go to a Chat as Tools(server.Tools("prefix__")); a
'' call the model writes goes to CallTool without the prefix, and Result to
'' the Chat's SendResults. A tool runs with this program's rights: show the
'' user each call before it is made.
type McpServer
public:
    declare constructor()
    declare constructor(byref serverName as const string, _
                        byref configFile as const string = "", _
                        byval seconds as long = 0)
    declare destructor()
    declare function Open(byref serverName as const string, _
                          byref configFile as const string = "", _
                          byval seconds as long = 0) as boolean
    declare function OpenCommand(args() as string, _
                                 byref logName as const string = "", _
                                 byval seconds as long = 0) as boolean
    declare function OpenCommand(args() as string, env() as string, _
                                 byref logName as const string = "", _
                                 byval seconds as long = 0) as boolean
    declare function OpenUrl(byref url as const string, _
                             byref serverName as const string = "", _
                             byval seconds as long = 0) as boolean
    declare function OpenUrl(byref url as const string, headers() as string, _
                             byref serverName as const string = "", _
                             byval seconds as long = 0) as boolean
    declare sub Close()
    declare const property IsOpen() as boolean
    declare const property ErrorCode() as long
    declare const property ErrorText() as string
    declare const property Handle() as janas_mcp ptr '' for the procedures of mcp.bi
    declare property Info() as string
    declare property Instructions() as string
    declare const property ToolCount() as long
    declare property ToolNames() as string '' one per line
    declare function Tools(byref prefix as const string = "") as string
    '' 1 when a tool says it only reads, 0, -1 when it says nothing
    declare function ReadOnly(byref toolName as const string) as long
    declare function Refresh() as boolean
    '' True when the server answered: ResultFailed says whether the tool did
    declare function CallTool(byref toolName as const string, _
                              byref args as const string = "", _
                              byval seconds as long = 0) as boolean
    declare property Result() as string
    declare const property ResultFailed() as boolean
    declare property ResultUser() as string '' parts for the user alone
    declare property ResultJson() as string
    declare sub Cancel()
    declare static function AbiVersion() as long
    declare static function ConfigPath() as string
    declare static function ConfigServers(byref configFile as const string = "") _
                                          as string
private:
    declare constructor(byref other as const McpServer) '' no copies
    declare operator let(byref other as const McpServer)
    declare function Fail(byval rc as long) as boolean
    declare function Done_() as boolean
    declare function Text(byval what as long) as string
    srv as janas_mcp ptr
    lastCode as long
    lastText as string
end type

end namespace

#ifndef JANAS_DECLARE_ONLY

namespace Janas

'' what the library writes as text: into a buffer, its full length aside
type TextGetter as function cdecl(byval h as const any ptr, _
                                  byval buf as zstring ptr, _
                                  byval cap as long, _
                                  byval length as long ptr) as long

private function TakeText(byval getter as TextGetter, _
                          byval h as const any ptr, byref rc as long) _
                          as string
    dim as zstring * 1024 small
    dim as long n = 0
    rc = getter(h, @small, sizeof(small), @n)
    if rc = JANAS_LLM_OK andalso n < sizeof(small) then return small
    if (rc <> JANAS_LLM_OK andalso rc <> JANAS_LLM_ESMALL) orelse n <= 0 then
        return ""
    end if
    dim as zstring ptr big = callocate(n + 1)
    if big = 0 then
        rc = JANAS_LLM_ENOMEM
        return ""
    end if
    rc = getter(h, big, n + 1, @n)
    dim as string s
    if rc = JANAS_LLM_OK then s = *big
    deallocate(big)
    return s
end function

dim shared as zstring * 1 empty_

'' a string for the library: a pointer it may read len(s) bytes from
private function Z(byref s as const string) as zstring ptr
    if len(s) = 0 then return @empty_
    return cast(zstring ptr, strptr(s))
end function

'' the same, or NULL for an empty string (the library's "none")
private function ZN(byref s as const string) as zstring ptr
    if len(s) = 0 then return 0
    return cast(zstring ptr, strptr(s))
end function

'' the first element's address, or NULL for an empty array
private function First_(arr() as zstring ptr) as zstring ptr ptr
    if ubound(arr) < lbound(arr) then return 0
    return @arr(lbound(arr))
end function

private function Count_(arr() as string) as long
    if ubound(arr) < lbound(arr) then return 0
    return ubound(arr) - lbound(arr) + 1
end function

'' ---- OpenOptions, ChatOptions, ReplyStats

constructor OpenOptions()
    janas_llm_params_default(cast(janas_llm_params ptr, @this))
end constructor

property OpenOptions.CacheGiB(byval gib as double)
    cache_bytes = culngint(gib * 1073741824.0)
end property

property OpenOptions.ReserveGiB(byval gib as double)
    reserve_bytes = culngint(gib * 1073741824.0)
end property

constructor ChatOptions()
    janas_llm_chat_params_default(cast(janas_llm_chat_params ptr, @this))
end constructor

constructor ReplyStats()
    size = sizeof(janas_llm_chat_stats)
end constructor

function ReplyStats.TokensPerSecond() as double
    if output_seconds <= 0 then return 0
    return output_tokens / output_seconds
end function

function ReplyStats.PromptTokensPerSecond() as double
    if input_seconds <= 0 then return 0
    return input_tokens / input_seconds
end function

function ReplyStats.Reason() as FinishReason
    return cast(FinishReason, finish)
end function

function ReplyStats.FinishText() as string
    select case finish
    case JANAS_LLM_FINISH_STOP
        return "stop"
    case JANAS_LLM_FINISH_LENGTH
        return "length"
    case JANAS_LLM_FINISH_CONTEXT
        return "context"
    case JANAS_LLM_FINISH_ERROR
        return "error"
    case JANAS_LLM_FINISH_TOOLS
        return "tools"
    end select
    return ""
end function

'' ---- Conversation

sub Conversation.Add(byval role as MessageRole, byref text as const string)
    redim preserve roles(0 to n)
    redim preserve texts(0 to n)
    roles(n) = role
    texts(n) = text
    n += 1
end sub

sub Conversation.Clear()
    erase roles, texts
    n = 0
end sub

property Conversation.Count() as long
    return n
end property

function Conversation.RoleOf(byval i as long) as MessageRole
    if i < 0 orelse i >= n then return MessageRole.User
    return cast(MessageRole, roles(i))
end function

function Conversation.TextOf(byval i as long) as string
    if i < 0 orelse i >= n then return ""
    return texts(i)
end function

'' ---- ReplyHandler, Printer

function ReplyHandler.OnPiece(byref text as const string, _
                              byval reasoning as boolean) as boolean
    return true
end function

sub ReplyHandler.OnProgress(byval readTokens as ulong, _
                            byval totalTokens as ulong)
end sub

sub ReplyHandler.OnEnd(byref st as const ReplyStats)
end sub

destructor ReplyHandler()
end destructor

function Printer.OnPiece(byref text as const string, _
                         byval reasoning as boolean) as boolean
    if reasoning = false orelse ShowReasoning then print text;
    return true
end function

'' ---- the handles a Model and its Chat share

private function HandlesNew() as Handles ptr
    dim as Handles ptr h = callocate(sizeof(Handles))
    if h then h->refs = 1
    return h
end function

private sub HandlesDrop(byref h as Handles ptr)
    if h = 0 then exit sub
    h->refs -= 1
    if h->refs = 0 then deallocate(h)
    h = 0
end sub

'' ---- Model

constructor Model()
end constructor

constructor Model(byref modelFile as const string)
    this.Open(modelFile)
end constructor

constructor Model(byref modelFile as const string, _
                  byref settings as const OpenOptions)
    this.Open(modelFile, settings)
end constructor

destructor Model()
    this.Close()
end destructor

function Model.Fail(byval rc as long) as boolean
    lastCode = rc
    lastText = *janas_llm_last_error()
    return false
end function

function Model.Done_() as boolean
    lastCode = JANAS_LLM_OK
    lastText = ""
    return true
end function

function Model.Open(byref modelFile as const string) as boolean
    dim as OpenOptions settings
    return this.Open(modelFile, settings)
end function

function Model.Open(byref modelFile as const string, _
                    byref settings as const OpenOptions) as boolean
    this.Close()
    dim as janas_llm_params p = *cast(const janas_llm_params ptr, @settings)
    dim as zstring * 4096 found
    if len(settings.Mtp) then
        p.mtp_path = strptr(settings.Mtp)
    elseif settings.FindMtp andalso _
           janas_llm_find_mtp(Z(modelFile), @found, sizeof(found)) = _
           JANAS_LLM_OK then
        p.mtp_path = @found
    end if
    if len(settings.Draft) then p.draft_path = strptr(settings.Draft)
    dim as janas_llm ptr llm
    dim as long rc = janas_llm_open(Z(modelFile), @p, @llm)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    hs = HandlesNew()
    if hs = 0 then
        janas_llm_close(llm)
        lastCode = JANAS_LLM_ENOMEM
        lastText = "janas: out of memory"
        return false
    end if
    hs->llm = llm
    if settings.Gpu = false then janas_llm_set_gpu(llm, 0)
    opened = modelFile
    return Done_()
end function

sub Model.Close()
    if hs = 0 then exit sub
    '' the library destroys the chat with the model: the Chat learns it
    if hs->llm then janas_llm_close(hs->llm)
    hs->llm = 0
    hs->chat = 0
    HandlesDrop(hs)
    opened = ""
end sub

property Model.IsOpen() as boolean
    return hs <> 0 andalso hs->llm <> 0
end property

property Model.ErrorCode() as long
    return lastCode
end property

property Model.ErrorText() as string
    return lastText
end property

property Model.Handle() as janas_llm ptr
    if hs = 0 then return 0
    return hs->llm
end property

property Model.Path() as string
    return opened
end property

property Model.Shared_() as Handles ptr
    return hs
end property

property Model.Name() as string
    if this.IsOpen = false then return ""
    dim as long rc
    return TakeText(cast(TextGetter, @janas_llm_name), hs->llm, rc)
end property

property Model.Description() as string
    if this.IsOpen = false then return ""
    dim as long rc
    return TakeText(cast(TextGetter, @janas_llm_describe), hs->llm, rc)
end property

property Model.DefaultSystem() as string
    if this.IsOpen = false then return ""
    dim as long rc
    return TakeText(cast(TextGetter, @janas_llm_default_system), hs->llm, rc)
end property

property Model.Tuning() as string
    if this.IsOpen = false then return ""
    dim as long rc
    return TakeText(cast(TextGetter, @janas_llm_tuning), hs->llm, rc)
end property

property Model.Reasons() as boolean
    return this.IsOpen andalso janas_llm_reasons(hs->llm) <> 0
end property

property Model.Mode(byval power as PowerMode)
    if this.IsOpen then janas_llm_set_mode(hs->llm, power)
end property

property Model.Gpu(byval allowed as boolean)
    if this.IsOpen then janas_llm_set_gpu(hs->llm, iif(allowed, 1, 0))
end property

property Model.Experts() as long
    dim as long used = 0, most = 0
    if this.IsOpen then janas_llm_experts(hs->llm, @used, @most)
    return used
end property

property Model.Experts(byval count_ as long)
    if this.IsOpen then janas_llm_set_experts(hs->llm, count_)
end property

property Model.ExpertsMost() as long
    dim as long used = 0, most = 0
    if this.IsOpen then janas_llm_experts(hs->llm, @used, @most)
    return most
end property

property Model.ExpertBits() as long
    if this.IsOpen = false then return 0
    return janas_llm_expert_bits(hs->llm)
end property

property Model.Attention() as AttentionMode
    if this.IsOpen = false then return AttentionMode.Auto
    return cast(AttentionMode, janas_llm_attn_scores(hs->llm))
end property

property Model.EmbedDim() as long
    if this.IsOpen = false then return 0
    return janas_llm_embed_dim(hs->llm)
end property

function Model.Embed(byref text as const string, vector() as single, _
                     byval dims as long = 0) as boolean
    erase vector
    if this.IsOpen = false then return Fail(JANAS_LLM_EINVAL)
    dim as long full = janas_llm_embed_dim(hs->llm)
    if full <= 0 then
        lastCode = JANAS_LLM_EMODEL
        lastText = "janas: the model gives no embeddings"
        return false
    end if
    redim vector(0 to full - 1)
    dim as long n = 0, tokens = 0
    dim as long rc = janas_llm_embed(hs->llm, Z(text), len(text), dims, _
                                     @vector(0), @n, @tokens)
    if rc <> JANAS_LLM_OK then
        erase vector
        return Fail(rc)
    end if
    if n > 0 andalso n < full then redim preserve vector(0 to n - 1)
    return Done_()
end function

function Model.Tokenize(byref text as const string, ids() as long, _
                        byval special as boolean = false) as boolean
    erase ids
    if this.IsOpen = false then return Fail(JANAS_LLM_EINVAL)
    dim as long cap = len(text) + 16, n = 0
    redim ids(0 to cap - 1)
    dim as long rc = janas_llm_tokenize(hs->llm, Z(text), len(text), _
                                        iif(special, 1, 0), @ids(0), cap, @n)
    if rc = JANAS_LLM_ESMALL andalso n > cap then
        redim ids(0 to n - 1)
        rc = janas_llm_tokenize(hs->llm, Z(text), len(text), _
                                iif(special, 1, 0), @ids(0), n, @n)
    end if
    if rc <> JANAS_LLM_OK then
        erase ids
        return Fail(rc)
    end if
    if n = 0 then erase ids else redim preserve ids(0 to n - 1)
    return Done_()
end function

function Model.TokenText(byval id as long) as string
    if this.IsOpen = false then return ""
    dim as zstring * 256 buf
    dim as long n = 0
    if janas_llm_token_text(hs->llm, id, @buf, sizeof(buf), @n) <> _
       JANAS_LLM_OK then return ""
    return left(buf, n)
end function

function Model.Memory(byref resident as ulongint, byref cache as ulongint, _
                      byref kv as ulongint) as boolean
    resident = 0 : cache = 0 : kv = 0
    if this.IsOpen = false then return Fail(JANAS_LLM_EINVAL)
    dim as long rc = janas_llm_memory(hs->llm, @resident, @cache, @kv)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Model.Preload(byref loaded as longint, _
                       byref total as longint) as boolean
    loaded = 0 : total = 0
    if this.IsOpen = false then return Fail(JANAS_LLM_EINVAL)
    dim as long rc = janas_llm_preload(hs->llm, @loaded, @total)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Model.NpuPreparing(byref graphs as long, byref total as long, _
                            byref seconds as double) as boolean
    graphs = 0 : total = 0 : seconds = 0
    if this.IsOpen = false then return false
    return janas_llm_npu_preparing(hs->llm, @graphs, @total, @seconds) = 1
end function

function Model.AbiVersion() as long
    return janas_llm_abi_version()
end function

function Model.FindMtp(byref modelFile as const string) as string
    dim as zstring * 4096 found
    if janas_llm_find_mtp(Z(modelFile), @found, sizeof(found)) <> _
       JANAS_LLM_OK then return ""
    return found
end function

function Model.Estimate(byref modelFile as const string, _
                        byref settings as const OpenOptions, _
                        byref fixed as ulongint, _
                        byref cacheAll as ulongint) as boolean
    fixed = 0 : cacheAll = 0
    dim as janas_llm_params p = *cast(const janas_llm_params ptr, @settings)
    if len(settings.Mtp) then p.mtp_path = strptr(settings.Mtp)
    return janas_llm_estimate(Z(modelFile), @p, @fixed, @cacheAll) = _
           JANAS_LLM_OK
end function

'' ---- Chat

constructor Chat(byref owner as Model)
    dim as ChatOptions settings
    Make(owner, settings)
end constructor

constructor Chat(byref owner as Model, byref settings as const ChatOptions)
    Make(owner, settings)
end constructor

sub Chat.Make(byref owner as Model, byref settings as const ChatOptions)
    current = settings
    over = true
    dim as Handles ptr h = owner.Shared_
    if h = 0 orelse h->llm = 0 then
        lastCode = JANAS_LLM_EINVAL
        lastText = "janas: the model is not open"
        exit sub
    end if
    dim as janas_llm_chat ptr c
    dim as long rc = janas_llm_chat_create(h->llm, _
                        cast(const janas_llm_chat_params ptr, @current), @c)
    if rc <> JANAS_LLM_OK then
        Fail(rc)
        exit sub
    end if
    h->chat = c
    h->refs += 1
    hs = h
    Done_()
end sub

destructor Chat()
    this.Close()
end destructor

sub Chat.Close()
    if hs = 0 then exit sub
    '' a model closed first has destroyed it already
    if hs->chat then janas_llm_chat_destroy(hs->chat)
    hs->chat = 0
    HandlesDrop(hs)
end sub

function Chat.Fail(byval rc as long) as boolean
    lastCode = rc
    lastText = *janas_llm_last_error()
    return false
end function

function Chat.Done_() as boolean
    lastCode = JANAS_LLM_OK
    lastText = ""
    return true
end function

'' whether the chat is gone (never opened, closed, or its model closed)
function Chat.Closed() as boolean
    if hs <> 0 andalso hs->chat <> 0 then return false
    lastCode = JANAS_LLM_EINVAL
    lastText = "janas: the chat is closed"
    return true
end function

'' a reply asked for: it follows through NextPiece
function Chat.Started(byval rc as long) as boolean
    lastReasoning = false
    over = rc <> JANAS_LLM_OK
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

property Chat.IsOpen() as boolean
    return hs <> 0 andalso hs->chat <> 0
end property

property Chat.ErrorCode() as long
    return lastCode
end property

property Chat.ErrorText() as string
    return lastText
end property

property Chat.Handle() as janas_llm_chat ptr
    if hs = 0 then return 0
    return hs->chat
end property

property Chat.Options() as ChatOptions
    return current
end property

function Chat.SetOptions(byref settings as const ChatOptions) as boolean
    if Closed() then return false
    current = settings
    dim as long rc = janas_llm_chat_set_params(hs->chat, _
                        cast(const janas_llm_chat_params ptr, @current))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

property Chat.Temperature(byval t as single)
    dim as ChatOptions settings = current
    settings.temperature = t
    SetOptions(settings)
end property

property Chat.MaxReply(byval limit as long)
    dim as ChatOptions settings = current
    settings.max_reply = limit
    SetOptions(settings)
end property

property Chat.Thinking(byval think as ThinkMode)
    dim as ChatOptions settings = current
    settings.thinking = think
    SetOptions(settings)
end property

property Chat.System(byref text as const string)
    if Closed() then exit property
    dim as long rc = janas_llm_chat_system(hs->chat, Z(text), len(text))
    if rc <> JANAS_LLM_OK then Fail(rc) else Done_()
end property

function Chat.Prepare(byref readTokens as long, _
                      byref totalTokens as long) as long
    readTokens = 0 : totalTokens = 0
    if Closed() then return lastCode
    dim as long rc = janas_llm_chat_prepare(hs->chat, @readTokens, _
                                            @totalTokens)
    if rc < 0 then Fail(rc) else Done_()
    return rc
end function

function Chat.Send(byref text as const string) as boolean
    if Closed() then return false
    return Started(janas_llm_chat_send(hs->chat, Z(text), len(text)))
end function

function Chat.Load(byref conv as const Conversation) as boolean
    if Closed() then return false
    dim as long n = conv.n
    if n <= 0 then
        lastCode = JANAS_LLM_EINVAL
        lastText = "janas: an empty conversation"
        return false
    end if
    redim as long roles(0 to n - 1), lens(0 to n - 1)
    redim as zstring ptr texts(0 to n - 1)
    for i as long = 0 to n - 1
        roles(i) = conv.roles(i)
        texts(i) = Z(conv.texts(i))
        lens(i) = len(conv.texts(i))
    next
    return Started(janas_llm_chat_load(hs->chat, n, @roles(0), @texts(0), _
                                       @lens(0)))
end function

function Chat.Prompt(byref text as const string) as boolean
    if Closed() then return false
    return Started(janas_llm_chat_prompt(hs->chat, Z(text), len(text)))
end function

function Chat.Infill(byref prefix as const string, _
                     byref suffix as const string) as boolean
    if Closed() then return false
    return Started(janas_llm_chat_infill(hs->chat, Z(prefix), len(prefix), _
                                         Z(suffix), len(suffix)))
end function

function Chat.SendResults(results() as string) as boolean
    if Closed() then return false
    dim as long n = Count_(results())
    if n = 0 then return Started(janas_llm_chat_send_results(hs->chat, 0, _
                                                             0, 0))
    redim as zstring ptr texts(0 to n - 1)
    redim as long lens(0 to n - 1)
    dim as long first = lbound(results)
    for i as long = 0 to n - 1
        texts(i) = Z(results(first + i))
        lens(i) = len(results(first + i))
    next
    return Started(janas_llm_chat_send_results(hs->chat, n, @texts(0), _
                                               @lens(0)))
end function

function Chat.NextPiece(byref piece as string) as boolean
    piece = ""
    if over then return false
    if Closed() then
        over = true
        return false
    end if
    dim as zstring * 4097 buf
    dim as long n = 0
    dim as long rc = janas_llm_chat_next(hs->chat, @buf, sizeof(buf) - 1, @n)
    if rc = JANAS_LLM_DONE then
        over = true
        Done_()
        return false
    end if
    if rc <> JANAS_LLM_OK then
        over = true
        return Fail(rc)
    end if
    buf[n] = 0
    piece = buf
    lastReasoning = janas_llm_chat_thinking(hs->chat) <> 0
    return true
end function

property Chat.Reasoning() as boolean
    return lastReasoning
end property

property Chat.Finished() as boolean
    return over
end property

function Chat.Reply() as string
    dim as string piece, answer
    while NextPiece(piece)
        if lastReasoning = false then answer &= piece
    wend
    return answer
end function

function Chat.ReadReply(byref handler as ReplyHandler) as boolean
    dim as string piece
    while NextPiece(piece)
        if len(piece) then
            if handler.OnPiece(piece, lastReasoning) = false then return true
        else
            dim as ReplyStats st = Stats()
            if st.stage = JANAS_LLM_STAGE_INPUT then
                handler.OnProgress(st.input_done, st.input_tokens)
            end if
        end if
    wend
    if lastCode <> JANAS_LLM_OK then return false
    handler.OnEnd(Stats())
    return true
end function

function Chat.Ask(byref text as const string) as string
    if Send(text) = false then return ""
    return Reply()
end function

function Chat.Stats() as ReplyStats
    dim as ReplyStats st
    if hs <> 0 andalso hs->chat <> 0 then
        janas_llm_chat_stats(hs->chat, cast(janas_llm_chat_stats ptr, @st))
    end if
    return st
end function

function Chat.Reset() as boolean
    if Closed() then return false
    over = true
    dim as long rc = janas_llm_chat_reset(hs->chat)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.Tools(byref json as const string) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_tools(hs->chat, _
                        ZN(json), len(json))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.ToolsAdd(byref json as const string) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_tools_add(hs->chat, _
                        ZN(json), len(json))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.ToolChoice(byval choice as ToolMode, _
                         byref toolName as const string = "", _
                         byval parallel as boolean = true) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_tool_choice(hs->chat, choice, _
                        Z(toolName), len(toolName), iif(parallel, 1, 0))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.Format(byval kind as AnswerForm, _
                     byref schema as const string = "") as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_format(hs->chat, kind, Z(schema), _
                                           len(schema))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

property Chat.Calls() as long
    if hs = 0 orelse hs->chat = 0 then return 0
    return janas_llm_chat_calls(hs->chat)
end property

function Chat.ToolCall(byval i as long, byref toolName as string, _
                       byref args as string) as boolean
    toolName = "" : args = ""
    if Closed() then return false
    dim as zstring * 256 nm
    dim as long nlen = 0, alen = 0
    dim as long rc = janas_llm_chat_call(hs->chat, i, @nm, sizeof(nm), _
                                         @nlen, 0, 0, @alen)
    if rc <> JANAS_LLM_OK andalso rc <> JANAS_LLM_ESMALL then return Fail(rc)
    dim as zstring ptr a = callocate(alen + 1)
    dim as zstring ptr nb = callocate(nlen + 1)
    if a = 0 orelse nb = 0 then
        deallocate(a)
        deallocate(nb)
        lastCode = JANAS_LLM_ENOMEM
        lastText = "janas: out of memory"
        return false
    end if
    rc = janas_llm_chat_call(hs->chat, i, nb, nlen + 1, @nlen, a, alen + 1, _
                             @alen)
    if rc = JANAS_LLM_OK then
        toolName = *nb
        args = *a
    end if
    deallocate(a)
    deallocate(nb)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.LogitBias(ids() as long, bias() as single) as boolean
    if Closed() then return false
    dim as long n = ubound(ids) - lbound(ids) + 1
    if n <= 0 then
        dim as long rc = janas_llm_chat_logit_bias(hs->chat, 0, 0, 0)
        if rc <> JANAS_LLM_OK then return Fail(rc)
        return Done_()
    end if
    if ubound(bias) - lbound(bias) + 1 <> n then
        lastCode = JANAS_LLM_EINVAL
        lastText = "janas: as many biases as tokens"
        return false
    end if
    dim as long rc = janas_llm_chat_logit_bias(hs->chat, n, _
                        @ids(lbound(ids)), @bias(lbound(bias)))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.Vocab(ids() as long) as boolean
    if Closed() then return false
    dim as long n = ubound(ids) - lbound(ids) + 1
    dim as long rc
    if n <= 0 then
        rc = janas_llm_chat_vocab(hs->chat, 0, 0)
    else
        rc = janas_llm_chat_vocab(hs->chat, n, @ids(lbound(ids)))
    end if
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

property Chat.Tokens() as long
    if hs = 0 orelse hs->chat = 0 then return 0
    return janas_llm_chat_tokens(hs->chat)
end property

function Chat.Token(byval i as long, byref id as long, _
                    byref part as TokenPart, byref logprob as single, _
                    byref bytes as string) as boolean
    id = 0 : part = TokenPart.Answer : logprob = 0 : bytes = ""
    if Closed() then return false
    dim as zstring * 257 buf
    dim as long n = 0, p = 0
    dim as long rc = janas_llm_chat_token(hs->chat, i, @id, @p, @logprob, _
                                          @buf, sizeof(buf) - 1, @n)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    part = cast(TokenPart, p)
    bytes = left(buf, n)
    return Done_()
end function

function Chat.TokenTop(byval i as long, ids() as long, _
                       logprobs() as single, _
                       byval most as long = 20) as boolean
    erase ids, logprobs
    if Closed() then return false
    if most < 1 then most = 1
    redim ids(0 to most - 1)
    redim logprobs(0 to most - 1)
    dim as long n = 0
    dim as long rc = janas_llm_chat_token_top(hs->chat, i, most, @ids(0), _
                                              @logprobs(0), @n)
    if rc <> JANAS_LLM_OK then
        erase ids, logprobs
        return Fail(rc)
    end if
    if n <= 0 then
        erase ids, logprobs
    else
        redim preserve ids(0 to n - 1)
        redim preserve logprobs(0 to n - 1)
    end if
    return Done_()
end function

function Chat.Keep(byval conversations as long, _
                   byval maxBytes as ulongint = 0) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_keep(hs->chat, conversations, maxBytes)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.Kept(byref conversations as long, _
                   byref usedBytes as ulongint) as boolean
    conversations = 0 : usedBytes = 0
    if Closed() then return false
    dim as long rc = janas_llm_chat_kept(hs->chat, @conversations, @usedBytes)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.KeepDisk(byval maxBytes as ulongint) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_keep_disk(hs->chat, maxBytes)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.KeptDisk(byref files as long, _
                       byref usedBytes as ulongint) as boolean
    files = 0 : usedBytes = 0
    if Closed() then return false
    dim as long rc = janas_llm_chat_kept_disk(hs->chat, @files, @usedBytes)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.Aside(byref systemText as const string, _
                    byref text as const string, _
                    byval maxTokens as long = 256) as string
    if Closed() then return ""
    '' room for the whole answer at once: asking again would compute it again
    dim as long cap = maxTokens * 32 + 256, n = 0
    dim as zstring ptr buf = callocate(cap)
    if buf = 0 then
        lastCode = JANAS_LLM_ENOMEM
        lastText = "janas: out of memory"
        return ""
    end if
    dim as long rc = janas_llm_chat_aside(hs->chat, _
                        ZN(systemText), _
                        len(systemText), Z(text), len(text), maxTokens, _
                        buf, cap, @n)
    if rc = JANAS_LLM_ESMALL andalso n >= cap then
        deallocate(buf)
        cap = n + 1
        buf = callocate(cap)
        if buf = 0 then
            lastCode = JANAS_LLM_ENOMEM
            lastText = "janas: out of memory"
            return ""
        end if
        rc = janas_llm_chat_aside(hs->chat, _
                ZN(systemText), len(systemText), _
                Z(text), len(text), maxTokens, buf, cap, @n)
    end if
    dim as string answer
    if rc = JANAS_LLM_OK then answer = *buf
    deallocate(buf)
    if rc <> JANAS_LLM_OK then
        Fail(rc)
        return ""
    end if
    Done_()
    return answer
end function

function Chat.Begin() as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_begin(hs->chat)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.Add(byval role as MessageRole, byref text as const string) _
                  as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_add(hs->chat, role, Z(text), len(text))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.AddCall(byref toolName as const string, _
                      byref args as const string) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_add_call(hs->chat, Z(toolName), _
                        len(toolName), Z(args), len(args))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.Submit() as boolean
    if Closed() then return false
    return Started(janas_llm_chat_run(hs->chat))
end function

function Chat.Count() as long
    if Closed() then return 0
    dim as long n = 0
    dim as long rc = janas_llm_chat_count(hs->chat, @n)
    if rc <> JANAS_LLM_OK then
        Fail(rc)
        return 0
    end if
    Done_()
    return n
end function

function Chat.Messages() as long
    if Closed() then return 0
    dim as long n = janas_llm_chat_messages(hs->chat)
    if n < 0 then
        Fail(n)
        return 0
    end if
    Done_()
    return n
end function

'' a part of message i, its length asked first
function Chat.MessagePart(byval i as long, byval what as long, _
                   byref role as long) as string
    if Closed() then return ""
    dim as long n = 0
    janas_llm_chat_message(hs->chat, i, what, @role, 0, 0, @n)
    dim as string s = space(n)
    dim as long rc = janas_llm_chat_message(hs->chat, i, what, @role, _
                                            strptr(s), n + 1, @n)
    if rc <> JANAS_LLM_OK then
        Fail(rc)
        return ""
    end if
    Done_()
    return s
end function

function Chat.Message(byval i as long, byref role as MessageRole, _
                      byref text as string) as boolean
    dim as long r = 0
    text = MessagePart(i, JANAS_LLM_MESSAGE_TEXT, r)
    role = cast(MessageRole, r)
    return lastCode = JANAS_LLM_OK
end function

function Chat.MessageReasoning(byval i as long) as string
    dim as long r = 0
    return MessagePart(i, JANAS_LLM_MESSAGE_REASONING, r)
end function

function Chat.MessageCalls(byval i as long) as string
    dim as long r = 0
    return MessagePart(i, JANAS_LLM_MESSAGE_CALLS, r)
end function

function Chat.Title() as string
    if Closed() then return ""
    dim as zstring * 1024 buf
    dim as long n = 0
    if janas_llm_chat_title(hs->chat, @buf, sizeof(buf), @n) <> JANAS_LLM_OK _
        then return ""
    return left(buf, n)
end function

function Chat.SetTitle(byref text as const string) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_set_title(hs->chat, strptr(text), len(text))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.AutoSave(byref folder as const string = "") as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_autosave(hs->chat, strptr(folder))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.NoAutoSave() as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_autosave(hs->chat, 0)
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.SavedTo() as string
    if Closed() then return ""
    dim as zstring * 4096 buf
    dim as long n = 0
    if janas_llm_chat_file(hs->chat, @buf, sizeof(buf), @n) <> JANAS_LLM_OK _
        then return ""
    return left(buf, n)
end function

function Chat.SaveTo(byref path as const string) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_save(hs->chat, strptr(path))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.RestoreFrom(byref path as const string) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_restore(hs->chat, strptr(path))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

function Chat.ExportTo(byref path as const string, _
                       byval plainText as boolean = false) as boolean
    if Closed() then return false
    dim as long rc = janas_llm_chat_export(hs->chat, strptr(path), _
        iif(plainText, JANAS_LLM_EXPORT_TEXT, JANAS_LLM_EXPORT_MARKDOWN))
    if rc <> JANAS_LLM_OK then return Fail(rc)
    return Done_()
end function

'' ---- McpServer

constructor McpServer()
end constructor

constructor McpServer(byref serverName as const string, _
                      byref configFile as const string = "", _
                      byval seconds as long = 0)
    this.Open(serverName, configFile, seconds)
end constructor

destructor McpServer()
    this.Close()
end destructor

function McpServer.Fail(byval rc as long) as boolean
    lastCode = rc
    lastText = *janas_mcp_last_error()
    return false
end function

function McpServer.Done_() as boolean
    lastCode = JANAS_MCP_OK
    lastText = ""
    return true
end function

function McpServer.Open(byref serverName as const string, _
                        byref configFile as const string = "", _
                        byval seconds as long = 0) as boolean
    this.Close()
    dim as long rc = janas_mcp_open(ZN(configFile), _
                                    Z(serverName), seconds, @srv)
    if rc <> JANAS_MCP_OK then
        srv = 0
        return Fail(rc)
    end if
    return Done_()
end function

function McpServer.OpenCommand(args() as string, _
                               byref logName as const string = "", _
                               byval seconds as long = 0) as boolean
    dim as string noEnv()
    return OpenCommand(args(), noEnv(), logName, seconds)
end function

function McpServer.OpenCommand(args() as string, env() as string, _
                               byref logName as const string = "", _
                               byval seconds as long = 0) as boolean
    this.Close()
    dim as long argc = Count_(args()), nenv = Count_(env())
    if argc = 0 then
        lastCode = JANAS_MCP_EINVAL
        lastText = "janas: no command"
        return false
    end if
    redim as zstring ptr argv(0 to argc - 1)
    for i as long = 0 to argc - 1
        argv(i) = Z(args(lbound(args) + i))
    next
    dim as zstring ptr envv()
    if nenv > 0 then redim envv(0 to nenv - 1)
    for i as long = 0 to nenv - 1
        envv(i) = Z(env(lbound(env) + i))
    next
    dim as long rc = janas_mcp_open_command(ZN(logName), argc, @argv(0), _
                                            nenv, First_(envv()), seconds, _
                                            @srv)
    if rc <> JANAS_MCP_OK then
        srv = 0
        return Fail(rc)
    end if
    return Done_()
end function

function McpServer.OpenUrl(byref url as const string, _
                           byref serverName as const string = "", _
                           byval seconds as long = 0) as boolean
    dim as string noHeaders()
    return OpenUrl(url, noHeaders(), serverName, seconds)
end function

function McpServer.OpenUrl(byref url as const string, headers() as string, _
                           byref serverName as const string = "", _
                           byval seconds as long = 0) as boolean
    this.Close()
    dim as long nh = Count_(headers())
    dim as zstring ptr hv()
    if nh > 0 then redim hv(0 to nh - 1)
    for i as long = 0 to nh - 1
        hv(i) = Z(headers(lbound(headers) + i))
    next
    dim as long rc = janas_mcp_open_url(ZN(serverName), Z(url), nh, _
                                        First_(hv()), seconds, @srv)
    if rc <> JANAS_MCP_OK then
        srv = 0
        return Fail(rc)
    end if
    return Done_()
end function

sub McpServer.Close()
    if srv then janas_mcp_close(srv)
    srv = 0
end sub

property McpServer.IsOpen() as boolean
    return srv <> 0
end property

property McpServer.ErrorCode() as long
    return lastCode
end property

property McpServer.ErrorText() as string
    return lastText
end property

property McpServer.Handle() as janas_mcp ptr
    return srv
end property

'' the server's texts, one getter each
function McpServer.Text(byval what as long) as string
    if srv = 0 then return ""
    dim as long rc
    dim as TextGetter getter
    select case what
    case 0
        getter = cast(TextGetter, @janas_mcp_info)
    case 1
        getter = cast(TextGetter, @janas_mcp_instructions)
    case 2
        getter = cast(TextGetter, @janas_mcp_tool_names)
    case 3
        getter = cast(TextGetter, @janas_mcp_result)
    case 4
        getter = cast(TextGetter, @janas_mcp_result_user)
    case else
        getter = cast(TextGetter, @janas_mcp_result_json)
    end select
    dim as string s = TakeText(getter, srv, rc)
    if rc <> JANAS_MCP_OK then Fail(rc) else Done_()
    return s
end function

property McpServer.Info() as string
    return Text(0)
end property

property McpServer.Instructions() as string
    return Text(1)
end property

property McpServer.ToolCount() as long
    if srv = 0 then return 0
    return janas_mcp_tool_count(srv)
end property

property McpServer.ToolNames() as string
    return Text(2)
end property

function McpServer.Tools(byref prefix as const string = "") as string
    if srv = 0 then return ""
    dim as zstring * 4096 small
    dim as long n = 0
    dim as long rc = janas_mcp_tools(srv, ZN(prefix), _
                                     len(prefix), @small, sizeof(small), @n)
    if rc = JANAS_MCP_OK andalso n < sizeof(small) then
        Done_()
        return small
    end if
    if rc <> JANAS_MCP_ESMALL then
        Fail(rc)
        return ""
    end if
    dim as zstring ptr big = callocate(n + 1)
    if big = 0 then
        lastCode = JANAS_MCP_ENOMEM
        lastText = "janas: out of memory"
        return ""
    end if
    rc = janas_mcp_tools(srv, ZN(prefix), len(prefix), _
                         big, n + 1, @n)
    dim as string s
    if rc = JANAS_MCP_OK then s = *big
    deallocate(big)
    if rc <> JANAS_MCP_OK then Fail(rc) else Done_()
    return s
end function

function McpServer.ReadOnly(byref toolName as const string) as long
    if srv = 0 then return -1
    return janas_mcp_tool_read_only(srv, Z(toolName), len(toolName))
end function

function McpServer.Refresh() as boolean
    if srv = 0 then return false
    dim as long rc = janas_mcp_refresh(srv)
    if rc <> JANAS_MCP_OK then return Fail(rc)
    return Done_()
end function

function McpServer.CallTool(byref toolName as const string, _
                            byref args as const string = "", _
                            byval seconds as long = 0) as boolean
    if srv = 0 then
        lastCode = JANAS_MCP_EINVAL
        lastText = "janas: the server is not open"
        return false
    end if
    dim as long rc = janas_mcp_call(srv, Z(toolName), len(toolName), _
                                    ZN(args), len(args), _
                                    seconds)
    if rc <> JANAS_MCP_OK then return Fail(rc)
    return Done_()
end function

property McpServer.Result() as string
    return Text(3)
end property

property McpServer.ResultFailed() as boolean
    return srv <> 0 andalso janas_mcp_result_error(srv) = 1
end property

property McpServer.ResultUser() as string
    return Text(4)
end property

property McpServer.ResultJson() as string
    return Text(5)
end property

sub McpServer.Cancel()
    if srv then janas_mcp_cancel(srv)
end sub

function McpServer.AbiVersion() as long
    return janas_mcp_abi_version()
end function

function McpServer.ConfigPath() as string
    dim as zstring * 4096 buf
    dim as long n = 0
    if janas_mcp_config_path(@buf, sizeof(buf), @n) <> JANAS_MCP_OK then
        return ""
    end if
    return buf
end function

function McpServer.ConfigServers(byref configFile as const string = "") _
                                 as string
    dim as zstring * 8192 buf
    dim as long n = 0
    if janas_mcp_config_servers(ZN(configFile), _
                                @buf, sizeof(buf), @n) <> JANAS_MCP_OK then
        return ""
    end if
    return buf
end function

end namespace

#endif
