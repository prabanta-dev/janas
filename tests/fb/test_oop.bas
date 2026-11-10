'' SPDX-License-Identifier: GPL-3.0-or-later
'' Copyright (C) 2026 Maurizio Cammalleri
'' test_oop.bas - janas_oop.bi without a model, run by tests/test_fb.c:
'' the parameter blocks as the C header lays them out, the options with the
'' library's defaults, a conversation kept by the program, a model and a
'' chat that do not open and say why, a handler's methods overridden, and
'' an MCP server - the fake one of the tests, given as the first argument -
'' started, its tools listed and called, and closed.
''
''   fbc tests/fb/test_oop.bas -i include -p bin/x86_64-linux -x test_oop
''   LD_LIBRARY_PATH=bin/x86_64-linux ./test_oop bin/x86_64-linux/mcp_fake_server

#include once "janas/janas_oop.bi"

dim shared as long checks, failures

sub check(byval ok as boolean, byref what as const string)
    checks += 1
    if ok = false then
        failures += 1
        print "FAIL "; what
    end if
end sub

'' the parameter blocks as llm.h lays them out on x86-64
#ifdef __FB_64BIT__
#assert sizeof(janas_llm_params) = 64
#assert sizeof(janas_llm_chat_params) = 80
#assert sizeof(janas_llm_chat_stats) = 232
#assert offsetof(janas_llm_params, reserve_bytes) = 48
#assert offsetof(janas_llm_params, min_quality) = 60
#assert offsetof(janas_llm_chat_params, max_reasoning) = 72
#assert offsetof(janas_llm_chat_stats, reasoning_cut) = 188
#assert offsetof(janas_llm_chat_stats, route) = 192
#assert offsetof(janas_llm_chat_stats, route_domain) = 196
#assert sizeof(janas_llm_monitor) = 208
#assert offsetof(janas_llm_monitor, replies) = 8
#assert offsetof(janas_llm_monitor, stage) = 32
#assert offsetof(janas_llm_monitor, memory_available) = 200
#assert sizeof(janas_llm_speech_params) = 24
#assert sizeof(janas_llm_transcribe_params) = 112
#assert offsetof(janas_llm_transcribe_params, speaker_names) = 88
#assert offsetof(janas_llm_transcribe_params, chunking) = 64
#assert offsetof(janas_llm_transcribe_params, seed) = 24
#assert offsetof(janas_llm_transcribe_params, temperature) = 32
#assert offsetof(janas_llm_transcribe_params, words) = 36
#assert offsetof(janas_llm_transcribe_params, on_segment) = 40
#assert offsetof(janas_llm_transcribe_params, stop) = 56
#endif

check(Janas.Model.AbiVersion() = JANAS_LLM_ABI, "the library's ABI")
check(Janas.McpServer.AbiVersion() = JANAS_MCP_ABI, "the MCP library's ABI")

scope '' the options, with the library's defaults
    dim as Janas.ChatOptions co
    check(co.size = sizeof(janas_llm_chat_params), "ChatOptions.size")
    check(abs(co.Temperature - 0.7) < 1e-6, "the default temperature")
    check(co.top_k = 20 andalso co.speculate = 1 andalso co.recap = 1, _
          "the default top_k, speculate, recap")
    check(co.Thinking = Janas.ThinkMode.ModelDefault, "the default thinking")
    co.Temperature = 0
    co.max_reply = 48
    check(co.temperature = 0 andalso co.MAX_REPLY = 48, "fields by any case")
    dim as Janas.OpenOptions oo
    check(oo.size = sizeof(janas_llm_params), "OpenOptions.size")
    check(oo.n_ctx = 16384, "the default context")
    check(oo.Gpu = true andalso oo.FindMtp = false andalso oo.Mtp = "", _
          "the default files and GPU")
    oo.CacheGiB = 2
    check(oo.cache_bytes = 2147483648ull, "CacheGiB")
    check(oo.min_quality = Janas.Quality.Low, "the default quality")
    oo.min_quality = Janas.Quality.High
    check(oo.min_quality = JANAS_LLM_QUALITY_HIGH, "Quality.High")
    dim as Janas.ReplyStats st
    check(st.size = sizeof(janas_llm_chat_stats), "ReplyStats.size")
    check(st.TokensPerSecond() = 0 andalso st.FinishText() = "" andalso _
          st.Reason() = Janas.FinishReason.Running, "an empty ReplyStats")
    st.finish = JANAS_LLM_FINISH_LENGTH
    st.output_tokens = 30
    st.output_seconds = 1.5
    check(st.FinishText() = "length" andalso _
          abs(st.TokensPerSecond() - 20) < 1e-9, "ReplyStats' methods")
end scope

'' read-only methods on const objects (a handler's OnEnd gets one)
sub constCalls(byref st as const Janas.ReplyStats, _
               byref conv as const Janas.Conversation, _
               byref m as const Janas.Model, byref c as const Janas.Chat, _
               byref srv as const Janas.McpServer)
    check(st.TokensPerSecond() = 0 andalso st.PromptTokensPerSecond() = 0 _
          andalso st.FinishText() = "", "a const ReplyStats")
    check(conv.Count = 1 andalso conv.TextOf(0) = "x" andalso _
          conv.RoleOf(0) = Janas.MessageRole.User, "a const Conversation")
    check(m.IsOpen = false andalso m.Name = "" andalso m.EmbedDim = 0 _
          andalso m.Experts = 0, "a const Model")
    check(c.IsOpen = false andalso c.Tokens = 0 andalso c.Calls = 0, _
          "a const Chat")
    check(srv.IsOpen = false andalso srv.ToolCount = 0, "a const McpServer")
end sub

scope
    dim as Janas.ReplyStats st
    dim as Janas.Conversation conv
    conv.Add(Janas.MessageRole.User, "x")
    dim as Janas.Model m
    dim as Janas.Chat c = Janas.Chat(m)
    dim as Janas.McpServer srv
    constCalls(st, conv, m, c, srv)
end scope

scope '' a conversation kept by the program
    dim as Janas.Conversation conv
    conv.Add(Janas.MessageRole.System, "Be brief.")
    conv.Add(Janas.MessageRole.User, "Ciao, città!")
    conv.Add(Janas.MessageRole.Assistant, "")
    check(conv.Count = 3, "three messages")
    check(conv.RoleOf(1) = Janas.MessageRole.User andalso _
          conv.TextOf(1) = "Ciao, città!" andalso conv.TextOf(2) = "", _
          "a message's role and text")
    check(conv.TextOf(7) = "", "a message that is not there")
    conv.Clear()
    check(conv.Count = 0, "cleared")
end scope

scope '' a model that does not open, and its chat
    dim as Janas.Model model = Janas.Model("/nonexistent/janas-test.jns")
    check(model.IsOpen = false, "a missing file does not open")
    check(model.ErrorCode = JANAS_LLM_EOPEN andalso len(model.ErrorText) > 0, _
          "and says why: " & model.ErrorText)
    check(model.Name = "" andalso model.Handle = 0, "nothing of it")
    dim as Janas.Chat chat = Janas.Chat(model)
    check(chat.IsOpen = false, "no chat without a model")
    check(chat.ErrorText = "janas: the model is not open", _
          "and it says so: " & chat.ErrorText)
    check(chat.Send("Hello") = false andalso chat.Ask("Hello") = "", _
          "a closed chat sends nothing")
    dim as string piece
    check(chat.NextPiece(piece) = false andalso chat.Finished, _
          "and reads nothing")
    check(Janas.Model.FindMtp("/nonexistent/janas-test.jns") = "", _
          "no MTP beside a missing file")
    dim as Janas.OpenOptions oo
    dim as ulongint fixed, cacheAll
    check(Janas.Model.Estimate("/nonexistent/janas-test.jns", oo, fixed, _
          cacheAll) = false, "no estimate of a missing file")
end scope

'' a handler: the methods overridden are the ones called
type Collect extends Janas.ReplyHandler
    pieces as long
    answer as string
    declare function OnPiece(byref text as const string, _
                             byval reasoning as boolean) as boolean override
    declare sub OnEnd(byref st as const Janas.ReplyStats) override
    ended as boolean
end type

function Collect.OnPiece(byref text as const string, _
                         byval reasoning as boolean) as boolean
    pieces += 1
    if reasoning = false then answer &= text
    return pieces < 3
end function

sub Collect.OnEnd(byref st as const Janas.ReplyStats)
    ended = true
end sub

scope
    dim as Collect c
    dim as Janas.ReplyHandler ptr h = @c
    check(h->OnPiece("a", false) andalso h->OnPiece("b", true) andalso _
          h->OnPiece("c", false) = false, "OnPiece overridden, and its stop")
    check(c.answer = "ac", "the reasoning left out")
    dim as Janas.ReplyStats st
    h->OnEnd(st)
    h->OnProgress(1, 2) '' the default: nothing
    check(c.ended, "OnEnd overridden")
end scope

'' an MCP server: the fake one, started from its command line
dim as string fake = command(1)
if len(fake) then
    dim as string argv(0 to 1) = {fake, "modern"}
    dim as Janas.McpServer server
    check(server.OpenCommand(argv()), "the fake server starts: " & _
          server.ErrorText)
    check(server.IsOpen andalso server.ToolCount >= 3, "it has tools")
    check(instr(server.ToolNames, "echo") > 0, "echo among them")
    dim as string tools = server.Tools("fake__")
    check(instr(tools, """fake__add""") > 0, "the tools as JSON, prefixed")
    check(server.CallTool("echo", "{""text"": ""ciao""}", 10), _
          "echo called: " & server.ErrorText)
    check(instr(server.Result, "ciao") > 0 andalso server.ResultFailed = false, _
          "echo answered: " & server.Result)
    check(server.CallTool("add", "{""a"": 2, ""b"": 3}", 10) andalso _
          instr(server.Result, "5") > 0, "add answered: " & server.Result)
    check(server.CallTool("fail", "", 10) andalso server.ResultFailed, _
          "a tool that fails says so")
    check(server.CallTool("no-such-tool", "", 10) = false andalso _
          server.ErrorCode <> 0, "an unknown tool")
    server.Close()
    check(server.IsOpen = false andalso server.CallTool("echo") = false, _
          "closed")
end if
dim as Janas.McpServer none = Janas.McpServer("no-such-server", _
                                              "/nonexistent/mcp.json")
check(none.IsOpen = false andalso none.ErrorCode = JANAS_MCP_EOPEN, _
      "no such server: " & none.ErrorText)
check(Janas.McpServer.ConfigServers("/nonexistent/mcp.json") = "", _
      "no servers in a missing file")

if failures then
    print "test_oop:"; failures; " of"; checks; " checks failed"
    end 1
end if
print "test_oop:"; checks; " checks, ok"
