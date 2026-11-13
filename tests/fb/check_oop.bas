'' SPDX-License-Identifier: GPL-3.0-or-later
'' Copyright (C) 2026 Maurizio Cammalleri
'' check_oop.bas - janas_oop.bi on a model, run by tests/fb_check.c, which
'' compares the replies with the library's own in C:
''
''   check_oop model.jns "first message" "second message"
''
'' Greedy replies, read piece by piece, whole, through a handler, after a
'' reset and in a new chat; a conversation loaded, its start reused; one chat
'' at a time; the tokens of a text and back; a request on the side; the
'' model closed under a live chat, which then says it is closed and closes
'' nothing twice. Each reply written as "key hex", the rest as "FAIL ..."
'' lines; the exit status the count of failures.

#include once "janas/janas_oop.bi"

dim shared as long failures

sub check(byval ok as boolean, byref what as const string)
    if ok = false then
        failures += 1
        print "FAIL "; what
    end if
end sub

function hexOf(byref s as const string) as string
    dim as string h
    for i as long = 0 to len(s) - 1
        h &= lcase(hex(s[i], 2))
    next
    return h
end function

sub say(byref key as const string, byref value as const string)
    print key; " "; value
end sub

type Collect extends Janas.ReplyHandler
    answer as string
    pieces as long
    ended as boolean
    declare function OnPiece(byref text as const string, _
                             byval reasoning as boolean) as boolean override
    declare sub OnEnd(byref st as const Janas.ReplyStats) override
end type

function Collect.OnPiece(byref text as const string, _
                         byval reasoning as boolean) as boolean
    pieces += 1
    if reasoning = false then answer &= text
    return true
end function

sub Collect.OnEnd(byref st as const Janas.ReplyStats)
    ended = st.output_tokens > 0
end sub

dim as string modelFile = command(1), p1 = command(2), p2 = command(3)
dim as Janas.OpenOptions oo
oo.n_ctx = 4096
dim as Janas.Model model = Janas.Model(modelFile, oo)
if model.IsOpen = false then
    print "FAIL open: "; model.ErrorText
    end 1
end if
say("model", model.Name)
dim as Janas.ChatOptions greedy
greedy.Temperature = 0
greedy.Thinking = Janas.ThinkMode.Off
greedy.max_reply = 48

scope
    dim as Janas.Chat chat = Janas.Chat(model, greedy)
    check(chat.IsOpen, "the chat: " & chat.ErrorText)
    scope
        dim as Janas.Chat other = Janas.Chat(model)
        check(other.IsOpen = false andalso other.ErrorCode = JANAS_LLM_EBUSY, _
              "one chat at a time: " & other.ErrorText)
    end scope

    '' piece by piece
    check(chat.Send(p1), "send: " & chat.ErrorText)
    dim as string piece, t1
    while chat.NextPiece(piece)
        if chat.Reasoning = false then t1 &= piece
    wend
    check(chat.ErrorCode = 0 andalso chat.Finished, "the reply's end: " & _
          chat.ErrorText)
    say("t1", hexOf(t1))
    dim as Janas.ReplyStats st = chat.Stats()
    check(st.output_tokens > 0 andalso st.TokensPerSecond() > 0, _
          "the reply's counters")

    '' the whole conversation, its start already computed
    dim as Janas.Conversation conv
    conv.Add(Janas.MessageRole.User, p1)
    conv.Add(Janas.MessageRole.Assistant, t1)
    conv.Add(Janas.MessageRole.User, p2)
    check(chat.Load(conv), "load: " & chat.ErrorText)
    say("t2", hexOf(chat.Reply()))
    st = chat.Stats()
    say("cached", st.cached_tokens & " of " & st.prompt_tokens)
    check(st.cached_tokens > 0, "nothing of the prompt reused")

    '' whole, after a reset
    check(chat.Reset(), "reset: " & chat.ErrorText)
    say("t1again", hexOf(chat.Ask(p1)))

    '' through a handler
    check(chat.Reset(), "reset: " & chat.ErrorText)
    dim as Collect handler
    check(chat.Send(p1) andalso chat.ReadReply(handler), "the handler: " & _
          chat.ErrorText)
    say("t1handler", hexOf(handler.answer))
    check(handler.ended andalso handler.pieces > 0, "the handler's end")

    '' the conversation written down, saved, taken up and exported
    dim as Janas.MessageRole role
    dim as string said, tmp = environ("TMPDIR")
    if tmp = "" then tmp = "/tmp"
    dim as long nm = chat.Messages()
    check(nm >= 2 andalso chat.Message(nm - 1, role, said) andalso _
          role = Janas.MessageRole.Assistant andalso len(said) > 0, _
          "the messages written down: " & nm & " " & chat.ErrorText)
    check(chat.SetTitle("FB check") andalso chat.Title() = "FB check", _
          "the title: " & chat.Title())
    dim as string saved = tmp & "/janas-fb-check.jsonl"
    check(chat.SaveTo(saved) andalso chat.RestoreFrom(saved) andalso _
          chat.Messages() = nm andalso chat.Title() = "FB check", _
          "saved and taken up: " & chat.ErrorText)
    check(chat.ExportTo(tmp & "/janas-fb-check.md"), "exported: " & _
          chat.ErrorText)
    kill saved
    kill tmp & "/janas-fb-check.md"

    '' tokens, and a request on the side
    dim as long ids()
    check(model.Tokenize("Hello, world!", ids()), "tokenize: " & _
          model.ErrorText)
    dim as string back
    for i as long = lbound(ids) to ubound(ids)
        back &= model.TokenText(ids(i))
    next
    check(back = "Hello, world!", "the tokens' text: " & back)
    dim as string side = chat.Aside("", "Say yes.", 8)
    check(len(side) > 0, "a request on the side: " & chat.ErrorText)

    '' the model closed under a live chat
    model.Close()
    check(chat.IsOpen = false, "the chat closed with its model")
    check(chat.Send("x") = false andalso chat.ErrorText = _
          "janas: the chat is closed", "and it says so: " & chat.ErrorText)
end scope '' the chat's own close finds nothing to close

check(model.Open(modelFile, oo), "open again: " & model.ErrorText)
scope
    dim as Janas.Chat chat2 = Janas.Chat(model, greedy)
    say("t1chat2", hexOf(chat2.Ask(p1)))
end scope
model.Close()
say("done", str(failures))
end failures
