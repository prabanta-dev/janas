'' SPDX-License-Identifier: GPL-3.0-or-later
'' Copyright (C) 2026 Maurizio Cammalleri
'' chat.bas - the smallest chat on Janas-LLM, in FreeBASIC.
''
'' What a BASIC MODERN or FreeBASIC program needs to do: open a model, make a
'' conversation, send a message, read the reply as it is written.
''
''   fbc tools/chat.bas -i include -p bin/x86_64-linux -x janas-chat-fb
''   LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-fb model.jns
''   echo "Ciao" | LD_LIBRARY_PATH=bin/x86_64-linux ./janas-chat-fb model.jns
''
#include once "janas/llm.bi"

'' FreeBASIC reads the console, not the standard input: its LINE INPUT waits for
'' a terminal and never sees a pipe. So the terminal keeps LINE INPUT, with the
'' line editing it brings, and a pipe or a file is read through the CONS device.
declare function isatty cdecl alias "isatty" (byval fd as long) as long

dim shared as long tty_input, stdin_file
tty_input = isatty(0)

'' Asks for one line and puts it in msg. Returns 0 at the end of the input.
function ask(byref msg as string) as long
    if tty_input <> 0 then
        line input "> ", msg
        return 1
    end if
    print "> ";
    if eof(stdin_file) then
        print
        return 0
    end if
    line input #stdin_file, msg
    print msg           '' a pipe has no echo of its own
    return 1
end function

if tty_input = 0 then
    stdin_file = freefile
    if open cons(for input as #stdin_file) <> 0 then
        print "cannot read the standard input"
        end 1
    end if
end if

dim as string path = command(1)
if path = "" then
    print "usage: chat <model.jns>"
    end 2
end if
if janas_llm_abi_version() <> JANAS_LLM_ABI then
    print "the library speaks another ABI"
    end 1
end if

dim as janas_llm_params mp
janas_llm_params_default(@mp)

dim as janas_llm ptr llm
if janas_llm_open(strptr(path), @mp, @llm) <> JANAS_LLM_OK then
    print *janas_llm_last_error()
    end 1
end if

dim as janas_llm_chat_params cp
janas_llm_chat_params_default(@cp)
cp.max_reply = 400   '' a model left to itself can fill the whole context
dim as janas_llm_chat ptr chat
if janas_llm_chat_create(llm, @cp, @chat) <> JANAS_LLM_OK then
    print *janas_llm_last_error()
    end 1
end if

dim as zstring * 512 desc
dim as long n
janas_llm_describe(llm, @desc, sizeof(desc), @n)
print desc

do
    dim as string msg
    if ask(msg) = 0 then exit do
    if msg = "" orelse msg = "/quit" then exit do
    if janas_llm_chat_send(chat, strptr(msg), -1) <> JANAS_LLM_OK then
        print *janas_llm_last_error()
        continue do
    end if
    dim as zstring * 256 piece
    do
        '' The piece is not closed by a NUL: one byte is left for it.
        dim as long r = janas_llm_chat_next(chat, @piece, sizeof(piece) - 1, @n)
        if r = JANAS_LLM_DONE then exit do
        if r <> JANAS_LLM_OK then
            print !"\n"; *janas_llm_last_error()
            exit do
        end if
        piece[n] = 0
        print piece;
    loop
    print

    dim as janas_llm_chat_stats st
    st.size = sizeof(st)
    if janas_llm_chat_stats(chat, @st) = JANAS_LLM_OK then
        print using "[#### token in ###.## s, ##.# token/s, context #####/#####]"; _
            st.output_tokens; st.output_seconds; _
            st.output_tokens / st.output_seconds; _
            st.context_used; st.context_size
    end if
loop

janas_llm_chat_destroy(chat)
janas_llm_close(llm)
if tty_input = 0 then close #stdin_file
