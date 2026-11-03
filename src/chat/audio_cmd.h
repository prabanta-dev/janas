/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_cmd.h - /audio in janas-chat: a recording transcribed (or
 * translated into English) by a Whisper model, its text going with the next
 * message, as /image's picture does.
 */
#ifndef JANAS_CHAT_AUDIO_CMD_H
#define JANAS_CHAT_AUDIO_CMD_H

struct chat_audio;

/* speech: --speech-model, or NULL for a Whisper model beside the language
   model (model_path's directory); nothing is opened until the first
   /audio. NULL when memory ran out. */
struct chat_audio *chat_audio_new(const char *speech, const char *model_path);
void chat_audio_free(struct chat_audio *a);

/* /audio's argument: "[translate] <file>". The segments are said as they
   come (say prints a line); the text to go with the next message is
   returned, malloc'd, or NULL after saying why. */
char *chat_audio_run(struct chat_audio *a, const char *arg,
                     void (*say)(void *user, const char *line), void *user);

#endif
