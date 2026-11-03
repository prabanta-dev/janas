/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - janas-files's tools (see files.h): those that read (files_list,
 * files_read, files_search) say so; those that change something
 * (files_write, files_edit, files_mkdir, in write.c) say they do not only
 * read, so that a client asks the user every time (janas-chat does,
 * --mcp-auto or not). The answers are text for the model.
 */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "files.h"

#define READS "\"annotations\": {\"readOnlyHint\": true}"
#define WRITES                                                                 \
    "\"annotations\": {\"readOnlyHint\": false, \"destructiveHint\": false}"
#define PATH_ARG(what)                                                         \
    "\"path\": {\"type\": \"string\", \"description\": \"" what                \
    ": ~/..., absolute, or relative to the folder janas-chat runs in\"}"

void fl_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b,
        "\"tools\": [{\"name\": \"files_list\", \"title\": \"A folder\", "
        "\"description\": \"What a folder holds: its folders and files "
        "with their sizes, down to depth levels.\", \"inputSchema\": "
        "{\"type\": \"object\", \"properties\": {" PATH_ARG(
            "the folder; leave it out for the one janas-chat runs in") ", "
                                                                       "\"depth\": {\"type\": \"number\", \"description\": \"1 (default) "
                                                                       "to 3\"}, \"hidden\": {\"type\": \"boolean\", \"description\": "
                                                                       "\"also the names starting with a dot\"}}, "
                                                                       "\"additionalProperties\": false}, " READS
                                                                       "}, "
                                                                       "{\"name\": \"files_read\", \"title\": \"Read a file\", "
                                                                       "\"description\": \"A text file's lines: 400 from the start unless "
                                                                       "asked; the answer says how many there are.\", \"inputSchema\": "
                                                                       "{\"type\": \"object\", \"properties\": {" PATH_ARG("the file") ", "
                                                                                                                                       "\"offset\": {\"type\": \"number\", \"description\": \"the first "
                                                                                                                                       "line, from 1\"}, \"lines\": {\"type\": \"number\", "
                                                                                                                                       "\"description\": \"how many, at most 2000\"}}, \"required\": "
                                                                                                                                       "[\"path\"], \"additionalProperties\": false}, " READS
                                                                                                                                       "}, "
                                                                                                                                       "{\"name\": \"files_search\", \"title\": \"Find files\", "
                                                                                                                                       "\"description\": \"Files under a folder by their name (a pattern "
                                                                                                                                       "such as *.class) and/or the lines holding a text (any case).\", "
                                                                                                                                       "\"inputSchema\": {\"type\": \"object\", \"properties\": "
                                                                                                                                       "{" PATH_ARG(
                                                                                                                                           "the folder; leave it out for the one janas-chat runs "
                                                                                                                                           "in") ", "
                                                                                                                                                 "\"name\": {\"type\": \"string\"}, \"text\": {\"type\": "
                                                                                                                                                 "\"string\"}, \"hidden\": {\"type\": \"boolean\", \"description\": \"false leaves out the names starting with a dot\"}}, "
                                                                                                                                                 "\"additionalProperties\": false}, " READS
                                                                                                                                                 "}, "
                                                                                                                                                 "{\"name\": \"files_write\", \"title\": \"Write a file\", "
                                                                                                                                                 "\"description\": \"Writes a text file. One that is there is "
                                                                                                                                                 "replaced only with overwrite true, and a copy of it is kept. The "
                                                                                                                                                 "user is asked first.\", \"inputSchema\": {\"type\": \"object\", "
                                                                                                                                                 "\"properties\": {" PATH_ARG("the file") ", \"content\": "
                                                                                                                                                                                          "{\"type\": \"string\"}, \"overwrite\": {\"type\": \"boolean\"}, "
                                                                                                                                                                                          "\"make_dirs\": {\"type\": \"boolean\", \"description\": \"false: "
                                                                                                                                                                                          "its folder must be there (made by default)\"}}, \"required\": "
                                                                                                                                                                                          "[\"path\", \"content\"], \"additionalProperties\": false}, " WRITES
                                                                                                                                                                                          "}, "
                                                                                                                                                                                          "{\"name\": \"files_edit\", \"title\": \"Edit a file\", "
                                                                                                                                                                                          "\"description\": \"Replaces a piece of a text file, old_text as "
                                                                                                                                                                                          "it is there (it must be there once, or all true), with new_text; "
                                                                                                                                                                                          "a copy of the file is kept. The user is asked first.\", "
                                                                                                                                                                                          "\"inputSchema\": {\"type\": \"object\", \"properties\": "
                                                                                                                                                                                          "{" PATH_ARG(
                                                                                                                                                                                              "the file") ", \"old_text\": {\"type\": \"string\"}, "
                                                                                                                                                                                                          "\"new_text\": {\"type\": \"string\"}, \"all\": {\"type\": "
                                                                                                                                                                                                          "\"boolean\"}}, \"required\": [\"path\", \"old_text\", "
                                                                                                                                                                                                          "\"new_text\"], \"additionalProperties\": false}, " WRITES
                                                                                                                                                                                                          "}, "
                                                                                                                                                                                                          "{\"name\": \"files_mkdir\", \"title\": \"Make a folder\", "
                                                                                                                                                                                                          "\"description\": \"Makes a folder, and those above it that are "
                                                                                                                                                                                                          "not there. The user is asked first.\", \"inputSchema\": "
                                                                                                                                                                                                          "{\"type\": \"object\", \"properties\": {" PATH_ARG("the folder") "}, "
                                                                                                                                                                                                                                                                            "\"required\": [\"path\"], \"additionalProperties\": false}, " WRITES
                                                                                                                                                                                                                                                                            "}]");
}

const char *fl_arg_str(const struct janas_json *args, const char *name)
{
    const char *s = janas_json_str(janas_json_get(args, name));
    return s && *s ? s : NULL;
}

int fl_arg_bool(const struct janas_json *args, const char *name)
{
    const struct janas_json *v = janas_json_get(args, name);
    return v && v->type == JANAS_JSON_TRUE;
}

double fl_arg_num(const struct janas_json *args, const char *name, double def)
{
    const struct janas_json *v = janas_json_get(args, name);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, def) : def;
}

int fl_ok(struct janas_buf *b, struct janas_buf *text)
{
    if (text->oom) {
        janas_buf_free(text);
        return fl_fail(b, "out of memory");
    }
    janas_mcps_text_result(b, text->p ? text->p : "", text->n, 0);
    janas_buf_free(text);
    return 0;
}

int fl_fail(struct janas_buf *b, const char *fmt, ...)
{
    char line[4096 + PATH_MAX];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    janas_mcps_text_result(b, line, strlen(line), 1);
    return 0;
}

int fl_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    struct fl_roots *r = ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    /* the list of folders is read again at each call: a line added to it
       counts at once */
    fl_roots_load(r);
    if (janas_json_is(name, "files_list"))
        return fl_tool_list(r, args, b);
    if (janas_json_is(name, "files_read"))
        return fl_tool_read(r, args, b);
    if (janas_json_is(name, "files_search"))
        return fl_tool_search(r, args, b);
    if (janas_json_is(name, "files_write"))
        return fl_tool_write(r, args, b);
    if (janas_json_is(name, "files_edit"))
        return fl_tool_edit(r, args, b);
    if (janas_json_is(name, "files_mkdir"))
        return fl_tool_mkdir(r, args, b);
    return -1;
}
