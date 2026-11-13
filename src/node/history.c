/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * history.c - a Chat's conversation as it was said, and conversations in
 * files (issue #69): chat.messages(), title(), setTitle(), autosave(),
 * file(), save(), restore(), export() on the engine's thread, in order
 * with the messages; and the module's listChats() and chatsDir(), which
 * read files only and answer at once.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/json.h"
#include "node/node_impl.h"

enum op {
    OP_MESSAGES,
    OP_TITLE,
    OP_SET_TITLE,
    OP_AUTOSAVE,
    OP_FILE,
    OP_SAVE,
    OP_RESTORE,
    OP_EXPORT
};

struct hist_job {
    enum op op;
    char *arg; /* the text, path or folder; NULL: none (autosave off) */
    int32_t format;
    struct janas_buf out; /* a string, or JSON (messages) */
};

static void hist_drop(void *data)
{
    struct hist_job *h = data;
    if (!h)
        return;
    free(h->arg);
    janas_buf_free(&h->out);
    free(h);
}

/* A string of the library into out: cap 0 asks its length first. */
static int32_t take(struct janas_buf *out, janas_llm_chat *c, int32_t i,
                    int32_t what, int32_t *role, int title)
{
    int32_t len = 0;
    if (title)
        janas_llm_chat_title(c, NULL, 0, &len);
    else
        janas_llm_chat_message(c, i, what, role, NULL, 0, &len);
    char *s = malloc((size_t)len + 1);
    if (!s)
        return JANAS_LLM_ENOMEM;
    int32_t rc =
        title ? janas_llm_chat_title(c, s, len + 1, &len)
              : janas_llm_chat_message(c, i, what, role, s, len + 1, &len);
    if (rc == JANAS_LLM_OK)
        janas_buf_put(out, s, (size_t)len);
    free(s);
    return rc;
}

static int32_t messages(struct janas_buf *b, janas_llm_chat *c)
{
    static const char *const ROLES[] = {"system", "user", "assistant", "tool"};
    int32_t n = janas_llm_chat_messages(c);
    if (n < 0)
        return n;
    janas_buf_puts(b, "[");
    for (int32_t i = 0; i < n; i++) {
        struct janas_buf t = {0};
        int32_t role = 0, rc = take(&t, c, i, JANAS_LLM_MESSAGE_TEXT, &role, 0);
        if (rc == JANAS_LLM_OK) {
            janas_buf_printf(b, "%s{\"role\":\"%s\",\"text\":", i ? "," : "",
                             ROLES[role & 3]);
            janas_json_write_str(b, t.p ? t.p : "", t.n);
            t.n = 0;
            rc = take(&t, c, i, JANAS_LLM_MESSAGE_REASONING, NULL, 0);
        }
        if (rc == JANAS_LLM_OK && t.n) {
            janas_buf_puts(b, ",\"reasoning\":");
            janas_json_write_str(b, t.p, t.n);
        }
        t.n = 0;
        if (rc == JANAS_LLM_OK)
            rc = take(&t, c, i, JANAS_LLM_MESSAGE_CALLS, NULL, 0);
        if (rc == JANAS_LLM_OK && t.n > 2) { /* not "[]" */
            janas_buf_puts(b, ",\"calls\":");
            janas_buf_put(b, t.p, t.n);
        }
        janas_buf_free(&t);
        if (rc != JANAS_LLM_OK)
            return rc;
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]");
    return b->oom ? JANAS_LLM_ENOMEM : JANAS_LLM_OK;
}

static void hist_run(struct janas_node_job *j)
{
    struct hist_job *h = j->data;
    janas_llm_chat *c = j->m->chat;
    if (!c) {
        j->rc = JANAS_NODE_ECLOSED;
        snprintf(j->error, sizeof(j->error), "janas: the chat is closed");
        return;
    }
    int32_t len = 0;
    switch (h->op) {
    case OP_MESSAGES:
        j->rc = messages(&h->out, c);
        break;
    case OP_TITLE:
        j->rc = take(&h->out, c, 0, 0, NULL, 1);
        break;
    case OP_SET_TITLE:
        j->rc = janas_llm_chat_set_title(c, h->arg ? h->arg : "", -1);
        break;
    case OP_AUTOSAVE:
        j->rc = janas_llm_chat_autosave(c, h->arg);
        break;
    case OP_FILE: {
        char p[4096];
        j->rc = janas_llm_chat_file(c, p, sizeof p, &len);
        if (j->rc == JANAS_LLM_OK)
            janas_buf_put(&h->out, p, (size_t)len);
        break;
    }
    case OP_SAVE:
        j->rc = janas_llm_chat_save(c, h->arg);
        break;
    case OP_RESTORE:
        j->rc = janas_llm_chat_restore(c, h->arg);
        break;
    case OP_EXPORT:
        j->rc = janas_llm_chat_export(c, h->arg, h->format);
        break;
    }
    if (j->rc < 0 && j->rc != JANAS_NODE_ECLOSED)
        snprintf(j->error, sizeof(j->error), "%s", janas_llm_last_error());
}

/* JSON text as a JavaScript value, by JSON.parse. */
static napi_value parse(napi_env env, const char *s, size_t n)
{
    napi_value global, json, fn, arg, out = NULL;
    if (napi_get_global(env, &global) != napi_ok ||
        napi_get_named_property(env, global, "JSON", &json) != napi_ok ||
        napi_get_named_property(env, json, "parse", &fn) != napi_ok ||
        !(arg = janas_node_string(env, s, n)) ||
        napi_call_function(env, json, fn, 1, &arg, &out) != napi_ok)
        return NULL;
    return out;
}

static napi_value hist_value(napi_env env, struct janas_node_job *j)
{
    struct hist_job *h = j->data;
    if (h->op == OP_MESSAGES)
        return parse(env, h->out.p ? h->out.p : "[]", h->out.p ? h->out.n : 2);
    if (h->op == OP_TITLE || h->op == OP_FILE)
        return janas_node_string(env, h->out.p ? h->out.p : "", h->out.n);
    return janas_node_undefined(env);
}

/* A method of Chat: its argument (a string, or for autosave null too) and,
   for export, the format. */
static napi_value post(napi_env env, napi_callback_info info, enum op op,
                       const char *what)
{
    size_t argc = 2;
    napi_value argv[2] = {NULL, NULL};
    struct janas_node_chat *c =
        janas_node_self(env, info, &janas_node_chat_tag, &argc, argv, what);
    if (!c)
        return NULL;
    if (c->closed || c->m->closing) {
        char msg[160];
        snprintf(msg, sizeof(msg), "janas: %s: the chat is closed", what);
        return janas_node_rejected(
            env, janas_node_error(env, JANAS_NODE_ECLOSED, msg));
    }
    struct hist_job *h = calloc(1, sizeof(*h));
    if (!h)
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    h->op = op;
    int wants = op != OP_MESSAGES && op != OP_TITLE && op != OP_FILE;
    if (wants) {
        napi_valuetype t = napi_undefined;
        if (argc)
            napi_typeof(env, argv[0], &t);
        size_t n = 0;
        if (op == OP_AUTOSAVE && (t == napi_null || t == napi_undefined)) {
            h->arg = NULL; /* no longer */
        } else if (op == OP_SET_TITLE && t == napi_undefined) {
            h->arg = NULL; /* from the first question again */
        } else if (!argc || !(h->arg = janas_node_text(env, argv[0], &n))) {
            int oom = n == SIZE_MAX;
            hist_drop(h);
            return oom ? janas_node_throw(env, JANAS_LLM_ENOMEM,
                                          "janas: out of memory")
                       : janas_node_type_error(env, "janas: %s: a string",
                                               what);
        }
        if (op == OP_AUTOSAVE && t == napi_undefined)
            h->arg = strdup(""); /* the default folder */
    }
    if (op == OP_EXPORT) {
        size_t fn;
        char *f = argc > 1 ? janas_node_text(env, argv[1], &fn) : NULL;
        h->format = f && !strcmp(f, "text") ? JANAS_LLM_EXPORT_TEXT
                                            : JANAS_LLM_EXPORT_MARKDOWN;
        int bad =
            argc > 1 && (!f || (strcmp(f, "text") && strcmp(f, "markdown")));
        free(f);
        if (bad) {
            hist_drop(h);
            return janas_node_type_error(
                env, "janas: chat.export(path, format): format \"markdown\" "
                     "or \"text\"");
        }
    }
    struct janas_node_job *j = janas_node_job(c->m, hist_run);
    napi_value p = j ? janas_node_promise(env, &j->deferred) : NULL;
    if (!p) {
        hist_drop(h);
        free(j);
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    }
    j->data = h;
    j->drop = hist_drop;
    j->value = hist_value;
    janas_node_post(env, j);
    return p;
}

napi_value janas_node_chat_messages(napi_env env, napi_callback_info info)
{
    return post(env, info, OP_MESSAGES, "Chat.messages");
}

napi_value janas_node_chat_title(napi_env env, napi_callback_info info)
{
    return post(env, info, OP_TITLE, "Chat.title");
}

napi_value janas_node_chat_set_title(napi_env env, napi_callback_info info)
{
    return post(env, info, OP_SET_TITLE, "Chat.setTitle");
}

napi_value janas_node_chat_autosave(napi_env env, napi_callback_info info)
{
    return post(env, info, OP_AUTOSAVE, "Chat.autosave");
}

napi_value janas_node_chat_file(napi_env env, napi_callback_info info)
{
    return post(env, info, OP_FILE, "Chat.file");
}

napi_value janas_node_chat_save(napi_env env, napi_callback_info info)
{
    return post(env, info, OP_SAVE, "Chat.save");
}

napi_value janas_node_chat_restore(napi_env env, napi_callback_info info)
{
    return post(env, info, OP_RESTORE, "Chat.restore");
}

napi_value janas_node_chat_export(napi_env env, napi_callback_info info)
{
    return post(env, info, OP_EXPORT, "Chat.export");
}

/* The module's: the default folder, and a folder's conversations. */

napi_value janas_node_chats_dir(napi_env env, napi_callback_info info)
{
    (void)info;
    char d[4096];
    int32_t len = 0;
    if (janas_llm_chats_dir(d, sizeof d, &len) != JANAS_LLM_OK)
        return janas_node_throw(env, JANAS_LLM_EFAIL, janas_llm_last_error());
    return janas_node_string(env, d, (size_t)len);
}

napi_value janas_node_list_chats(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1] = {NULL};
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    char *dir = NULL;
    size_t dn;
    napi_valuetype t = napi_undefined;
    if (argc)
        napi_typeof(env, argv[0], &t);
    if (t != napi_undefined && t != napi_null &&
        !(dir = janas_node_text(env, argv[0], &dn)))
        return janas_node_type_error(env, "janas: listChats(dir): a string");
    int32_t len = 0;
    janas_llm_chats_list(dir, NULL, 0, &len);
    char *list = malloc((size_t)len + 1);
    if (!list ||
        janas_llm_chats_list(dir, list, len + 1, &len) != JANAS_LLM_OK) {
        free(list);
        free(dir);
        return janas_node_throw(env, JANAS_LLM_EFAIL, janas_llm_last_error());
    }
    free(dir);
    struct janas_buf b = {0};
    janas_buf_puts(&b, "[");
    int k = 0;
    for (char *p = list, *e; *p && (e = strchr(p, '\n')); p = e + 1) {
        *e = 0;
        char title[1024], model[256], file[4096];
        int64_t updated = 0;
        int32_t nm = 0;
        if (janas_llm_chat_file_info(p, title, sizeof title, model,
                                     sizeof model, file, sizeof file, &updated,
                                     &nm) != JANAS_LLM_OK)
            continue;
        janas_buf_puts(&b, k++ ? ",{\"path\":" : "{\"path\":");
        janas_json_write_str(&b, p, strlen(p));
        janas_buf_puts(&b, ",\"title\":");
        janas_json_write_str(&b, title, strlen(title));
        janas_buf_puts(&b, ",\"model\":");
        janas_json_write_str(&b, model, strlen(model));
        janas_buf_puts(&b, ",\"modelFile\":");
        janas_json_write_str(&b, file, strlen(file));
        janas_buf_printf(&b, ",\"updated\":%lld,\"messages\":%d}",
                         (long long)updated * 1000, (int)nm);
    }
    janas_buf_puts(&b, "]");
    free(list);
    napi_value v = b.oom ? NULL : parse(env, b.p, b.n);
    janas_buf_free(&b);
    return v ? v
             : janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
}
