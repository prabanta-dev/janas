/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * addon.c - the module for Node.js, janas.node: what Node calls when it
 * loads it, the environment of each Node thread that does, and the small
 * conversions between JavaScript's values and the library's.
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "node/node_impl.h"

const napi_type_tag janas_node_model_tag = {0x6a616e61736d6f64ull,
                                            0x9c1f3b2a5e7d4c81ull};
const napi_type_tag janas_node_chat_tag = {0x6a616e6173636874ull,
                                           0x2b8e4f7a1c3d5e90ull};
const napi_type_tag janas_node_reply_tag = {0x6a616e6173726570ull,
                                            0x7d3a9e1b4c6f2a58ull};
const napi_type_tag janas_node_speech_tag = {0x6a616e6173737063ull,
                                             0x4e2c8a6f1b9d3e75ull};

struct janas_node_env *janas_node_env(napi_env env)
{
    void *st = NULL;
    if (napi_get_instance_data(env, &st) != napi_ok)
        return NULL;
    return st;
}

void janas_node_env_unref(struct janas_node_env *st)
{
    if (--st->refs == 0)
        free(st);
}

/* Node is leaving the environment: the models close first, so that each
   writes what it learnt, and their threads end */
static void env_cleanup(void *arg)
{
    janas_node_teardown(arg);
}

static void env_free(napi_env env, void *data, void *hint)
{
    (void)env;
    (void)hint;
    janas_node_env_unref(data);
}

napi_value janas_node_undefined(napi_env env)
{
    napi_value v = NULL;
    napi_get_undefined(env, &v);
    return v;
}

napi_value janas_node_string(napi_env env, const char *s, size_t len)
{
    napi_value v = NULL;
    if (napi_create_string_utf8(env, s ? s : "", s ? len : 0, &v) != napi_ok)
        return janas_node_undefined(env);
    return v;
}

napi_value janas_node_number(napi_env env, double x)
{
    napi_value v = NULL;
    if (napi_create_double(env, x, &v) != napi_ok)
        return janas_node_undefined(env);
    return v;
}

napi_value janas_node_bool(napi_env env, int b)
{
    napi_value v = NULL;
    napi_get_boolean(env, b != 0, &v);
    return v;
}

void janas_node_put(napi_env env, napi_value obj, const char *name,
                    napi_value v)
{
    napi_set_named_property(env, obj, name, v);
}

/* the codes of the library's failures, as Node names a system's (ENOENT):
   the suffix of JANAS_LLM_E..., in a namespace of their own - not JANAS_,
   the environment's */
const char *janas_node_code(int32_t rc)
{
    switch (rc) {
    case JANAS_LLM_EINVAL:
        return "EINVAL";
    case JANAS_LLM_ENOMEM:
        return "ENOMEM";
    case JANAS_LLM_EOPEN:
        return "EOPEN";
    case JANAS_LLM_EMODEL:
        return "EMODEL";
    case JANAS_LLM_EFULL:
        return "EFULL";
    case JANAS_LLM_EBUSY:
        return "EBUSY";
    case JANAS_LLM_ESMALL:
        return "ESMALL";
    case JANAS_LLM_ELIMIT:
        return "ELIMIT";
    case JANAS_NODE_ECLOSED:
        return "ECLOSED";
    default:
        return "EFAIL";
    }
}

napi_value janas_node_error(napi_env env, int32_t rc, const char *msg)
{
    napi_value e = NULL;
    if (!msg || !*msg)
        msg = "janas: the engine failed, and did not say why";
    napi_create_error(
        env,
        janas_node_string(env, janas_node_code(rc), JANAS_NAPI_AUTO_LENGTH),
        janas_node_string(env, msg, JANAS_NAPI_AUTO_LENGTH), &e);
    return e;
}

napi_value janas_node_throw(napi_env env, int32_t rc, const char *msg)
{
    napi_throw_error(env, janas_node_code(rc), msg);
    return NULL;
}

napi_value janas_node_type_error(napi_env env, const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    napi_throw_type_error(env, NULL, buf);
    return NULL;
}

napi_value janas_node_promise(napi_env env, napi_deferred *d)
{
    napi_value p = NULL;
    if (napi_create_promise(env, d, &p) != napi_ok) {
        *d = NULL;
        return NULL;
    }
    return p;
}

napi_value janas_node_resolved(napi_env env, napi_value v)
{
    napi_deferred d;
    napi_value p = janas_node_promise(env, &d);
    if (p)
        napi_resolve_deferred(env, d, v ? v : janas_node_undefined(env));
    return p;
}

napi_value janas_node_rejected(napi_env env, napi_value err)
{
    napi_deferred d;
    napi_value p = janas_node_promise(env, &d);
    if (p)
        napi_reject_deferred(env, d, err ? err : janas_node_undefined(env));
    return p;
}

napi_value janas_node_new(napi_env env, napi_ref cls, const napi_type_tag *tag,
                          void *native, napi_finalize fin)
{
    struct janas_node_env *st = janas_node_env(env);
    napi_value ctor, obj = NULL;
    if (!st || napi_get_reference_value(env, cls, &ctor) != napi_ok)
        return NULL;
    st->constructing = 1;
    napi_status s = napi_new_instance(env, ctor, 0, NULL, &obj);
    st->constructing = 0;
    if (s != napi_ok || napi_type_tag_object(env, obj, tag) != napi_ok ||
        napi_wrap(env, obj, native, fin, NULL, NULL) != napi_ok)
        return NULL;
    return obj;
}

napi_value janas_node_constructor(napi_env env, napi_callback_info info)
{
    struct janas_node_env *st = janas_node_env(env);
    napi_value self = NULL;
    void *where = NULL;
    napi_get_cb_info(env, info, NULL, NULL, &self, &where);
    if (!st || !st->constructing)
        return janas_node_type_error(env, "janas: %s",
                                     where ? (const char *)where
                                           : "not made with new");
    return self;
}

void *janas_node_self(napi_env env, napi_callback_info info,
                      const napi_type_tag *tag, size_t *argc, napi_value *argv,
                      const char *what)
{
    napi_value self = NULL;
    void *native = NULL;
    size_t none = 0;
    bool ours = false;
    if (napi_get_cb_info(env, info, argc ? argc : &none, argv, &self, NULL) !=
            napi_ok ||
        napi_check_object_type_tag(env, self, tag, &ours) != napi_ok || !ours ||
        napi_unwrap(env, self, &native) != napi_ok || !native) {
        janas_node_type_error(env, "janas: %s called on another object", what);
        return NULL;
    }
    return native;
}

char *janas_node_text(napi_env env, napi_value v, size_t *len)
{
    napi_valuetype t;
    size_t n = 0;
    *len = 0;
    if (!v || napi_typeof(env, v, &t) != napi_ok || t != napi_string ||
        napi_get_value_string_utf8(env, v, NULL, 0, &n) != napi_ok)
        return NULL;
    char *s = malloc(n + 1);
    if (!s) {
        *len = SIZE_MAX;
        return NULL;
    }
    if (napi_get_value_string_utf8(env, v, s, n + 1, &n) != napi_ok) {
        free(s);
        return NULL;
    }
    *len = n;
    return s;
}

void *janas_node_bytes(napi_env env, napi_value v, size_t *len)
{
    bool is = false;
    void *data = NULL;
    size_t n = 0;
    *len = 0;
    if (!v)
        return NULL;
    if (napi_is_typedarray(env, v, &is) == napi_ok && is) {
        napi_typedarray_type t;
        size_t count = 0;
        if (napi_get_typedarray_info(env, v, &t, &count, &data, NULL, NULL) !=
            napi_ok)
            return NULL;
        static const unsigned char size[] = {1, 1, 1, 2, 2, 4, 4, 4, 8, 8, 8};
        n = count * ((unsigned)t < sizeof(size) ? size[t] : 1);
    } else if (napi_is_arraybuffer(env, v, &is) != napi_ok || !is ||
               napi_get_arraybuffer_info(env, v, &data, &n) != napi_ok)
        return NULL;
    if (!n || !data)
        return NULL;
    void *copy = malloc(n);
    if (!copy) {
        *len = SIZE_MAX;
        return NULL;
    }
    memcpy(copy, data, n);
    *len = n;
    return copy;
}

int janas_node_options(napi_env env, napi_value obj, const char *what,
                       const char *const *known)
{
    napi_valuetype t;
    if (!obj || napi_typeof(env, obj, &t) != napi_ok || t == napi_undefined ||
        t == napi_null)
        return 0;
    if (t != napi_object) {
        janas_node_type_error(env, "janas: %s: the options must be an object",
                              what);
        return -1;
    }
    napi_value names;
    uint32_t n = 0;
    if (napi_get_property_names(env, obj, &names) != napi_ok ||
        napi_get_array_length(env, names, &n) != napi_ok)
        return 0;
    for (uint32_t i = 0; i < n; i++) {
        napi_value k;
        char name[64];
        size_t len = 0;
        if (napi_get_element(env, names, i, &k) != napi_ok ||
            napi_get_value_string_utf8(env, k, name, sizeof(name), &len) !=
                napi_ok)
            continue;
        const char *const *p = known;
        while (*p && strcmp(*p, name))
            p++;
        if (*p)
            continue;
        char list[512];
        size_t at = 0;
        list[0] = 0;
        for (p = known; *p && at < sizeof(list); p++) {
            int w = snprintf(list + at, sizeof(list) - at, "%s%s",
                             p == known ? "" : ", ", *p);
            at += w > 0 ? (size_t)w : 0;
        }
        janas_node_type_error(env, "janas: %s: no option '%s' (there are %s)",
                              what, name, list);
        return -1;
    }
    return 0;
}

static const char *type_name(napi_valuetype type)
{
    switch (type) {
    case napi_number:
        return "a number";
    case napi_boolean:
        return "a boolean";
    case napi_string:
        return "a string";
    default:
        return "an object";
    }
}

int janas_node_opt(napi_env env, napi_value obj, const char *what,
                   const char *name, napi_valuetype type, napi_value *v)
{
    napi_valuetype t;
    if (!obj || napi_typeof(env, obj, &t) != napi_ok || t != napi_object ||
        napi_get_named_property(env, obj, name, v) != napi_ok ||
        napi_typeof(env, *v, &t) != napi_ok || t == napi_undefined)
        return 0;
    if (t != type) {
        janas_node_type_error(env, "janas: %s: %s must be %s", what, name,
                              type_name(type));
        return -1;
    }
    return 1;
}

int janas_node_opt_number(napi_env env, napi_value obj, const char *what,
                          const char *name, double lo, double hi, double *out)
{
    napi_value v;
    int r = janas_node_opt(env, obj, what, name, napi_number, &v);
    if (r <= 0)
        return r;
    double x = 0;
    napi_get_value_double(env, v, &x);
    if (!isfinite(x) || x < lo || x > hi) {
        janas_node_type_error(env, "janas: %s: %s must be from %g to %g", what,
                              name, lo, hi);
        return -1;
    }
    *out = x;
    return 1;
}

int janas_node_opt_int(napi_env env, napi_value obj, const char *what,
                       const char *name, int64_t lo, int64_t hi, int64_t *out)
{
    double x;
    int r =
        janas_node_opt_number(env, obj, what, name, (double)lo, (double)hi, &x);
    if (r <= 0)
        return r;
    if (x != floor(x)) {
        janas_node_type_error(env, "janas: %s: %s must be a whole number", what,
                              name);
        return -1;
    }
    *out = (int64_t)x;
    return 1;
}

int janas_node_opt_bool(napi_env env, napi_value obj, const char *what,
                        const char *name, int *out)
{
    napi_value v;
    int r = janas_node_opt(env, obj, what, name, napi_boolean, &v);
    if (r <= 0)
        return r;
    bool b = false;
    napi_get_value_bool(env, v, &b);
    *out = b;
    return 1;
}

int janas_node_opt_string(napi_env env, napi_value obj, const char *what,
                          const char *name, char **out)
{
    napi_value v;
    int r = janas_node_opt(env, obj, what, name, napi_string, &v);
    if (r <= 0)
        return r;
    size_t len;
    *out = janas_node_text(env, v, &len);
    if (!*out) {
        janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
        return -1;
    }
    return 1;
}

static napi_value abi_version(napi_env env, napi_callback_info info)
{
    (void)info;
    return janas_node_number(env, janas_llm_abi_version());
}

static int keep_class(napi_env env, napi_value cls, napi_ref *ref)
{
    return cls && napi_create_reference(env, cls, 1, ref) == napi_ok;
}

/* What Node calls when it loads the module, once in each environment. */
__attribute__((visibility("default"))) napi_value
napi_register_module_v1(napi_env env, napi_value exports)
{
    struct janas_node_env *st = calloc(1, sizeof(*st));
    if (!st)
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    st->refs = 1;
    if (napi_set_instance_data(env, st, env_free, NULL) != napi_ok) {
        free(st);
        return janas_node_throw(env, JANAS_LLM_EFAIL,
                                "janas: the module cannot keep its data");
    }
    napi_value model = janas_node_model_class(env);
    napi_value chat = janas_node_chat_class(env);
    napi_value reply = janas_node_reply_class(env);
    napi_value speech = janas_node_speech_class(env);
    if (!keep_class(env, model, &st->model_class) ||
        !keep_class(env, chat, &st->chat_class) ||
        !keep_class(env, reply, &st->reply_class) ||
        !keep_class(env, speech, &st->speech_class) ||
        napi_add_env_cleanup_hook(env, env_cleanup, st) != napi_ok)
        return janas_node_throw(env, JANAS_LLM_EFAIL,
                                "janas: the module cannot define its classes");
    napi_property_descriptor d[] = {
        {.utf8name = "open",
         .method = janas_node_open,
         .attributes = napi_enumerable},
        {.utf8name = "openSpeech",
         .method = janas_node_open_speech,
         .attributes = napi_enumerable},
        {.utf8name = "chatsDir",
         .method = janas_node_chats_dir,
         .attributes = napi_enumerable},
        {.utf8name = "listChats",
         .method = janas_node_list_chats,
         .attributes = napi_enumerable},
        {.utf8name = "abiVersion",
         .method = abi_version,
         .attributes = napi_enumerable},
        {.utf8name = "version",
         .value = janas_node_string(env, JANAS_VERSION, JANAS_NAPI_AUTO_LENGTH),
         .attributes = napi_enumerable},
        {.utf8name = "Model", .value = model, .attributes = napi_enumerable},
        {.utf8name = "Chat", .value = chat, .attributes = napi_enumerable},
        {.utf8name = "Reply", .value = reply, .attributes = napi_enumerable},
        {.utf8name = "Speech", .value = speech, .attributes = napi_enumerable},
    };
    if (napi_define_properties(env, exports, sizeof(d) / sizeof(d[0]), d) !=
        napi_ok)
        return NULL;
    return exports;
}
