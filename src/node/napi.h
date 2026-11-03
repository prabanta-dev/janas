/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * napi.h - the part of Node-API that the module for Node.js calls, declared
 * here so that Janas builds without Node's headers. Node-API is Node's C
 * interface for native modules, the same in every version of Node from 12.22
 * on (this is its version 8): its functions are the node process's own, found
 * when the module is loaded.
 *
 * Types, values and layouts as Node's js_native_api_types.h and
 * node_api_types.h have them, only those the module uses. Where Node's
 * headers are installed, tests/test_node.c declares these functions after
 * them, and the compiler checks that the two agree. Private to src/node/.
 */
#ifndef JANAS_NODE_NAPI_H
#define JANAS_NODE_NAPI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* a string's length when it ends with a NUL */
#define JANAS_NAPI_AUTO_LENGTH SIZE_MAX

#ifndef JANAS_NAPI_NODE_TYPES /* the check takes Node's own */
typedef struct napi_env__ *napi_env;
typedef struct napi_value__ *napi_value;
typedef struct napi_ref__ *napi_ref;
typedef struct napi_callback_info__ *napi_callback_info;
typedef struct napi_deferred__ *napi_deferred;
typedef struct napi_threadsafe_function__ *napi_threadsafe_function;

typedef enum {
    napi_ok,
    napi_invalid_arg,
    napi_object_expected,
    napi_string_expected,
    napi_name_expected,
    napi_function_expected,
    napi_number_expected,
    napi_boolean_expected,
    napi_array_expected,
    napi_generic_failure,
    napi_pending_exception,
    napi_cancelled,
    napi_escape_called_twice,
    napi_handle_scope_mismatch,
    napi_callback_scope_mismatch,
    napi_queue_full,
    napi_closing,
    napi_bigint_expected,
    napi_date_expected,
    napi_arraybuffer_expected,
    napi_detachable_arraybuffer_expected,
    napi_would_deadlock
} napi_status;

typedef enum {
    napi_undefined,
    napi_null,
    napi_boolean,
    napi_number,
    napi_string,
    napi_symbol,
    napi_object,
    napi_function,
    napi_external,
    napi_bigint
} napi_valuetype;

typedef enum {
    napi_default = 0,
    napi_writable = 1 << 0,
    napi_enumerable = 1 << 1,
    napi_configurable = 1 << 2,
    napi_static = 1 << 10
} napi_property_attributes;

typedef enum {
    napi_tsfn_release,
    napi_tsfn_abort
} napi_threadsafe_function_release_mode;

typedef enum {
    napi_tsfn_nonblocking,
    napi_tsfn_blocking
} napi_threadsafe_function_call_mode;

typedef napi_value (*napi_callback)(napi_env env, napi_callback_info info);
typedef void (*napi_finalize)(napi_env env, void *finalize_data,
                              void *finalize_hint);
typedef void (*napi_threadsafe_function_call_js)(napi_env env,
                                                 napi_value js_callback,
                                                 void *context, void *data);
typedef void (*napi_cleanup_hook)(void *arg);

typedef struct {
    const char *utf8name; /* one of utf8name and name is NULL */
    napi_value name;
    napi_callback method;
    napi_callback getter;
    napi_callback setter;
    napi_value value;
    napi_property_attributes attributes;
    void *data;
} napi_property_descriptor;

typedef struct {
    const char *error_message;
    void *engine_reserved;
    uint32_t engine_error_code;
    napi_status error_code;
} napi_extended_error_info;

typedef struct {
    uint64_t lower;
    uint64_t upper;
} napi_type_tag;

typedef enum {
    napi_int8_array,
    napi_uint8_array,
    napi_uint8_clamped_array,
    napi_int16_array,
    napi_uint16_array,
    napi_int32_array,
    napi_uint32_array,
    napi_float32_array,
    napi_float64_array,
    napi_bigint64_array,
    napi_biguint64_array
} napi_typedarray_type;
#endif

/* js_native_api.h */
napi_status napi_get_last_error_info(napi_env env,
                                     const napi_extended_error_info **result);
napi_status napi_get_undefined(napi_env env, napi_value *result);
napi_status napi_get_null(napi_env env, napi_value *result);
napi_status napi_get_global(napi_env env, napi_value *result);
napi_status napi_get_boolean(napi_env env, bool value, napi_value *result);
napi_status napi_create_object(napi_env env, napi_value *result);
napi_status napi_create_double(napi_env env, double value, napi_value *result);
napi_status napi_create_string_utf8(napi_env env, const char *str,
                                    size_t length, napi_value *result);
napi_status napi_create_function(napi_env env, const char *utf8name,
                                 size_t length, napi_callback cb, void *data,
                                 napi_value *result);
napi_status napi_create_error(napi_env env, napi_value code, napi_value msg,
                              napi_value *result);
napi_status napi_typeof(napi_env env, napi_value value, napi_valuetype *result);
napi_status napi_get_value_double(napi_env env, napi_value value,
                                  double *result);
napi_status napi_get_value_bool(napi_env env, napi_value value, bool *result);
napi_status napi_get_value_string_utf8(napi_env env, napi_value value,
                                       char *buf, size_t bufsize,
                                       size_t *result);
napi_status napi_get_property_names(napi_env env, napi_value object,
                                    napi_value *result);
napi_status napi_set_named_property(napi_env env, napi_value object,
                                    const char *utf8name, napi_value value);
napi_status napi_get_named_property(napi_env env, napi_value object,
                                    const char *utf8name, napi_value *result);
napi_status napi_get_element(napi_env env, napi_value object, uint32_t index,
                             napi_value *result);
napi_status napi_is_array(napi_env env, napi_value value, bool *result);
napi_status napi_create_array_with_length(napi_env env, size_t length,
                                          napi_value *result);
napi_status napi_set_element(napi_env env, napi_value object, uint32_t index,
                             napi_value value);
napi_status napi_is_typedarray(napi_env env, napi_value value, bool *result);
napi_status napi_get_typedarray_info(napi_env env, napi_value typedarray,
                                     napi_typedarray_type *type, size_t *length,
                                     void **data, napi_value *arraybuffer,
                                     size_t *byte_offset);
napi_status napi_is_arraybuffer(napi_env env, napi_value value, bool *result);
napi_status napi_get_arraybuffer_info(napi_env env, napi_value arraybuffer,
                                      void **data, size_t *byte_length);
napi_status napi_get_array_length(napi_env env, napi_value value,
                                  uint32_t *result);
napi_status napi_call_function(napi_env env, napi_value recv, napi_value func,
                               size_t argc, const napi_value *argv,
                               napi_value *result);
napi_status napi_new_instance(napi_env env, napi_value constructor, size_t argc,
                              const napi_value *argv, napi_value *result);
napi_status napi_get_cb_info(napi_env env, napi_callback_info cbinfo,
                             size_t *argc, napi_value *argv,
                             napi_value *this_arg, void **data);
napi_status napi_define_class(napi_env env, const char *utf8name, size_t length,
                              napi_callback constructor, void *data,
                              size_t property_count,
                              const napi_property_descriptor *properties,
                              napi_value *result);
napi_status napi_define_properties(napi_env env, napi_value object,
                                   size_t property_count,
                                   const napi_property_descriptor *properties);
napi_status napi_wrap(napi_env env, napi_value js_object, void *native_object,
                      napi_finalize finalize_cb, void *finalize_hint,
                      napi_ref *result);
napi_status napi_unwrap(napi_env env, napi_value js_object, void **result);
napi_status napi_create_reference(napi_env env, napi_value value,
                                  uint32_t initial_refcount, napi_ref *result);
napi_status napi_delete_reference(napi_env env, napi_ref ref);
napi_status napi_get_reference_value(napi_env env, napi_ref ref,
                                     napi_value *result);
napi_status napi_throw_error(napi_env env, const char *code, const char *msg);
napi_status napi_throw_type_error(napi_env env, const char *code,
                                  const char *msg);
napi_status napi_is_exception_pending(napi_env env, bool *result);
napi_status napi_get_and_clear_last_exception(napi_env env, napi_value *result);
napi_status napi_create_promise(napi_env env, napi_deferred *deferred,
                                napi_value *promise);
napi_status napi_resolve_deferred(napi_env env, napi_deferred deferred,
                                  napi_value resolution);
napi_status napi_reject_deferred(napi_env env, napi_deferred deferred,
                                 napi_value rejection);
napi_status napi_set_instance_data(napi_env env, void *data,
                                   napi_finalize finalize_cb,
                                   void *finalize_hint);
napi_status napi_get_instance_data(napi_env env, void **data);
napi_status napi_type_tag_object(napi_env env, napi_value value,
                                 const napi_type_tag *type_tag);
napi_status napi_check_object_type_tag(napi_env env, napi_value value,
                                       const napi_type_tag *type_tag,
                                       bool *result);

/* node_api.h */
napi_status napi_add_env_cleanup_hook(napi_env env, napi_cleanup_hook fun,
                                      void *arg);
napi_status napi_create_threadsafe_function(
    napi_env env, napi_value func, napi_value async_resource,
    napi_value async_resource_name, size_t max_queue_size,
    size_t initial_thread_count, void *thread_finalize_data,
    napi_finalize thread_finalize_cb, void *context,
    napi_threadsafe_function_call_js call_js_cb,
    napi_threadsafe_function *result);
napi_status
napi_call_threadsafe_function(napi_threadsafe_function func, void *data,
                              napi_threadsafe_function_call_mode is_blocking);
napi_status
napi_release_threadsafe_function(napi_threadsafe_function func,
                                 napi_threadsafe_function_release_mode mode);
napi_status napi_ref_threadsafe_function(napi_env env,
                                         napi_threadsafe_function func);
napi_status napi_unref_threadsafe_function(napi_env env,
                                           napi_threadsafe_function func);

#endif
