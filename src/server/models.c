/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * models.c - GET /models and GET /models/{model}: the one model this server
 * runs, under the name its clients use for it; DELETE /models/{model}, which
 * a server that runs its model from a file does not do.
 */
#include "server.h"

#include <microhttpd.h>
#include <string.h>
#include <yyjson.h>

/* The embedding model, when there is one. */
static yyjson_mut_val *embed_obj(struct srv_req *r, yyjson_mut_doc *d)
{
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, o, "id", r->srv->cfg.embed_id);
    yyjson_mut_obj_add_str(d, o, "object", "model");
    yyjson_mut_obj_add_int(d, o, "created", srv_engine_created(r->srv->engine));
    yyjson_mut_obj_add_str(d, o, "owned_by", "janas");
    yyjson_mut_obj_add_int(d, o, "embedding_dimensions",
                           srv_engine_embed_dim(r->srv->engine));
    return o;
}

static yyjson_mut_val *model_obj(struct srv_req *r, yyjson_mut_doc *d)
{
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, o, "id", r->srv->cfg.model_id);
    yyjson_mut_obj_add_str(d, o, "object", "model");
    yyjson_mut_obj_add_int(d, o, "created", srv_engine_created(r->srv->engine));
    yyjson_mut_obj_add_str(d, o, "owned_by", "janas");
    /* not in OpenAI's schema, which allows more: what the model is and how
       it runs on this machine */
    yyjson_mut_obj_add_str(d, o, "description",
                           srv_engine_describe(r->srv->engine));
    if (r->srv->cfg.test) { /* for the test page: Janas presents itself */
        yyjson_mut_val *j = yyjson_mut_obj_add_obj(d, o, "janas");
        yyjson_mut_obj_add_str(d, j, "system",
                               srv_engine_system(r->srv->engine));
    }
    return o;
}

static int reply_doc(struct srv_req *r, yyjson_mut_doc *d)
{
    size_t n = 0;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_NOFLAG, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, n);
}

int srv_models_list(struct srv_req *r)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, root);
    yyjson_mut_obj_add_str(d, root, "object", "list");
    yyjson_mut_val *arr = yyjson_mut_obj_add_arr(d, root, "data");
    yyjson_mut_arr_append(arr, model_obj(r, d));
    if (r->srv->cfg.embed_id)
        yyjson_mut_arr_append(arr, embed_obj(r, d));
    return reply_doc(r, d);
}

int srv_models_get(struct srv_req *r)
{
    if (r->srv->cfg.embed_id && !strcmp(r->param[0], r->srv->cfg.embed_id)) {
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_doc_set_root(d, embed_obj(r, d));
        return reply_doc(r, d);
    }
    if (strcmp(r->param[0], r->srv->cfg.model_id) != 0)
        return srv_reply_error(r, 404, "invalid_request_error",
                               "model_not_found",
                               "The model '%s' does not exist here: this "
                               "server runs '%s'.",
                               r->param[0], r->srv->cfg.model_id);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_doc_set_root(d, model_obj(r, d));
    return reply_doc(r, d);
}

int srv_models_delete(struct srv_req *r)
{
    if (strcmp(r->param[0], r->srv->cfg.model_id) != 0)
        return srv_reply_error(r, 404, "invalid_request_error",
                               "model_not_found",
                               "The model '%s' does not exist here: this "
                               "server runs '%s'.",
                               r->param[0], r->srv->cfg.model_id);
    return srv_reply_error(r, 403, "invalid_request_error", "permission_denied",
                           "The model '%s' is the one this server runs, from "
                           "its file: it cannot be deleted through the API.",
                           r->param[0]);
}
