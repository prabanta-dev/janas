/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * responses_mcp.c - the tool of type "mcp" of OpenAI's Responses API: the
 * server reaches the MCP servers a request names (server_url, with its
 * headers and authorization), gives their tools to the model, and runs the
 * calls the model writes, through libjanas_mcp.
 *
 * A response begins with an mcp_list_tools item for each server whose tools
 * the history has not listed yet. When the model calls one of its tools,
 * the call either needs a person's approval (require_approval, "always" by
 * default, as OpenAI has it) - then the response ends with an
 * mcp_approval_request, and the next request answers it with an
 * mcp_approval_response - or it is run at once: an mcp_call item with its
 * output, and the model goes on in the same response, a round at a time,
 * each round a job of its own whose prompt the engine finds computed but
 * for the new tokens.
 *
 * The model sees the tools as server_label__tool; the items say the label
 * and the tool apart, as OpenAI's do.
 */
#include "request.h"
#include "responses.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "janas/mcp.h"

#define MAX_ROUNDS 8
#define CALL_SECONDS 120

/* A tool filter: names (NULL: any), and read_only (-1: either). */
struct filter {
    char **names;
    int n_names;
    int read_only;
    int set;
};

struct server {
    char label[64];
    janas_mcp *m;
    struct filter allowed;
    int approval; /* 1: always, 0: never, 2: by the filters */
    struct filter always, never;
};

struct resp_mcp {
    struct server *s;
    int n;
    int rounds;
};

static void filter_free(struct filter *f)
{
    for (int i = 0; i < f->n_names; i++)
        free(f->names[i]);
    free(f->names);
}

void resp_mcp_free(struct resp_mcp *mc)
{
    if (!mc)
        return;
    for (int i = 0; i < mc->n; i++) {
        janas_mcp_close(mc->s[i].m);
        filter_free(&mc->s[i].allowed);
        filter_free(&mc->s[i].always);
        filter_free(&mc->s[i].never);
    }
    free(mc->s);
    free(mc);
}

/* An array of names, or {tool_names, read_only}; 0, or -1 when neither. */
static int read_filter(yyjson_val *v, struct filter *f)
{
    *f = (struct filter){.read_only = -1};
    if (!v || yyjson_is_null(v))
        return 0;
    yyjson_val *names = v;
    if (yyjson_is_obj(v)) {
        yyjson_val *ro = yyjson_obj_get(v, "read_only");
        if (ro && !yyjson_is_bool(ro))
            return -1;
        if (ro)
            f->read_only = yyjson_get_bool(ro);
        names = yyjson_obj_get(v, "tool_names");
    } else if (!yyjson_is_arr(v)) {
        return -1;
    }
    f->set = 1;
    if (!names)
        return 0;
    if (!yyjson_is_arr(names))
        return -1;
    f->names = calloc(yyjson_arr_size(names) + 1, sizeof(*f->names));
    if (!f->names)
        return -1;
    size_t idx, max;
    yyjson_val *n;
    yyjson_arr_foreach(names, idx, max, n)
    {
        if (!yyjson_is_str(n))
            return -1;
        f->names[f->n_names++] = strdup(yyjson_get_str(n));
    }
    return 0;
}

/* Whether the filter takes the tool; an empty one takes them all. */
static int takes(const struct filter *f, janas_mcp *m, const char *name)
{
    if (f->read_only >= 0 &&
        janas_mcp_tool_read_only(m, name, -1) != (f->read_only ? 1 : 0))
        return 0;
    if (!f->names)
        return 1;
    for (int i = 0; i < f->n_names; i++)
        if (f->names[i] && !strcmp(f->names[i], name))
            return 1;
    return 0;
}

static struct server *by_label(struct resp_mcp *mc, const char *label, size_t n)
{
    for (int i = 0; mc && i < mc->n; i++)
        if (strlen(mc->s[i].label) == n && !memcmp(mc->s[i].label, label, n))
            return &mc->s[i];
    return NULL;
}

/* The server and the tool of a name the model uses (label__tool). */
static struct server *owner(struct resp_mcp *mc, const char *name,
                            const char **tool)
{
    const char *sep = name ? strstr(name, "__") : NULL;
    struct server *s = sep ? by_label(mc, name, (size_t)(sep - name)) : NULL;
    if (s)
        *tool = sep + 2;
    return s;
}

void resp_mcp_cancel(struct resp_mcp *mc)
{
    for (int i = 0; mc && i < mc->n; i++)
        if (mc->s[i].m)
            janas_mcp_cancel(mc->s[i].m);
}

int resp_mcp_owns(const struct resp *x, const char *name)
{
    const char *tool;
    return x->mcp && owner(x->mcp, name, &tool) != NULL;
}

static int needs_approval(const struct server *s, const char *tool)
{
    if (s->approval != 2)
        return s->approval;
    if (s->never.set && takes(&s->never, s->m, tool))
        return 0;
    if (s->always.set && takes(&s->always, s->m, tool))
        return 1;
    return 1;
}

/* The mcp_list_tools item of a server. */
static yyjson_mut_val *list_item(yyjson_mut_doc *d, const struct server *s,
                                 yyjson_val *tools)
{
    yyjson_mut_val *it = yyjson_mut_obj(d);
    char id[48];
    srv_new_id(id, sizeof(id), "mcpl_");
    yyjson_mut_obj_add_strcpy(d, it, "id", id);
    yyjson_mut_obj_add_str(d, it, "type", "mcp_list_tools");
    yyjson_mut_obj_add_strcpy(d, it, "server_label", s->label);
    yyjson_mut_val *arr = yyjson_mut_obj_add_arr(d, it, "tools");
    size_t idx, max;
    yyjson_val *t;
    yyjson_arr_foreach(tools, idx, max, t)
    {
        yyjson_val *f = yyjson_obj_get(t, "function");
        const char *name = yyjson_get_str(yyjson_obj_get(f, "name"));
        if (!name || strlen(name) <= strlen(s->label) + 2)
            continue;
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, arr);
        yyjson_mut_obj_add_strcpy(d, o, "name", name + strlen(s->label) + 2);
        yyjson_val *desc = yyjson_obj_get(f, "description");
        if (yyjson_get_len(desc))
            yyjson_mut_obj_add_val(d, o, "description",
                                   yyjson_val_mut_copy(d, desc));
        else
            yyjson_mut_obj_add_null(d, o, "description");
        yyjson_mut_obj_add_val(
            d, o, "input_schema",
            yyjson_val_mut_copy(d, yyjson_obj_get(f, "parameters")));
        int ro =
            janas_mcp_tool_read_only(s->m, name + strlen(s->label) + 2, -1);
        if (ro >= 0) {
            yyjson_mut_val *an = yyjson_mut_obj_add_obj(d, o, "annotations");
            yyjson_mut_obj_add_bool(d, an, "read_only", ro);
        } else {
            yyjson_mut_obj_add_null(d, o, "annotations");
        }
    }
    yyjson_mut_obj_add_null(d, it, "error");
    return it;
}

/* Runs a call: the mcp_call item into arr, the text for the model (the
   output, or what went wrong) into *text. */
static void run_call(yyjson_mut_doc *d, yyjson_mut_val *arr, struct server *s,
                     const char *tool, const char *args,
                     const char *approval_id, char **text)
{
    int32_t rc = janas_mcp_call(s->m, tool, -1, args, -1, CALL_SECONDS);
    yyjson_mut_val *it = yyjson_mut_arr_add_obj(d, arr);
    char id[48];
    srv_new_id(id, sizeof(id), "mcp_");
    yyjson_mut_obj_add_strcpy(d, it, "id", id);
    yyjson_mut_obj_add_str(d, it, "type", "mcp_call");
    yyjson_mut_obj_add_strcpy(d, it, "server_label", s->label);
    yyjson_mut_obj_add_strcpy(d, it, "name", tool);
    yyjson_mut_obj_add_strcpy(d, it, "arguments", args);
    if (approval_id)
        yyjson_mut_obj_add_strcpy(d, it, "approval_request_id", approval_id);
    else
        yyjson_mut_obj_add_null(d, it, "approval_request_id");
    char *out = NULL;
    int32_t len = 0;
    if (rc == JANAS_MCP_OK) {
        janas_mcp_result(s->m, NULL, 0, &len);
        out = malloc((size_t)len + 1);
        if (out && janas_mcp_result(s->m, out, len + 1, &len) != JANAS_MCP_OK)
            out[0] = 0;
    }
    if (rc == JANAS_MCP_OK && out && !janas_mcp_result_error(s->m)) {
        yyjson_mut_obj_add_strcpy(d, it, "output", out);
        yyjson_mut_obj_add_null(d, it, "error");
        yyjson_mut_obj_add_str(d, it, "status", "completed");
        *text = out;
        return;
    }
    yyjson_mut_obj_add_null(d, it, "output");
    yyjson_mut_val *e = yyjson_mut_obj_add_obj(d, it, "error");
    const char *why = rc == JANAS_MCP_OK && out ? out : janas_mcp_last_error();
    if (rc == JANAS_MCP_OK) { /* the tool ran and said it failed */
        yyjson_mut_obj_add_str(d, e, "type", "mcp_tool_execution_error");
        yyjson_mut_val *c = yyjson_mut_obj_add_arr(d, e, "content");
        yyjson_mut_val *p = yyjson_mut_arr_add_obj(d, c);
        yyjson_mut_obj_add_str(d, p, "type", "text");
        yyjson_mut_obj_add_strcpy(d, p, "text", why);
    } else {
        yyjson_mut_obj_add_str(d, e, "type", "mcp_protocol_error");
        yyjson_mut_obj_add_int(d, e, "code", -32603);
        yyjson_mut_obj_add_strcpy(d, e, "message", why);
    }
    yyjson_mut_obj_add_str(d, it, "status", "failed");
    size_t n = strlen(why) + 8;
    *text = malloc(n);
    if (*text)
        snprintf(*text, n, "Error: %s", why);
    free(out);
}

/* The approval responses of the input: the calls they approve run now,
   their mcp_call items first in the output, their outputs kept on the job
   for the messages (resp_items_to_msgs). */
static int approvals(struct resp *x, yyjson_val *ctx, yyjson_mut_doc *d,
                     yyjson_mut_val *out)
{
    yyjson_doc *in = yyjson_read(x->input, strlen(x->input), 0);
    size_t idx, max;
    yyjson_val *it;
    int rc = 0;
    yyjson_arr_foreach(yyjson_doc_get_root(in), idx, max, it)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(it, "type"));
        yyjson_val *ok = yyjson_obj_get(it, "approve");
        const char *rid =
            yyjson_get_str(yyjson_obj_get(it, "approval_request_id"));
        if (!type || strcmp(type, "mcp_approval_response") != 0 || !rid ||
            !yyjson_get_bool(ok))
            continue;
        /* the request it answers, in the history */
        yyjson_val *q = NULL, *h;
        size_t i2, m2;
        yyjson_arr_foreach(ctx, i2, m2, h)
        {
            if (!strcmp(yyjson_get_str(yyjson_obj_get(h, "id")) ?: "", rid) &&
                !strcmp(yyjson_get_str(yyjson_obj_get(h, "type")) ?: "",
                        "mcp_approval_request"))
                q = h;
        }
        const char *label = yyjson_get_str(yyjson_obj_get(q, "server_label"));
        const char *tool = yyjson_get_str(yyjson_obj_get(q, "name"));
        const char *args = yyjson_get_str(yyjson_obj_get(q, "arguments"));
        struct server *s =
            label ? by_label(x->mcp, label, strlen(label)) : NULL;
        char *text = NULL;
        if (!q || !s || !tool) {
            text = strdup("Error: the approved call is not known here (its "
                          "server is not among this request's tools).");
        } else {
            run_call(d, out, s, tool, args ? args : "{}", rid, &text);
        }
        struct srv_job *j = x->j;
        char **ni = realloc(j->mcp_id, (size_t)(j->n_mcp + 1) * sizeof(*ni));
        if (ni)
            j->mcp_id = ni;
        char **no = realloc(j->mcp_out, (size_t)(j->n_mcp + 1) * sizeof(*no));
        if (no)
            j->mcp_out = no;
        char *idc = strdup(rid);
        if (!ni || !no || !idc || !text) {
            free(idc);
            free(text);
            rc = -1;
            break;
        }
        j->mcp_id[j->n_mcp] = idc;
        j->mcp_out[j->n_mcp++] = text;
    }
    yyjson_doc_free(in);
    return rc;
}

/* Whether the history already listed the tools of a server. */
static int listed(yyjson_val *ctx, const char *label)
{
    size_t idx, max;
    yyjson_val *it;
    yyjson_arr_foreach(ctx, idx, max, it)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(it, "type"));
        const char *l = yyjson_get_str(yyjson_obj_get(it, "server_label"));
        if (type && l && !strcmp(type, "mcp_list_tools") && !strcmp(l, label) &&
            !yyjson_get_str(yyjson_obj_get(it, "error")))
            return 1;
    }
    return 0;
}

/* One tool of type mcp, reached: 0, or -1 with the answer given. */
static int reach(struct srv_req *r, struct server *s, yyjson_val *t, int *ret)
{
    const char *url = yyjson_get_str(yyjson_obj_get(t, "server_url"));
    if (!url) {
        *ret = srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "MCP server '%s': only 'server_url' is served "
                               "here (no connectors, no tunnels).",
                               s->label);
        return -1;
    }
    /* the headers, and the authorization as a bearer token */
    const char *h[34];
    char *own[34];
    int nh = 0;
    yyjson_val *hs = yyjson_obj_get(t, "headers");
    size_t idx, max;
    yyjson_val *k, *v;
    if (yyjson_is_obj(hs))
        yyjson_obj_foreach(hs, idx, max, k, v)
        {
            if (nh >= 32 || !yyjson_is_str(v))
                continue;
            size_t n = yyjson_get_len(k) + yyjson_get_len(v) + 3;
            own[nh] = malloc(n);
            if (own[nh]) {
                snprintf(own[nh], n, "%s: %s", yyjson_get_str(k),
                         yyjson_get_str(v));
                h[nh] = own[nh];
                nh++;
            }
        }
    const char *auth = yyjson_get_str(yyjson_obj_get(t, "authorization"));
    if (auth) {
        size_t n = strlen(auth) + 24;
        own[nh] = malloc(n);
        if (own[nh]) {
            snprintf(own[nh], n, "Authorization: Bearer %s", auth);
            h[nh] = own[nh];
            nh++;
        }
    }
    int32_t rc = janas_mcp_open_url(s->label, url, nh, h, 30, &s->m);
    for (int i = 0; i < nh; i++)
        free(own[i]);
    if (rc != JANAS_MCP_OK) {
        *ret = srv_reply_error(r, 424, "external_connector_error", NULL,
                               "Error retrieving tool list from MCP server "
                               "'%s': %s",
                               s->label, janas_mcp_last_error());
        return -1;
    }
    return 0;
}

int resp_mcp_read(struct srv_req *r, struct resp *x, yyjson_val *root, int *ret)
{
    yyjson_val *tools = yyjson_obj_get(root, "tools");
    size_t idx, max, count = 0;
    yyjson_val *t;
    yyjson_arr_foreach(tools, idx, max, t)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
        count += type && !strcmp(type, "mcp");
    }
    if (!count)
        return 0;
    if (r->srv->cfg.no_mcp) {
        *ret = srv_reply_error(r, 400, "invalid_request_error",
                               "unsupported_value",
                               "This server was started with --no-mcp: it "
                               "reaches no MCP server.");
        return -1;
    }
    x->mcp = calloc(1, sizeof(*x->mcp));
    if (x->mcp)
        x->mcp->s = calloc(count, sizeof(*x->mcp->s));
    if (!x->mcp || !x->mcp->s)
        goto oom;
    yyjson_doc *cd = yyjson_read(x->context, strlen(x->context), 0);
    yyjson_val *ctx = yyjson_doc_get_root(cd);
    /* the model's tools: the functions of the request, then the servers' */
    yyjson_doc *fd =
        x->j->tools ? yyjson_read(x->j->tools, x->j->n_tools, 0) : NULL;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *all = fd ? yyjson_val_mut_copy(d, yyjson_doc_get_root(fd))
                             : yyjson_mut_arr(d);
    yyjson_mut_val *items = yyjson_mut_arr(d);
    yyjson_doc_free(fd);
    int rc = 0;
    yyjson_arr_foreach(tools, idx, max, t)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
        if (!type || strcmp(type, "mcp") != 0)
            continue;
        const char *label = yyjson_get_str(yyjson_obj_get(t, "server_label"));
        struct server *s = &x->mcp->s[x->mcp->n];
        if (!label || !*label || strlen(label) >= sizeof(s->label) ||
            strstr(label, "__") || strpbrk(label, " \"<>\n")) {
            rc = -1;
            *ret = srv_reply_error(r, 400, "invalid_request_error",
                                   "invalid_value",
                                   "tools[%zu]: an MCP tool needs a "
                                   "'server_label' (a plain name, no '__').",
                                   idx);
            break;
        }
        if (by_label(x->mcp, label, strlen(label))) {
            rc = -1;
            *ret = srv_reply_error(
                r, 400, "invalid_request_error", "invalid_value",
                "tools[%zu]: the label '%s' is used twice.", idx, label);
            break;
        }
        snprintf(s->label, sizeof(s->label), "%s", label);
        yyjson_val *ra = yyjson_obj_get(t, "require_approval");
        const char *ras = yyjson_get_str(ra);
        s->approval = !ra || yyjson_is_null(ra) ? 1
                      : ras                     ? strcmp(ras, "never") != 0
                                                : 2;
        if (read_filter(yyjson_obj_get(t, "allowed_tools"), &s->allowed) ||
            (s->approval == 2 &&
             (read_filter(yyjson_obj_get(ra, "always"), &s->always) ||
              read_filter(yyjson_obj_get(ra, "never"), &s->never)))) {
            x->mcp->n++;
            rc = -1;
            *ret = srv_reply_error(r, 400, "invalid_request_error",
                                   "invalid_value",
                                   "tools[%zu]: allowed_tools and the filters "
                                   "of require_approval are lists of names or "
                                   "{tool_names, read_only}.",
                                   idx);
            break;
        }
        x->mcp->n++;
        if (reach(r, s, t, ret) != 0) {
            rc = -1;
            break;
        }
        /* its tools, as the model sees them, those allowed */
        int32_t len = 0;
        char pre[80];
        snprintf(pre, sizeof(pre), "%s__", s->label);
        janas_mcp_tools(s->m, pre, -1, NULL, 0, &len);
        char *tj = malloc((size_t)len + 1);
        if (!tj || janas_mcp_tools(s->m, pre, -1, tj, len + 1, &len) != 0) {
            free(tj);
            goto oom_in;
        }
        yyjson_doc *td = yyjson_read(tj, (size_t)len, 0);
        free(tj);
        yyjson_mut_val *mine = yyjson_mut_arr(d);
        yyjson_val *one;
        size_t i2, m2;
        yyjson_arr_foreach(yyjson_doc_get_root(td), i2, m2, one)
        {
            const char *name = yyjson_get_str(
                yyjson_obj_get(yyjson_obj_get(one, "function"), "name"));
            if (name && takes(&s->allowed, s->m, name + strlen(pre))) {
                yyjson_mut_arr_append(all, yyjson_val_mut_copy(d, one));
                yyjson_mut_arr_append(mine, yyjson_val_mut_copy(d, one));
            }
        }
        if (!listed(ctx, s->label)) {
            yyjson_doc *md = yyjson_mut_val_imut_copy(mine, NULL);
            yyjson_mut_arr_append(items,
                                  list_item(d, s, yyjson_doc_get_root(md)));
            yyjson_doc_free(md);
        }
        yyjson_doc_free(td);
    }
    if (rc == 0 && approvals(x, ctx, d, items) != 0)
        goto oom_in;
    if (rc == 0) {
        size_t n;
        free(x->j->tools);
        x->j->tools =
            yyjson_mut_val_write(all, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
        x->j->n_tools = n;
        free(x->done_items);
        x->done_items =
            yyjson_mut_val_write(items, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
        if (!x->j->tools || !x->done_items)
            goto oom_in;
        /* tool_choice {type: mcp, server_label, name} */
        yyjson_val *tc = yyjson_obj_get(root, "tool_choice");
        const char *tt = yyjson_get_str(yyjson_obj_get(tc, "type"));
        if (tt && !strcmp(tt, "mcp")) {
            const char *l = yyjson_get_str(yyjson_obj_get(tc, "server_label"));
            const char *nm = yyjson_get_str(yyjson_obj_get(tc, "name"));
            if (l && nm) {
                x->j->tool_choice = JANAS_LLM_TOOLS_FUNCTION;
                snprintf(x->j->tool_name, sizeof(x->j->tool_name), "%s__%s", l,
                         nm);
            } else {
                x->j->tool_choice = JANAS_LLM_TOOLS_REQUIRED;
            }
        }
    }
    yyjson_mut_doc_free(d);
    yyjson_doc_free(cd);
    return rc;
oom_in:
    yyjson_mut_doc_free(d);
    yyjson_doc_free(cd);
oom:
    *ret = srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    return -1;
}

/* A copy of the job's settings and messages, for the next round. */
static struct srv_job *next_job(const struct srv_job *o)
{
    struct srv_job *j = srv_job_new();
    if (!j || srv_job_choices(j, 1, 1) != 0)
        goto bad;
    j->kind = o->kind;
    j->p = o->p;
    j->seeded = o->seeded;
    j->parallel = o->parallel;
    j->format = o->format;
    /* a choice forced once is not forced again: the model now answers */
    j->tool_choice = JANAS_LLM_TOOLS_AUTO;
    if ((o->tools && !(j->tools = srv_dup_n(o->tools, o->n_tools))) ||
        (o->schema && !(j->schema = srv_dup_n(o->schema, o->n_schema))))
        goto bad;
    j->n_tools = o->n_tools;
    j->n_schema = o->n_schema;
    j->msg = calloc((size_t)o->n_msg + 1, sizeof(*j->msg));
    if (!j->msg)
        goto bad;
    for (int32_t i = 0; i < o->n_msg; i++) {
        const struct srv_msg *m = &o->msg[i];
        struct srv_msg *c = &j->msg[j->n_msg++];
        *c = (struct srv_msg){.role = m->role, .n = m->n};
        if (!(c->text = srv_dup_n(m->text ? m->text : "", m->n)))
            goto bad;
        for (int k = 0; k < m->n_calls; k++)
            if (resp_add_call(
                    c, m->call_name[k],
                    srv_dup_n(m->call_args[k], strlen(m->call_args[k]))) != 0)
                goto bad;
    }
    return j;
bad:
    srv_job_release(j);
    return NULL;
}

int resp_mcp_round(struct resp *x)
{
    struct srv_job *j = x->j;
    if (!x->mcp || j->err != JANAS_LLM_OK || j->cancel || x->consumed)
        return 0;
    const struct srv_choice *ch = &j->ch[0];
    int mine = 0, others = 0;
    for (int i = 0; i < ch->n_calls; i++) {
        if (resp_mcp_owns(x, ch->calls[i].name))
            mine++;
        else
            others++;
    }
    if (!mine)
        return 0;
    /* the round's items: those of the job, then a call or a request for
       approval for each of the MCP calls */
    yyjson_doc *dd = x->done_items
                         ? yyjson_read(x->done_items, strlen(x->done_items), 0)
                         : NULL;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *arr = dd ? yyjson_val_mut_copy(d, yyjson_doc_get_root(dd))
                             : yyjson_mut_arr(d);
    yyjson_doc_free(dd);
    size_t before = yyjson_mut_arr_size(arr);
    resp_job_items(d, arr, x);
    if (x->stream) /* told already, as they were written */
        x->items_told += (int)(yyjson_mut_arr_size(arr) - before);
    char **texts = calloc((size_t)ch->n_calls, sizeof(*texts));
    int pending = 0;
    for (int i = 0; texts && i < ch->n_calls; i++) {
        const struct srv_call *k = &ch->calls[i];
        const char *tool;
        struct server *s = owner(x->mcp, k->name, &tool);
        if (!s)
            continue;
        if (needs_approval(s, tool)) {
            yyjson_mut_val *it = yyjson_mut_arr_add_obj(d, arr);
            char id[48];
            srv_new_id(id, sizeof(id), "mcpr_");
            yyjson_mut_obj_add_strcpy(d, it, "id", id);
            yyjson_mut_obj_add_str(d, it, "type", "mcp_approval_request");
            yyjson_mut_obj_add_strcpy(d, it, "server_label", s->label);
            yyjson_mut_obj_add_strcpy(d, it, "name", tool);
            yyjson_mut_obj_add_strcpy(d, it, "arguments", k->args);
            pending = 1;
        } else {
            run_call(d, arr, s, tool, k->args, NULL, &texts[i]);
        }
    }
    size_t n;
    char *items =
        yyjson_mut_val_write(arr, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    if (items) {
        free(x->done_items);
        x->done_items = items;
        x->consumed = 1;
    }
    int again =
        items && texts && !pending && !others && ++x->mcp->rounds < MAX_ROUNDS;
    struct srv_job *nj = again ? next_job(j) : NULL;
    int ok = nj != NULL;
    if (ok) {
        /* the model's turn with its calls, then what the tools answered */
        struct srv_msg *a = resp_push(
            nj, JANAS_LLM_ROLE_ASSISTANT,
            srv_dup_n(ch->text ? ch->text : "", ch->n_text), ch->n_text);
        for (int i = 0; ok && a && i < ch->n_calls; i++)
            ok = resp_add_call(&nj->msg[nj->n_msg - 1], ch->calls[i].name,
                               srv_dup_n(ch->calls[i].args,
                                         strlen(ch->calls[i].args))) == 0;
        ok = ok && a;
        for (int i = 0; ok && i < ch->n_calls; i++) {
            const char *t = texts[i] ? texts[i] : "";
            ok = resp_push(nj, JANAS_LLM_ROLE_TOOL, srv_dup_n(t, strlen(t)),
                           strlen(t)) != NULL;
        }
    }
    for (int i = 0; texts && i < ch->n_calls; i++)
        free(texts[i]);
    free(texts);
    /* the next round, unless the response was cancelled meanwhile */
    if (!ok || resp_adopt(x, nj) != 0) {
        srv_job_release(nj);
        return 0; /* the response ends with what it has */
    }
    /* its counts added to those before it */
    x->prev_in += j->prompt_tokens;
    x->prev_cached += j->st.cached_tokens;
    x->prev_out += j->completion_tokens;
    x->prev_reason += j->reasoning_tokens;
    x->consumed = 0;
    srv_job_release(j);
    resp_round_reset(x);
    return 1;
}
