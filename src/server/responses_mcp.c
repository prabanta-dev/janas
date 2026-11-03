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

#include "common/mcp_client.h"
#include "containers.h"
#include "janas/mcp.h"
#include "vectors.h"
#include "web_search.h"

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

/* a source the web_search and file_search calls found: cited as [n], n
   its place + 1; a page (url) or a file (file_id) */
struct web_src {
    char url[512], title[256];
    char file_id[48];
};

struct resp_mcp {
    struct server *s;
    int n;
    int rounds;
    int web; /* a web_search tool: the server searches */
    struct web_src *src;
    int n_src;
    /* a file_search tool: its vector stores, results, filters, threshold */
    char **vs;
    int n_vs, fs_max;
    char *fs_filters;
    double fs_threshold;
    /* a code_interpreter tool: its container, and the files the code
       wrote (cited where the answer names them) */
    char ci[48];
    struct ci_file {
        char id[48], name[256];
    } *ci_files;
    int n_ci_files;
};

#define WEB_NAME "web_search"
#define FILE_NAME "file_search"
#define CODE_NAME "python"
#define WEB_HITS 5

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
    free(mc->src);
    for (int i = 0; i < mc->n_vs; i++)
        free(mc->vs[i]);
    free(mc->vs);
    free(mc->fs_filters);
    free(mc->ci_files);
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
    if (x->mcp && x->mcp->web && name && !strcmp(name, WEB_NAME))
        return 1;
    if (x->mcp && x->mcp->n_vs && name && !strcmp(name, FILE_NAME))
        return 1;
    if (x->mcp && x->mcp->ci[0] && name && !strcmp(name, CODE_NAME))
        return 1;
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

/* A web_search call: Wikipedia asked, a web_search_call item with its
   sources, and for the model the pages' first sentences, numbered on from
   the sources found before so that [n] means one page in the response. */
static void run_web(struct resp_mcp *mc, yyjson_mut_doc *d, yyjson_mut_val *arr,
                    const char *args, char **text)
{
    yyjson_doc *ad = yyjson_read(args, strlen(args), 0);
    yyjson_val *a = yyjson_doc_get_root(ad);
    const char *q = yyjson_get_str(yyjson_obj_get(a, "query"));
    const char *lang = yyjson_get_str(yyjson_obj_get(a, "language"));
    struct srv_web_hit hits[WEB_HITS];
    char err[256] = "";
    int n =
        q && *q ? srv_web_search(q, lang, hits, WEB_HITS, err, sizeof err) : -1;
    if (!q || !*q)
        snprintf(err, sizeof err, "no query");
    yyjson_mut_val *it = yyjson_mut_arr_add_obj(d, arr);
    char id[48];
    srv_new_id(id, sizeof(id), "ws_");
    yyjson_mut_obj_add_strcpy(d, it, "id", id);
    yyjson_mut_obj_add_str(d, it, "type", "web_search_call");
    yyjson_mut_obj_add_str(d, it, "status", n >= 0 ? "completed" : "failed");
    yyjson_mut_val *act = yyjson_mut_obj_add_obj(d, it, "action");
    yyjson_mut_obj_add_str(d, act, "type", "search");
    yyjson_mut_obj_add_strcpy(d, act, "query", q ? q : "");
    yyjson_mut_val *srcs = yyjson_mut_obj_add_arr(d, act, "sources");
    struct janas_buf t = {0};
    if (n < 0) {
        janas_buf_printf(&t, "The search failed: %s.", err);
    } else if (n == 0) {
        janas_buf_printf(&t, "Wikipedia has no page for \"%s\".", q);
    } else {
        janas_buf_printf(&t,
                         "Wikipedia's pages for \"%s\", best first. Answer "
                         "from them, and put the number of the page a fact "
                         "comes from right after it, as [1].\n",
                         q);
        struct web_src *ns =
            realloc(mc->src, (size_t)(mc->n_src + n) * sizeof(*ns));
        for (int i = 0; ns && i < n; i++) {
            mc->src = ns;
            struct web_src *w = &mc->src[mc->n_src++];
            snprintf(w->url, sizeof w->url, "%.*s", (int)sizeof w->url - 1,
                     hits[i].url);
            snprintf(w->title, sizeof w->title, "%.*s",
                     (int)sizeof w->title - 1, hits[i].title);
            yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, srcs);
            yyjson_mut_obj_add_str(d, o, "type", "url");
            yyjson_mut_obj_add_strcpy(d, o, "url", hits[i].url);
            janas_buf_printf(&t, "\n[%d] %s (%s)\n%s\n", mc->n_src,
                             hits[i].title, hits[i].url, hits[i].text);
        }
    }
    yyjson_doc_free(ad);
    *text = t.oom ? NULL : (janas_buf_put(&t, "", 1), t.p);
    if (t.oom)
        janas_buf_free(&t);
}

/* A file_search call: the tool's vector stores searched, a
   file_search_call item with the results, and for the model the chunks
   numbered on from the sources found before. */
static void run_file(struct resp_mcp *mc, yyjson_mut_doc *d,
                     yyjson_mut_val *arr, const char *args, char **text)
{
    yyjson_doc *ad = yyjson_read(args, strlen(args), 0);
    const char *q =
        yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(ad), "query"));
    yyjson_doc *fd =
        mc->fs_filters ? yyjson_read(mc->fs_filters, strlen(mc->fs_filters), 0)
                       : NULL;
    /* every store's best, then the best of them all */
    struct srv_vhit *all = NULL;
    int n = 0;
    char err[200] = "";
    for (int i = 0; q && *q && i < mc->n_vs; i++) {
        struct srv_vhit *h;
        int k = srv_vectors_search(mc->vs[i], &q, 1, yyjson_doc_get_root(fd),
                                   mc->fs_max, mc->fs_threshold, &h, err,
                                   sizeof err);
        if (k <= 0) {
            srv_vhits_free(h, 0);
            continue;
        }
        struct srv_vhit *t = realloc(all, (size_t)(n + k) * sizeof(*t));
        if (!t) {
            srv_vhits_free(h, k);
            continue;
        }
        all = t;
        memcpy(all + n, h, (size_t)k * sizeof(*h));
        n += k;
        free(h); /* the texts moved */
    }
    yyjson_doc_free(fd);
    for (int i = 1; i < n; i++) /* best first, insertion: a few dozen */
        for (int j = i; j > 0 && all[j].score > all[j - 1].score; j--) {
            struct srv_vhit t = all[j];
            all[j] = all[j - 1];
            all[j - 1] = t;
        }
    int keep = n < mc->fs_max ? n : mc->fs_max;
    yyjson_mut_val *it = yyjson_mut_arr_add_obj(d, arr);
    char id[48];
    srv_new_id(id, sizeof(id), "fs_");
    yyjson_mut_obj_add_strcpy(d, it, "id", id);
    yyjson_mut_obj_add_str(d, it, "type", "file_search_call");
    yyjson_mut_obj_add_str(d, it, "status", q && *q ? "completed" : "failed");
    yyjson_mut_val *qs = yyjson_mut_obj_add_arr(d, it, "queries");
    yyjson_mut_arr_add_strcpy(d, qs, q ? q : "");
    yyjson_mut_val *res = yyjson_mut_obj_add_arr(d, it, "results");
    struct janas_buf t = {0};
    if (!q || !*q)
        janas_buf_printf(&t, "The search needs a query.");
    else if (!keep)
        janas_buf_printf(&t, "The files hold nothing about \"%s\".", q);
    else
        janas_buf_printf(&t,
                         "The files' passages for \"%s\", best first. "
                         "Answer from them, and put the number of the "
                         "passage a fact comes from right after it, as "
                         "[1].\n",
                         q);
    for (int i = 0; i < keep; i++) {
        struct web_src *ns =
            realloc(mc->src, (size_t)(mc->n_src + 1) * sizeof(*ns));
        if (!ns)
            break;
        mc->src = ns;
        struct web_src *w = &mc->src[mc->n_src++];
        memset(w, 0, sizeof(*w));
        snprintf(w->file_id, sizeof w->file_id, "%s", all[i].file_id);
        snprintf(w->title, sizeof w->title, "%s", all[i].filename);
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, res);
        yyjson_mut_obj_add_strcpy(d, o, "file_id", all[i].file_id);
        yyjson_mut_obj_add_strcpy(d, o, "filename", all[i].filename);
        yyjson_mut_obj_add_real(d, o, "score", all[i].score);
        yyjson_mut_obj_add_strcpy(d, o, "text", all[i].text);
        yyjson_doc *atd =
            all[i].attributes
                ? yyjson_read(all[i].attributes, strlen(all[i].attributes), 0)
                : NULL;
        if (atd)
            yyjson_mut_obj_add_val(
                d, o, "attributes",
                yyjson_val_mut_copy(d, yyjson_doc_get_root(atd)));
        yyjson_doc_free(atd);
        janas_buf_printf(&t, "\n[%d] %s\n%s\n", mc->n_src, all[i].filename,
                         all[i].text);
    }
    srv_vhits_free(all, n);
    yyjson_doc_free(ad);
    *text = t.oom ? NULL : (janas_buf_put(&t, "", 1), t.p);
    if (t.oom)
        janas_buf_free(&t);
}

/* The tail of t, at most max bytes (whole characters), into b. */
static void put_tail(struct janas_buf *b, const char *t, size_t max)
{
    size_t n = strlen(t);
    if (n > max) {
        t += n - max;
        while ((*t & 0xc0) == 0x80)
            t++;
        janas_buf_puts(b, "[...]\n");
    }
    janas_buf_puts(b, t);
}

/* A code_interpreter call: the code run in the container's session, a
   code_interpreter_call item with its output and images, and for the model
   what it printed, its error and the files it wrote. */
static void run_code(struct resp_mcp *mc, yyjson_mut_doc *d,
                     yyjson_mut_val *arr, const char *args, char **text)
{
    yyjson_doc *ad = yyjson_read(args, strlen(args), 0);
    const char *code =
        yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(ad), "code"));
    struct srv_cell c = {0};
    int rc = code ? srv_container_run(NULL, mc->ci, code, &c) : -1;
    yyjson_mut_val *it = yyjson_mut_arr_add_obj(d, arr);
    char id[48];
    srv_new_id(id, sizeof(id), "ci_");
    yyjson_mut_obj_add_strcpy(d, it, "id", id);
    yyjson_mut_obj_add_str(d, it, "type", "code_interpreter_call");
    yyjson_mut_obj_add_str(d, it, "status", rc == 0 ? "completed" : "failed");
    yyjson_mut_obj_add_strcpy(d, it, "container_id", mc->ci);
    yyjson_mut_obj_add_strcpy(d, it, "code", code ? code : "");
    yyjson_mut_val *outs = yyjson_mut_obj_add_arr(d, it, "outputs");
    struct janas_buf t = {0};
    if (!code) {
        janas_buf_puts(&t, "The call needs the code to run, as "
                           "{\"code\": \"...\"}.");
    } else if (rc != 0) {
        janas_buf_puts(&t, "The container is gone (it expired): the code "
                           "was not run.");
    } else {
        struct janas_buf lg = {0};
        if (c.logs)
            janas_buf_puts(&lg, c.logs);
        if (c.error) {
            if (lg.n && lg.p[lg.n - 1] != '\n')
                janas_buf_put(&lg, "\n", 1);
            janas_buf_puts(&lg, c.error);
        }
        janas_buf_put(&lg, "", 1);
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, outs);
        yyjson_mut_obj_add_str(d, o, "type", "logs");
        yyjson_mut_obj_add_strcpy(d, o, "logs", lg.oom ? "" : lg.p);
        janas_buf_free(&lg);
        if (c.logs && *c.logs) {
            janas_buf_puts(&t, "Output:\n");
            put_tail(&t, c.logs, 8000);
            janas_buf_puts(&t, "\n");
        }
        if (c.error) {
            janas_buf_puts(&t, "Error:\n");
            put_tail(&t, c.error, 4000);
            janas_buf_puts(&t, "\n");
        }
        for (int i = 0; i < c.n_files; i++) {
            if (i < c.n_images) {
                char url[200];
                snprintf(url, sizeof url, "/v1/containers/%s/files/%s/content",
                         mc->ci, c.file_id[i]);
                yyjson_mut_val *im = yyjson_mut_arr_add_obj(d, outs);
                yyjson_mut_obj_add_str(d, im, "type", "image");
                yyjson_mut_obj_add_strcpy(d, im, "url", url);
            }
            janas_buf_printf(&t, "%s %s\n",
                             i ? "     " : "Files written:", c.path[i]);
            struct ci_file *nf = realloc(
                mc->ci_files, (size_t)(mc->n_ci_files + 1) * sizeof(*nf));
            if (nf) {
                mc->ci_files = nf;
                struct ci_file *f = &mc->ci_files[mc->n_ci_files++];
                snprintf(f->id, sizeof f->id, "%s", c.file_id[i]);
                const char *b = strrchr(c.path[i], '/');
                snprintf(f->name, sizeof f->name, "%s", b ? b + 1 : c.path[i]);
            }
        }
        if (!t.n)
            janas_buf_puts(&t, "The code ran and printed nothing.");
    }
    srv_cell_free(&c);
    yyjson_doc_free(ad);
    janas_buf_put(&t, "", 1);
    *text = t.oom ? NULL : t.p;
    if (t.oom)
        janas_buf_free(&t);
}

/* The files the code wrote, where the text names them: their
   container_file_citation annotations (offsets in characters). */
static void code_annotate(yyjson_mut_doc *d, yyjson_mut_val *arr,
                          const struct resp_mcp *mc, const char *text, size_t n)
{
    for (int i = 0; i < mc->n_ci_files; i++) {
        const struct ci_file *f = &mc->ci_files[i];
        size_t fl = strlen(f->name);
        const char *hit = NULL;
        for (const char *p = text; fl && p + fl <= text + n; p++)
            if (!memcmp(p, f->name, fl)) {
                hit = p;
                break;
            }
        if (!hit)
            continue;
        size_t at = 0, len = 0;
        for (const char *p = text; p < hit; p++)
            at += ((unsigned char)*p & 0xc0) != 0x80;
        for (size_t k = 0; k < fl; k++)
            len += ((unsigned char)f->name[k] & 0xc0) != 0x80;
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, arr);
        yyjson_mut_obj_add_str(d, o, "type", "container_file_citation");
        yyjson_mut_obj_add_strcpy(d, o, "container_id", mc->ci);
        yyjson_mut_obj_add_strcpy(d, o, "file_id", f->id);
        yyjson_mut_obj_add_strcpy(d, o, "filename", f->name);
        yyjson_mut_obj_add_uint(d, o, "start_index", at);
        yyjson_mut_obj_add_uint(d, o, "end_index", at + len);
    }
}

void resp_web_annotate(yyjson_mut_doc *d, yyjson_mut_val *arr,
                       const struct resp *x, const char *text, size_t n)
{
    if (!x || !x->mcp || !text)
        return;
    code_annotate(d, arr, x->mcp, text, n);
    if (!x->mcp->n_src)
        return;
    /* offsets in characters, as OpenAI counts them, not bytes */
    size_t chars = 0;
    for (size_t i = 0; i < n; i++) {
        if (text[i] == '[') {
            size_t k = i + 1;
            long v = 0;
            while (k < n && text[k] >= '0' && text[k] <= '9' && v < 100000)
                v = v * 10 + (text[k++] - '0');
            if (k > i + 1 && k < n && text[k] == ']' && v >= 1 &&
                v <= x->mcp->n_src) {
                const struct web_src *w = &x->mcp->src[v - 1];
                yyjson_mut_val *o = yyjson_mut_arr_add_obj(d, arr);
                if (w->file_id[0]) { /* a passage of a file */
                    yyjson_mut_obj_add_str(d, o, "type", "file_citation");
                    yyjson_mut_obj_add_strcpy(d, o, "file_id", w->file_id);
                    yyjson_mut_obj_add_strcpy(d, o, "filename", w->title);
                    yyjson_mut_obj_add_uint(d, o, "index", chars);
                    if (((unsigned char)text[i] & 0xc0) != 0x80)
                        chars++;
                    continue;
                }
                yyjson_mut_obj_add_str(d, o, "type", "url_citation");
                yyjson_mut_obj_add_uint(d, o, "start_index", chars);
                yyjson_mut_obj_add_uint(d, o, "end_index", chars + (k + 1 - i));
                yyjson_mut_obj_add_strcpy(d, o, "url", w->url);
                yyjson_mut_obj_add_strcpy(d, o, "title", w->title);
            }
        }
        if (((unsigned char)text[i] & 0xc0) != 0x80)
            chars++;
    }
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
    int32_t rc = janas_mcp_open_url_policy(s->label, url, nh, h, 30,
                                           &r->srv->cfg.mcp_policy, &s->m);
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
    size_t idx, max, count = 0, web = 0, files = 0, code = 0;
    yyjson_val *t;
    yyjson_arr_foreach(tools, idx, max, t)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
        count += type && !strcmp(type, "mcp");
        /* web_search and its dated and preview names */
        web += type && !strncmp(type, "web_search", 10);
        files += type && !strcmp(type, "file_search");
        code += type && !strcmp(type, "code_interpreter");
    }
    if (!count && !web && !files && !code)
        return 0;
    if (code && srv_containers_off()) {
        *ret = srv_reply_error(r, 400, "invalid_request_error",
                               "unsupported_value", "%s", srv_containers_off());
        return -1;
    }
    if (web && r->srv->cfg.no_web) {
        *ret = srv_reply_error(r, 400, "invalid_request_error",
                               "unsupported_value",
                               "This server was started with --no-web: it "
                               "does not search.");
        return -1;
    }
    if (count && r->srv->cfg.no_mcp) {
        *ret = srv_reply_error(r, 400, "invalid_request_error",
                               "unsupported_value",
                               "This server was started with --no-mcp: it "
                               "reaches no MCP server.");
        return -1;
    }
    x->mcp = calloc(1, sizeof(*x->mcp));
    if (x->mcp)
        x->mcp->s = calloc(count ? count : 1, sizeof(*x->mcp->s));
    if (!x->mcp || !x->mcp->s)
        goto oom;
    x->mcp->web = web > 0;
    yyjson_arr_foreach(tools, idx, max, t)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
        if (!type || strcmp(type, "code_interpreter") != 0 || x->mcp->ci[0])
            continue;
        /* its container: one named, or one made now */
        const char *why = NULL;
        if (srv_container_for(r->srv, yyjson_obj_get(t, "container"),
                              x->mcp->ci, sizeof x->mcp->ci, &why) != 0) {
            *ret = srv_reply_error(r, 400, "invalid_request_error",
                                   "invalid_value", "tools[%zu]: %s", idx, why);
            return -1;
        }
    }
    yyjson_arr_foreach(tools, idx, max, t)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
        if (!type || strcmp(type, "file_search") != 0)
            continue;
        yyjson_val *ids = yyjson_obj_get(t, "vector_store_ids");
        yyjson_val *mx = yyjson_obj_get(t, "max_num_results");
        yyjson_val *f = yyjson_obj_get(t, "filters");
        yyjson_val *th = yyjson_obj_get(yyjson_obj_get(t, "ranking_options"),
                                        "score_threshold");
        if (!yyjson_is_arr(ids) || !yyjson_arr_size(ids) ||
            (mx && (!yyjson_is_int(mx) || yyjson_get_sint(mx) < 1 ||
                    yyjson_get_sint(mx) > 50))) {
            *ret = srv_reply_error(r, 400, "invalid_request_error",
                                   "invalid_value",
                                   "tools[%zu]: file_search needs "
                                   "'vector_store_ids', and max_num_results "
                                   "from 1 to 50.",
                                   idx);
            return -1;
        }
        size_t i2, m2;
        yyjson_val *v;
        yyjson_arr_foreach(ids, i2, m2, v)
        {
            char **nv =
                realloc(x->mcp->vs, (size_t)(x->mcp->n_vs + 1) * sizeof(*nv));
            if (!nv || !yyjson_is_str(v))
                continue;
            x->mcp->vs = nv;
            x->mcp->vs[x->mcp->n_vs++] = strdup(yyjson_get_str(v));
        }
        /* a few passages a call: they all go into the model's context */
        x->mcp->fs_max = mx ? (int)yyjson_get_sint(mx) : 6;
        if (f && !yyjson_is_null(f))
            x->mcp->fs_filters = yyjson_val_write(f, 0, NULL);
        x->mcp->fs_threshold = th ? yyjson_get_num(th) : 0;
    }
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
    if (files) { /* the files' search, as a function the model calls */
        static const char FILE_TOOL[] =
            "{\"type\":\"function\",\"function\":{\"name\":\"" FILE_NAME
            "\",\"description\":\"Search the user's files for passages "
            "that answer the question. Returns the passages, numbered; cite "
            "them as [1], [2] after the facts they give.\",\"parameters\":"
            "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":"
            "\"string\",\"description\":\"what to look for, in a few "
            "words or a question\"}},\"required\":[\"query\"]}}}";
        yyjson_doc *fd2 = yyjson_read(FILE_TOOL, sizeof(FILE_TOOL) - 1, 0);
        if (!fd2)
            goto oom_in;
        yyjson_mut_arr_append(all,
                              yyjson_val_mut_copy(d, yyjson_doc_get_root(fd2)));
        yyjson_doc_free(fd2);
    }
    if (code) { /* the session, as a function the model calls */
        static const char CODE_TOOL[] =
            "{\"type\":\"function\",\"function\":{\"name\":\"" CODE_NAME
            "\",\"description\":\"Run Python 3 code in a sandboxed session, "
            "as in a Jupyter notebook: variables and imports stay from one "
            "call to the next, the value of the last expression is shown, "
            "and print() output is returned. numpy, scipy, sympy, "
            "matplotlib and PIL are there; there is no internet. The user's "
            "files are in /mnt/data; save there the files you make for the "
            "user (plots drawn with matplotlib are saved there by "
            "themselves). Use it for calculations, data analysis and "
            "charts.\",\"parameters\":{\"type\":\"object\",\"properties\":"
            "{\"code\":{\"type\":\"string\",\"description\":\"the Python "
            "code to run\"}},\"required\":[\"code\"]}}}";
        yyjson_doc *cd2 = yyjson_read(CODE_TOOL, sizeof(CODE_TOOL) - 1, 0);
        if (!cd2)
            goto oom_in;
        yyjson_mut_arr_append(all,
                              yyjson_val_mut_copy(d, yyjson_doc_get_root(cd2)));
        yyjson_doc_free(cd2);
    }
    if (web) { /* the search, as a function the model calls */
        static const char WEB_TOOL[] =
            "{\"type\":\"function\",\"function\":{\"name\":\"" WEB_NAME
            "\",\"description\":\"Search the web (Wikipedia) for facts to "
            "answer with: people, places, history, science, definitions. "
            "Returns the matching pages' first sentences, numbered; cite "
            "them as [1], [2] after the facts they give. Not for news or "
            "today's events.\",\"parameters\":{\"type\":\"object\","
            "\"properties\":{\"query\":{\"type\":\"string\","
            "\"description\":\"a few words, as for a search engine\"},"
            "\"language\":{\"type\":\"string\",\"description\":\"the "
            "Wikipedia to search, as a code: en, it, fr...; the user's "
            "language\"}},\"required\":[\"query\"]}}}";
        yyjson_doc *wd = yyjson_read(WEB_TOOL, sizeof(WEB_TOOL) - 1, 0);
        if (!wd)
            goto oom_in;
        yyjson_mut_arr_append(all,
                              yyjson_val_mut_copy(d, yyjson_doc_get_root(wd)));
        yyjson_doc_free(wd);
    }
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
        /* tool_choice {type: mcp, server_label, name}, {type: web_search} */
        yyjson_val *tc = yyjson_obj_get(root, "tool_choice");
        const char *tt = yyjson_get_str(yyjson_obj_get(tc, "type"));
        if (tt && code && !strcmp(tt, "code_interpreter")) {
            x->j->tool_choice = JANAS_LLM_TOOLS_FUNCTION;
            snprintf(x->j->tool_name, sizeof(x->j->tool_name), "%s", CODE_NAME);
        } else if (tt && files && !strcmp(tt, "file_search")) {
            x->j->tool_choice = JANAS_LLM_TOOLS_FUNCTION;
            snprintf(x->j->tool_name, sizeof(x->j->tool_name), "%s", FILE_NAME);
        } else if (tt && web && !strncmp(tt, "web_search", 10)) {
            x->j->tool_choice = JANAS_LLM_TOOLS_FUNCTION;
            snprintf(x->j->tool_name, sizeof(x->j->tool_name), "%s", WEB_NAME);
        } else if (tt && !strcmp(tt, "mcp")) {
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
        if (x->mcp->web && !strcmp(k->name, WEB_NAME)) {
            run_web(x->mcp, d, arr, k->args, &texts[i]);
            continue;
        }
        if (x->mcp->n_vs && !strcmp(k->name, FILE_NAME)) {
            run_file(x->mcp, d, arr, k->args, &texts[i]);
            continue;
        }
        if (x->mcp->ci[0] && !strcmp(k->name, CODE_NAME)) {
            run_code(x->mcp, d, arr, k->args, &texts[i]);
            continue;
        }
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
