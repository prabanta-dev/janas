/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_chatfile.c - conversations in files (see include/janas/llm.h,
 * janas_llm_chat_save): saved as they go, one file each, taken up again,
 * exported in Markdown or plain text and read back from either.
 *
 * The saved form is JSON Lines: a first line that says what the file is
 * ({"janas_chat":1, the title, when it began, the model, the settings}),
 * then one line a message ({"role", "text", "reasoning", "calls", "at"}).
 * The whole file is written again at each message, to a file beside it
 * renamed over it: a conversation is never left half written.
 *
 * The Markdown has a line of its own before each message, an HTML comment
 * that does not show (<!-- janas:user -->), then a heading; reading it
 * back goes by those lines, or by the headings alone (## User, ##
 * Assistant) in a file written by hand.
 */
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>

#include "common/catalog.h"
#include "llm/chat.h"

static const char *const ROLES[] = {"system", "user", "assistant", "tool"};
static const char *const HEADS[] = {"System", "User", "Assistant", "Tool"};

static int role_of(const char *s, size_t n)
{
    for (int r = 0; r < 4; r++)
        if (strlen(ROLES[r]) == n && !strncasecmp(s, ROLES[r], n))
            return r;
    return -1;
}

static int dir_default(char *buf, size_t len)
{
    const char *x = getenv("XDG_DATA_HOME"), *h = getenv("HOME");
    int n;
    if (x && *x)
        n = snprintf(buf, len, "%s/janas/chats", x);
    else if (h && *h)
        n = snprintf(buf, len, "%s/.local/share/janas/chats", h);
    else
        return -1;
    return n > 0 && (size_t)n < len ? 0 : -1;
}

int32_t janas_llm_chats_dir(char *buf, int32_t cap, int32_t *len)
{
    char d[4096];
    if (dir_default(d, sizeof d) != 0)
        return janas_api_fail(JANAS_LLM_EFAIL, "no home directory");
    return janas_api_copy_out(d, strlen(d), buf, cap, len);
}

/* The model's name, or its file's when it has none. */
static const char *model_name(const janas_llm_chat *c)
{
    const janas_llm *llm = c->llm;
    if (llm->name[0])
        return llm->name;
    const char *p = llm->path ? llm->path : "";
    const char *b = strrchr(p, '/');
#ifdef _WIN32
    const char *w = strrchr(p, '\\');
    if (w && (!b || w > b))
        b = w;
#endif
    return b ? b + 1 : p;
}

static void put_str(struct janas_buf *b, const char *key, const char *s)
{
    janas_buf_printf(b, ",\"%s\":", key);
    janas_json_write_str(b, s ? s : "", s ? strlen(s) : 0);
}

static void write_jsonl(const janas_llm_chat *c, struct janas_buf *b)
{
    janas_buf_puts(b, "{\"janas_chat\":1");
    put_str(b, "title", c->tr_title);
    if (c->tr_titled)
        janas_buf_puts(b, ",\"titled\":true");
    janas_buf_printf(b, ",\"created\":%lld", (long long)c->tr_created);
    put_str(b, "model", model_name(c));
    put_str(b, "model_file", c->llm->path);
    const struct janas_llm_chat_params *p = &c->p;
    janas_buf_printf(b,
                     ",\"settings\":{\"temperature\":%g,\"top_k\":%d,"
                     "\"top_p\":%g,\"min_p\":%g,\"thinking\":%d,"
                     "\"max_reply\":%d}}\n",
                     (double)p->temperature, (int)p->top_k, (double)p->top_p,
                     (double)p->min_p, (int)p->thinking, (int)p->max_reply);
    for (size_t i = 0; i < c->n_tr; i++) {
        const struct tmsg *m = &c->tr[i];
        janas_buf_printf(b, "{\"role\":\"%s\"", ROLES[m->role & 3]);
        put_str(b, "text", m->text);
        if (m->reasoning)
            put_str(b, "reasoning", m->reasoning);
        if (m->n_calls) {
            janas_buf_puts(b, ",\"calls\":[");
            for (size_t k = 0; k < m->n_calls; k++) {
                janas_buf_puts(b, k ? ",{\"name\":" : "{\"name\":");
                janas_json_write_str(b, m->calls[k].name,
                                     strlen(m->calls[k].name));
                put_str(b, "arguments", m->calls[k].args);
                janas_buf_puts(b, "}");
            }
            janas_buf_puts(b, "]");
        }
        janas_buf_printf(b, ",\"at\":%lld}\n", (long long)m->at);
    }
}

/* Bytes into path, by way of a file beside it renamed over it. */
static int32_t write_file(const char *path, const char *s, size_t n)
{
    size_t pn = strlen(path);
    char *tmp = malloc(pn + 8);
    if (!tmp)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    snprintf(tmp, pn + 8, "%s.part", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        int32_t rc = janas_api_fail(JANAS_LLM_EFAIL, "cannot write %s: %s", tmp,
                                    strerror(errno));
        free(tmp);
        return rc;
    }
    int bad = fwrite(s, 1, n, f) != n;
    bad |= fclose(f) != 0;
    if (bad || rename(tmp, path) != 0) {
        int32_t rc = janas_api_fail(JANAS_LLM_EFAIL, "cannot write %s: %s",
                                    path, strerror(errno));
        remove(tmp);
        free(tmp);
        return rc;
    }
    free(tmp);
    return JANAS_LLM_OK;
}

static int has_question(const janas_llm_chat *c)
{
    for (size_t i = 0; i < c->n_tr; i++)
        if (c->tr[i].role == JANAS_LLM_ROLE_USER)
            return 1;
    return 0;
}

/* A file of its own in the folder, named after the time it began. */
static int new_path(janas_llm_chat *c)
{
    time_t t = (time_t)(c->tr_created ? c->tr_created : time(NULL));
    struct tm tm;
    localtime_r(&t, &tm);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%Y-%m-%d-%H%M%S", &tm);
    size_t n = strlen(c->tr_dir) + 48;
    char *p = malloc(n);
    if (!p)
        return -1;
    struct stat st;
    for (int k = 1; k < 1000; k++) {
        if (k == 1)
            snprintf(p, n, "%s/%s.jsonl", c->tr_dir, stamp);
        else
            snprintf(p, n, "%s/%s-%d.jsonl", c->tr_dir, stamp, k);
        if (stat(p, &st) != 0) {
            c->tr_path = p;
            return 0;
        }
    }
    free(p);
    return -1;
}

void janas_api_tr_saved(janas_llm_chat *c)
{
    if (!c->tr_dir || !has_question(c))
        return;
    if (!c->tr_path && new_path(c) != 0)
        return;
    struct janas_buf b = {0};
    write_jsonl(c, &b);
    if (!b.oom)
        write_file(c->tr_path, b.p, b.n);
    janas_buf_free(&b);
}

int32_t janas_llm_chat_autosave(janas_llm_chat *c, const char *dir)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    free(c->tr_dir);
    c->tr_dir = NULL;
    if (!dir)
        return JANAS_LLM_OK;
    char d[4096];
    if (!*dir && dir_default(d, sizeof d) != 0)
        return janas_api_fail(JANAS_LLM_EFAIL, "no home directory");
    if (*dir)
        snprintf(d, sizeof d, "%s", dir);
    if (janas_mkdirs(d) != 0)
        return janas_api_fail(JANAS_LLM_EFAIL, "cannot make %s", d);
    if (!(c->tr_dir = strdup(d)))
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    /* a file in another folder is left: the next one goes here */
    if (c->tr_path && strncmp(c->tr_path, d, strlen(d)) != 0) {
        free(c->tr_path);
        c->tr_path = NULL;
    }
    janas_api_tr_settle(c);
    janas_api_tr_saved(c);
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_file(janas_llm_chat *c, char *buf, int32_t cap,
                            int32_t *len)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    const char *p = c->tr_path ? c->tr_path : "";
    return janas_api_copy_out(p, strlen(p), buf, cap, len);
}

int32_t janas_llm_chat_save(janas_llm_chat *c, const char *path)
{
    if (!c || !path || !*path)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    janas_api_tr_settle(c);
    struct janas_buf b = {0};
    write_jsonl(c, &b);
    int32_t rc = b.oom ? janas_api_fail(JANAS_LLM_ENOMEM, "out of memory")
                       : write_file(path, b.p, b.n);
    janas_buf_free(&b);
    return rc;
}

/* Export. */

static void when(struct janas_buf *b, int64_t at)
{
    time_t t = (time_t)at;
    struct tm tm;
    localtime_r(&t, &tm);
    char s[32];
    strftime(s, sizeof s, "%Y-%m-%d %H:%M", &tm);
    janas_buf_puts(b, s);
}

/* A fence longer than any run of backquotes in s. */
static void fence(struct janas_buf *b, const char *s)
{
    size_t most = 0, run = 0;
    for (; *s; s++) {
        run = *s == '`' ? run + 1 : 0;
        if (run > most)
            most = run;
    }
    for (size_t k = 0; k < (most < 3 ? 3 : most + 1); k++)
        janas_buf_put(b, "`", 1);
}

static void export_md(const janas_llm_chat *c, struct janas_buf *b)
{
    janas_buf_printf(b, "# %s\n\n",
                     c->tr_title && *c->tr_title ? c->tr_title
                                                 : "Conversation");
    janas_buf_puts(b, "*");
    when(b, c->tr_created ? c->tr_created : (int64_t)time(NULL));
    janas_buf_printf(b, ", %s, with Janas-LLM.*\n", model_name(c));
    for (size_t i = 0; i < c->n_tr; i++) {
        const struct tmsg *m = &c->tr[i];
        janas_buf_printf(b, "\n<!-- janas:%s -->\n## %s\n\n",
                         ROLES[m->role & 3], HEADS[m->role & 3]);
        if (m->reasoning && *m->reasoning)
            janas_buf_printf(b,
                             "<details><summary>Reasoning</summary>\n\n%s\n\n"
                             "</details>\n\n",
                             m->reasoning);
        if (*m->text)
            janas_buf_printf(b, "%s\n", m->text);
        for (size_t k = 0; k < m->n_calls; k++) {
            janas_buf_printf(b, "\n<!-- janas:call %s -->\n**Calls** `%s`\n\n",
                             m->calls[k].name, m->calls[k].name);
            fence(b, m->calls[k].args);
            janas_buf_printf(b, "json\n%s\n", m->calls[k].args);
            fence(b, m->calls[k].args);
            janas_buf_puts(b, "\n");
        }
    }
}

static void export_text(const janas_llm_chat *c, struct janas_buf *b)
{
    const char *t = c->tr_title && *c->tr_title ? c->tr_title : "Conversation";
    janas_buf_printf(b, "%s\n", t);
    when(b, c->tr_created ? c->tr_created : (int64_t)time(NULL));
    janas_buf_printf(b, ", %s, with Janas-LLM.\n", model_name(c));
    for (size_t i = 0; i < c->n_tr; i++) {
        const struct tmsg *m = &c->tr[i];
        janas_buf_printf(b, "\n[%s]\n", ROLES[m->role & 3]);
        if (*m->text)
            janas_buf_printf(b, "%s\n", m->text);
        for (size_t k = 0; k < m->n_calls; k++)
            janas_buf_printf(b, "[call %s]\n%s\n", m->calls[k].name,
                             m->calls[k].args);
    }
}

int32_t janas_llm_chat_export(janas_llm_chat *c, const char *path,
                              int32_t format)
{
    if (!c || !path || !*path ||
        (format != JANAS_LLM_EXPORT_MARKDOWN &&
         format != JANAS_LLM_EXPORT_TEXT))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    janas_api_tr_settle(c);
    struct janas_buf b = {0};
    if (format == JANAS_LLM_EXPORT_MARKDOWN)
        export_md(c, &b);
    else
        export_text(c, &b);
    int32_t rc = b.oom ? janas_api_fail(JANAS_LLM_ENOMEM, "out of memory")
                       : write_file(path, b.p, b.n);
    janas_buf_free(&b);
    return rc;
}

/* Reading back. */

static char *read_all(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        janas_api_fail(JANAS_LLM_EOPEN, "cannot open %s: %s", path,
                       strerror(errno));
        return NULL;
    }
    struct janas_buf b = {0};
    char tmp[65536];
    size_t k;
    while ((k = fread(tmp, 1, sizeof tmp, f)) > 0 && b.n < (512u << 20))
        janas_buf_put(&b, tmp, k);
    fclose(f);
    janas_buf_put(&b, "", 0);
    if (b.oom) {
        janas_buf_free(&b);
        janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
        return NULL;
    }
    *n = b.n;
    return b.p ? b.p : calloc(1, 1);
}

static char *str_of(const struct janas_json *v)
{
    if (!v || v->type != JANAS_JSON_STRING)
        return NULL;
    char *d = malloc(v->n + 1);
    if (d) {
        memcpy(d, v->s, v->n);
        d[v->n] = 0;
    }
    return d;
}

/* One line of the saved form: the header (*header set), or a message. */
static int read_line(janas_llm_chat *c, const char *s, size_t n, int *header)
{
    char err[128];
    struct janas_json_doc *d = janas_json_parse(s, n, err, sizeof err);
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_root(d);
    int rc = 0;
    if (janas_json_get(o, "janas_chat")) {
        *header = 1;
        free(c->tr_title);
        c->tr_title = str_of(janas_json_get(o, "title"));
        c->tr_titled = janas_json_get(o, "titled") &&
                       janas_json_get(o, "titled")->type == JANAS_JSON_TRUE;
        c->tr_created =
            (int64_t)janas_json_num(janas_json_get(o, "created"), 0);
    } else {
        const struct janas_json *r = janas_json_get(o, "role");
        int role = r && r->type == JANAS_JSON_STRING ? role_of(r->s, r->n) : -1;
        const struct janas_json *t = janas_json_get(o, "text");
        struct tmsg *m = NULL;
        if (role >= 0 && t && t->type == JANAS_JSON_STRING)
            m = janas_api_tr_push(c, role, t->s, t->n);
        if (!m) {
            rc = -1;
        } else {
            m->reasoning = str_of(janas_json_get(o, "reasoning"));
            int64_t at = (int64_t)janas_json_num(janas_json_get(o, "at"), 0);
            if (at)
                m->at = at;
            const struct janas_json *cl = janas_json_get(o, "calls");
            for (const struct janas_json *k = cl ? cl->child : NULL; k;
                 k = k->next) {
                const struct janas_json *nm = janas_json_get(k, "name"),
                                        *a = janas_json_get(k, "arguments");
                if (nm && nm->type == JANAS_JSON_STRING && a &&
                    a->type == JANAS_JSON_STRING)
                    janas_api_tr_call(m, nm->s, nm->n, a->s, a->n);
            }
        }
    }
    janas_json_free(d);
    return rc;
}

static int read_jsonl(janas_llm_chat *c, const char *s, size_t n)
{
    int header = 0;
    for (size_t i = 0; i < n;) {
        size_t e = i;
        while (e < n && s[e] != '\n')
            e++;
        size_t k = i;
        while (k < e && (unsigned char)s[k] <= ' ')
            k++;
        if (k < e && read_line(c, s + k, e - k, &header) != 0)
            return -1;
        i = e + 1;
    }
    return header ? 0 : -1;
}

/* A line of s at i (to e, without its newline): the next one at *next. */
static size_t line_end(const char *s, size_t n, size_t i, size_t *next)
{
    size_t e = i;
    while (e < n && s[e] != '\n')
        e++;
    *next = e < n ? e + 1 : n;
    if (e > i && s[e - 1] == '\r')
        e--;
    return e;
}

/* The role a line of the Markdown starts: by its marker, or (heads) by a
   heading alone; -1 when it starts none. */
static int md_role(const char *l, size_t n, int heads)
{
    if (!heads) {
        static const char pre[] = "<!-- janas:";
        size_t pn = sizeof pre - 1;
        if (n < pn + 4 || strncmp(l, pre, pn) || strncmp(l + n - 4, " -->", 4))
            return -1;
        return role_of(l + pn, n - pn - 4);
    }
    if (n < 4 || strncmp(l, "## ", 3))
        return -1;
    size_t k = n;
    while (k > 3 && (unsigned char)l[k - 1] <= ' ')
        k--;
    return role_of(l + 3, k - 3);
}

/* The text of a message, trimmed of the blank lines around it, with its
   reasoning (<details> first) and its calls taken out. */
static int md_message(janas_llm_chat *c, int role, const char *s, size_t n)
{
    struct janas_buf text = {0}, why = {0};
    struct janas_buf names = {0}, args = {0}; /* NUL-separated */
    size_t n_calls = 0;
    int in_why = 0;
    for (size_t i = 0, next; i < n; i = next) {
        size_t e = line_end(s, n, i, &next);
        const char *l = s + i;
        size_t ln = e - i;
        static const char det[] = "<details><summary>Reasoning</summary>";
        static const char call[] = "<!-- janas:call ";
        if (!in_why && !why.n && ln == sizeof det - 1 && !memcmp(l, det, ln)) {
            in_why = 1;
            continue;
        }
        if (in_why) {
            if (ln == 10 && !memcmp(l, "</details>", 10))
                in_why = 0;
            else
                janas_buf_put(&why, s + i, next - i);
            continue;
        }
        if (ln > sizeof call + 3 && !memcmp(l, call, sizeof call - 1) &&
            !memcmp(l + ln - 4, " -->", 4)) {
            /* the name, then (after a line of its own) a fenced block */
            janas_buf_put(&names, l + sizeof call - 1,
                          ln - (sizeof call - 1) - 4);
            janas_buf_put(&names, "", 1);
            size_t f = next, fl = 0;
            while (f < n) { /* to the fence's opening */
                size_t fe = line_end(s, n, f, &next);
                if (fe > f && s[f] == '`') {
                    while (f + fl < fe && s[f + fl] == '`')
                        fl++;
                    f = next;
                    break;
                }
                f = next;
            }
            size_t a0 = f;
            while (f < n) { /* to its closing */
                size_t fe = line_end(s, n, f, &next);
                size_t k = 0;
                while (f + k < fe && s[f + k] == '`')
                    k++;
                if (k >= fl && f + k == fe && fl)
                    break;
                f = next;
            }
            size_t a1 = f > a0 ? f - 1 : a0; /* the newline before it */
            if (a1 > a0 && s[a1 - 1] == '\r')
                a1--;
            janas_buf_put(&args, s + a0, a1 - a0);
            janas_buf_put(&args, "", 1);
            n_calls++;
            continue;
        }
        janas_buf_put(&text, s + i, next - i);
    }
    /* blank lines around, off */
    struct janas_buf *parts[2] = {&text, &why};
    for (int p = 0; p < 2; p++) {
        struct janas_buf *b = parts[p];
        size_t a = 0;
        while (a < b->n && (b->p[a] == '\n' || b->p[a] == '\r'))
            a++;
        size_t z = b->n;
        while (z > a && (unsigned char)b->p[z - 1] <= ' ')
            z--;
        if (b->p) {
            memmove(b->p, b->p + a, z - a);
            b->n = z - a;
        }
    }
    int rc = -1;
    struct tmsg *m = text.oom || why.oom || names.oom || args.oom
                         ? NULL
                         : janas_api_tr_push(c, role, text.p, text.n);
    if (m) {
        rc = 0;
        if (why.n && (m->reasoning = malloc(why.n + 1))) {
            memcpy(m->reasoning, why.p, why.n);
            m->reasoning[why.n] = 0;
        }
        const char *nm = names.p, *ag = args.p;
        for (size_t k = 0; k < n_calls; k++) {
            janas_api_tr_call(m, nm, strlen(nm), ag, strlen(ag));
            nm += strlen(nm) + 1;
            ag += strlen(ag) + 1;
        }
    }
    janas_buf_free(&text);
    janas_buf_free(&why);
    janas_buf_free(&names);
    janas_buf_free(&args);
    return rc;
}

static int read_md(janas_llm_chat *c, const char *s, size_t n)
{
    /* the markers when there are any, the headings otherwise */
    int heads = 1;
    for (size_t i = 0, next; i < n && heads; i = next) {
        size_t e = line_end(s, n, i, &next);
        heads = md_role(s + i, e - i, 0) < 0;
    }
    int role = -1;
    size_t body = 0;
    for (size_t i = 0, next; i <= n; i = next) {
        size_t e = i < n ? line_end(s, n, i, &next) : (next = n + 1, n);
        int r = i < n ? md_role(s + i, e - i, heads) : -1;
        if (i == n || r >= 0) {
            if (role >= 0 && md_message(c, role, s + body, i - body) != 0)
                return -1;
            if (i == n)
                break;
            role = r;
            body = next;
            if (!heads && body < n) { /* the heading under the marker */
                size_t nx, he = line_end(s, n, body, &nx);
                if (he - body >= 3 && !strncmp(s + body, "## ", 3))
                    body = nx;
            }
        } else if (role < 0 && !c->tr_title && e - i > 2 &&
                   !strncmp(s + i, "# ", 2)) {
            c->tr_title = malloc(e - i - 1);
            if (c->tr_title) {
                memcpy(c->tr_title, s + i + 2, e - i - 2);
                c->tr_title[e - i - 2] = 0;
                c->tr_titled = 1;
            }
        }
    }
    return c->n_tr ? 0 : -1;
}

static int read_text(janas_llm_chat *c, const char *s, size_t n)
{
    int role = -1;
    size_t body = 0;
    struct tmsg *last = NULL;
    for (size_t i = 0, next; i <= n; i = next) {
        size_t e = i < n ? line_end(s, n, i, &next) : (next = n + 1, n);
        const char *l = s + i;
        size_t ln = e - i;
        int r = -1, call = 0;
        if (i < n && ln > 2 && l[0] == '[' && l[ln - 1] == ']') {
            r = role_of(l + 1, ln - 2);
            call = ln > 7 && !strncmp(l, "[call ", 6);
        }
        if (i < n && r < 0 && !call) {
            if (role < 0 && i == 0 && ln && !c->tr_title &&
                (c->tr_title = malloc(ln + 1))) {
                memcpy(c->tr_title, l, ln);
                c->tr_title[ln] = 0;
                c->tr_titled = 1;
            }
            continue;
        }
        /* the part before ends here */
        size_t z = i;
        while (z > body && (unsigned char)s[z - 1] <= ' ')
            z--;
        size_t a = body;
        while (a < z && (s[a] == '\n' || s[a] == '\r'))
            a++;
        if (role >= 0 && !(last = janas_api_tr_push(c, role, s + a, z - a)))
            return -1;
        if (i == n)
            break;
        if (call && last && last->role == JANAS_LLM_ROLE_ASSISTANT) {
            /* the arguments: the lines up to the next marker */
            size_t f = next, g = next;
            while (g < n) {
                size_t nx, ge = line_end(s, n, g, &nx);
                if (ge - g > 2 && s[g] == '[' && s[ge - 1] == ']' &&
                    (role_of(s + g + 1, ge - g - 2) >= 0 ||
                     !strncmp(s + g, "[call ", 6)))
                    break;
                g = nx;
            }
            size_t gz = g;
            while (gz > f && (unsigned char)s[gz - 1] <= ' ')
                gz--;
            janas_api_tr_call(last, l + 6, ln - 7, s + f, gz - f);
            next = g;
            role = -1;
            body = g;
            continue;
        }
        role = r;
        body = next;
    }
    return c->n_tr ? 0 : -1;
}

int janas_api_tr_read(janas_llm_chat *c, const char *s, size_t n,
                      const char *path, int *saved_form)
{
    size_t k = 0;
    while (k < n && (unsigned char)s[k] <= ' ')
        k++;
    int saved = k < n && s[k] == '{' && strstr(s + k, "\"janas_chat\"");
    size_t pn = strlen(path);
    int md = pn > 3 && (!strcasecmp(path + pn - 3, ".md") ||
                        (pn > 9 && !strcasecmp(path + pn - 9, ".markdown")));
    int bad = saved ? read_jsonl(c, s, n)
              : md  ? read_md(c, s, n)
                    : read_text(c, s, n);
    if (bad && !saved && !md) { /* text that is Markdown after all */
        janas_api_tr_clear(c);
        bad = read_md(c, s, n);
    }
    if (bad) {
        janas_api_tr_clear(c);
        return -1;
    }
    if (!c->tr_created)
        c->tr_created = c->tr[0].at;
    if (!c->tr_title) { /* from its first question */
        c->tr_titled = 0;
        janas_api_tr_autotitle(c);
    }
    *saved_form = saved;
    return 0;
}

int32_t janas_llm_chat_restore(janas_llm_chat *c, const char *path)
{
    if (!c || !path)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    if (c->replying && !c->ended)
        return janas_api_fail(JANAS_LLM_EBUSY, "a reply is being read");
    size_t n;
    char *s = read_all(path, &n);
    if (!s)
        return JANAS_LLM_EOPEN;
    janas_llm_chat_reset(c); /* the conversation before is over */
    int saved = 0, bad = janas_api_tr_read(c, s, n, path, &saved);
    free(s);
    if (bad)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "%s: no conversation found in it", path);
    /* the system message is the chat's, until the caller gives another */
    if (c->n_tr && c->tr[0].role == JANAS_LLM_ROLE_SYSTEM &&
        !(c->system = strdup(c->tr[0].text))) {
        janas_api_tr_clear(c);
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    }
    c->tr_pending = c->n_tr > (c->system ? 1u : 0u);
    /* saved as it goes: in the same file when it is one of the folder's */
    if (c->tr_dir && saved && !strncmp(path, c->tr_dir, strlen(c->tr_dir)))
        c->tr_path = strdup(path);
    janas_api_tr_saved(c);
    return JANAS_LLM_OK;
}

/* The folder's conversations. */

struct entry {
    char *path;
    int64_t t;
};

static int newest_first(const void *a, const void *b)
{
    int64_t x = ((const struct entry *)a)->t, y = ((const struct entry *)b)->t;
    return x < y ? 1 : x > y ? -1 : 0;
}

int32_t janas_llm_chats_list(const char *dir, char *buf, int32_t cap,
                             int32_t *len)
{
    char d[4096];
    if ((!dir || !*dir) && dir_default(d, sizeof d) != 0)
        return janas_api_fail(JANAS_LLM_EFAIL, "no home directory");
    if (dir && *dir)
        snprintf(d, sizeof d, "%s", dir);
    struct entry *v = NULL;
    size_t nv = 0, cv = 0;
    int32_t rc = JANAS_LLM_OK;
    DIR *dd = opendir(d);
    struct dirent *e;
    while (dd && (e = readdir(dd))) {
        size_t n = strlen(e->d_name);
        if (n < 7 || strcmp(e->d_name + n - 6, ".jsonl"))
            continue;
        if (nv == cv) {
            cv = cv ? 2 * cv : 64;
            struct entry *t = realloc(v, cv * sizeof(*t));
            if (!t) {
                rc = janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
                break;
            }
            v = t;
        }
        size_t pn = strlen(d) + n + 2;
        char *p = malloc(pn);
        struct stat st;
        if (!p) {
            rc = janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
            break;
        }
        snprintf(p, pn, "%s/%s", d, e->d_name);
        if (stat(p, &st) != 0) {
            free(p);
            continue;
        }
        v[nv++] = (struct entry){.path = p, .t = (int64_t)st.st_mtime};
    }
    if (dd)
        closedir(dd);
    if (nv)
        qsort(v, nv, sizeof(*v), newest_first);
    struct janas_buf b = {0};
    for (size_t i = 0; i < nv; i++) {
        janas_buf_printf(&b, "%s\n", v[i].path);
        free(v[i].path);
    }
    free(v);
    if (rc == JANAS_LLM_OK)
        rc = b.oom ? janas_api_fail(JANAS_LLM_ENOMEM, "out of memory")
                   : janas_api_copy_out(b.p ? b.p : "", b.n, buf, cap, len);
    janas_buf_free(&b);
    return rc;
}

/* A string into a buffer of cap bytes, cut at a whole character. */
static void put_cut(const char *s, size_t n, char *buf, int32_t cap)
{
    if (!buf || cap <= 0)
        return;
    if (n >= (size_t)cap) {
        n = (size_t)cap - 1;
        while (n > 0 && ((unsigned char)s[n] & 0xC0) == 0x80)
            n--;
    }
    memcpy(buf, s, n);
    buf[n] = 0;
}

int32_t janas_llm_chat_file_info(const char *path, char *title,
                                 int32_t title_cap, char *model,
                                 int32_t model_cap, char *model_file,
                                 int32_t file_cap, int64_t *updated,
                                 int32_t *messages)
{
    if (!path)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    put_cut("", 0, title, title_cap);
    put_cut("", 0, model, model_cap);
    put_cut("", 0, model_file, file_cap);
    size_t n;
    char *s = read_all(path, &n);
    if (!s)
        return JANAS_LLM_EOPEN;
    size_t e = 0;
    while (e < n && s[e] != '\n')
        e++;
    char err[128];
    struct janas_json_doc *d = janas_json_parse(s, e, err, sizeof err);
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    if (!o || !janas_json_get(o, "janas_chat")) {
        janas_json_free(d);
        free(s);
        return janas_api_fail(JANAS_LLM_EINVAL, "%s: not a saved conversation",
                              path);
    }
    const struct janas_json *t = janas_json_get(o, "title"),
                            *m = janas_json_get(o, "model"),
                            *mf = janas_json_get(o, "model_file");
    if (t && t->type == JANAS_JSON_STRING)
        put_cut(t->s, t->n, title, title_cap);
    if (m && m->type == JANAS_JSON_STRING)
        put_cut(m->s, m->n, model, model_cap);
    if (mf && mf->type == JANAS_JSON_STRING)
        put_cut(mf->s, mf->n, model_file, file_cap);
    janas_json_free(d);
    if (messages) { /* a line a message, the system's not counted */
        int32_t k = 0;
        for (size_t i = e; i < n; i++)
            if (s[i] == '\n' && i + 1 < n && s[i + 1] == '{' &&
                strncmp(s + i + 1, "{\"role\":\"system\"", 16))
                k++;
        *messages = k;
    }
    free(s);
    struct stat st;
    if (updated)
        *updated = stat(path, &st) == 0 ? (int64_t)st.st_mtime : 0;
    return JANAS_LLM_OK;
}
