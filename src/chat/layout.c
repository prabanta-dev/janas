/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layout.c - the layouts of the services' answers in janas-chat (see
 * layout.h).
 */
#define _GNU_SOURCE /* memmem, strndup */
#include "layout.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "services/common/template.h"
#include "glossary.h"

#define META "dev.prabanta.janas/layout"

static char lang[16] = "en";
static int (*translator)(const char *, const char *, const char *, char **);

void layout_set_lang(const char *l)
{
    snprintf(lang, sizeof lang, "%s", l && *l ? l : "en");
}

const char *layout_lang(void)
{
    return lang;
}

void layout_set_translator(int (*fn)(const char *, const char *, const char *,
                                     char **))
{
    translator = fn;
}

/* The language's name, for the request. */
static const char *lang_name(const char *code)
{
    static const char *const names[][2] = {
        {"it", "Italian"},  {"fr", "French"},     {"de", "German"},
        {"es", "Spanish"},  {"pt", "Portuguese"}, {"nl", "Dutch"},
        {"ca", "Catalan"},  {"sc", "Sardinian"},  {"pl", "Polish"},
        {"ro", "Romanian"}, {"el", "Greek"},      {"sv", "Swedish"},
    };
    for (size_t i = 0; i < sizeof names / sizeof *names; i++)
        if (strcmp(names[i][0], code) == 0)
            return names[i][1];
    return code;
}

/* FNV-1a: the English layout's mark in the file's name, so that a layout
   changed by a new version of the service is translated again */
static uint64_t mark(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++)
        h = (h ^ (unsigned char)*s) * 1099511628211ull;
    return h;
}

static int dir_of(char *buf, size_t cap)
{
    const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    int n = xdg && *xdg ? snprintf(buf, cap, "%s/janas", xdg)
            : home      ? snprintf(buf, cap, "%s/.config/janas", home)
                        : -1;
    if (n < 0 || (size_t)n >= cap)
        return -1;
    if (mkdir(buf, 0700) != 0 && errno != EEXIST)
        return -1;
    if ((size_t)n + 9 >= cap)
        return -1;
    strcat(buf, "/layouts");
    if (mkdir(buf, 0700) != 0 && errno != EEXIST)
        return -1;
    return 0;
}

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    struct janas_buf b = {0};
    char tmp[4096];
    size_t n;
    while ((n = fread(tmp, 1, sizeof tmp, f)) > 0)
        janas_buf_put(&b, tmp, n);
    fclose(f);
    janas_buf_put(&b, "", 1);
    if (b.oom) {
        janas_buf_free(&b);
        return NULL;
    }
    return b.p;
}

/* The model's answer as a layout: without the fence of a code block, and
   without blank lines around it. */
static void unwrap(char *s)
{
    char *p = s;
    while (*p == '\n' || *p == ' ')
        p++;
    if (strncmp(p, "```", 3) == 0) {
        char *nl = strchr(p, '\n');
        p = nl ? nl + 1 : p + 3;
        char *end = strstr(p, "\n```");
        if (end)
            *end = 0;
    }
    memmove(s, p, strlen(p) + 1);
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == ' '))
        s[--n] = 0;
}

/* The languages that write a decimal comma. */
static int comma_lang(const char *l)
{
    static const char *const c[] = {
        "it", "fr", "de", "es", "pt", "nl", "ca", "sc", "pl", "ro",
        "el", "sv", "da", "fi", "no", "nb", "cs", "sk", "hu", "ru",
        "uk", "tr", "hr", "sl", "sr", "bg", "lt", "lv", "et", "id"};
    for (size_t i = 0; i < sizeof c / sizeof *c; i++)
        if (strcmp(c[i], l) == 0)
            return 1;
    return 0;
}

/*
 * The model never sees a tag or a key: asked to keep them, Qwen3.6-35B-A3B
 * still wrote dir.NNN for dir.NNW and {{sst}} for {{sea}} in two layouts
 * of the weather. It gets the text with each tag as a numbered marker
 * ({7}, as in the message files programs are translated with) and the
 * words' texts numbered one a line; the tags and the keys are put back
 * here.
 */
static const char SYSTEM[] =
    "You translate the texts of a program. The text holds markers such as "
    "{7}: keep every marker exactly as it is, once each, where the sense "
    "puts it; never translate, change, add or drop one. Translate all the "
    "text between them, however short (\", by \", \", comments: \"). "
    "After the line "
    "\"=== words\" come words and short phrases, one a line, each after "
    "its number and a colon (\"12: north-east\"): translate what follows "
    "the colon, keeping the numbers and the lines; what is in square "
    "brackets says what the word is (a term of a scale: give the "
    "language's own term) and is not to be written. Compass points as the "
    "language abbreviates them (in Italian W is O and SW is SO). Answer "
    "with the translated text, the line \"=== words\" and the words, "
    "nothing before or after.";

struct coded {
    struct janas_buf text; /* for the model */
    char **tags;           /* {N}: tags[N - 1] */
    char **gaps;           /* gaps[N - 1]: the spaces and lines after {N} */
    char *to_tag;          /* to_tag[N - 1]: {N + 1} comes right after */
    size_t n_tags;
    char **keys; /* word N: keys[N - 1], and its English text */
    char **english;
    size_t n_keys, n_english;
};

static void coded_free(struct coded *c)
{
    janas_buf_free(&c->text);
    for (size_t i = 0; i < c->n_tags; i++) {
        free(c->tags[i]);
        free(c->gaps ? c->gaps[i] : NULL);
    }
    free(c->to_tag);
    for (size_t i = 0; i < c->n_keys; i++)
        free(c->keys[i]);
    for (size_t i = 0; i < c->n_english; i++)
        free(c->english[i]);
    free(c->english);
    free(c->tags);
    free(c->gaps);
    free(c->keys);
}

static int push(char ***v, size_t *n, const char *s, size_t len)
{
    char **g = realloc(*v, (*n + 1) * sizeof **v);
    if (!g)
        return -1;
    *v = g;
    if (!(g[*n] = malloc(len + 1)))
        return -1;
    memcpy(g[*n], s, len);
    g[*n][len] = 0;
    (*n)++;
    return 0;
}

/* What a word is, for the model to find the language's own term: the
   names of the scales are terms of art (Beaufort 1 is "bava di vento" in
   Italian, not a word for "light air"). Left out of the answer. */
static const char *hint(const char *key, size_t n)
{
    static char h[96];
    static const struct {
        const char *prefix, *what;
    } k[] = {{"bft.", "Beaufort wind force"},
             {"sea.", "Douglas sea state"},
             {"wmo.", "WMO weather code"},
             {"cover.", "METAR cloud cover"},
             {"wx.", "METAR present weather"},
             {"dir.", "compass point, abbreviated"},
             {"from.", "wind direction"},
             {"wd.", "day of the week"},
             {"status.", "the state of one issue or pull request"},
             {"kind.", "a plural, after a number"},
             {"change.", "what happened to one file"},
             {"done.", "what a git command did"},
             {"mon.", "month"}};
    for (size_t i = 0; i < sizeof k / sizeof *k; i++) {
        size_t pn = strlen(k[i].prefix);
        if (n > pn && strncmp(key, k[i].prefix, pn) == 0) {
            snprintf(h, sizeof h, "   [%s %.*s]", k[i].what, (int)(n - pn),
                     key + pn);
            return h;
        }
    }
    return "";
}

/* The layout without its tags and keys, for the model. */
static int encode(const char *en, struct coded *c)
{
    memset(c, 0, sizeof *c);
    const char *sep = strstr(en, "\n---\n");
    size_t ln = sep ? (size_t)(sep - en) : strlen(en);
    for (size_t i = 0; i < ln;) {
        const char *open = memmem(en + i, ln - i, "{{", 2);
        size_t upto = open ? (size_t)(open - en) : ln;
        janas_buf_put(&c->text, en + i, upto - i);
        if (!open)
            break;
        const char *close = memmem(open, ln - upto, "}}", 2);
        if (!close)
            return -1;
        size_t end = (size_t)(close - en) + 2;
        size_t nt = c->n_tags;
        char **g = realloc(c->gaps, (nt + 1) * sizeof *g);
        char *t = g ? realloc(c->to_tag, nt + 1) : NULL;
        if (g)
            c->gaps = g;
        if (!t)
            return -1;
        c->to_tag = t;
        g[nt] = NULL;
        if (push(&c->tags, &c->n_tags, open, end - upto) != 0)
            return -1;
        size_t sp = end;
        while (sp < ln && (en[sp] == ' ' || en[sp] == '\n'))
            sp++;
        t[nt] = sp + 1 < ln && en[sp] == '{' && en[sp + 1] == '{';
        if (!(g[nt] = strndup(en + end, sp - end)))
            return -1;
        janas_buf_printf(&c->text, "{%zu}", c->n_tags);
        i = end;
    }
    janas_buf_puts(&c->text, "\n=== words\n");
    for (const char *p = sep ? sep + 5 : en + ln; *p;) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        const char *eq = memchr(p, '=', len);
        if (eq && *p != '#') {
            const char *a = p, *b = eq;
            while (b > a && b[-1] == ' ')
                b--;
            const char *v = eq + 1;
            while (*v == ' ')
                v++;
            if (push(&c->keys, &c->n_keys, a, (size_t)(b - a)) != 0 ||
                push(&c->english, &c->n_english, v, (size_t)(p + len - v)) != 0)
                return -1;
            /* a term of the glossary is not asked: decode puts it */
            if (!glossary_word(lang, a, (size_t)(b - a)))
                janas_buf_printf(&c->text, "%zu: %.*s%s\n", c->n_keys,
                                 (int)(p + len - v), v,
                                 hint(a, (size_t)(b - a)));
        }
        p = nl ? nl + 1 : p + len;
    }
    return c->text.oom ? -1 : 0;
}

/* The model's answer with the tags and keys put back; NULL with the
   reason in why when a marker or a word is missing, doubled or made up. */
static char *decode(const struct coded *c, const char *ans, char *why,
                    size_t why_len)
{
    const char *w = strstr(ans, "=== words");
    if (!w) {
        snprintf(why, why_len, "no words");
        return NULL;
    }
    size_t tn = (size_t)(w - ans);
    while (tn && (ans[tn - 1] == '\n' || ans[tn - 1] == ' '))
        tn--;
    int *seen = calloc(c->n_tags + 1, sizeof *seen);
    char **val = calloc(c->n_keys + 1, sizeof *val);
    struct janas_buf out = {0};
    int bad = !seen || !val;
    for (size_t i = 0; !bad && i < tn;) {
        if (ans[i] == '{') {
            char *end;
            unsigned long k = strtoul(ans + i + 1, &end, 10);
            if (end > ans + i + 1 && *end == '}') {
                if (k < 1 || k > c->n_tags || seen[k]++) {
                    snprintf(why, why_len, "marker {%lu} made up or doubled",
                             k);
                    bad = 1;
                    break;
                }
                janas_buf_puts(&out, c->tags[k - 1]);
                i = (size_t)(end - ans) + 1;
                /* the lines after a tag are the English ones: the 35B
                   added blank lines between the legs of a journey. So are
                   the spaces between two tags with nothing else between
                   them; those within a sentence are the language's. */
                size_t j = i;
                int nl = 0;
                while (j < tn &&
                       (ans[j] == ' ' || ans[j] == '\n' || ans[j] == '\r'))
                    nl |= ans[j++] != ' ';
                const char *g = c->gaps[k - 1];
                if ((c->to_tag[k - 1] && j < tn && ans[j] == '{' &&
                     strtoul(ans + j + 1, &end, 10) == k + 1 && *end == '}') ||
                    (nl && strchr(g, '\n'))) {
                    janas_buf_puts(&out, g);
                    i = j;
                }
                continue;
            }
        }
        janas_buf_put(&out, ans + i, 1);
        i++;
    }
    for (size_t k = 1; !bad && k <= c->n_tags; k++)
        if (!seen[k]) {
            snprintf(why, why_len, "marker {%zu} dropped", k);
            bad = 1;
        }
    for (size_t k = 1; !bad && k <= c->n_keys; k++) {
        const char *g =
            glossary_word(lang, c->keys[k - 1], strlen(c->keys[k - 1]));
        if (g)
            val[k] = strdup(g);
    }
    for (const char *p = w; !bad && *p;) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        char *end;
        unsigned long k = strtoul(p, &end, 10);
        if (end > p && *end == ':' && k >= 1 && k <= c->n_keys && !val[k]) {
            const char *v = end + 1;
            while (*v == ' ')
                v++;
            size_t vn = (size_t)(p + len - v);
            const char *br = memchr(v, '[', vn); /* the hint, if repeated */
            if (br)
                vn = (size_t)(br - v);
            while (vn && (v[vn - 1] == ' ' || v[vn - 1] == '\r'))
                vn--;
            val[k] = strndup(v, vn);
        }
        p = nl ? nl + 1 : p + len;
    }
    /* a word left out stays English: the layout still holds (a marker
       left out would not) - Qwen3.6-35B-A3B skipped one line in a hundred */
    for (size_t k = 1; !bad && k <= c->n_keys; k++)
        if (!val[k])
            val[k] = strdup(c->english[k - 1]);
    if (!bad) {
        janas_buf_puts(&out, "\n---\n");
        for (size_t k = 1; k <= c->n_keys; k++)
            janas_buf_printf(&out, "%s = %s\n", c->keys[k - 1], val[k]);
        if (comma_lang(lang))
            janas_buf_puts(&out, "decimal = ,\n");
        janas_buf_put(&out, "", 1);
    }
    for (size_t k = 0; val && k <= c->n_keys; k++)
        free(val[k]);
    free(val);
    free(seen);
    if (bad || out.oom) {
        janas_buf_free(&out);
        return NULL;
    }
    return out.p;
}

/* The layout of name in the user's language: from the file, or asked of
   the model and kept; NULL when there is none (English is used). */
static char *translated(const char *name, const char *en)
{
    char dir[1024], path[1200];
    if (dir_of(dir, sizeof dir) != 0)
        return NULL;
    snprintf(path, sizeof path, "%s/%s.%s.%016llx.txt", dir, name, lang,
             (unsigned long long)glossary_mark(lang, name, mark(en)));
    char *have = read_file(path);
    char why[200];
    if (have && janas_tpl_same_shape(en, strlen(en), have, strlen(have), why,
                                     sizeof why))
        return have;
    free(have);
    struct coded c;
    if (!translator || encode(en, &c) != 0)
        return NULL;
    struct janas_buf ask = {0};
    const char *terms = glossary_terms(lang, name);
    janas_buf_printf(&ask, "Translate into %s%s%s:\n\n%.*s", lang_name(lang),
                     terms ? ", with these terms: " : "", terms ? terms : "",
                     (int)c.text.n, c.text.p);
    char *ans = NULL;
    int r = ask.oom ? -1 : translator(name, SYSTEM, ask.p, &ans);
    janas_buf_free(&ask);
    char *out = NULL;
    if (r == 0 && ans) {
        unwrap(ans);
        /* the text given back as it came: the 35B did so with GitHub's
           issues, translating only the words. Not kept, so that it is
           asked again the next time. */
        const char *w = strstr(ans, "=== words");
        const char *cw = strstr(c.text.p, "\n=== words");
        size_t an = w ? (size_t)(w - ans) : strlen(ans);
        size_t cn = cw ? (size_t)(cw - c.text.p) : c.text.n;
        while (an && (ans[an - 1] == '\n' || ans[an - 1] == ' '))
            an--;
        while (cn && (c.text.p[cn - 1] == '\n' || c.text.p[cn - 1] == ' '))
            cn--;
        if (an == cn && memcmp(ans, c.text.p, cn) == 0)
            snprintf(why, sizeof why, "the text came back untranslated");
        else
            out = decode(&c, ans, why, sizeof why);
    } else {
        snprintf(why, sizeof why, "no answer");
    }
    free(ans);
    coded_free(&c);
    if (out && !janas_tpl_same_shape(en, strlen(en), out, strlen(out), why,
                                     sizeof why)) {
        free(out);
        out = NULL;
    }
    if (!out) {
        fprintf(stderr,
                "janas-chat: the layout %s in %s is not usable (%s): "
                "English is used\n",
                name, lang, why);
        return NULL;
    }
    FILE *f = fopen(path, "wb");
    if (f) {
        fputs(out, f);
        fclose(f);
    }
    return out;
}

static char *fill(const char *tpl, const struct janas_json *data)
{
    struct janas_buf b = {0};
    char err[200];
    if (janas_tpl_render(tpl, strlen(tpl), data, &b, err, sizeof err) != 0) {
        fprintf(stderr, "janas-chat: a layout could not be filled: %s\n", err);
        janas_buf_free(&b);
        return NULL;
    }
    janas_buf_put(&b, "", 1);
    return b.oom ? NULL : b.p;
}

int layout_take(const char *result, size_t n, char **user, char **model)
{
    *user = *model = NULL;
    struct janas_json_doc *d = janas_json_parse(result, n, NULL, 0);
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    const struct janas_json *meta =
        janas_json_get(janas_json_get(r, "_meta"), META);
    const struct janas_json *data = janas_json_get(r, "structuredContent");
    const char *name = janas_json_str(janas_json_get(meta, "name"));
    const char *en = janas_json_str(janas_json_get(meta, "text"));
    const char *brief = janas_json_str(janas_json_get(meta, "brief"));
    int ok = 0;
    if (name && en && brief && data && data->type == JANAS_JSON_OBJECT &&
        !strchr(name, '/') && name[0] != '.') {
        char *mine = strcmp(lang, "en") != 0 ? translated(name, en) : NULL;
        *user = fill(mine ? mine : en, data);
        if (!*user && mine) /* a translation that does not fill: English */
            *user = fill(en, data);
        free(mine);
        *model = fill(brief, data);
        ok = *user && *model;
        if (!ok) {
            free(*user);
            free(*model);
            *user = *model = NULL;
        }
    }
    janas_json_free(d);
    return ok;
}
