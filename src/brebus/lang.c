/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * lang.c - the languages of this computer, for Brebus's window: its
 * address says them (?window&lang=it-IT,en), and the page (page/i18n.js)
 * speaks the first of them it knows. On Linux and the BSDs LANGUAGE,
 * LC_ALL, LC_MESSAGES and LANG, as gettext reads them; on Windows the
 * user's languages of the interface. macOS's window asks its own
 * (window_mac.m).
 */
#include "brebus.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* one language at the end of out: "it_IT.UTF-8" becomes "it-IT"; C and
   POSIX say none */
static void add(char *out, size_t len, const char *l, size_t n)
{
    char one[40];
    size_t k = 0;
    for (size_t i = 0; i < n && k + 1 < sizeof one; i++) {
        char c = l[i];
        if (c == '.' || c == '@')
            break;
        if (c == '_')
            c = '-';
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-'))
            return;
        one[k++] = c;
    }
    one[k] = 0;
    if (k < 2 || !strcmp(one, "C") || !strcmp(one, "POSIX"))
        return;
    size_t at = strlen(out);
    if (at + k + 2 > len)
        return;
    snprintf(out + at, len - at, "%s%s", at ? "," : "", one);
}

void brebus_sys_langs(char *out, size_t len)
{
    if (!len)
        return;
    out[0] = 0;
#ifdef _WIN32
    ULONG count = 0, n = 0;
    if (GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, &count, NULL, &n) && n &&
        n < 512) {
        WCHAR buf[512];
        if (GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, &count, buf, &n))
            for (WCHAR *p = buf; *p; p += wcslen(p) + 1) {
                char a[40];
                int k = WideCharToMultiByte(CP_UTF8, 0, p, -1, a, sizeof a,
                                            NULL, NULL);
                if (k > 1)
                    add(out, len, a, strlen(a));
            }
    }
#else
    /* LANGUAGE is a list (it:en), and holds only beside a locale that is
       not C */
    const char *all = getenv("LC_ALL"), *msg = getenv("LC_MESSAGES"),
               *lang = getenv("LANG");
    const char *loc = all && *all ? all : msg && *msg ? msg : lang;
    const char *list = getenv("LANGUAGE");
    if (list && *list && loc && *loc && strcmp(loc, "C") &&
        strcmp(loc, "POSIX"))
        for (const char *p = list; *p;) {
            size_t n = strcspn(p, ":");
            add(out, len, p, n);
            p += n;
            if (*p == ':')
                p++;
        }
    if (loc && *loc)
        add(out, len, loc, strlen(loc));
#endif
}

/*
 * The page's languages but English are files of their own beside brebus
 * (lang/<code>.json in the source, brebus-lang/ beside the program), data
 * alone: {"name": "Français", "locale": "fr", "texts": {English: translation,
 * ...}}, a translation a string or [one, other]. A file is never handed to the
 * page as it is: brebus reads it, keeps each text whose translation is as
 * safe as its English, and writes the script itself; a file changed by
 * anyone can make a text wrong, never run anything.
 *
 * A translation is kept when it holds the same markup as its English,
 * tag for tag, attributes and all, and no other '<'; a '"' outside the
 * markup only where its English has one (the page puts texts inside
 * attributes); the parts {x} of its English alone, and {n} in a number's
 * forms; no NUL; and no longer than LANG_MAX_TEXT.
 */
#include <dirent.h>
#include <sys/stat.h>

#include "llm/json.h"

#define LANG_MAX_FILE (4u << 20)
#define LANG_MAX_TEXT 8192
#define LANG_MAX_TEXTS 20000
#define LANG_MAX_LANGS 200
#define LANG_MAX_TAGS 32
#define LANG_MAX_PARTS 8

/* a code a language's file may have: it, pt-BR, zh-Hans, es-419 */
static int code_ok(const char *c, size_t n)
{
    size_t i = 0;
    while (i < n && c[i] >= 'a' && c[i] <= 'z')
        i++;
    if (i < 2 || i > 3)
        return 0;
    if (i == n)
        return 1;
    if (c[i] != '-' || n - i - 1 < 2 || n - i - 1 > 4)
        return 0;
    for (size_t k = i + 1; k < n; k++)
        if (!((c[k] >= 'a' && c[k] <= 'z') || (c[k] >= 'A' && c[k] <= 'Z') ||
              (c[k] >= '0' && c[k] <= '9')))
            return 0;
    return 1;
}

/* a locale for the numbers and the dates: pt-PT, sr-Latn-RS */
static int locale_ok(const char *c, size_t n)
{
    if (n > 35)
        return 0;
    size_t i = 0;
    while (i < n && c[i] >= 'a' && c[i] <= 'z')
        i++;
    if (i < 2 || i > 3)
        return 0;
    while (i < n) {
        if (c[i++] != '-')
            return 0;
        size_t s = i;
        while (i < n &&
               ((c[i] >= 'a' && c[i] <= 'z') || (c[i] >= 'A' && c[i] <= 'Z') ||
                (c[i] >= '0' && c[i] <= '9')))
            i++;
        if (i - s < 2 || i - s > 8)
            return 0;
    }
    return 1;
}

/* the tags of a text in order, each from its '<' to its '>': their count,
   or -1 for a '<' that ends no tag, or more than max */
static int tags_of(const char *s, size_t n, const char **at, size_t *len,
                   int max)
{
    int k = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] != '<')
            continue;
        const char *e = memchr(s + i, '>', n - i);
        if (!e || k == max || memchr(s + i + 1, '<', (size_t)(e - s - i - 1)))
            return -1;
        at[k] = s + i;
        len[k++] = (size_t)(e - s - i) + 1;
        i = (size_t)(e - s);
    }
    return k;
}

/* the '"' of a text outside its tags */
static int quotes_of(const char *s, size_t n)
{
    int q = 0, in = 0;
    for (size_t i = 0; i < n; i++)
        if (s[i] == '<')
            in = 1;
        else if (s[i] == '>')
            in = 0;
        else if (s[i] == '"' && !in)
            q++;
    return q;
}

/* the parts {x} of a text: whether each of v's is one of k's (or n, for
   a number's forms); -1 when k has more than LANG_MAX_PARTS */
static int parts_ok(const char *k, size_t kn, const char *v, size_t vn,
                    int forms)
{
    int nk = 0;
    for (size_t i = 0; i < kn; i++)
        if (k[i] == '{' && ++nk > LANG_MAX_PARTS)
            return 0;
    for (size_t i = 0; i < vn; i++) {
        if (v[i] != '{')
            continue;
        size_t j = i + 1;
        while (j < vn &&
               ((v[j] >= 'a' && v[j] <= 'z') || (v[j] >= 'A' && v[j] <= 'Z') ||
                (v[j] >= '0' && v[j] <= '9') || v[j] == '_'))
            j++;
        if (j == i + 1 || j == vn || v[j] != '}')
            continue; /* a brace, no part */
        size_t pl = j - i + 1;
        if (forms && pl == 3 && v[i + 1] == 'n')
            continue;
        int found = 0;
        for (size_t a = 0; a + pl <= kn && !found; a++)
            found = !memcmp(k + a, v + i, pl);
        if (!found)
            return 0;
        i = j;
    }
    return 1;
}

int brebus_lang_text_ok(const char *k, size_t kn, const char *v, size_t vn,
                        int forms)
{
    if (!k || !v || kn > LANG_MAX_TEXT || vn > LANG_MAX_TEXT ||
        strlen(k) != kn || strlen(v) != vn)
        return 0;
    const char *ka[LANG_MAX_TAGS], *va[LANG_MAX_TAGS];
    size_t kl[LANG_MAX_TAGS], vl[LANG_MAX_TAGS];
    int nk = tags_of(k, kn, ka, kl, LANG_MAX_TAGS),
        nv = tags_of(v, vn, va, vl, LANG_MAX_TAGS);
    if (nk < 0 || nv != nk)
        return 0;
    /* tag for tag: each of the translation's one of its English's, once */
    char used[LANG_MAX_TAGS] = {0};
    for (int i = 0; i < nv; i++) {
        int j = 0;
        while (j < nk &&
               (used[j] || kl[j] != vl[i] || memcmp(ka[j], va[i], vl[i])))
            j++;
        if (j == nk)
            return 0;
        used[j] = 1;
    }
    if (quotes_of(v, vn) && !quotes_of(k, kn))
        return 0;
    return parts_ok(k, kn, v, vn, forms);
}

/* the folder of the languages: BREBUS_LANG_DIR, or brebus-lang beside
   brebus (every package has it there: the folder, the AppImage, /opt of
   the system's packages, Windows's, Brebus.app's Contents/MacOS) */
static int lang_dir(const char *bin_dir, char *dir, size_t len)
{
    const char *env = getenv("BREBUS_LANG_DIR");
    struct stat st;
    if ((size_t)(env && *env
                     ? snprintf(dir, len, "%s", env)
                     : snprintf(dir, len, "%s/brebus-lang", bin_dir)) >= len)
        return -1;
    return stat(dir, &st) == 0 && S_ISDIR(st.st_mode) ? 0 : -1;
}

static char *read_file(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    char *p = malloc(LANG_MAX_FILE + 1);
    size_t got = p ? fread(p, 1, LANG_MAX_FILE + 1, f) : 0;
    fclose(f);
    if (!p || got > LANG_MAX_FILE) {
        free(p);
        return NULL;
    }
    *n = got;
    return p;
}

struct lang {
    char code[16];
    char name[96];
};

static int lang_cmp(const void *a, const void *b)
{
    return strcmp(((const struct lang *)a)->code,
                  ((const struct lang *)b)->code);
}

/* a language's file read: its document, or NULL */
static struct janas_json_doc *lang_read(const char *dir, const char *code)
{
    char path[4200];
    if ((size_t)snprintf(path, sizeof path, "%s/%s.json", dir, code) >=
        sizeof path)
        return NULL;
    size_t n = 0;
    char *t = read_file(path, &n);
    if (!t)
        return NULL;
    char err[128];
    struct janas_json_doc *d = janas_json_parse(t, n, err, sizeof err);
    free(t);
    if (!d)
        fprintf(stderr, "brebus: %s: %s\n", path, err);
    else if (!janas_json_root(d) ||
             janas_json_root(d)->type != JANAS_JSON_OBJECT) {
        janas_json_free(d);
        d = NULL;
    }
    return d;
}

/* its name, in itself: plain text alone */
static int name_of(const struct janas_json *root, char *name, size_t len)
{
    const struct janas_json *v = janas_json_get(root, "name");
    const char *s = janas_json_str(v);
    if (!s || !v->n || v->n >= len || strlen(s) != v->n ||
        strpbrk(s, "<>&\"'\\"))
        return -1;
    memcpy(name, s, v->n + 1);
    return 0;
}

/* the languages there are, sorted by code */
static int lang_list(const char *dir, struct lang *l, int max)
{
    DIR *d = opendir(dir);
    if (!d)
        return 0;
    int n = 0;
    struct dirent *e;
    while (n < max && (e = readdir(d))) {
        size_t len = strlen(e->d_name);
        if (len < 6 || len - 5 >= sizeof l->code ||
            strcmp(e->d_name + len - 5, ".json") ||
            !code_ok(e->d_name, len - 5) || !strncmp(e->d_name, "en", 2))
            continue;
        memcpy(l[n].code, e->d_name, len - 5);
        l[n].code[len - 5] = 0;
        struct janas_json_doc *doc = lang_read(dir, l[n].code);
        if (!doc)
            continue;
        if (name_of(janas_json_root(doc), l[n].name, sizeof l[n].name) == 0)
            n++;
        janas_json_free(doc);
    }
    closedir(d);
    qsort(l, (size_t)n, sizeof *l, lang_cmp);
    return n;
}

static int ieq(const char *a, const char *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z')
            x = (char)(x - 'A' + 'a');
        if (y >= 'A' && y <= 'Z')
            y = (char)(y - 'A' + 'a');
        if (x != y)
            return 0;
    }
    return 1;
}

/* the first of want (it-IT,en: the choice, then the system's) there is a
   file for: its index in l, or -1 for English (asked, or none there) */
static int lang_pick(const char *want, const struct lang *l, int n)
{
    for (const char *p = want ? want : ""; *p;) {
        size_t m = strcspn(p, ",");
        char w[16];
        size_t k = 0;
        for (size_t i = 0; i < m && k + 1 < sizeof w; i++) {
            char c = p[i];
            if (c == '.' || c == '@' || c == ';')
                break;
            w[k++] = c == '_' ? '-' : c;
        }
        w[k] = 0;
        p += m + (p[m] == ',');
        size_t base = strcspn(w, "-");
        if (base < 2)
            continue;
        if (base == 2 && ieq(w, "en", 2))
            return -1;
        for (int i = 0; i < n; i++) /* it-IT for it-IT */
            if (strlen(l[i].code) == k && ieq(l[i].code, w, k))
                return i;
        for (int i = 0; i < n; i++) /* pt for pt-AO */
            if (strlen(l[i].code) == base && ieq(l[i].code, w, base))
                return i;
        for (int i = 0; i < n; i++) /* pt-BR for pt, when it is alone */
            if (l[i].code[base] == '-' && ieq(l[i].code, w, base))
                return i;
    }
    return -1;
}

int brebus_lang_js(const char *bin_dir, const char *want, struct janas_buf *out)
{
    char dir[4200];
    static struct lang l[LANG_MAX_LANGS];
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    int n = lang_dir(bin_dir, dir, sizeof dir) == 0
                ? lang_list(dir, l, LANG_MAX_LANGS)
                : 0;
    janas_buf_puts(out, "LANGS = [[\"en\", \"English\"]");
    for (int i = 0; i < n; i++) {
        janas_buf_puts(out, ", [");
        janas_json_write_str(out, l[i].code, strlen(l[i].code));
        janas_buf_puts(out, ", ");
        janas_json_write_str(out, l[i].name, strlen(l[i].name));
        janas_buf_puts(out, "]");
    }
    janas_buf_puts(out, "];\n");
    int at = lang_pick(want, l, n);
    struct janas_json_doc *doc = at < 0 ? NULL : lang_read(dir, l[at].code);
    const struct janas_json *root = doc ? janas_json_root(doc) : NULL,
                            *texts =
                                root ? janas_json_get(root, "texts") : NULL;
    if (!texts || texts->type != JANAS_JSON_OBJECT ||
        texts->n > LANG_MAX_TEXTS) {
        janas_json_free(doc);
        pthread_mutex_unlock(&mu);
        janas_buf_puts(out, "LANG = LOCALE = \"en\";\n"
                            "document.documentElement.lang = LOCALE;\n");
        return out->oom ? -1 : 0;
    }
    const struct janas_json *lc = janas_json_get(root, "locale");
    const char *loc = janas_json_str(lc);
    if (!loc || !locale_ok(loc, lc->n))
        loc = l[at].code;
    janas_buf_puts(out, "LANG = ");
    janas_json_write_str(out, l[at].code, strlen(l[at].code));
    janas_buf_puts(out, ";\nLOCALE = ");
    janas_json_write_str(out, loc, strlen(loc));
    janas_buf_puts(out, ";\nI18N[LANG] = {");
    int kept = 0, out_n = 0;
    for (const struct janas_json *e = texts->child; e; e = e->next) {
        int ok = e->key_n && strlen(e->key) == e->key_n &&
                 strcmp(e->key, "__proto__");
        if (ok && e->type == JANAS_JSON_STRING)
            ok = brebus_lang_text_ok(e->key, e->key_n, e->s, e->n, 0);
        else if (ok && e->type == JANAS_JSON_ARRAY && e->n == 2 &&
                 e->child->type == JANAS_JSON_STRING &&
                 e->child->next->type == JANAS_JSON_STRING)
            ok = brebus_lang_text_ok(e->key, e->key_n, e->child->s, e->child->n,
                                     1) &&
                 brebus_lang_text_ok(e->key, e->key_n, e->child->next->s,
                                     e->child->next->n, 1);
        else
            ok = 0;
        if (!ok) {
            out_n++;
            continue;
        }
        janas_buf_puts(out, kept++ ? ",\n" : "\n");
        janas_json_write_str(out, e->key, e->key_n);
        janas_buf_puts(out, ": ");
        if (e->type == JANAS_JSON_STRING)
            janas_json_write_str(out, e->s, e->n);
        else {
            janas_buf_puts(out, "[");
            janas_json_write_str(out, e->child->s, e->child->n);
            janas_buf_puts(out, ", ");
            janas_json_write_str(out, e->child->next->s, e->child->next->n);
            janas_buf_puts(out, "]");
        }
    }
    janas_buf_puts(out, "};\ndocument.documentElement.lang = LOCALE;\n");
    if (out_n)
        fprintf(stderr,
                "brebus: %s/%s.json: %d texts left in English, their "
                "translation not as safe as the English\n",
                dir, l[at].code, out_n);
    janas_json_free(doc);
    pthread_mutex_unlock(&mu);
    return out->oom ? -1 : 0;
}
