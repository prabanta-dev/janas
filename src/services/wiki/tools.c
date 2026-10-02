/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - the tools janas-wiki offers (see wiki.h): wiki_search,
 * wiki_page and wiki_nearby, on the API of the Wikipedia of a language
 * (action=query: TextExtracts for the text, GeoData for the places).
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "services/common/geo.h"
#include "services/common/locate.h"
#include "common/mcp_server.h"
#include "wiki.h"

#define KEEP_S 3600       /* a page changes seldom within the hour */
#define WHOLE_MAX 8000    /* a page this long is given whole */
#define INTRO_MAX 6000    /* else its introduction, at most this */
#define SECTION_MAX 10000 /* and a section, at most this */
#define LICENSE                                                                \
    "Text from Wikipedia, under CC BY-SA 4.0: give the page's address as "     \
    "the source."

static const char *arg_str(const struct janas_json *args, const char *name)
{
    const char *s = janas_json_str(janas_json_get(args, name));
    return s && *s ? s : NULL;
}

static double arg_num(const struct janas_json *args, const char *name,
                      double def)
{
    const struct janas_json *v = janas_json_get(args, name);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, def) : def;
}

static int result(struct janas_buf *out, struct janas_buf *b, int is_error)
{
    janas_mcps_text_result(b, out->p ? out->p : "", out->n, is_error);
    janas_buf_free(out);
    return 0;
}

static int fail(struct janas_buf *b, const char *text)
{
    struct janas_buf out = {0};
    janas_buf_puts(&out, text);
    return result(&out, b, 1);
}

static int lang_ok(const char *s, size_t n)
{
    int ok = n >= 2 && n <= 12;
    for (size_t i = 0; ok && i < n; i++)
        ok = islower((unsigned char)s[i]) || s[i] == '-';
    return ok;
}

/* The language of the Wikipedia asked for: "it", "en", "zh-yue"; it goes
   into the host's name, so nothing else is taken. When the model gives
   none, the computer's (LANG=it_IT.UTF-8: it), else English. */
static const char *arg_lang(const struct janas_json *args, char *to, size_t cap)
{
    const char *s = arg_str(args, "language");
    if (s && lang_ok(s, strlen(s)) && strlen(s) < cap) {
        snprintf(to, cap, "%s", s);
        return to;
    }
    static const char *const vars[] = {"LC_ALL", "LC_MESSAGES", "LANG"};
    for (size_t i = 0; i < sizeof vars / sizeof *vars; i++) {
        const char *v = getenv(vars[i]);
        if (!v || !*v)
            continue;
        size_t n = strcspn(v, "_.@");
        if (lang_ok(v, n) && n < cap) {
            snprintf(to, cap, "%.*s", (int)n, v);
            return to;
        }
        break; /* the first one set rules, "C" too */
    }
    snprintf(to, cap, "en");
    return to;
}

static void url_put(struct janas_buf *b, const char *s)
{
    static const char hex[] = "0123456789ABCDEF";
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            janas_buf_put(b, (const char *)&c, 1);
        else
            janas_buf_printf(b, "%%%c%c", hex[c >> 4], hex[c & 15]);
    }
}

static void api_url(struct janas_buf *u, const char *lang)
{
    janas_buf_printf(u,
                     "https://%s.wikipedia.org/w/api.php?action=query&"
                     "format=json&formatversion=2&",
                     lang);
}

/* The answer to url into body: 0, or -1 with the reason in err. */
static int ask(struct janas_buf *url, struct janas_buf *body, char *err,
               size_t err_len)
{
    if (url->oom) {
        snprintf(err, err_len, "out of memory");
        janas_buf_free(url);
        return -1;
    }
    char why[256] = "";
    int status = wiki_fetch(url->p, KEEP_S, body, why, sizeof why);
    janas_buf_free(url);
    if (status == 200)
        return 0;
    if (status == 429)
        snprintf(err, err_len,
                 "Wikipedia asks to slow down (HTTP 429): try again in a "
                 "minute");
    else if (status > 0)
        snprintf(err, err_len, "Wikipedia answered HTTP %d", status);
    else
        snprintf(err, err_len, "Wikipedia did not answer (%s)", why);
    return -1;
}

static void put_point(struct janas_buf *out, double lat, double lon)
{
    janas_buf_printf(out, "%.4f, %.4f", lat, lon);
}

/* ---- wiki_search ---- */

static int tool_search(const struct janas_json *args, struct janas_buf *b)
{
    const char *q = arg_str(args, "query");
    if (!q)
        return fail(b, "No query given: what to look for.");
    char lang[16];
    arg_lang(args, lang, sizeof lang);
    int limit = (int)arg_num(args, "limit", 8);
    limit = limit < 1 ? 1 : limit > 20 ? 20 : limit;
    struct janas_buf url = {0}, body = {0};
    api_url(&url, lang);
    janas_buf_puts(&url, "generator=search&gsrsearch=");
    url_put(&url, q);
    janas_buf_printf(&url, "&gsrlimit=%d&prop=description%%7Ccoordinates",
                     limit);
    char err[384];
    struct wiki_hit *hits = calloc(WIKI_HITS, sizeof *hits);
    int n = !hits ? -1
            : ask(&url, &body, err, sizeof err) != 0
                ? -1
                : wiki_read_hits(body.p, body.n, hits, limit, err, sizeof err);
    janas_buf_free(&body);
    /* titles and a line each: what the model needs to pick one, and no
       more - every token of it is read before it answers */
    struct janas_buf out = {0};
    if (!hits)
        janas_buf_puts(&out, "out of memory");
    else if (n < 0)
        janas_buf_printf(&out, "%s.", err);
    else if (n == 0)
        janas_buf_printf(&out,
                         "Nothing found on Wikipedia (%s) for \"%s\". Try "
                         "other words, or another language's Wikipedia "
                         "(en has the most pages).",
                         lang, q);
    else {
        janas_buf_printf(&out, "Wikipedia (%s), pages for \"%s\":\n", lang, q);
        for (int i = 0; i < n; i++) {
            const struct wiki_hit *h = &hits[i];
            janas_buf_printf(&out, "%d. %s", i + 1, h->title);
            if (h->desc[0])
                janas_buf_printf(&out, " - %s", h->desc);
            janas_buf_puts(&out, "\n");
        }
        janas_buf_puts(&out, "A page is shown to the user by its title, as "
                             "written above.");
    }
    free(hits);
    return result(&out, b, n < 0 || !hits);
}

/* ---- wiki_page ---- */

/* Words in n bytes of text. */
static size_t words(const char *s, size_t n)
{
    size_t w = 0;
    for (size_t i = 0; i < n; i++)
        if (s[i] != ' ' && s[i] != '\n' &&
            (i == 0 || s[i - 1] == ' ' || s[i - 1] == '\n'))
            w++;
    return w;
}

/*
 * The page goes to the user as it is, and the model gets a note of it: a
 * page is thousands of tokens, which a model reads before it can answer -
 * over a minute of a large one for a page, and a few pages fill its
 * context - while the user asked to read it, not to have it retold.
 */
static int tool_page(const struct janas_json *args, struct janas_buf *b)
{
    const char *title = arg_str(args, "title");
    if (!title)
        return fail(b, "No title given: the page's title, as a search "
                       "gives it.");
    const char *section = arg_str(args, "section");
    char lang[16];
    arg_lang(args, lang, sizeof lang);
    struct janas_buf url = {0}, body = {0};
    api_url(&url, lang);
    janas_buf_puts(&url, "redirects=1&titles=");
    url_put(&url, title);
    janas_buf_puts(&url, "&prop=extracts%7Ccoordinates%7Cdescription%7C"
                         "info%7Cpageprops&inprop=url&explaintext=1&"
                         "exsectionformat=wiki&ppprop=disambiguation");
    char err[384];
    struct wiki_page p;
    int rc = ask(&url, &body, err, sizeof err) != 0
                 ? -1
                 : wiki_read_page(body.p, body.n, &p, err, sizeof err);
    janas_buf_free(&body);
    struct janas_buf out = {0}, note = {0}, list = {0};
    if (rc != 0) {
        janas_buf_printf(&out, "%s.", err);
        return result(&out, b, 1);
    }
    if (p.missing || !p.text) {
        janas_buf_printf(&out,
                         "There is no page \"%s\" on Wikipedia (%s): search "
                         "for it, and ask for a page by the title the search "
                         "gives.",
                         title, lang);
        wiki_page_free(&p);
        return result(&out, b, 1);
    }
    /* the user's: the page as it is */
    janas_buf_printf(&out, "%s", p.title);
    if (p.desc[0])
        janas_buf_printf(&out, " - %s", p.desc);
    janas_buf_printf(&out, "\nWikipedia (%s): %s\n\n", lang, p.url);
    const char *what = "its introduction";
    char what_s[300];
    size_t from = 0, n = 0, shown;
    int found = 0;
    if (p.disamb) {
        what = "a disambiguation page, the pages of that name";
        shown = wiki_cut(p.text, p.len, WHOLE_MAX);
        janas_buf_put(&out, p.text, shown);
    } else if (section &&
               wiki_section(p.text, p.len, section, &from, &n) == 0) {
        found = 1;
        shown = wiki_drop_heads(p.text + from,
                                wiki_cut(p.text + from, n, SECTION_MAX));
        janas_buf_put(&out, p.text + from, shown);
        snprintf(
            what_s, sizeof what_s, "its section \"%.*s\"",
            (int)strcspn(p.text + from + strspn(p.text + from, "= "), "=\n"),
            p.text + from + strspn(p.text + from, "= "));
        what = what_s;
        if (shown < n)
            wiki_section_list(p.text + from, n, &list);
    } else {
        if (p.len <= WHOLE_MAX) {
            what = "whole";
            shown = p.len;
            janas_buf_put(&out, p.text, p.len);
        } else {
            shown = wiki_cut(p.text, wiki_intro_len(p.text, p.len), INTRO_MAX);
            janas_buf_put(&out, p.text, shown);
            wiki_section_list(p.text, p.len, &list);
        }
    }
    size_t nw = words(out.p ? out.p : "", out.n);
    if (list.p)
        janas_buf_printf(&out, "\n\nSections: %s.", list.p);
    janas_buf_puts(&out, "\n\nText from Wikipedia, CC BY-SA 4.0.");
    /* the model's: a note */
    janas_buf_printf(&note,
                     "The page \"%s\" of Wikipedia (%s), %s, is shown to "
                     "the user as it is: %s, %zu words.",
                     p.title, lang, p.url, what, nw);
    if (p.from[0])
        janas_buf_printf(&note, " Asked as \"%s\", which leads to it.", p.from);
    if (section && !found && !p.disamb)
        janas_buf_printf(&note, " It has no section \"%s\".", section);
    if (list.p)
        janas_buf_printf(&note, " Its sections, to show one when asked: %s.",
                         list.p);
    janas_buf_puts(&note, " Do not repeat, translate or summarize it unless "
                          "the user asks; tell them, in a line in their "
                          "language, that the page is above, with its "
                          "address.");
    janas_buf_free(&list);
    wiki_page_free(&p);
    if (out.oom || note.oom) {
        janas_buf_free(&out);
        janas_buf_free(&note);
        return fail(b, "out of memory");
    }
    janas_mcps_split_result(b, note.p, note.n, out.p, out.n);
    janas_buf_free(&out);
    janas_buf_free(&note);
    return 0;
}

/* ---- wiki_nearby ---- */

/* "38.02, 12.51" */
static int as_point(const char *q, double *lat, double *lon)
{
    char *end;
    double a = strtod(q, &end);
    if (end == q)
        return 0;
    while (*end == ' ' || *end == ',' || *end == ';')
        end++;
    char *end2;
    double b = strtod(end, &end2);
    while (*end2 == ' ')
        end2++;
    if (end2 == end || *end2 || fabs(a) > 90 || fabs(b) > 180)
        return 0;
    *lat = a;
    *lon = b;
    return 1;
}

/* A place's point from its page, or else from the first page with one
   that a search for it finds. 0, or -1 with the reason in err. */
static int place_point(const char *place, const char *lang, double *lat,
                       double *lon, char *name, size_t name_cap, char *err,
                       size_t err_len)
{
    struct janas_buf url = {0}, body = {0};
    api_url(&url, lang);
    janas_buf_puts(&url, "redirects=1&titles=");
    url_put(&url, place);
    janas_buf_puts(&url, "&prop=coordinates%7Cdescription");
    struct wiki_page p;
    if (ask(&url, &body, err, err_len) == 0 &&
        wiki_read_page(body.p, body.n, &p, err, err_len) == 0) {
        wiki_page_free(&p);
        if (!p.missing && !isnan(p.lat)) {
            *lat = p.lat;
            *lon = p.lon;
            snprintf(name, name_cap, "%s", p.title);
            janas_buf_free(&body);
            return 0;
        }
    }
    body.n = 0;
    api_url(&url, lang);
    janas_buf_puts(&url, "generator=search&gsrlimit=5&gsrsearch=");
    url_put(&url, place);
    janas_buf_puts(&url, "&prop=coordinates");
    struct wiki_hit *hits = calloc(5, sizeof *hits);
    int n = !hits || ask(&url, &body, err, err_len) != 0
                ? -1
                : wiki_read_hits(body.p, body.n, hits, 5, err, err_len);
    janas_buf_free(&body);
    int found = -1;
    for (int i = 0; i < n && found < 0; i++)
        if (!isnan(hits[i].lat)) {
            *lat = hits[i].lat;
            *lon = hits[i].lon;
            snprintf(name, name_cap, "%s", hits[i].title);
            found = 0;
        }
    free(hits);
    if (found != 0 && n >= 0)
        snprintf(err, err_len,
                 "no page of Wikipedia (%s) with coordinates for \"%s\"", lang,
                 place);
    return found;
}

static int by_km(const void *a, const void *b)
{
    double x = ((const struct wiki_hit *)a)->km;
    double y = ((const struct wiki_hit *)b)->km;
    return x < y ? -1 : x > y;
}

/* A point in words: its town, or "3 km N of Erice" */
static void point_name(double lat, double lon, char *name, size_t cap)
{
    double km;
    const struct geo_city *c = geo_city_near(lat, lon, &km);
    if (!c)
        snprintf(name, cap, "%.4f, %.4f", lat, lon);
    else if (km < 3)
        snprintf(name, cap, "%s", c->name);
    else
        snprintf(
            name, cap, "%.0f km %s of %s", km,
            geo_compass(geo_bearing(c->lat * 1e-5, c->lon * 1e-5, lat, lon)),
            c->name);
}

static int tool_nearby(const struct janas_json *args, struct janas_buf *b)
{
    const char *place = arg_str(args, "place");
    double lat = arg_num(args, "latitude", NAN);
    double lon = arg_num(args, "longitude", NAN);
    char lang[16], name[256] = "", how[96] = "", err[384];
    arg_lang(args, lang, sizeof lang);
    double km = arg_num(args, "radius_km", 2);
    km = km < 0.1 ? 0.1 : km > 10 ? 10 : km;
    int limit = (int)arg_num(args, "limit", 15);
    limit = limit < 1 ? 1 : limit > WIKI_HITS ? WIKI_HITS : limit;
    struct janas_buf out = {0};
    if (place && as_point(place, &lat, &lon)) {
        place = NULL;
    } else if (place) {
        if (place_point(place, lang, &lat, &lon, name, sizeof name, err,
                        sizeof err) != 0) {
            janas_buf_printf(&out,
                             "%s. Give the place's coordinates, or "
                             "its name as its page has it.",
                             err);
            return result(&out, b, 1);
        }
    } else if (isnan(lat) || isnan(lon)) {
        struct janas_where w;
        if (janas_where(&w, err, sizeof err) != 0) {
            janas_buf_printf(&out, "No place given, and %s.", err);
            return result(&out, b, 1);
        }
        lat = w.lat;
        lon = w.lon;
        snprintf(name, sizeof name, "%s", w.city);
        snprintf(how, sizeof how, "%s", w.how);
    }
    if (!name[0])
        point_name(lat, lon, name, sizeof name);
    struct janas_buf url = {0}, body = {0};
    api_url(&url, lang);
    janas_buf_printf(&url,
                     "generator=geosearch&ggscoord=%.5f%%7C%.5f&"
                     "ggsradius=%d&ggslimit=%d&prop=coordinates%%7C"
                     "description&colimit=max",
                     lat, lon, (int)(km * 1000), limit);
    struct wiki_hit *hits = calloc(WIKI_HITS, sizeof *hits);
    int n = !hits ? -1
            : ask(&url, &body, err, sizeof err) != 0
                ? -1
                : wiki_read_hits(body.p, body.n, hits, limit, err, sizeof err);
    janas_buf_free(&body);
    if (n < 0) {
        janas_buf_printf(&out, "%s.", hits ? err : "out of memory");
        free(hits);
        return result(&out, b, 1);
    }
    for (int i = 0; i < n; i++) /* nearest first; without a point last */
        hits[i].km = isnan(hits[i].lat)
                         ? 1e9
                         : geo_km(lat, lon, hits[i].lat, hits[i].lon);
    qsort(hits, (size_t)n, sizeof *hits, by_km);
    janas_buf_printf(&out, "Wikipedia (%s): %d page%s within %.1f km of %s (",
                     lang, n, n == 1 ? "" : "s", km, name);
    put_point(&out, lat, lon);
    janas_buf_puts(&out, ")");
    if (how[0])
        janas_buf_printf(&out, " - where the user is, %s", how);
    janas_buf_puts(&out, n ? ":\n" : ".\n");
    for (int i = 0; i < n; i++) {
        const struct wiki_hit *h = &hits[i];
        double d = h->km;
        janas_buf_printf(&out, "\n%d. %s", i + 1, h->title);
        if (h->desc[0])
            janas_buf_printf(&out, " - %s", h->desc);
        if (d >= 1e8)
            continue;
        if (d < 0.05)
            janas_buf_puts(&out, ": here");
        else {
            const char *dir =
                geo_compass(geo_bearing(lat, lon, h->lat, h->lon));
            if (d < 1)
                janas_buf_printf(&out, ": %.0f m %s", round(d * 100) * 10, dir);
            else
                janas_buf_printf(&out, ": %.1f km %s", d, dir);
        }
    }
    if (n)
        janas_buf_puts(&out, "\n\nEach is read by its title, as written "
                             "above.");
    else
        janas_buf_puts(&out, "A wider radius (up to 10 km) may find some.");
    if (how[0])
        janas_buf_printf(&out,
                         "\nNo place was named: this is where the user is, "
                         "%s. Tell the user which place it is and that it "
                         "may be off: they can name theirs.",
                         how);
    janas_buf_puts(&out, "\n" LICENSE);
    free(hits);
    return result(&out, b, 0);
}

/* ---- the list ---- */

static void schema_str(struct janas_buf *b, const char *name, const char *what)
{
    janas_buf_printf(b,
                     "\"%s\": {\"type\": \"string\", \"description\": ", name);
    janas_json_write_str(b, what, strlen(what));
    janas_buf_puts(b, "}");
}

static void lang_prop(struct janas_buf *b)
{
    schema_str(b, "language",
               "The Wikipedia's language: the user's (it, en...).");
}

void wiki_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b, "\"tools\": [{\"name\": \"wiki_search\", \"title\": \"Search "
           "Wikipedia\", \"description\": \"The pages of Wikipedia that "
           "match some words, best first: their titles, each with what it "
           "is about in a line.\", \"inputSchema\": {\"type\": \"object\", "
           "\"properties\": {");
    schema_str(b, "query", "A name, a subject, a few words.");
    janas_buf_puts(b, ", ");
    lang_prop(b);
    janas_buf_puts(
        b, ", \"limit\": {\"type\": \"integer\", \"minimum\": 1, "
           "\"maximum\": 20, \"description\": \"How many pages (default "
           "8).\"}}, \"required\": [\"query\"], \"additionalProperties\": "
           "false}}, {\"name\": \"wiki_page\", \"title\": \"A page of "
           "Wikipedia\", \"description\": \"Shows the user a page of "
           "Wikipedia as it is: a short one whole, a long one its "
           "introduction, or the section asked for. You get a note of it "
           "(its address, its sections), not the text.\", "
           "\"inputSchema\": {\"type\": \"object\", \"properties\": {");
    schema_str(b, "title", "The page's title, as a search gives it.");
    janas_buf_puts(b, ", ");
    schema_str(b, "section", "A section, by the name the note lists.");
    janas_buf_puts(b, ", ");
    lang_prop(b);
    janas_buf_puts(
        b, "}, \"required\": [\"title\"], \"additionalProperties\": "
           "false}}, {\"name\": \"wiki_nearby\", \"title\": \"Wikipedia "
           "around a place\", \"description\": \"The pages of Wikipedia "
           "about what lies around a place - monuments, churches, "
           "museums, stations, villages - nearest first, with how far and "
           "which way.\", \"inputSchema\": {\"type\": \"object\", "
           "\"properties\": {");
    schema_str(b, "place",
               "A place, as its page is titled, or \"lat,lon\". Leave out "
               "place and coordinates only when the user names no place: "
               "then where the user is.");
    janas_buf_puts(
        b, ", \"latitude\": {\"type\": \"number\", \"minimum\": -90, "
           "\"maximum\": 90}, \"longitude\": {\"type\": \"number\", "
           "\"minimum\": -180, \"maximum\": 180}, \"radius_km\": "
           "{\"type\": \"number\", \"minimum\": 0.1, \"maximum\": 10, "
           "\"description\": \"How far around, in km (default 2).\"}, "
           "\"limit\": {\"type\": \"integer\", \"minimum\": 1, "
           "\"maximum\": 30, \"description\": \"How many pages (default "
           "15).\"}, ");
    lang_prop(b);
    janas_buf_puts(b, "}, \"additionalProperties\": false}}]");
}

int wiki_tools_call(void *ctx, const struct janas_json *params,
                    struct janas_buf *b)
{
    (void)ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    if (janas_json_is(name, "wiki_search"))
        return tool_search(args, b);
    if (janas_json_is(name, "wiki_page"))
        return tool_page(args, b);
    if (janas_json_is(name, "wiki_nearby"))
        return tool_nearby(args, b);
    return -1;
}
