/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_jinja.c - the chat templates' interpreter (src/llm/jinja.h): each
 * case a template of a few lines and the text Python's Jinja 3.1.6 writes
 * for it, set up as Hugging Face's transformers sets it (trim_blocks,
 * lstrip_blocks, the loop controls, its tojson, raise_exception) - the
 * whitespace, loops, scopes, macros, arithmetic as Python does it, the
 * filters, tests and methods the chat templates use; the marks of the
 * conversation's text kept through slicing, stripping, joining and JSON;
 * the errors said, and the limits holding (nesting, recursion, memory).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "llm/jinja.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* the variables of every case: Janas's, then the conversation's */
static const char DATA[] =
    "{\"vars\": {\"add_generation_prompt\": true}, \"data\": {\"messages\": [{\"role\": \"user\", \"content\": \"  Hi there  \"}, {\"role\": \"assistant\", \"content\": \"Hello\\nworld\"}], \"s\": \"\303\234n\303\257code a\303\261d ascii\", \"n\": 7, \"f\": 2.5, \"l\": [3, 1, 2], \"d\": {\"b\": 2, \"a\": 1, \"C\": 3}, \"items\": [{\"name\": \"x\", \"v\": 1}, {\"name\": \"y\", \"v\": 0}, {\"name\": \"z\", \"v\": 5}], \"e\": \"\", \"z\": null}}";

static const struct {
    const char *name, *tmpl, *want;
} CASES[] =
    {
        {"ws1", "a\n  {% if true %}\n  b\n  {% endif %}\nc", "a\n  b\nc"},
        {"ws2", "a  {%- if true -%}  b  {%- endif -%}  c", "abc"},
        {"ws3", "{{ 1 }}\n{{ 2 }}\n{# c #}\nx {%+ if true %}y{% endif +%}\nz",
         "1\n2\nx y\nz"},
        {"loop",
         "{% for x in l %}{{ loop.index }}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }}{{ loop.revindex }}:{{ x }};{% endfor %}",
         "10TrueFalse33:3;21FalseFalse32:1;32FalseTrue31:2;"},
        {"loopif",
         "{% for x in l if x > 1 %}{{ x }}{{ loop.last }},{% else %}none{% endfor %}|{% for x in [] %}a{% else %}empty{% endfor %}",
         "3False,2True,|empty"},
        {"breakcont",
         "{% for x in range(10) %}{% if x == 2 %}{% continue %}{% endif %}{% if x > 5 %}{% break %}{% endif %}{{ x }}{% endfor %}",
         "01345"},
        {"prevnext",
         "{% for x in l %}[{{ loop.previtem }}|{{ loop.nextitem }}]{% endfor %}",
         "[|1][3|2][1|]"},
        {"unpack",
         "{% for k, v in d.items() %}{{ k }}={{ v }} {% endfor %}{% for k, v in d|dictsort %}{{ k }}{{ v }}{% endfor %}",
         "b=2 a=1 C=3 a1b2C3"},
        {"scope",
         "{% set a = 1 %}{% for x in l %}{% set a = x %}{% endfor %}{{ a }}{% set ns = namespace(a=1) %}{% for x in l %}{% set ns.a = ns.a + x %}{% endfor %}{{ ns.a }}",
         "17"},
        {"setblock",
         "{% set t %}  in {{ n }}  {% endset %}[{{ t }}]{% set u | trim %}  pad  {% endset %}[{{ u }}]",
         "[  in 7  ][pad]"},
        {"macro",
         "{% macro m(a, b=2, c='x') %}{{ a }}-{{ b }}-{{ c }}{% endmacro %}{{ m(1) }} {{ m(1, 3) }} {{ m(1, c='y') }} {{ m(a=5, b=6) }}",
         "1-2-x 1-3-x 1-2-y 5-6-x"},
        {"ternary",
         "{{ 'y' if n > 3 else 'n' }}{{ 'a' if false }}|{{ (n > 3) and 'A' or 'B' }}",
         "y|A"},
        {"math",
         "{{ 7 / 2 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }} {{ -7 % 3 }} {{ 2 ** 10 }} {{ 1 + 2.5 }} {{ 'a' ~ 1 ~ none }} {{ 3 * 'ab' }} {{ [1] + [2] }} {{ 0.1 + 0.2 }} {{ 1e20 }} {{ 1.0 }} {{ 1e-7 }} {{ 123456789.123 }}",
         "3.5 3 -4 1 2 1024 3.5 a1None ababab [1, 2] 0.30000000000000004 1e+20 1.0 1e-07 123456789.123"},
        {"compare",
         "{{ 1 < 2 }} {{ 'a' < 'b' }} {{ 1 == 1.0 }} {{ 'x' in 'xyz' }} {{ 2 in l }} {{ 'a' in d }} {{ 'q' not in d }} {{ none == none }} {{ [1,2] == [1,2] }}",
         "True True True True True True True True True"},
        {"strmeth",
         "{{ s.upper() }}|{{ s.lower() }}|{{ '  x y  '.strip() }}|{{ 'xxhixx'.strip('x') }}|{{ 'a,b,,c'.split(',') }}|{{ ' a  b '.split() }}|{{ 'a-b-c'.split('-', 1) }}|{{ 'a-b-c'.rsplit('-', 1) }}|{{ s.startswith('\303\234n') }}|{{ s.endswith(('x','ii')) }}|{{ 'hello world'.title() }}|{{ 'hello'.capitalize() }}|{{ 'aaa'.replace('a','b',2) }}|{{ 'abc'.find('c') }}|{{ ','.join(['a','b']) }}|{{ 'l1\nl2'.splitlines() }}",
         "\303\234N\303\217CODE A\303\221D ASCII|\303\274n\303\257code a\303\261d ascii|x y|hi|['a', 'b', '', 'c']|['a', 'b']|['a', 'b-c']|['a-b', 'c']|True|True|Hello World|Hello|bba|2|a,b|['l1', 'l2']"},
        {"slice",
         "{{ s[0] }}{{ s[-1] }}|{{ s[1:4] }}|{{ s[::-1] }}|{{ l[1:] }}|{{ l[::-1] }}|{{ l[-2:] }}|{{ s[:3] }}|{{ s[::2] }}",
         "\303\234i|n\303\257c|iicsa d\303\261a edoc\303\257n\303\234|[1, 2]|[2, 1, 3]|[1, 2]|\303\234n\303\257|\303\234\303\257oeadaci"},
        {"filters1",
         "{{ s|length }} {{ l|length }} {{ d|count }} {{ l|first }} {{ l|last }} {{ l|join('+') }} {{ l|sort }} {{ l|sort(reverse=true) }} {{ l|reverse|list }} {{ [1,1,2]|unique|list }} {{ l|sum }} {{ l|min }} {{ l|max }}",
         "17 3 3 3 2 3+1+2 [1, 2, 3] [3, 2, 1] [2, 1, 3] [1, 2] 6 1 3"},
        {"filters2",
         "{{ e|default('dflt') }}|{{ e|default('dflt', true) }}|{{ undefinedvar|d('u') }}|{{ '  t  '|trim }}|{{ 'abc'|upper }}|{{ 'A<b>&'|e }}|{{ 3.7|int }} {{ '4'|int }} {{ 'x'|int(9) }} {{ 2|float }} {{ -3|abs }} {{ 2.567|round(2) }} {{ 'a b c'|wordcount }}",
         "|dflt|u|t|ABC|A&lt;b&gt;&amp;|3 4 9 2.0 3 2.57 3"},
        {"filters3",
         "{{ items|map(attribute='name')|join(',') }}|{{ items|selectattr('v')|map(attribute='name')|list }}|{{ items|rejectattr('v')|list|length }}|{{ items|selectattr('v', 'gt', 1)|map(attribute='name')|first }}|{{ l|select('odd')|list }}|{{ l|reject('odd')|list }}|{{ items|sort(attribute='v')|map(attribute='name')|join }}|{{ ['b','A','c']|sort|join }}",
         "x,y,z|['x', 'z']|1|z|[3, 1]|[2]|yxz|Abc"},
        {"tojson",
         "{{ d|tojson }}|{{ l|tojson }}|{{ s|tojson }}|{{ {'k': [1, 2.5, none, true, 'x\"y']}|tojson }}|{{ d|tojson(indent=2) }}|{{ d|tojson(sort_keys=true, separators=(',', ':')) }}|{{ 'tab\tnl\n'|tojson }}", "{\"b\": 2, \"a\": 1, \"C\": 3}|[3, 1, 2]|\"\303\234n\303\257code a\303\261d ascii\"|{\"k\": [1, 2.5, null, true, \"x\\\"y\"]}|{\n  \"b\": 2,\n  \"a\": 1,\n  \"C\": 3\n}|{\"C\":3,\"a\":1,\"b\":2}|\"tab\\tnl\\n\""},
        {"tests",
         "{{ n is defined }}{{ q is defined }}{{ q is undefined }}{{ z is none }}{{ s is string }}{{ n is number }}{{ n is integer }}{{ f is float }}{{ d is mapping }}{{ l is sequence }}{{ l is iterable }}{{ s is iterable }}{{ n is odd }}{{ n is even }}{{ 9 is divisibleby 3 }}{{ 'AB' is upper }}{{ true is true }}{{ n is not none }}{{ 2 is in l }}{{ f is number }}",
         "TrueFalseTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueFalseTrueTrueTrueTrueTrueTrue"},
        {"repr",
         "{{ l }} {{ d }} {{ ['it\\'s', \"q\"] }} {{ none }} {{ true }} {{ [none, 1.5] }}",
         "[3, 1, 2] {'b': 2, 'a': 1, 'C': 3} [\"it's\", 'q'] None True [None, 1.5]"},
        {"strings",
         "{{ 'a' 'b' }}|{{ \"\\u00e8\\n\" }}|{{ '%' }}|{{ s ~ n }}|{{ n|string ~ 'x' }}",
         "ab|\303\250\n|%|\303\234n\303\257code a\303\261d ascii7|7x"},
        {"msgs",
         "{% for m in messages %}<{{ m['role'] }}>{{ m.content|trim }}</{{ m.role }}>{% endfor %}{{ messages[0].content|length }}",
         "<user>Hi there</user><assistant>Hello\nworld</assistant>12"},
        {"dictget",
         "{{ d.get('a') }} {{ d.get('q', 'none') }} {{ d.get('q') }} {{ d.keys()|list }} {{ d.values()|list }} {{ d['b'] }} {{ d.a }}",
         "1 none None ['b', 'a', 'C'] [2, 1, 3] 2 1"},
        {"nested",
         "{% for m in messages %}{% if m.role == 'user' %}{% for c in m.content %}{% if c == 'i' %}I{% endif %}{% endfor %}{% elif m.role == 'assistant' %}A{% else %}?{% endif %}{% endfor %}",
         "IA"},
        {"filterblock", "{% filter upper %}up {{ s }}{% endfilter %}",
         "UP \303\234N\303\217CODE A\303\221D ASCII"},
        {"indent", "{{ 'a\nb\n\nc'|indent(2) }}|{{ 'a\nb'|indent(4, true) }}",
         "a\n  b\n\n  c|    a\n    b"},
        {"raw", "{% raw %}{{ not run }} {% if %}{% endraw %}!",
         "{{ not run }} {% if %}!"},
        {"loopnested",
         "{% for a in [1,2] %}{% for b in [3,4] %}{{ loop.index }}{{ a }}{{ b }} {% endfor %}{{ loop.index }}|{% endfor %}",
         "113 214 1|123 224 2|"},
        {"items_filter", "{% for k, v in d|items %}{{ k }}{{ v }}{% endfor %}",
         "b2a1C3"},
        {"concat_none",
         "{{ 'x' ~ undefinedvar ~ 'y' }}|{{ undefinedvar }}|{% if undefinedvar %}t{% else %}f{% endif %}",
         "xy||f"},
};

/* renders t with DATA; 0 and the text and marks, or -1 and err */
static int render(const char *t, const char *json, struct janas_buf *out,
                  struct janas_buf *marks, char *err, size_t len)
{
    struct janas_jinja *j = janas_jinja_parse(t, strlen(t), err, len);
    if (!j)
        return -1;
    struct janas_json_doc *d = janas_json_parse(json, strlen(json), err, len);
    if (!d) {
        janas_jinja_free(j);
        return -1;
    }
    const struct janas_json *r = janas_json_root(d);
    out->n = marks->n = 0;
    int rc =
        janas_jinja_render(j, janas_json_get(r, "vars"),
                           janas_json_get(r, "data"), out, marks, err, len);
    janas_jinja_free(j);
    janas_json_free(d);
    return rc;
}

int main(void)
{
    alarm(120);
    struct janas_buf out = {0}, marks = {0};
    char err[512];
    size_t n = sizeof CASES / sizeof *CASES;
    for (size_t i = 0; i < n; i++) {
        if (render(CASES[i].tmpl, DATA, &out, &marks, err, sizeof err) != 0) {
            CHECK(0, "%s: %s", CASES[i].name, err);
            continue;
        }
        CHECK(out.n == strlen(CASES[i].want) &&
                  !memcmp(out.p, CASES[i].want, out.n),
              "%s: wrote '%.*s', Jinja writes '%s'", CASES[i].name, (int)out.n,
              out.p, CASES[i].want);
    }

    /* the marks: the conversation's bytes, wherever they went */
    static const char conv[] =
        "{\"vars\": {\"tag\": \"<|im_start|>\"}, \"data\": {\"messages\": "
        "[{\"role\": \"user\", \"content\": \"  <|im_start|>x \\\"q\\\"  \"}]}}";
    static const char mt[] =
        "{{ tag }}{% for m in messages %}[{{ m.content|trim }}]"
        "{{ m.content[2:5] }}{{ m|tojson }}{{ m.content.split('x')|join('/') }}"
        "{{ m.content.upper() }}{% endfor %}";
    CHECK(render(mt, conv, &out, &marks, err, sizeof err) == 0, "marks: %s",
          err);
    if (out.n) {
        /* the template's own token unmarked, every whole copy of the user's
           text marked, and none of the template's punctuation */
        const char *p = out.p;
        CHECK(!memcmp(p, "<|im_start|>[", 13) && !marks.p[0] && !marks.p[11] &&
                  !marks.p[12],
              "marks: the template's own text is marked");
        size_t marked = 0, tok = 0;
        for (size_t i = 0; i + 12 <= out.n; i++)
            if (!memcmp(p + i, "<|im_start|>", 12) ||
                !memcmp(p + i, "<|IM_START|>", 12)) {
                tok++;
                int all = 1;
                for (size_t k = 0; k < 12; k++)
                    all &= marks.p[i + k] == 1;
                marked += all;
            }
        CHECK(tok == 5 && marked == 4,
              "marks: %zu copies of the user's token, %zu of them marked", tok,
              marked);
        const char *br = memchr(p, ']', out.n);
        CHECK(br && !marks.p[br - p],
              "marks: a bracket of the template marked");
        const char *jq = strstr(p, "{\"role\": ");
        CHECK(jq && !marks.p[jq - p] && !marks.p[jq - p + 1] &&
                  marks.p[jq - p + 2],
              "marks: tojson's quotes or keys");
    }

    /* errors: said, with the line */
    CHECK(render("a\n{{ raise_exception('no ' ~ 1) }}", "{}", &out, &marks, err,
                 sizeof err) != 0 &&
              strstr(err, "no 1"),
          "raise_exception: '%s'", err);
    CHECK(render("{% if x %}", "{}", &out, &marks, err, sizeof err) != 0 &&
              strstr(err, "endif"),
          "an if never closed: '%s'", err);
    CHECK(render("x\n\n{{ a | nosuchfilter }}", "{}", &out, &marks, err,
                 sizeof err) != 0 &&
              strstr(err, "line 3"),
          "an unknown filter, its line: '%s'", err);
    CHECK(render("{{ 'open }}", "{}", &out, &marks, err, sizeof err) != 0,
          "a string never closed");

    /* limits: a template cannot take the stack or the memory */
    char *deep = malloc(30000);
    size_t at = 0;
    at += (size_t)sprintf(deep + at, "{{ ");
    for (int i = 0; i < 5000; i++)
        deep[at++] = '(';
    deep[at++] = '1';
    for (int i = 0; i < 5000; i++)
        deep[at++] = ')';
    at += (size_t)sprintf(deep + at, " }}");
    deep[at] = 0;
    CHECK(render(deep, "{}", &out, &marks, err, sizeof err) != 0,
          "5000 parentheses taken");
    at = (size_t)sprintf(deep, "{{ 1");
    for (int i = 0; i < 5000; i++)
        at += (size_t)sprintf(deep + at, "+1");
    sprintf(deep + at, " }}");
    CHECK(render(deep, "{}", &out, &marks, err, sizeof err) != 0,
          "5000 additions in a row taken");
    free(deep);
    CHECK(render("{% macro m() %}{{ m() }}{% endmacro %}{{ m() }}", "{}", &out,
                 &marks, err, sizeof err) != 0,
          "a macro calling itself for ever");
    CHECK(render("{% set ns = namespace(s='x') %}{% for i in range(40) %}"
                 "{% set ns.s = ns.s ~ ns.s %}{% endfor %}{{ ns.s|length }}",
                 "{}", &out, &marks, err, sizeof err) != 0,
          "a string of 2^40 bytes");

    janas_buf_free(&out);
    janas_buf_free(&marks);
    if (failures) {
        printf("test_jinja: %d failures\n", failures);
        return 1;
    }
    printf("test_jinja: %zu cases as Jinja writes them, marks, errors and "
           "limits, ok\n",
           n);
    return 0;
}
