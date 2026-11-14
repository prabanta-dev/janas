/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * reports_table.c - the table of docs/reports.md made from the testers'
 * reports: every janas-bench output pasted into an issue or a comment of
 * the repository, as `gh` gives them, one JSON object a line
 * ({"number", "url", "created_at", "user", "body"}). A report is found by
 * its lines, which janas-bench writes the same way since its first
 * version: "janas-bench <version>, <stamp>", then "system:", "CPU:",
 * "memory:", "GPU:", "model:", and "result:" with the speeds the engine
 * chose. The table is a ranking for each model, the fastest reply first;
 * the project's own reports (the author's) are left out.
 *
 *   reports_table <items.jsonl>                print the table
 *   reports_table <items.jsonl> docs/reports.md
 *       put it between the file's lines <!-- reports:begin --> and
 *       <!-- reports:end -->, the rest of the file as it was
 *
 * The items come from (jq's filters in -q):
 *   gh api --paginate 'repos/prabanta-dev/janas/issues?state=all&per_page=100'
 *     -q '.[] | {number, url: .html_url, created_at, user: .user.login, body}'
 *   gh api --paginate 'repos/prabanta-dev/janas/issues/comments?per_page=100'
 *     -q '.[] | {number: (.issue_url | split("/") | last | tonumber),
 *                url: .html_url, created_at, user: .user.login, body}'
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/bench_report.h"
#include "llm/json.h"

#define OWNER "camauri"

struct row {
    struct janas_bench_row b;
    char date[16], url[256];
    int number;
};

static struct row *rows;
static size_t n_rows, cap_rows;

static int add_row(const struct row *r)
{
    /* the same report quoted again (a reply that pastes it) counts once */
    for (size_t i = 0; i < n_rows; i++)
        if (!strcmp(rows[i].b.model, r->b.model) &&
            !strcmp(rows[i].b.cpu, r->b.cpu) &&
            rows[i].b.decode == r->b.decode &&
            rows[i].b.prefill == r->b.prefill)
            return 0;
    if (n_rows == cap_rows) {
        cap_rows = cap_rows ? 2 * cap_rows : 64;
        struct row *t = realloc(rows, cap_rows * sizeof(*t));
        if (!t)
            return -1;
        rows = t;
    }
    rows[n_rows++] = *r;
    return 0;
}

static const char *MONTHS[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

/* The reports in one item. */
struct item {
    int number;
    const char *url, *created;
};

static int take(void *ctx, const struct janas_bench_row *b)
{
    const struct item *it = ctx;
    struct row r = {.b = *b, .number = it->number};
    snprintf(r.url, sizeof r.url, "%s", it->url);
    int y, mo, d;
    if (sscanf(it->created, "%d-%d-%d", &y, &mo, &d) == 3 && mo >= 1 &&
        mo <= 12)
        snprintf(r.date, sizeof r.date, "%d %s %d", d, MONTHS[mo - 1], y);
    return add_row(&r);
}

static int by_model_speed(const void *a, const void *b)
{
    const struct row *x = a, *y = b;
    int c = strcmp(x->b.model, y->b.model);
    if (c)
        return c;
    return (x->b.decode < y->b.decode) - (x->b.decode > y->b.decode);
}

static void table(struct janas_buf *o)
{
    qsort(rows, n_rows, sizeof(*rows), by_model_speed);
    const char *model = "";
    for (size_t i = 0; i < n_rows; i++) {
        const struct row *row = &rows[i];
        const struct janas_bench_row *r = &row->b;
        if (strcmp(r->model, model)) {
            model = r->model;
            janas_buf_printf(
                o,
                "\n### %s\n\n| Reply, tok/s | Prompt, tok/s | "
                "With MTP, tok/s | CPU | GPU | RAM | System | "
                "Janas | Report |\n|---:|---:|---:|---|---|---|---|"
                "---|---|\n",
                model);
        }
        char mtp[48] = "-";
        if (r->mtp_greedy > 0)
            snprintf(mtp, sizeof mtp, "%.1f / %.1f", r->mtp_greedy,
                     r->mtp_sampled);
        janas_buf_printf(o,
                         "| %.1f | %.1f | %s | %s | %s | %s | %s | %s | "
                         "[#%d](%s), %s |\n",
                         r->decode, r->prefill, mtp, r->cpu,
                         r->gpu[0] ? r->gpu : "-", r->ram[0] ? r->ram : "-",
                         r->sys[0] ? r->sys : "-", r->version, row->number,
                         row->url, row->date);
    }
}

static char *read_all(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    struct janas_buf b = {0};
    char tmp[65536];
    size_t k;
    while ((k = fread(tmp, 1, sizeof tmp, f)) > 0)
        janas_buf_put(&b, tmp, k);
    fclose(f);
    janas_buf_put(&b, "", 0);
    *n = b.n;
    return b.oom ? NULL : b.p;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: reports_table <items.jsonl> [reports.md]\n");
        return 2;
    }
    size_t n;
    char *s = read_all(argv[1], &n);
    if (!s) {
        fprintf(stderr, "reports_table: cannot read %s\n", argv[1]);
        return 1;
    }
    for (size_t i = 0; i < n;) {
        size_t e = i;
        while (e < n && s[e] != '\n')
            e++;
        char err[128];
        struct janas_json_doc *d =
            e > i ? janas_json_parse(s + i, e - i, err, sizeof err) : NULL;
        const struct janas_json *o = d ? janas_json_root(d) : NULL;
        const char *user = o ? janas_json_str(janas_json_get(o, "user")) : NULL;
        const struct janas_json *body = o ? janas_json_get(o, "body") : NULL;
        const char *url = o ? janas_json_str(janas_json_get(o, "url")) : NULL;
        const char *at =
            o ? janas_json_str(janas_json_get(o, "created_at")) : NULL;
        if (body && body->type == JANAS_JSON_STRING && user && url && at &&
            strcmp(user, OWNER) != 0 &&
            janas_bench_report_scan(
                body->s, body->n, take,
                &(struct item){.number = (int)janas_json_num(
                                   janas_json_get(o, "number"), 0),
                               .url = url,
                               .created = at}) < 0) {
            fprintf(stderr, "reports_table: out of memory\n");
            return 1;
        }
        janas_json_free(d);
        i = e + 1;
    }
    free(s);
    struct janas_buf t = {0};
    table(&t);
    if (t.oom)
        return 1;
    if (argc < 3) {
        fwrite(t.p ? t.p : "", 1, t.n, stdout);
        printf("\n%zu reports\n", n_rows);
        return 0;
    }
    size_t mn;
    char *md = read_all(argv[2], &mn);
    static const char B[] = "<!-- reports:begin -->",
                      E[] = "<!-- reports:end -->";
    char *b = md ? strstr(md, B) : NULL, *x = b ? strstr(b, E) : NULL;
    if (!b || !x) {
        fprintf(stderr, "reports_table: %s has no %s ... %s\n", argv[2], B, E);
        return 1;
    }
    FILE *f = fopen(argv[2], "wb");
    if (!f)
        return 1;
    fwrite(md, 1, (size_t)(b - md) + sizeof B - 1, f);
    fputc('\n', f);
    fwrite(t.p, 1, t.n, f);
    fputc('\n', f);
    fputs(x, f);
    fclose(f);
    printf("%s: %zu reports\n", argv[2], n_rows);
    return 0;
}
