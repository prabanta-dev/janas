/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - janas-prices' tools (see prices.h) and what they share. They
 * only read, and say so (readOnlyHint): currencies, crypto-assets, shares
 * (with a free key of the user's), inflation, central banks' rates,
 * electricity and fuel, each for as many countries as open data covers.
 */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "services/common/template.h"
#include "prices.h"

#define READS                                                                  \
    "\"annotations\": {\"readOnlyHint\": true, \"openWorldHint\": true}"
#define PLACE_ARG                                                              \
    "\"place\": {\"type\": \"string\", \"description\": \"a town or an "       \
    "address with its town; leave it out for where the user is\"}"

void pr_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b,
        "\"tools\": [{\"name\": \"prices_currency\", \"title\": "
        "\"Currencies\", \"description\": \"An amount in other currencies "
        "at the ECB's reference rates, of today or of a day.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": {\"from\": "
        "{\"type\": \"string\", \"description\": \"ISO code, EUR by "
        "default\"}, \"to\": {\"type\": \"string\", \"description\": \"ISO "
        "codes, comma-separated\"}, \"amount\": {\"type\": \"number\"}, "
        "\"date\": {\"type\": \"string\", \"description\": \"YYYY-MM-DD; "
        "leave it out for the latest\"}}, \"additionalProperties\": false}, " READS
        "}, "
        "{\"name\": \"prices_crypto\", \"title\": \"Crypto-assets\", "
        "\"description\": \"The price of a crypto-asset now, and its change "
        "in 24 hours.\", \"inputSchema\": {\"type\": \"object\", "
        "\"properties\": {\"coin\": {\"type\": \"string\", \"description\": "
        "\"name or symbol: bitcoin, ETH\"}, \"currency\": {\"type\": "
        "\"string\", \"description\": \"EUR by default\"}}, \"required\": "
        "[\"coin\"], \"additionalProperties\": false}, " READS "}, "
        "{\"name\": \"prices_stock\", \"title\": \"Shares\", "
        "\"description\": \"A share's last price and its day, from its "
        "symbol (AAPL, TSCO.LON) or its company's name.\", \"inputSchema\": "
        "{\"type\": \"object\", \"properties\": {\"share\": {\"type\": "
        "\"string\"}}, \"required\": [\"share\"], \"additionalProperties\": "
        "false}, " READS "}, "
        "{\"name\": \"prices_inflation\", \"title\": \"Inflation\", "
        "\"description\": \"A country's inflation: monthly in Europe, yearly "
        "elsewhere.\", \"inputSchema\": {\"type\": \"object\", "
        "\"properties\": {\"country\": {\"type\": \"string\", "
        "\"description\": \"ISO code (IT, US) or EA for the euro area; "
        "leave it out for the user's\"}}, \"additionalProperties\": false}, " READS
        "}, "
        "{\"name\": \"prices_rates\", \"title\": \"Central banks' rates\", "
        "\"description\": \"The rates of a central bank.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": {\"bank\": "
        "{\"type\": \"string\", \"enum\": [\"ecb\", \"fed\", \"boe\", "
        "\"snb\"], \"description\": \"leave it out for the user's\"}}, "
        "\"additionalProperties\": false}, " READS "}, "
        "{\"name\": \"prices_electricity\", \"title\": \"Electricity\", "
        "\"description\": \"The wholesale price of electricity hour by hour, "
        "where it is open data.\", \"inputSchema\": {\"type\": \"object\", "
        "\"properties\": {" PLACE_ARG ", \"day\": {\"type\": \"string\", "
        "\"enum\": [\"today\", \"tomorrow\"]}}, \"additionalProperties\": "
        "false}, " READS "}, "
        "{\"name\": \"prices_fuel\", \"title\": \"Fuel\", \"description\": "
        "\"The cheapest fuel stations near a place, with their prices.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": "
        "{" PLACE_ARG ", \"fuel\": {\"type\": \"string\", \"enum\": "
        "[\"petrol\", \"diesel\", \"lpg\", \"cng\"]}, \"radius\": {\"type\": "
        "\"number\", \"description\": \"km, 5 by default, at most 20\"}}, "
        "\"required\": [\"fuel\"], \"additionalProperties\": false}, " READS
        "}]");
}

const char *pr_arg_str(const struct janas_json *args, const char *name)
{
    const char *s = janas_json_str(janas_json_get(args, name));
    return s && *s ? s : NULL;
}

double pr_arg_num(const struct janas_json *args, const char *name, double def)
{
    const struct janas_json *v = janas_json_get(args, name);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, def) : def;
}

int pr_fail(struct janas_buf *b, const char *fmt, ...)
{
    char line[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    janas_mcps_text_result(b, line, strlen(line), 1);
    return 0;
}

int pr_answer(struct janas_buf *b, const char *name, struct janas_buf *d,
              const char *layout, const char *brief)
{
    char err[300];
    int oom = d->oom;
    int r = oom ? -1
                : janas_tpl_result(b, name, d->p, d->n, layout, brief, err,
                                   sizeof err);
    janas_buf_free(d);
    if (r != 0)
        return pr_fail(b, "The answer could not be written: %s.",
                       oom ? "out of memory" : err);
    return 0;
}

int pr_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    (void)ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    static const struct {
        const char *name;
        int (*fn)(const struct janas_json *, struct janas_buf *);
    } T[] = {{"prices_currency", pr_tool_currency},
             {"prices_crypto", pr_tool_crypto},
             {"prices_stock", pr_tool_stock},
             {"prices_inflation", pr_tool_inflation},
             {"prices_rates", pr_tool_rates},
             {"prices_electricity", pr_tool_power},
             {"prices_fuel", pr_tool_fuel}};
    for (size_t i = 0; i < sizeof T / sizeof *T; i++)
        if (janas_json_is(name, T[i].name))
            return T[i].fn(args, b);
    return -1;
}
