/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-prices: prices and finance as an MCP server over stdio,
 * for information only, for janas-chat (which starts it by itself) and any
 * other MCP client: currencies, crypto-assets, shares, inflation, central
 * banks' rates, electricity, fuel. Open data without a key, but for
 * shares, which need a free key of the user's.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/mcp_server.h"
#include "prices.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static const char INSTRUCTIONS[] =
    "Use these tools by yourself when the user asks about a price or a "
    "figure of money: what a currency is worth, a crypto-asset, a share, "
    "inflation, a central bank's rates, electricity, the cheapest fuel near "
    "a place; never name them. The figures are for information only: never "
    "advise buying or selling. With no place or country named, leave it "
    "out: the tool takes where the user is; say which place.";

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-prices [--config FILE]\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - prices_currency, prices_crypto, prices_stock,\n"
            "prices_inflation, prices_rates, prices_electricity and\n"
            "prices_fuel. For information only.\n"
            "Open data without a key: the ECB (through Frankfurter),\n"
            "CoinGecko and Coinbase, Eurostat and the World Bank, the ECB,\n"
            "the New York Fed, the Bank of England and the SNB,\n"
            "energy-charts.info (only the zones it gives under CC BY), the\n"
            "fuel prices of Italy (MIMIT), France, Spain and Austria.\n"
            "Shares need a free key of Alpha Vantage\n"
            "(https://www.alphavantage.co/support/#api-key): the line\n"
            "alphavantage_key = KEY  in FILE (by default\n"
            "~/.config/janas/prices.conf), or JANAS_ALPHAVANTAGE_KEY.\n"
            "janas-chat starts it by itself when it is beside it.\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"prices\": {\"command\": "
            "\"janas-prices\"}}}\n");
}

int main(int argc, char **argv)
{
    const char *config = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config = argv[++i];
            continue;
        }
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    char path[4096];
    if (!config) {
        const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
        if (xdg && *xdg)
            snprintf(path, sizeof path, "%s/janas/prices.conf", xdg);
        else
            snprintf(path, sizeof path, "%s/.config/janas/prices.conf",
                     home ? home : ".");
        config = path;
    }
    pr_key_config(config);
    struct janas_mcps srv = {.name = "janas-prices",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .tools_list = pr_tools_list,
                             .tools_call = pr_tools_call};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-prices: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr, "janas-prices %s: ready, 7 tools; shares: %s\n",
            JANAS_VERSION, pr_key() ? "Alpha Vantage" : "none (no key)");
    janas_mcps_run(&srv);
    pr_fetch_free();
    return 0;
}
