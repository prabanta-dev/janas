/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layouts.c - the layouts of janas-prices' answers (see prices.h and
 * services/common/template.h): the text the user reads, filled with the data in
 * English here and in the user's language by a client that translates the
 * layout once (janas-chat), and the brief, all the model reads of it.
 * Every answer says where its figures come from, how old they are, and
 * that they are for information only.
 */
#include "prices.h"

#define MON_WORDS                                                              \
    "mon.1 = January\n"                                                        \
    "mon.2 = February\n"                                                       \
    "mon.3 = March\n"                                                          \
    "mon.4 = April\n"                                                          \
    "mon.5 = May\n"                                                            \
    "mon.6 = June\n"                                                           \
    "mon.7 = July\n"                                                           \
    "mon.8 = August\n"                                                         \
    "mon.9 = September\n"                                                      \
    "mon.10 = October\n"                                                       \
    "mon.11 = November\n"                                                      \
    "mon.12 = December\n"

#define DIR_WORDS                                                              \
    "dir.N = N\n"                                                              \
    "dir.NNE = NNE\n"                                                          \
    "dir.NE = NE\n"                                                            \
    "dir.ENE = ENE\n"                                                          \
    "dir.E = E\n"                                                              \
    "dir.ESE = ESE\n"                                                          \
    "dir.SE = SE\n"                                                            \
    "dir.SSE = SSE\n"                                                          \
    "dir.S = S\n"                                                              \
    "dir.SSW = SSW\n"                                                          \
    "dir.SW = SW\n"                                                            \
    "dir.WSW = WSW\n"                                                          \
    "dir.W = W\n"                                                              \
    "dir.WNW = WNW\n"                                                          \
    "dir.NW = NW\n"                                                            \
    "dir.NNW = NNW\n"

#define HOW_WORDS                                                              \
    "how.as set on the computer (JANAS_LOCATION) = as set on the computer "    \
    "(JANAS_LOCATION)\n"                                                       \
    "how.estimated from the internet connection (GeoJS) = estimated from "     \
    "the internet connection (GeoJS)\n"                                        \
    "how.estimated from the internet connection (ipwho.is) = estimated from "  \
    "the internet connection (ipwho.is)\n"                                     \
    "how.a rough guess, the city of the computer's time zone = a rough "       \
    "guess, the city of the computer's time zone\n"

#define INFO "For information only, not advice to buy or sell."

const char PR_FX_LAYOUT[] =
    "{{amount}} {{from}}, at the exchange rates of {{date.day}} "
    "{{date.mon|mon}} {{date.year}}:\n"
    "{{#items}}\n"
    "  {{value}} {{code}} (1 {{from}} = {{rate}} {{code}})\n"
    "{{/items}}\n"
    "Reference exchange rates: {{source|src}}, set once a working day: "
    "a bank or an exchange office applies its own. " INFO "\n"
    "---\n" MON_WORDS
    "src.ecb_frankfurter = European Central Bank, through Frankfurter\n"
    "src.ecb = European Central Bank\n";

const char PR_FX_BRIEF[] =
    "{{amount}} {{from}} = {{#items}}{{value}} {{code}} (exchange rate "
    "{{rate}}); {{/items}}reference exchange rates of "
    "{{date.day}}/{{date.mon}}/{{date.year}} ({{source}}). Shown to "
    "the user: answer in a line, in the user's language.";

const char PR_COIN_LAYOUT[] =
    "{{name}} ({{symbol}}): {{price}} {{cur}}{{#has_change}}, "
    "{{#up}}+{{/up}}{{change}}% in 24 "
    "hours{{/has_change}}{{#has_cap_bn}}; market value {{cap_bn}} "
    "billion {{cur}}{{/has_cap_bn}}{{#at.day}}; at {{at.at}} of "
    "{{at.day}} {{at.mon|mon}}{{/at.day}}.\n"
    "Data provided by {{source|src}}. The prices of crypto-assets "
    "swing widely. " INFO "\n"
    "---\n" MON_WORDS
    "src.coingecko = CoinGecko (https://www.coingecko.com/en/api)\n"
    "src.coinbase = Coinbase\n";

const char PR_COIN_BRIEF[] =
    "{{name}} ({{symbol}}): {{price}} {{cur}}{{#has_change}}, "
    "{{change}}% in 24 hours{{/has_change}}; source {{source}}. Shown "
    "to the user: answer in a line, in the user's language.";

const char PR_QUOTE_LAYOUT[] =
    "{{name}} ({{symbol}}{{#region}}, {{region}}{{/region}}): "
    "{{price}} {{currency}}{{#has_change}}, {{#up}}+{{/up}}{{change}} "
    "({{#up}}+{{/up}}{{pct}}%){{/has_change}}; the trading day of "
    "{{day.day}} {{day.mon|mon}} {{day.year}}\n"
    "open {{open}}, high {{high}}, low {{low}}, previous close "
    "{{prev}}{{#volume}}, shares traded {{volume}}{{/volume}}\n"
    "From Alpha Vantage: the prices may be delayed. " INFO "\n"
    "---\n" MON_WORDS;

const char PR_QUOTE_BRIEF[] =
    "{{name}} ({{symbol}}{{#region}}, {{region}}{{/region}}): "
    "{{price}} {{currency}}, {{change}} ({{pct}}%), trading day "
    "{{day.day}}/{{day.mon}}/{{day.year}}. Shown to the user: answer "
    "in a line, in the user's language.";

const char PR_SERIES_LAYOUT[] =
    "{{what|what}}{{#who}}, {{who}}{{/who}}:\n"
    "{{#items}}\n"
    "  {{when}}: {{v}}{{unit}}\n"
    "{{/items}}\n"
    "{{#has_min}}minimum {{min}}{{unit}} ({{min_when}}), maximum "
    "{{max}}{{unit}} ({{max_when}}), average {{avg}}{{unit}}\n"
    "{{/has_min}}{{#note}}{{note|note}}\n"
    "{{/note}}Source: {{source|src}}{{#licence}}, "
    "{{licence}}{{/licence}}. " INFO "\n"
    "---\n"
    "what.inflation = Inflation, the yearly change of consumer prices\n"
    "what.power = Electricity, the wholesale price on the day-ahead "
    "market\n"
    "src.eurostat = Eurostat, the harmonised index of consumer prices "
    "(HICP)\n"
    "src.worldbank = World Bank (CC BY 4.0), consumer prices\n"
    "src.energy_charts = energy-charts.info (Fraunhofer ISE)\n"
    "note.flash = The last month may be a first estimate.\n"
    "note.yearly = Yearly figures: the latest may be one or two years "
    "back.\n"
    "note.exchange = The price on the power exchange, before taxes, "
    "charges and the seller's margin: not what a household pays.\n";

const char PR_SERIES_BRIEF[] =
    "{{what}}{{#who}}, {{who}}{{/who}}: last {{last.v}}{{unit}} "
    "({{last.when}}); minimum {{min}} ({{min_when}}), maximum {{max}} "
    "({{max_when}}), average {{avg}}. {{#note}}Note: {{note}}. "
    "{{/note}}Shown to the user with every value: answer in a line or "
    "two, in the user's language.";

const char PR_RATES_LAYOUT[] =
    "{{bank|bank}}:\n"
    "{{#items}}\n"
    "  {{name|rate}}: {{v}}% ({{when.day}} {{when.mon|mon}} "
    "{{when.year}})\n"
    "{{/items}}\n"
    "{{#range}}  the target range of the federal funds: "
    "{{lo}}-{{hi}}%\n"
    "{{/range}}Source: {{source|src}}. " INFO "\n"
    "---\n" MON_WORDS "rate.deposit = deposit facility rate\n"
    "rate.mro = main refinancing operations\n"
    "rate.mlf = marginal lending facility\n"
    "rate.estr = euro short-term rate (€STR)\n"
    "rate.bank = Bank Rate\n"
    "rate.policy = policy rate\n"
    "rate.effr = effective federal funds rate\n"
    "bank.ecb = European Central Bank\n"
    "bank.fed = Federal Reserve\n"
    "bank.boe = Bank of England\n"
    "bank.snb = Swiss National Bank\n"
    "src.ecb_portal = ECB data portal\n"
    "src.nyfed = Federal Reserve Bank of New York\n"
    "src.boe = Bank of England database\n"
    "src.snb = Swiss National Bank data portal\n";

const char PR_RATES_BRIEF[] =
    "{{bank}}: {{#items}}{{name}} {{v}}% "
    "({{when.day}}/{{when.mon}}/{{when.year}}); "
    "{{/items}}{{#range}}target range {{lo}}-{{hi}}%. {{/range}}Shown "
    "to the user: answer in a line, in the user's language.";

const char PR_FUEL_LAYOUT[] =
    "{{kind|kind}} near {{place.name}}{{#place.how}} (where the user "
    "is, {{place.how|how}}){{/place.how}}{{#place.country}}, "
    "{{place.country}}{{/place.country}}: {{#found}}stations within "
    "{{radius}} km, {{found}}; the cheapest {{cheapest}} "
    "{{unit}}{{#has_average}}, the average {{average}} "
    "{{unit}}{{/has_average}}{{/found}}{{^found}}no station within "
    "{{radius}} km sells it{{/found}}{{#stale}}; left out, as their "
    "price was not told in the last {{stale_days}} days: "
    "{{stale}}{{/stale}}\n"
    "{{#items}}\n"
    "{{@n}}. {{price}} {{unit}}{{#self}} "
    "(self-service){{/self}}{{#served}} (served){{/served}}: "
    "{{#name}}{{name}}, {{/name}}{{address}}{{#city}}, "
    "{{city}}{{/city}} ({{#km}}{{km}} km{{/km}}{{#m}}{{m}} m{{/m}} "
    "{{dir|dir}}{{#updated}}; price of {{updated}}{{/updated}})\n"
    "{{/items}}\n"
    "Source: {{source|src}} ({{fuel}}); the prices as each station "
    "told them. " INFO "\n"
    "---\n" DIR_WORDS HOW_WORDS "kind.petrol = Petrol\n"
    "kind.diesel = Diesel\n"
    "kind.lpg = LPG\n"
    "kind.cng = Methane (CNG)\n"
    "src.mimit = Italian Ministry of Enterprises (MIMIT), the day's "
    "file\n"
    "src.fr = French government (prix-carburants), as they come\n"
    "src.es = Spanish Ministry for the Ecological Transition\n"
    "src.at = E-Control, Austria's fuel price calculator, which gives "
    "some stations, the cheapest\n";

const char PR_FUEL_BRIEF[] =
    "{{kind}} near {{place.name}}{{#place.how}} (where the user is, "
    "{{place.how}}: say which place, and that it may be "
    "off){{/place.how}}: {{found}} stations within {{radius}} km, the "
    "cheapest {{cheapest}} {{unit}}{{#has_average}}, average "
    "{{average}}{{/has_average}}{{#stale}}; {{stale}} left out, their "
    "price older than {{stale_days}} days{{/stale}}. "
    "{{#items}}{{price}} {{name}} {{address}} ({{#km}}{{km}} "
    "km{{/km}}{{#m}}{{m}} m{{/m}}); {{/items}}The list is shown to the "
    "user: answer in a line or two, in the user's language.";
