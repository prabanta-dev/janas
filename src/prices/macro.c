/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * macro.c - janas-prices' inflation, central banks' rates and electricity
 * (see prices.h). Inflation from Eurostat for Europe, monthly, and from the
 * World Bank for every other country, yearly; the rates of the ECB, the
 * Fed (through the New York Fed), the Bank of England and the Swiss
 * National Bank; electricity from energy-charts (Fraunhofer ISE), only for
 * the bidding zones whose prices it gives under an open licence.
 */
#define _GNU_SOURCE /* strcasestr */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "common/geo.h"
#include "common/locate.h"
#include "prices.h"

#define DAY_KEEP_S 3600
#define MS 30000

/* The countries of Eurostat's consumer prices: the EU's, Iceland's,
   Norway's, the euro area (EA) and the Union (EU). Greece is EL there. */
static const char EUROSTAT[] = " AT BE BG CY CZ DE DK EE EL ES FI FR HR HU "
                               "IE IS IT LT LU LV MT NL NO PL PT RO SE SI "
                               "SK EA EU ";

static int in_eurostat(const char *cc)
{
    char k[6];
    snprintf(k, sizeof k, " %s ", cc);
    return strstr(EUROSTAT, k) != NULL;
}

/* A country's name from its ISO code, the code itself when unknown */
static const char *country_name(const char *cc)
{
    for (uint16_t i = 0; *geo_country_iso2(i); i++)
        if (!strcmp(geo_country_iso2(i), cc))
            return geo_country_name(i);
    return cc;
}

/* The user's country's code, "" when not known */
static void user_cc(char *cc)
{
    struct janas_where w;
    char err[200];
    cc[0] = 0;
    if (janas_where(&w, err, sizeof err) == 0)
        snprintf(cc, 3, "%s", w.cc);
}

int pr_tool_inflation(const struct janas_json *args, struct janas_buf *b)
{
    char cc[4] = "", err[400] = "", url[512];
    const char *c = pr_arg_str(args, "country");
    if (c) {
        if (strlen(c) != 2 || !isalpha((unsigned char)c[0]) ||
            !isalpha((unsigned char)c[1]))
            return pr_fail(b,
                           "\"%s\" is not a country's ISO code (IT, US, "
                           "EA for the euro area).",
                           c);
        cc[0] = (char)toupper((unsigned char)c[0]);
        cc[1] = (char)toupper((unsigned char)c[1]);
    } else {
        user_cc(cc);
        if (!cc[0])
            return pr_fail(b, "No country given, and where the user is is "
                              "not known: ask which country.");
    }
    if (!strcmp(cc, "GR"))
        snprintf(cc, sizeof cc, "EL");
    struct janas_buf body = {0}, d = {0};
    struct pr_series s;
    if (in_eurostat(cc)) {
        snprintf(url, sizeof url,
                 "https://ec.europa.eu/eurostat/api/dissemination/statistics/"
                 "1.0/data/prc_hicp_minr?geo=%s&coicop18=TOTAL&unit=RCH_A"
                 "&lastTimePeriod=13",
                 cc);
        int ok =
            pr_fetch(url, DAY_KEEP_S, MS, 1, &body, err, sizeof err) == 200 &&
            pr_read_eurostat(body.p, body.n, &s, err, sizeof err) == 0;
        janas_buf_free(&body);
        if (!ok)
            return pr_fail(b,
                           "Eurostat's inflation of %s could not be read: "
                           "%s.",
                           cc, err);
        snprintf(s.unit, sizeof s.unit, "%%");
        const char *who = !strcmp(cc, "EA")   ? "the euro area"
                          : !strcmp(cc, "EU") ? "the European Union"
                          : !strcmp(cc, "EL") ? country_name("GR")
                                              : country_name(cc);
        pr_series_data(&d, "inflation", who, &s, 13, "eurostat", "flash", "");
    } else {
        snprintf(url, sizeof url,
                 "https://api.worldbank.org/v2/country/%s/indicator/"
                 "FP.CPI.TOTL.ZG?format=json&mrv=6",
                 cc);
        char country[64] = "";
        int ok =
            pr_fetch(url, DAY_KEEP_S, MS, 1, &body, err, sizeof err) == 200 &&
            pr_read_worldbank(body.p, body.n, &s, country, sizeof country, err,
                              sizeof err) == 0;
        janas_buf_free(&body);
        if (!ok)
            return pr_fail(b,
                           "The World Bank's inflation of %s could not be "
                           "read: %s.",
                           cc, err);
        snprintf(s.unit, sizeof s.unit, "%%");
        for (int i = 0; i < s.n; i++) /* as Eurostat gives them */
            s.p[i].v = round(s.p[i].v * 10) / 10;
        pr_series_data(&d, "inflation", country, &s, 6, "worldbank", "yearly",
                       "");
    }
    return pr_answer(b, "prices_inflation", &d, PR_SERIES_LAYOUT,
                     PR_SERIES_BRIEF);
}

/* ---- central banks ---- */

static int ecb_series(const char *key, struct pr_rate *r, const char *name,
                      char *err, size_t err_len)
{
    char url[300];
    snprintf(url, sizeof url,
             "https://data-api.ecb.europa.eu/service/data/%s"
             "?lastNObservations=1&format=csvdata",
             key);
    struct janas_buf body = {0};
    struct pr_series s;
    int ok = pr_fetch(url, DAY_KEEP_S, MS, 0.5, &body, err, err_len) == 200 &&
             pr_read_ecb_csv(body.p, body.n, &s, err, err_len) == 0;
    janas_buf_free(&body);
    if (ok) {
        snprintf(r->name, sizeof r->name, "%s", name);
        snprintf(r->when, sizeof r->when, "%s", s.p[s.n - 1].when);
        r->v = s.p[s.n - 1].v;
    }
    return ok ? 0 : -1;
}

int pr_tool_rates(const struct janas_json *args, struct janas_buf *b)
{
    const char *bank = pr_arg_str(args, "bank");
    char cc[3] = "", err[400] = "";
    if (!bank) {
        user_cc(cc);
        bank = !strcmp(cc, "US")   ? "fed"
               : !strcmp(cc, "GB") ? "boe"
               : !strcmp(cc, "CH") ? "snb"
                                   : "ecb";
    }
    struct pr_rate r[4];
    int n = 0;
    double lo = NAN, hi = NAN;
    const char *name, *source;
    struct janas_buf body = {0};
    struct pr_series s;
    if (!strcmp(bank, "ecb")) {
        static const struct {
            const char *key, *name;
        } E[] = {{"FM/D.U2.EUR.4F.KR.DFR.LEV", "deposit"},
                 {"FM/D.U2.EUR.4F.KR.MRR_FR.LEV", "mro"},
                 {"FM/D.U2.EUR.4F.KR.MLFR.LEV", "mlf"},
                 {"EST/B.EU000A2X2A25.WT", "estr"}};
        for (int i = 0; i < 4; i++)
            if (ecb_series(E[i].key, &r[n], E[i].name, err, sizeof err) == 0)
                n++;
        name = "ecb";
        source = "ecb_portal";
    } else if (!strcmp(bank, "fed")) {
        if (pr_fetch("https://markets.newyorkfed.org/api/rates/unsecured/effr/"
                     "last/1.json",
                     DAY_KEEP_S, MS, 0.5, &body, err, sizeof err) == 200 &&
            pr_read_nyfed(body.p, body.n, &s, &lo, &hi, err, sizeof err) == 0) {
            snprintf(r[0].name, sizeof r[0].name, "effr");
            snprintf(r[0].when, sizeof r[0].when, "%s", s.p[s.n - 1].when);
            r[0].v = s.p[s.n - 1].v;
            n = 1;
        }
        name = "fed";
        source = "nyfed";
    } else if (!strcmp(bank, "boe")) {
        char url[400], from[16];
        time_t t = time(NULL) - 40 * 86400;
        struct tm g;
        gmtime_r(&t, &g);
        strftime(from, sizeof from, "%d/%b/%Y", &g);
        snprintf(url, sizeof url,
                 "https://www.bankofengland.co.uk/boeapps/database/"
                 "_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=%s&Dateto=now"
                 "&SeriesCodes=IUDBEDR&CSVF=TN&UsingCodes=Y",
                 from);
        if (pr_fetch(url, DAY_KEEP_S, MS, 0.5, &body, err, sizeof err) == 200 &&
            pr_read_boe_csv(body.p, body.n, &s, err, sizeof err) == 0) {
            snprintf(r[0].name, sizeof r[0].name, "bank");
            snprintf(r[0].when, sizeof r[0].when, "%s", s.p[s.n - 1].when);
            r[0].v = s.p[s.n - 1].v;
            n = 1;
        }
        name = "boe";
        source = "boe";
    } else if (!strcmp(bank, "snb")) {
        char url[300], from[16];
        time_t t = time(NULL) - 40 * 86400;
        struct tm g;
        gmtime_r(&t, &g);
        strftime(from, sizeof from, "%Y-%m", &g);
        snprintf(url, sizeof url,
                 "https://data.snb.ch/api/cube/snbgwdzid/data/json/en"
                 "?fromDate=%s",
                 from);
        if (pr_fetch(url, DAY_KEEP_S, MS, 0.5, &body, err, sizeof err) == 200 &&
            pr_read_snb(body.p, body.n, &s, err, sizeof err) == 0) {
            snprintf(r[0].name, sizeof r[0].name, "policy");
            snprintf(r[0].when, sizeof r[0].when, "%s", s.p[s.n - 1].when);
            r[0].v = s.p[s.n - 1].v;
            n = 1;
        }
        name = "snb";
        source = "snb";
    } else {
        return pr_fail(b, "No such bank: ecb, fed, boe or snb.");
    }
    janas_buf_free(&body);
    if (!n)
        return pr_fail(b, "The rates of %s could not be read: %s.", name, err);
    struct janas_buf d = {0};
    pr_rates_data(&d, name, r, n, lo, hi, source);
    return pr_answer(b, "prices_rates", &d, PR_RATES_LAYOUT, PR_RATES_BRIEF);
}

/* ---- electricity ---- */

/* The bidding zone of a place: Italy's by its region, the Nordic
   countries' roughly by the point, the others' the country */
static void zone_of(const struct pr_place *p, char *z, size_t cap)
{
    static const struct {
        const char *word, *zone;
    } IT[] = {{"sicil", "IT-Sicily"},         {"sardegna", "IT-Sardinia"},
              {"sardinia", "IT-Sardinia"},    {"calabr", "IT-Calabria"},
              {"apulia", "IT-South"},         {"puglia", "IT-South"},
              {"basilicata", "IT-South"},     {"molise", "IT-South"},
              {"lazio", "IT-Centre-South"},   {"campania", "IT-Centre-South"},
              {"abruzz", "IT-Centre-South"},  {"tuscan", "IT-Centre-North"},
              {"toscana", "IT-Centre-North"}, {"umbria", "IT-Centre-North"},
              {"marche", "IT-Centre-North"}};
    if (!strcmp(p->cc, "IT")) {
        snprintf(z, cap, "IT-North");
        for (size_t i = 0; i < sizeof IT / sizeof *IT; i++)
            if (strcasestr(p->region, IT[i].word))
                snprintf(z, cap, "%s", IT[i].zone);
    } else if (!strcmp(p->cc, "DE") || !strcmp(p->cc, "LU")) {
        snprintf(z, cap, "DE-LU");
    } else if (!strcmp(p->cc, "DK")) {
        snprintf(z, cap, p->lon < 11 ? "DK1" : "DK2");
    } else if (!strcmp(p->cc, "SE")) {
        snprintf(z, cap,
                 p->lat > 65.5   ? "SE1"
                 : p->lat > 61.5 ? "SE2"
                 : p->lat > 57.5 ? "SE3"
                                 : "SE4");
    } else if (!strcmp(p->cc, "NO")) {
        snprintf(z, cap,
                 p->lat > 66                   ? "NO4"
                 : p->lat > 62                 ? "NO3"
                 : p->lon < 7.5                ? "NO5"
                 : p->lat < 59.5 && p->lon < 9 ? "NO2"
                                               : "NO1");
    } else {
        snprintf(z, cap, "%s", p->cc);
    }
}

int pr_tool_power(const struct janas_json *args, struct janas_buf *b)
{
    struct pr_place p;
    char err[400] = "", zone[24], url[300], day[11], licence[600];
    if (pr_place_find(pr_arg_str(args, "place"), &p, err, sizeof err) != 0)
        return pr_fail(b, "%s. Ask the user which place.", err);
    if (!p.cc[0])
        return pr_fail(b, "The country of %s is not known.", p.name);
    zone_of(&p, zone, sizeof zone);
    int tomorrow = janas_json_is(janas_json_get(args, "day"), "tomorrow");
    time_t t = time(NULL) + (tomorrow ? 86400 : 0);
    struct tm lt;
    localtime_r(&t, &lt);
    strftime(day, sizeof day, "%Y-%m-%d", &lt);
    snprintf(url, sizeof url,
             "https://api.energy-charts.info/price?bzn=%s&start=%s&end=%s",
             zone, day, day);
    struct janas_buf body = {0};
    struct pr_series s;
    int st = pr_fetch(url, DAY_KEEP_S, MS, 3, &body, err, sizeof err);
    int open = st == 200 ? pr_read_power(body.p, body.n, p.tz, &s, licence,
                                         sizeof licence, err, sizeof err)
                         : -1;
    janas_buf_free(&body);
    if (st == 404 || (st == 200 && open < 0 && !licence[0]))
        return pr_fail(b,
                       "energy-charts has no prices for the zone %s (%s) "
                       "%s.",
                       zone, p.name, tomorrow ? "for tomorrow yet" : "today");
    if (open < 0)
        return pr_fail(b, "The prices of electricity could not be read: %s.",
                       st > 0 && st != 200 ? "energy-charts is busy" : err);
    if (!open)
        return pr_fail(b,
                       "The wholesale prices of the zone %s (%s) are not "
                       "open data: their source allows only private and "
                       "internal use, so they are not given. Tell the "
                       "user so.",
                       zone, p.name);
    /* the hours: quarters of an hour averaged */
    struct pr_series h;
    memset(&h, 0, sizeof h);
    snprintf(h.unit, sizeof h.unit, " €/MWh");
    for (int i = 0; i < s.n;) {
        char hour[24];
        snprintf(hour, sizeof hour, "%.13s:00", s.p[i].when);
        double sum = 0;
        int k = 0;
        for (; i < s.n && !strncmp(s.p[i].when, hour, 13); i++, k++)
            sum += s.p[i].v;
        if (h.n < PR_POINTS) {
            snprintf(h.p[h.n].when, sizeof h.p[h.n].when, "%s", hour + 11);
            h.p[h.n++].v = round(sum / k * 100) / 100;
        }
    }
    char who[160];
    snprintf(who, sizeof who, "%s, %s, %s", p.name, zone, day);
    struct janas_buf d = {0};
    pr_series_data(&d, "power", who, &h, 24, "energy_charts", "exchange",
                   licence);
    return pr_answer(b, "prices_electricity", &d, PR_SERIES_LAYOUT,
                     PR_SERIES_BRIEF);
}
