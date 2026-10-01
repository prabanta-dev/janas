/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * warn.c - weather_alerts (see weather.h): MeteoAlarm's warnings in force
 * for the place's region, those of the rest of its country after them,
 * and in Italy the Civil Protection's official levels for the place's
 * zone of alert. No warning is news too, and is said.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "weather.h"

#define OTHERS 8 /* warnings elsewhere in the country, listed at most */

static void alert_line(const struct wx_alert *a, int full, const char *tz,
                       struct janas_buf *out)
{
    /* the feeds give their own zone (Norway's UTC): the place's instead */
    char from[64], to[64];
    time_t t0 = wx_iso_time(a->onset), t1 = wx_iso_time(a->expires);
    if (t0)
        wx_local_time(t0, tz, from, sizeof from);
    else
        snprintf(from, sizeof from, "%.16s", a->onset);
    if (t1)
        wx_local_time(t1, tz, to, sizeof to);
    else
        snprintf(to, sizeof to, "%.16s", a->expires);
    janas_buf_printf(out, "- %s, %s level, %s: from %s to %s", a->event,
                     a->level, a->area, from, to);
    if (full && a->text[0]) {
        /* the first line of the description: the rest is a disclaimer */
        const char *nl = strchr(a->text, '\n');
        int n = nl ? (int)(nl - a->text) : (int)strlen(a->text);
        janas_buf_printf(out, ". %.*s", n, a->text);
    }
    if (full && a->sender[0])
        janas_buf_printf(out, " (%s)", a->sender);
    janas_buf_puts(out, "\n");
}

static const char *dpc_words(const char *level)
{
    /* "Ordinaria criticità / ALLERTA GIALLA" -> the level, in English */
    if (strstr(level, "ROSSA"))
        return "red alert (high criticality)";
    if (strstr(level, "ARANCIONE"))
        return "orange alert (moderate criticality)";
    if (strstr(level, "GIALLA"))
        return "yellow alert (ordinary criticality)";
    if (strstr(level, "NESSUNA"))
        return "no alert (no significant phenomena expected)";
    return level;
}

int wx_tool_alerts(const struct janas_json *args, struct janas_buf *b)
{
    struct wx_place p;
    if (wx_arg_place(args, &p, b) != 0)
        return 0;
    const char *lang = wx_arg_str(args, "language");
    struct janas_buf out = {0};
    char name[256], err[512];
    wx_place_name(&p, name, sizeof name);
    janas_buf_printf(&out, "Weather warnings for %s:\n", name);

    struct wx_alert *v = NULL;
    size_t n = 0;
    int r = wx_meteoalarm(&p, lang ? lang : "en", &v, &n, err, sizeof err);
    if (r < 0) {
        janas_buf_printf(&out, "MeteoAlarm could not be asked: %s.\n", err);
    } else if (r > 0) {
        janas_buf_printf(&out, "%s.\n", err);
    } else {
        size_t mine = 0;
        while (mine < n && v[mine].mine)
            mine++;
        const char *region = p.region[0] ? p.region : p.country;
        if (mine)
            janas_buf_printf(&out, "MeteoAlarm, in force for %s:\n", region);
        else
            janas_buf_printf(&out,
                             "MeteoAlarm: no warning in force for "
                             "%s.\n",
                             region);
        for (size_t i = 0; i < mine; i++)
            alert_line(&v[i], 1, p.tz, &out);
        if (n > mine) {
            janas_buf_printf(&out, "Elsewhere in %s, %zu more:\n",
                             p.country[0] ? p.country : "the country",
                             n - mine);
            for (size_t i = mine; i < n && i < mine + OTHERS; i++)
                alert_line(&v[i], 0, p.tz, &out);
            if (n - mine > OTHERS)
                janas_buf_printf(&out, "- and %zu more\n", n - mine - OTHERS);
        }
        if (strcmp(p.cc, "IT") == 0)
            janas_buf_puts(&out, "MeteoAlarm's Italian warnings are the Air "
                                 "Force weather service's, about the "
                                 "phenomena; the official alerts are the "
                                 "Civil Protection's, below.\n");
    }
    free(v);

    if (strcmp(p.cc, "IT") == 0) {
        struct wx_dpc d;
        int k = wx_dpc(&p, &d, err, sizeof err);
        if (k < 0) {
            janas_buf_printf(&out,
                             "The Civil Protection's bulletin could not "
                             "be read: %s.\n",
                             err);
        } else if (k == 0) {
            janas_buf_printf(&out,
                             "The Civil Protection's bulletin has no "
                             "municipality called %s.\n",
                             p.name);
        } else {
            janas_buf_printf(&out,
                             "The Civil Protection's national bulletin of "
                             "criticality (issued %.4s-%.2s-%.2s at "
                             "%.2s:%.2s), zone of alert %s:\n",
                             d.issued, d.issued + 4, d.issued + 6, d.issued + 9,
                             d.issued + 11, d.zones);
            static const char *const what[3] = {"floods (hydraulic)",
                                                "landslides (hydrogeological)",
                                                "thunderstorms"};
            for (int i = 0; i < 2 && d.day[i][0]; i++) {
                char dn[48];
                wx_day_name(d.day[i], dn, sizeof dn);
                janas_buf_printf(&out, "- %s:", dn);
                for (int j = 0; j < 3; j++)
                    janas_buf_printf(&out, "%s %s: %s", j ? ";" : "", what[j],
                                     d.level[i][j][0] ? dpc_words(d.level[i][j])
                                                      : "not given");
                janas_buf_puts(&out, "\n");
            }
            if (!d.day[1][0])
                janas_buf_puts(&out, "Tomorrow's levels come with the next "
                                     "bulletin, usually by 16:00.\n");
        }
    }
    janas_buf_puts(&out, "Sources: MeteoAlarm (meteoalarm.org, EUMETNET)");
    if (strcmp(p.cc, "IT") == 0)
        janas_buf_puts(&out, "; Dipartimento della Protezione Civile "
                             "(CC BY 4.0)");
    janas_buf_puts(&out, ".");
    wx_place_note(p.how, &out);
    return wx_result(&out, b, 0);
}
