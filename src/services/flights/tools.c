/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - the tools janas-flights offers: flight_status (a flight by its
 * number or callsign), flights_between (the flights of a route, with an
 * AviationStack key), flights_over (what flies over a sea now) and
 * flights_nearby (what flies over a place now). The lists - over a sea,
 * near a place - use open data only: AviationStack's few calls a month are
 * for the questions about one flight or one route.
 *
 * Every fact comes out already worked out - the state, where the aircraft
 * is in words, distances, an arrival estimated from its speed, times in the
 * airport's own zone - so that the model telling it has nothing to compute
 * and nothing to guess. What is not known is said to be not known.
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "services/common/locate.h"
#include "common/mcp_server.h"
#include "flights.h"
#include "services/common/geo.h"
#include "services/common/template.h"

#define DEG(x) ((x) * 1e-5)
#define KM_PER_NM 1.852
#define MAX_LIST 40
#define ROUTES_ASKED 12 /* adsbdb calls in a list, at most */
#define LIVE_ASKED 8    /* ADS-B calls in a list of flights, at most */
#define MAX_BETWEEN 20
#define MAX_OVER 80
#define MAX_TILES 6    /* ADS-B calls for a sea, at most */
#define ROUTES_OVER 25 /* adsbdb calls for a sea, at most */

static void schema_str(struct janas_buf *b, const char *name, const char *what)
{
    janas_buf_printf(b,
                     "\"%s\": {\"type\": \"string\", \"description\": ", name);
    janas_json_write_str(b, what, strlen(what));
    janas_buf_puts(b, "}");
}

/* The model reads all of it before it can use a tool: every word costs. */
void fl_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b, "\"tools\": [{\"name\": \"flight_status\", \"title\": \"Status "
           "of a flight\", \"description\": \"A flight's times, gate and "
           "delay today (with an AviationStack key), and where it is now "
           "(ADS-B): on the ground or in the air, where in words, an "
           "arrival estimate.\", \"inputSchema\": {\"type\": \"object\", "
           "\"properties\": {");
    schema_str(b, "flight", "Flight number (AZ1631) or callsign (VLG1595).");
    janas_buf_puts(
        b, "}, \"required\": [\"flight\"], \"additionalProperties\": false}}, "
           "{\"name\": \"flights_between\", \"title\": \"Flights of a "
           "route\", \"description\": \"Today's flights from one airport to "
           "another, with their times and where those flying are (needs an "
           "AviationStack key).\", \"inputSchema\": {\"type\": \"object\", "
           "\"properties\": {");
    schema_str(b, "from", "Airport code (TRN) or city in English.");
    janas_buf_puts(b, ", ");
    schema_str(b, "to", "Airport code (TPS) or city in English.");
    janas_buf_puts(
        b, "}, \"required\": [\"from\", \"to\"], \"additionalProperties\": "
           "false}}, "
           "{\"name\": \"flights_over\", \"title\": \"Flights over a "
           "sea\", \"description\": \"The aircraft flying now over a sea "
           "(ADS-B, open data), with where each is and its route.\", "
           "\"inputSchema\": {\"type\": \"object\", \"properties\": {");
    schema_str(b, "area", "A sea in English (Tyrrhenian Sea, North Sea).");
    janas_buf_puts(
        b, ", \"limit\": {\"type\": \"integer\", \"minimum\": 1, "
           "\"maximum\": 80, \"description\": \"At most this many listed "
           "(default 40).\"}}, \"required\": [\"area\"], "
           "\"additionalProperties\": false}}, "
           "{\"name\": \"flights_nearby\", \"title\": \"Flights near a "
           "place\", \"description\": \"The aircraft flying now near an "
           "airport, a city or a point (ADS-B, open data).\", "
           "\"inputSchema\": {\"type\": \"object\", \"properties\": {");
    schema_str(b, "place",
               "Airport code (TRN) or city in English. Leave out place and "
               "coordinates only when the user names no place: then where "
               "the user is.");
    janas_buf_puts(
        b, ", \"latitude\": {\"type\": \"number\", \"minimum\": -90, "
           "\"maximum\": 90}, \"longitude\": {\"type\": \"number\", "
           "\"minimum\": -180, \"maximum\": 180}, \"radius_km\": {\"type\": "
           "\"number\", \"minimum\": 1, \"maximum\": 460, \"description\": "
           "\"Default 50.\"}, \"limit\": {\"type\": \"integer\", "
           "\"minimum\": 1, \"maximum\": 40, \"description\": \"Nearest "
           "first (default 15).\"}}, \"additionalProperties\": false}}]");
}

/* ---- words ---- */

/* 33000 -> "33,000" */
static void thousands(struct janas_buf *b, double v)
{
    long n = lround(v);
    if (n < 0) {
        janas_buf_puts(b, "-");
        n = -n;
    }
    if (n >= 1000)
        janas_buf_printf(b, "%ld,%03ld", n / 1000, n % 1000);
    else
        janas_buf_printf(b, "%ld", n);
}

/* The time t as HH:MM in the zone tz (an IANA name). */
static void hhmm(time_t t, const char *tz, char *out, size_t n)
{
    char *old = getenv("TZ") ? strdup(getenv("TZ")) : NULL;
    setenv("TZ", tz, 1);
    tzset();
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(out, n, "%H:%M", &tm);
    if (old) {
        setenv("TZ", old, 1);
        free(old);
    } else {
        unsetenv("TZ");
    }
    tzset();
}

/* The time t as "HH:MM local time (HH:MM UTC)", with the day when it is
   not today there. */
static void when_text(struct janas_buf *b, time_t t, const char *tz)
{
    char loc[16], utc[16], day[16], today[16];
    hhmm(t, tz, loc, sizeof loc);
    hhmm(t, "UTC", utc, sizeof utc);
    janas_buf_printf(b, "%s local time (%s UTC)", loc, utc);
    char *old = getenv("TZ") ? strdup(getenv("TZ")) : NULL;
    setenv("TZ", tz, 1);
    tzset();
    struct tm tm;
    time_t now = time(NULL);
    localtime_r(&now, &tm);
    strftime(today, sizeof today, "%d %b", &tm);
    localtime_r(&t, &tm);
    strftime(day, sizeof day, "%d %b", &tm);
    if (old) {
        setenv("TZ", old, 1);
        free(old);
    } else {
        unsetenv("TZ");
    }
    tzset();
    if (strcmp(day, today) != 0)
        janas_buf_printf(b, " on %s", day);
}

static const char *squawk_meaning(const char *sq)
{
    if (strcmp(sq, "7500") == 0)
        return "7500, unlawful interference (hijacking)";
    if (strcmp(sq, "7600") == 0)
        return "7600, radio failure";
    if (strcmp(sq, "7700") == 0)
        return "7700, general emergency";
    return NULL;
}

/* What the aircraft is doing, in words. */
static void state_text(const struct fl_ac *a, struct janas_buf *b)
{
    double km = 1e30;
    const struct geo_airport *ap =
        a->has_pos ? geo_airport_near(a->lat, a->lon, &km) : NULL;
    if (a->ground) {
        janas_buf_puts(b, "on the ground");
        if (ap && km < 6)
            janas_buf_printf(b, " at %s (%s)", ap->name, ap->iata);
        if (a->gs >= 0 && a->gs < 3)
            janas_buf_puts(b, ", stationary");
        else if (a->gs >= 3 && a->gs < 40)
            janas_buf_printf(b, ", taxiing at %.0f kts", a->gs);
        else if (a->gs >= 40)
            janas_buf_printf(b,
                             ", on the runway at %.0f kts (taking off or "
                             "just landed)",
                             a->gs);
    } else if (a->has_alt && a->alt < 100) {
        /* barometric altitude reads below zero on a short final */
        janas_buf_puts(b, "airborne at ground level (touching down or just "
                          "lifted off)");
        if (a->gs >= 0)
            janas_buf_printf(b, ", %.0f kts", a->gs);
        if (ap && km < 6)
            janas_buf_printf(b, ", at %s (%s)", ap->name, ap->iata);
    } else if (a->has_alt) {
        const char *phase = a->rate > 400    ? "climbing"
                            : a->rate < -400 ? "descending"
                            : a->alt > 20000 ? "cruising"
                                             : "level";
        janas_buf_printf(b, "airborne, %s, ", phase);
        if (strcmp(phase, "climbing") == 0 || strcmp(phase, "descending") == 0)
            janas_buf_puts(b, "now at ");
        thousands(b, a->alt);
        janas_buf_puts(b, " ft");
        if (a->alt >= 10000)
            janas_buf_printf(b, " (FL%03ld)", lround(a->alt / 100));
        if (a->gs >= 0)
            janas_buf_printf(b, ", %.0f kts over the ground (%.0f km/h)", a->gs,
                             a->gs * KM_PER_NM);
        if (a->track >= 0)
            janas_buf_printf(b, ", heading %.0f degrees (%s)", a->track,
                             geo_compass(a->track));
        if (fabs(a->rate) > 400)
            janas_buf_printf(b, ", %s %.0f ft a minute",
                             a->rate > 0 ? "up" : "down", fabs(a->rate));
        if (ap && km < 30 && a->alt < 6000)
            janas_buf_printf(b, ", %.0f km from %s (%s)", km, ap->name,
                             ap->iata);
    } else {
        janas_buf_puts(b, "altitude not reported");
    }
    const char *sq = squawk_meaning(a->squawk);
    if (sq)
        janas_buf_printf(b, ". EMERGENCY: transponder code %s", sq);
}

/* The airport of a place of a route: the table's, for its time zone. */
static const struct geo_airport *route_airport(const struct fl_place *p)
{
    if (p->iata[0])
        return geo_airport_find(p->iata);
    if (p->icao[0])
        return geo_airport_find(p->icao);
    return NULL;
}

/* Whether the position fits the route: not more than a detour of a
   quarter of the route (200 km at least) off the great circle. */
static int route_fits(const struct fl_ac *a, const struct fl_route *r)
{
    if (r->scheduled || !a->has_pos || (!r->from.lat && !r->from.lon) ||
        (!r->to.lat && !r->to.lon))
        return 1;
    double whole = geo_km(r->from.lat, r->from.lon, r->to.lat, r->to.lon);
    double via = geo_km(r->from.lat, r->from.lon, a->lat, a->lon) +
                 geo_km(a->lat, a->lon, r->to.lat, r->to.lon);
    double slack = whole / 4 > 200 ? whole / 4 : 200;
    return via - whole <= slack;
}

static void place_text(const struct fl_place *p, struct janas_buf *b)
{
    janas_buf_printf(b, "%s", p->city[0] ? p->city : p->name);
    if (p->iata[0])
        janas_buf_printf(b, " (%s", p->iata);
    if (p->iata[0] && p->name[0])
        janas_buf_printf(b, ", %s", p->name);
    if (p->iata[0])
        janas_buf_puts(b, ")");
}

/* The route line, said as uncertain, and checked against the position. */
static void route_text(const struct fl_ac *a, const struct fl_route *r,
                       struct janas_buf *b)
{
    janas_buf_puts(b, r->scheduled ? "route: " : "route on record: ");
    place_text(&r->from, b);
    janas_buf_puts(b, " to ");
    place_text(&r->to, b);
    if (r->airline[0])
        janas_buf_printf(b, ", %s", r->airline);
    if (r->flight_iata[0])
        janas_buf_printf(b, " flight %s", r->flight_iata);
    if (r->scheduled)
        janas_buf_puts(b, " (from the schedule)");
    else if (!route_fits(a, r))
        janas_buf_puts(b, " - BUT the aircraft is far off that route, so the "
                          "record is probably wrong for this callsign (the "
                          "airline reuses it): do not state the route");
    else
        janas_buf_puts(b, " (from adsbdb's community records, not from a "
                          "schedule: usually right, sometimes not)");
}

/* An arrival estimated from the speed, toward the route's destination. */
static void eta_text(const struct fl_ac *a, const struct fl_route *r,
                     struct janas_buf *b)
{
    if (a->ground || !a->has_pos || a->gs < 80 || !route_fits(a, r))
        return;
    double km = geo_km(a->lat, a->lon, r->to.lat, r->to.lon);
    if (km < 5)
        return;
    double hours = km / (a->gs * KM_PER_NM) + (km > 40 ? 10.0 / 60 : 0);
    time_t t = time(NULL) + (time_t)(hours * 3600);
    const struct geo_airport *to = route_airport(&r->to);
    char loc[16], utc[16];
    hhmm(t, to ? geo_tz_name(to->tz) : "UTC", loc, sizeof loc);
    hhmm(t, "UTC", utc, sizeof utc);
    janas_buf_printf(b,
                     "\n- %.0f km to %s; arrival estimated from the "
                     "aircraft's speed: about %s local time there (%s "
                     "UTC). An estimate, not the airline's.",
                     km, r->to.city[0] ? r->to.city : r->to.name, loc, utc);
}

/* One aircraft, the lines of its status. */
static void ac_text(const struct fl_ac *a, const struct fl_route *r,
                    int have_route, struct janas_buf *b)
{
    janas_buf_printf(b, "- callsign %s", a->flight[0] ? a->flight : "unknown");
    if (a->reg[0] || a->type[0])
        janas_buf_printf(b, " (aircraft %s%s%s)", a->reg,
                         a->reg[0] && a->type[0] ? ", type " : "", a->type);
    janas_buf_puts(b, "\n- ");
    state_text(a, b);
    if (a->has_pos) {
        janas_buf_puts(b, "\n- position: ");
        geo_describe(a->lat, a->lon, b);
        if (a->seen_pos > 60)
            janas_buf_printf(b, " (last heard %.0f minutes ago)",
                             a->seen_pos / 60);
    }
    if (have_route) {
        janas_buf_puts(b, "\n- ");
        route_text(a, r, b);
        eta_text(a, r, b);
    }
}

/* The sources of an answer: net is the ADS-B network that answered (""
   when none was asked), sched says whether AviationStack's times are in. */
static void sources(struct janas_buf *b, const char *net, int routes, int sched)
{
    janas_buf_puts(b, "\n\nSources: ");
    if (sched)
        janas_buf_puts(b, "schedules and times from AviationStack");
    if (net[0])
        janas_buf_printf(b, "%slive positions from %s (%s)", sched ? "; " : "",
                         net,
                         strcmp(net, "adsb.lol") == 0
                             ? "data under the ODbL, https://adsb.lol"
                             : "https://adsb.fi");
    if (routes)
        janas_buf_puts(b, "; routes from adsbdb.com");
    janas_buf_puts(b, ".");
    if (!fl_sched_have())
        janas_buf_puts(b, " No schedules (the server has no AviationStack "
                          "key): scheduled and official estimated times are "
                          "not known to this tool.");
}

static void now_text(struct janas_buf *b)
{
    char utc[16];
    hhmm(time(NULL), "UTC", utc, sizeof utc);
    janas_buf_printf(b, "As of %s UTC.\n", utc);
}

/* ---- schedules ---- */

/* An end of a scheduled flight as a place of a route, from the tables. */
static void sched_place(const struct fl_end *e, struct fl_place *p)
{
    const struct geo_airport *a =
        geo_airport_find(e->iata[0] ? e->iata : e->icao);
    snprintf(p->iata, sizeof p->iata, "%s", e->iata);
    snprintf(p->icao, sizeof p->icao, "%s", e->icao);
    snprintf(p->name, sizeof p->name, "%s", a ? a->name : e->airport);
    snprintf(p->city, sizeof p->city, "%s", a ? a->city : "");
    char *prov = strstr(p->city, " ("); /* "Trapani (TP)": the province */
    if (prov)
        *prov = 0;
    if (a) {
        p->lat = DEG(a->lat);
        p->lon = DEG(a->lon);
    }
}

static void sched_route(const struct fl_sched *s, struct fl_route *r)
{
    memset(r, 0, sizeof(*r));
    snprintf(r->airline, sizeof r->airline, "%s", s->airline);
    snprintf(r->flight_iata, sizeof r->flight_iata, "%s", s->flight_iata);
    sched_place(&s->dep, &r->from);
    sched_place(&s->arr, &r->to);
    r->scheduled = 1;
}

static const char *end_tz(const struct fl_end *e)
{
    if (e->tz[0])
        return e->tz;
    const struct geo_airport *a =
        geo_airport_find(e->iata[0] ? e->iata : e->icao);
    return a ? geo_tz_name(a->tz) : "UTC";
}

/* "- departure from Turin (TRN): scheduled 18:25 local time (16:25 UTC),
   actual 18:23 ...; gate 20" */
static void end_text(struct janas_buf *b, const char *what,
                     const struct fl_end *e)
{
    struct fl_place p = {0};
    sched_place(e, &p);
    janas_buf_printf(b, "\n- %s %s ", what,
                     strcmp(what, "departure") == 0 ? "from" : "at");
    place_text(&p, b);
    janas_buf_puts(b, ":");
    const char *tz = end_tz(e);
    const char *sep = " ";
    if (e->scheduled) {
        janas_buf_puts(b, " scheduled ");
        when_text(b, e->scheduled, tz);
        sep = "; ";
    }
    if (e->estimated && !e->actual && e->estimated != e->scheduled) {
        janas_buf_printf(b, "%sestimated ", sep);
        when_text(b, e->estimated, tz);
        sep = "; ";
    }
    if (e->actual) {
        janas_buf_printf(b, "%sactual %s ", sep,
                         strcmp(what, "departure") == 0 ? "(departed)"
                                                        : "(landed)");
        when_text(b, e->actual, tz);
        sep = "; ";
    }
    if (!e->scheduled && !e->estimated && !e->actual)
        janas_buf_puts(b, " times not known");
    if (e->delay > 0)
        janas_buf_printf(b, "; delay %d minutes", e->delay);
    if (e->terminal[0])
        janas_buf_printf(b, "; terminal %s", e->terminal);
    if (e->gate[0])
        janas_buf_printf(b, "; gate %s", e->gate);
    if (e->baggage[0])
        janas_buf_printf(b, "; baggage belt %s", e->baggage);
}

/* What a flight is doing, from its times: AviationStack's own status
   lags (a flight landed an hour ago still "active", one an hour before
   departure "active" already). */
enum fl_state {
    ST_SCHEDULED,  /* no departure recorded, not due yet */
    ST_UNRECORDED, /* due, or said active, but no departure recorded */
    ST_AIR,        /* departed, not landed */
    ST_LANDED,
    ST_CANCELLED,
    ST_DIVERTED,
    ST_INCIDENT,
};

static enum fl_state sched_state(const struct fl_sched *s)
{
    if (strcmp(s->status, "cancelled") == 0)
        return ST_CANCELLED;
    if (strcmp(s->status, "diverted") == 0)
        return ST_DIVERTED;
    if (strcmp(s->status, "incident") == 0)
        return ST_INCIDENT;
    if (s->arr.actual || strcmp(s->status, "landed") == 0)
        return ST_LANDED;
    if (s->dep.actual)
        return ST_AIR;
    /* "active" comes up to an hour early: before 15 minutes to departure
       it is still scheduled */
    time_t due = s->dep.estimated ? s->dep.estimated : s->dep.scheduled,
           now = time(NULL);
    if (due ? due <= now + 15 * 60 : strcmp(s->status, "active") == 0)
        return ST_UNRECORDED;
    return ST_SCHEDULED;
}

static const char *state_words(enum fl_state st)
{
    switch (st) {
    case ST_SCHEDULED:
        return "scheduled, not departed yet";
    case ST_UNRECORDED:
        return "departure time not recorded yet (it may be about to leave, "
               "or have just left)";
    case ST_AIR:
        return "departed, not landed yet";
    case ST_LANDED:
        return "landed";
    case ST_CANCELLED:
        return "CANCELLED";
    case ST_DIVERTED:
        return "DIVERTED to another airport";
    case ST_INCIDENT:
        return "INCIDENT reported";
    }
    return "not known";
}

/* The schedule's lines of a flight. */
static void sched_text(struct janas_buf *b, const struct fl_sched *s)
{
    janas_buf_printf(b, "Flight %s",
                     s->flight_iata[0] ? s->flight_iata : s->flight_icao);
    if (s->airline[0])
        janas_buf_printf(b, ", %s", s->airline);
    if (s->flight_icao[0] && s->flight_iata[0])
        janas_buf_printf(b, " (ICAO %s)", s->flight_icao);
    janas_buf_printf(b, ", %s: %s", s->date, state_words(sched_state(s)));
    if (s->operated_by[0])
        janas_buf_printf(b, "\n- a codeshare: the flight is %s, operated by %s",
                         s->operated_by,
                         s->operator_name[0] ? s->operator_name
                                             : "another "
                                               "airline");
    end_text(b, "departure", &s->dep);
    end_text(b, "arrival", &s->arr);
    if (s->reg[0] || s->hex[0])
        janas_buf_printf(b, "\n- aircraft%s%s%s%s", s->reg[0] ? " " : "",
                         s->reg, s->hex[0] ? ", transponder " : "", s->hex);
}

/* The flight a list of days means: the one in the air, or the one whose
   departure is nearest to now. Of the answers to a number, the flight of
   that number: a query by AZ1731 brings its codeshares too, V76064 and
   the like, numbers of other airlines for the same flight. */
static const struct fl_sched *sched_pick(const struct fl_sched *v, size_t n,
                                         const char *want)
{
    const struct fl_sched *best = NULL;
    double most = 1e30;
    time_t now = time(NULL);
    int exact = 0;
    for (size_t i = 0; i < n; i++)
        exact |= strcmp(v[i].flight_iata, want) == 0 ||
                 strcmp(v[i].flight_icao, want) == 0;
    for (size_t i = 0; i < n; i++) {
        if (exact && strcmp(v[i].flight_iata, want) != 0 &&
            strcmp(v[i].flight_icao, want) != 0)
            continue;
        if (!exact && v[i].operated_by[0])
            continue;
        if (sched_state(&v[i]) == ST_AIR)
            return &v[i];
        time_t t = v[i].dep.actual      ? v[i].dep.actual
                   : v[i].dep.estimated ? v[i].dep.estimated
                                        : v[i].dep.scheduled;
        double d = fabs(difftime(t, now));
        if (!best || d < most) {
            most = d;
            best = &v[i];
        }
    }
    return best ? best : n ? &v[0] : NULL;
}

/* The aircraft of a scheduled flight, by its transponder: 1 when seen. */
static int live_by_hex(const char *hex, struct fl_ac *a, const char **net)
{
    char query[32], err[400];
    snprintf(query, sizeof query, "hex/%s", hex);
    struct fl_ac *ac = NULL;
    size_t n = 0;
    if (fl_adsb(query, &ac, &n, net, err, sizeof err) != 0)
        return 0;
    int seen = n > 0;
    if (seen)
        *a = ac[0];
    free(ac);
    return seen;
}

/* Whether an aircraft seen live is flying the scheduled flight, and not
   the flight before or after it with the same transponder: on the ground
   at the departure airport (before it leaves) or at the arrival one (after
   it lands), or in the air on the route and heading for its end. */
static int live_fits(const struct fl_ac *a, const struct fl_sched *s,
                     enum fl_state st)
{
    struct fl_route r;
    sched_route(s, &r);
    if (!a->has_pos || (!r.to.lat && !r.to.lon) || (!r.from.lat && !r.from.lon))
        return 0;
    double to_from = geo_km(a->lat, a->lon, r.from.lat, r.from.lon),
           to_to = geo_km(a->lat, a->lon, r.to.lat, r.to.lon);
    if (a->ground || (a->has_alt && a->alt < 100))
        return (st != ST_AIR && to_from < 6) || (st == ST_AIR && to_to < 6);
    if (st == ST_SCHEDULED)
        return 0; /* in the air before departure: the flight before */
    double whole = geo_km(r.from.lat, r.from.lon, r.to.lat, r.to.lon);
    double slack = whole / 4 > 200 ? whole / 4 : 200;
    if (to_from + to_to - whole > slack)
        return 0;
    if (a->track >= 0 && to_to > 30) {
        double turn =
            fabs(geo_bearing(a->lat, a->lon, r.to.lat, r.to.lon) - a->track);
        if (turn > 180)
            turn = 360 - turn;
        if (turn > 70)
            return 0; /* on the route, the other way: the flight back */
    }
    return 1;
}

/* ---- flight_status ---- */

/* "VY 1595", "vy1595" -> "VY1595"; empty when it has other characters */
static void normalize(const char *in, char *out, size_t cap)
{
    size_t n = 0;
    for (; *in && n + 1 < cap; in++) {
        if (*in == ' ' || *in == '-')
            continue;
        if (!isalnum((unsigned char)*in)) {
            n = 0;
            break;
        }
        out[n++] = (char)toupper((unsigned char)*in);
    }
    out[n] = 0;
}

/* A flight number: two characters (not both digits) and digits, IATA, or
   three letters and digits, ICAO. 2, 3, or 0 when it is neither. */
static int flight_number(const char *f)
{
    size_t n = strlen(f);
    if (n >= 4 && n <= 7 && isalpha((unsigned char)f[0]) &&
        isalpha((unsigned char)f[1]) && isalpha((unsigned char)f[2]) &&
        strspn(f + 3, "0123456789") == n - 3)
        return 3;
    if (n >= 3 && n <= 7 &&
        !(isdigit((unsigned char)f[0]) && isdigit((unsigned char)f[1])) &&
        isalnum((unsigned char)f[0]) && isalnum((unsigned char)f[1]) &&
        strspn(f + 2, "0123456789") == n - 2)
        return 2;
    return 0;
}

static int flight_status(const struct janas_json *args, struct janas_buf *b)
{
    char want[16];
    normalize(janas_json_str(janas_json_get(args, "flight"))
                  ? janas_json_str(janas_json_get(args, "flight"))
                  : "",
              want, sizeof want);
    if (strlen(want) < 3) {
        const char *m = "Give a flight number (VY1595) or a callsign "
                        "(VLG1595).";
        janas_mcps_text_result(b, m, strlen(m), 1);
        return 0;
    }
    struct janas_buf out = {0};
    now_text(&out);
    /* the schedule first: the times, the route, the aircraft */
    struct fl_sched *sv = NULL;
    size_t sn = 0;
    const struct fl_sched *s = NULL;
    int kind = flight_number(want);
    if (fl_sched_have() && kind) {
        char q[64], serr[300];
        snprintf(q, sizeof q, "%s=%s",
                 kind == 3 ? "flight_icao" : "flight_iata", want);
        if (fl_schedule(q, &sv, &sn, serr, sizeof serr) != 0)
            janas_buf_printf(&out, "Schedule not available: %s.\n", serr);
        else if (!(s = sched_pick(sv, sn, want)))
            janas_buf_printf(&out,
                             "AviationStack has no flight %s today (its free "
                             "plan knows today's flights only).\n",
                             want);
    }
    /* the callsigns to look for: as given, the schedule's ICAO number, and
       from an IATA flight number the airline's ICAO code with the number
       (VY1595 -> VLG1595) */
    char cands[3][24];
    int n_cands = 0;
    snprintf(cands[n_cands++], sizeof cands[0], "%s", want);
    char airline[96] = "";
    if (s && s->flight_icao[0] && strcmp(s->flight_icao, want) != 0) {
        snprintf(cands[n_cands++], sizeof cands[0], "%s", s->flight_icao);
        snprintf(airline, sizeof airline, "%s", s->airline);
    } else if (!s && kind == 2) {
        char iata[3] = {want[0], want[1], 0}, icao[4];
        if (fl_airline_icao(iata, icao, airline, sizeof airline) == 1) {
            const char *num = want + 2;
            while (*num == '0' && num[1])
                num++;
            snprintf(cands[n_cands++], sizeof cands[0], "%s%s", icao, num);
        }
    }
    const char *net = "";
    int found = 0, routes = 0;
    enum fl_state st = ST_SCHEDULED;
    if (s) {
        sched_text(&out, s);
        /* the aircraft by its transponder, while the flight is flying: at
           another time it is flying another flight */
        struct fl_ac a;
        struct fl_route r;
        sched_route(s, &r);
        st = sched_state(s);
        if ((st == ST_AIR || st == ST_UNRECORDED || st == ST_SCHEDULED) &&
            s->hex[0] && live_by_hex(s->hex, &a, &net) &&
            live_fits(&a, s, st)) {
            janas_buf_puts(&out, "\nNow, from the ADS-B receivers:\n");
            ac_text(&a, &r, 1, &out);
            found = 1;
        }
        if (st != ST_AIR && st != ST_UNRECORDED)
            found = 1; /* not in the air: the times say it all */
    }
    char err[400] = "";
    for (int i = n_cands - 1; i >= 0 && !found; i--) {
        struct fl_ac *ac = NULL;
        size_t n = 0;
        char query[64];
        snprintf(query, sizeof query, "callsign/%s", cands[i]);
        if (fl_adsb(query, &ac, &n, &net, err, sizeof err) != 0)
            continue;
        if (n > 0 && (!s || live_fits(&ac[0], s, st))) {
            struct fl_route r = {0};
            int have;
            if (s) {
                sched_route(s, &r);
                have = 1;
                janas_buf_puts(&out, "\nNow, from the ADS-B receivers:\n");
            } else {
                have = fl_route(ac[0].flight[0] ? ac[0].flight : cands[i], &r);
                routes = have == 1;
                janas_buf_printf(&out, "Flight %s", want);
                if (strcmp(cands[i], want) != 0)
                    janas_buf_printf(&out, " (callsign %s, %s)", cands[i],
                                     airline);
                janas_buf_puts(&out, " is being tracked now:\n");
            }
            ac_text(&ac[0], &r, have == 1, &out);
            found = 1;
        }
        free(ac);
    }
    if (!found && !s && err[0] && strstr(err, "did not answer")) {
        janas_mcps_text_result(b, err, strlen(err), 1);
        janas_buf_free(&out);
        free(sv);
        return 0;
    }
    if (!found && s) {
        janas_buf_puts(&out, "\nThe ADS-B receivers do not see the aircraft "
                             "now (out of their range, or its callsign is "
                             "not its flight number): the times above are "
                             "the schedule's.");
    } else if (!found) {
        janas_buf_printf(&out,
                         "Flight %s is not seen by the ADS-B receivers now "
                         "(looked for callsign%s ",
                         want, n_cands > 1 ? "s" : "");
        for (int i = 0; i < n_cands; i++)
            janas_buf_printf(&out, "%s%s", i ? " and " : "", cands[i]);
        janas_buf_puts(&out,
                       "). It may not be flying now (not yet departed, or "
                       "already landed), it may be out of the receivers' "
                       "range, or its airline may use a callsign different "
                       "from the flight number (Ryanair, easyJet and Wizz "
                       "Air often do).");
        struct fl_route r;
        struct fl_ac none = {0};
        if (fl_route(cands[n_cands - 1], &r) == 1) {
            janas_buf_puts(&out, "\n- ");
            route_text(&none, &r, &out);
            routes = 1;
        }
    }
    sources(&out, net, routes, s != NULL);
    free(sv);
    janas_mcps_text_result(b, out.p ? out.p : "", out.n, 0);
    janas_buf_free(&out);
    return 0;
}

/* ---- flights_between ---- */

static int by_departure(const void *x, const void *y)
{
    time_t a = (*(const struct fl_sched *const *)x)->dep.scheduled,
           b = (*(const struct fl_sched *const *)y)->dep.scheduled;
    return (a > b) - (a < b);
}

static int flights_between(const struct janas_json *args, struct janas_buf *b)
{
    const char *from = janas_json_str(janas_json_get(args, "from")),
               *to = janas_json_str(janas_json_get(args, "to"));
    const struct geo_airport *f = from ? geo_airport_find(from) : NULL,
                             *t = to ? geo_airport_find(to) : NULL;
    if (!f || !t) {
        char m[300];
        snprintf(m, sizeof m,
                 "Airport not known: %.100s. Give an IATA or ICAO code (TRN, "
                 "LICT) or a city's name in English.",
                 !f ? (from ? from : "(none)") : to);
        janas_mcps_text_result(b, m, strlen(m), 1);
        return 0;
    }
    if (!fl_sched_have()) {
        const char *m = "This server has no AviationStack key, so it knows "
                        "no schedules: flights_between is not available. "
                        "flights_nearby shows what flies near an airport "
                        "now.";
        janas_mcps_text_result(b, m, strlen(m), 1);
        return 0;
    }
    char q[64], err[300];
    snprintf(q, sizeof q, "dep_iata=%s&arr_iata=%s", f->iata, t->iata);
    struct fl_sched *v = NULL;
    size_t n = 0;
    if (fl_schedule(q, &v, &n, err, sizeof err) != 0) {
        janas_mcps_text_result(b, err, strlen(err), 1);
        return 0;
    }
    /* the flights as operated: a codeshare is another number of one */
    const struct fl_sched **op = n ? malloc(n * sizeof(*op)) : NULL;
    size_t m = 0;
    for (size_t i = 0; op && i < n; i++)
        if (!v[i].operated_by[0])
            op[m++] = &v[i];
    if (m)
        qsort(op, m, sizeof(*op), by_departure);
    struct janas_buf out = {0};
    now_text(&out);
    janas_buf_printf(&out,
                     "%zu flight%s from %s (%s) to %s (%s) in AviationStack's "
                     "schedule (its free plan knows today's flights and the "
                     "latest ones)",
                     m, m == 1 ? "" : "s", f->name, f->iata, t->name, t->iata);
    /* the latest: the oldest are yesterday's, landed */
    size_t first = m > MAX_BETWEEN ? m - MAX_BETWEEN : 0;
    if (first)
        janas_buf_printf(&out, "; the latest %d, %zu older ones left out",
                         MAX_BETWEEN, first);
    janas_buf_puts(&out, m ? ":\n" : ".");
    const char *net = "";
    int asked = 0;
    for (size_t i = first; i < m; i++) {
        const struct fl_sched *s = op[i];
        janas_buf_printf(&out, "\n%zu. ", i - first + 1);
        sched_text(&out, s);
        int shared = 0;
        for (size_t k = 0; k < n; k++)
            if (v[k].operated_by[0] &&
                strcmp(v[k].operated_by, s->flight_iata) == 0) {
                if (shared < 4)
                    janas_buf_printf(&out, "%s%s",
                                     shared ? ", " : "\n- also sold as ",
                                     v[k].flight_iata);
                shared++;
            }
        if (shared > 4)
            janas_buf_printf(&out, " and %d more", shared - 4);
        struct fl_ac a;
        struct fl_route r;
        enum fl_state st = sched_state(s);
        if ((st == ST_AIR || st == ST_UNRECORDED) && s->hex[0] &&
            asked < LIVE_ASKED) {
            asked++;
            if (live_by_hex(s->hex, &a, &net) && live_fits(&a, s, st)) {
                sched_route(s, &r);
                janas_buf_puts(&out, "\n  Now:\n");
                ac_text(&a, &r, 0, &out);
                eta_text(&a, &r, &out);
            }
        }
        janas_buf_puts(&out, "\n");
    }
    sources(&out, net, 0, 1);
    free(op);
    free(v);
    janas_mcps_text_result(b, out.p ? out.p : "", out.n, 0);
    janas_buf_free(&out);
    return 0;
}

/* ---- flights_over ---- */

/* "key": "value", as JSON */
static void jstr(struct janas_buf *b, const char *key, const char *v)
{
    janas_buf_printf(b, "\"%s\": ", key);
    janas_json_write_str(b, v, strlen(v));
}

/* One aircraft as an item of a layout's "aircraft" (layouts.c): what
   ac_line says, as data. */
static void ac_json(const struct fl_ac *a, const struct fl_route *r,
                    int have_route, struct janas_buf *b)
{
    janas_buf_puts(b, "{");
    jstr(b, "hex", a->hex);
    if (a->flight[0]) {
        janas_buf_puts(b, ", ");
        jstr(b, "callsign", a->flight);
    }
    if (a->reg[0]) {
        janas_buf_puts(b, ", ");
        jstr(b, "reg", a->reg);
    }
    if (a->type[0]) {
        janas_buf_puts(b, ", ");
        jstr(b, "type", a->type);
    }
    if (a->ground) {
        janas_buf_puts(b, ", \"ground\": true");
        double km = 1e30;
        const struct geo_airport *ap =
            a->has_pos ? geo_airport_near(a->lat, a->lon, &km) : NULL;
        if (ap && km < 6) {
            char at[160];
            snprintf(at, sizeof at, "%s (%s)", ap->name, ap->iata);
            janas_buf_puts(b, ", ");
            jstr(b, "airport", at);
        }
        if (a->gs >= 0)
            janas_buf_printf(b, ", \"on_ground\": \"%s\"",
                             a->gs < 3    ? "stationary"
                             : a->gs < 40 ? "taxiing"
                                          : "runway");
    } else if (a->has_alt && a->alt >= 10000)
        janas_buf_printf(b, ", \"fl\": \"%03ld\"", lround(a->alt / 100));
    else if (a->has_alt)
        janas_buf_printf(b, ", \"ft\": %ld", lround(a->alt));
    if (!a->ground && a->has_alt && fabs(a->rate) > 400)
        janas_buf_puts(b, a->rate > 0 ? ", \"climbing\": true"
                                      : ", \"descending\": true");
    if (a->gs >= 0)
        janas_buf_printf(b, ", \"kts\": %.0f", a->gs);
    if (a->track >= 0) {
        janas_buf_puts(b, ", ");
        jstr(b, "heading", geo_compass(a->track));
    }
    janas_buf_puts(b, ", \"where\": ");
    geo_describe_json(a->lat, a->lon, b);
    if (squawk_meaning(a->squawk)) {
        janas_buf_puts(b, ", ");
        jstr(b, "emergency", a->squawk);
    }
    if (have_route && route_fits(a, r)) {
        janas_buf_puts(b, ", \"route\": {");
        jstr(b, "from", r->from.city[0] ? r->from.city : r->from.name);
        janas_buf_puts(b, ", ");
        jstr(b, "from_code", r->from.iata[0] ? r->from.iata : r->from.icao);
        janas_buf_puts(b, ", ");
        jstr(b, "to", r->to.city[0] ? r->to.city : r->to.name);
        janas_buf_puts(b, ", ");
        jstr(b, "to_code", r->to.iata[0] ? r->to.iata : r->to.icao);
        if (r->airline[0]) {
            janas_buf_puts(b, ", ");
            jstr(b, "airline", r->airline);
        }
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "}");
}

static int by_lon(const void *x, const void *y)
{
    double a = ((const struct fl_ac *)x)->lon,
           b = ((const struct fl_ac *)y)->lon;
    return (a > b) - (a < b);
}

static int flights_over(const struct janas_json *args, struct janas_buf *b)
{
    const char *name = janas_json_str(janas_json_get(args, "area"));
    int limit = (int)janas_json_num(janas_json_get(args, "limit"), 40);
    if (limit < 1)
        limit = 1;
    if (limit > MAX_OVER)
        limit = MAX_OVER;
    struct geo_area area;
    if (!name || !geo_area_find(name, &area)) {
        char m[300];
        snprintf(m, sizeof m,
                 "Sea not known: %.100s. Give a sea's name in English "
                 "(Tyrrhenian Sea, Adriatic Sea, Ligurian Sea, North Sea).",
                 name ? name : "(none)");
        janas_mcps_text_result(b, m, strlen(m), 1);
        return 0;
    }
    /* the sea's box, in cells of at most 600 km a side: each a circle of
       the networks' (250 nautical miles at most) */
    double la0 = DEG(area.s_lat_min), la1 = DEG(area.s_lat_max),
           lo0 = DEG(area.s_lon_min), lo1 = DEG(area.s_lon_max);
    double near_eq = fabs(la0) < fabs(la1) ? fabs(la0) : fabs(la1);
    if (la0 < 0 && la1 > 0)
        near_eq = 0;
    double h_km = (la1 - la0) * 111.2,
           w_km = (lo1 - lo0) * 111.2 * cos(near_eq * M_PI / 180);
    int ny = (int)ceil(h_km / 600), nx = (int)ceil(w_km / 600);
    if (ny < 1)
        ny = 1;
    if (nx < 1)
        nx = 1;
    if (nx * ny > MAX_TILES || lo1 - lo0 > 180) {
        char m[300];
        snprintf(m, sizeof m,
                 "The %s is too large to search at once (%d areas of 600 km "
                 "a side). Ask for one of its parts: a smaller sea within "
                 "it, or flights_nearby around a place.",
                 area.name, nx * ny);
        janas_mcps_text_result(b, m, strlen(m), 1);
        return 0;
    }
    double ch = h_km / ny, cw = w_km / nx;
    double nm = ceil((sqrt(ch * ch + cw * cw) / 2 + 10) / KM_PER_NM);
    if (nm > 250)
        nm = 250;
    struct fl_ac *all = NULL;
    size_t n_all = 0, cap = 0;
    const char *net = "";
    char err[400] = "";
    int failed = 0;
    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++) {
            double lat = la0 + (la1 - la0) * (y + 0.5) / ny,
                   lon = lo0 + (lo1 - lo0) * (x + 0.5) / nx;
            char query[96];
            snprintf(query, sizeof query, "lat/%.4f/lon/%.4f/dist/%.0f", lat,
                     lon, nm);
            struct fl_ac *ac = NULL;
            size_t n = 0;
            if (fl_adsb(query, &ac, &n, &net, err, sizeof err) != 0) {
                failed++;
                continue;
            }
            for (size_t i = 0; i < n; i++) {
                if (!ac[i].has_pos ||
                    !geo_area_has(&area, ac[i].lat, ac[i].lon))
                    continue;
                int dup = 0;
                for (size_t k = 0; k < n_all && !dup; k++)
                    dup = strcmp(all[k].hex, ac[i].hex) == 0;
                if (dup)
                    continue;
                if (n_all == cap) {
                    size_t nc = cap ? cap * 2 : 64;
                    struct fl_ac *g = realloc(all, nc * sizeof(*g));
                    if (!g)
                        break;
                    all = g;
                    cap = nc;
                }
                all[n_all++] = ac[i];
            }
            free(ac);
            if (strcmp(net, "adsb.fi") == 0)
                usleep(1100000); /* adsb.fi: one request a second */
        }
    if (failed == nx * ny) {
        janas_mcps_text_result(b, err, strlen(err), 1);
        free(all);
        return 0;
    }
    if (n_all)
        qsort(all, n_all, sizeof(*all), by_lon);
    /* the data, for the layout (layouts.c) */
    struct janas_buf d = {0};
    char utc[16];
    hhmm(time(NULL), "UTC", utc, sizeof utc);
    size_t airborne = 0;
    int emergencies = 0;
    for (size_t i = 0; i < n_all; i++) {
        airborne += !all[i].ground;
        emergencies += squawk_meaning(all[i].squawk) != NULL;
    }
    janas_buf_printf(&d, "{\"time\": \"%s\", ", utc);
    jstr(&d, "sea", area.name);
    janas_buf_printf(&d, ", \"count\": %zu, \"airborne\": %zu", n_all,
                     airborne);
    if (failed)
        janas_buf_printf(&d, ", \"failed\": %d, \"parts\": %d", failed,
                         nx * ny);
    if (n_all > (size_t)limit)
        janas_buf_printf(&d, ", \"listed\": %d", limit);
    if (emergencies)
        janas_buf_printf(&d, ", \"emergencies\": %d", emergencies);
    janas_buf_puts(&d, ", \"aircraft\": [");
    int routes = 0;
    for (size_t i = 0; i < n_all && i < (size_t)limit; i++) {
        struct fl_route r = {0};
        int have = 0;
        if (all[i].flight[0] && i < ROUTES_OVER) {
            have = fl_route(all[i].flight, &r) == 1;
            routes |= have && route_fits(&all[i], &r);
        }
        janas_buf_puts(&d, i ? ", " : "");
        ac_json(&all[i], &r, have, &d);
    }
    janas_buf_puts(&d, "], ");
    /* open data only: no schedule is asked for a list */
    jstr(&d, "net", net);
    janas_buf_puts(&d, ", ");
    jstr(&d, "net_url",
         strcmp(net, "adsb.lol") == 0 ? "https://adsb.lol" : "https://adsb.fi");
    janas_buf_printf(&d, "%s%s}",
                     strcmp(net, "adsb.lol") == 0 ? ", \"odbl\": true" : "",
                     routes ? ", \"routes\": true" : "");
    free(all);
    if (d.oom || janas_tpl_result(b, "flights_over", d.p, d.n, FL_OVER_LAYOUT,
                                  FL_OVER_BRIEF, err, sizeof err) != 0) {
        char m[480];
        snprintf(m, sizeof m, "The answer could not be written: %s",
                 d.oom ? "out of memory" : err);
        janas_mcps_text_result(b, m, strlen(m), 1);
    }
    janas_buf_free(&d);
    return 0;
}

/* ---- flights_nearby ---- */

struct near {
    const struct fl_ac *a;
    double km;
};

static int by_km(const void *x, const void *y)
{
    double a = ((const struct near *)x)->km, b = ((const struct near *)y)->km;
    return (a > b) - (a < b);
}

static int flights_nearby(const struct janas_json *args, struct janas_buf *b)
{
    const char *place = janas_json_str(janas_json_get(args, "place"));
    double lat = janas_json_num(janas_json_get(args, "latitude"), 1000),
           lon = janas_json_num(janas_json_get(args, "longitude"), 1000);
    double radius = janas_json_num(janas_json_get(args, "radius_km"), 50);
    int limit = (int)janas_json_num(janas_json_get(args, "limit"), 15);
    if (radius < 1)
        radius = 1;
    if (radius > 460)
        radius = 460;
    if (limit < 1)
        limit = 1;
    if (limit > MAX_LIST)
        limit = MAX_LIST;
    char where[256] = "", here[128] = "", how[128] = "";
    if (place && *place) {
        const struct geo_airport *ap = geo_airport_find(place);
        const struct geo_city *c = ap ? NULL : geo_city_find(place);
        if (ap) {
            lat = DEG(ap->lat);
            lon = DEG(ap->lon);
            snprintf(where, sizeof where, "%s (%s)", ap->name, ap->iata);
        } else if (c) {
            lat = DEG(c->lat);
            lon = DEG(c->lon);
            snprintf(where, sizeof where, "%s, %s", c->name,
                     geo_country_name(c->country));
        } else {
            char m[300];
            snprintf(m, sizeof m,
                     "Place not known: %.100s. Give an airport code (TRN, "
                     "LIMF), a city's name in English, or a latitude and "
                     "longitude.",
                     place);
            janas_mcps_text_result(b, m, strlen(m), 1);
            return 0;
        }
    } else if (lat > 90 || lat < -90 || lon > 180 || lon < -180) {
        struct janas_where w;
        char why[300];
        if (janas_where(&w, why, sizeof why) != 0) {
            char m[400];
            snprintf(m, sizeof m, "No place given, and %s.", why);
            janas_mcps_text_result(b, m, strlen(m), 1);
            return 0;
        }
        lat = w.lat;
        lon = w.lon;
        snprintf(where, sizeof where, "%s (where the user is, %s)",
                 w.city[0] ? w.city : "the user's position", w.how);
        snprintf(here, sizeof here, "%s",
                 w.city[0] ? w.city : "the user's position");
        snprintf(how, sizeof how, "%s", w.how);
        place = where; /* named: no point to put into words */
    } else {
        snprintf(where, sizeof where, "%.3f, %.3f", lat, lon);
    }
    struct fl_ac *ac = NULL;
    size_t n = 0;
    const char *net = "";
    char query[96], err[400];
    snprintf(query, sizeof query, "lat/%.4f/lon/%.4f/dist/%.0f", lat, lon,
             ceil(radius / KM_PER_NM));
    if (fl_adsb(query, &ac, &n, &net, err, sizeof err) != 0) {
        janas_mcps_text_result(b, err, strlen(err), 1);
        return 0;
    }
    struct near *v = n ? malloc(n * sizeof(*v)) : NULL;
    size_t m = 0;
    for (size_t i = 0; v && i < n; i++)
        if (ac[i].has_pos) {
            double d = geo_km(lat, lon, ac[i].lat, ac[i].lon);
            if (d <= radius)
                v[m++] = (struct near){&ac[i], d};
        }
    if (m)
        qsort(v, m, sizeof(*v), by_km);
    /* the data, for the layout (layouts.c) */
    struct janas_buf d = {0};
    char utc[16];
    hhmm(time(NULL), "UTC", utc, sizeof utc);
    janas_buf_printf(&d, "{\"time\": \"%s\", ", utc);
    jstr(&d, "place", here[0] ? here : where);
    if (how[0]) {
        janas_buf_puts(&d, ", ");
        jstr(&d, "how", how);
    }
    if (!(place && *place)) { /* a point: where it is */
        janas_buf_puts(&d, ", \"point\": ");
        geo_describe_json(lat, lon, &d);
    }
    janas_buf_printf(&d, ", \"radius\": %.0f, \"count\": %zu", radius, m);
    if (m > (size_t)limit)
        janas_buf_printf(&d, ", \"listed\": %d", limit);
    janas_buf_puts(&d, ", \"aircraft\": [");
    int routes = 0;
    for (size_t i = 0; i < m && i < (size_t)limit; i++) {
        const struct fl_ac *a = v[i].a;
        struct fl_route r = {0};
        int have = 0;
        if (a->flight[0] && i < ROUTES_ASKED) {
            have = fl_route(a->flight, &r) == 1;
            routes |= have && route_fits(a, &r);
        }
        janas_buf_puts(&d, i ? ", " : "");
        ac_json(a, &r, have, &d);
        /* how far and which way from the place, into the item */
        d.n--; /* its "}" */
        janas_buf_printf(&d, ", \"dist_km\": %.0f, ", v[i].km);
        jstr(&d, "dist_dir",
             geo_compass(geo_bearing(lat, lon, a->lat, a->lon)));
        janas_buf_puts(&d, "}");
    }
    janas_buf_puts(&d, "], ");
    jstr(&d, "net", net);
    janas_buf_puts(&d, ", ");
    jstr(&d, "net_url",
         strcmp(net, "adsb.lol") == 0 ? "https://adsb.lol" : "https://adsb.fi");
    janas_buf_printf(&d, "%s%s%s}",
                     strcmp(net, "adsb.lol") == 0 ? ", \"odbl\": true" : "",
                     routes ? ", \"routes\": true" : "",
                     fl_sched_have() ? "" : ", \"no_schedules\": true");
    free(v);
    free(ac);
    if (d.oom || janas_tpl_result(b, "flights_nearby", d.p, d.n, FL_NEAR_LAYOUT,
                                  FL_NEAR_BRIEF, err, sizeof err) != 0) {
        char msg[480];
        snprintf(msg, sizeof msg, "The answer could not be written: %s",
                 d.oom ? "out of memory" : err);
        janas_mcps_text_result(b, msg, strlen(msg), 1);
    }
    janas_buf_free(&d);
    return 0;
}

int fl_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    (void)ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    if (janas_json_is(name, "flight_status"))
        return flight_status(args, b);
    if (janas_json_is(name, "flights_between"))
        return flights_between(args, b);
    if (janas_json_is(name, "flights_over"))
        return flights_over(args, b);
    if (janas_json_is(name, "flights_nearby"))
        return flights_nearby(args, b);
    return -1;
}
