/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_maps_parse.c - janas-maps' reading of its sources' answers, without
 * the network, and the layouts filled with what was read: Valhalla's route
 * from Trapani to Palermo and another way (cut to a few steps), an error
 * of it, OSRM's route, Transitous' journey, Overpass' pharmacies (and a
 * way without a name), Nominatim's and Photon's places, as they came on 2
 * October 2026.
 * The sources belong to the program, so they are compiled in here.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "common/template.h"
#include "maps/data.c"
#include "maps/layouts.c"
#include "maps/nearby.c"
#include "maps/read.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* nearby.c's tool, and what the tools need, are not tested here */
int mp_arg_place(const struct janas_json *args, const char *name, int here,
                 double near_lat, double near_lon, struct mp_place *p,
                 struct janas_buf *b)
{
    (void)args, (void)name, (void)here, (void)near_lat, (void)near_lon;
    (void)p, (void)b;
    return -1;
}
double mp_arg_num(const struct janas_json *args, const char *name, double def)
{
    (void)args, (void)name;
    return def;
}
const char *mp_arg_str(const struct janas_json *args, const char *name)
{
    (void)args, (void)name;
    return NULL;
}
int mp_result(struct janas_buf *out, struct janas_buf *b, int is_error)
{
    (void)b, (void)is_error;
    janas_buf_free(out);
    return 0;
}
int mp_answer(struct janas_buf *b, const char *name, struct janas_buf *d,
              const char *layout, const char *brief)
{
    (void)b, (void)name, (void)layout, (void)brief;
    janas_buf_free(d);
    return 0;
}
int mp_fetch(const char *url, int keep_s, struct janas_buf *out, char *err,
             size_t err_len)
{
    (void)url, (void)keep_s, (void)out, (void)err, (void)err_len;
    return -1;
}
const char *mp_base(const char *env, const char *def)
{
    (void)env;
    return def;
}
void mp_url_put(struct janas_buf *b, const char *s)
{
    janas_buf_puts(b, s);
}

static const char VALHALLA[] =
    "{\"trip\": {\"language\": \"it\", \"summary\": {\"has_toll\": true, \"has_hig"
    "hway\": true, \"has_ferry\": false, \"time\": 5950.149, \"length\": 110.4"
    "9}, \"legs\": [{\"maneuvers\": [{\"type\": 3, \"instruction\": \"Guida vers"
    "o est su Via Giardini.\", \"length\": 0.028, \"street_names\": [\"Via Gi"
    "ardini\"]}, {\"type\": 10, \"instruction\": \"Svolta a destra su Via Pal"
    "merio Abate.\", \"length\": 0.048, \"street_names\": [\"Via Palmerio Aba"
    "te\"]}, {\"type\": 27, \"instruction\": \"Esci dalla rotonda verso A29di"
    "r/Palermo.\", \"length\": 7.424, \"street_names\": [\"A29dir\", \"E 933\"]}"
    ", {\"type\": 24, \"instruction\": \"Mantieni la sinistra per rimanere s"
    "u E 933 verso Palermo.\", \"length\": 29.193, \"street_names\": [\"E 933"
    "\"]}, {\"type\": 20, \"instruction\": \"Prendi l'uscita A29 verso Palerm"
    "o/aeroporto Falcone Borsellino.\", \"length\": 58.84, \"street_names\":"
    " [\"E 90\"]}, {\"type\": 4, \"instruction\": \"Sei arrivato a destinazion"
    "e.\", \"length\": 0.0}]}]}, \"alternates\": [{\"trip\": {\"summary\": {\"has"
    "_toll\": false, \"has_highway\": true, \"has_ferry\": false, \"time\": 71"
    "57.349, \"length\": 104.988}, \"legs\": [{\"maneuvers\": [{\"type\": 24, \""
    "instruction\": \"Mantieni la sinistra per rimanere su E 933 verso Pa"
    "lermo.\", \"length\": 29.193, \"street_names\": [\"E 933\"]}, {\"type\": 20"
    ", \"instruction\": \"Prendi l'uscita A29 verso Palermo/aeroporto Falc"
    "one Borsellino.\", \"length\": 21.372, \"street_names\": [\"A29\", \"E 90\""
    ", \"Autostrada Palermo-Mazara del Vallo\"]}, {\"type\": 20, \"instructi"
    "on\": \"Prendi l'uscita.\", \"length\": 17.315, \"street_names\": [\"SS624"
    "\"]}]}]}}]}";

static const char VALHALLA_ERROR[] =
    "{\"error_code\": 442, \"error\": \"No path could be found for input\", \""
    "status_code\": 400, \"status\": \"Bad Request\"}";

static const char OSRM[] =
    "{\"code\": \"Ok\", \"routes\": [{\"distance\": 109363, \"duration\": 5731.7,"
    " \"legs\": [{\"summary\": \"Autostrada Alcamo-Trapani, Autostrada Paler"
    "mo-Mazara del Vallo\"}]}]}";

static const char TRANSIT[] =
    "{\"itineraries\": [{\"duration\": 13200, \"startTime\": \"2026-10-02T03:4"
    "2:00Z\", \"endTime\": \"2026-10-02T07:22:00Z\", \"transfers\": 2, \"legs\":"
    " [{\"mode\": \"WALK\", \"startTime\": \"2026-10-02T03:42:00Z\", \"endTime\":"
    " \"2026-10-02T03:50:00Z\", \"from\": {\"name\": \"START\", \"tz\": \"Europe/R"
    "ome\"}, \"to\": {\"name\": \"TRAPANI\"}, \"distance\": 494.0, \"realTime\": f"
    "alse}, {\"mode\": \"REGIONAL_RAIL\", \"startTime\": \"2026-10-02T03:50:00"
    "Z\", \"endTime\": \"2026-10-02T04:26:00Z\", \"from\": {\"name\": \"TRAPANI\","
    " \"tz\": \"Europe/Rome\"}, \"to\": {\"name\": \"MARSALA\"}, \"displayName\": \""
    "REG 21854\", \"routeShortName\": \"REG 21854\", \"agencyName\": \"TRENITAL"
    "IA\", \"headsign\": \"\", \"realTime\": false}, {\"mode\": \"WALK\", \"startTi"
    "me\": \"2026-10-02T04:26:00Z\", \"endTime\": \"2026-10-02T04:28:00Z\", \"f"
    "rom\": {\"name\": \"MARSALA\", \"tz\": \"Europe/Rome\"}, \"to\": {\"name\": \"MA"
    "RSALA-VIA AMERIGO FAZIO\"}, \"distance\": 418.0, \"realTime\": false}, "
    "{\"mode\": \"BUS\", \"startTime\": \"2026-10-02T04:30:00Z\", \"endTime\": \"2"
    "026-10-02T07:05:00Z\", \"from\": {\"name\": \"MARSALA-VIA AMERIGO FAZIO\""
    ", \"tz\": \"Europe/Rome\"}, \"to\": {\"name\": \"PALERMO (Piazza Cairoli)\"}"
    ", \"displayName\": \"MARSALA-VIA AMERIGO FAZIO - PALERMO P.zza Cairol"
    "i\", \"routeShortName\": \"MARSALA-VIA AMERIGO FAZIO - PALERMO P.zza C"
    "airoli\", \"agencyName\": \"Salemi\", \"headsign\": \"Palermo P.Zza Cairol"
    "i\", \"realTime\": false}, {\"mode\": \"WALK\", \"startTime\": \"2026-10-02T"
    "07:05:00Z\", \"endTime\": \"2026-10-02T07:09:00Z\", \"from\": {\"name\": \"P"
    "ALERMO (Piazza Cairoli)\", \"tz\": \"Europe/Rome\"}, \"to\": {\"name\": \"ST"
    "AZIONE CENTRALE GIULIO CESARE\"}, \"distance\": 485.0, \"realTime\": fa"
    "lse}, {\"mode\": \"BUS\", \"startTime\": \"2026-10-02T07:12:00Z\", \"endTim"
    "e\": \"2026-10-02T07:17:00Z\", \"from\": {\"name\": \"STAZIONE CENTRALE GI"
    "ULIO CESARE\", \"tz\": \"Europe/Rome\"}, \"to\": {\"name\": \"ROMA LATTARINI"
    "\"}, \"displayName\": \"101\", \"routeShortName\": \"101\", \"agencyName\": \""
    "AMAT Palermo S.p.A.\", \"headsign\": \"STADIO\", \"realTime\": false}, {\""
    "mode\": \"WALK\", \"startTime\": \"2026-10-02T07:17:00Z\", \"endTime\": \"20"
    "26-10-02T07:22:00Z\", \"from\": {\"name\": \"ROMA LATTARINI\", \"tz\": \"Eur"
    "ope/Rome\"}, \"to\": {\"name\": \"END\"}, \"distance\": 300.0, \"realTime\": "
    "false}]}]}";

static const char OVERPASS[] =
    "{\"version\": 0.6, \"elements\": [{\"type\": \"node\", \"id\": 1073287420, \""
    "lat\": 38.0152454, \"lon\": 12.5137922, \"tags\": {\"amenity\": \"pharmacy"
    "\", \"dispensing\": \"yes\", \"healthcare\": \"pharmacy\", \"name\": \"Occhipi"
    "nti\", \"operator\": \"Vito Occhipinti\", \"phone\": \"+39092321663\", \"ref"
    ":msal\": \"15725\", \"ref:vatin\": \"IT01922580814\", \"source\": \"DatiOpen"
    ".it - Provincia Regionale di Trapani\", \"source:url\": \"http://www.d"
    "atiopen.it/it/data/Farmacie_della_Provincia_di_Trapani\"}}, {\"type\""
    ": \"node\", \"id\": 1073287769, \"lat\": 38.0158833, \"lon\": 12.5107126, "
    "\"tags\": {\"addr:housenumber\": \"37\", \"addr:postcode\": \"91100\", \"addr"
    ":street\": \"Via della Cuba\", \"amenity\": \"pharmacy\", \"dispensing\": \""
    "yes\", \"healthcare\": \"pharmacy\", \"name\": \"Giglio Ruggiero & C. s.n."
    "c.\", \"ref:msal\": \"15690\", \"ref:vatin\": \"IT02166410817\", \"source\": "
    "\"DatiOpen.it - Provincia Regionale di Trapani\", \"source:url\": \"htt"
    "p://www.datiopen.it/it/data/Farmacie_della_Provincia_di_Trapani\"}}"
    ", {\"type\": \"node\", \"id\": 1073453284, \"lat\": 38.0150992, \"lon\": 12."
    "5075134, \"tags\": {\"addr:housenumber\": \"32\", \"addr:postcode\": \"9110"
    "0\", \"addr:street\": \"Via San Francesco d'Assisi\", \"amenity\": \"pharm"
    "acy\", \"dispensing\": \"yes\", \"healthcare\": \"pharmacy\", \"name\": \"Rest"
    "ivo Ilaria\", \"ref:msal\": \"15695\", \"ref:vatin\": \"IT00094860814\", \"s"
    "ource\": \"DatiOpen.it - Provincia Regionale di Trapani\", \"source:ur"
    "l\": \"http://www.datiopen.it/it/data/Farmacie_della_Provincia_di_Tr"
    "apani\"}}, {\"type\": \"node\", \"id\": 1624173475, \"lat\": 38.0193717, \"l"
    "on\": 12.5215576, \"tags\": {\"addr:housenumber\": \"119\", \"addr:street\""
    ": \"Via Giovanbattista Fardella\", \"amenity\": \"pharmacy\", \"dispensin"
    "g\": \"yes\", \"healthcare\": \"pharmacy\", \"name\": \"Garraffa Vincenzo\", "
    "\"ref:msal\": \"15710\", \"ref:vatin\": \"IT02373400817\", \"source\": \"Dati"
    "Open.it - Provincia Regionale di Trapani\", \"source:url\": \"http://w"
    "ww.datiopen.it/it/data/Farmacie_della_Provincia_di_Trapani\"}}, {\"t"
    "ype\": \"way\", \"id\": 4242, \"center\": {\"lat\": 38.0176, \"lon\": 12.5151"
    "}, \"tags\": {\"amenity\": \"pharmacy\", \"opening_hours\": \"Mo-Sa 08:30-1"
    "3:00,16:30-20:00\"}}]}";

static const char NOMINATIM[] =
    "[{\"place_id\": 53253714, \"licence\": \"Data © OpenStreetMap contribut"
    "ors, ODbL 1.0. http://osm.org/copyright\", \"osm_type\": \"way\", \"osm_"
    "id\": 381762269, \"lat\": \"38.1101703\", \"lon\": \"13.3680470\", \"categor"
    "y\": \"railway\", \"type\": \"platform\", \"place_rank\": 30, \"importance\":"
    " 7.561226825355943e-05, \"addresstype\": \"railway\", \"name\": \"Stazion"
    "e Centrale\", \"display_name\": \"Stazione Centrale, Via Paolo Balsamo"
    ", Brancaccio, II Circoscrizione, Palermo, Sicilia, 90123, Italia\","
    " \"address\": {\"railway\": \"Stazione Centrale\", \"road\": \"Via Paolo Ba"
    "lsamo\", \"suburb\": \"Brancaccio\", \"city\": \"Palermo\", \"county\": \"Pale"
    "rmo\", \"ISO3166-2-lvl6\": \"IT-PA\", \"state\": \"Sicilia\", \"ISO3166-2-lv"
    "l4\": \"IT-82\", \"postcode\": \"90123\", \"country\": \"Italia\", \"country_c"
    "ode\": \"it\"}, \"boundingbox\": [\"38.1100702\", \"38.1102703\", \"13.36783"
    "15\", \"13.3682624\"]}]";

static const char PHOTON[] =
    "{\"type\": \"FeatureCollection\", \"features\": [{\"type\": \"Feature\", \"pr"
    "operties\": {\"osm_type\": \"W\", \"osm_id\": 105508402, \"osm_key\": \"high"
    "way\", \"osm_value\": \"residential\", \"type\": \"street\", \"name\": \"Via M"
    "ichelangelo Fardella\", \"district\": \"Trapani\", \"city\": \"Trapani\", \""
    "county\": \"Trapani\", \"state\": \"Sicilia\", \"country\": \"Italia\", \"post"
    "code\": \"91100\", \"countrycode\": \"IT\", \"extent\": [12.5251702, 38.017"
    "7973, 12.5256313, 38.0154919]}, \"geometry\": {\"type\": \"Point\", \"coo"
    "rdinates\": [12.5253913, 38.0166427]}}]}";

/* Nominatim's "Trapani", cut: the municipality, the province, the town */
static const char TOWN[] =
    "[{\"osm_type\": \"relation\", \"osm_id\": 39150, \"lat\": "
    "\"37.9003731\", \"lon\": \"12.7116255\", \"category\": \"boundary\", "
    "\"type\": \"administrative\", \"name\": \"Trapani\", \"display_name\": "
    "\"Trapani, Sicilia, 91100, Italia\"}, {\"osm_type\": \"relation\", "
    "\"osm_id\": 39151, \"lat\": \"37.8745486\", \"lon\": \"12.7179792\", "
    "\"category\": \"boundary\", \"type\": \"administrative\", \"name\": "
    "\"Trapani\"}, {\"osm_type\": \"node\", \"osm_id\": 1, \"lat\": "
    "\"38.0173961\", \"lon\": \"12.5160225\", \"category\": \"place\", "
    "\"type\": \"city\", \"name\": \"Trapani\"}]";

/* The layout filled with the data: the text, or "" when it failed. */
static char text[16384];

static const char *fill(const char *layout, const struct janas_buf *d)
{
    text[0] = 0;
    struct janas_json_doc *doc =
        janas_json_parse(d->p ? d->p : "", d->n, NULL, 0);
    CHECK(doc != NULL, "the data is not JSON:\n%.*s", (int)d->n, d->p);
    if (!doc)
        return text;
    struct janas_buf out = {0};
    char err[200];
    if (janas_tpl_render(layout, strlen(layout), janas_json_root(doc), &out,
                         err, sizeof err) == 0)
        snprintf(text, sizeof text, "%.*s", (int)out.n, out.p);
    else
        CHECK(0, "a layout not filled: %s", err);
    janas_buf_free(&out);
    janas_json_free(doc);
    return text;
}

static void place(struct mp_place *p, const char *name, double lat, double lon)
{
    memset(p, 0, sizeof *p);
    snprintf(p->name, sizeof p->name, "%s", name);
    p->lat = lat;
    p->lon = lon;
}

static void routes(void)
{
    struct mp_routes rs;
    char err[256];
    CHECK(mp_read_valhalla(VALHALLA, strlen(VALHALLA), &rs, err, sizeof err) ==
              0,
          "Valhalla: %s", err);
    CHECK(rs.n == 2 && fabs(rs.r[0].km - 110.49) < 0.01 &&
              fabs(rs.r[0].min - 99.17) < 0.01 && rs.r[0].toll &&
              rs.r[0].highway && !rs.r[1].toll,
          "Valhalla: %d routes, %.2f km, %.2f min", rs.n, rs.r[0].km,
          rs.r[0].min);
    CHECK(strcmp(rs.r[0].roads, "A29dir, E 933, E 90") == 0,
          "the roads: \"%s\"", rs.r[0].roads);
    CHECK(rs.r[0].n_step == 6 && rs.r[1].step == NULL &&
              strcmp(rs.r[0].step[5].text, "Sei arrivato a destinazione.") == 0,
          "the steps: %d", rs.r[0].n_step);

    struct mp_place from, to;
    place(&from, "Trapani", 38.0175, 12.515);
    snprintf(from.how, sizeof from.how,
             "estimated from the internet connection (GeoJS)");
    place(&to, "Palermo", 38.1157, 13.3615);
    struct janas_buf d = {0};
    mp_route_data(&d, &from, NULL, &to, "car", &rs);
    mp_routes_free(&rs);
    const char *t = fill(MP_ROUTE_LAYOUT, &d);
    CHECK(strstr(t, "From Trapani (where the user is, estimated from the "
                    "internet connection (GeoJS)) to Palermo by car: 110 km, "
                    "about 1 h 39 min without traffic, by A29dir, E 933, E "
                    "90; with tolls.\n") == t,
          "the route:\n%s", t);
    CHECK(
        strstr(t,
               "Another way: 105 km, about 1 h 59 min, by E 933, A29, SS624."),
        "the other way, without tolls:\n%s", t);
    CHECK(strstr(t, "\n1. Guida verso est su Via Giardini. (30 m)\n") &&
              strstr(t, "\n5. Prendi l'uscita A29 verso Palermo/aeroporto "
                        "Falcone Borsellino. (59 km)\n") &&
              strstr(t, "\n6. Sei arrivato a destinazione.\n"),
          "the steps, each with its own length:\n%s", t);
    CHECK(strstr(t, "directions?engine=fossgis_valhalla_car&route=38.01750%"
                    "2C12.51500%3B38.11570%2C13.36150\n"),
          "the link:\n%s", t);
    t = fill(MP_ROUTE_BRIEF, &d);
    CHECK(strstr(t, "car from Trapani (where the user is") == t &&
              !strstr(t, "Guida"),
          "the brief:\n%s", t);
    janas_buf_free(&d);

    CHECK(mp_read_valhalla(VALHALLA_ERROR, strlen(VALHALLA_ERROR), &rs, err,
                           sizeof err) == -1 &&
              strstr(err, "No path could be found"),
          "Valhalla's error: %s", err);
    CHECK(mp_read_osrm(OSRM, strlen(OSRM), &rs, err, sizeof err) == 0 &&
              rs.n == 1 && fabs(rs.r[0].km - 109.363) < 0.001 &&
              strcmp(rs.r[0].roads, "Autostrada Alcamo-Trapani, Autostrada "
                                    "Palermo-Mazara del Vallo") == 0 &&
              rs.r[0].step == NULL,
          "OSRM: %s", err);
    mp_routes_free(&rs);
    static const char far[] =
        "{\"code\": \"Ok\", \"routes\": [{\"distance\": 3556000, "
        "\"duration\": 139380, \"legs\": [{\"summary\": \"\"}]}], "
        "\"waypoints\": [{\"distance\": 6.0}, {\"distance\": 2915000.5}]}";
    CHECK(mp_read_osrm(far, strlen(far), &rs, err, sizeof err) == -1 &&
              strstr(err, "2915 km from the nearest road"),
          "OSRM, a place off its map: %s", err);
}

static void transit(void)
{
    CHECK(mp_utc("2026-10-02T03:42:00Z") == 1790912520 && mp_utc("x") == -1,
          "mp_utc");
    struct mp_transit tr;
    char err[256];
    CHECK(mp_read_transit(TRANSIT, strlen(TRANSIT), &tr, err, sizeof err) ==
                  0 &&
              tr.n == 1 && tr.t[0].n_leg == 7 && tr.t[0].transfers == 2 &&
              strcmp(tr.tz, "Europe/Rome") == 0,
          "Transitous: %s", err);
    struct mp_place from, to;
    place(&from, "Via Giardini, Trapani", 38.0175, 12.515);
    place(&to, "Palermo", 38.1157, 13.3615);
    struct janas_buf d = {0};
    mp_transit_data(&d, &from, &to, &tr);
    const char *t = fill(MP_TRANSIT_LAYOUT, &d);
    CHECK(strstr(t, "By public transport from Via Giardini, Trapani to "
                    "Palermo; journeys found: 1.\n") == t &&
              strstr(t, "\n1. Leaves at 05:42") &&
              strstr(t, "(3 h 40 min, changes: 2):\n") &&
              strstr(t, "   - walk 8 min, 490 m to TRAPANI\n") &&
              strstr(t, "   - 05:50 regional train REG 21854 (TRENITALIA) "
                        "from TRAPANI; 06:26 get off at MARSALA\n") &&
              strstr(t, " to Palermo\n") && strstr(t, "local (Europe/Rome)"),
          "the journeys:\n%s", t);
    janas_buf_free(&d);
}

static void near(void)
{
    struct mp_near nr;
    char err[256];
    CHECK(mp_read_overpass(OVERPASS, strlen(OVERPASS), 38.0175, 12.515, 10, &nr,
                           err, sizeof err) == 0 &&
              nr.found == 5 && nr.n == 5,
          "Overpass: %s", err);
    for (int i = 1; i < nr.n; i++)
        CHECK(nr.p[i].m >= nr.p[i - 1].m, "not the nearest first");
    CHECK(strcmp(nr.p[0].osm, "way/4242") == 0 && !nr.p[0].name[0],
          "the nearest: %s", nr.p[0].osm);
    struct mp_place at;
    place(&at, "Trapani", 38.0175, 12.515);
    struct janas_buf d = {0};
    mp_near_data(&d, &at, "pharmacy", 1, &nr);
    const char *t = fill(MP_NEAR_LAYOUT, &d);
    CHECK(strstr(
              t,
              "Within 1 km of Trapani: 5 pharmacies; these are the 5 nearest:\n"
              "1. (no name) - 10 m ") == t &&
              strstr(t, "; open: Mo-Sa 08:30-13:00,16:30-20:00\n") &&
              strstr(t, "Via della Cuba 37"),
          "the pharmacies:\n%s", t);
    janas_buf_free(&d);
    static const char busy[] =
        "<?xml version=\"1.0\"?>\n<p><strong>Error</strong>: runtime error: "
        "Dispatcher_Client::request_read_and_idx::timeout. The server is "
        "probably too busy to handle your request. </p>";
    CHECK(mp_read_overpass(busy, strlen(busy), 38, 12, 10, &nr, err,
                           sizeof err) == -1 &&
              strstr(err, "too busy now"),
          "Overpass, too busy: %s", err);
    /* every kind has its words */
    for (size_t i = 0; i < MP_N_KINDS; i++) {
        char w[64];
        snprintf(w, sizeof w, "\nwhat.%s = ", MP_KINDS[i].name);
        CHECK(strstr(MP_NEAR_LAYOUT, w), "no words for %s", MP_KINDS[i].name);
    }
}

static void places(void)
{
    struct mp_place p;
    CHECK(mp_read_nominatim(NOMINATIM, strlen(NOMINATIM), &p) == 0 &&
              strcmp(p.name, "Stazione Centrale") == 0 &&
              strcmp(p.city, "Palermo") == 0 &&
              strcmp(p.osm, "way/381762269") == 0 &&
              strcmp(p.kind, "railway=platform") == 0 &&
              fabs(p.lat - 38.1101703) < 1e-7,
          "Nominatim: %s, %s, %s", p.name, p.city, p.osm);
    CHECK(mp_read_nominatim("[]", 2, &p) == -1, "Nominatim, nothing found");
    struct mp_place rev;
    CHECK(mp_read_nominatim(NOMINATIM + 1, strlen(NOMINATIM) - 2, &rev) == 0 &&
              strcmp(rev.name, "Stazione Centrale") == 0,
          "Nominatim's reverse lookup, one place");
    struct janas_buf d = {0};
    mp_find_data(&d, &p);
    const char *t = fill(MP_FIND_LAYOUT, &d);
    CHECK(strstr(t, "Stazione Centrale:\n- address: Stazione Centrale, Via "
                    "Paolo Balsamo") == t &&
              strstr(t, "- on the map: https://www.openstreetmap.org/?mlat="
                        "38.11017&mlon=13.36805#map=17/38.11017/13.36805\n"),
          "the place:\n%s", t);
    janas_buf_free(&d);
    CHECK(mp_read_nominatim(TOWN, strlen(TOWN), &p) == 0 &&
              fabs(p.lat - 38.0173961) < 1e-7 &&
              strcmp(p.osm, "relation/39150") == 0,
          "a town by its point, not its boundary's: %f", p.lat);
    CHECK(mp_read_photon(PHOTON, strlen(PHOTON), &p) == 0 &&
              strcmp(p.name, "Via Michelangelo Fardella") == 0 &&
              strcmp(p.city, "Trapani") == 0 &&
              strcmp(p.osm, "way/105508402") == 0 &&
              fabs(p.lon - 12.5253913) < 1e-7,
          "Photon: %s, %s, %s", p.name, p.city, p.osm);
}

int main(void)
{
    setenv("TZ", "UTC", 1);
    routes();
    transit();
    near();
    places();
    if (failures) {
        printf("test_maps_parse: %d failures\n", failures);
        return 1;
    }
    printf("test_maps_parse: ok\n");
    return 0;
}
