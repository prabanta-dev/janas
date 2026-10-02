/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_quakes_parse.c - janas-quakes' reading of its sources, without the
 * network, and the layouts filled with what was read: INGV's two latest
 * earthquakes within 300 km of Palermo, the USGS's two strongest of a
 * week and its answer for one event's id, as they gave them on 3 October
 * 2026 (the USGS's cut to the fields read). The times are told in UTC.
 * The sources belong to the program, so they are compiled in here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "services/common/template.h"
#include "services/quakes/data.c"
#include "services/quakes/layouts.c"
#include "services/quakes/read.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char S_INGV[] =
    "{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"p"
    "roperties\":{\"eventId\":46749431,\"originId\":147776221,\"time\":\""
    "2026-10-01T22:46:42.590000\",\"author\":\"SURVEY-INGV\",\"magType\""
    ":\"ML\",\"mag\":2.4,\"magAuthor\":\"--\",\"type\":\"earthquake\",\"place\""
    ":\"Costa Siciliana nord-orientale (Messina)\",\"version\":100,\"g"
    "eojson_creationTime\":\"2026-10-02T22:41:03\"},\"geometry\":{\"typ"
    "e\":\"Point\",\"coordinates\":[15.2578,38.3912,168.3]}},{\"type\":\""
    "Feature\",\"properties\":{\"eventId\":46749241,\"originId\":1477758"
    "31,\"time\":\"2026-10-01T20:28:14.590000\",\"author\":\"SURVEY-INGV"
    "\",\"magType\":\"ML\",\"mag\":3,\"magAuthor\":\"--\",\"type\":\"earthquake"
    "\",\"place\":\"Costa Calabra nord-occidentale (Cosenza)\",\"versio"
    "n\":100,\"geojson_creationTime\":\"2026-10-02T22:41:03\"},\"geomet"
    "ry\":{\"type\":\"Point\",\"coordinates\":[15.6732,39.5417,271.4]}}]"
    "}";

static const char S_USGS[] =
    "{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"p"
    "roperties\":{\"mag\":5.8,\"place\":\"165 km SSE of Vilyuchinsk, Ru"
    "ssia\",\"time\":1790958881010,\"felt\":null,\"mmi\":3.738,\"alert\":\""
    "green\",\"status\":\"reviewed\",\"tsunami\":0,\"magType\":\"mww\",\"type"
    "\":\"earthquake\",\"url\":\"https://earthquake.usgs.gov/earthquake"
    "s/eventpage/us6000tz62\"},\"geometry\":{\"type\":\"Point\",\"coordin"
    "ates\":[159.7398,51.6873,29.477]},\"id\":\"us6000tz62\"},{\"type\":"
    "\"Feature\",\"properties\":{\"mag\":5.6,\"place\":\"72 km ESE of Koko"
    "po, Papua New Guinea\",\"time\":1790431698875,\"felt\":1,\"mmi\":4."
    "2,\"alert\":\"green\",\"status\":\"reviewed\",\"tsunami\":0,\"magType\":"
    "\"mww\",\"type\":\"earthquake\",\"url\":\"https://earthquake.usgs.gov"
    "/earthquakes/eventpage/us6000txtc\"},\"geometry\":{\"type\":\"Poin"
    "t\",\"coordinates\":[152.8611,-4.6288,69.51]},\"id\":\"us6000txtc\""
    "}],\"bbox\":[152.8611,-4.6288,29.477,159.7398,51.6873,69.51]}";

static const char S_USGS_ONE[] =
    "{\"type\":\"Feature\",\"properties\":{\"mag\":5.8,\"place\":\"165 km SS"
    "E of Vilyuchinsk, Russia\",\"time\":1790958881010,\"felt\":null,\""
    "mmi\":3.738,\"alert\":\"green\",\"status\":\"reviewed\",\"tsunami\":0,\""
    "magType\":\"mww\",\"type\":\"earthquake\",\"url\":\"https://earthquake"
    ".usgs.gov/earthquakes/eventpage/us6000tz62\"},\"geometry\":{\"ty"
    "pe\":\"Point\",\"coordinates\":[159.7398,51.6873,29.477]},\"id\":\"u"
    "s6000tz62\"}";

/* the layout filled with the data d; the text, or "" (counted) */
static char *fill(const char *layout, struct janas_buf *d)
{
    static char out[16384];
    char why[200];
    struct janas_json_doc *doc = janas_json_parse(d->p, d->n, why, sizeof why);
    CHECK(doc, "the data is not JSON: %s\n%.*s", why, (int)d->n, d->p);
    out[0] = 0;
    if (doc) {
        struct janas_buf o = {0};
        char err[200];
        int r = janas_tpl_render(layout, strlen(layout), janas_json_root(doc),
                                 &o, err, sizeof err);
        CHECK(r == 0, "not filled: %s", err);
        if (r == 0 && o.p)
            snprintf(out, sizeof out, "%.*s", (int)o.n, o.p);
        janas_buf_free(&o);
        janas_json_free(doc);
    }
    janas_buf_free(d);
    return out;
}

static struct qk_list l;

static void test_read(void)
{
    char err[300] = "";
    CHECK(qk_utc("2026-10-01T22:46:42.590000") == 1790894802LL, "utc %lld",
          qk_utc("2026-10-01T22:46:42.590000"));
    CHECK(qk_utc("1970-01-01T00:00:00") == 0 && qk_utc("x") == -1, "utc ends");
    memset(&l, 0, sizeof l);
    CHECK(qk_read(S_INGV, sizeof S_INGV - 1, "ingv", &l, err, sizeof err) == 0,
          "ingv: %s", err);
    CHECK(
        l.n == 2 && !strcmp(l.e[0].id, "46749431") && l.e[0].mag == 2.4 &&
            l.e[0].depth_km == 168.3 && l.e[0].lat == 38.3912 &&
            l.e[0].lon == 15.2578 && !strcmp(l.e[0].magtype, "ML") &&
            !strcmp(l.e[0].place, "Costa Siciliana nord-orientale (Messina)") &&
            !strcmp(l.e[0].url, "https://terremoti.ingv.it/event/46749431") &&
            l.e[0].felt == -1 && l.e[0].mmi == -1,
        "ingv event: %d %s", l.n, l.e[0].id);
    CHECK(l.n == 2 && l.e[1].mag == 3, "an integer magnitude");
    CHECK(qk_read("", 0, "ingv", &l, err, sizeof err) == 0 && l.n == 2,
          "INGV's 204, none");
    memset(&l, 0, sizeof l);
    CHECK(qk_read(S_USGS, sizeof S_USGS - 1, "usgs", &l, err, sizeof err) == 0,
          "usgs: %s", err);
    CHECK(l.n == 2 && !strcmp(l.e[0].id, "us6000tz62") && l.e[0].mag == 5.8 &&
              l.e[0].at == 1790958881LL && l.e[0].felt == -1 &&
              l.e[0].mmi == 3.738 && !strcmp(l.e[0].alert, "green") &&
              l.e[0].reviewed && !l.e[0].tsunami && l.e[1].felt == 1,
          "usgs event: %s %lld %d", l.e[0].id, l.e[0].at, l.e[0].felt);
    struct qk_list one;
    memset(&one, 0, sizeof one);
    CHECK(qk_read(S_USGS_ONE, sizeof S_USGS_ONE - 1, "usgs", &one, err,
                  sizeof err) == 0 &&
              one.n == 1 && !strcmp(one.e[0].id, "us6000tz62"),
          "one event: %d", one.n);
    CHECK(qk_read("{\"x\": 1}", 8, "usgs", &one, err, sizeof err) == -1,
          "not an answer of events");
    qk_by_time(&l);
    CHECK(l.e[0].at > l.e[1].at, "the latest first");
}

static void test_layouts(void)
{
    char err[300];
    struct qk_place p = {
        .name = "Palermo", .country = "Italy", .lat = 38.1157, .lon = 13.3615};
    memset(&l, 0, sizeof l);
    qk_read(S_INGV, sizeof S_INGV - 1, "ingv", &l, err, sizeof err);
    qk_by_time(&l);
    struct janas_buf d = {0};
    qk_near_data(&d, &p, 300, 7, 2.5, &l, "ingv");
    const char *t = fill(QK_NEAR_LAYOUT, &d);
    CHECK(strstr(t, "Earthquakes within 300 km of Palermo in the last 7 "
                    "days, of magnitude 2.5 or more:\n"),
          "near head:\n%s", t);
    CHECK(strstr(t, "  1/10 22:46: magnitude 2.4, 168 km deep, 168 km E of "
                    "Palermo: Costa Siciliana nord-orientale (Messina)\n"),
          "near item:\n%s", t);
    CHECK(strstr(t, "magnitude 3.0, 271 km deep"), "a magnitude 3.0:\n%s", t);
    CHECK(strstr(t, "Source: INGV (Istituto Nazionale di Geofisica e "
                    "Vulcanologia, CC BY 4.0)."),
          "source:\n%s", t);
    memset(&l, 0, sizeof l);
    qk_near_data(&d, &p, 300, 7, 2.5, &l, "ingv");
    t = fill(QK_NEAR_LAYOUT, &d);
    CHECK(strstr(t, "No earthquake of magnitude 2.5 or more within 300 km of "
                    "Palermo in the last 7 days.\n"),
          "none:\n%s", t);

    memset(&l, 0, sizeof l);
    qk_read(S_USGS, sizeof S_USGS - 1, "usgs", &l, err, sizeof err);
    l.e[1].tsunami = 1; /* to see the words */
    qk_by_mag(&l);
    qk_strong_data(&d, 7, 5, &l);
    t = fill(QK_STRONG_LAYOUT, &d);
    /* the day is "today" or 2/10, as the test is run: left out */
    CHECK(strstr(t, "  magnitude 5.8, ") &&
              strstr(t, " 16:34: 165 km SSE of Vilyuchinsk, Russia, 29 km "
                        "deep; impact: green, little damage expected\n"),
          "strong:\n%s", t);
    CHECK(strstr(t, "at sea and strong enough for a tsunami: see the warning "
                    "centres"),
          "tsunami words:\n%s", t);
    qk_strong_data(&d, 7, 5, &l);
    t = fill(QK_STRONG_BRIEF, &d);
    CHECK(strstr(t, "tsunami flag (not a tsunami seen)"), "brief:\n%s", t);

    qk_event_data(&d, &l.e[1], NULL);
    t = fill(QK_EVENT_LAYOUT, &d);
    CHECK(strstr(t, "Earthquake of magnitude 5.6 (mww): 72 km ESE of Kokopo, "
                    "Papua New Guinea\n") &&
              strstr(t, "  felt: 1 people told the USGS they felt it\n") &&
              strstr(t, "  the strongest shaking, estimated: 4 on the "
                        "Mercalli scale\n") &&
              strstr(t, "  more: https://earthquake.usgs.gov/earthquakes/"
                        "eventpage/us6000txtc\n") &&
              !strstr(t, "from "),
          "event:\n%s", t);
}

int main(void)
{
    setenv("TZ", "UTC", 1);
    tzset();
    test_read();
    test_layouts();
    if (failures) {
        printf("test_quakes_parse: %d failures\n", failures);
        return 1;
    }
    printf("test_quakes_parse: ok\n");
    return 0;
}
