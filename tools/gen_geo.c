/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gen_geo.c - generates src/services/common/geo_data.h, the tables of Janas's
 * services' geography (src/services/common/geo.h): countries, first-level
 * regions, time zones, cities of 30,000 people or more and every capital, the
 * airports with an IATA code and scheduled flights, and the seas as polygons.
 *
 * It reads, from one directory:
 *   airports.csv            OurAirports, public domain
 *       https://davidmegginson.github.io/ourairports-data/airports.csv
 *   cities15000.txt         GeoNames, CC BY 4.0 (unzipped)
 *       https://download.geonames.org/export/dump/cities15000.zip
 *   admin1CodesASCII.txt    GeoNames, the same
 *   countryInfo.txt         GeoNames, the same
 *   ne_10m_geography_marine_polys.geojson   Natural Earth, public domain
 *       https://github.com/nvkelso/natural-earth-vector (geojson/)
 *
 * An airport's time zone is that of the nearest city. The seas' outlines
 * are simplified to about a kilometre (Douglas-Peucker, 0.01 degree).
 *
 * Usage: gen_geo <data-directory> > src/services/common/geo_data.h
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/json.h"

#define MIN_POP 30000
#define SIMPLIFY 0.01 /* degrees */

static char dir[4096];

static FILE *open_in(const char *name)
{
    char path[8192];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "r");
    if (!f)
        perror(path);
    return f;
}

static void *grow(void *p, size_t *cap, size_t n, size_t size)
{
    if (n < *cap)
        return p;
    *cap = *cap ? *cap * 2 : 1024;
    p = realloc(p, *cap * size);
    if (!p) {
        fprintf(stderr, "gen_geo: out of memory\n");
        exit(1);
    }
    return p;
}

/* A C string literal of s. */
static void put_str(const char *s)
{
    putchar('"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\')
            printf("\\%c", c);
        else if (c < 0x20)
            printf("\\%03o", c);
        else
            putchar(c);
    }
    putchar('"');
}

/* Splits a line at tabs, in place: the number of fields. */
static int split_tab(char *line, char **f, int most)
{
    int n = 0;
    line[strcspn(line, "\r\n")] = 0;
    f[n++] = line;
    for (char *p = line; *p && n < most; p++)
        if (*p == '\t') {
            *p = 0;
            f[n++] = p + 1;
        }
    return n;
}

/* Splits a CSV line (quoted fields, "" inside quotes) in place. */
static int split_csv(char *line, char **f, int most)
{
    int n = 0;
    char *r = line, *w = line;
    line[strcspn(line, "\r\n")] = 0;
    while (n < most) {
        f[n++] = w;
        int quoted = *r == '"';
        if (quoted)
            r++;
        for (;;) {
            if (!*r)
                break;
            if (quoted && *r == '"') {
                if (r[1] == '"') {
                    *w++ = '"';
                    r += 2;
                    continue;
                }
                quoted = 0;
                r++;
                continue;
            }
            if (!quoted && *r == ',')
                break;
            *w++ = *r++;
        }
        int more = *r == ',';
        *w++ = 0;
        if (!more)
            break;
        r++;
    }
    return n;
}

/* ---- countries, regions, time zones ---- */

struct country {
    char iso2[3];
    char *name;
};
static struct country *countries;
static size_t n_countries, cap_countries;

struct region {
    char code[24]; /* "IT.16" */
    int country;
    char *name;
};
static struct region *regions;
static size_t n_regions, cap_regions;

static char **tzs;
static size_t n_tzs, cap_tzs;

static int country_of(const char *iso2)
{
    for (size_t i = 0; i < n_countries; i++)
        if (strcmp(countries[i].iso2, iso2) == 0)
            return (int)i;
    return -1;
}

static int region_of(const char *code)
{
    for (size_t i = 0; i < n_regions; i++)
        if (strcmp(regions[i].code, code) == 0)
            return (int)i;
    return -1;
}

static int tz_of(const char *name)
{
    for (size_t i = 0; i < n_tzs; i++)
        if (strcmp(tzs[i], name) == 0)
            return (int)i;
    tzs = grow(tzs, &cap_tzs, n_tzs, sizeof(*tzs));
    tzs[n_tzs] = strdup(name);
    return (int)n_tzs++;
}

static int read_countries(void)
{
    FILE *f = open_in("countryInfo.txt");
    if (!f)
        return -1;
    char line[4096], *fl[20];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#')
            continue;
        if (split_tab(line, fl, 20) < 5 || strlen(fl[0]) != 2)
            continue;
        countries =
            grow(countries, &cap_countries, n_countries, sizeof(*countries));
        memcpy(countries[n_countries].iso2, fl[0], 3);
        countries[n_countries++].name = strdup(fl[4]);
    }
    fclose(f);
    return 0;
}

static int read_regions(void)
{
    FILE *f = open_in("admin1CodesASCII.txt");
    if (!f)
        return -1;
    char line[4096], *fl[4];
    while (fgets(line, sizeof line, f)) {
        if (split_tab(line, fl, 4) < 3 || strlen(fl[0]) >= 24)
            continue;
        char iso2[3] = {fl[0][0], fl[0][1], 0};
        int c = country_of(iso2);
        if (c < 0)
            continue;
        regions = grow(regions, &cap_regions, n_regions, sizeof(*regions));
        snprintf(regions[n_regions].code, sizeof regions[0].code, "%s", fl[0]);
        regions[n_regions].country = c;
        /* the ASCII name: "Tuscany", as the rest of the text is written */
        regions[n_regions++].name = strdup(fl[2]);
    }
    fclose(f);
    return 0;
}

/* ---- cities ---- */

struct city {
    double lat, lon;
    long pop;
    int country, region, tz;
    char *name, *ascii;
};
static struct city *cities;
static size_t n_cities, cap_cities;

static int read_cities(void)
{
    FILE *f = open_in("cities15000.txt");
    if (!f)
        return -1;
    static char line[1 << 20];
    char *fl[20];
    while (fgets(line, sizeof line, f)) {
        if (split_tab(line, fl, 20) < 19)
            continue;
        long pop = atol(fl[14]);
        if (pop < MIN_POP && strcmp(fl[7], "PPLC") != 0)
            continue;
        int c = country_of(fl[8]);
        if (c < 0)
            continue;
        char code[64];
        snprintf(code, sizeof code, "%s.%s", fl[8], fl[10]);
        cities = grow(cities, &cap_cities, n_cities, sizeof(*cities));
        struct city *y = &cities[n_cities++];
        y->lat = atof(fl[4]);
        y->lon = atof(fl[5]);
        y->pop = pop;
        y->country = c;
        y->region = region_of(code);
        y->tz = tz_of(fl[17]);
        y->name = strdup(fl[1]);
        y->ascii = strdup(fl[2]);
    }
    fclose(f);
    return 0;
}

static double dist2(double la1, double lo1, double la2, double lo2)
{
    double dx = (lo2 - lo1) * cos((la1 + la2) * M_PI / 360), dy = la2 - la1;
    return dx * dx + dy * dy;
}

/* ---- airports ---- */

struct airport {
    double lat, lon;
    int country, tz;
    char iata[4], icao[5];
    char *name, *city;
};
static struct airport *airports;
static size_t n_airports, cap_airports;

static int read_airports(void)
{
    FILE *f = open_in("airports.csv");
    if (!f)
        return -1;
    static char line[1 << 16];
    char *fl[24];
    int first = 1;
    while (fgets(line, sizeof line, f)) {
        if (first) { /* the header */
            first = 0;
            continue;
        }
        if (split_csv(line, fl, 24) < 14)
            continue;
        const char *type = fl[2], *iata = fl[13], *icao = fl[12];
        if (strlen(iata) != 3 || strcmp(fl[11], "yes") != 0 ||
            (strcmp(type, "large_airport") && strcmp(type, "medium_airport") &&
             strcmp(type, "small_airport")))
            continue;
        int c = country_of(fl[8]);
        if (c < 0)
            continue;
        airports = grow(airports, &cap_airports, n_airports, sizeof(*airports));
        struct airport *a = &airports[n_airports++];
        a->lat = atof(fl[4]);
        a->lon = atof(fl[5]);
        a->country = c;
        memcpy(a->iata, iata, 4);
        snprintf(a->icao, sizeof a->icao, "%s", strlen(icao) == 4 ? icao : "");
        a->name = strdup(fl[3]);
        a->city = strdup(fl[10]);
        /* the time zone of the nearest city */
        double best = 1e30;
        a->tz = 0;
        for (size_t i = 0; i < n_cities; i++) {
            double d = dist2(a->lat, a->lon, cities[i].lat, cities[i].lon);
            if (d < best) {
                best = d;
                a->tz = cities[i].tz;
            }
        }
    }
    fclose(f);
    return 0;
}

/* ---- seas ---- */

struct pt {
    int32_t lon, lat;
};
static struct pt *pts;
static size_t n_pts, cap_pts;
struct ring {
    uint32_t pt0, n;
};
static struct ring *rings;
static size_t n_rings, cap_rings;
struct sea {
    char *name;
    uint32_t ring0, n_rings;
    int32_t lat_min, lat_max, lon_min, lon_max;
};
static struct sea *seas;
static size_t n_seas, cap_seas;

/* Douglas-Peucker over v[a..b]: keep[] marks the points kept. */
static void simplify(const double (*v)[2], size_t a, size_t b, char *keep)
{
    if (b <= a + 1)
        return;
    double x1 = v[a][0], y1 = v[a][1], x2 = v[b][0], y2 = v[b][1];
    double dx = x2 - x1, dy = y2 - y1, len = sqrt(dx * dx + dy * dy);
    size_t far = a;
    double most = -1;
    for (size_t i = a + 1; i < b; i++) {
        double d =
            len > 0
                ? fabs(dy * v[i][0] - dx * v[i][1] + x2 * y1 - y2 * x1) / len
                : hypot(v[i][0] - x1, v[i][1] - y1);
        if (d > most) {
            most = d;
            far = i;
        }
    }
    if (most > SIMPLIFY) {
        keep[far] = 1;
        simplify(v, a, far, keep);
        simplify(v, far, b, keep);
    }
}

static void add_ring(const struct janas_json *ring, struct sea *s)
{
    size_t n = ring->n;
    if (n < 4)
        return;
    double(*v)[2] = malloc(n * sizeof(*v));
    char *keep = calloc(n, 1);
    size_t i = 0;
    for (const struct janas_json *p = ring->child; p && i < n; p = p->next) {
        v[i][0] = janas_json_num(p->child, 0);
        v[i][1] = janas_json_num(p->child ? p->child->next : NULL, 0);
        i++;
    }
    keep[0] = keep[n - 1] = 1;
    /* a closed ring: split at the point farthest from the first, so the
       two halves have distinct ends */
    size_t far = 0;
    double most = -1;
    for (size_t k = 1; k < n; k++) {
        double d = hypot(v[k][0] - v[0][0], v[k][1] - v[0][1]);
        if (d > most) {
            most = d;
            far = k;
        }
    }
    keep[far] = 1;
    simplify((const double(*)[2])v, 0, far, keep);
    simplify((const double(*)[2])v, far, n - 1, keep);
    rings = grow(rings, &cap_rings, n_rings, sizeof(*rings));
    rings[n_rings].pt0 = (uint32_t)n_pts;
    uint32_t kept = 0;
    for (size_t k = 0; k < n; k++) {
        if (!keep[k])
            continue;
        pts = grow(pts, &cap_pts, n_pts, sizeof(*pts));
        int32_t lon = (int32_t)lround(v[k][0] * 1e5),
                lat = (int32_t)lround(v[k][1] * 1e5);
        pts[n_pts++] = (struct pt){lon, lat};
        kept++;
        if (lat < s->lat_min)
            s->lat_min = lat;
        if (lat > s->lat_max)
            s->lat_max = lat;
        if (lon < s->lon_min)
            s->lon_min = lon;
        if (lon > s->lon_max)
            s->lon_max = lon;
    }
    rings[n_rings++].n = kept;
    s->n_rings++;
    free(v);
    free(keep);
}

static int read_seas(void)
{
    FILE *f = open_in("ne_10m_geography_marine_polys.geojson");
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)size);
    if (!text || fread(text, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "gen_geo: cannot read the seas\n");
        return -1;
    }
    fclose(f);
    char err[256];
    struct janas_json_doc *doc =
        janas_json_parse(text, (size_t)size, err, sizeof err);
    if (!doc) {
        fprintf(stderr, "gen_geo: the seas: %s\n", err);
        return -1;
    }
    const struct janas_json *feats =
        janas_json_get(janas_json_root(doc), "features");
    for (const struct janas_json *ft = feats ? feats->child : NULL; ft;
         ft = ft->next) {
        const char *name = janas_json_str(
            janas_json_get(janas_json_get(ft, "properties"), "name"));
        const struct janas_json *g = janas_json_get(ft, "geometry");
        const char *type = janas_json_str(janas_json_get(g, "type"));
        const struct janas_json *co = janas_json_get(g, "coordinates");
        if (!name || !*name || !type || !co)
            continue;
        seas = grow(seas, &cap_seas, n_seas, sizeof(*seas));
        struct sea *s = &seas[n_seas++];
        *s = (struct sea){.name = strdup(name),
                          .ring0 = (uint32_t)n_rings,
                          .lat_min = INT32_MAX,
                          .lat_max = INT32_MIN,
                          .lon_min = INT32_MAX,
                          .lon_max = INT32_MIN};
        if (strcmp(type, "Polygon") == 0) {
            for (const struct janas_json *r = co->child; r; r = r->next)
                add_ring(r, s);
        } else if (strcmp(type, "MultiPolygon") == 0) {
            for (const struct janas_json *p = co->child; p; p = p->next)
                for (const struct janas_json *r = p->child; r; r = r->next)
                    add_ring(r, s);
        }
        if (!s->n_rings)
            n_seas--;
    }
    janas_json_free(doc);
    free(text);
    return 0;
}

/* ---- output ---- */

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(
            stderr,
            "usage: gen_geo <data-directory> > src/services/common/geo_data.h\n");
        return 2;
    }
    snprintf(dir, sizeof dir, "%s", argv[1]);
    if (read_countries() || read_regions() || read_cities() ||
        read_airports() || read_seas())
        return 1;
    if (n_countries > 0xFFFF || n_regions >= 0xFFFF || n_tzs > 0xFFFF) {
        fprintf(stderr, "gen_geo: too many for 16 bits\n");
        return 1;
    }
    printf("/* SPDX-License-Identifier: GPL-3.0-or-later\n"
           "   Copyright (C) 2026 Maurizio Cammalleri */\n");
    printf("/* Generated by tools/gen_geo.c. Do not edit. From: OurAirports "
           "(public domain),\n   GeoNames (CC BY 4.0, "
           "https://www.geonames.org), Natural Earth (public\n   domain). "
           "%zu countries, %zu regions, %zu time zones, %zu cities, %zu\n"
           "   airports, %zu seas in %zu rings of %zu points. */\n",
           n_countries, n_regions, n_tzs, n_cities, n_airports, n_seas, n_rings,
           n_pts);
    printf("#ifndef JANAS_FLIGHTS_GEO_DATA_H\n#define JANAS_FLIGHTS_GEO_DATA_H"
           "\n\n#include \"geo.h\"\n\n");
    printf("static const struct geo_country geo_countries[] = {\n");
    for (size_t i = 0; i < n_countries; i++) {
        printf("    {\"%s\", ", countries[i].iso2);
        put_str(countries[i].name);
        printf("},\n");
    }
    printf("};\n\nstatic const struct geo_region geo_regions[] = {\n");
    for (size_t i = 0; i < n_regions; i++) {
        printf("    {%d, ", regions[i].country);
        put_str(regions[i].name);
        printf("},\n");
    }
    printf("};\n\nstatic const char *const geo_tzs[] = {\n");
    for (size_t i = 0; i < n_tzs; i++) {
        printf("    ");
        put_str(tzs[i]);
        printf(",\n");
    }
    printf("};\n\nstatic const struct geo_city geo_cities[] = {\n");
    for (size_t i = 0; i < n_cities; i++) {
        const struct city *y = &cities[i];
        printf("    {%ld, %ld, %ld, %d, %d, %d, ", lround(y->lat * 1e5),
               lround(y->lon * 1e5), y->pop, y->country,
               y->region < 0 ? 0xFFFF : y->region, y->tz);
        put_str(y->name);
        printf(", ");
        put_str(y->ascii);
        printf("},\n");
    }
    printf("};\n\nstatic const struct geo_airport geo_airports[] = {\n");
    for (size_t i = 0; i < n_airports; i++) {
        const struct airport *a = &airports[i];
        printf("    {%ld, %ld, %d, %d, \"%s\", \"%s\", ", lround(a->lat * 1e5),
               lround(a->lon * 1e5), a->country, a->tz, a->iata, a->icao);
        put_str(a->name);
        printf(", ");
        put_str(a->city);
        printf("},\n");
    }
    printf("};\n\nstatic const struct geo_sea geo_seas[] = {\n");
    for (size_t i = 0; i < n_seas; i++) {
        const struct sea *s = &seas[i];
        printf("    {");
        put_str(s->name);
        printf(", %u, %u, %d, %d, %d, %d},\n", s->ring0, s->n_rings, s->lat_min,
               s->lat_max, s->lon_min, s->lon_max);
    }
    printf("};\n\nstatic const struct geo_ring geo_rings[] = {\n");
    for (size_t i = 0; i < n_rings; i++)
        printf("    {%u, %u},\n", rings[i].pt0, rings[i].n);
    printf("};\n\nstatic const int32_t geo_points[][2] = {\n");
    for (size_t i = 0; i < n_pts; i++)
        printf("%s{%d,%d},%s", i % 6 == 0 ? "    " : "", pts[i].lon, pts[i].lat,
               i % 6 == 5 || i + 1 == n_pts ? "\n" : " ");
    printf("};\n\n#endif\n");
    return 0;
}
