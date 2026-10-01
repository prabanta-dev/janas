/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * glossary.c - the terms of art of the layouts (see glossary.h). The
 * scales' names are those of the national services' tables: Aeronautica
 * Militare, Météo-France, AEMET, Deutscher Wetterdienst. The sea's state
 * is the Douglas degree (weather/words.c), 1 to 9; 0, the glassy sea, is
 * there for the table to be whole.
 */
#include "glossary.h"

#include <string.h>

static const struct {
    const char *lang;
    const char *bft[13]; /* Beaufort 0-12 */
    const char *sea[10]; /* Douglas 0-9 */
    const char *terms;
} G[] = {
    {"it",
     {"calma", "bava di vento", "brezza leggera", "brezza tesa",
      "vento moderato", "vento teso", "vento fresco", "vento forte", "burrasca",
      "burrasca forte", "tempesta", "tempesta violenta", "uragano"},
     {"calmo", "quasi calmo", "poco mosso", "mosso", "molto mosso", "agitato",
      "molto agitato", "grosso", "molto grosso", "tempestoso"},
     "swell = mare lungo; wind waves = mare di vento; significant height = "
     "altezza significativa; gusts = raffiche"},
    {"fr",
     {"calme", "très légère brise", "légère brise", "petite brise",
      "jolie brise", "bonne brise", "vent frais", "grand frais", "coup de vent",
      "fort coup de vent", "tempête", "violente tempête", "ouragan"},
     {"calme", "calme, ridée", "belle", "peu agitée", "agitée", "forte",
      "très forte", "grosse", "très grosse", "énorme"},
     "swell = houle; wind waves = mer du vent; significant height = "
     "hauteur significative; gusts = rafales"},
    {"es",
     {"calma", "ventolina", "flojito", "flojo", "bonancible", "fresquito",
      "fresco", "frescachón", "temporal", "temporal fuerte", "temporal duro",
      "temporal muy duro", "temporal huracanado"},
     {"calma", "rizada", "marejadilla", "marejada", "fuerte marejada", "gruesa",
      "muy gruesa", "arbolada", "montañosa", "enorme"},
     "swell = mar de fondo; wind waves = mar de viento; significant height "
     "= altura significativa; gusts = rachas"},
    {"de",
     {"Windstille", "leiser Zug", "leichte Brise", "schwache Brise",
      "mäßige Brise", "frische Brise", "starker Wind", "steifer Wind",
      "stürmischer Wind", "Sturm", "schwerer Sturm", "orkanartiger Sturm",
      "Orkan"},
     {"spiegelglatte See", "ruhige, gekräuselte See", "schwach bewegte See",
      "leicht bewegte See", "mäßig bewegte See", "grobe See", "sehr grobe See",
      "hohe See", "sehr hohe See", "außergewöhnlich schwere See"},
     "swell = Dünung; wind waves = Windsee; significant height = "
     "signifikante Wellenhöhe; gusts = Böen"},
};

static int find(const char *lang)
{
    for (size_t i = 0; lang && i < sizeof G / sizeof *G; i++)
        if (strcmp(G[i].lang, lang) == 0)
            return (int)i;
    return -1;
}

/* The degree after prefix in key, or -1. */
static int degree(const char *key, size_t n, const char *prefix, int top)
{
    size_t pn = strlen(prefix);
    if (n <= pn || n > pn + 2 || strncmp(key, prefix, pn) != 0)
        return -1;
    int d = 0;
    for (size_t i = pn; i < n; i++) {
        if (key[i] < '0' || key[i] > '9')
            return -1;
        d = d * 10 + (key[i] - '0');
    }
    return d <= top ? d : -1;
}

const char *glossary_word(const char *lang, const char *key, size_t n)
{
    int g = find(lang), d;
    if (g < 0)
        return NULL;
    if ((d = degree(key, n, "bft.", 12)) >= 0)
        return G[g].bft[d];
    if ((d = degree(key, n, "sea.", 9)) >= 0)
        return G[g].sea[d];
    return NULL;
}

const char *glossary_terms(const char *lang)
{
    int g = find(lang);
    return g < 0 ? NULL : G[g].terms;
}

static uint64_t fnv(uint64_t h, const char *s)
{
    for (; *s; s++)
        h = (h ^ (unsigned char)*s) * 1099511628211ull;
    return (h ^ '\n') * 1099511628211ull;
}

uint64_t glossary_mark(const char *lang, uint64_t h)
{
    int g = find(lang);
    if (g < 0)
        return h;
    for (int i = 0; i < 13; i++)
        h = fnv(h, G[g].bft[i]);
    for (int i = 0; i < 10; i++)
        h = fnv(h, G[g].sea[i]);
    return fnv(h, G[g].terms);
}
