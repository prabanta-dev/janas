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
    const char *terms;   /* of the running text */
    const char *dev;     /* the same, of git's and GitHub's layouts */
} G[] = {
    {"it",
     {"calma", "bava di vento", "brezza leggera", "brezza tesa",
      "vento moderato", "vento teso", "vento fresco", "vento forte", "burrasca",
      "burrasca forte", "tempesta", "tempesta violenta", "uragano"},
     {"calmo", "quasi calmo", "poco mosso", "mosso", "molto mosso", "agitato",
      "molto agitato", "grosso", "molto grosso", "tempestoso"},
     "swell = mare lungo; wind waves = mare di vento; significant height = "
     "altezza significativa; gusts = raffiche",
     "commit = commit; push = push; pull = pull; staged = nell'area di "
     "stage; working tree = cartella di lavoro; branch = ramo; repository = "
     "repository; issue = issue; pull request = pull request; release = "
     "release; tag = tag"},
    {"fr",
     {"calme", "très légère brise", "légère brise", "petite brise",
      "jolie brise", "bonne brise", "vent frais", "grand frais", "coup de vent",
      "fort coup de vent", "tempête", "violente tempête", "ouragan"},
     {"calme", "calme, ridée", "belle", "peu agitée", "agitée", "forte",
      "très forte", "grosse", "très grosse", "énorme"},
     "swell = houle; wind waves = mer du vent; significant height = "
     "hauteur significative; gusts = rafales",
     "commit = commit; push = push; pull = pull; staged = indexé; working "
     "tree = répertoire de travail; branch = branche; repository = dépôt; "
     "pull request = pull request; release = version; tag = étiquette"},
    {"es",
     {"calma", "ventolina", "flojito", "flojo", "bonancible", "fresquito",
      "fresco", "frescachón", "temporal", "temporal fuerte", "temporal duro",
      "temporal muy duro", "temporal huracanado"},
     {"calma", "rizada", "marejadilla", "marejada", "fuerte marejada", "gruesa",
      "muy gruesa", "arbolada", "montañosa", "enorme"},
     "swell = mar de fondo; wind waves = mar de viento; significant height "
     "= altura significativa; gusts = rachas",
     "commit = commit; push = push; pull = pull; staged = preparado; "
     "working tree = directorio de trabajo; branch = rama; repository = "
     "repositorio; pull request = pull request; release = versión"},
    {"de",
     {"Windstille", "leiser Zug", "leichte Brise", "schwache Brise",
      "mäßige Brise", "frische Brise", "starker Wind", "steifer Wind",
      "stürmischer Wind", "Sturm", "schwerer Sturm", "orkanartiger Sturm",
      "Orkan"},
     {"spiegelglatte See", "ruhige, gekräuselte See", "schwach bewegte See",
      "leicht bewegte See", "mäßig bewegte See", "grobe See", "sehr grobe See",
      "hohe See", "sehr hohe See", "außergewöhnlich schwere See"},
     "swell = Dünung; wind waves = Windsee; significant height = "
     "signifikante Wellenhöhe; gusts = Böen",
     "commit = Commit; push = Push; pull = Pull; staged = in der "
     "Staging-Area; working tree = Arbeitsverzeichnis; branch = Branch; "
     "repository = Repository; pull request = Pull-Request; release = "
     "Release"},
};

/* The state of one issue or pull request (github/layouts.c): a model
   asked wrote the plural ("aperte") for one. Feminine, as "issue" and
   "pull request" are in these languages. */
static const struct {
    const char *lang, *open, *closed, *merged, *draft;
} STATUS[] = {
    {"it", "aperta", "chiusa", "unita", "bozza"},
    {"fr", "ouverte", "fermée", "fusionnée", "brouillon"},
    {"es", "abierta", "cerrada", "fusionada", "borrador"},
    {"de", "offen", "geschlossen", "zusammengeführt", "Entwurf"},
};

static const char *status_word(const char *lang, const char *key, size_t n)
{
    for (size_t i = 0; lang && i < sizeof STATUS / sizeof *STATUS; i++) {
        if (strcmp(STATUS[i].lang, lang) != 0)
            continue;
        const char *k[4] = {"status.open", "status.closed", "status.merged",
                            "status.draft"};
        const char *v[4] = {STATUS[i].open, STATUS[i].closed, STATUS[i].merged,
                            STATUS[i].draft};
        for (int j = 0; j < 4; j++)
            if (strlen(k[j]) == n && strncmp(key, k[j], n) == 0)
                return v[j];
    }
    return NULL;
}

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
    return status_word(lang, key, n);
}

/* git_status, github_issues: the layouts of git and GitHub */
static int dev_layout(const char *name)
{
    return name && strncmp(name, "git", 3) == 0;
}

const char *glossary_terms(const char *lang, const char *name)
{
    int g = find(lang);
    return g < 0 ? NULL : dev_layout(name) ? G[g].dev : G[g].terms;
}

static uint64_t fnv(uint64_t h, const char *s)
{
    for (; *s; s++)
        h = (h ^ (unsigned char)*s) * 1099511628211ull;
    return (h ^ '\n') * 1099511628211ull;
}

uint64_t glossary_mark(const char *lang, const char *name, uint64_t h)
{
    int g = find(lang);
    if (g < 0)
        return h;
    for (int i = 0; i < 13; i++)
        h = fnv(h, G[g].bft[i]);
    for (int i = 0; i < 10; i++)
        h = fnv(h, G[g].sea[i]);
    if (dev_layout(name)) {
        for (size_t i = 0; i < sizeof STATUS / sizeof *STATUS; i++)
            if (strcmp(STATUS[i].lang, lang) == 0)
                h = fnv(fnv(fnv(fnv(h, STATUS[i].open), STATUS[i].closed),
                            STATUS[i].merged),
                        STATUS[i].draft);
        return fnv(h, G[g].dev);
    }
    return fnv(h, G[g].terms);
}
