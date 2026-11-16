/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * lib.c - the libraries an artifact may use (React, Recharts, D3, Chart.js,
 * Three.js, Plotly, Mermaid, KaTeX...), served by Brebus at /brebus/lib/
 * from a copy of its own: each file fetched once from jsdelivr's npm when
 * first asked, checked against the SHA-256 written below, and kept in
 * ~/.local/share/janas/brebus/lib. An artifact has no network of its own;
 * it gets these files, and only these: a path not in the list is refused.
 * They are not part of Janas, and are not in its repository: each comes
 * with its licence (MIT, BSD, ISC or Apache 2.0, docs/brebus.md).
 */
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/hf_fetch.h"
#include "common/sha256.h"

static const struct lib {
    const char *path, *sha256;
    size_t bytes;
} LIBS[] = {
    {"@babel/standalone@7.26.4/babel.min.js",
     "a12872ea8da3d29b2a296c51bfac7c482e81419c755f2207a49ad9b77200f4ea",
     2983904},
    {"chart.js@4.4.6/dist/chart.umd.js",
     "3850656abbdc319141e6e8ce8eacde2622fc767c30e20d81704af2bf3159f92d",
     205615},
    {"d3@7.9.0/dist/d3.min.js",
     "f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539",
     279706},
    {"@highlightjs/cdn-assets@11.10.0/highlight.min.js",
     "471ef9ae90c407af440fcdc48edfeeb562106b3267bd12d99071c162fb52ed32",
     124980},
    {"@highlightjs/cdn-assets@11.10.0/styles/github-dark.min.css",
     "9f208d022102b1d0c7aebfecd8e42ca7997d5de636649d2b31ea63093d809019", 1315},
    {"@highlightjs/cdn-assets@11.10.0/styles/github.min.css",
     "3a9a5def8b9c311e5ae43abde85c63133185eed4f0d9f67fea4b00a8308cf066", 1309},
    {"katex@0.16.11/dist/contrib/auto-render.min.js",
     "7b57d427ac6270677daf8d8380ded2cc73336f9149a167b8e1fe0d6ef66604ae", 3481},
    {"katex@0.16.11/dist/fonts/KaTeX_AMS-Regular.woff2",
     "0cdd387c9590a1a9f9794560022dbb59654a7d86f187aa0c81495ad42d3a7308", 28076},
    {"katex@0.16.11/dist/fonts/KaTeX_Caligraphic-Bold.woff2",
     "de7701e42cf1f4cf0b766c03fb27977207eee2f4fd5d76fa82188406da43ea4c", 6912},
    {"katex@0.16.11/dist/fonts/KaTeX_Caligraphic-Regular.woff2",
     "5d53e70ad607c2352162dec9e0923fb54ecdafaccbf604cd8dcf7d00facb989b", 6908},
    {"katex@0.16.11/dist/fonts/KaTeX_Fraktur-Bold.woff2",
     "74444efd593c005e3f4573b44524704c0af0a937fe911cca9e94068d0d140d3f", 11348},
    {"katex@0.16.11/dist/fonts/KaTeX_Fraktur-Regular.woff2",
     "51814d270d06ff0255dba0799994fa4d8c84d11f09951d47595f4abb1f3602dc", 11316},
    {"katex@0.16.11/dist/fonts/KaTeX_Main-BoldItalic.woff2",
     "99cd42a3c072d918f2f44984a807cf7aa16e13545fd0875fc07c6c65f99e715b", 16780},
    {"katex@0.16.11/dist/fonts/KaTeX_Main-Bold.woff2",
     "0f60d1b897938ec918c8ce073092411baf9438f6739465693ff18b0f9d20b021", 25324},
    {"katex@0.16.11/dist/fonts/KaTeX_Main-Italic.woff2",
     "97479ca6cce906abc961ecac96faa5f9ca2e61b8e7670d475826bcdee9a7c267", 16988},
    {"katex@0.16.11/dist/fonts/KaTeX_Main-Regular.woff2",
     "c2342cd8b869e01752a9321dc17213fc40d4d04c79688c1d43f2cf316abd7866", 26272},
    {"katex@0.16.11/dist/fonts/KaTeX_Math-BoldItalic.woff2",
     "dc47344dbb6cb5b655c8460d561f4df5f501b90c804ad3c6cec65fe322351ab1", 16400},
    {"katex@0.16.11/dist/fonts/KaTeX_Math-Italic.woff2",
     "7af58c5ec8f132a2ddde9027c6d7814decce4d3b822a11192a42a20e2e973264", 16440},
    {"katex@0.16.11/dist/fonts/KaTeX_SansSerif-Bold.woff2",
     "e99ae51144bf1232efcc1bfe5add36262c6866b0faab24fa75740e1b98577a62", 12216},
    {"katex@0.16.11/dist/fonts/KaTeX_SansSerif-Italic.woff2",
     "00b26ac825e2095056396e0553b8ac26d3f8ad158c3826e28b4c45b385c4714a", 12028},
    {"katex@0.16.11/dist/fonts/KaTeX_SansSerif-Regular.woff2",
     "68e8c73ef42afd3ccec58bf0fba302cce448938e7fc020a5e31f8a952eee1342", 10344},
    {"katex@0.16.11/dist/fonts/KaTeX_Script-Regular.woff2",
     "036d4e95149b69ff9bcc0cd55771efeb25ffa3947293e69acd78d5ac328c684b", 9644},
    {"katex@0.16.11/dist/fonts/KaTeX_Size1-Regular.woff2",
     "6b47c40166b6dbe21a5dfca7718413f2147fd2399be1ba605d8ad39cedf25dfe", 5468},
    {"katex@0.16.11/dist/fonts/KaTeX_Size2-Regular.woff2",
     "d04c54219f9eaec6d4d4fd42dfb28785975a4794d6b2fc71e566b9cd6db842dd", 5208},
    {"katex@0.16.11/dist/fonts/KaTeX_Size3-Regular.woff2",
     "73d591271b1604960cb10bb90fee021670af7297017e0e98480b332d11f51995", 3624},
    {"katex@0.16.11/dist/fonts/KaTeX_Size4-Regular.woff2",
     "a4af7d414440a1c1790825cfb700cf9cf43b0f2c4b04f0ebc523011ad9853ec0", 4928},
    {"katex@0.16.11/dist/fonts/KaTeX_Typewriter-Regular.woff2",
     "71d517d67827787cfabdf186914cc3358eda539e37931941f2b2fd4a21f68c0b", 13568},
    {"katex@0.16.11/dist/katex.min.css",
     "717bc9ae7853b61f0f76455dddf0ecd4f527a783f42de2ac24684899c1c46258", 23335},
    {"katex@0.16.11/dist/katex.min.js",
     "e6bfe5deebd4c7ccd272055bab63bd3ab2c73b907b6e6a22d352740a81381fd4",
     275414},
    {"lodash@4.17.21/lodash.min.js",
     "a9705dfc47c0763380d851ab1801be6f76019f6b67e40e9b873f8b4a0603f7a9", 73015},
    {"lucide-react@0.460.0/dist/umd/lucide-react.min.js",
     "aa5a38ea2b8ccd1a4cbdf414210c5c6ac0682bca128894b90e530f3bd64c8304",
     512021},
    {"mammoth@1.8.0/mammoth.browser.min.js",
     "deb07bf230d1cb3e190bc5adc6743f35c6531b6571d1e5469b24f452a7f0f4ab",
     642194},
    {"mathjs@13.2.2/lib/browser/math.js",
     "5e6c9e880e91f029b6665148c4d76a73488bd0e95ef0653537cefab826cf6646",
     676922},
    {"mermaid@11.4.1/dist/mermaid.min.js",
     "a43bc1afd446f9c4cc66ac5dd45d02e8d65e26fc5344ec0ef787f88d6ddb6f9e",
     2571900},
    {"papaparse@5.4.1/papaparse.min.js",
     "b8e870c5d2b29772f10c9fa9a693c8b896aac8540ed6701e3cc6304c683febdb", 19469},
    {"plotly.js-dist-min@2.35.2/plotly.min.js",
     "6d21266ce1bd7d9e5ab4e115989c70c20de0382fd973a8f26ab58619eba4d603",
     4558696},
    {"prop-types@15.8.1/prop-types.min.js",
     "e653471aba824786aee5dce1bcb5a86ed30c8518d346d2ace0460a5633a9cbdb", 1722},
    {"react@18.3.1/umd/react.production.min.js",
     "d949f1c3687aedadcedac85261865f29b17cd273997e7f6b2bfc53b2f9d4c4dd", 10751},
    {"react-dom@18.3.1/umd/react-dom.production.min.js",
     "35f4f974f4b2bcd44da73963347f8952e341f83909e4498227d4e26b98f66f0d",
     131835},
    {"recharts@2.13.3/umd/Recharts.js",
     "1fdc7e37a2e57b796931bfb08ac49b436b68bb2a4aa9bf296117789231da2d54",
     514753},
    {"@tailwindcss/browser@4.1.10/dist/index.global.js",
     "12107ff9df38c723a8c961b0a9aa2bb6b416a2f47b15e0743680ee03f80e40f8",
     259295},
    {"three@0.128.0/build/three.min.js",
     "9274bbcec8d96168626c732b5d31c775aa8cfb7eaa0599bec0c175908a2c1ce2",
     603445},
    {"tone@15.0.4/build/Tone.js",
     "33c3fdc2f26f66f203631deb2c51a5ade54cdffb8271d2d98106c295de89bc0d",
     345744},
    {"xlsx@0.18.5/dist/xlsx.full.min.js",
     "c9506197caf809a075b6dee1da0d36fb19da7158ffe8a88e7b0c96c5d8623c99",
     881727},
};

static pthread_mutex_t fetch_mu = PTHREAD_MUTEX_INITIALIZER;

static int slurp(const char *path, struct janas_buf *out)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    char tmp[65536];
    size_t n;
    while ((n = fread(tmp, 1, sizeof tmp, f)) > 0)
        janas_buf_put(out, tmp, n);
    int bad = ferror(f) || out->oom;
    fclose(f);
    return bad ? -1 : 0;
}

static int checked(const struct lib *l, const struct janas_buf *b)
{
    struct janas_sha256 s;
    char hex[65];
    janas_sha256_init(&s);
    janas_sha256_update(&s, b->p, b->n);
    janas_sha256_hex(&s, hex);
    return b->n == l->bytes && !strcmp(hex, l->sha256);
}

int brebus_lib(struct brebus *b, const char *rel, struct janas_buf *out,
               const char **mime, char *err, size_t en)
{
    const struct lib *l = NULL;
    for (size_t i = 0; rel && i < sizeof LIBS / sizeof *LIBS; i++)
        if (!strcmp(rel, LIBS[i].path))
            l = &LIBS[i];
    if (!l) {
        snprintf(err, en, "not a library Brebus serves");
        return -1;
    }
    const char *dot = strrchr(l->path, '.');
    *mime = !strcmp(dot, ".css")     ? "text/css; charset=utf-8"
            : !strcmp(dot, ".woff2") ? "font/woff2"
                                     : "text/javascript; charset=utf-8";
    char path[4800], dir[4800];
    snprintf(path, sizeof path, "%s/lib/%s", b->data_dir, l->path);
    pthread_mutex_lock(&fetch_mu);
    out->n = 0;
    if (slurp(path, out) == 0 && checked(l, out)) {
        pthread_mutex_unlock(&fetch_mu);
        return 0;
    }
    /* not here yet, or not the file it should be: fetched again */
    out->n = 0;
    char url[600];
    snprintf(url, sizeof url, "https://cdn.jsdelivr.net/npm/%s", l->path);
    int rc = janas_hf_get_url(url, NULL, NULL, out, NULL, 0, err, en);
    if (rc == 0 && !checked(l, out)) {
        snprintf(err, en, "%s came different from the file expected", l->path);
        rc = -1;
    }
    if (rc == 0) {
        /* kept for next time, written whole under another name first */
        snprintf(dir, sizeof dir, "%s", path);
        *strrchr(dir, '/') = 0;
        char tmp[4900];
        snprintf(tmp, sizeof tmp, "%s.part", path);
        FILE *f = janas_mkdirs(dir) == 0 ? fopen(tmp, "wb") : NULL;
        int bad = !f || fwrite(out->p, 1, out->n, f) != out->n;
        if (f)
            bad |= fclose(f) != 0;
        if (bad || rename(tmp, path) != 0)
            remove(tmp); /* served all the same, fetched again next time */
    }
    pthread_mutex_unlock(&fetch_mu);
    return rc;
}
