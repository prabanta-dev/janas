/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_docs.c - janas-docs (src/services/docs/): the text of a docx and an
 * odt made by Python's zipfile, every tag the extractor knows among them;
 * the archive and the XML cut at every byte and mutated, the index file
 * too, the answers of an embeddings server as well - nothing may crash or
 * read past its input (the asan build sees to the reading). Then a home
 * made in ~/tmp: its Documents folder indexed by words, searched, read,
 * a document changed and one removed, the private and hidden files and a
 * link out never indexed; and a small embeddings server on 127.0.0.1 of
 * the test's own, whose vectors make the search by meaning, and whose
 * model, changed, makes the vectors again.
 */
#include <arpa/inet.h>
#include <fcntl.h>
#include <ftw.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include "services/docs/chunks.c"
#include "services/docs/embed.c"
#include "services/docs/extract.c"
#include "services/docs/index.c"
#include "services/docs/roots.c"
#include "services/docs/search.c"
#include "services/docs/store.c"
#include "services/docs/tools.c"
#include "services/docs/xml.c"
#include "services/docs/zip.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* made with Python's zipfile: [Content_Types].xml stored and
   word/document.xml deflated (level 9); mimetype stored and content.xml
   deflated */
static const unsigned char DOCX[593] = {
    0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5a, 0x27,
    0x6e, 0x5d, 0xee, 0x47, 0x58, 0x66, 0x1d, 0x00, 0x00, 0x00, 0x1d, 0x00,
    0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x5b, 0x43, 0x6f, 0x6e, 0x74, 0x65,
    0x6e, 0x74, 0x5f, 0x54, 0x79, 0x70, 0x65, 0x73, 0x5d, 0x2e, 0x78, 0x6d,
    0x6c, 0x3c, 0x3f, 0x78, 0x6d, 0x6c, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69,
    0x6f, 0x6e, 0x3d, 0x22, 0x31, 0x2e, 0x30, 0x22, 0x3f, 0x3e, 0x3c, 0x54,
    0x79, 0x70, 0x65, 0x73, 0x2f, 0x3e, 0x50, 0x4b, 0x03, 0x04, 0x14, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x5a, 0x27, 0x6e, 0x5d, 0xea, 0x34, 0x4b, 0xbf,
    0x3e, 0x01, 0x00, 0x00, 0x37, 0x02, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
    0x77, 0x6f, 0x72, 0x64, 0x2f, 0x64, 0x6f, 0x63, 0x75, 0x6d, 0x65, 0x6e,
    0x74, 0x2e, 0x78, 0x6d, 0x6c, 0x75, 0x51, 0xdb, 0x6a, 0x02, 0x31, 0x10,
    0x7d, 0xf7, 0x2b, 0x42, 0x0a, 0xbe, 0xd5, 0xac, 0x4a, 0xa5, 0xac, 0x6e,
    0x44, 0x4a, 0x6f, 0x6f, 0x82, 0xf6, 0x03, 0xc6, 0x64, 0xaa, 0x81, 0xdc,
    0x48, 0xa2, 0xab, 0x7e, 0x51, 0xff, 0xa3, 0x3f, 0xd6, 0xec, 0x8a, 0x8a,
    0x60, 0x5f, 0x32, 0xc9, 0xcc, 0x99, 0x73, 0xe6, 0x64, 0x26, 0xd3, 0xbd,
    0xd1, 0x64, 0x87, 0x21, 0x2a, 0x67, 0x2b, 0xda, 0xef, 0x15, 0x94, 0xa0,
    0x15, 0x4e, 0x2a, 0xbb, 0xae, 0xe8, 0xd7, 0xf2, 0xed, 0xf1, 0x99, 0x92,
    0x98, 0xc0, 0x4a, 0xd0, 0xce, 0x62, 0x45, 0x0f, 0x18, 0xe9, 0x94, 0x77,
    0x26, 0x75, 0x29, 0x9d, 0xd8, 0x1a, 0xb4, 0x89, 0x64, 0x06, 0x1b, 0xcb,
    0xba, 0xa2, 0x9b, 0x94, 0x7c, 0xc9, 0x58, 0x14, 0x1b, 0x34, 0x10, 0x7b,
    0xce, 0xa3, 0xcd, 0xb5, 0x6f, 0x17, 0x0c, 0xa4, 0xfc, 0x0c, 0x6b, 0x56,
    0xbb, 0x20, 0x7d, 0x70, 0x02, 0x63, 0xcc, 0x02, 0x46, 0xb3, 0x41, 0x51,
    0x8c, 0x98, 0x01, 0x65, 0x29, 0xcf, 0x94, 0x2b, 0x27, 0x0f, 0x4d, 0xf4,
    0xed, 0x31, 0x0f, 0x6d, 0x58, 0xa4, 0x83, 0x46, 0x52, 0x97, 0x3b, 0xd0,
    0x15, 0xfd, 0x40, 0x68, 0x46, 0xeb, 0x53, 0xc6, 0x27, 0xec, 0x82, 0x69,
    0x8f, 0xc4, 0x5f, 0x9c, 0x4d, 0x01, 0x52, 0x72, 0x44, 0x2a, 0xa2, 0x9d,
    0x80, 0x63, 0x36, 0x85, 0x0d, 0x2e, 0xb5, 0xe8, 0x70, 0xea, 0xb9, 0x28,
    0x9c, 0xda, 0x9a, 0xf9, 0xcb, 0xe8, 0x41, 0x64, 0x77, 0x3e, 0x60, 0xc4,
    0xb0, 0x43, 0xca, 0x3f, 0x35, 0x11, 0x60, 0x73, 0x3b, 0xc9, 0x1e, 0xa3,
    0xca, 0x13, 0xfc, 0xfe, 0x34, 0xb4, 0x37, 0x6c, 0x17, 0xe5, 0xe7, 0xa7,
    0x82, 0x74, 0xc1, 0xf8, 0x31, 0xc1, 0x6d, 0x70, 0x77, 0x31, 0xb0, 0x62,
    0x27, 0x6c, 0x14, 0x20, 0xd1, 0x1e, 0xe1, 0x1e, 0x4a, 0xa2, 0x5e, 0xe2,
    0x3e, 0xf1, 0x2c, 0x2d, 0x50, 0x6b, 0x48, 0x2d, 0xd7, 0x39, 0x7b, 0x8b,
    0x55, 0x36, 0xa6, 0xd0, 0xe6, 0xe7, 0xb3, 0xf7, 0xd7, 0xa6, 0x76, 0xcd,
    0xfc, 0xef, 0x96, 0x2f, 0x50, 0xb8, 0xbc, 0x4d, 0x12, 0xd4, 0x1a, 0x48,
    0xf7, 0x61, 0x3f, 0x18, 0x0d, 0x67, 0xe3, 0x7c, 0x19, 0x0c, 0x87, 0xe3,
    0x3b, 0x5f, 0xc5, 0xce, 0x5b, 0x61, 0xd7, 0x8d, 0xf3, 0xce, 0x1f, 0x50,
    0x4b, 0x01, 0x02, 0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5a,
    0x27, 0x6e, 0x5d, 0xee, 0x47, 0x58, 0x66, 0x1d, 0x00, 0x00, 0x00, 0x1d,
    0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x5b, 0x43, 0x6f,
    0x6e, 0x74, 0x65, 0x6e, 0x74, 0x5f, 0x54, 0x79, 0x70, 0x65, 0x73, 0x5d,
    0x2e, 0x78, 0x6d, 0x6c, 0x50, 0x4b, 0x01, 0x02, 0x14, 0x03, 0x14, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x5a, 0x27, 0x6e, 0x5d, 0xea, 0x34, 0x4b, 0xbf,
    0x3e, 0x01, 0x00, 0x00, 0x37, 0x02, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x4e, 0x00,
    0x00, 0x00, 0x77, 0x6f, 0x72, 0x64, 0x2f, 0x64, 0x6f, 0x63, 0x75, 0x6d,
    0x65, 0x6e, 0x74, 0x2e, 0x78, 0x6d, 0x6c, 0x50, 0x4b, 0x05, 0x06, 0x00,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x02, 0x00, 0x80, 0x00, 0x00, 0x00, 0xbb,
    0x01, 0x00, 0x00, 0x00, 0x00,
};

static const unsigned char ODT[531] = {
    0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5a, 0x27,
    0x6e, 0x5d, 0x5e, 0xc6, 0x32, 0x0c, 0x27, 0x00, 0x00, 0x00, 0x27, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x6d, 0x69, 0x6d, 0x65, 0x74, 0x79,
    0x70, 0x65, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f,
    0x6e, 0x2f, 0x76, 0x6e, 0x64, 0x2e, 0x6f, 0x61, 0x73, 0x69, 0x73, 0x2e,
    0x6f, 0x70, 0x65, 0x6e, 0x64, 0x6f, 0x63, 0x75, 0x6d, 0x65, 0x6e, 0x74,
    0x2e, 0x74, 0x65, 0x78, 0x74, 0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00,
    0x00, 0x08, 0x00, 0x5a, 0x27, 0x6e, 0x5d, 0xb6, 0x2c, 0xa9, 0x38, 0x18,
    0x01, 0x00, 0x00, 0x58, 0x02, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x63,
    0x6f, 0x6e, 0x74, 0x65, 0x6e, 0x74, 0x2e, 0x78, 0x6d, 0x6c, 0x8d, 0x92,
    0xc1, 0x6e, 0xc3, 0x20, 0x0c, 0x86, 0x5f, 0x05, 0x71, 0xe8, 0x2d, 0xa3,
    0xdb, 0x34, 0x69, 0xa2, 0x84, 0xde, 0xfa, 0x00, 0xd3, 0xf6, 0x00, 0x84,
    0xb8, 0x1d, 0x2a, 0xb1, 0xbb, 0xe0, 0x54, 0xed, 0xdb, 0x2f, 0x85, 0xac,
    0xed, 0x34, 0x45, 0xda, 0x05, 0x0b, 0xfc, 0xfd, 0xfe, 0x6d, 0x0b, 0xb3,
    0x3e, 0x75, 0x51, 0x1c, 0xa1, 0x4f, 0x81, 0xb0, 0x96, 0x8f, 0x0f, 0x4b,
    0x29, 0x00, 0x3d, 0xb5, 0x01, 0x77, 0xb5, 0xfc, 0x78, 0xdf, 0x54, 0xaf,
    0x72, 0x6d, 0x0d, 0x6d, 0xb7, 0xc1, 0x83, 0x6e, 0xc9, 0x0f, 0x1d, 0x20,
    0x57, 0x9e, 0x90, 0xc7, 0x28, 0x46, 0x31, 0x26, 0x5d, 0xb2, 0xb5, 0x1c,
    0x7a, 0xd4, 0xe4, 0x52, 0x48, 0x1a, 0x5d, 0x07, 0x49, 0xb3, 0xd7, 0x74,
    0x00, 0xfc, 0x51, 0xe9, 0x7b, 0x5a, 0x67, 0xab, 0xf2, 0xc2, 0x70, 0xe2,
    0xff, 0xaa, 0x2f, 0x6c, 0xd6, 0x5e, 0x9b, 0x72, 0x03, 0x53, 0xe7, 0x38,
    0xf8, 0x2a, 0xf1, 0x39, 0x42, 0xb2, 0x26, 0x47, 0x9d, 0x4f, 0x9b, 0x38,
    0x44, 0x10, 0xe8, 0x92, 0xa7, 0xc4, 0x64, 0xd4, 0x7d, 0xce, 0xa8, 0xd9,
    0x12, 0x53, 0xa2, 0xa1, 0xf6, 0x7c, 0xbd, 0x5c, 0xac, 0xad, 0xc9, 0x0d,
    0x7c, 0x8a, 0x1c, 0x68, 0xe0, 0x18, 0x10, 0xaa, 0x08, 0x47, 0x88, 0xb5,
    0x7c, 0x92, 0xf6, 0x6d, 0x04, 0x99, 0x9d, 0x51, 0x05, 0x9b, 0xf0, 0x83,
    0xdd, 0xb8, 0x3e, 0xa0, 0x2b, 0xb7, 0x54, 0xc4, 0xbe, 0x96, 0xcf, 0x52,
    0xd9, 0x97, 0xe5, 0x52, 0xec, 0x4a, 0x82, 0x5d, 0xa3, 0xac, 0xf3, 0x5f,
    0xc3, 0x04, 0xe6, 0xda, 0x4d, 0x0f, 0x6e, 0xaf, 0x6c, 0x0c, 0x70, 0x0c,
    0x4c, 0xa2, 0x73, 0x6d, 0x0f, 0x53, 0xf9, 0xc3, 0x6d, 0x07, 0x88, 0xc4,
    0xe3, 0x00, 0x84, 0x57, 0xc7, 0x01, 0x85, 0xa7, 0xee, 0xb2, 0x39, 0xba,
    0xe1, 0x6a, 0x9e, 0xdf, 0x8c, 0x66, 0x62, 0x11, 0x79, 0x45, 0xfb, 0xc5,
    0x8e, 0x57, 0x7f, 0x35, 0x65, 0x7c, 0xf5, 0x6b, 0x33, 0x6a, 0xe6, 0x63,
    0xd8, 0x6f, 0x50, 0x4b, 0x01, 0x02, 0x14, 0x03, 0x14, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x5a, 0x27, 0x6e, 0x5d, 0x5e, 0xc6, 0x32, 0x0c, 0x27, 0x00,
    0x00, 0x00, 0x27, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x6d, 0x69, 0x6d, 0x65, 0x74, 0x79, 0x70, 0x65, 0x50, 0x4b, 0x01, 0x02,
    0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x5a, 0x27, 0x6e, 0x5d,
    0xb6, 0x2c, 0xa9, 0x38, 0x18, 0x01, 0x00, 0x00, 0x58, 0x02, 0x00, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x80, 0x01, 0x4d, 0x00, 0x00, 0x00, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x6e,
    0x74, 0x2e, 0x78, 0x6d, 0x6c, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00,
    0x00, 0x02, 0x00, 0x02, 0x00, 0x6f, 0x00, 0x00, 0x00, 0x8e, 0x01, 0x00,
    0x00, 0x00, 0x00,
};

static const char DOCX_TEXT[] =
    "# Contratto di locazione\nIl canone mensile \xc3\xa8 di 850 & euro\t"
    "scadenza\nSeconda riga \xe2\x98\xba \xc3\xa9";
static const char ODT_TEXT[] =
    "## Ricetta\nFarina   500 g\tacqua\nlievito madre\nFine <ok>";

static void documents(void)
{
    struct janas_buf t = {0};
    char why[256];
    CHECK(dz_extract_mem(DOCX, sizeof DOCX, DZ_DOCX, &t, why, sizeof why) ==
                  0 &&
              t.n == strlen(DOCX_TEXT) && !memcmp(t.p, DOCX_TEXT, t.n),
          "docx: [%.*s]", (int)t.n, t.p ? t.p : "");
    t.n = 0;
    CHECK(dz_extract_mem(ODT, sizeof ODT, DZ_ODT, &t, why, sizeof why) == 0 &&
              t.n == strlen(ODT_TEXT) && !memcmp(t.p, ODT_TEXT, t.n),
          "odt: [%.*s]", (int)t.n, t.p ? t.p : "");
    uint8_t *x;
    size_t xn;
    CHECK(dz_zip_get(DOCX, sizeof DOCX, "content.xml", &x, &xn, why,
                     sizeof why) == 1,
          "a name not there: %s", why);
    CHECK(dz_kind("/a/b.DOCX") == DZ_DOCX && dz_kind("/a/b.md") == DZ_TEXT &&
              dz_kind("/a/.md") == DZ_NONE && dz_kind("/a.b/c") == DZ_NONE,
          "kinds");

    /* cut at every byte, then mutated: the archives and the XML */
    const unsigned char *arch[2] = {DOCX, ODT};
    size_t an[2] = {sizeof DOCX, sizeof ODT};
    int kinds[2] = {DZ_DOCX, DZ_ODT};
    unsigned char *m = malloc(4096);
    srand(4242);
    int read_ok = 0;
    for (int a = 0; a < 2 && m; a++) {
        for (size_t k = 0; k <= an[a]; k++) {
            memcpy(m, arch[a], k);
            t.n = 0;
            dz_extract_mem(m, k, kinds[a], &t, why, sizeof why);
        }
        for (int r = 0; r < 20000; r++) {
            memcpy(m, arch[a], an[a]);
            int nm = 1 + rand() % 6;
            for (int i = 0; i < nm; i++)
                m[rand() % an[a]] = (unsigned char)rand();
            t.n = 0;
            read_ok +=
                dz_extract_mem(m, an[a], kinds[a], &t, why, sizeof why) == 0;
        }
    }
    free(m);
    /* XML alone: cut and mutated, tags left open, entities broken */
    const char *xs[] = {"<w:p><w:t>a &#xFFFFFFFFFF; &#0; &# &amp b</w:t>",
                        "<text:h text:outline-level=\"9\">x<text:s text:c=",
                        "<?x <!-- <![CDATA[ a",
                        "<office:body><text:p>&#55296;"
                        "</text:p><w:t \"'>\"",
                        "<office:body><text:s text:c=\"99999\"/>"};
    for (size_t i = 0; i < sizeof xs / sizeof xs[0]; i++)
        for (size_t k = 0; k <= strlen(xs[i]); k++) {
            t.n = 0;
            dz_xml_text(xs[i], k, DZ_DOCX, &t);
            t.n = 0;
            dz_xml_text(xs[i], k, DZ_ODT, &t);
        }
    janas_buf_free(&t);
    printf("test_docs: %d of 40000 mutated archives read\n", read_ok);
}

static void pieces(void)
{
    size_t n = 20000;
    char *s = malloc(n + 1);
    for (size_t i = 0; i < n; i++)
        s[i] = i % 7 == 6 ? ' ' : i % 500 == 499 ? '\n' : (char)('a' + i % 26);
    memcpy(s + 3000, "\xc3\xa8\xc3\xa8\xc3\xa8", 6);
    s[n] = 0;
    struct dz_piece *p;
    long k = dz_cut(s, n, DZ_CHUNK, DZ_OVERLAP, &p);
    int ok = k > 10;
    for (long i = 0; ok && i < k; i++) {
        ok = p[i].n > 0 && p[i].n <= DZ_CHUNK + 8 &&
             ((unsigned char)s[p[i].at] & 0xc0) != 0x80 &&
             (i == 0 ||
              (p[i].at > p[i - 1].at && p[i].at <= p[i - 1].at + p[i - 1].n));
    }
    /* to the end, but for the spaces there (s ends in " \n") */
    CHECK(ok && p[k - 1].at + p[k - 1].n == n - 2, "pieces: %ld", k);
    free(p);
    /* one word longer than a piece, and spaces alone */
    memset(s, 'x', n);
    k = dz_cut(s, n, DZ_CHUNK, DZ_OVERLAP, &p);
    CHECK(k > 0 && p[k - 1].at + p[k - 1].n == n, "one long word: %ld", k);
    free(p);
    memset(s, ' ', n);
    k = dz_cut(s, n, DZ_CHUNK, DZ_OVERLAP, &p);
    CHECK(k == 0, "spaces: %ld", k);
    free(p);
    free(s);
}

/* ---- a small embeddings server ---- */

static int srv_fd = -1, srv_port;
static char srv_model[64] = "fake-embed";
static int srv_requests;

/* 8 numbers: each word counted in its bucket */
static void vec_of(const char *t, size_t n, float *v)
{
    memset(v, 0, 8 * sizeof(float));
    const char *p = t, *e = t + n;
    char w[64];
    while (dz_word(&p, e, w, sizeof w)) {
        if (strlen(w) < 3 || dz_stopword(w) || !strcmp(w, "instruct") ||
            !strcmp(w, "given") || !strcmp(w, "search") ||
            !strcmp(w, "query") || !strcmp(w, "retrieve") ||
            !strcmp(w, "relevant") || !strcmp(w, "passages") ||
            !strcmp(w, "answer"))
            continue;
        unsigned h = 5381;
        for (const char *c = w; *c; c++)
            h = h * 33 + (unsigned char)*c;
        v[h % 8] += 1;
    }
}

static void *serve(void *arg)
{
    (void)arg;
    for (;;) { /* until the socket is shut down */
        int c = accept(srv_fd, NULL, NULL);
        if (c < 0)
            break;
        struct janas_buf in = {0};
        char b[4096];
        ssize_t r;
        long want = -1;
        size_t head = 0;
        while ((r = read(c, b, sizeof b)) > 0) {
            janas_buf_put(&in, b, (size_t)r);
            janas_buf_put(&in, "", 1);
            in.n--;
            char *e = strstr(in.p, "\r\n\r\n");
            if (e && want < 0) {
                head = (size_t)(e - in.p) + 4;
                char *cl = strcasestr(in.p, "Content-Length:");
                want = cl ? atol(cl + 15) : 0;
            }
            if (want >= 0 && in.n >= head + (size_t)want)
                break;
        }
        struct janas_buf o = {0};
        char err[64];
        struct janas_json_doc *d =
            want > 0
                ? janas_json_parse(in.p + head, (size_t)want, err, sizeof err)
                : NULL;
        const struct janas_json *input =
            d ? janas_json_get(janas_json_root(d), "input") : NULL;
        janas_buf_puts(&o, "{\"object\": \"list\", \"data\": [");
        size_t i = 0;
        for (const struct janas_json *it = input ? input->child : NULL; it;
             it = it->next, i++) {
            float v[8];
            vec_of(it->s, it->n, v);
            janas_buf_printf(&o,
                             "%s{\"object\": \"embedding\", \"index\": %zu, "
                             "\"embedding\": [",
                             i ? ", " : "", i);
            for (int k = 0; k < 8; k++)
                janas_buf_printf(&o, "%s%g", k ? ", " : "", v[k] + 0.01);
            janas_buf_puts(&o, "]}");
        }
        janas_buf_printf(&o, "], \"model\": \"%s\"}", srv_model);
        janas_json_free(d);
        __atomic_add_fetch(&srv_requests, 1, __ATOMIC_SEQ_CST);
        char h[256];
        int hn =
            snprintf(h, sizeof h,
                     "HTTP/1.1 200 OK\r\nContent-Type: application/json"
                     "\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                     o.n);
        if (write(c, h, (size_t)hn) < 0 || write(c, o.p, o.n) < 0)
            perror("write");
        close(c);
        janas_buf_free(&in);
        janas_buf_free(&o);
    }
    return NULL;
}

static int serve_start(pthread_t *t)
{
    srv_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET,
                            .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t al = sizeof a;
    if (srv_fd < 0 || bind(srv_fd, (struct sockaddr *)&a, sizeof a) != 0 ||
        listen(srv_fd, 8) != 0 ||
        getsockname(srv_fd, (struct sockaddr *)&a, &al) != 0)
        return -1;
    srv_port = ntohs(a.sin_port);
    return pthread_create(t, NULL, serve, NULL);
}

static void answers(void)
{
    static const char A[] =
        "{\"data\": [{\"index\": 0, \"embedding\": [3, 4]}, {\"index\": 1, "
        "\"embedding\": [0, 2]}], \"model\": \"m\"}";
    float *v;
    int dim;
    char model[16], why[128];
    CHECK(dz_embed_parse(A, strlen(A), 2, &v, &dim, model, sizeof model, why,
                         sizeof why) == 0 &&
              dim == 2 && v[0] == 0.6f && v[1] == 0.8f && v[3] == 1 &&
              !strcmp(model, "m"),
          "an answer: %s", why);
    free(v);
    CHECK(dz_embed_parse(A, strlen(A), 3, &v, &dim, model, sizeof model, why,
                         sizeof why) == -1,
          "one vector short");
    static const char E[] = "{\"error\": {\"message\": \"no model\"}}";
    CHECK(dz_embed_parse(E, strlen(E), 1, &v, &dim, NULL, 0, why, sizeof why) ==
                  -1 &&
              !strcmp(why, "no model"),
          "an error: %s", why);
    char m[sizeof A];
    srand(77);
    for (int r = 0; r < 20000; r++) {
        memcpy(m, A, sizeof A);
        for (int i = 0; i < 3; i++)
            m[rand() % (sizeof A - 1)] = (char)(rand() % 128);
        if (dz_embed_parse(m, sizeof A - 1, 2, &v, &dim, model, sizeof model,
                           why, sizeof why) == 0)
            free(v);
    }
}

/* ---- the index file ---- */

static void store(const char *dir)
{
    struct dz_index x = {0}, y = {0};
    snprintf(x.source, sizeof x.source, "server x, model m, 4 numbers");
    x.dim = 4;
    x.f = calloc(2, sizeof(*x.f));
    x.n = x.cap = 2;
    const char *paths[2] = {"/a/one.md", "/a/two.md"};
    for (int i = 0; i < 2; i++) {
        x.f[i].path = strdup(paths[i]);
        x.f[i].mtime = 1000 + i;
        x.f[i].size = 10 + i;
        x.f[i].n_chunk = 2;
        x.f[i].c = calloc(2, sizeof(struct dz_chunk));
        for (int k = 0; k < 2; k++) {
            x.f[i].c[k].text = strdup(k ? "second piece" : "first");
            x.f[i].c[k].n = (uint32_t)strlen(x.f[i].c[k].text);
            if (i == k) {
                x.f[i].c[k].vec = malloc(16);
                for (int j = 0; j < 4; j++)
                    x.f[i].c[k].vec[j] = (float)(j + i);
            }
        }
    }
    snprintf(x.path, sizeof x.path, "%s/store-test.bin", dir);
    CHECK(dz_index_save(&x) == 0, "saved");
    char why[PATH_MAX + 64];
    CHECK(dz_index_load(&y, x.path, why, sizeof why) == 0 && y.n == 2 &&
              y.dim == 4 && !strcmp(y.source, x.source) &&
              y.f[1].mtime == 1001 && y.f[1].c[1].vec &&
              y.f[1].c[1].vec[3] == 4 && !y.f[1].c[0].vec &&
              !strcmp(y.f[0].c[1].text, "second piece"),
          "loaded: %s", why);
    dz_index_free(&y);
    /* cut at every byte, mutated */
    FILE *f = fopen(x.path, "rb");
    uint8_t buf[512];
    size_t n = f ? fread(buf, 1, sizeof buf, f) : 0;
    if (f)
        fclose(f);
    int whole = 0;
    for (size_t k = 0; k <= n; k++) {
        struct dz_index z = {0};
        whole += dz_index_parse(&z, buf, k) == 0;
        dz_index_free(&z);
    }
    CHECK(whole == 1, "only the whole file reads: %d", whole);
    uint8_t m[512];
    srand(99);
    for (int r = 0; r < 20000; r++) {
        memcpy(m, buf, n);
        for (int i = 0; i < 4; i++)
            m[rand() % n] = (uint8_t)rand();
        struct dz_index z = {0};
        dz_index_parse(&z, m, n);
        dz_index_free(&z);
    }
    dz_index_free(&x);
}

/* ---- a home with documents ---- */

static char base[PATH_MAX - 64], home[PATH_MAX], docs[PATH_MAX + 16];

static void put_doc(const char *name, const void *s, size_t n)
{
    char p[2 * PATH_MAX];
    snprintf(p, sizeof p, "%s/%s", docs, name);
    FILE *f = fopen(p, "wb");
    if (f) {
        fwrite(s, 1, n ? n : strlen(s), f);
        fclose(f);
    }
}

static int rm_one(const char *p, const struct stat *st, int flag,
                  struct FTW *ftw)
{
    (void)st, (void)flag, (void)ftw;
    return remove(p);
}

static struct janas_buf o;

static const char *search(struct dz_ctx *c, const char *q, const char *folder)
{
    o.n = 0;
    dz_tool_search(c, q, folder, 6, &o);
    janas_buf_put(&o, "", 1);
    return o.p;
}

static void setup_ctx(struct dz_ctx *c, const char *server)
{
    memset(c, 0, sizeof *c);
    dz_roots_load(&c->roots);
    char why[256];
    CHECK(dz_embedder_open(&c->emb, server, NULL, NULL, why, sizeof why) == 0,
          "embedder: %s", why);
    pthread_mutex_init(&c->index.mu, NULL);
    pthread_cond_init(&c->index.wake, NULL);
    c->index.roots = &c->roots;
    c->index.emb = &c->emb;
    char p[PATH_MAX + 32];
    snprintf(p, sizeof p, "%s/index.bin", base);
    CHECK(dz_index_load(&c->index, p, why, sizeof why) == 0, "load: %s", why);
}

static void end_ctx(struct dz_ctx *c)
{
    dz_index_free(&c->index);
    dz_embedder_close(&c->emb);
    pthread_mutex_destroy(&c->index.mu);
    pthread_cond_destroy(&c->index.wake);
}

static void home_tests(void)
{
    char p[2 * PATH_MAX], q[2 * PATH_MAX];
    snprintf(home, sizeof home, "%s/home", base);
    snprintf(docs, sizeof docs, "%s/Documents", home);
    mkdir(home, 0755);
    mkdir(docs, 0755);
    snprintf(p, sizeof p, "%s/.hidden", docs);
    mkdir(p, 0755);
    snprintf(p, sizeof p, "%s/sub", docs);
    mkdir(p, 0755);
    put_doc("notes.md",
            "# Notes\nThe boiler was serviced on 3 March by "
            "Idraulica Rossi.\n",
            0);
    put_doc("contratto.docx", DOCX, sizeof DOCX);
    put_doc("sub/ricetta.odt", ODT, sizeof ODT);
    put_doc(".env", "TOKEN=canone\n", 0);
    put_doc("secrets.txt", "canone segreto\n", 0);
    put_doc(".hidden/x.md", "canone nascosto\n", 0);
    put_doc("bin.txt", "a\0b canone", 10);
    put_doc("photo.jpg", "canone", 0);
    snprintf(p, sizeof p, "%s/outside.md", base);
    FILE *f = fopen(p, "w");
    if (f) {
        fputs("canone fuori\n", f);
        fclose(f);
    }
    snprintf(q, sizeof q, "%s/link.md", docs);
    CHECK(symlink(p, q) == 0, "link");
    setenv("HOME", home, 1);
    unsetenv("XDG_CONFIG_HOME");

    /* by the words */
    struct dz_ctx c;
    setup_ctx(&c, NULL);
    CHECK(c.roots.n_root == 1 && !c.roots.listed, "the Documents folder");
    dz_index_round(&c.index);
    CHECK(c.index.n == 4, "4 documents (bin.txt with no text): %zu", c.index.n);
    CHECK(c.index.failed == 1 && strstr(c.index.last_fail, "bin.txt"),
          "bin.txt not text: %s", c.index.last_fail);
    const char *r = search(&c, "canone", NULL);
    CHECK(strstr(r, "1 passage from 1 document") &&
              strstr(r, "~/Documents/contratto.docx") && strstr(r, "850") &&
              !strstr(r, "segreto") && !strstr(r, "nascosto") &&
              !strstr(r, "fuori") && !strstr(r, "TOKEN"),
          "canone: %s", r);
    r = search(&c, "lievito", NULL);
    CHECK(strstr(r, "sub/ricetta.odt") && strstr(r, "(words)"), "lievito: %s",
          r);
    snprintf(p, sizeof p, "%s/sub", docs);
    r = search(&c, "boiler lievito", p);
    CHECK(strstr(r, "ricetta") && !strstr(r, "notes"), "a folder: %s", r);
    r = search(&c, "Idraulica Rossi boiler", NULL);
    CHECK(strstr(r, "notes.md") && strstr(r, "3 March"), "notes: %s", r);
    r = search(&c, "nothing like it", NULL);
    CHECK(strstr(r, "0 passages"), "nothing: %s", r);

    /* reading */
    o.n = 0;
    CHECK(dz_tool_read(&c, "~/Documents/contratto.docx", 0, 0, &o) == 0 &&
              strstr(o.p, "Seconda riga"),
          "read: %.*s", (int)o.n, o.p);
    o.n = 0;
    CHECK(dz_tool_read(&c, "~/Documents/contratto.docx", 10, 5, &o) == 0 &&
              strstr(o.p, "from 10 to 15"),
          "a stretch: %.*s", (int)o.n, o.p);
    const char *refused[] = {"~/Documents/.env",    "~/Documents/secrets.txt",
                             "~/Documents/link.md", "~/Documents/photo.jpg",
                             "/etc/passwd",         "relative.md",
                             "~/nothing.md"};
    for (size_t i = 0; i < sizeof refused / sizeof refused[0]; i++) {
        o.n = 0;
        CHECK(dz_tool_read(&c, refused[i], 0, 0, &o) == -1, "%s read: %.*s",
              refused[i], (int)o.n, o.p);
    }
    o.n = 0;
    dz_tool_status(&c, &o);
    janas_buf_put(&o, "", 1);
    CHECK(strstr(o.p, "4 documents indexed") && strstr(o.p, "words alone"),
          "status: %s", o.p);

    /* a document changed, one removed: the next round sees both */
    put_doc("notes.md", "# Notes\nThe heat pump replaced the boiler.\n", 0);
    snprintf(p, sizeof p, "%s/notes.md", docs);
    struct stat st;
    stat(p, &st);
    st.st_mtime += 10; /* a second may not have passed */
    struct timespec ts[2] = {{.tv_sec = st.st_mtime}, {.tv_sec = st.st_mtime}};
    utimensat(AT_FDCWD, p, ts, 0);
    snprintf(p, sizeof p, "%s/sub/ricetta.odt", docs);
    remove(p);
    dz_index_round(&c.index);
    r = search(&c, "heat pump", NULL);
    CHECK(strstr(r, "notes.md") && strstr(r, "heat pump"), "changed: %s", r);
    r = search(&c, "lievito", NULL);
    CHECK(strstr(r, "0 passages"), "removed: %s", r);
    end_ctx(&c);

    /* the same index read again from its file */
    setup_ctx(&c, NULL);
    CHECK(c.index.n == 3, "reloaded: %zu", c.index.n);
    r = search(&c, "canone", NULL);
    CHECK(strstr(r, "contratto.docx"), "after reload: %s", r);
    end_ctx(&c);

    /* by the meaning, from the server */
    pthread_t t;
    if (serve_start(&t) != 0) {
        CHECK(0, "the test's server");
        return;
    }
    char url[64];
    snprintf(url, sizeof url, "http://127.0.0.1:%d", srv_port);
    setup_ctx(&c, url);
    CHECK(dz_embed_state(&c.emb, NULL, 0, NULL) && c.emb.dim == 8 &&
              strstr(c.emb.name, "fake-embed"),
          "probed: %s", c.emb.name);
    dz_index_round(&c.index);
    size_t vec = 0, pieces = 0;
    for (size_t i = 0; i < c.index.n; i++)
        for (uint32_t k = 0; k < c.index.f[i].n_chunk; k++, pieces++)
            vec += c.index.f[i].c[k].vec != NULL;
    CHECK(pieces && vec == pieces && strstr(c.index.source, "fake-embed"),
          "vectors: %zu of %zu", vec, pieces);
    r = search(&c, "canone", NULL);
    CHECK(strstr(r, "meaning and their words") &&
              strstr(r, "contratto.docx, passage 1 of 1 (words and meaning"),
          "by meaning: %s", r);
    end_ctx(&c);

    /* another model: the vectors made again, the text kept */
    snprintf(srv_model, sizeof srv_model, "fake-embed-2");
    int before = __atomic_load_n(&srv_requests, __ATOMIC_SEQ_CST);
    setup_ctx(&c, url);
    CHECK(strstr(c.index.source, "fake-embed,") && c.index.n == 3,
          "the old vectors loaded: %s", c.index.source);
    dz_index_round(&c.index);
    CHECK(strstr(c.index.source, "fake-embed-2") &&
              __atomic_load_n(&srv_requests, __ATOMIC_SEQ_CST) > before + 1,
          "made again: %s, %d requests", c.index.source,
          __atomic_load_n(&srv_requests, __ATOMIC_SEQ_CST) - before);
    vec = 0;
    for (size_t i = 0; i < c.index.n; i++)
        for (uint32_t k = 0; k < c.index.f[i].n_chunk; k++)
            vec += c.index.f[i].c[k].vec != NULL;
    CHECK(vec == pieces, "all again: %zu", vec);
    /* the thread, started and stopped */
    CHECK(dz_indexer_start(&c.index) == 0, "thread");
    dz_indexer_poke(&c.index);
    r = search(&c, "boiler", NULL);
    CHECK(strstr(r, "notes.md"), "while the thread runs: %s", r);
    dz_indexer_stop(&c.index);
    end_ctx(&c);

    /* a start without the model keeps them, and says so */
    setup_ctx(&c, NULL);
    dz_index_round(&c.index);
    o.n = 0;
    dz_tool_status(&c, &o);
    janas_buf_put(&o, "", 1);
    CHECK(strstr(c.index.source, "fake-embed-2") &&
              strstr(o.p, "keep their vectors from server"),
          "kept: %s", o.p);
    end_ctx(&c);

    shutdown(srv_fd, SHUT_RDWR);
    close(srv_fd);
    pthread_join(t, NULL);
}

int main(void)
{
    documents();
    pieces();
    answers();
    const char *h = getenv("HOME");
    char parent[PATH_MAX - 128];
    snprintf(parent, sizeof parent, "%s/tmp", h ? h : "/tmp");
    mkdir(parent, 0700);
    snprintf(base, sizeof base, "%s/janas-docs-test-XXXXXX", parent);
    if (!mkdtemp(base)) {
        perror("mkdtemp");
        return 1;
    }
    store(base);
    home_tests();
    janas_buf_free(&o);
    nftw(base, rm_one, 16, FTW_DEPTH | FTW_PHYS);
    printf("test_docs: %s\n", failures ? "FAILED" : "ok");
    return failures != 0;
}
