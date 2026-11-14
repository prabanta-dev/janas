/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * extract.c - the text of a document, by the kind its name says: plain
 * text and Markdown as they are (when they are text: no NUL, valid UTF-8
 * but for a byte in a hundred), a docx's word/document.xml and an odt's
 * content.xml through xml.c. Anything else is not a document here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

#include "docs.h"

static const char *const TEXT_EXT[] = {"txt",      "text", "md",  "markdown",
                                       "mdown",    "rst",  "org", "adoc",
                                       "asciidoc", "tex",  "csv", "tsv"};

int dz_kind(const char *path)
{
    const char *slash = strrchr(path, '/'), *dot = strrchr(path, '.');
    if (!dot || (slash && dot < slash) || dot == (slash ? slash + 1 : path))
        return DZ_NONE;
    dot++;
    if (!strcasecmp(dot, "docx"))
        return DZ_DOCX;
    if (!strcasecmp(dot, "odt"))
        return DZ_ODT;
    for (size_t i = 0; i < sizeof TEXT_EXT / sizeof TEXT_EXT[0]; i++)
        if (!strcasecmp(dot, TEXT_EXT[i]))
            return DZ_TEXT;
    return DZ_NONE;
}

int dz_is_text(const char *t, size_t n)
{
    if (memchr(t, 0, n))
        return 0;
    size_t bad = 0;
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)t[i];
        size_t k = c < 0x80         ? 1
                   : (c >> 5) == 6  ? 2
                   : (c >> 4) == 14 ? 3
                   : (c >> 3) == 30 ? 4
                                    : 0;
        if (!k || i + k > n) {
            bad++;
            i++;
            continue;
        }
        for (size_t j = 1; j < k; j++)
            if (((unsigned char)t[i + j] & 0xc0) != 0x80) {
                bad++;
                k = 1;
                break;
            }
        i += k;
    }
    return bad * 100 <= n;
}

int dz_extract_mem(const uint8_t *p, size_t n, int kind, struct janas_buf *text,
                   char *why, size_t wn)
{
    if (kind == DZ_TEXT) {
        if (n >= 3 && !memcmp(p, "\xef\xbb\xbf", 3)) /* a byte order mark */
            p += 3, n -= 3;
        if (!dz_is_text((const char *)p, n)) {
            snprintf(why, wn, "not text");
            return -1;
        }
        janas_buf_put(text, (const char *)p, n);
        return text->oom ? -1 : 0;
    }
    if (kind != DZ_DOCX && kind != DZ_ODT) {
        snprintf(why, wn, "not a kind of document read here");
        return -1;
    }
    uint8_t *x;
    size_t xn;
    if (dz_zip_get(p, n, kind == DZ_DOCX ? "word/document.xml" : "content.xml",
                   &x, &xn, why, wn) != 0)
        return -1;
    int rc = dz_xml_text((const char *)x, xn, kind, text);
    free(x);
    if (rc != 0)
        snprintf(why, wn, "out of memory");
    return rc;
}

int dz_extract(const char *path, struct janas_buf *text, char *why, size_t wn)
{
    int kind = dz_kind(path);
    if (kind == DZ_NONE) {
        snprintf(why, wn, "not a kind of document read here");
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(why, wn, "cannot be opened");
        return -1;
    }
    struct stat st;
    if (fstat(fileno(f), &st) != 0 || !S_ISREG(st.st_mode) ||
        st.st_size > DZ_MAX_FILE) {
        fclose(f);
        snprintf(why, wn, "not a file of at most %d MiB", DZ_MAX_FILE >> 20);
        return -1;
    }
    size_t n = (size_t)st.st_size;
    uint8_t *p = malloc(n + 1);
    if (!p || fread(p, 1, n, f) != n) {
        free(p);
        fclose(f);
        snprintf(why, wn, "cannot be read");
        return -1;
    }
    fclose(f);
    int rc = dz_extract_mem(p, n, kind, text, why, wn);
    free(p);
    return rc;
}
