/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * icons.c - Brebus's icons, the Prabanta logo drawn for each size the
 * desktops ask for (icons/README.md), carried in the program: the page's,
 * the window's, and those --install-desktop writes.
 */
#include "brebus.h"

#include "brebus_icon128_png.h" /* made by build.sh from icons/ */
#include "brebus_icon16_png.h"
#include "brebus_icon22_png.h"
#include "brebus_icon24_png.h"
#include "brebus_icon256_png.h"
#include "brebus_icon32_png.h"
#include "brebus_icon48_png.h"
#include "brebus_icon512_png.h"
#include "brebus_icon64_png.h"
#include "brebus_icon_svg.h"

#define ICON(s) {s, janas_brebus_icon##s##_png, janas_brebus_icon##s##_png_len}

const struct brebus_icon brebus_icons[] = {
    ICON(16), ICON(22),  ICON(24),  ICON(32),  ICON(48),
    ICON(64), ICON(128), ICON(256), ICON(512), {0, NULL, 0},
};

const unsigned char *const brebus_icon_svg = janas_brebus_icon_svg;
const unsigned brebus_icon_svg_len = janas_brebus_icon_svg_len;

const struct brebus_icon *brebus_icon(int size)
{
    for (const struct brebus_icon *i = brebus_icons; i->size; i++)
        if (i->size == size)
            return i;
    return NULL;
}
