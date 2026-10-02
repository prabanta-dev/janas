/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layouts.c - the layouts of janas-quakes' answers (see quakes.h and
 * services/common/template.h): the text the user reads, filled with the
 * data in English here and in the user's language by a client that
 * translates the layout once (janas-chat), and the brief, all the model
 * reads of it. The sources' names of places stay as they give them (INGV's
 * in Italian, the USGS's in English).
 *
 * The USGS's tsunami flag is set for large events at sea: it does not say
 * that a tsunami was, and the words say so. Its alert is PAGER's, the
 * impact it expects.
 */
#include "quakes.h"

#define WORDS                                                                  \
    "dir.N = N\n"                                                              \
    "dir.NNE = NNE\n"                                                          \
    "dir.NE = NE\n"                                                            \
    "dir.ENE = ENE\n"                                                          \
    "dir.E = E\n"                                                              \
    "dir.ESE = ESE\n"                                                          \
    "dir.SE = SE\n"                                                            \
    "dir.SSE = SSE\n"                                                          \
    "dir.S = S\n"                                                              \
    "dir.SSW = SSW\n"                                                          \
    "dir.SW = SW\n"                                                            \
    "dir.WSW = WSW\n"                                                          \
    "dir.W = W\n"                                                              \
    "dir.WNW = WNW\n"                                                          \
    "dir.NW = NW\n"                                                            \
    "dir.NNW = NNW\n"                                                          \
    "alert.green = green, little damage expected\n"                            \
    "alert.yellow = yellow, some damage expected\n"                            \
    "alert.orange = orange, serious damage expected\n"                         \
    "alert.red = red, extensive damage expected\n"                             \
    "src.ingv = INGV (Istituto Nazionale di Geofisica e Vulcanologia, CC BY "  \
    "4.0)\n"                                                                   \
    "src.usgs = USGS (U.S. Geological Survey)\n"

#define WHEN                                                                   \
    "{{#when.today}}today{{/when.today}}{{^when.today}}{{when.day}}/"          \
    "{{when.mon}}{{/when.today}} {{when.at}}"
#define AGO                                                                    \
    "{{#ago.d}}{{ago.d}} d {{/ago.d}}{{#ago.h}}{{ago.h}} h "                   \
    "{{/ago.h}}{{ago.m}} min ago"
#define CARE                                                                   \
    "Source: {{source|src}}. The first estimates change in the first hours; "  \
    "in an emergency, follow the civil protection's word.\n"
#define SHOWN                                                                  \
    " Shown to the user: do not repeat the list; answer in a line or two, "    \
    "in the user's language."

const char QK_NEAR_LAYOUT[] =
    "{{#n}}Earthquakes {{#sea}}in the {{name}}{{/sea}}{{^sea}}within "
    "{{radius}} km of {{name}}{{/sea}} in the last {{days}} days, of "
    "magnitude {{min_mag}} or more{{#more}} ({{n}} listed, {{more}} "
    "more){{/more}}:"
    "{{/n}}{{^n}}No earthquake of magnitude {{min_mag}} or more "
    "{{#sea}}in the {{name}}{{/sea}}{{^sea}}within {{radius}} km of "
    "{{name}}{{/sea}} in the last {{days}} days.{{/n}}\n"
    "{{#items}}\n"
    "  " WHEN "{{#when.today}} (" AGO ")"
    "{{/when.today}}: magnitude {{mag}}, {{depth}} km deep, {{^sea}}{{km}} "
    "km {{dir|dir}} of {{name}}: {{/sea}}{{place}}{{#auto}} (first "
    "estimate){{/auto}}\n"
    "{{/items}}\n" CARE "---\n" WORDS;

const char QK_NEAR_BRIEF[] =
    "{{n}}{{#more}}+{{more}}{{/more}} earthquakes M{{min_mag}}+ "
    "{{#sea}}in the {{name}}{{/sea}}{{^sea}}within {{radius}} km of "
    "{{name}}{{/sea}}{{#how}} ({{how}}){{/how}}, last {{days}} "
    "days{{#n}}, strongest M{{top}}{{/n}}: {{#items}}[{{id}}] " WHEN
    " M{{mag}} "
    "{{^sea}}{{km}} km {{dir}} {{/sea}}{{place}}, {{depth}} km deep; "
    "{{/items}}" SHOWN;

const char QK_STRONG_LAYOUT[] =
    "{{#n}}The strongest earthquakes in the world in the last {{days}} days, "
    "of magnitude {{min_mag}} or more{{#more}} ({{n}} listed, {{more}} "
    "more){{/more}}:"
    "{{/n}}{{^n}}No earthquake of magnitude {{min_mag}} or more in the world "
    "in the last {{days}} days.{{/n}}\n"
    "{{#items}}\n"
    "  magnitude {{mag}}, " WHEN ": {{place}}, {{depth}} km "
    "deep{{#alert}}; impact: {{alert|alert}}{{/alert}}{{#tsunami}}; at sea "
    "and strong enough for a tsunami: see the warning centres{{/tsunami}}\n"
    "{{/items}}\n" CARE "---\n" WORDS;

const char QK_STRONG_BRIEF[] =
    "{{n}} earthquakes M{{min_mag}}+ in the world, last {{days}} days; "
    "impact alerts above green: {{alerts}}; at sea and strong: {{tsunamis}}. "
    "{{#items}}[{{id}}] M{{mag}} " WHEN " {{place}}{{#alert}} alert "
    "{{alert}}{{/alert}}{{#tsunami}} tsunami flag (not a tsunami "
    "seen){{/tsunami}}; {{/items}}" SHOWN;

const char QK_EVENT_LAYOUT[] =
    "{{#e}}\n"
    "Earthquake of magnitude {{mag}} ({{magtype}}){{#kind}}, {{kind}}"
    "{{/kind}}: {{place}}\n"
    "  when: " WHEN " (" AGO ")\n"
    "  where: {{#where.sea}}in the {{where.sea}}{{/where.sea}}{{^where.sea}}"
    "{{where.region}}{{/where.sea}}{{#where.city}}, {{#where.km}}{{where.km}} "
    "km {{where.dir|dir}} of {{/where.km}}{{where.city}}{{/where.city}}; "
    "{{depth}} km deep{{#deep}}, a deep one{{/deep}}\n"
    "{{#from}}\n"
    "  from {{from}}: {{km}} km {{dir|dir}}\n"
    "{{/from}}\n"
    "{{#has_felt}}\n"
    "  felt: {{felt}} people told the USGS they felt it\n"
    "{{/has_felt}}\n"
    "{{#has_mmi}}\n"
    "  the strongest shaking, estimated: {{mmi}} on the Mercalli scale\n"
    "{{/has_mmi}}\n"
    "{{#alert}}\n"
    "  impact: {{alert|alert}}\n"
    "{{/alert}}\n"
    "{{#tsunami}}\n"
    "  at sea and strong enough for a tsunami: see the warning centres\n"
    "{{/tsunami}}\n"
    "{{#auto}}\n"
    "  an automatic first estimate: it may change\n"
    "{{/auto}}\n"
    "  more: {{url}}\n"
    "{{/e}}\n" CARE "---\n" WORDS;

const char QK_EVENT_BRIEF[] =
    "{{#e}}[{{id}}] M{{mag}} {{magtype}}, " WHEN " (" AGO
    "), {{place}}, {{depth}} km "
    "deep{{#from}}, {{km}} km {{dir}} of {{from}}{{/from}}{{#has_felt}}, "
    "felt by {{felt}}{{/has_felt}}{{#has_mmi}}, MMI {{mmi}}{{/has_mmi}}"
    "{{#alert}}, alert {{alert}}{{/alert}}{{#tsunami}}, tsunami flag (not a "
    "tsunami seen){{/tsunami}}{{#auto}}, automatic{{/auto}}.{{/e}}" SHOWN;
