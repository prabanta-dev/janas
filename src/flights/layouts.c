/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layouts.c - the layouts of janas-flights' answers (see flights.h and
 * common/template.h): the text the user reads, filled with the data in
 * English here and in the user's language by a client that translates the
 * layout once (janas-chat), and the brief, all the model reads of it.
 */
#include "flights.h"

/* The words every layout of aircraft shares: the compass, the codes of an
   emergency. */
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
    "squawk.7500 = 7500, unlawful interference (hijacking)\n"                  \
    "squawk.7600 = 7600, radio failure\n"                                      \
    "squawk.7700 = 7700, general emergency\n"                                  \
    "ground.stationary = stationary\n"                                         \
    "ground.taxiing = taxiing\n"                                               \
    "ground.runway = on the runway (taking off or just landed)\n"

/* One aircraft, as an item of "aircraft". */
#define AIRCRAFT "{{@n}}. " AIRCRAFT_BODY
#define AIRCRAFT_BODY                                                          \
    "{{#callsign}}{{callsign}}{{/callsign}}{{^callsign}}no callsign "          \
    "(transponder {{hex}}){{/callsign}}{{#reg}} ({{reg}}{{#type}}, "           \
    "{{type}}{{/type}}){{/reg}}{{^reg}}{{#type}} ({{type}}){{/type}}"          \
    "{{/reg}}: {{#ground}}on the ground{{#airport}} at {{airport}}"            \
    "{{/airport}}{{#on_ground}}, {{on_ground|ground}}{{/on_ground}}"           \
    "{{/ground}}{{^ground}}{{#fl}}FL{{fl}}"                                    \
    "{{/fl}}{{#ft}}{{ft}} ft{{/ft}}{{^fl}}{{^ft}}altitude not reported"        \
    "{{/ft}}{{/fl}}{{#climbing}} climbing{{/climbing}}{{#descending}} "        \
    "descending{{/descending}}{{/ground}}{{#kts}}, {{kts}} kts{{/kts}}"        \
    "{{#heading}}, towards {{heading|dir}}{{/heading}}; "                      \
    "{{#where.sea}}over the {{where.sea}}{{#where.and}}, between "             \
    "{{where.region}} and {{where.and}}{{/where.and}}{{/where.sea}}"           \
    "{{^where.sea}}over {{where.region}}{{/where.sea}}; {{#where.km}}"         \
    "{{where.km}} km {{where.dir|dir}} of {{where.city}}{{/where.km}}"         \
    "{{^where.km}}over {{where.city}}{{/where.km}}{{#emergency}}; "            \
    "EMERGENCY: transponder code {{emergency|squawk}}{{/emergency}}"           \
    "{{#route}}; route on record: {{route.from}} ({{route.from_code}}) to "    \
    "{{route.to}} ({{route.to_code}}){{#route.airline}}, {{route.airline}}"    \
    "{{/route.airline}}{{/route}}"

/* The sources, at the foot. */
#define SOURCES                                                                \
    "Sources: live positions from {{net}} ({{#odbl}}data under the ODbL, "     \
    "{{/odbl}}{{net_url}}){{#routes}}; routes from adsbdb.com{{/routes}}"

const char FL_OVER_LAYOUT[] =
    "As of {{time}} UTC: {{count}} aircraft over the {{sea}} ({{airborne}} "
    "in the air){{#failed}}; {{failed}} of the {{parts}} parts of the sea "
    "could not be asked, so there may be more{{/failed}}{{#listed}}; "
    "{{listed}} of them listed{{/listed}}{{#count}}, from west to "
    "east:{{/count}}{{^count}}.{{/count}}\n"
    "{{#aircraft}}\n" AIRCRAFT "\n"
    "{{/aircraft}}\n"
    "{{#routes}}\n"
    "\nRoutes on record come from the community (adsbdb), not from "
    "schedules: usually right, sometimes not; those that do not fit the "
    "position are left out.\n"
    "{{/routes}}\n"
    "Only aircraft with a transponder heard by the community's receivers "
    "are counted: nearly all airliners, not every small or military "
    "aircraft.\n\n" SOURCES "; sea boundaries from Natural Earth.\n"
    "---\n" WORDS;

const char FL_OVER_BRIEF[] =
    "{{count}} aircraft over the {{sea}} at {{time}} UTC ({{airborne}} in "
    "the air){{#failed}}; {{failed}} of {{parts}} parts of the sea not "
    "asked, so there may be more{{/failed}}.{{#emergencies}} "
    "{{emergencies}} of them squawk an emergency: say so.{{/emergencies}} "
    "The list, aircraft by aircraft, is shown to the user as it is: do not "
    "repeat it; answer in a line or two from these figures, in the user's "
    "language.";

/* Where the user is, as the locator says how it knows. */
#define HOW_WORDS                                                              \
    "how.as set on the computer (JANAS_LOCATION) = as set on the computer "    \
    "(JANAS_LOCATION)\n"                                                       \
    "how.estimated from the internet connection (GeoJS) = estimated from "     \
    "the internet connection (GeoJS)\n"                                        \
    "how.estimated from the internet connection (ipwho.is) = estimated from "  \
    "the internet connection (ipwho.is)\n"                                     \
    "how.a rough guess, the city of the computer's time zone = a rough "       \
    "guess, the city of the computer's time zone\n"

const char FL_NEAR_LAYOUT[] =
    "As of {{time}} UTC.\n"
    "{{#point}}\n"
    "The point is {{#point.sea}}over the {{point.sea}}{{#point.and}}, "
    "between {{point.region}} and {{point.and}}{{/point.and}}{{/point.sea}}"
    "{{^point.sea}}over {{point.region}}{{/point.sea}}; {{#point.km}}"
    "{{point.km}} km {{point.dir|dir}} of {{point.city}}{{/point.km}}"
    "{{^point.km}}over {{point.city}}{{/point.km}}.\n"
    "{{/point}}\n"
    "{{count}} aircraft within {{radius}} km of {{place}}{{#how}} (where "
    "the user is, {{how|how}}){{/how}}{{#listed}}; the nearest {{listed}}"
    "{{/listed}}{{#count}}:{{/count}}{{^count}}.{{/count}}\n"
    "{{#aircraft}}\n"
    "{{@n}}. {{dist_km}} km {{dist_dir|dir}}: " AIRCRAFT_BODY "\n"
    "{{/aircraft}}\n"
    "\n" SOURCES ".{{#no_schedules}} No schedules (the server has no "
    "AviationStack key): scheduled and official estimated times are not "
    "known to this tool.{{/no_schedules}}\n"
    "---\n" WORDS HOW_WORDS;

const char FL_NEAR_BRIEF[] =
    "{{count}} aircraft within {{radius}} km of {{place}} at {{time}} UTC"
    "{{#how}}; the place is where the user is, {{how}}: say which place, "
    "and that it may be off{{/how}}. The list, aircraft by aircraft, is "
    "shown to the user as it is: do not repeat it; answer in a line or two "
    "from these figures, in the user's language.";
