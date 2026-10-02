/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layouts.c - the layouts of janas-maps' answers (see maps.h and
 * services/common/template.h): the text the user reads, filled with the data in
 * English here and in the user's language by a client that translates the
 * layout once (janas-chat), and the brief, all the model reads of it. A
 * route's steps come from Valhalla already in the computer's language.
 */
#include "maps.h"

/* The compass, abbreviated. */
#define DIR_WORDS                                                              \
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
    "dir.NNW = NNW\n"

/* The days of the week (0 Sunday) and the months. */
#define DATE_WORDS                                                             \
    "wd.0 = Sunday\n"                                                          \
    "wd.1 = Monday\n"                                                          \
    "wd.2 = Tuesday\n"                                                         \
    "wd.3 = Wednesday\n"                                                       \
    "wd.4 = Thursday\n"                                                        \
    "wd.5 = Friday\n"                                                          \
    "wd.6 = Saturday\n"                                                        \
    "mon.1 = January\n"                                                        \
    "mon.2 = February\n"                                                       \
    "mon.3 = March\n"                                                          \
    "mon.4 = April\n"                                                          \
    "mon.5 = May\n"                                                            \
    "mon.6 = June\n"                                                           \
    "mon.7 = July\n"                                                           \
    "mon.8 = August\n"                                                         \
    "mon.9 = September\n"                                                      \
    "mon.10 = October\n"                                                       \
    "mon.11 = November\n"                                                      \
    "mon.12 = December\n"

/* Where the user is, as the locator says how it knows it. */
#define HOW_WORDS                                                              \
    "how.as set on the computer (JANAS_LOCATION) = as set on the computer "    \
    "(JANAS_LOCATION)\n"                                                       \
    "how.estimated from the internet connection (GeoJS) = estimated from "     \
    "the internet connection (GeoJS)\n"                                        \
    "how.estimated from the internet connection (ipwho.is) = estimated from "  \
    "the internet connection (ipwho.is)\n"                                     \
    "how.a rough guess, the city of the computer's time zone = a rough "       \
    "guess, the city of the computer's time zone\n"

/* A place, with where the user is when it was not named. */
#define PLACE(p)                                                               \
    "{{" p ".name}}{{#" p ".how}} (where the user is, {{" p ".how|how}})"      \
    "{{/" p ".how}}"

/* A time, with its day when it is not today. */
#define CLOCK(t)                                                               \
    "{{" t ".at}}{{#" t ".date}} {{" t ".wd|wd}} {{" t ".day}} {{" t           \
    ".mon|mon}}{{/" t ".date}}"

#define FROM PLACE("from")
#define AT PLACE("place")
#define DEP CLOCK("dep")
#define ARR CLOCK("arr")

#define DIST "{{#km}}{{km}} km{{/km}}{{#m}}{{m}} m{{/m}}"
#define TIME "{{#h}}{{h}} h {{/h}}{{min}} min"

#define OSM "data © OpenStreetMap contributors (ODbL)"

const char MP_ROUTE_LAYOUT[] =
    "From " FROM "{{#via}} through {{via.name}}{{/via}} to "
    "{{to.name}} {{mode|mode}}: " DIST ", about " TIME " without traffic"
    "{{#roads}}, by {{roads}}{{/roads}}{{#toll}}; with tolls{{/toll}}"
    "{{#ferry}}; with a ferry{{/ferry}}.\n"
    "{{#from.address}}\n"
    "- from: {{from.address}}\n"
    "{{/from.address}}\n"
    "{{#via.address}}\n"
    "- through: {{via.address}}\n"
    "{{/via.address}}\n"
    "{{#to.address}}\n"
    "- to: {{to.address}}\n"
    "{{/to.address}}\n"
    "{{#alternatives}}\n"
    "Another way: " DIST ", about " TIME "{{#roads}}, by {{roads}}{{/roads}}"
    "{{#toll}}; with tolls{{/toll}}{{#ferry}}; with a ferry{{/ferry}}.\n"
    "{{/alternatives}}\n"
    "{{#n_steps}}\n"
    "\nThe way, step by step:\n"
    "{{/n_steps}}\n"
    "{{#steps}}\n"
    "{{@n}}. {{text}}{{#km}} ({{km}} km){{/km}}{{#m}} ({{m}} m){{/m}}\n"
    "{{/steps}}\n"
    "{{#cut}}\n"
    "... and {{cut}} more steps.\n"
    "{{/cut}}\n"
    "{{#link}}\n"
    "\nOn the map: {{link}}\n"
    "{{/link}}\n"
    "\nRoute: {{source}} on the servers of FOSSGIS; places: Nominatim; " OSM
    ". Times are without traffic.\n"
    "---\n"
    "mode.car = by car\n"
    "mode.bike = by bike\n"
    "mode.foot = on foot\n" HOW_WORDS;

const char MP_ROUTE_BRIEF[] =
    "{{mode}} from {{from.name}}{{#from.how}} (where the user is, "
    "{{from.how}}: say which place, and that it may be off){{/from.how}}"
    "{{#via}} through {{via.name}}{{/via}} to {{to.name}}: {{#km}}{{km}} "
    "km{{/km}}{{#m}}{{m}} m{{/m}}, {{#h}}{{h}} h {{/h}}{{min}} min without "
    "traffic{{#roads}}, by {{roads}}{{/roads}}{{#toll}}, tolls{{/toll}}"
    "{{#ferry}}, a ferry{{/ferry}}.{{#alternatives}} Another way: {{#km}}"
    "{{km}} km{{/km}}, {{#h}}{{h}} h {{/h}}{{min}} min{{#roads}}, by "
    "{{roads}}{{/roads}}.{{/alternatives}} The steps and the map's link are "
    "shown to the user as they are: do not repeat them; answer in a line or "
    "two from these figures, in the user's language.";

/* The means of public transport, as Transitous names them. */
#define MODE_WORDS                                                             \
    "mode.BUS = bus\n"                                                         \
    "mode.COACH = coach\n"                                                     \
    "mode.TRAM = tram\n"                                                       \
    "mode.SUBWAY = underground\n"                                              \
    "mode.METRO = metro\n"                                                     \
    "mode.RAIL = train\n"                                                      \
    "mode.HIGHSPEED_RAIL = high-speed train\n"                                 \
    "mode.LONG_DISTANCE = long-distance train\n"                               \
    "mode.NIGHT_RAIL = night train\n"                                          \
    "mode.REGIONAL_FAST_RAIL = fast regional train\n"                          \
    "mode.REGIONAL_RAIL = regional train\n"                                    \
    "mode.SUBURBAN = suburban train\n"                                         \
    "mode.FERRY = ferry\n"                                                     \
    "mode.AIRPLANE = plane\n"                                                  \
    "mode.FUNICULAR = funicular\n"                                             \
    "mode.AERIAL_LIFT = cable car\n"                                           \
    "mode.CABLE_CAR = cable tram\n"                                            \
    "mode.OTHER = other\n"

const char MP_TRANSIT_LAYOUT[] =
    "By public transport from " FROM " to {{to.name}}"
    "{{#count}}; journeys found: {{count}}{{/count}}{{^count}}: no journey "
    "found"
    "{{/count}}.\n"
    "{{#trips}}\n"
    "\n{{@n}}. Leaves at " DEP ", arrives at " ARR
    " ({{#h}}{{h}} h {{/h}}{{min}} min{{#transfers}}, changes: {{transfers}}"
    "{{/transfers}}{{^transfers}}, direct{{/transfers}}):\n"
    "{{#legs}}\n"
    "{{#walk}}   - walk " TIME "{{#km}}, {{km}} km{{/km}}{{#m}}, {{m}} m{{/m}} "
    "to {{to}}\n"
    "{{/walk}}{{^walk}}   - {{dep.at}} {{mode|mode}} {{line}}{{#agency}} "
    "({{agency}}){{/agency}}{{#headsign}} towards {{headsign}}{{/headsign}} "
    "from {{from}}; {{arr.at}} get off at {{to}}{{#live}}, live times{{/live}}\n"
    "{{/walk}}\n"
    "{{/legs}}\n"
    "{{#cut}}\n"
    "   ... and {{cut}} more legs.\n"
    "{{/cut}}\n"
    "{{/trips}}\n"
    "\nTimes are local ({{tz}}) and from the timetables; those marked as live "
    "times come from the operator in real time. "
    "Source: Transitous (transitous.org), from the operators' open data; "
    "places: Nominatim, " OSM ". Check the operator before you go.\n"
    "---\n" MODE_WORDS DATE_WORDS HOW_WORDS;

const char MP_TRANSIT_BRIEF[] =
    "By public transport from {{from.name}}{{#from.how}} (where the user "
    "is, {{from.how}}: say which place, and that it may be off){{/from.how}} "
    "to {{to.name}}; journeys found: {{count}}. {{#trips}}{{@n}}: {{dep.at}}"
    "{{#dep.date}} (day {{dep.day}}){{/dep.date}} to {{arr.at}}, {{#h}}{{h}} h "
    "{{/h}}{{min}} min, changes: {{transfers}}; {{/trips}}The journeys, leg "
    "by leg, are shown to the user as they are: do not repeat them; answer "
    "in a line or two from these figures, in the user's language.";

/* The kinds of place, as maps_nearby names them (nearby.c). */
#define WHAT_WORDS                                                             \
    "what.fuel = fuel stations\n"                                              \
    "what.charging = charging stations\n"                                      \
    "what.parking = car parks\n"                                               \
    "what.pharmacy = pharmacies\n"                                             \
    "what.hospital = hospitals\n"                                              \
    "what.doctor = doctors and clinics\n"                                      \
    "what.atm = cash machines\n"                                               \
    "what.bank = banks\n"                                                      \
    "what.post_office = post offices\n"                                        \
    "what.police = police\n"                                                   \
    "what.supermarket = supermarkets\n"                                        \
    "what.restaurant = restaurants\n"                                          \
    "what.cafe = cafés\n"                                                      \
    "what.bar = bars and pubs\n"                                               \
    "what.hotel = hotels and guest houses\n"                                   \
    "what.toilets = public toilets\n"                                          \
    "what.train_station = railway stations\n"                                  \
    "what.bus_stop = bus stops\n"                                              \
    "what.museum = museums\n"                                                  \
    "what.attraction = sights\n"

const char MP_NEAR_LAYOUT[] =
    "Within {{radius}} km of " AT ": {{found}} {{what|what}}"
    "{{#count}}; these are the {{count}} nearest:{{/count}}{{^count}}.{{/count}}\n"
    "{{#items}}\n"
    "{{@n}}. {{#name}}{{name}}{{/name}}{{^name}}(no name){{/name}} - " DIST
    "{{#dir}} {{dir|dir}}{{/dir}}{{#address}}; {{address}}{{/address}}"
    "{{#hours}}; open: {{hours}}{{/hours}}{{#phone}}; {{phone}}{{/phone}}\n"
    "{{/items}}\n"
    "\nDistances as the crow flies. Opening hours as mapped, in "
    "OpenStreetMap's notation: they may be out of date. Source: Overpass, "
    "" OSM ".\n"
    "---\n" WHAT_WORDS DIR_WORDS HOW_WORDS;

const char MP_NEAR_BRIEF[] =
    "{{found}} {{what}} within {{radius}} km of {{place.name}}{{#place.how}}"
    " (where the user is, {{place.how}}: say which place, and that it may "
    "be off){{/place.how}}{{#count}}; the nearest: {{#items}}{{name}} "
    "{{#km}}{{km}} km{{/km}}{{#m}}{{m}} m{{/m}}; {{/items}}{{/count}}. The "
    "list is shown to the user as it is: do not repeat it; answer in a line "
    "or two, in the user's language.";

const char MP_FIND_LAYOUT[] = AT ":\n"
                                 "{{#place.address}}\n"
                                 "- address: {{place.address}}\n"
                                 "{{/place.address}}\n"
                                 "- coordinates: {{place.lat}}, {{place.lon}}\n"
                                 "{{#kind}}\n"
                                 "- on the map as: {{kind}}\n"
                                 "{{/kind}}\n"
                                 "- on the map: {{link}}\n"
                                 "\nSource: Nominatim, " OSM ".\n"
                                 "---\n" HOW_WORDS;

const char MP_FIND_BRIEF[] =
    "{{place.name}}{{#place.how}} (where the user is, {{place.how}}: say "
    "which place, and that it may be off){{/place.how}}: {{place.address}} "
    "({{place.lat}}, {{place.lon}}). The address and a link to the map are "
    "shown to the user as they are: do not repeat them; answer in a line, "
    "in the user's language.";
