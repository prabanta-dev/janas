/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layouts.c - the layouts of janas-weather's answers (see weather.h and
 * services/common/template.h): the report the user reads, filled in English
 * here and in the user's language by a client that translates the layout once
 * (janas-chat), and the brief, the figures the model reads to answer.
 */
#include "weather.h"

/* The sky, as the WMO codes Open-Meteo gives (words.c). */
#define WMO_WORDS                                                              \
    "wmo.0 = clear sky\n"                                                      \
    "wmo.1 = mainly clear\n"                                                   \
    "wmo.2 = partly cloudy\n"                                                  \
    "wmo.3 = overcast\n"                                                       \
    "wmo.45 = fog\n"                                                           \
    "wmo.48 = fog depositing rime\n"                                           \
    "wmo.51 = light drizzle\n"                                                 \
    "wmo.53 = drizzle\n"                                                       \
    "wmo.55 = dense drizzle\n"                                                 \
    "wmo.56 = light freezing drizzle\n"                                        \
    "wmo.57 = dense freezing drizzle\n"                                        \
    "wmo.61 = light rain\n"                                                    \
    "wmo.63 = rain\n"                                                          \
    "wmo.65 = heavy rain\n"                                                    \
    "wmo.66 = light freezing rain\n"                                           \
    "wmo.67 = heavy freezing rain\n"                                           \
    "wmo.71 = light snow\n"                                                    \
    "wmo.73 = snow\n"                                                          \
    "wmo.75 = heavy snow\n"                                                    \
    "wmo.77 = snow grains\n"                                                   \
    "wmo.80 = light rain showers\n"                                            \
    "wmo.81 = rain showers\n"                                                  \
    "wmo.82 = violent rain showers\n"                                          \
    "wmo.85 = light snow showers\n"                                            \
    "wmo.86 = heavy snow showers\n"                                            \
    "wmo.95 = thunderstorm\n"                                                  \
    "wmo.96 = thunderstorm with light hail\n"                                  \
    "wmo.99 = thunderstorm with heavy hail\n"

/* The wind's force, and where it blows from. */
#define WIND_WORDS                                                             \
    "bft.0 = calm\n"                                                           \
    "bft.1 = light air\n"                                                      \
    "bft.2 = light breeze\n"                                                   \
    "bft.3 = gentle breeze\n"                                                  \
    "bft.4 = moderate breeze\n"                                                \
    "bft.5 = fresh breeze\n"                                                   \
    "bft.6 = strong breeze\n"                                                  \
    "bft.7 = near gale\n"                                                      \
    "bft.8 = gale\n"                                                           \
    "bft.9 = strong gale\n"                                                    \
    "bft.10 = storm\n"                                                         \
    "bft.11 = violent storm\n"                                                 \
    "bft.12 = hurricane force\n"                                               \
    "from.N = from the north\n"                                                \
    "from.NNE = from the north-north-east\n"                                   \
    "from.NE = from the north-east\n"                                          \
    "from.ENE = from the east-north-east\n"                                    \
    "from.E = from the east\n"                                                 \
    "from.ESE = from the east-south-east\n"                                    \
    "from.SE = from the south-east\n"                                          \
    "from.SSE = from the south-south-east\n"                                   \
    "from.S = from the south\n"                                                \
    "from.SSW = from the south-south-west\n"                                   \
    "from.SW = from the south-west\n"                                          \
    "from.WSW = from the west-south-west\n"                                    \
    "from.W = from the west\n"                                                 \
    "from.WNW = from the west-north-west\n"                                    \
    "from.NW = from the north-west\n"                                          \
    "from.NNW = from the north-north-west\n"

/* The compass, for where a station is. */
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

/* A station's report: the weather as coded, the clouds. */
#define STATION_WORDS                                                          \
    "wx.- = light\n"                                                           \
    "wx.+ = heavy\n"                                                           \
    "wx.TS = thunderstorm\n"                                                   \
    "wx.SH = showers of\n"                                                     \
    "wx.FZ = freezing\n"                                                       \
    "wx.RA = rain\n"                                                           \
    "wx.DZ = drizzle\n"                                                        \
    "wx.SN = snow\n"                                                           \
    "wx.GR = hail\n"                                                           \
    "wx.GS = small hail\n"                                                     \
    "wx.FG = fog\n"                                                            \
    "wx.BR = mist\n"                                                           \
    "wx.HZ = haze\n"                                                           \
    "wx.VC = nearby\n"                                                         \
    "cover.FEW = few clouds\n"                                                 \
    "cover.SCT = scattered clouds\n"                                           \
    "cover.BKN = broken clouds\n"                                              \
    "cover.OVC = overcast\n"                                                   \
    "cover.VV = sky obscured\n"                                                \
    "cover.CLR = clear\n"                                                      \
    "cover.SKC = clear\n"                                                      \
    "cover.NSC = no significant cloud\n"                                       \
    "cover.NCD = no cloud detected\n"                                          \
    "cover.CAVOK = no significant cloud\n"

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

#define PLACE "{{place}}{{#how}} - where the user is, {{how|how}}{{/how}}"

/* The model's cell, and the source. */
#define MODEL                                                                  \
    "Forecast: {{model.source}}'s model, at the point {{model.lat}}, "         \
    "{{model.lon}}{{#model.elev}} ({{model.elev}} m){{/model.elev}}, "         \
    "{{model.km}} km from the place: a model's value, not a measure.\n"

const char WX_NOW_LAYOUT[] =
    "The weather now in " PLACE ", at {{time}} {{tz}} (local time):\n"
    "- sky: {{sky|wmo}}{{sky_text}}{{#cloud}} (cloud cover {{cloud}}%)"
    "{{/cloud}}\n"
    "- temperature {{temp}} °C{{#feels}}, feels like {{feels}} °C{{/feels}}"
    "{{#humidity}}; humidity {{humidity}}%{{/humidity}}\n"
    "- wind {{#wind_from}}{{wind_from|from}} ({{wind_deg}}°) {{/wind_from}}"
    "at {{wind}} km/h ({{bft|bft}}){{#gust}}, gusts up to {{gust}} km/h"
    "{{/gust}}\n"
    "{{#has_precip}}\n"
    "- rain in the last quarter of an hour: {{precip}} mm\n"
    "{{/has_precip}}\n"
    "{{#pressure}}\n"
    "- pressure at sea level {{pressure}} hPa\n"
    "{{/pressure}}\n"
    "{{#today}}\n"
    "- today: lowest {{today.tmin}} °C, highest {{today.tmax}} °C"
    "{{#today.has_prob}}, chance of rain {{today.rain_prob}}%"
    "{{/today.has_prob}}{{#today.sunrise}}; sunrise {{today.sunrise}}, "
    "sunset {{today.sunset}}{{/today.sunrise}}\n"
    "{{/today}}\n" MODEL "{{#station}}\n"
    "Measured by the station of {{station.name}} ({{station.icao}}), "
    "{{station.km}} km {{station.dir|dir}} of the place, at "
    "{{station.time}} ({{station.ago}} minutes ago): {{station.temp}} °C, "
    "dew point {{station.dew}} °C{{#station.calm}}, calm{{/station.calm}}"
    "{{#station.variable}}, wind of variable direction at {{station.kt}} kt "
    "({{station.kmh}} km/h){{/station.variable}}{{#station.from}}, wind "
    "{{station.from|from}} at {{station.kt}} kt ({{station.kmh}} km/h, "
    "{{station.bft|bft}}){{/station.from}}{{#station.gust_kt}}, gusts "
    "{{station.gust_kt}} kt ({{station.gust_kmh}} km/h){{/station.gust_kt}}"
    "{{#station.vis_km}}, visibility {{station.vis_km}} km"
    "{{#station.vis_more}} or more{{/station.vis_more}}{{/station.vis_km}}"
    "{{#station.wx}}, {{#station.wx_parts}}{{.|wx}} {{/station.wx_parts}}"
    "({{station.wx}}){{/station.wx}}{{^station.wx}}, no rain or other "
    "weather{{/station.wx}}{{#station.clouds}}, {{cover|cover}}{{#ft}} at "
    "{{ft}} ft ({{m}} m){{/ft}}{{/station.clouds}}{{#station.pressure}}, "
    "pressure {{station.pressure}} hPa{{/station.pressure}}. As reported: "
    "{{station.raw}}\n"
    "{{/station}}\n"
    "{{#temp_diff}}\n"
    "The model's temperature and the station's differ by {{temp_diff}} °C: "
    "the station's is the measure (the station is {{station.km}} km away).\n"
    "{{/temp_diff}}\n"
    "{{#rain_disagree}}\n"
    "The model and the station disagree on rain: the station "
    "{{#station_rain}}reports it{{/station_rain}}{{^station_rain}}reports "
    "none{{/station_rain}}, the model {{#model_rain}}has it{{/model_rain}}"
    "{{^model_rain}}has none{{/model_rain}}.\n"
    "{{/rain_disagree}}\n"
    "{{#no_station}}\n"
    "No weather station reported within {{no_station}} km.\n"
    "{{/no_station}}\n"
    "{{#station_error}}\n"
    "The stations could not be asked ({{station_error}}).\n"
    "{{/station_error}}\n"
    "Sources: {{model.source}} ({{model.url}}, CC BY 4.0){{#station}}; the "
    "station's METAR from aviationweather.gov (NOAA){{/station}}.\n"
    "---\n" WMO_WORDS WIND_WORDS DIR_WORDS STATION_WORDS HOW_WORDS;

const char WX_NOW_BRIEF[] =
    "The weather now in {{place}} at {{time}} {{tz}}{{#how}} (where the "
    "user is, {{how}}: say which place, and that it may be off){{/how}}: "
    "{{sky|wmo}}{{sky_text}}, {{temp}} °C{{#feels}} (feels {{feels}})"
    "{{/feels}}, wind {{wind}} km/h{{#wind_from}} {{wind_from|from}}"
    "{{/wind_from}}{{#gust}}, gusts {{gust}}{{/gust}}{{#today}}; today "
    "{{today.tmin}} to {{today.tmax}} °C{{#today.has_prob}}, chance of "
    "rain {{today.rain_prob}}%{{/today.has_prob}}{{/today}}{{#station}}; "
    "the station of {{station.name}}, {{station.km}} km away, measured "
    "{{station.temp}} °C{{/station}}.{{#temp_diff}} The model and the "
    "station differ by {{temp_diff}} °C: the station's is the "
    "measure.{{/temp_diff}}{{#rain_disagree}} They disagree on "
    "rain.{{/rain_disagree}} The full report is shown to the user as it "
    "is: do not repeat it; answer in a line or two from these figures, in "
    "the user's language.\n"
    "---\n" WMO_WORDS WIND_WORDS;

#define DATE "{{wd|wd}} {{day}} {{mon|mon}}"

const char WX_FORECAST_LAYOUT[] =
    "Forecast for " PLACE ", by the day (times in local time, {{tz}}):\n"
    "{{#days}}\n"
    "- " DATE ": {{sky|wmo}}{{sky_text}}; {{tmin}} to {{tmax}} °C"
    "{{#no_rain}}; no rain{{/no_rain}}{{#rain}}; rain {{rain}} mm{{/rain}}"
    "{{#has_prob}} (chance {{rain_prob}}%){{/has_prob}}{{#wind}}; wind up "
    "to {{wind}} km/h ({{bft|bft}}){{#wind_from}} {{wind_from|from}}"
    "{{/wind_from}}{{/wind}}{{#gust}}, gusts up to {{gust}} km/h{{/gust}}"
    "{{#uv}}; UV index {{uv}}{{/uv}}{{#sunrise}}; sunrise {{sunrise}}, "
    "sunset {{sunset}}{{/sunrise}}\n"
    "{{/days}}\n"
    "{{#fewer}}\n"
    "({{model.source}} gives {{fewer}} days.)\n"
    "{{/fewer}}\n"
    "{{#hours}}\n"
    "{{#newday}}\n"
    "{{newday.wd|wd}} {{newday.day}} {{newday.mon|mon}}\n"
    "{{/newday}}\n"
    "- {{time}}: {{sky|wmo}}{{sky_text}}, {{temp}} °C{{#rain}}, rain "
    "{{rain}} mm{{/rain}}{{#has_prob}} (chance {{rain_prob}}%){{/has_prob}}"
    ", wind {{wind}} km/h{{#wind_from}} {{wind_from|from}}{{/wind_from}}"
    "{{#gust}}, gusts {{gust}}{{/gust}}\n"
    "{{/hours}}\n" MODEL "Source: {{model.source}} ({{model.url}}, CC BY "
    "4.0).\n"
    "---\n" WMO_WORDS WIND_WORDS DATE_WORDS HOW_WORDS;

const char WX_FORECAST_BRIEF[] =
    "The forecast for {{place}}{{#how}} (where the user is, {{how}}: say "
    "which place, and that it may be off){{/how}}, by the day: "
    "{{#days}}" DATE ": {{sky|wmo}}{{sky_text}}, {{tmin}} to {{tmax}} °C"
    "{{#has_prob}}, rain {{rain_prob}}%{{/has_prob}}{{#rain}} ({{rain}} "
    "mm){{/rain}}{{#wind}}, wind {{wind}} km/h{{/wind}}; {{/days}}The full "
    "forecast, hour by hour too, is shown to the user as it is: do not "
    "repeat it; answer in a line or two from these figures, in the user's "
    "language.\n"
    "---\n" WMO_WORDS DATE_WORDS;

/* The sea's state, on the Douglas scale (words.c). */
#define SEA_WORDS                                                              \
    "sea.1 = calm\n"                                                           \
    "sea.2 = smooth\n"                                                         \
    "sea.3 = slight\n"                                                         \
    "sea.4 = moderate\n"                                                       \
    "sea.5 = rough\n"                                                          \
    "sea.6 = very rough\n"                                                     \
    "sea.7 = high\n"                                                           \
    "sea.8 = very high\n"                                                      \
    "sea.9 = phenomenal\n"

#define WHERE                                                                  \
    "{{#where.sea}}over the {{where.sea}}{{#where.and}}, between "             \
    "{{where.region}} and {{where.and}}{{/where.and}}{{/where.sea}}"           \
    "{{^where.sea}}over {{where.region}}{{/where.sea}}; {{#where.km}}"         \
    "{{where.km}} km {{where.dir|dir}} of {{where.city}}{{/where.km}}"         \
    "{{^where.km}}over {{where.city}}{{/where.km}}"

const char WX_SEA_LAYOUT[] =
    "{{#sea}}The state of the {{sea}} at {{time}} local time ({{tz}}), and "
    "the next {{days}} days:{{/sea}}{{^sea}}The state of the sea off " PLACE
    " at {{time}} local time ({{tz}}), and the next {{days}} days:{{/sea}}\n"
    "{{#summary}}\n"
    "Now, at {{summary.points}} points: waves {{summary.lo}} to "
    "{{summary.hi}} m ({{summary.lo_code|sea}}{{#summary.hi_other}} to "
    "{{summary.hi_code|sea}}{{/summary.hi_other}}){{#summary.t_lo}}, water "
    "{{summary.t_lo}} to {{summary.t_hi}} °C{{/summary.t_lo}}.\n"
    "{{#summary.days}}\n"
    "{{wd|wd}} {{day}} {{mon|mon}}: highest waves {{max}} m ({{code|sea}}), " WHERE
    ".\n"
    "{{/summary.days}}\n"
    "{{/summary}}\n"
    "{{#points}}\n"
    "- {{#where}}" WHERE ": {{/where}}waves {{wave}} m ({{wave_code|sea}})"
    "{{#wave_from}} {{wave_from|from}}{{/wave_from}}{{#period}}, period "
    "{{period}} s{{/period}}{{#has_wind_wave}}; wind waves {{wind_wave}} m"
    "{{/has_wind_wave}}{{#has_swell}}, swell {{swell}} m{{#swell_from}} "
    "{{swell_from|from}}{{/swell_from}}{{/has_swell}}{{#sst}}; water "
    "{{sst}} °C{{/sst}}{{#has_wind}}; wind {{wind}} km/h ({{bft|bft}})"
    "{{#wind_from}} {{wind_from|from}}{{/wind_from}}{{#gust}}, gusts "
    "{{gust}} km/h{{/gust}}{{/has_wind}}\n"
    "{{#days}}\n"
    "    {{wd|wd}} {{day}} {{mon|mon}}: waves up to {{max}} m ({{code|sea}})"
    "{{#from}} {{from|from}}{{/from}}{{#has_wind}}, wind up to {{wind}} "
    "km/h ({{bft|bft}}){{#gust}}, gusts {{gust}}{{/gust}}{{/has_wind}}\n"
    "{{/days}}\n"
    "{{/points}}\n"
    "The waves' height is the significant height (the mean of the highest "
    "third); single waves can be almost twice as high. Source: Open-Meteo's "
    "sea and weather models (open-meteo.com, CC BY 4.0): forecasts, not "
    "measures.\n"
    "---\n" SEA_WORDS WIND_WORDS DIR_WORDS DATE_WORDS HOW_WORDS;

const char WX_SEA_BRIEF[] =
    "The sea {{#sea}}of the {{sea}}{{/sea}}{{^sea}}off {{place}}{{#how}} "
    "(where the user is, {{how}}: say which place, and that it may be off)"
    "{{/how}}{{/sea}} at {{time}}: {{#summary}}waves {{summary.lo}} to "
    "{{summary.hi}} m ({{summary.lo_code|sea}} to {{summary.hi_code|sea}})"
    "{{#summary.t_lo}}, water {{summary.t_lo}} to {{summary.t_hi}} "
    "°C{{/summary.t_lo}}; highest by the day: {{#summary.days}}{{wd|wd}} "
    "{{max}} m ({{code|sea}}); {{/summary.days}}{{/summary}}{{^summary}}"
    "{{#points}}waves {{wave}} m ({{wave_code|sea}}){{#sst}}, water {{sst}} "
    "°C{{/sst}}{{#has_wind}}, wind {{wind}} km/h{{/has_wind}}; by the day: "
    "{{#days}}{{wd|wd}} up to {{max}} m ({{code|sea}}); {{/days}}"
    "{{/points}}{{/summary}}The full report is shown to the user as it is: "
    "do not repeat it; answer in a line or two from these figures, in the "
    "user's language.\n"
    "---\n" SEA_WORDS DATE_WORDS;
