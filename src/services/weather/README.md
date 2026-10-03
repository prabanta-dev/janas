# janas-weather

The weather now and in the next days, the sea and its waves, and the weather warnings in force - for `janas-chat` or any other MCP client (see [the services](../README.md)). Free sources only: no key, no account.

## The tools

- **`weather_now`** - the weather now at a place: sky, temperature and how it feels, humidity, wind and gusts (with their name on the Beaufort scale), rain, pressure, today's lowest and highest, sunrise and sunset, from a forecast model; and next to it what the nearest weather station (an airport's, within 40 km) measured, with when and how far away. When the model and the station disagree - by 3 °C or more, or on rain - the answer says so, and says which is the measure.
- **`weather_forecast`** - day by day up to 16 days (sky, lowest and highest, rain and its chance, wind and gusts, UV index, sunrise and sunset), and the next 24 hours every three hours, or the next 48 every hour.
- **`weather_sea`** - the state of the sea, now and up to 8 days ahead: waves with their name on the Douglas scale (calm, smooth, slight, moderate, rough...), wind waves and swell, the water's temperature and the wind. For a whole sea by its name in English or Italian ("Tyrrhenian Sea", "Tirreno", "Adriatico", "mar Ligure"), at up to nine points across it with a summary on top, or for the sea off a coastal place.
- **`weather_alerts`** - the warnings in force for a place, from the national weather services gathered by MeteoAlarm (wind, thunderstorms, heat, snow; yellow, orange, red), the place's region first and the expired and green ones left out; and in Italy the Civil Protection's official levels for the zone of alert the place is in, for floods, landslides and thunderstorms, today and tomorrow.

A place is a name in Italian or English ("Trapani", "Firenze", "Florence"), with its region or country when several share it ("Trapani, Sicilia"), an airport's code (`TRN`, `LICT`), or a latitude and longitude. A point is put into words (the town it is in or near, or the sea it is over: "the Tyrrhenian Sea, between Lazio, Italy and Sicily, Italy; 191 km SSW of Anzio"). Times are given in the place's local time.

## When no place is named

"What's the weather like?" names no place: the service then takes where the user is, and says how it knows it, so that the model can tell the user which place that is and that it may be off. In this order:

1. **`JANAS_LOCATION`**, when set in the environment: a city of the built-in tables by its English name (`Trapani`), or `lat,lon` (`38.02,12.51`); `off` for no guess at all - the model is then told to ask.
2. **The internet connection's position**, from [GeoJS](https://www.geojs.io), or [ipwho.is](https://ipwho.is) when it does not answer: free and with no key, but the address seen is the connection's, so the town can be the provider's rather than yours (a town nearby, more often than not). Asked once an hour at most.
3. **The city of the computer's time zone** (`Europe/Rome`: Rome), a rough guess, when neither answers.

The same applies to `janas-flights`' `flights_nearby`; the code is shared (`src/services/common/locate.c`), for the services to come. `janas-chat` passes its environment to the services it starts, so `JANAS_LOCATION` set for the chat reaches them.

## Where the data comes from

| Data | Source | Key |
|---|---|---|
| forecasts, the weather now, places by name | [Open-Meteo](https://open-meteo.com) (CC BY 4.0; free for non-commercial use) | none |
| the same, when Open-Meteo does not answer | [MET Norway](https://api.met.no) (CC BY 4.0) | none |
| the sea | Open-Meteo's marine models | none |
| what a station measured | the airports' METAR, from [aviationweather.gov](https://aviationweather.gov) (NOAA) | none |
| warnings | [MeteoAlarm](https://meteoalarm.org) (EUMETNET) | none |
| Italy's alert levels | the [Dipartimento della Protezione Civile](https://github.com/pcm-dpc/DPC-Bollettini-Criticita-Idrogeologica-Idraulica)'s bulletin (CC BY 4.0) | none |
| where the user is | [GeoJS](https://www.geojs.io), [ipwho.is](https://ipwho.is) | none |
| airports, cities, seas | the tables `janas-flights` uses: OurAirports, GeoNames, Natural Earth | - |

A forecast is a model's value and a station's report a measure: the answers say which is which, and where the model's cell lies. The answers are kept for a few minutes (forecasts and warnings 15, a station's report 10, the Civil Protection's bulletin an hour), so a conversation that comes back to a place does not ask again.

What the service does with its sources:

- **MeteoAlarm's Italian warnings** are the Air Force weather service's, about the phenomena; the official alerts are the Civil Protection's: the answer gives both and says so.
- **The Civil Protection's bulletin** comes out around 16:00 for today and tomorrow: until then, yesterday's "tomorrow" is today's. The zone of alert is found by the municipality's name, or, when the bulletin does not name it (an airport, a point), by the point in the zones' outlines.
- **Names in the country's language.** MeteoAlarm and the Civil Protection write regions and municipalities in Italian (Piemonte, Torino); a place found in the built-in tables, in English, takes its names from Open-Meteo's geocoding.
- **MET Norway** asks for a User-Agent naming the program: requests carry `janas-weather/<version> https://github.com/prabanta-dev/janas`.

## Running it

```sh
janas-weather
```

It speaks MCP over its standard input and output (both eras of the protocol, as `janas-mcp`) and writes what it does on its standard error. In other clients:

```sh
claude mcp add weather -- /path/to/janas-weather
claude mcp add weather -e JANAS_LOCATION=Trapani -- /path/to/janas-weather
```

```json
{"mcpServers": {"weather": {"command": "/path/to/janas-weather"}}}
```

## What has been run

1 October 2026:

- Over the protocol, every tool, in the release and AddressSanitizer builds: the weather now in Trapani, with the station of Trapani-Birgi 13 km away; the forecast for Florence hour by hour and for Bolzano for three days; the Tyrrhenian Sea at nine points and the sea off Trapani; the warnings for Trapani (none from MeteoAlarm, the Civil Protection's levels for its zone), for a point in Turin and in Florence (found in Piemonte and Toscana, and in their zones of alert by the point), for the airport `LICT`, and for Oslo (three gale warnings elsewhere in Norway, given in Oslo's time though the feed is in UTC); the weather with no place, placed in Palermo by GeoJS. Once, Open-Meteo did not answer and MET Norway did.
- `janas-chat` with Qwen3.6-35B-A3B, asked in Italian about the weather in Trapani, the Tyrrhenian Sea today and tomorrow, warnings in Genoa, the next three days in Bolzano, and "Che tempo fa?": it called the tools by itself and told what they returned, the last answer as "where you are estimated to be from the internet connection".
- `janas-chat` starting the service beside it. In Claude Code and the other clients it has not been run yet.
