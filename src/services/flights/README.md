# janas-flights

Where a flight is and what it is doing, its scheduled and actual times, the flights of a route, what flies over a sea or near a place - for `janas-chat` or any other MCP client (see [the services](../README.md)). The live data needs no key and no account; the schedules need a free key of your own.

## The tools

- **`flight_status`** - a flight by its number (`VY1595`, `AZ1631`) or its callsign (`VLG1595`). With a key: scheduled, estimated and actual departure and arrival, in each airport's local time and in UTC, delay, terminal, gate and baggage belt. On the ground: parked, taxiing or on the runway, and at which airport. In the air: altitude, speed, climbing or descending, where it is in words ("over the Tyrrhenian Sea, between Sicily, Italy and Campania, Italy; 86 km NNE of Palermo"), its route, and an arrival time estimated from its speed, next to the airline's.
- **`flights_between`** - the flights from one airport to another, today's and the latest ones, each with its times and, if in the air, where it is now. Needs the key.
- **`flights_over`** - how many and which aircraft fly now over a sea, by its name in English or Italian ("Tyrrhenian Sea", "Tirreno", "mar Ligure"), one line each: callsign, aircraft, altitude, speed, heading, where, and its route when one on record fits its position.
- **`flights_nearby`** - what flies now near an airport (`TRN`, `LICT`, "Birgi"), a city ("Turin") or a point, nearest first; with none of them, near where the user is (`JANAS_LOCATION`, or the internet connection's position: [as for the weather](../weather/README.md#when-no-place-is-named)).

Airports are found by IATA or ICAO code, by city, or by the names people use ("Trapani-Birgi", "Birgi", "Fiumicino").

## Where the data comes from

| Data | Source | Key |
|---|---|---|
| positions, altitude, speed | [adsb.lol](https://adsb.lol) (Open Database License), [adsb.fi](https://adsb.fi) when it does not answer | none |
| a callsign's route, without a key | [adsbdb.com](https://www.adsbdb.com) | none |
| schedules and times | [AviationStack](https://aviationstack.com) | yours, free: 100 calls a month |
| airports, cities, seas | [OurAirports](https://ourairports.com) (public domain), [GeoNames](https://www.geonames.org) (CC BY 4.0), [Natural Earth](https://www.naturalearthdata.com) (public domain), built into the program by `tools/gen_geo` | - |

The lists - over a sea, near a place - use open data only; AviationStack is asked only about one flight (`flight_status`) or one route (`flights_between`), and the service tells the model not to call those for each aircraft of a list.

The routes on record at adsbdb are often wrong for airlines that reuse their callsigns (Ryanair, Wizz Air): a route that does not fit the aircraft's position is reported as probably wrong, or left out of a list.

## The key

Register at [aviationstack.com](https://aviationstack.com) for the free plan: 100 calls a month, today's flights only. Every question about a flight number or a route is one call; the answers are kept for ten minutes. Put the key in `~/.config/janas/flights.conf`:

```
aviationstack_key = YOUR_KEY
```

or in another file named by `--config FILE` or `JANAS_FLIGHTS_CONFIG`, or in the variable `JANAS_AVIATIONSTACK_KEY` (which wins). Without a key the service runs on open data only; `flight_status` then tells where a flight is, not its times.

## What it does with AviationStack's answers

- The times come in each airport's local time though marked `+00:00`: they are read in the airport's zone.
- AviationStack's status of a flight lags (a flight still "active" an hour after landing, another "active" an hour before leaving): the state reported comes from the actual times.
- An aircraft flies several flights a day: a live position is given to a flight only when the aircraft with its transponder is on its route and heading for its destination, or on the ground at the right airport.
- A question about a number brings its codeshares too (AZ1731 and Volotea's V76064, the same flight): the number asked is the one answered, and in a list the others show as "also sold as".

## Running it

```sh
janas-flights [--config FILE]
```

It speaks MCP over its standard input and output (both eras of the protocol, as `janas-mcp`) and writes what it does on its standard error. In other clients:

```sh
claude mcp add flights -- /path/to/janas-flights
claude mcp add flights -e JANAS_AVIATIONSTACK_KEY=YOUR_KEY -- /path/to/janas-flights
```

```json
{"mcpServers": {"flights": {"command": "/path/to/janas-flights"}}}
```

## What has been run

1 October 2026:

- Over the protocol, with a key: ITA Airways' AZ1709 from Fiumicino to Catania, departed at 08:19 with 19 minutes of delay, seen over the Tyrrhenian Sea with the airline's estimate (09:11) and the one from its speed (09:08); the 21 flights from Fiumicino to Catania; Turin-Trapani, whose last flight on record was the evening before.
- Without a key: `flights_over` the Tyrrhenian Sea at 09:07 found 45 aircraft in the air, from a business jet north of Bastia to a Ryanair descending towards Calabria.
- `janas-chat` with Qwen3-Next-80B-A3B asked in Italian where Vueling's VY1595 from Cairo to Barcelona was and what flew near Fiumicino: it called `flight_status` and `flights_nearby` by itself and told what they returned (configured by hand, before the chat started the services by itself).
- `janas-chat` finding and starting the service beside it, with its tools and instructions, checked without a model. In Claude Code and the other clients it has not been run yet.
