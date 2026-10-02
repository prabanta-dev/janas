# janas-maps

Roads and places for `janas-chat` or any other MCP client (see [the services](README.md)): a route by car, bike or on foot, journeys by public transport, what is near a place, where a place or an address is. OpenStreetMap's data through free services: no key, no account.

## The tools

- **`maps_route`** - the route between two places, or through a third (`via`), by car (the default), bike or on foot (`mode`), avoiding tolls, motorways or ferries when asked (`avoid`): its length, its time without traffic, the roads it mostly takes, whether it has tolls or a ferry, up to two other ways, the way step by step, and a link to the route on openstreetmap.org. The steps are written by the router in the computer's language (Italian with `LANG=it_IT.UTF-8`; English for a language it does not know). When Valhalla does not answer, OSRM gives the route without its steps.
- **`maps_transit`** - journeys by public transport between two places, leaving now or at a time (`when`: `HH:MM` or `YYYY-MM-DD HH:MM`, the start's local time; a time of the day already gone is tomorrow's), or arriving by it (`arrive_by`): up to five, each with its times, length and changes, and leg by leg the line, its operator, the stops and the walks between them.
- **`maps_nearby`** - the places of a kind within a radius of a place (2 km unless asked, up to 10), nearest first, at most ten, with how far and which way, their address, opening hours and phone when mapped. The kinds are a list, so that the model picks one and never writes a query: fuel, charging, parking, pharmacy, hospital, doctor, atm, bank, post_office, police, supermarket, restaurant, cafe, bar, hotel, toilets, train_station, bus_stop, museum, attraction.
- **`maps_find`** - where a place or an address is: its full address, its coordinates, what it is on the map, and a link to it; for a point, what is there.

A place is a name or an address, best with its town ("Via Roma 10, Trapani", "Stazione Centrale, Palermo", "Colosseo"), or `lat,lon`. The second place of a route is looked for near the first, so that "Stazione Centrale" on the way from Trapani is Palermo's. With no start named, the start is where the user is, as for the weather ([details](weather.md#when-no-place-is-named)), and the answer says so.

## The answers

Every answer is data with a layout (see [the services](README.md)): `janas-chat` shows the user the whole of it in the user's language - the steps, the journeys leg by leg, the list of places - and gives the model only a brief of the figures, from which it answers in a line or two. A route of a hundred steps costs the model the same as one of ten.

## Where the data comes from

| Data | Source | Key |
|---|---|---|
| places and addresses | [Nominatim](https://nominatim.org) (OpenStreetMap Foundation), [Photon](https://photon.komoot.io) (komoot) when Nominatim does not answer | none |
| routes | [Valhalla](https://github.com/valhalla/valhalla) and [OSRM](https://project-osrm.org) on the servers of [FOSSGIS](https://fossgis.de/arbeitsgruppen/osm-server/nutzungsbedingungen/) | none |
| public transport | [Transitous](https://transitous.org), from the operators' open timetables | none |
| what is near | [Overpass API](https://overpass-api.de), [Private.coffee's instance](https://overpass.private.coffee) when the first is busy | none |
| where the user is | [GeoJS](https://www.geojs.io), [ipwho.is](https://ipwho.is) | none |

- **The data is OpenStreetMap's**, © OpenStreetMap contributors, under the [Open Database License](https://www.openstreetmap.org/copyright): every answer says so.
- **The services are run by volunteers for moderate use**, and each asks for the same: a User-Agent naming the program and where it lives (`janas-maps/<version> (+https://github.com/prabanta-dev/janas)`) and at most one request a second; `janas-maps` waits as long as it must between two requests to a server, and keeps the answers a while (places a week, routes ten minutes, journeys two minutes, places around an hour). Transitous asks to be used by open-source, non-commercial projects, as Janas is.
- **Each server can be changed**, as FOSSGIS asks of programs, with `JANAS_NOMINATIM_URL`, `JANAS_PHOTON_URL`, `JANAS_VALHALLA_URL`, `JANAS_OSRM_URL`, `JANAS_TRANSITOUS_URL`, `JANAS_OVERPASS_URL` and `JANAS_OVERPASS_URL2`: an instance of your own, for instance.
- **Times by road are without traffic**: no free source has it.
- **Long routes**: FOSSGIS' Valhalla routes up to 1,500 km, and its OSRM knows Europe only; longer routes, or routes off Europe's roads when Valhalla does not answer, are said to be unavailable rather than made up.
- **Public transport is as good as the open timetables**: Transitous has those the operators publish, the trains of Trenitalia among them, not every bus company's. Live times are marked when there are any; check the operator before you go.
- **What is near is what is mapped**: a pharmacy may be named after its pharmacist, and opening hours are in OpenStreetMap's notation, sometimes old. Distances are as the crow flies. Overpass is at times too busy to answer: the second instance is asked, and when neither answers the answer says to try again in a minute.

## Running it

```sh
janas-maps
```

It speaks MCP over its standard input and output and writes what it does on its standard error. In other clients:

```sh
claude mcp add maps -- /path/to/janas-maps
```

```json
{"mcpServers": {"maps": {"command": "/path/to/janas-maps"}}}
```

## What has been run

2 October 2026, over the protocol, in the release and AddressSanitizer builds, with `LANG=it_IT.UTF-8`:

- Trapani to Palermo by car: 110 km, 1 h 37 min, by the A29, with two other ways and the steps in Italian; through Alcamo, avoiding tolls; to Erice on foot; to Marsala by bike; from "here" (Palermo, by GeoJS) to Erice. Trapani to New York: no route, with the reasons (Valhalla's limit, OSRM's map).
- Trapani to Palermo's central station by public transport at 9:00: five journeys, the first a regional train to Marsala and a coach to Palermo; Milan's central station to Rome Termini arriving by 8:00 on 5 October.
- Pharmacies within 1 km of Piazza Vittorio Emanuele in Trapani, cash machines within 500 m of the Colosseum, fuel near "here".
- The Colosseum, a point beside it, and a place that does not exist.

In `janas-chat` with Qwen3.6-35B-A3B, the same day, in Italian: "Come arrivo a Palermo da Trapani?" (the route by car, the layout written in Italian once), "E in treno?" (five journeys, leg by leg), "C'è una farmacia vicino a me?" (where the user is, by GeoJS: 22 pharmacies within 2 km). The first run showed three faults, mended and run again: blank lines the model had added between the legs, "14:53 alle PIRAINETO" for an arrival (the English said "at" for both a time and a place), and a place taken from the conversation for "near me". The second instance of Overpass was asked when the first was made to fail; on 2 October it did not answer in time.
