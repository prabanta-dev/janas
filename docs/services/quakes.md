# janas-quakes

The earthquakes for `janas-chat` or any other MCP client (see [the services](README.md)): the latest near a place or where you are, the strongest in the world, one in detail. It only reads, from open sources, without a key.

## The tools

- **`quakes_near`** - the earthquakes around a place (`place`: a town, a region, or a sea, whose earthquakes are those within it; where you are unless asked), within a radius (`radius`, 300 km unless asked, 1,000 at most), of the last days (`days`, 7 unless asked, 30 at most), of a magnitude or more (`min_mag`, 2.5 unless asked): the latest first, each with its time (and how long ago, when it was today), its magnitude, its depth, its distance and way from the place, and where the source says it was.
- **`quakes_strong`** - the strongest earthquakes in the world of the last days (`days`, 7), of a magnitude or more (`min_mag`, 5), the strongest first, each with the impact the USGS expects (PAGER's alert: green, yellow, orange, red) and whether it was at sea and strong enough for a tsunami.
- **`quakes_event`** - one earthquake in detail: by its id (each answer gives them, in brackets, to the model), or the latest near a place, or with neither the latest near you ("the one just felt"): its magnitude and its kind, the time, where it was (a sea or a region, the nearest town), its depth, its distance from the place asked; from the USGS, how many people told it they felt it, the strongest shaking estimated (on the Mercalli scale), the impact; the link to the source's page of it.

## Where the data comes from

| Data | Source | Where | Licence |
|---|---|---|---|
| earthquakes in Italy and the seas around it (35-48° N, 6-19° E, and the seas that reach into it) | [INGV](https://terremoti.ingv.it), Istituto Nazionale di Geofisica e Vulcanologia, through its [FDSN event service](https://webservices.ingv.it) | the Italian network's; the strongest of the Mediterranean and of the world too | CC BY 4.0 |
| earthquakes everywhere else, the strongest in the world | the [USGS](https://earthquake.usgs.gov/fdsnws/event/1/), U.S. Geological Survey | the world; not every small one (in the Aegean, on 3 October, it gave none of magnitude 3 or more in a week) | public domain |
| places | [Nominatim](https://nominatim.org) (OpenStreetMap), the built-in tables of cities and seas | - | - |

- **Each source the other's fallback.** When the one for a place does not answer, the other is asked, and the answer says which gave it.
- **As the sources tell them.** The magnitude, the time, the point and its depth are the sources' own, as they stood when asked: the first estimates change in the first hours, and every answer says so, and that in an emergency the civil protection's word counts. The names of places are the sources' (INGV's in Italian, the USGS's in English); where an event was is also told by the built-in tables (a sea, a region, the nearest town), in the same words for both.
- **The tsunami flag is not a tsunami.** The USGS sets it for large earthquakes at sea; the answer says they were at sea and strong enough for one, and to look at the warning centres.
- **The sources' rules are kept.** A User-Agent naming the program, a second between two requests to a server, the answers kept a minute.
- **No source run from Russia or China.**

## The answers

Every answer is data with a layout (see [the services](README.md)): `janas-chat` shows you the whole list in your language, and gives the model only a brief, from which it answers in a line or two.

## Running it

```sh
janas-quakes
```

It speaks MCP over its standard input and output and writes what it does on its standard error. In other clients:

```sh
claude mcp add quakes -- /path/to/janas-quakes
```

```json
{"mcpServers": {"quakes": {"command": "/path/to/janas-quakes"}}}
```

## What has been run

3 October 2026, over the protocol, in the release build: the earthquakes near the user (Palermo, by GeoJS: INGV), near Norcia in 30 days and 50 km, in the Tyrrhenian and the Ionian seas (INGV, within their polygons), in the Aegean (the USGS: none of magnitude 3 or more in 7 days), near Tokyo of magnitude 4 or more (the USGS); the strongest in the world; an event by the USGS's id, one by INGV's, the latest near the user; an id that is not one and a place that does not exist (refused). The reading of both sources' answers and the layouts are checked by `tests/test_quakes_parse.c`, on answers they gave that night.

In `janas-chat` with Qwen3.6-35B-A3B, the same night, in Italian, on a pipe: "C'è stato un terremoto qui vicino?" (INGV's four near Palermo), "Quali sono stati i terremoti più forti di questa settimana nel mondo?", "Ci sono stati terremoti nel mar Tirreno questo mese?" (thirteen): each answered from its tool, the lists shown in Italian. "Dov'era esattamente quello di poco fa?", after the world's list, was taken by the model as a town of that list, and answered with an earthquake near it: the answers now give the model each event's id, to name the one meant, and run again, the model passed the id of the latest near Palermo. The seas' names of the built-in tables are English: the line telling where an event was says "Tyrrhenian Sea" in an Italian answer too.
