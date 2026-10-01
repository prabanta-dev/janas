# Janas's services

A service of Janas is a program that brings data of the world - flights, the weather, an encyclopedia - to a model, as tools of the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). Each one is an MCP server of its own, over its standard input and output, so that any MCP client can use it: `janas-chat`, and the assistants of others.

| Service | What it answers | Key |
|---|---|---|
| [`janas-flights`](flights.md) | where a flight is and what it is doing, its times, the flights of a route, what flies over a sea or near a place | none for live data; a free one of your own for schedules |
| [`janas-weather`](weather.md) | the weather now (a model's and a station's), the forecast, the sea and its waves, the warnings in force | none |
| [`janas-wiki`](wiki.md) | Wikipedia's search, its pages as text, the pages about what lies around a place | none |

What every service does the same way:

- **Facts already worked out.** The answers are text for the model, with places, distances and times already computed and put into words, so that the model telling them has nothing to compute; what is not known is said to be not known.
- **Open data first.** Questions about an area or a list are answered from open sources only; a source with a small quota (a key of yours) is kept for questions about one thing.
- **It says when to use it.** A service tells the client's model when its tools apply: a question on the subject is enough, the tools need not be named, and the model is told not to name them in its answers.
- **Keys stay put.** A key is read from the service's file under `~/.config/janas/` or from a variable of its environment, and never printed, not even in an error.
- **Only what was asked.** A service's tools are used for its subject when the user asks about it, and never because of another's answer: no weather or flights for the town a page names, unless the user asks for them.
- **What is the user's goes to the user.** A page of Wikipedia is shown as it is, and the model reads only a note of it: MCP marks each part of an answer with its audience, and `janas-chat` shows the parts meant for the user alone instead of giving them to the model. A client that does not heed the audience gives the model both.
- **Here, when no place is named.** A question about "here" takes where the user is: `JANAS_LOCATION` when set (a city, `lat,lon`, or `off`), else the internet connection's position (GeoJS, ipwho.is), else the city of the computer's time zone; the answer says which, so the model can say it is a guess ([details](weather.md#when-no-place-is-named)).

## In janas-chat

`janas-chat` starts by itself the services it finds beside it (in the same directory as the program) and runs their calls without asking: they read public data and change nothing. Their tools are not written into the system message: it holds a catalog, a line a service, and the model opens a service when a question needs it, getting its instructions and tools then (`--all-tools` writes them all at the start instead). A call shows as a dimmed line. Nothing to configure, nothing to name: ask your question.

- `/mcp` lists the servers running, the services marked as Janas's.
- `--no-services` starts none of them; `--no-mcp` no server at all.
- A server of the same name in `~/.config/janas/mcp.json` takes the place of the service (to give it options, say).

## In other clients

A service is configured like any other stdio MCP server. In [Claude Code](https://code.claude.com/docs/en/mcp):

```sh
claude mcp add flights -- /path/to/janas-flights
```

`--scope user` makes it available in every project, `-e NAME=value` gives it a variable (a key), and `claude mcp list` shows whether it started. In the clients that read an `mcpServers` file - Claude Desktop's `claude_desktop_config.json`, Cursor's `mcp.json`, `janas-chat`'s own `~/.config/janas/mcp.json`:

```json
{"mcpServers": {"flights": {
    "command": "/path/to/janas-flights",
    "env": {"JANAS_AVIATIONSTACK_KEY": "YOUR_KEY"}}}}
```

Each service's page says which variables and files it reads.
