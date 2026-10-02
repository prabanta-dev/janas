# Janas's services

A service of Janas is a program that brings data of the world - flights, the weather, roads and places, an encyclopedia, GitHub's projects, the git repositories of your computer - to a model, as tools of the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). Each one is an MCP server of its own, over its standard input and output, so that any MCP client can use it: `janas-chat`, and the assistants of others.

| Service | What it answers | Key |
|---|---|---|
| [`janas-flights`](flights.md) | where a flight is and what it is doing, its times, the flights of a route, what flies over a sea or near a place | none for live data; a free one of your own for schedules |
| [`janas-git`](git.md) | the git repositories of this computer: what has changed, the commits, the branches, a diff; a commit, pull, push or switch of branch, each asked of you first | none |
| [`janas-github`](github.md) | a project's issues and pull requests (the new ones), its latest releases, its recent commits, what it is | none; the gh command's token when you have one |
| [`janas-maps`](maps.md) | a route by car, bike or on foot, step by step; journeys by public transport; what is near a place; where an address is | none |
| [`janas-weather`](weather.md) | the weather now (a model's and a station's), the forecast, the sea and its waves, the warnings in force | none |
| [`janas-wiki`](wiki.md) | Wikipedia's search, its pages as text, the pages about what lies around a place | none |

What every service does the same way:

- **Reading by default.** Every tool only reads, but those of `janas-git` that commit, pull, push or switch branch; they say so, and nothing that throws work away is offered.
- **Facts already worked out.** The answers are text for the model, with places, distances and times already computed and put into words, so that the model telling them has nothing to compute; what is not known is said to be not known.
- **Open data first.** Questions about an area or a list are answered from open sources only; a source with a small quota (a key of yours) is kept for questions about one thing.
- **It says when to use it.** A service tells the client's model when its tools apply: a question on the subject is enough, the tools need not be named, and the model is told not to name them in its answers.
- **Keys stay put.** A key is read from the service's file under `~/.config/janas/` or from a variable of its environment, and never printed, not even in an error.
- **Only what was asked.** A service's tools are used for its subject when the user asks about it, and never because of another's answer: no weather or flights for the town a page names, unless the user asks for them.
- **Data and a layout.** The longest answers - the aircraft over a sea or near a place, the weather now, the forecast, the sea, the routes and journeys and places of the maps - come as data with a layout (MCP's `structuredContent`, the layout in `_meta`) besides their English text. `janas-chat` fills the layout in the user's language and shows it, and gives the model only a brief: the figures it needs to answer in a line or two. A list of forty aircraft cost the model 3,000 tokens to read and as many to write again; it now reads about a hundred. The layout in another language than English is asked of the model once, on the side of the conversation: it gets the text with the layout's tags as numbered markers and its words numbered one a line, never the tags themselves, which are put back and checked (a word it leaves out stays English; a marker left out, and the English layout is used). The terms of the scales - the Beaufort forces, the Douglas sea states - are not asked: in Italian, French, Spanish and German they come from a glossary in `janas-chat`, with the names the national weather services give them, and the request gives the model the terms of the sea for the rest of the text. It is kept in `~/.config/janas/layouts`, one file a layout and language, where it can be corrected by hand; a new version of a layout is translated anew. The language is the computer's (`LANG`).
- **What is the user's goes to the user.** A page of Wikipedia is shown as it is, and the model reads only a note of it: MCP marks each part of an answer with its audience, and `janas-chat` shows the parts meant for the user alone instead of giving them to the model. A client that does not heed the audience gives the model both.
- **Here, when no place is named.** A question about "here" takes where the user is: `JANAS_LOCATION` when set (a city, `lat,lon`, or `off`), else the internet connection's position (GeoJS, ipwho.is), else the city of the computer's time zone; the answer says which, so the model can say it is a guess ([details](weather.md#when-no-place-is-named)).

## In janas-chat

`janas-chat` starts by itself the services it finds beside it (in the same directory as the program) and runs without asking the calls of the tools that only read. A tool that changes something - `janas-git`'s commit, pull, push, switch - says so (MCP's `readOnlyHint` false), and its call is shown whole and asked of you every time, with `--mcp-auto` too; on a pipe it is not run. Their tools are not written into the system message: it holds a catalog, a line a service, and the model opens a service when a question needs it, getting its instructions and tools then (`--all-tools` writes them all at the start instead). A call shows as a dimmed line. Nothing to configure, nothing to name: ask your question.

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
