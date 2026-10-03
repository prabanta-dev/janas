# janas-wiki

Wikipedia for `janas-chat` or any other MCP client (see [the services](../README.md)), when the user asks for it: its search, a page shown as it is, and the pages about what lies around a place. Wikipedia's own API: no key, no account.

## The tools

- **`wiki_search`** - the pages that match some words, best first: their titles, each with what it is about in a line (Wikidata's description).
- **`wiki_page`** - shows the user a page as it is: a short page whole, a long one its introduction and the names of its sections, or the section asked for, by its name or the start of it (a section longer than 10,000 characters is cut at a paragraph, and the answer names its parts). A title that leads to another page is followed and said; a disambiguation page comes as the list of the pages it leads to. The model gets a note of it, not its text: the page's address, how many words were shown, its sections.
- **`wiki_nearby`** - the pages about what lies around a place, nearest first, with how far and which way ("Palazzo D'Alì: 170 m NW"), within up to 10 km. The place is a page's title (its point is the page's), `lat,lon`, or none: then where the user is, as for the weather ([details](../weather/README.md#when-no-place-is-named)).

Every tool takes `language`, the Wikipedia to ask: the model gives the user's (`it`, `en`, `fr`...), and the English one when the first has nothing; when it gives none, the computer's language is taken (`LANG=it_IT.UTF-8`: `it`), else English.

## The page is yours, not the model's

A page is thousands of tokens, which a model reads before it can say a word - over a minute of a large model on a laptop for one page - and a few pages fill its context. And a page asked for is meant to be read, not retold. So the page goes to the user as it is, and the model answers from what it knows: it is told to use Wikipedia only when the user asks for it, or asks what there is around a place, and never to ask the other services for the weather or the flights of a place a page names.

The answer of `wiki_page` has two parts, each marked with its audience as MCP provides: a note for the model, and the page for the user. `janas-chat` shows the user's part as it is and gives the model only the note; a client that does not heed the audience gives the model both.

## Where the data comes from

| Data | Source | Key |
|---|---|---|
| search, pages, places | the [MediaWiki Action API](https://www.mediawiki.org/wiki/API:Main_page) of each Wikipedia (TextExtracts, GeoData) | none |
| where the user is | [GeoJS](https://www.geojs.io), [ipwho.is](https://ipwho.is) | none |

- **The text** is Wikipedia's, under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/): every answer gives the page's address, and the model is told to give it as the source.
- **It is plain text**: the tables and the boxes beside the text (the infobox's population, dates) are not in it.
- **Requests carry a User-Agent** naming the program and where it lives, as [Wikimedia's policy](https://meta.wikimedia.org/wiki/User-Agent_policy) asks: `janas-wiki/<version> (https://github.com/prabanta-dev/janas)`. One request at a time, and the answers are kept for an hour, so that a page read and then a section of it cost one request.
- **Wikipedia is not news**: the model is told to say so when a question is about today.

## Running it

```sh
janas-wiki
```

It speaks MCP over its standard input and output and writes what it does on its standard error. In other clients:

```sh
claude mcp add wiki -- /path/to/janas-wiki
```

```json
{"mcpServers": {"wiki": {"command": "/path/to/janas-wiki"}}}
```

## What has been run

1 October 2026:

- Over the protocol, every tool, in the release and AddressSanitizer builds, the note for the model and the page for the user apart: a search for "promessi sposi" on the Italian Wikipedia; "Manzoni", which leads to "Alessandro Manzoni", its introduction and the names of its sections, then its section "Biografia", cut and with its parts named; "Mercurio", a disambiguation page; a page that does not exist; the pages around Trapani (the castle 200 m NNW, the town hall 170 m NW), around the Colosseum on the English Wikipedia, and around "here", placed in Palermo by GeoJS; a language that is not one, taken as the computer's (Italian).
- `janas-chat` with Qwen3.6-35B-A3B, in Italian, with the page shown to the user: "Who was Franco Zeffirelli? Briefly." was answered from the model's memory with no tool, in 6.6 s - and with a mistake, the cast of another *Romeo and Juliet*; "show me his page on Wikipedia" showed the page's introduction and sections as they are, the model reading a note of 174 tokens in 3 s; "and what's the weather in Florence today?" asked for the weather, as asked. Before, with the page given to the model: "what is there to see nearby, and what's the weather?" gave the pages around where the user is and the weather there.
