# janas-github

GitHub for `janas-chat` or any other MCP client (see [the services](README.md)): a project's issues and pull requests (the new ones since you last asked), its latest releases, its recent commits, what it is. It only reads: nothing on GitHub is changed.

## The tools

- **`github_issues`** - a repository's issues, or its pull requests (`kind`), open, closed or all (`state`), the newest first, with how many there are; `new` gives only those opened since the last time you asked ("are there new issues on Janas?"), and `days` those opened in the last days. Each with its number, title, author, date, comments, labels and link; a pull request says whether it was merged or is a draft.
- **`github_releases`** - the latest releases (`count`, 1 for the latest), each with its tag, name, date, author, whether it is a pre-release, its files, the start of its notes and its link; a repository with no releases gives its latest tags.
- **`github_commits`** - a branch's commits of the last days (`days`, 7 unless asked; `branch`, the main one unless asked; `author`), the newest first, ten at most.
- **`github_repo`** - what a repository is: its description, stars, forks, watchers, open issues and pull requests, language, licence, main branch, last push, topics, site.

A repository is `owner/name`, a github.com link, or a name alone ("Janas", "llama.cpp"): among your own repositories first (with a token), else the one of that name with the most stars on GitHub; the answer says which it took.

## The answers

Every answer is data with a layout (see [the services](README.md)): `janas-chat` shows the user the list in the user's language - titles, messages and notes as their authors wrote them - and gives the model only a brief, from which it answers in a line or two. The terms of the craft (commit, branch, pull request, release...) come from the glossary of `janas-chat`, so that the translation does not turn them into words of everyday speech.

## The token

GitHub answers 60 requests an hour without a token, and 10 searches a minute; with one 5,000 and 30, and your private repositories too. `janas-github` takes the token of `GH_TOKEN` or `GITHUB_TOKEN` when set, else that of the [gh](https://cli.github.com) command (`gh auth token`) when you have logged in with it (`gh auth login`); without one it asks GitHub anonymously. The token stays in the program's memory: it is never written, printed or sent anywhere but to GitHub's API, and a redirect to another host loses it.

When you last looked at a repository's issues or pull requests is kept in `~/.config/janas/github-seen.txt`, a line a repository; the first time, "new" means the last seven days.

## Running it

```sh
janas-github
```

It speaks MCP over its standard input and output and writes what it does on its standard error. In other clients:

```sh
claude mcp add github -- /path/to/janas-github
```

```json
{"mcpServers": {"github": {"command": "/path/to/janas-github"}}}
```

## What has been run

2 October 2026, over the protocol, in the release and AddressSanitizer builds, with the gh command's token: the new issues of Janas (eight in the last seven days the first time, none the second), its closed issues, the latest release of llama.cpp (found by its name), Janas's commits of three days and llama.cpp's of one, the open pull requests of llama.cpp (1,656), the repository of Janas from its link, and a repository that does not exist (a clear error).

In `janas-chat` with Qwen3.6-35B-A3B, the same day, in Italian: "Ci sono nuove issue su Janas?" (the issues of the last seven days the first time, none the second), the latest release of llama.cpp and its commits of the week. The first run showed the issues' layout given back untranslated (now refused and asked again) and the model calling eight issues open when four were closed (the brief now counts them). Run again with the layouts translated anew: the open issues ("4 issue aperte", each "aperta": the states of an issue come from the glossary, as the model wrote "aperte" for one), the latest release, the commits and the repository of llama.cpp, all in Italian.
