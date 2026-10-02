# janas-git

The git repositories of your computer for `janas-chat` or any other MCP client (see [the services](README.md)): what has changed, the latest commits, the branches, a diff; and, asked of you every time, a commit, a pull, a push, a switch of branch. Nothing that throws work away is offered.

## The tools

They only read, and run without asking:

- **`git_status`** - the branch, the remote branch it follows and how far ahead or behind it is, the files staged, changed, new and in conflict.
- **`git_log`** - a branch's latest commits (`count`, 10 unless asked, 20 at most), only those changing a path (`path`), of an author (`author`), of the last days (`days`).
- **`git_diff`** - the changes not staged, or those staged for the next commit (`staged`), in a path or all: the files with their lines added and removed, and the diff, shown to you (400 lines at most); the model reads its start, enough to write a commit's message.
- **`git_branches`** - the local branches, the latest worked on first, each with the remote branch it follows and how far ahead or behind.

They change something, and say so (MCP's `readOnlyHint` false), so that the client asks you first:

- **`git_commit`** - commits what is staged, after staging the files named (`files`) or every tracked file changed (`all`); a `signoff` when asked. The answer lists the files the commit holds, those staged before included.
- **`git_pull`** - brings the branch up to its remote one, only when that needs no merge (`git pull --ff-only`).
- **`git_push`** - sends the branch's commits to the remote branch it follows; a branch that follows none is pushed to `origin` and followed only when asked (`set_upstream`). Never by force.
- **`git_switch`** - goes to another branch, or makes a new one and goes to it (`create`); git refuses when changes would be lost, and the refusal is the answer.

Never offered: `reset`, `restore`, `checkout` of files, `clean`, `stash drop`, `rebase`, `commit --amend`, `--no-verify`, `push --force`, deleting a branch. git runs with its arguments as they are, no shell between, and nothing the model writes is taken as an option: a branch's name or a path beginning with `-` is refused, paths come after `--`. git runs without a terminal, so that a password it asks for (ssh, https) fails at once instead of waiting; your ssh agent and git's credential helpers work as usual.

A repository is a path (`~/src/x`), a name listed in `~/.config/janas/git-repos.txt` (one path a line; the name is its last part), or, when none is named, the one `janas-chat` was started in.

## In janas-chat

A call that changes something shows whole, highlighted, and waits for your `y` - every time, with `--mcp-auto` too, and never "always". On a pipe, where nobody can answer, it is not run.

## The answers

Every answer is data with a layout (see [the services](README.md)): the list of files, the commits, the diff are shown to you in your language, and the model reads a brief. After a command that changed something, the answer is git's own words.

## Running it

```sh
janas-git
```

It speaks MCP over its standard input and output and writes what it does on its standard error. In other clients (which decide by themselves whether to ask before a call that changes something):

```sh
claude mcp add git -- /path/to/janas-git
```

```json
{"mcpServers": {"git": {"command": "/path/to/janas-git"}}}
```

## What has been run

2 October 2026, over the protocol, in the release and AddressSanitizer builds, on a repository made for the purpose with a remote of its own: the status with a file changed, one renamed, one new; the diffs staged and not; a commit with nothing staged (refused), of the files named and of all the tracked ones; a push; a branch named `--force` (refused); a new branch, its push without a remote branch (refused) and with `set_upstream`; the branches and the log; a pull with nothing to bring; a switch with changes that would be lost (refused by git); a path that is not there and a directory outside any repository.

In `janas-chat` with Qwen3.6-35B-A3B, in a terminal, in that repository: "Cosa è cambiato in questo repository?" (the status and the diff), then "Fai il commit di tutto con un messaggio adatto.": the call shown whole, asked, confirmed with `y`, one commit made with a message the model wrote in Italian. Then, with `--mcp-auto`: on a pipe, "Fai il commit di a.txt con il messaggio 'otto'." was called by the model and not run ("not run: on a pipe nobody can say yes"); in a terminal it was asked all the same, answered `n`, and not run. The status, diff, branches and log were shown in Italian, git's terms as the glossary gives them.
