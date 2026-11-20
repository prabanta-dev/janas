# janas-skills

Your skills for `janas-chat` or any other MCP client (see [the services](../README.md)): instructions for particular tasks, in the format of Claude's skills, listed, read, and their scripts run in a cage. Nothing reaches the network.

A skill is a folder with a `SKILL.md`: a frontmatter with its `name` and `description`, then the instructions, and the files they name (scripts, templates, references). A skill made for Claude works here as it is. Those of every chat are in `skills/` beside your `mcp.json` (`~/.config/janas/skills`, `%APPDATA%\janas\skills`), the same folder Brebus keeps them in; `--dir DIR` adds a folder of skills, searched first (up to 8), such as a project's.

## The tools

- **`skills_list`** - your skills, each with its description.
- **`skill_read`** - a skill's `SKILL.md` and the list of its files, or one of its files (`file`, by its path in the skill's folder). Nothing outside the skill's folder is read, not even by a link.
- **`skill_run`** - a script of a skill (`script`, by its path in the skill's folder), with its arguments (`args`) and text for its standard input (`input`): what it printed, how it ended, and the files it wrote. Your client asks you before each run, as the tool says it is not a mere reading.
- **`skill_<skill>_<script>_<extension>`** - one tool for each script of a skill (`skill_hello_hi_py` for `scripts/hi.py` of the skill `hello`), with the same `args` and `input`, run as `skill_run` runs it: a client that offers tools by name offers the scripts so. Up to 32.

## The cage

A script runs in the cage of `janas-chat`'s code interpreter: Linux namespaces of its own, without privileges.

- **No network.**
- **Read-only:** the system's `/usr` and the skill's folder; nothing else of your home is there.
- **Writable:** one folder of its own, its current folder. When the script writes something, the folder is kept under `~/.cache/janas/skill-runs/` and the answer names it; those of more than a day ago are removed.
- **Limits:** 120 seconds, 2 GiB of memory, files of 512 MiB at most, 64 KiB of each output kept.

On Linux, scripts in Python (`.py`, the system's `/usr/bin/python3`), shell (`.sh`) and JavaScript (`.js`, `.mjs`, `/usr/bin/node`) are run; Python's packages are those the system has. On Windows, scripts in Python alone, by the Python Brebus installs from python.org (*Settings*, *Run code*), in the AppContainer cage of its code interpreter: no network, the skill's folder read, its own folder written (`%LOCALAPPDATA%\janas\skill-runs`); Windows has no limit on a file's size.

## Running it

`janas-chat` starts it by itself, beside the other services, when you have at least one skill in `~/.config/janas/skills`. Brebus does not need it: it gives its model your skills on its own. In another client:

```json
{"mcpServers": {"skills": {"command": "janas-skills"}}}
```
