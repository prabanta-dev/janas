# janas-files

Your files for `janas-chat` or any other MCP client (see [the services](../README.md)): what a folder holds, what a file says, the files found by their name or by the text they hold; and, asked of you every time, a file written or edited, a folder made. It reaches your home and the folders you list, never the private files, and it deletes nothing.

## The tools

They only read, and run without asking:

- **`files_list`** - what a folder holds, its folders first, the files with their sizes, down to three levels (`depth`). The names starting with a dot are left out unless asked (`hidden`), and the answer says how many there were; a folder that holds nothing else - a Gambas project's `.project` and `.src` - shows them.
- **`files_read`** - a text file's lines: 400 from the start unless asked (`offset`, `lines`, at most 2000 and 100 KB at a time), and how many there are in all. A file that is not text, or larger than 32 MiB, is not read.
- **`files_search`** - the files under a folder by their name (`name`, a pattern such as `*.class`, any case) and the lines holding a text (`text`, any case): 100 hits at most, files of up to 4 MiB, 50,000 files looked at. Hidden folders are searched (a project's own may be all it has); the folders that only programs fill - `.git`, `.cache`, `node_modules`, virtual environments - are not.

They change something, and say so (MCP's `readOnlyHint` false), so that the client asks you first:

- **`files_write`** - writes a text file (16 MiB at most), making the folders it goes in (`make_dirs` false to have them be there already). A file that is there is replaced only with `overwrite`, and a copy of it is kept first.
- **`files_edit`** - replaces a piece of a text file, `old_text` exactly as it is there, with `new_text`; the piece must be there once, or `all` replaces every one. A copy of the file is kept first.
- **`files_mkdir`** - makes a folder, and those above it that are not there.

A file is written beside itself and renamed over the old one, so that it is never left half written; it keeps the old one's permissions. The copies go to `~/.local/share/janas/files-backup/<date-time>/`, under the path the file has in your home, and the answer says where. Nothing is deleted, moved or renamed.

## Where it reaches

A path is `~/...`, absolute, or relative to the folder `janas-chat` runs in. It is followed to its end - every link, every `..` - before it is judged, so that a link in your home that points elsewhere leads nowhere: what counts is the file it reaches. It reaches:

- your home;
- the folders you list in `~/.config/janas/files-dirs.txt`, a path a line (`#` starts a comment), read again at every call.

Never, wherever they are, the private files: `~/.ssh`, `~/.gnupg`, `~/.password-store`, the keyrings, `~/.config/janas` (the services' keys), the browsers' and the mail's profiles (Firefox, Thunderbird, Chrome, Chromium, Brave, Edge, Vivaldi, Opera), the clouds' and the tools' credentials (`~/.aws`, `~/.azure`, `~/.config/gcloud`, `~/.kube`, `~/.docker`, `~/.config/gh`, `.netrc`, `.git-credentials`, `.pgpass`, `.npmrc`, `.pypirc`, Cargo's), the shells' histories; and any file named `.env`, `.env.*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `id_rsa*`, `id_ed25519*`, `id_ecdsa*`, `id_dsa*`, `*.kdbx`, `*.keystore`, `*.jks`, `credentials.json`, `secrets.*`. They are neither listed, nor read, nor searched, nor written, nor reached by a link. A file outside the home - `/etc/machine-id`, say - is refused with the reason, and the model is told to say so.

## In janas-chat

A call that changes something shows whole, highlighted, and waits for your `y` - every time, with `--mcp-auto` too, and never "always". On a pipe, where nobody can answer, it is not run.

## Running it

```sh
janas-files
```

It speaks MCP over its standard input and output and writes what it does on its standard error. Linux only for now: on Windows a path, the home and the private folders are other things, still to be written. In other clients (which decide by themselves whether to ask before a call that changes something):

```sh
claude mcp add files -- /path/to/janas-files
```

```json
{"mcpServers": {"files": {"command": "/path/to/janas-files"}}}
```

## What has been run

2 November 2026, in `tests/test_files.c` (release and AddressSanitizer builds), on a home made for the purpose: a listing, one with the hidden names, a folder of dotted names alone; lines 2-3 of a file; a binary file refused; a search by name and by text; and refused, reading as writing - a link to a file outside, `../../out/secret.txt`, `~/.ssh/id_rsa`, a link to `~/.ssh`, `.env`, `~/.config/janas/key.txt`, `/etc/hostname`, `~/proj/../.ssh/id_rsa`, a folder made outside -, the file outside unchanged; a file written with its folders, one replaced only with `overwrite` and its copy found, an edit, an ambiguous one refused, one of all the pieces, one not found; a dangling link replaced by the file rather than followed out of the home; a folder listed in `files-dirs.txt` reached, a private key in it not.

In `janas-chat` with Qwen3-Coder-30B-A3B, in a folder holding a small Gambas3 project (`ref/Notes`: `.project`, `.src/FMain.class`, `.src/FMain.form`): "Nella cartella ref/Notes c'è un progetto Gambas3. Leggi i suoi file e spiegami in poche righe cosa fa il programma e dove salva i dati." on a pipe: the folder listed, `.src` listed, the three files read, the program explained (notes kept in `~/.notes.txt`). Then in a terminal, "Scrivi in new/Todo una versione del programma di ref/Notes per una lista di cose da fare, con un pulsante Done che toglie l'elemento scelto...": the three files read, `new/Todo` and `.src` made, `.project`, `FMain.class` with its `btnDone_Click` and `FMain.form` with the Done button written, each call shown and confirmed with `y`.
