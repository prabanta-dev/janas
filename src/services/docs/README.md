# janas-docs

Your own documents for `janas-chat` or any other MCP client (see [the services](../README.md)): the passages of your notes, letters, contracts, manuals and papers that answer a question, each with the document it comes from, and a document read whole. It indexes the folders you name, in the background, and keeps the index on your computer; nothing reaches the network unless you point it at an embeddings server of your own.

## The tools

All three only read, and run without asking:

- **`docs_search`** - the passages that answer a question best (`query`; 6 unless asked, at most 20, `max`), only under a folder if asked (`folder`). Each comes with its document, its place in it ("passage 3 of 12") and how it was found: by its words, by its meaning, or both.
- **`docs_read`** - a document's text, 12,000 bytes from `offset` unless asked (`length`, at most 50,000), and how long the whole text is. A Word or LibreOffice document is read as its text.
- **`docs_status`** - the folders indexed, how many documents and passages, whether they are searched by meaning too and with which model, how far the indexing is, the last document that could not be read and why.

## What it reads

The folders listed, a path a line (`~/...` or absolute, `#` starts a comment), in `~/.config/janas/docs-dirs.txt`; without that file, your Documents folder (the one your desktop names in `~/.config/user-dirs.dirs`, `~/Documents` otherwise). The list is read again at every round of indexing.

Under them: plain text (`.txt`, `.text`, `.csv`, `.tsv`), Markdown (`.md`, `.markdown`, `.mdown`), reStructuredText, Org, AsciiDoc and TeX files, Word documents (`.docx`), LibreOffice ones (`.odt`) and PDF, of up to 64 MiB each. Not yet: spreadsheets, presentations, web pages.

Never: the names starting with a dot (hidden files and folders), `node_modules`, links (neither to files nor to folders: a link cannot take the indexing out of the folders you named), and the private files, the same as [`janas-files`](../files/README.md#where-it-reaches) keeps away from - keys, passwords, tokens, `.env` files, the browsers' and the mail's data. `docs_read` reads only a document under one of the folders, of a kind read here, and not private.

A Word document is read from its `word/document.xml`, a LibreOffice one from its `content.xml`, with the archive opened and decompressed by Janas's own code: the text a reader sees, its headings marked as in Markdown (`#`, `##`...), without the deleted text of tracked changes, the codes of fields, or the comments.

A PDF is read by Janas's own code too:

- **The objects** are found by scanning the file, not through its cross-reference table, which a file updated many times often has wrong. The later definition of an object wins. The objects kept in object streams are read from them.
- **The pages** are walked in their order, each with the resources it inherits. Their text is taken from the content streams and from the forms they draw.
- **Characters** come through each font's ToUnicode map, or through WinAnsi for a simple font without one.
- **Spaces and new lines** are found from the glyphs' widths and the text's moves. A gap wider than a fifth of the font's size is a space; a change of line is a new line.

Not read:

- encrypted files (refused, and said so);
- pages that are only pictures (scanned pages: no OCR yet);
- fonts of two-byte codes without a map;
- filters other than FlateDecode.

## By words, and by meaning

The text of each document is cut into passages of about 1,200 bytes (some 300 tokens) that overlap by 200, ending at a paragraph or a space. A search is always made by the words: the passages holding the question's words, the rarer ones in your documents counting more (BM25), so that names, codes and numbers are found as they are written. With an embedding model it is made by the meaning as well: the passages whose vectors are nearest to the question's, which finds what is said with other words. The two rankings are joined by their ranks, so that a passage high in both comes first.

The vectors come, in this order, from:

1. **A server** (`server = http://127.0.0.1:8080` in `~/.config/janas/docs.conf`, or `--server URL`): a `janas-server` started with `--embedding-model`, or any server that speaks OpenAI's `/v1/embeddings`; `server_key =` (or `--key`) when it wants an API key. A server not answering yet is asked again at every round; meanwhile the search goes by the words.
2. **A model of its own** (`embedding_model = ~/models/qwen3-embedding-0.6b-q8.jns`, or `--embedding-model FILE`), opened by `janas-docs` itself through the engine: `janas-get qwen3-embedding-0.6b` fetches one.
3. **Neither:** the words alone.

The index says which model made its vectors. Another one makes them again, keeping the text: two models' vectors are not comparable. Started without any, the service keeps the vectors it has, unused, until that model is back.

## The index

In `~/.local/share/janas/docs/index.bin` (`$XDG_DATA_HOME/janas/docs/` when set), written aside and renamed, so that it is never left half written. At start it is read; then a round of indexing, in a thread of its own, looks at the folders, reads the documents new or changed since the last round (by their time and size) and leaves out those gone; then it makes the vectors missing, eight passages at a time. Every passage is searched by its words as soon as its document is read, and by its meaning as soon as its vector is made. A round comes every five minutes, and at a search when the last one is more than a minute old. A document that cannot be read (not text, a broken archive) is said in `docs_status` and not read again until it changes.

As a server it runs at a lower priority (nice 10), it and the threads of its model: the indexing gives way to what you are using - the model of a chat first.

## Running it

```sh
janas-docs                      # the MCP server
janas-docs --index              # a round of indexing now, what is there, the end
janas-docs --search "the boiler's service"   # the index as it is, searched
```

It speaks MCP over its standard input and output and writes what it does on its standard error. Linux only for now, as `janas-files`. In other clients:

```sh
claude mcp add docs -- /path/to/janas-docs
```

```json
{"mcpServers": {"docs": {"command": "/path/to/janas-docs"}}}
```

## What has been run

14 November 2026, in `tests/test_docs.c` (release, AddressSanitizer and ThreadSanitizer builds): a docx and an odt made by Python's `zipfile` read to their exact text (a heading, a tab, `&amp;`, `&#x263A;`, tracked deletions and field codes left out; an odt's repeated spaces, line break, heading level and comment); both archives cut at every byte and mutated 20,000 times each, the XML cut at every byte, the index file cut at every byte (only the whole one reads) and mutated 20,000 times, an embeddings answer mutated 20,000 times - no crash, no read past the input. A home made for the purpose, its Documents folder indexed by words: four documents, `.env`, `secrets.txt`, a hidden folder, a link out and a `.jpg` left out, a text file holding a NUL byte said not to be text; searches by words, one under a folder; a document read, a stretch of it, and refused - `.env`, `secrets.txt`, the link, the `.jpg`, `/etc/passwd`, a relative path, a missing file; a document changed and one removed, seen at the next round; the index read again from its file. Then with a small embeddings server of the test's own on 127.0.0.1: every passage given its vector, a search by meaning and words; the server's model changed, the vectors made again with the text kept; the thread started, searched while it runs, stopped; a start with no model keeping them.

On the development laptop (Intel Core Ultra 9 185H, 32 GB), with the 15 pages of Janas's `docs/` copied into a Documents folder (264 KB, 273 passages) and Qwen3-Embedding-0.6B (Q8): the first round, vectors and all, took 1 min 55 s with the model opened by `janas-docs`, and 1 min 39 s from a `janas-server` started with `--embedding-model` on the same machine; the same question ranked the same passage first both ways (cosine 0.70). "how do I make Obsidian talk to the server" found the Obsidian section of `compatibility.md` first. Over MCP on stdio, `initialize`, `tools/list` and a call of each tool answered as above.

PDF, 15 November 2026: in `tests/test_docs.c`, a PDF made by the test, two pages read to their exact text:

- a Type0 font with its ToUnicode map, glyph by glyph as Skia writes them, the spaces found from the widths;
- a WinAnsi font with `\351`, a `TJ`'s kerning gap;
- a compressed stream, a page of two streams (one with a wrong `/Length`), a font inside an object stream, a form with text of its own;
- the same file with `/Encrypt` refused;
- the file cut at every byte and mutated 20,000 times.

On the development laptop, 25 PDFs at hand, compared word by word with poppler's `pdftotext` 25.03:

- **Research papers** (pdfTeX, Nature, Springer, Skia): 90-99.8% of poppler's words found, and 92-99.7% of the words found are poppler's too.
- **The three lowest:** 71%, 78% and 84%. Their missing words are the labels of plots drawn inside the figures ("L2", "0H", digits).
- **Speed:** a paper of 2.2 MB was read in 0.03 s, against 0.11 s for `pdftotext`.
- **Fuzzing under AddressSanitizer and UBSan:** that paper and a Skia PDF, each mutated 300 times and cut at 200 places, gave no error.
- **Not read, and said so:** an encrypted PDF.

In `janas-chat` with Qwen3-4B (Q4_K_M), on a pipe, the same folder searched by words: "Cerca nei miei documenti cosa dicono di Obsidian e riassumilo." opened the service, called `docs_search` with "Obsidian", and summed up the Obsidian section of `compatibility.md` with its `--allow-origin app://obsidian.md` command. Asked less directly ("Nei miei documenti c'è scritto come collegare Obsidian a janas-server?"), Qwen3-4B did not open the service and answered from its own guesses: a small model's limit.
