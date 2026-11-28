# Brebus's languages

Brebus's page is written in English, and English is the only language inside the program. Every other language is a file of this folder, `<code>.json`, installed in `brebus-lang/` beside the program, in every package. Brebus lists the files it finds, so a language more is a file more: nothing else changes.

```json
{
  "name": "Français",
  "locale": "fr",
  "texts": {
    "New chat": "Nouvelle conversation",
    "{n} chats found": ["{n} conversation trouvée", "{n} conversations trouvées"]
  }
}
```

- **The name of the file** is the language's code: `fr`, `pt` (Portugal's Portuguese), `pt-BR`, `zh-Hans`, `es-419`. The system's language finds its own code first, then its language alone (`pt-AO` finds `pt`), then the language's only variant.
- **`name`** is the language's name in itself, shown in Settings. Plain text.
- **`locale`** writes the numbers and the dates, and chooses a number's form (`pt-PT`: one for 1 alone).
- **`texts`** holds, for each English text of the page exactly as the page says it, its translation. A text the file lacks stays in English. A text with a number has two forms, `[one, other]`. Every key of `it.json` is a text of the page.

The files are data, never code. Brebus reads the file and writes the script the page runs itself, and keeps a translation only when it is as safe as its English:

- the same markup, tag for tag, with the same attributes, and no other `<`;
- a `"` outside the markup only where the English has one, because the page puts some texts inside attributes: write quotes as the language does (« », “ ”, „ “);
- only the parts `{x}` of its English, and `{n}` in a number's forms;
- 8192 bytes at most.

A translation that breaks a rule stays in English, and Brebus says so on its standard error. A changed file can make a text wrong, but it cannot run anything in the page.
