# Changelog

## 0.1.2 - 12 November 2026

- A model is downloaded with its progress in a notification (GB, MB/s), which can cancel it; the setting up goes on by itself when it is done. It ran in a terminal, and waited until that terminal was closed by hand.
- Once Janas is set up, it says which model to choose in Continue ("Janas … (code)"), where Continue may keep a model of its own selected.
- The last step of the first start says how to have the code written into a file: Apply, Ctrl+I, or Continue's Agent mode with a model made for code.
- "Janas: set up" in the status bar has the editor's warning background.
- **Show the engine at work**: Janas's page of the engine at work (`/monitor`), the servers started with it (setting `janas.monitor`, on), with a Janas newer than v2026.11.12.
- The servers stopped are said so in the log ("the servers stopped", not "ended (null)"); Windows paths given to janas-setup with their drive in capitals.

## 0.1.1 - 10 November 2026

- The first start: a page of four steps opens after the extension is installed (set up Janas, a model, an extension to chat and complete code with, a first question), each step ticking itself when done.
- Until Janas is set up, every start says so in a notice (kept in the editor's notifications once it hides), and the status bar reads "Janas: set up"; once set up, a start with no extension to chat or complete with says that too.
- New commands: "Install an extension to chat and complete code with" and "Show the steps to set it up".
- An icon: janas-chat's octopus.
- Janas's folder taken away means a first start again: the page of the steps opens and the notices come back, whatever the editor remembers of an earlier installation.
- The setting up no longer stops in silence: when no model or no extension to chat with is chosen, or one cannot be installed, it says so and offers to go on, and the log says every step (on Windows 11 with VS Code 1.136 it stopped after the list of the models, thirteen times, saying nothing). The lists of models and of extensions stay open when the focus goes elsewhere.

## 0.1.0 - 7 November 2026

- Janas installed in its own folder from its GitHub releases, checked against the SHA-256 published beside the package and the SHA256SUMS inside it; versions kept side by side, with going back to the previous one.
- A model chosen for the machine's memory, the models for code first.
- Continue, llama.vscode or Cline installed where none is, and set up by janas-setup.
- Janas's servers started with the editor, their state in the status bar; once a day, a word when a new version is out.
