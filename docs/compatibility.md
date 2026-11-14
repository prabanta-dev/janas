# Programs that work with Janas

`janas-server` speaks three APIs that other programs already know: OpenAI's (`/v1/...`), Ollama's (`/api/...`, on Ollama's port 11434 if it is started with `--port 11434`) and Anthropic's Messages API. A program that talks to any of them can talk to a model on your own machine. This page lists the programs that were actually tried, with their version, the API they used, what was done with them and on what day. A program not listed here may well work too; it has simply not been tried.

All the trials below ran on the development laptop (Intel Core Ultra 9 185H, 32 GB, Debian 13) with `janas-server` on the same machine. How each one is set up step by step is in [editor.md](editor.md) for the editors and in the sections below for the others.

| Program | Version | API | What was tried | Result | Tried |
|---|---|---|---|---|---|
| [AnythingLLM](https://anythingllm.com) Desktop | 1.19.0 | Ollama | chat, streamed; a document embedded with `--embedding-model` and asked about; the agent, with a tool call | works | 14 Nov 2026 |
| [Obsidian](https://obsidian.md) with [Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) | 1.15.1, Copilot 4.0.20 | OpenAI (Copilot's "Ollama" provider) | the model listed and verified, a question in Copilot's chat | works with `--allow-origin app://obsidian.md`, and Janas from the develop branch of 14 November 2026 on (below) | 14 Nov 2026 |
| [Zed](https://zed.dev) | 1.28.1 | Ollama | the Agent panel: a question about a file of the project, the model calling Zed's tool to read it | works; the first message is slow (below) | 14 Nov 2026 |
| [Continue](https://www.continue.dev) (VSCodium, VS Code) | 2.0.0 | Ollama | chat, edits, completion at the cursor, its command-line agent | works ([editor.md](editor.md#continue-vscodium-vs-code)) | Nov 2026 |
| [Cline](https://cline.bot) | 4.1.26 | Ollama | agent tasks with tools | works ([editor.md](editor.md#cline-vscodium-vs-code)) | Nov 2026 |
| [Roo Code](https://roocode.com) | 3.54.0 | Ollama | agent tasks, a project written | works ([editor.md](editor.md#roo-code-vscodium-vs-code)) | Nov 2026 |
| [Claude Code](https://code.claude.com) | 2.1.311 | Anthropic | a project read and a new one written in its style | works ([editor.md](editor.md#claude-code)) | Nov 2026 |
| [aider](https://aider.chat) | 0.86.2 | Ollama | edits to a project | works ([editor.md](editor.md#aider)) | Nov 2026 |
| [Open WebUI](https://openwebui.com) | | Ollama | chat, whole and streamed | works ([editor.md](editor.md#open-webui)); start the server with `--think off` ([serving.md](serving.md)) | Nov 2026 |
| VS Code's own chat | | Ollama, then OpenAI | its sequence of requests replayed with `curl` and the `openai` SDK; VS Code itself not yet | not tried in VS Code ([editor.md](editor.md#vs-codes-own-chat)) | Nov 2026 |

## AnythingLLM

In AnythingLLM's settings choose **Ollama** as the LLM provider with the base URL `http://127.0.0.1:11434` and the model `janas-server` was started with, and Ollama as the embedding provider too if the server has an embedding model (`--embedding-model`, such as `qwen3-embedding-0.6b` from `janas-get`):

```sh
janas-server qwen3-4b.jns --port 11434 --think off \
    --embedding-model qwen3-embedding-0.6b-q8.jns
```

What was tried, with Qwen3-4B and Qwen3-Embedding-0.6B: a question in a workspace in chat mode ("Rome." in 1.6 s, streamed); a text file uploaded and embedded through `/api/embed`, and a question only that file could answer, answered right with the file given as its source; the agent, which AnythingLLM starts by itself in its "automatic" mode when the model can call tools, answering "17 times 23 is 391"; and the agent calling one of its tools, `document-summarizer`, with valid JSON arguments, AnythingLLM running it and sending its result back. Qwen3-4B gave that tool a file name it had made up: a small model's limit, not the connection's. In automatic mode the agent's first reply reads about 900 tokens of AnythingLLM's instructions (6.3 s to its first token here).

## Obsidian, with Copilot for Obsidian

In Copilot's settings, **BYOK**, **Add a provider**, **Ollama**: it lists the server's models (`/v1/models`), marks the provider "Verified", and the model enabled there appears in Copilot's chat. Obsidian's window is a web page whose origin is `app://obsidian.md`, and `janas-server` answers only the web pages it is told to, so start it with:

```sh
janas-server qwen3-4b.jns --port 11434 --think off --allow-origin app://obsidian.md
```

Without it Copilot says "Connection error" and the server's answer, which Copilot does not show, says which option lets it in. Janas from the release of 13 November 2026 and earlier still fails with the option: the browser's preliminary request (CORS) was allowed two headers only, and the JavaScript client Copilot uses sends more of its own. From the develop branch of 14 November 2026 the server allows the headers such a page asks for. Asked "Name three Italian islands", Qwen3-4B answered through it in about 5 s.

## Zed

In Zed's `settings.json`:

```json
{
  "language_models": { "ollama": { "api_url": "http://127.0.0.1:11434" } },
  "agent": { "default_model": { "provider": "ollama", "model": "qwen3-4b" } }
}
```

Zed asks the server for its models when it starts (`/api/tags`, `/api/show`). In the Agent panel, asked to explain a file of the project, Qwen3-4B called Zed's tool to read it and then answered right. The Agent panel's first message carries Zed's instructions and the descriptions of its tools, 10,984 tokens of which 7,721 are the tools': Qwen3-4B read them in 204.8 s on the laptop before its first word. The server keeps what it has read, so the next message started after 1.3 s, reading only its 55 new tokens. Reading a long prompt that slowly is being looked into.
