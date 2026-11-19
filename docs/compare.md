# Brebus beside other desktop apps for local models

As of **19 November 2026**, for Brebus v2026.11.19. Updated every week.

> **About this table.** The other apps' columns come from what their makers' official documentation, changelogs and releases say (the sources are below), not from trying them; they can be out of date or wrong. A correction is welcome as an [issue](https://github.com/prabanta-dev/janas/issues), with the page that shows it. Brebus's column says what has been tried; what is written but not yet tried says so.

✓ yes · ◐ partly · — no, or not in the documentation · B only in Bionic, LM Studio's separate agent app

## Platforms and licence

| Feature | Brebus | LM Studio | Jan | Ollama app | AnythingLLM | Msty Studio |
|---|---|---|---|---|---|---|
| *version* | *v2026.11.19 · 19 Nov* | *0.4.28 · 31 Oct* | *0.8.8 · 28 Oct* | *0.40.16 · 9 Nov* | *1.20.0 · 17 Nov* | *3.0.1 · 17 Nov* |
| Linux app | ✓ portable tar.gz; deb, rpm being prepared | ✓ AppImage, deb; x64, ARM64 | ✓ Flatpak, deb, AppImage | — no window on Linux (CLI only) | ✓ AppImage x64, ARM64 | ✓ AppImage, deb |
| Windows app | ✓ x64 zip; exe, msi being prepared | ✓ x64, ARM64 | ✓ x64, ARM64 | ✓ x64, ARM64 | ✓ x64, ARM64 | ✓ x64, ARM64 |
| macOS app | — not yet | ◐ Apple Silicon only | ◐ Intel Macs: remote models only | ✓ | ✓ | ✓ |
| Open source | ✓ GPL-3.0 | — closed (CLI, SDK: MIT) | ✓ Apache-2.0 | ✓ MIT | ✓ MIT | — closed, paid tiers |
| Window | ✓ native C, the system's own webview | not stated | ✓ Tauri | ✓ Go, native webview | not stated | not stated |
| Telemetry, account | ✓ none; no account | ✓ none stated; no account | ✓ opt-in | ✓ none stated; account for cloud | — telemetry on until turned off | ✓ "zero"; account for paid tiers |

## Models

| Feature | Brebus | LM Studio | Jan | Ollama app | AnythingLLM | Msty Studio |
|---|---|---|---|---|---|---|
| *version* | *v2026.11.19 · 19 Nov* | *0.4.28 · 31 Oct* | *0.8.8 · 28 Oct* | *0.40.16 · 9 Nov* | *1.20.0 · 17 Nov* | *3.0.1 · 17 Nov* |
| Curated catalog | ✓ converted and checked | ✓ | ✓ | ✓ | ✓ | ✓ |
| Search Hugging Face | ✓ by words or address | ✓ | ✓ by model ID | — | ◐ pull a repo | ✓ |
| Will it run here? | ✓ fits / needs more; each Hub file checked: likely to run, not supported yet | ✓ memory estimate | ✓ fits / may be slow / won't fit | — | — | ✓ |
| Downloads | ✓ 4 at once, resumed, SHA-256 checked, room on disk checked | ◐ retried | ◐ retried | — | — | — |
| Models bigger than memory | ✓ experts streamed from the SSD | — mmap only | ◐ MoE experts kept on GPU | — | — | — |
| GPU | ◐ Vulkan; CUDA and HIP (ROCm) written, to be tried on a card; Metal with macOS | ✓ CUDA, Vulkan, ROCm, Metal, MLX | ✓ CUDA, Vulkan, Metal, MLX | ✓ CUDA, Vulkan, ROCm, Metal, MLX | ✓ CUDA, Vulkan, Metal | ◐ not named |
| NPU | ◐ Intel: Linux; Windows built, to be tried; chosen by Janas or by you | ◐ Apple Neural Engine, prompt only | — | — | ✓ AMD XDNA2, Qualcomm | — |
| Several models loaded | — one at a time, unloaded when idle (next) | ✓ JIT, TTL, auto-evict | ✓ | ✓ | ✓ | ✓ |

## Chat

| Feature | Brebus | LM Studio | Jan | Ollama app | AnythingLLM | Msty Studio |
|---|---|---|---|---|---|---|
| *version* | *v2026.11.19 · 19 Nov* | *0.4.28 · 31 Oct* | *0.8.8 · 28 Oct* | *0.40.16 · 9 Nov* | *1.20.0 · 17 Nov* | *3.0.1 · 17 Nov* |
| Projects / folders | ✓ a project is a folder of yours | ✓ folders | ✓ projects | — | ✓ workspaces | ✓ |
| Search all chats | ✓ with filters: where, when, model, archived | ✓ | ✓ | — | — | ✓ |
| Export | ✓ Markdown, text, PDF | ✓ PDF, Markdown, text | — | — | ◐ logs | ✓ |
| Edit, regenerate, branch | ✓ edit and ask again; replies paged; a branch from any reply; duplicate | ◐ regenerate, paged | ✓ branches | — | — | ✓ branches, split chats |
| Formulas, diagrams | ✓ KaTeX, Mermaid | ✓ LaTeX, Mermaid | ◐ LaTeX | ◐ LaTeX | — | ✓ LaTeX, Mermaid |
| Documents (RAG) | ✓ PDF, Office, LibreOffice, EPUB, web; words and meaning; no OCR | ✓ PDF, docx, txt | ✓ + xlsx, pptx | ◐ drop text or PDF | ✓ + OCR | ✓ |
| Pictures (vision) | ✓ | ✓ | ✓ + video, audio | ✓ | ✓ | ✓ |
| Dictation | ✓ Whisper on this computer | B | — | — | ✓ | — |
| Read aloud | ✓ system voices | B | — | ◐ CLI and API only | ✓ | — |
| Artifacts beside the chat | ✓ Claude's kinds: HTML, React, Mermaid, SVG, Markdown, code; no network | B Canvas | ◐ HTML, SVG, experimental | — | ◐ charts, slides | — |
| Run code in a sandbox | ✓ Python, Linux and Windows, each run allowed | — | ✓ shell | — | — | — |
| MCP tools | ✓ + OAuth, tried on Linux and Windows; asks before tools that write; Janas's own services | ✓ + OAuth | ✓ + OAuth | — | ✓ | ✓ + OAuth |
| Web search | ✓ Janas's web service | B | ✓ no key | ◐ account | ✓ agent | ◐ |
| JSON by schema | ✓ | ✓ | ✓ | ✓ | — | — |

## Server for other programs

| Feature | Brebus | LM Studio | Jan | Ollama app | AnythingLLM | Msty Studio |
|---|---|---|---|---|---|---|
| *version* | *v2026.11.19 · 19 Nov* | *0.4.28 · 31 Oct* | *0.8.8 · 28 Oct* | *0.40.16 · 9 Nov* | *1.20.0 · 17 Nov* | *3.0.1 · 17 Nov* |
| OpenAI API | ✓ | ✓ | ✓ | ✓ | ◐ unverified | — |
| Anthropic API | ✓ | ✓ | ✓ | ✓ | — | — |
| Ollama API | ✓ | — | — | ✓ | — | ✓ bundled Ollama |
| Parallel requests | ✓ from the memory | ✓ 4 | ✓ | ✓ | — | — |
| Network, key | ✓ LAN with a key | ✓ | ✓ + trusted hosts | ◐ no key | ✓ | ◐ |
| HTTPS | ✓ a certificate made by Brebus, its fingerprint shown; Linux and Windows | — | — | — | — | — |
| Without a window | ✓ at login, model loaded | ✓ daemon | ◐ | ✓ | — | — |
| CLI, SDK | ✓ janas-*; C, FreeBASIC, Node.js | ✓ lms; TS, Python | ◐ separate CLI | ✓ CLI; SDKs | ◐ API | — |
| Other devices | — | ✓ LM Link, iPhone app | — | ◐ cloud | ✓ mobile | ✓ |
| Plugins | ◐ skills in Claude's format, tried without a model yet; MCP services | ✓ | ✓ extensions | — | ✓ skills | ◐ |
| Connectors (services, desktop apps) | ◐ 13 web services a click each, OAuth; .mcpb extensions; not yet tried with a real service | ◐ MCP by hand, OAuth | ◐ MCP, OAuth | — | ◐ MCP | ◐ MCP, OAuth |

## Sources

Read on 18 November 2026.

- **LM Studio**: <https://lmstudio.ai/docs/app>, <https://lmstudio.ai/docs/app/system-requirements>, <https://lmstudio.ai/changelog>, <https://lmstudio.ai/docs/bionic>, <https://lmstudio.ai/app-privacy>
- **Jan**: <https://docs.jan.ai/desktop>, <https://docs.jan.ai/desktop/privacy>, <https://github.com/janhq/jan/releases>
- **Ollama**: <https://docs.ollama.com>, <https://ollama.com/blog/new-app>, <https://github.com/ollama/ollama/releases>
- **AnythingLLM**: <https://docs.anythingllm.com>, <https://github.com/Mintplex-Labs/anything-llm/releases>
- **Msty Studio**: <https://docs.msty.ai/studio>, <https://msty.ai/pricing/>, <https://msty.ai/resources/changelog/studio/>
- **Brebus**: [Brebus's guide](brebus.md), [ChangeLog](../ChangeLog.md)
