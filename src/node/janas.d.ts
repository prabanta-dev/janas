// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maurizio Cammalleri
/**
 * janas.node - Janas-LLM in Node.js: a model opened inside the process, its
 * replies read as they are written. Built by build.sh beside the programs
 * (bin/x86_64-linux/janas.node), Linux for now. It speaks Node-API version
 * 8, which Node has had since 12.22; it was tried with Node 20.
 *
 * It is a native module: load it with require, in an ES module through
 * createRequire, and take its types from this file:
 *
 *     import { createRequire } from "node:module";
 *     import type * as Janas from "./janas.js";
 *     const janas = createRequire(import.meta.url)("./janas.node") as typeof Janas;
 *
 *     const model = await janas.open("qwen3-4b-q4km.jns");
 *     const chat = await model.chat({ thinking: false });
 *     for await (const piece of chat.send("Hello!"))
 *         if (piece.type === "text") process.stdout.write(piece.text);
 *     await model.close();
 *
 * Each model runs on a thread of its own, so JavaScript's thread never waits
 * for it. A model holds one conversation: replies asked for while another is
 * written wait for it, in the order they were asked. Wrong arguments throw a
 * TypeError at once; what fails in the engine rejects the promise, or the
 * reply's iteration, with a JanasError. An idle model does not keep Node
 * running; one left open is closed when Node leaves.
 *
 * The engine's environment variables (JANAS_GPU, JANAS_NPU, JANAS_PROMPT_ON
 * and the others of docs/chat.md) apply as in the programs, read when a model
 * opens.
 */

/** The build of Janas it was made from (`git describe`). */
export declare const version: string;

/** The version of the library's ABI (JANAS_LLM_ABI_VERSION, 1). */
export declare function abiVersion(): number;

/**
 * Opens a model file (.jns): the weights stay on disk, the experts are
 * streamed from it and the most used kept in memory. Opening takes seconds:
 * the expert cache is set up, and the engine reads what it learnt of this
 * machine.
 */
export declare function open(path: string, options?: OpenOptions): Promise<Model>;

export interface OpenOptions {
    /** Context, in tokens (16384). */
    context?: number;
    /** Memory for the expert cache, GiB; 0 or absent: the free memory, less a margin. */
    cacheGiB?: number;
    /** Memory left to other programs when the cache takes the rest, GiB (a fifth of the machine's, 2 at least). */
    reserveGiB?: number;
    /** The model's multi-token prediction block, for faster replies: its path, or true to look beside the model. */
    mtp?: string | boolean;
    /** A small model of the same family that guesses the next tokens (Qwen3-0.6B for Qwen3-4B). */
    draft?: string;
    /** "auto" (default: max on mains power, eco on battery), "eco" (no GPU where it would only buy speed with power), "max". */
    mode?: "auto" | "eco" | "max";
    /** false: never give the GPU work. */
    gpu?: boolean;
    /** false: do not fill the expert cache, as the model opens, with the experts this machine used most. */
    preload?: boolean;
    /** Bits of the experts' down matrix read, for files that hold it in planes: 2, 4 or 6 (all of it). */
    expertBits?: 2 | 4 | 6;
    /** 4: the output head read at four bits (a few per cent faster, answers a little further from the weights). */
    headBits?: 4 | 6;
    /** How the attention scores are computed: "exact", "fast" (sixteen-bit integers), "auto". */
    attention?: "auto" | "exact" | "fast";
    /** The lowest quality the engine may choose to be faster where memory is short: "low" (default) the experts at four bits and the light ones not in memory left out, "medium" four bits only, "high" nothing lowered. expertBits and headBits win over it. */
    quality?: "low" | "medium" | "high";
}

/** An open model. */
export declare class Model {
    private constructor();
    readonly path: string;
    /** The name in its metadata ("Qwen3 4B"), empty if it has none. */
    readonly name: string;
    /** How it is made ("qwen3, 36 layers, ..."). */
    readonly description: string;
    /** The system message Janas's programs give it: its name is Janas, and it says which model it thinks with. */
    readonly defaultSystem: string;
    /** Whether it reasons before replying and can be told not to (ChatOptions.thinking). */
    readonly reasons: boolean;
    /** The MTP block opened with it, or false. */
    readonly mtp: string | false;
    /** Its conversation: one at a time per model (close it before asking another). */
    chat(options?: ChatOptions): Promise<Chat>;
    /** How its preparations go, without waiting for the engine: the expert cache filling, the NPU made ready. */
    status(): ModelStatus;
    /** Stops its replies, and closes it; called again, it waits for the same close. */
    close(): Promise<void>;
}

export interface ModelStatus {
    open: boolean;
    /** a job is posted and not done: opening, a reply, a close */
    busy: boolean;
    /** the expert cache filled from this machine's profile: done of total experts */
    preload: { done: number; total: number };
    /** the NPU made ready (the first prompts of a session wait for it): graphs done of total, seconds */
    npu: { preparing: boolean; done: number; total: number; seconds: number };
}

/** The parameters of a conversation, from the next reply on. */
export interface ChatOptions {
    /** The system message; only before the first message (or after reset). */
    system?: string;
    /** 0: always the most likely token (0.7). */
    temperature?: number;
    /** 0: no limit (20). */
    topK?: number;
    /** (0.8) */
    topP?: number;
    /** (0) */
    minP?: number;
    /** 0: a new one each chat (default). */
    seed?: number;
    /** Draft tokens ahead when it pays: the same text, sooner (true). */
    speculate?: boolean;
    /** Tokens of a reply, its reasoning included; 0: no limit. */
    maxReply?: number;
    /** Tokens of the answer, the reasoning not counted; 0: no limit. */
    maxAnswer?: number;
    /** Tokens of the reasoning, closed by the engine past them; 0: no limit. */
    maxReasoning?: number;
    /** The longest prompt, in tokens: a longer one fails with ELIMIT, nothing cut; 0: the context. */
    maxInput?: number;
    /** Reasoning before the reply, for the models that do it: true, false, or null for the model's own way. */
    thinking?: boolean | null;
    /** When the context fills, a recap of the oldest exchanges stays in it (true). */
    recap?: boolean;
    presencePenalty?: number;
    frequencyPenalty?: number;
    /** Conversations kept computed besides the current one, for load() switching between several (0). */
    keep?: number;
}

export type Role = "system" | "user" | "assistant" | "tool";

export interface Message {
    role: Role;
    content: string;
}

/** A model's conversation. */
export declare class Chat {
    private constructor();
    /** A user message, in the conversation kept here; the reply follows. */
    send(text: string, options?: ReplyOptions): Reply;
    /**
     * A picture (the bytes of a PNG or a JPEG) for the next message, with a
     * model that sees: Qwen3.5 or Gemma 4 with its .vision.jns beside it.
     */
    image(data: Uint8Array | ArrayBuffer): Promise<void>;
    /**
     * A whole conversation, as an HTTP client keeps it (only the first may be
     * the system's; the last is the user's): what it shares with the one
     * already computed is not read again.
     */
    load(messages: readonly Message[], options?: ReplyOptions): Reply;
    /** New parameters, from the next reply on (and the system message, before the first). */
    set(options: ChatOptions): Promise<void>;
    /** Forgets the conversation, and the system message. */
    reset(): Promise<void>;
    /** Stops its replies and ends it: the model may have another. */
    close(): Promise<void>;
    /**
     * The conversation as it was said: what was sent and what the model
     * replied (with its reasoning and its tool calls, once a reply is over),
     * or the conversation loaded and its reply.
     */
    messages(): Promise<SavedMessage[]>;
    /** Its title: the first line of the first question, until one is given. */
    title(): Promise<string>;
    /** A title of the caller's; none: from the first question again. */
    setTitle(title?: string): Promise<void>;
    /**
     * Saves the conversation as it goes, a file each, in dir (none: the
     * default, chatsDir(); null: no longer).
     */
    autosave(dir?: string | null): Promise<void>;
    /** The file it is saved to as it goes ("": none yet). */
    file(): Promise<string>;
    /** Writes it to a file, in the saved form (JSON Lines). */
    save(path: string): Promise<void>;
    /**
     * Takes a conversation up from a file (saved, or in Markdown or plain
     * text): it is loaded whole with the next message.
     */
    restore(path: string): Promise<void>;
    /** Writes it for people, in Markdown (the default) or plain text. */
    export(path: string, format?: "markdown" | "text"): Promise<void>;
}

export interface SavedMessage {
    role: "system" | "user" | "assistant" | "tool";
    text: string;
    reasoning?: string;
    calls?: { name: string; arguments: string }[];
}

export interface SavedChat {
    path: string;
    title: string;
    model: string;
    modelFile: string;
    /** when it was last written, in milliseconds since 1970 (as Date.now()) */
    updated: number;
    /** the system message not counted */
    messages: number;
}

/** The default folder of saved conversations. */
export declare function chatsDir(): string;
/** A folder's saved conversations (none: the default), the most recent first. */
export declare function listChats(dir?: string): SavedChat[];

export interface ReplyOptions {
    /** Stops the reply when it aborts (the client of a request gone). */
    signal?: AbortSignal;
}

/** A piece of the reply's text: the model's reasoning, or its answer. */
export interface TextPiece {
    readonly type: "text";
    readonly text: string;
    readonly reasoning: boolean;
}

/** How much of the prompt is read: one at each block of it. */
export interface ProgressPiece {
    readonly type: "progress";
    readonly read: number;
    readonly total: number;
}

export type Piece = TextPiece | ProgressPiece;

/**
 * A reply, as the model writes it: read it with for await, or whole with
 * text(). Leaving the loop early (break, return, an exception) stops it.
 */
export declare class Reply implements AsyncIterable<Piece> {
    private constructor();
    next(): Promise<IteratorResult<Piece, undefined>>;
    return(value?: unknown): Promise<IteratorResult<Piece, undefined>>;
    [Symbol.asyncIterator](): Reply;
    /** Stops it where it is; the pieces already written can still be read. */
    stop(): void;
    /** The answer, its reasoning left out, once the reply is over; the pieces go to it from then on. */
    text(): Promise<string>;
    /** Its counters, once it is over. */
    stats(): Promise<ReplyStats>;
}

export interface ReplyStats {
    /** why it ended: the model ended it, max tokens, the context full, an error, tool calls, stop() or an abort */
    finish: "stop" | "length" | "context" | "error" | "tools" | "stopped" | null;
    /** the whole prompt, of which cachedTokens already computed; inputTokens read for this reply */
    promptTokens: number;
    cachedTokens: number;
    inputTokens: number;
    outputTokens: number;
    reasoningTokens: number;
    contextUsed: number;
    contextSize: number;
    /** draft tokens guessed, and kept */
    drafted: number;
    accepted: number;
    /** seconds reading the prompt, writing the reply, in all, to the first token */
    inputSeconds: number;
    outputSeconds: number;
    totalSeconds: number;
    firstTokenSeconds: number;
    /** the threads the engine chose for reading and for writing */
    inputThreads: number;
    outputThreads: number;
    /** the routed experts used, read from the disk, the bytes read, the time waiting for them */
    expertsUsed: number;
    expertsRead: number;
    bytesRead: number;
    ioWaitSeconds: number;
    peakRssBytes: number;
    /** maxReasoning closed the reasoning, not the model */
    reasoningCut: boolean;
}

/**
 * Opens a Whisper model (.jns, made by hf2jns_whisper) on a thread of its
 * own: recordings are transcribed in the language they are spoken in, or
 * translated into English. A voice activity detector (silero-vad.jns) and a
 * speaker model (wespeaker-resnet34.jns) beside it are opened with it.
 */
export declare function openSpeech(path: string, options?: SpeechOptions): Promise<Speech>;

export interface SpeechOptions {
    /** Threads; 0 or absent: one a performance core. */
    threads?: number;
    /** The voice activity detector, for chunking (default: silero-vad.jns beside the model). */
    vadModel?: string;
    /** The speaker model, for diarize (default: wespeaker-resnet34.jns beside the model). */
    speakerModel?: string;
}

/** An open speech model: one recording at a time, the others waiting in order. */
export declare class Speech {
    private constructor();
    readonly path: string;
    /** The bytes of a sound file: WAV, FLAC, MP3, Ogg, M4A, WebM; any rate and channels. */
    transcribe(data: Uint8Array | ArrayBuffer, options?: TranscribeOptions): Promise<Transcription>;
    /** Closes it once the recordings asked for are done. */
    close(): Promise<void>;
}

export interface TranscribeOptions {
    /** Into English (false). */
    translate?: boolean;
    /** "it" or "italian"; absent: found out. */
    language?: string;
    /** Text said before, or the words and spelling expected (names, terms). */
    prompt?: string;
    /** 0: the likeliest words, a higher one only where they loop or are unlikely (default); up to 1: sampled at it. */
    temperature?: number;
    seed?: number;
    /** The words' times too (false). */
    words?: boolean;
    /** Only the speech read: "auto" (loudness normalized, then voice activity detection), or the detector's settings. */
    chunking?: "auto" | { threshold?: number; prefixMs?: number; silenceMs?: number };
    /** Who speaks when (false): turns, with the known speakers' names, at most 8, or A, B... */
    diarize?: boolean;
    /** Known speakers: a name and a short recording (2-10 s) of each. */
    speakers?: readonly { name: string; data: Uint8Array | ArrayBuffer }[];
    /** Stops it between windows of thirty seconds: it rejects then ("stopped"). */
    signal?: AbortSignal;
}

export interface Transcription {
    text: string;
    /** its code ("it") */
    language: string;
    /** seconds */
    duration: number;
    segments: {
        start: number;
        end: number;
        /** beginning with a space, as the model writes it */
        text: string;
        avgLogprob: number;
        noSpeechProb: number;
        temperature: number;
        compressionRatio: number;
    }[];
    /** with words: true */
    words?: { start: number; end: number; text: string; probability: number }[];
    /** with diarize: true */
    turns?: { start: number; end: number; speaker: string; text: string }[];
}

/** The error a promise or a reply rejects with when the engine fails: its code is
    the suffix of the library's JANAS_LLM_E... (EOPEN for JANAS_LLM_EOPEN), and
    ECLOSED for a model or a chat closed. */
export interface JanasError extends Error {
    code:
        | "EINVAL"
        | "ENOMEM"
        | "EOPEN"
        | "EMODEL"
        | "EFULL"
        | "EBUSY"
        | "ESMALL"
        | "EFAIL"
        | "ELIMIT"
        | "ECLOSED";
}
