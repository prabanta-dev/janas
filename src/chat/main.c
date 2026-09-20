/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-chat - a terminal chat with a model, on the public API of
 * libjanas_llm (include/janas/llm.h).
 *
 * On a terminal it takes the window: a banner, the conversation scrolling
 * under it, and a footer that keeps the model, the context used, the last
 * speed and the state of the expert cache under the eye. Lines are edited
 * and recalled with the arrows (src/chat/edit.h). Piped in or out it stays
 * the plain thing it was, for scripts and measurements.
 *
 * Lines starting with / are commands (/help lists them). A message may be
 * more than a line: a backslash and Enter add one, and the backslash is not
 * part of what is sent. Ctrl-C stops a reply, Ctrl-D quits.
 */
#define _XOPEN_SOURCE 700 /* wcwidth, for the width of what is printed */
#include <errno.h>
#include <locale.h>
#include <sys/stat.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/sysinfo.h"
#include "edit.h"
#include "janas/llm.h"
#include "markdown.h"
#include "mcp_chat.h"
#include "scroll.h"
#include "term.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static volatile sig_atomic_t interrupted;
static volatile sig_atomic_t resized;
static struct janas_term term;
static struct janas_scroll conv; /* what the conversation printed */
static struct janas_hist history;
/* what the whole conversation has cost, so the bar can say the average and
   not only how the last reply went */
static double sess_tokens, sess_seconds;

/* The commands, for Tab and for the help. */
static const char *const commands[] = {
    "/help",    "/about", "/stats", "/reset",   "/system",   "/temp",
    "/spec",    "/think", "/mode",  "/experts", "/markdown", "/mcp",
    "/context", "/quit",  "/exit",  NULL};

static void on_sigwinch(int sig)
{
    (void)sig;
    resized = 1;
}

/*
 * The footer is made of parts, and a part that does not fit is left out
 * rather than cut off at the edge. A big font is a narrow window, and the
 * line used to lose whatever came last in it - which was the speed, the
 * one number a reader watches. Each part says how long it is to be kept:
 * the speed goes last of all, the state of the cache first.
 */
#define FOOT_PARTS 8

struct foot {
    char text[112];
    int keep;
};

static void foot_add(struct foot *p, int *np, int keep, const char *fmt, ...)
{
    va_list ap;
    if (*np >= FOOT_PARTS)
        return;
    p[*np].keep = keep;
    va_start(ap, fmt);
    vsnprintf(p[*np].text, sizeof(p[*np].text), fmt, ap);
    va_end(ap);
    (*np)++;
}

/* The line at the bottom: what is worth having under the eye while the
   conversation scrolls above it. */
/* what main asked for, read by the footer: the address of its own, so a
   /think is seen without a second copy that could fall out of step */
static const struct janas_llm_chat_params *asked;

static void footer(const janas_llm *llm, const janas_llm_chat *c,
                   const char *state)
{
    if (!term.tty)
        return;
    char name[96] = "";
    int32_t len;
    janas_llm_name(llm, name, sizeof(name), &len);
    struct janas_llm_chat_stats s = {.size = sizeof(s)};
    int have = c && janas_llm_chat_stats(c, &s) == JANAS_LLM_OK;
    int32_t used = 0, most = 0;
    janas_llm_experts(llm, &used, &most);
    int64_t done = 0, total = 0;
    janas_llm_preload(llm, &done, &total);

    struct foot part[FOOT_PARTS];
    int np = 0;
    if (name[0])
        foot_add(part, &np, 3, "%s", name);
    foot_add(part, &np, 2, "%d bit", janas_llm_expert_bits(llm));
    foot_add(part, &np, 2, "%d/%d experts", used, most);
    if (have && s.context_size)
        foot_add(part, &np, 4, "context %u/%u", s.context_used, s.context_size);
    /* only where it is on, and only where it means something: it is the one
       switch that changes how long a reply takes before a word of it shows */
    if (asked && asked->thinking != 0 && janas_llm_reasons(llm))
        foot_add(part, &np, 4, "reasoning");
    if (have && s.output_seconds > 0)
        foot_add(part, &np, 7, "%.1f tok/s",
                 s.output_tokens / s.output_seconds);
    if (sess_seconds > 0)
        foot_add(part, &np, 6, "avg %.1f tok/s", sess_tokens / sess_seconds);
    if (total > done)
        foot_add(part, &np, 1, "cache %d%%",
                 (int)(100 * done / (total ? total : 1)));
    if (state && *state)
        foot_add(part, &np, 5, "%s", state);

    /* drop the least worth keeping until what is left fits, and never the
       last one standing: one part cut off says more than an empty line */
    int in[FOOT_PARTS];
    for (int i = 0; i < np; i++)
        in[i] = 1;
    for (;;) {
        int width = 1, worst = -1, kept = 0;
        for (int i = 0; i < np; i++) {
            if (!in[i])
                continue;
            width +=
                (kept ? 2 : 0) + term_cols(part[i].text, strlen(part[i].text));
            kept++;
            if (worst < 0 || part[i].keep < part[worst].keep)
                worst = i;
        }
        if (width <= term.cols || kept < 2)
            break;
        in[worst] = 0;
    }

    char line[512];
    size_t n = 1;
    line[0] = ' ';
    for (int i = 0; i < np && n < sizeof(line); i++) {
        if (!in[i])
            continue;
        n += (size_t)snprintf(line + n, sizeof(line) - n, "%s%s",
                              n > 1 ? "  " : "", part[i].text);
    }
    if (n >= sizeof(line))
        n = sizeof(line) - 1;
    line[n] = 0;
    term_footer(&term, line);
}

/*
 * The footer while the editor waits for a key: the filling of the cache
 * goes on meanwhile, and a footer drawn only at the start and after each
 * reply showed whatever it was at that moment - 60% at one start, 0% at
 * the next - and kept it. Drawn again when the share changes, and once
 * more when the filling is over, so that it goes.
 */
static const janas_llm *idle_llm;
static const janas_llm_chat *idle_chat;

static int footer_idle(void)
{
    static int shown = -2;
    int64_t done = 0, total = 0;
    janas_llm_preload(idle_llm, &done, &total);
    int pct = total > done ? (int)(100 * done / total) : -1;
    if (pct == shown)
        return 0;
    shown = pct;
    footer(idle_llm, idle_chat, NULL);
    return 1;
}

static void on_sigint(int sig)
{
    (void)sig;
    interrupted = 1;
    mcpc_cancel(); /* a tool running stops too */
}

static void usage(void)
{
    fprintf(
        stderr,
        "usage: janas-chat <model.jns> [options]\n"
        "  --mtp <file>     multi-token prediction block (faster replies)\n"
        "  --draft <file>   a small model of the same family to guess the\n"
        "                   next tokens; worth it on a dense model, where\n"
        "                   several can be checked in one pass for little\n"
        "                   more than one (Qwen3-0.6B for Qwen3-4B: 1.14x)\n"
        "  --ctx <tokens>   context length (default 16384)\n"
        "  --cache <GiB>    RAM for streamed experts (default: automatic)\n"
        "  --reserve <GiB>  memory left to other programs when the cache is\n"
        "                   automatic (default: a fifth of the machine's)\n"
        "  --bits <2|4|6>   bits per weight of the experts' down matrix\n"
        "                   (files that hold it in planes; default: the\n"
        "                   memory decides)\n"
        "  --attention <exact|fast>\n"
        "                   the attention scores: the query as it is, or\n"
        "                   to sixteen bits with an exact integer sum,\n"
        "                   twice as fast for an error of 3e-5\n"
        "                   (default: fast above 16384 tokens of context)\n"
        "  --no-preload     do not fill the expert cache at start with\n"
        "                   the experts used most on this machine\n"
        "  --no-recap       when the context fills, forget the oldest\n"
        "                   exchanges without summing them up first\n"
        "  --system <text>  system message (default: the assistant is "
        "called Janas,\n"
        "                   and which model it thinks with; \"\" for "
        "none)\n"
        "  --temp <t>       temperature (default 0.7; 0: greedy)\n"
        "  --top-k <k>      (default 20)   --top-p <p> (default 0.8)\n"
        "  --min-p <p>      (default 0)    --seed <n> (default: random)\n"
        "  --max <tokens>   longest reply (default: no limit)\n"
        "  --no-spec        no speculative decoding\n"
        "  --mode <m>       auto (default), eco or max\n"
        "  --think <on|off> reasoning before replying, for models that do\n"
        "                   it (default off: it is slow, and a small model\n"
        "                   reasons at length about very little)\n"
        "  --stats          speed report after every reply\n"
        "  --no-markdown    print the model's marks instead of "
        "reading them\n"
        "  --no-gpu         never give the GPU work, whatever the mode "
        "says\n"
        "  --mcp-config <file>\n"
        "                   the MCP servers whose tools the model may call\n"
        "                   (default ~/.config/janas/mcp.json, in the\n"
        "                   format of the other clients: mcpServers)\n"
        "  --no-mcp         start no MCP server\n"
        "  --progress <s>   a line on stderr every s seconds while a long\n"
        "                   prompt is read, and one after each reply\n"
        "  --mcp-instructions\n"
        "                   add what the MCP servers say of their tools to\n"
        "                   the system message, marked as theirs (off by\n"
        "                   default: it is text their authors wrote, which\n"
        "                   the model would read as instructions)\n"
        "  --mcp-auto       run the tools the model calls without asking\n"
        "                   (needed on a pipe, where nobody can answer)\n");
}

/*
 * The help, as a command and what it does under it rather than as two
 * columns: a column of descriptions is a column of wrapped fragments in a
 * narrow window, and the command is easier to find when it stands alone in
 * the colour of the program. The page lays the text out, so nothing here
 * decides where a line breaks.
 */
static const struct {
    const char *cmd; /* NULL: a paragraph of its own, at the end */
    const char *what;
} help_lines[] = {
    {"/help", "this list"},
    {"/about", "what this is, who wrote it, and what it is running on"},
    {"/stats [on|off]", "speed report after every reply (toggles)"},
    {"/context", "how full the context is, and with what"},
    {"/reset", "new conversation"},
    {"/system", "show the system message"},
    {"/system <text>", "new conversation with this system message"},
    {"/system off", "new conversation with no system message"},
    {"/temp <t>", "temperature (0: greedy)"},
    {"/spec [on|off]", "speculative decoding (toggles)"},
    {"/mode [auto|eco|max]", "power mode, and what the engine chose"},
    {"/experts [n]",
     "experts per token (fewer: faster, a little less accurate)"},
    {"/think [on|off]",
     "reasoning before replying, for models that do it (off by default: the "
     "bar says reasoning while it is on)"},
    {"/markdown [on|off]", "read the model's marks, or print them"},
    {"/gpu [on|off]", "whether the GPU may be given work"},
    {"/mcp", "the MCP servers running and the tools they give the model"},
    {"/mcp auto [on|off]",
     "run the tools the model calls without asking first (off by default: "
     "every call is shown and waits for y, n or a - always that tool)"},
    {"/save-config",
     "keep these settings for the next start (asks to overwrite)"},
    {"/load-config", "read them back, losing the current ones (asks first)"},
    {"/del-config", "forget them: the next start uses the chat's own"},
    {"/quit", "exit (also Ctrl-D)"},
    {NULL, "A backslash and Enter add a line to the message (the backslash "
           "is not sent); Enter alone sends it. Ctrl-C stops a reply."},
    {NULL, "The arrows move in the message - up and down between its lines "
           "when it has more than one - and bring back earlier prompts from "
           "its first and last line (kept between sessions); Home, End, "
           "Ctrl-A/E/K/U/W and Ctrl-L work as usual."},
    {NULL, "Page Up and Page Down look back over the conversation, Ctrl-Home "
           "and Ctrl-End go to its beginning and its end; the line being "
           "written stays where it is, and sending brings the end back."}};

static void help(const janas_llm *llm)
{
    (void)llm;
    for (size_t i = 0; i < sizeof(help_lines) / sizeof(*help_lines); i++)
        if (help_lines[i].cmd)
            term_field(&term, help_lines[i].cmd, help_lines[i].what);
        else
            term_printf(&term, "%s\n\n", help_lines[i].what);
}

static int on_off(const char *arg, int current)
{
    if (strcmp(arg, "on") == 0)
        return 1;
    if (strcmp(arg, "off") == 0)
        return 0;
    return !current;
}

static void report(const janas_llm_chat *c)
{
    struct janas_llm_chat_stats s = {.size = sizeof(s)};
    if (janas_llm_chat_stats(c, &s) != JANAS_LLM_OK)
        return;
    double in = s.input_seconds > 0 ? s.input_tokens / s.input_seconds : 0;
    double out = s.output_seconds > 0 ? s.output_tokens / s.output_seconds : 0;
    term_printf(&term,
                "%s[input %u tokens, %.2f s, %.1f tok/s | output %u tokens, "
                "%.2f s, %.1f tok/s | total %.2f s | context %u/%u",
                T_USER(&term), s.input_tokens, s.input_seconds, in,
                s.output_tokens, s.output_seconds, out, s.total_seconds,
                s.context_used, s.context_size);
    if (s.drafted > 0)
        term_printf(&term, " | drafts %u/%u accepted", s.accepted, s.drafted);
    if (s.auto_off_passes > 0)
        term_printf(&term, " | drafts off %u passes (the engine judged)",
                    s.auto_off_passes);
    if (s.experts_used > 0)
        term_printf(&term,
                    " | experts %.1f%% cached, %.0f MB read, %.2f s waiting",
                    100.0 * (double)(s.experts_used - s.experts_read) /
                        (double)s.experts_used,
                    (double)s.bytes_read / 1e6, s.io_wait_seconds);
    term_printf(&term,
                " | first token %.2f s, prompt built in %.3f s, read with %d "
                "threads (%.1f s of processor), written with %d | context "
                "memory %.0f MB, peak %.0f MB",
                s.first_token_seconds, s.prepare_seconds, s.input_threads,
                s.input_cpu_seconds, s.output_threads,
                (double)s.context_bytes / 1e6, (double)s.peak_rss_bytes / 1e6);
    term_printf(&term, "]%s\n", T_RESET(&term));
}

/* The context, and what fills it: the last reply's prompt by its parts. */
static void context_page(const janas_llm_chat *c)
{
    struct janas_llm_chat_stats s = {.size = sizeof(s)};
    if (janas_llm_chat_stats(c, &s) != JANAS_LLM_OK || !s.context_size) {
        term_printf(&term, "nothing sent yet\n");
        return;
    }
    uint32_t free_tokens =
        s.context_size > s.context_used ? s.context_size - s.context_used : 0;
    term_printf(&term, "context %u of %u tokens (%.1f%%), %u free\n",
                s.context_used, s.context_size,
                100.0 * s.context_used / s.context_size, free_tokens);
    term_printf(&term,
                "last prompt %u tokens: system %u (tools %u), conversation "
                "%u, last message %u; %u read, %u already computed\n",
                s.prompt_tokens, s.prompt_system, s.prompt_tools,
                s.prompt_history, s.prompt_last, s.input_tokens,
                s.cached_tokens);
    term_printf(&term,
                "its keys and values take %.0f MB; when the context fills, "
                "the oldest exchanges go (summed up first, unless "
                "--no-recap)\n",
                (double)s.context_bytes / 1e6);
}

/*
 * What this is and what it is running on, in one page: the thing a reader
 * wants when they meet the program, and the thing a report of a problem
 * needs - the version, the model, the machine - which until now had to be
 * gathered by hand from three places.
 */
/*
 * What about() writes, caught instead of printed: the page has to be laid
 * out and scrolled, not thrown at the screen, and the terminal already
 * knows how to hand its output to somebody else (the conversation store
 * does the same).
 */
struct about_cap {
    char *buf;
    size_t n, cap;
};

static void about_sink(void *arg, const char *s, size_t n)
{
    struct about_cap *c = arg;
    if (c->n + n + 1 > c->cap) {
        size_t want = (c->n + n + 1) * 2;
        char *bigger = realloc(c->buf, want);
        if (!bigger)
            return;
        c->buf = bigger;
        c->cap = want;
    }
    memcpy(c->buf + c->n, s, n);
    c->n += n;
    c->buf[c->n] = 0;
}

static void about(const janas_llm *llm)
{
    char head[80];
    snprintf(head, sizeof(head), "Janas-Chat %s", JANAS_VERSION);
    const char *text[4] = {head, "Maurizio \"camauri\" Cammalleri, 2026",
                           "GPL-3.0-or-later", "github.com/prabanta-dev/janas"};
    const char *face[4] = {T_BOLD(&term), "", T_DIM(&term), T_DIM(&term)};
    term_beside(&term, text, face, 4, term.rows - 1, 1);
    term_printf(&term, "\n");

    char desc[1024];
    int32_t dl;
    if (janas_llm_describe(llm, desc, sizeof(desc), &dl) == JANAS_LLM_OK)
        term_field(&term, "model", desc);
    char cpu[128] = "";
    janas_cpu_name(cpu, sizeof(cpu));
    uint64_t total = janas_mem_total(), free_now = janas_mem_available();
    int mains = janas_on_mains();
    char machine[320];
    snprintf(machine, sizeof(machine),
             "%s, %ld threads, %.1f GiB of memory (%.1f free)%s",
             cpu[0] ? cpu : "processor unknown", sysconf(_SC_NPROCESSORS_ONLN),
             (double)total / (1 << 30), (double)free_now / (1 << 30),
             mains < 0 ? ""
             : mains   ? ", on mains"
                       : ", on battery");
    term_field(&term, "machine", machine);
    term_field(&term, "engine",
               "Janas-LLM " JANAS_VERSION ", C11, no dependency beyond the C "
               "library; the experts stream from the disk, so a model larger "
               "than the memory still runs");
}

/* A page as text, written by whoever draws it into a sink of our own. */
static char *catch_page(void (*draw)(const janas_llm *), const janas_llm *llm)
{
    struct about_cap c = {NULL, 0, 0};
    void (*sink)(void *, const char *, size_t) = term.sink;
    void *arg = term.sink_arg;
    term.sink = about_sink;
    term.sink_arg = &c;
    draw(llm);
    term.sink = sink;
    term.sink_arg = arg;
    return c.buf;
}

/* The default system message: the assistant is called Janas, the model is
   what it thinks with, and both are said, so that it guesses neither. Left
   to itself a model answers out of its training, and they very often claim
   to be somebody else's assistant. */
static char *default_system(const janas_llm *llm)
{
    int32_t n = 0;
    janas_llm_default_system(llm, NULL, 0, &n); /* the length */
    char *s = malloc((size_t)n + 1);
    if (s && janas_llm_default_system(llm, s, n + 1, &n) != JANAS_LLM_OK) {
        free(s);
        s = NULL;
    }
    return s;
}

/* A new conversation with this system message (NULL: none). */
/* What the MCP servers say of their tools, when --mcp-instructions asks
   for it: added after the system message, marked as theirs. */
static char *mcp_extra;

/* The system message, with the servers' instructions after it. */
static int32_t set_system(janas_llm_chat *c, const char *system)
{
    if (!mcp_extra)
        return system ? janas_llm_chat_system(c, system, -1) : JANAS_LLM_OK;
    size_t a = system ? strlen(system) : 0, b = strlen(mcp_extra);
    char *s = malloc(a + b + 3);
    if (!s)
        return JANAS_LLM_ENOMEM;
    snprintf(s, a + b + 3, "%s%s%s", system ? system : "", a ? "\n\n" : "",
             mcp_extra);
    int32_t rc = janas_llm_chat_system(c, s, -1);
    free(s);
    return rc;
}

static void restart(janas_llm_chat *c, const char *system)
{
    janas_llm_chat_reset(c);
    if (set_system(c, system) != JANAS_LLM_OK)
        term_printf(&term, "%s\n", janas_llm_last_error());
}

/* Reads one message (lines joined while they end with a backslash).
   Returns its length, or -1 at the end of the input. */
static long read_message(char **buf, size_t *cap, int tty)
{
    size_t n = 0;
    char *line = NULL;
    size_t lcap = 0;
    for (;;) {
        ssize_t l;
        if (tty) { /* our own editor, with the previous prompts */
            static char edited[8192];
            int r = line_read(&term, n == 0 ? "\u203a " : "  ", edited,
                              sizeof(edited), &history, commands);
            if (r == 0) {
                free(line);
                return n > 0 ? (long)n : -1;
            }
            if (r < 0) { /* the line was given up */
                n = 0;
                continue;
            }
            l = (ssize_t)strlen(edited);
            if ((size_t)l + 2 > lcap) {
                char *t = realloc(line, (size_t)l + 2);
                if (!t) {
                    free(line);
                    return -1;
                }
                line = t;
                lcap = (size_t)l + 2;
            }
            memcpy(line, edited, (size_t)l + 1);
            goto have_line;
        }
        errno = 0;
        l = getline(&line, &lcap, stdin);
        if (l < 0) {
            if (errno == EINTR && interrupted) {
                interrupted = 0;
                clearerr(stdin);
                term_write(&term, "\n", 1);
                n = 0;
                continue;
            }
            free(line);
            return n > 0 ? (long)n : -1;
        }
    have_line:
        while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r'))
            l--;
        int more = l > 0 && line[l - 1] == '\\';
        if (more)
            line[--l] = '\n';
        else
            line[l] = 0;
        if (n + (size_t)l + 1 > *cap) {
            *cap = 2 * (n + (size_t)l) + 256;
            char *t = realloc(*buf, *cap);
            if (!t) {
                free(line);
                return -1;
            }
            *buf = t;
        }
        memcpy(*buf + n, line, (size_t)l + (more ? 1 : 0));
        n += (size_t)l + (more ? 1 : 0);
        (*buf)[n] = 0;
        if (!more)
            break;
    }
    free(line);
    return (long)n;
}

/*
 * The options, read from words: the command line gives them, and so does the
 * file /save-config writes, which holds the same words. One reader for both,
 * so a setting can never mean one thing typed and another saved.
 */
struct chat_opts { /* where the words land: main keeps them, this names them */
    struct janas_llm_params *mp;
    struct janas_llm_chat_params *cp;
    const char **system_arg;
    int *stats, *markdown, *gpu;
    const char **mcp_config;
    int *mcp, *mcp_auto, *mcp_instructions;
};

static int take_options(int n, char **w, struct chat_opts *o)
{
    for (int i = 0; i < n; i++) {
        const char *a = w[i], *v = i + 1 < n ? w[i + 1] : NULL;
        if (strcmp(a, "--no-spec") == 0) {
            o->cp->speculate = 0;
            continue;
        }
        if (strcmp(a, "--stats") == 0) {
            *o->stats = 1;
            continue;
        }
        if (strcmp(a, "--no-markdown") == 0) {
            *o->markdown = 0;
            continue;
        }
        if (strcmp(a, "--no-gpu") == 0) {
            *o->gpu = 0;
            continue;
        }
        if (strcmp(a, "--no-preload") == 0) {
            o->mp->no_preload = 1;
            continue;
        }
        if (strcmp(a, "--no-recap") == 0) {
            o->cp->recap = 0;
            continue;
        }
        if (strcmp(a, "--no-mcp") == 0) {
            *o->mcp = 0;
            continue;
        }
        if (strcmp(a, "--mcp-auto") == 0) {
            *o->mcp_auto = 1;
            continue;
        }
        if (strcmp(a, "--mcp-instructions") == 0) {
            *o->mcp_instructions = 1;
            continue;
        }
        if (!v) {
            return -1;
        }
        i++;
        if (strcmp(a, "--mtp") == 0)
            o->mp->mtp_path = v;
        else if (strcmp(a, "--draft") == 0)
            o->mp->draft_path = v;
        else if (strcmp(a, "--think") == 0)
            o->cp->thinking = strcmp(v, "off") == 0 ? 0 : 1;
        else if (strcmp(a, "--mode") == 0)
            o->mp->mode = strcmp(v, "eco") == 0   ? JANAS_LLM_MODE_ECO
                          : strcmp(v, "max") == 0 ? JANAS_LLM_MODE_MAX
                                                  : JANAS_LLM_MODE_AUTO;
        else if (strcmp(a, "--ctx") == 0)
            o->mp->n_ctx = (uint32_t)atol(v);
        else if (strcmp(a, "--cache") == 0)
            o->mp->cache_bytes = (uint64_t)(atof(v) * (1 << 30));
        else if (strcmp(a, "--reserve") == 0)
            o->mp->reserve_bytes = (uint64_t)(atof(v) * (1 << 30));
        else if (strcmp(a, "--bits") == 0)
            o->mp->expert_bits = atoi(v);
        else if (strcmp(a, "--attention") == 0)
            o->mp->attn_scores = strcmp(v, "exact") == 0  ? JANAS_LLM_ATTN_EXACT
                                 : strcmp(v, "fast") == 0 ? JANAS_LLM_ATTN_FAST
                                                          : JANAS_LLM_ATTN_AUTO;
        else if (strcmp(a, "--system") == 0)
            *o->system_arg = v;
        else if (strcmp(a, "--mcp-config") == 0)
            *o->mcp_config = v;
        else if (strcmp(a, "--progress") == 0)
            setenv("JANAS_PROGRESS", v, 1);
        else if (strcmp(a, "--temp") == 0)
            o->cp->temperature = (float)atof(v);
        else if (strcmp(a, "--top-k") == 0)
            o->cp->top_k = atoi(v);
        else if (strcmp(a, "--top-p") == 0)
            o->cp->top_p = (float)atof(v);
        else if (strcmp(a, "--min-p") == 0)
            o->cp->min_p = (float)atof(v);
        else if (strcmp(a, "--seed") == 0)
            o->cp->seed = strtoull(v, NULL, 10);
        else if (strcmp(a, "--max") == 0)
            o->cp->max_reply = atoi(v);
        else {
            return -1;
        }
    }
    return 0;
}

/*
 * The settings kept between sessions. The file holds the words of the
 * command line, one option to a line, the rest of the line being its value:
 * the same reader takes both, so a setting cannot mean one thing typed and
 * another saved, and the file is one anybody can open and edit.
 *
 * It is read before the command line, which therefore wins, and nothing is
 * written to it unless /save-config is asked for: what a command changes is
 * for this conversation only, which is how a knob turned to try something
 * does not become the way things are.
 */
/* A question answered with one key. Piped in, nobody is there to answer:
   the caller asked for it in writing and that is the answer. */
static int ask_yes(const char *question)
{
    if (!term.tty)
        return 1;
    term_printf(&term, "%s [y/N] ", question);
    term_raw(&term, 1);
    char k = 0;
    if (read(STDIN_FILENO, &k, 1) < 0)
        k = 0;
    term_raw(&term, 0);
    int yes = k == 'y' || k == 'Y';
    term_printf(&term, "%c\n", yes ? 'y' : 'n');
    return yes;
}

static int conf_path(char *buf, size_t n, int create)
{
    const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    char dir[768];
    if (xdg && *xdg)
        snprintf(dir, sizeof(dir), "%s/janas", xdg);
    else if (home)
        snprintf(dir, sizeof(dir), "%s/.config/janas", home);
    else
        return -1;
    if (create) {
        char up[832];
        snprintf(up, sizeof(up), "%s", dir);
        char *slash = strrchr(up, '/');
        if (slash) {
            *slash = 0;
            mkdir(up, 0700);
        }
        mkdir(dir, 0700);
    }
    snprintf(buf, n, "%s/chat.conf", dir);
    return 0;
}

/* The file as words, kept alive in *store (freed by the caller, or never). */
static char **conf_read(const char *path, int *n, char **store)
{
    *n = 0;
    *store = NULL;
    FILE *f = fopen(path, "r");
    if (!f)
        return NULL;
    static char line[1024];
    char **w = NULL;
    size_t cap = 0, used = 0, hold = 0;
    char *text = NULL;
    while (fgets(line, sizeof(line), f)) {
        char *a = line;
        while (*a == ' ' || *a == '\t')
            a++;
        a[strcspn(a, "\n")] = 0;
        if (!*a || *a == '#')
            continue;
        char *v = a;
        while (*v && *v != ' ' && *v != '\t')
            v++;
        if (*v) {
            *v++ = 0;
            while (*v == ' ' || *v == '\t')
                v++;
        }
        for (int k = 0; k < (*v ? 2 : 1); k++) {
            const char *piece = k ? v : a;
            size_t len = strlen(piece) + 1;
            char *grown = realloc(text, hold + len);
            if (!grown)
                break;
            text = grown;
            memcpy(text + hold, piece, len);
            hold += len;
            if (used == cap) {
                size_t c2 = cap ? cap * 2 : 16;
                char **g2 = realloc(w, c2 * sizeof(*w));
                if (!g2)
                    break;
                w = g2;
                cap = c2;
            }
            w[used++] = (char *)(uintptr_t)(hold - len); /* an offset for now */
        }
    }
    fclose(f);
    /* the offsets become pointers once the text has stopped moving */
    for (size_t k = 0; k < used; k++)
        w[k] = text + (uintptr_t)w[k];
    *n = (int)used;
    *store = text;
    return w;
}

static void conf_write(const char *path, const struct chat_opts *o)
{
    FILE *f = fopen(path, "w");
    if (!f) {
        term_printf(&term, "cannot write %s\n", path);
        return;
    }
    const struct janas_llm_params *m = o->mp;
    const struct janas_llm_chat_params *c = o->cp;
    fprintf(f, "# janas-chat: written by /save-config. The words are those of\n"
               "# the command line, one to a line; the command line wins over\n"
               "# this file, and /load-config reads it again.\n");
    if (m->mtp_path)
        fprintf(f, "--mtp %s\n", m->mtp_path);
    fprintf(f, "--ctx %u\n", m->n_ctx);
    if (m->cache_bytes)
        fprintf(f, "--cache %.2f\n", (double)m->cache_bytes / (1 << 30));
    if (m->reserve_bytes)
        fprintf(f, "--reserve %.2f\n", (double)m->reserve_bytes / (1 << 30));
    if (m->expert_bits)
        fprintf(f, "--bits %d\n", m->expert_bits);
    if (m->attn_scores == JANAS_LLM_ATTN_EXACT)
        fprintf(f, "--attention exact\n");
    else if (m->attn_scores == JANAS_LLM_ATTN_FAST)
        fprintf(f, "--attention fast\n");
    fprintf(f, "--mode %s\n",
            m->mode == JANAS_LLM_MODE_ECO   ? "eco"
            : m->mode == JANAS_LLM_MODE_MAX ? "max"
                                            : "auto");
    fprintf(f, "--temp %g\n--top-k %d\n--top-p %g\n--min-p %g\n",
            (double)c->temperature, c->top_k, (double)c->top_p,
            (double)c->min_p);
    if (c->max_reply)
        fprintf(f, "--max %d\n", c->max_reply);
    fprintf(f, "--think %s\n", c->thinking ? "on" : "off");
    if (!c->speculate)
        fprintf(f, "--no-spec\n");
    if (*o->stats)
        fprintf(f, "--stats\n");
    if (!*o->markdown)
        fprintf(f, "--no-markdown\n");
    if (!*o->gpu)
        fprintf(f, "--no-gpu\n");
    if (*o->mcp_config)
        fprintf(f, "--mcp-config %s\n", *o->mcp_config);
    if (!*o->mcp)
        fprintf(f, "--no-mcp\n");
    if (*o->mcp_auto)
        fprintf(f, "--mcp-auto\n");
    if (*o->mcp_instructions)
        fprintf(f, "--mcp-instructions\n");
    fclose(f);
    term_printf(&term, "written to %s (the system message is not kept)\n",
                path);
}

/* The MCP servers and their tools, for /mcp. */
static void mcp_list(void)
{
    if (!mcpc_count()) {
        term_printf(&term, "no MCP server is running: they are configured in "
                           "~/.config/janas/mcp.json, or the file --mcp-config "
                           "names\n");
        return;
    }
    for (size_t i = 0; i < mcpc_count(); i++) {
        char info[2048], names[4096];
        int32_t len;
        janas_mcp_info(mcpc_server(i), info, sizeof(info), &len);
        janas_mcp_tool_names(mcpc_server(i), names, sizeof(names), &len);
        term_printf(&term, "%s%s%s\n%s%s", T_BOLD(&term), mcpc_name(i),
                    T_RESET(&term), info, T_DIM(&term));
        for (char *t = strtok(names, "\n"); t; t = strtok(NULL, "\n"))
            term_printf(&term, "  %s__%s\n", mcpc_name(i), t);
        term_printf(&term, "%s\n", T_RESET(&term));
    }
}

/* A question answered with one key, lowercase (0 when none came). */
static int ask_key(const char *question)
{
    term_printf(&term, "%s ", question);
    term_raw(&term, 1);
    char k = 0;
    if (read(STDIN_FILENO, &k, 1) < 0)
        k = 0;
    term_raw(&term, 0);
    if (k >= 'A' && k <= 'Z')
        k = (char)(k - 'A' + 'a');
    term_printf(&term, "%c\n", k >= ' ' && k < 0x7f ? k : ' ');
    return k;
}

/* Text of the library, of any length, into a buffer of the caller's. */
static char *call_part(const janas_llm_chat *c, int32_t i, int args)
{
    int32_t nl = 0, al = 0;
    janas_llm_chat_call(c, i, NULL, 0, &nl, NULL, 0, &al);
    char *name = malloc((size_t)nl + 1), *arg = malloc((size_t)al + 1);
    if (!name || !arg ||
        janas_llm_chat_call(c, i, name, nl + 1, &nl, arg, al + 1, &al) !=
            JANAS_LLM_OK) {
        free(name);
        free(arg);
        return NULL;
    }
    free(args ? name : arg);
    return args ? arg : name;
}

/*
 * The tool calls of the reply that just ended: each shown, confirmed (or
 * not, in auto mode or for a tool always allowed), run, and what the tools
 * answered sent back to the model. 1 when a new reply follows.
 */
static int run_calls(janas_llm_chat *c, int autorun)
{
    int32_t n = janas_llm_chat_calls(c);
    if (n <= 0 || !mcpc_count())
        return 0;
    char **texts = calloc((size_t)n, sizeof(*texts));
    if (!texts)
        return 0;
    int ok = 1;
    for (int32_t i = 0; i < n && ok; i++) {
        char *name = call_part(c, i, 0), *args = call_part(c, i, 1);
        if (!name || !args) {
            free(name);
            free(args);
            ok = 0;
            break;
        }
        term_gap(&term);
        term_printf(&term, "%s→ %s%s %s%.*s%s%s\n", T_CMD(&term), name,
                    T_DIM(&term), "", 400, args,
                    strlen(args) > 400 ? " ..." : "", T_RESET(&term));
        int go = autorun || mcpc_always(name);
        if (!go && interrupted) {
            go = 0;
        } else if (!go && !term.tty) {
            term_printf(&term, "not run: on a pipe nobody can say yes "
                               "(--mcp-auto runs them)\n");
        } else if (!go) {
            int k = ask_key("run it? [y]es, [n]o, [a]lways this tool");
            go = k == 'y' || k == 'a';
            if (k == 'a')
                mcpc_set_always(name);
        }
        if (go) {
            footer(idle_llm, c, "running a tool");
            int rc = mcpc_call(name, args, &texts[i]);
            const char *t = texts[i] ? texts[i] : "";
            size_t line = strcspn(t, "\n");
            term_printf(
                &term, "%s%s%.*s%s%s\n", rc ? T_WARN(&term) : T_DIM(&term),
                rc > 0 ? "tool error: " : "", (int)(line < 200 ? line : 200), t,
                line < strlen(t) || line > 200 ? " ..." : "", T_RESET(&term));
        } else {
            texts[i] = strdup("The user did not allow this call: it was "
                              "not run.");
        }
        free(name);
        free(args);
        if (!texts[i])
            ok = 0;
    }
    int sent = 0;
    if (ok && !interrupted) {
        if (janas_llm_chat_send_results(c, n, (const char *const *)texts,
                                        NULL) == JANAS_LLM_OK)
            sent = 1;
        else
            term_printf(&term, "%s\n", janas_llm_last_error());
    }
    for (int32_t i = 0; i < n; i++)
        free(texts[i]);
    free(texts);
    interrupted = 0;
    return sent;
}

/* The reply, written as it comes; -1 when it failed or was stopped. */
static int show_reply(const janas_llm *llm, janas_llm_chat *c, int markdown,
                      int stats)
{
    interrupted = 0;
    term_gap(&term);
    footer(llm, c, janas_llm_chat_thinking(c) ? "reasoning" : "generating");
    char piece[256];
    int32_t len = 0, rc; /* written by chat_next when it says OK */
    int grey = 0;
    char tail = '\n'; /* the reply's last byte: a model often ends on a
                          newline of its own, and one is enough */
    unsigned pieces = 0;
    struct janas_md md;
    md_init(&md, &term, markdown);
    term_puts(&term, T_MODEL(&term));
    while ((rc = janas_llm_chat_next(c, piece, sizeof(piece), &len)) ==
           JANAS_LLM_OK) {
        /* the reasoning in grey: the colour changes only where the
           reasoning starts and ends */
        int th = janas_llm_chat_thinking(c);
        if (th != grey) {
            md.base = th ? T_THINK(&term) : "";
            term_puts(&term, th ? T_THINK(&term) : T_RESET(&term));
            grey = th;
        }
        if (resized) { /* the window changed: the footer follows it */
            resized = 0;
            term_size(&term);
            term_region(&term);
            if (conv.back)
                scroll_draw(&term, &conv); /* re-wrapped to the new width */
        }
        md_write(&md, piece, (size_t)len);
        if (len > 0)
            tail = piece[len - 1];
        fflush(stdout);
        struct janas_llm_chat_stats rs = {.size = sizeof(rs)};
        if (len == 0 && janas_llm_chat_stats(c, &rs) == JANAS_LLM_OK &&
            rs.stage == JANAS_LLM_STAGE_INPUT) {
            /* a long prompt, a block at a time: how far, how fast */
            char state[96];
            double rate =
                rs.input_seconds > 0 ? rs.input_done / rs.input_seconds : 0;
            double left = rs.size >= sizeof(rs) ? rs.input_eta_seconds : -1;
            char eta[32] = "";
            if (left >= 3600)
                snprintf(eta, sizeof(eta), ", ~%dh%02dm left",
                         (int)(left / 3600), (int)(left / 60) % 60);
            else if (left >= 60)
                snprintf(eta, sizeof(eta), ", ~%dm left", (int)(left / 60));
            else if (left >= 0)
                snprintf(eta, sizeof(eta), ", ~%ds left", (int)left);
            snprintf(state, sizeof(state), "reading %u%% %.0f tok/s%s",
                     rs.input_tokens ? 100 * rs.input_done / rs.input_tokens
                                     : 100,
                     rate, eta);
            footer(llm, c, state);
        } else if (++pieces % 16 == 0) { /* the footer follows the reply */
            struct janas_llm_chat_stats st = {.size = sizeof(st)};
            const char *what =
                janas_llm_chat_thinking(c) ? "reasoning" : "generating";
            char state[64];
            snprintf(state, sizeof(state), "%s", what);
            if (janas_llm_chat_stats(c, &st) == JANAS_LLM_OK &&
                st.output_tokens)
                snprintf(state, sizeof(state), "%s %u", what, st.output_tokens);
            footer(llm, c, state);
        }
        if (interrupted)
            break;
    }
    md_end(&md);
    term_puts(&term, T_RESET(&term));
    grey = 0;
    if (tail != '\n')
        term_write(&term, "\n", 1);
    if (rc < 0) {
        term_gap(&term);
        term_printf(&term, "[%s]\n", janas_llm_last_error());
    } else if (interrupted) {
        term_gap(&term);
        term_printf(&term, "[stopped]\n");
    }
    int stopped = interrupted || rc < 0;
    interrupted = 0;
    {
        struct janas_llm_chat_stats st = {.size = sizeof(st)};
        if (janas_llm_chat_stats(c, &st) == JANAS_LLM_OK &&
            st.output_seconds > 0) {
            sess_tokens += st.output_tokens;
            sess_seconds += st.output_seconds;
        }
    }
    if (stats) {
        term_gap(&term);
        report(c);
    }
    return stopped ? -1 : 0;
}

int main(int argc, char **argv)
{
    /* wcwidth() answers for the locale's character set, and in the C locale
       it answers -1 to everything past ASCII: ask for the user's own */
    setlocale(LC_CTYPE, "");
    if (argc < 2 || argv[1][0] == '-') {
        usage();
        return 2;
    }
    if (janas_llm_abi_version() != JANAS_LLM_ABI_VERSION) {
        fprintf(stderr, "janas-chat: library ABI %d, expected %d\n",
                janas_llm_abi_version(), JANAS_LLM_ABI_VERSION);
        return 1;
    }
    struct janas_llm_params mp;
    struct janas_llm_chat_params cp;
    janas_llm_params_default(&mp);
    janas_llm_chat_params_default(&cp);
    /* the chat asks for no reasoning unless it is asked for. The library's
       own default is the model's, which for Qwen3 is to reason: on a small
       model that is two hundred tokens of rough work before the first word
       of an answer to "what is seventeen times three". --think on, /think
       on, or --think on in the configuration file bring it back. */
    cp.thinking = 0;
    asked = &cp;
    const char *system_arg = NULL;
    int stats = 0;
    int markdown = 1; /* the marks read, not printed; never on a pipe */
    int gpu = 1;      /* the GPU may be given work, if there is one */
    const char *mcp_config = NULL; /* NULL: the default file */
    int mcp = 1, mcp_auto = 0, mcp_instructions = 0;
    struct chat_opts o = {
        &mp,  &cp,         &system_arg, &stats,    &markdown,
        &gpu, &mcp_config, &mcp,        &mcp_auto, &mcp_instructions};
    /* the file first, the command line after it: what is typed wins */
    char cpath[1024];
    char *conf_text = NULL;
    if (conf_path(cpath, sizeof(cpath), 0) == 0) {
        int cn = 0;
        char **cw = conf_read(cpath, &cn, &conf_text);
        if (cw) {
            if (take_options(cn, cw, &o) != 0)
                fprintf(stderr, "janas-chat: %s has a word it does not know\n",
                        cpath);
            free(cw);
        }
    }
    if (take_options(argc - 2, argv + 2, &o) != 0) {
        usage();
        return 2;
    }

    term_init(&term);
    if (!term.tty)
        markdown = 0; /* a pipe gets the marks: a script may want them */
    scroll_init(&conv);
    scroll_attach(&term, &conv); /* from here the conversation is kept */
    int tty = term.tty;
    fprintf(stderr, "loading %s ...\n", argv[1]);
    janas_llm *llm = NULL;
    janas_llm_chat *c = NULL; /* only written when the open succeeded, which
                                 a compiler looking across the whole program
                                 cannot always see */
    if (janas_llm_open(argv[1], &mp, &llm) != JANAS_LLM_OK ||
        janas_llm_chat_create(llm, &cp, &c) != JANAS_LLM_OK) {
        fprintf(stderr, "janas-chat: %s\n", janas_llm_last_error());
        return 1;
    }
    char *system = system_arg ? (*system_arg ? strdup(system_arg) : NULL)
                              : default_system(llm);
    if (system && janas_llm_chat_system(c, system, -1) != JANAS_LLM_OK) {
        fprintf(stderr, "janas-chat: %s\n", janas_llm_last_error());
        return 1;
    }
    if (!gpu && janas_llm_set_gpu(llm, 0) != JANAS_LLM_OK)
        fprintf(stderr, "janas-chat: %s\n", janas_llm_last_error());
    /* the tools of the MCP servers configured, given to the model */
    if (mcp && mcpc_start(mcp_config) > 0 && mcpc_tools() &&
        janas_llm_chat_tools(c, mcpc_tools(), -1) != JANAS_LLM_OK) {
        fprintf(stderr, "janas-chat: MCP tools: %s\n", janas_llm_last_error());
        mcpc_stop();
    }
    /* and, asked for, what the servers say of their tools */
    if (mcp_instructions && mcpc_count() && (mcp_extra = mcpc_instructions()) &&
        set_system(c, system) != JANAS_LLM_OK)
        fprintf(stderr, "janas-chat: %s\n", janas_llm_last_error());
    char desc[512];
    int32_t dl;
    janas_llm_describe(llm, desc, sizeof(desc), &dl);
    char name[96] = "";
    janas_llm_name(llm, name, sizeof(name), &dl);
    if (term.tty) {
        term_banner(&term, "Janas-Chat", name[0] ? name : argv[1],
                    "/help for the commands, /quit to leave");
        term_printf(&term, "%s%s%s\n\n", T_DIM(&term), desc, T_RESET(&term));
        char hp[512];
        const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
        if (xdg && *xdg)
            snprintf(hp, sizeof(hp), "%s/janas/chat_history", xdg);
        else if (home)
            snprintf(hp, sizeof(hp), "%s/.cache/janas/chat_history", home);
        else
            hp[0] = 0;
        hist_load(&history, hp);
        footer(llm, c, NULL);
        idle_llm = llm;
        idle_chat = c;
        line_idle = footer_idle;
    } else
        fprintf(stderr, "%s\n/help for the commands\n", desc);

    struct sigaction sa = {.sa_handler = on_sigint};
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL); /* no SA_RESTART: getline returns */
    struct sigaction sw = {.sa_handler = on_sigwinch};
    sigemptyset(&sw.sa_mask); /* no SA_RESTART: the editor redraws itself */
    sigaction(SIGWINCH, &sw, NULL);

    char *msg = NULL;
    size_t cap = 0;
    long n;
    while ((n = read_message(&msg, &cap, tty)) >= 0) {
        if (n == 0)
            continue;
        if (msg[0] == '/') {
            term_gap(&term); /* what a command answers is one block */
            char *arg = strchr(msg, ' ');
            if (arg)
                *arg++ = 0;
            else
                arg = msg + strlen(msg);
            if (strcmp(msg, "/quit") == 0 || strcmp(msg, "/exit") == 0)
                break;
            if (strcmp(msg, "/help") == 0 || strcmp(msg, "/about") == 0) {
                char *page =
                    catch_page(strcmp(msg, "/help") == 0 ? help : about, llm);
                if (!page)
                    term_printf(&term, "not enough memory for it\n");
                else if (term.tty) {
                    term_page_begin(&term);
                    term_pager(&term, page);
                    term_page_end(&term);
                } else
                    fputs(page, stdout);
                free(page);
            } else if (strcmp(msg, "/save-config") == 0) {
                if (conf_path(cpath, sizeof(cpath), 1) != 0)
                    term_printf(&term, "nowhere to write it\n");
                /* only when there is something to lose: asking about an
                   empty place is noise, and noise is what makes people
                   stop reading the questions that matter */
                else if (access(cpath, F_OK) != 0 ||
                         ask_yes("overwrite the saved settings?"))
                    conf_write(cpath, &o);
            } else if (strcmp(msg, "/del-config") == 0) {
                if (conf_path(cpath, sizeof(cpath), 0) != 0)
                    term_printf(&term, "nowhere to look for it\n");
                else if (access(cpath, F_OK) != 0)
                    term_printf(&term, "there is no %s\n", cpath);
                else if (ask_yes("delete the saved settings?")) {
                    if (remove(cpath) == 0)
                        term_printf(&term,
                                    "%s deleted: the next start uses the "
                                    "chat's own settings\n",
                                    cpath);
                    else
                        term_printf(&term, "cannot delete %s\n", cpath);
                }
            } else if (strcmp(msg, "/load-config") == 0) {
                if (!ask_yes("read the saved settings, losing these?"))
                    continue;
                struct janas_llm_params was = mp;
                int cn = 0;
                char *text = NULL;
                char **cw = conf_path(cpath, sizeof(cpath), 0) == 0
                                ? conf_read(cpath, &cn, &text)
                                : NULL;
                if (!cw) {
                    term_printf(&term, "no %s to read\n", cpath);
                    free(text);
                } else {
                    if (take_options(cn, cw, &o) != 0)
                        term_printf(&term, "a word in it is not known\n");
                    free(cw);
                    free(conf_text); /* the older words are let go */
                    conf_text = text;
                    if (janas_llm_chat_set_params(c, &cp) != JANAS_LLM_OK)
                        term_printf(&term, "%s\n", janas_llm_last_error());
                    janas_llm_set_mode(llm, mp.mode);
                    if (janas_llm_set_gpu(llm, gpu) != JANAS_LLM_OK)
                        term_printf(&term, "%s\n", janas_llm_last_error());
                    term_printf(&term, "read %s\n", cpath);
                    if (was.n_ctx != mp.n_ctx ||
                        was.cache_bytes != mp.cache_bytes ||
                        was.expert_bits != mp.expert_bits ||
                        was.attn_scores != mp.attn_scores ||
                        was.mtp_path != mp.mtp_path)
                        term_printf(&term,
                                    "context, cache, bits, attention and the "
                                    "MTP block are read at the next start\n");
                }
            } else if (strcmp(msg, "/gpu") == 0) {
                int want = on_off(arg, gpu);
                if (janas_llm_set_gpu(llm, want) != JANAS_LLM_OK)
                    term_printf(&term, "%s\n", janas_llm_last_error());
                else {
                    gpu = want;
                    term_printf(&term, "GPU %s\n", gpu ? "on" : "off");
                }
            } else if (strcmp(msg, "/mcp") == 0) {
                if (strncmp(arg, "auto", 4) == 0) {
                    const char *v = arg[4] == ' ' ? arg + 5 : "";
                    mcp_auto = on_off(v, mcp_auto);
                    term_printf(&term, "MCP tools %s\n",
                                mcp_auto ? "run without asking"
                                         : "run after asking");
                } else {
                    mcp_list();
                }
            } else if (strcmp(msg, "/context") == 0) {
                context_page(c);
            } else if (strcmp(msg, "/markdown") == 0) {
                markdown = on_off(arg, markdown);
                term_printf(&term, "markdown %s\n",
                            markdown ? "read" : "printed as it comes");
            } else if (strcmp(msg, "/stats") == 0) {
                stats = on_off(arg, stats);
                term_printf(&term, "stats %s\n", stats ? "on" : "off");
            } else if (strcmp(msg, "/reset") == 0) {
                restart(c, system);
                if (term.tty) { /* a clean window, as at the start */
                    term_banner(&term, "Janas-Chat", name[0] ? name : argv[1],
                                "/help for the commands, /quit to leave");
                    footer(llm, c, NULL);
                } else
                    term_printf(&term, "new conversation\n");
            } else if (strcmp(msg, "/system") == 0) {
                if (!*arg) {
                    term_printf(&term, "%s\n",
                                system ? system : "(no system message)");
                    continue;
                }
                free(system);
                system = strcmp(arg, "off") == 0 ? NULL : strdup(arg);
                restart(c, system);
                term_printf(&term, "new conversation %s\n",
                            system ? "with this system message"
                                   : "with no system message");
            } else if (strcmp(msg, "/experts") == 0) {
                int32_t used = 0, most = 0;
                if (*arg &&
                    janas_llm_set_experts(llm, atoi(arg)) != JANAS_LLM_OK) {
                    term_printf(&term, "%s\n", janas_llm_last_error());
                    continue;
                }
                janas_llm_experts(llm, &used, &most);
                term_printf(&term, "%d experts per token of %d\n", used, most);
            } else if (strcmp(msg, "/temp") == 0 && *arg) {
                cp.temperature = (float)atof(arg);
                if (janas_llm_chat_set_params(c, &cp) != JANAS_LLM_OK)
                    term_printf(&term, "%s\n", janas_llm_last_error());
            } else if (strcmp(msg, "/think") == 0) {
                cp.thinking = strcmp(arg, "off") == 0  ? 0
                              : strcmp(arg, "on") == 0 ? 1
                                                       : !cp.thinking;
                if (janas_llm_chat_set_params(c, &cp) == JANAS_LLM_OK)
                    term_printf(&term,
                                "reasoning %s (for models that reason)\n",
                                cp.thinking ? "on" : "off");
            } else if (strcmp(msg, "/mode") == 0) {
                int32_t md = strcmp(arg, "eco") == 0    ? JANAS_LLM_MODE_ECO
                             : strcmp(arg, "max") == 0  ? JANAS_LLM_MODE_MAX
                             : strcmp(arg, "auto") == 0 ? JANAS_LLM_MODE_AUTO
                                                        : -1;
                if (*arg && md < 0)
                    term_printf(&term, "modes: auto, eco, max\n");
                else if (*arg && janas_llm_set_mode(llm, md) != JANAS_LLM_OK)
                    term_printf(&term, "%s\n", janas_llm_last_error());
                char tb[512];
                int32_t tl;
                if (janas_llm_tuning(llm, tb, sizeof(tb), &tl) == JANAS_LLM_OK)
                    term_printf(&term, "%s\n", tb);
            } else if (strcmp(msg, "/spec") == 0) {
                cp.speculate = on_off(arg, cp.speculate);
                if (janas_llm_chat_set_params(c, &cp) == JANAS_LLM_OK)
                    term_printf(&term, "speculative decoding %s\n",
                                cp.speculate ? "on" : "off");
                else
                    term_printf(&term, "%s\n", janas_llm_last_error());
            } else {
                term_printf(&term, "unknown command %s: /help lists them\n",
                            msg);
            }
            continue;
        }
        if (janas_llm_chat_send(c, msg, (int32_t)n) != JANAS_LLM_OK) {
            term_gap(&term);
            term_printf(&term, "%s\n", janas_llm_last_error());
            continue;
        }
        /* the reply, and while it calls tools and they answer, the next */
        for (int round = 0; round < 16; round++)
            if (show_reply(llm, c, markdown, stats) != 0 ||
                run_calls(c, mcp_auto) <= 0)
                break;
        footer(llm, c, NULL);
    }
    free(msg);
    free(system);
    free(mcp_extra);
    mcpc_stop();
    janas_llm_chat_destroy(c);
    janas_llm_close(llm);
    scroll_free(&conv);
    free(conf_text);
    if (term.tty) {
        hist_save(&history);
        hist_free(&history);
        term_raw(&term, 0);
        term_region_off(&term);
    }
    return 0;
}
