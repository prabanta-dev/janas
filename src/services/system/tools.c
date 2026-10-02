/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - janas-system's tools (see system.h). They only read, and say
 * so (readOnlyHint); none of them reaches the network (openWorldHint
 * false): how the computer is, its disks and what fills a directory, its
 * processes, its errors, its updates, its network.
 */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "services/common/template.h"
#include "system.h"

#define READS                                                                  \
    "\"annotations\": {\"readOnlyHint\": true, \"openWorldHint\": false}"
#define NO_ARGS                                                                \
    "\"inputSchema\": {\"type\": \"object\", \"properties\": {}, "             \
    "\"additionalProperties\": false}"

void sy_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b,
        "\"tools\": [{\"name\": \"system_status\", \"title\": \"This "
        "computer\", \"description\": \"How this computer is: processor, "
        "memory, disks, battery, temperatures, how long it has been on. For "
        "'how is the computer', 'why is it slow'.\", " NO_ARGS ", " READS "}, "
        "{\"name\": \"system_disks\", \"title\": \"Disks\", "
        "\"description\": \"The disks and their free space; with a path, "
        "what fills that directory, biggest first.\", \"inputSchema\": "
        "{\"type\": \"object\", \"properties\": {\"path\": {\"type\": "
        "\"string\", \"description\": \"a directory (~ for the home); leave "
        "it out for the disks\"}}, \"additionalProperties\": false}, " READS
        "}, "
        "{\"name\": \"system_processes\", \"title\": \"Processes\", "
        "\"description\": \"The programs running that use the most "
        "processor or memory, or those of a name (is a program running?).\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": {\"by\": "
        "{\"type\": \"string\", \"enum\": [\"cpu\", \"memory\"]}, \"name\": "
        "{\"type\": \"string\", \"description\": \"part of a program's "
        "name\"}}, \"additionalProperties\": false}, " READS "}, "
        "{\"name\": \"system_errors\", \"title\": \"Errors\", "
        "\"description\": \"The errors in the system's logs and the "
        "services that failed.\", \"inputSchema\": {\"type\": \"object\", "
        "\"properties\": {\"hours\": {\"type\": \"number\", \"description\": "
        "\"the last hours; leave it out for since the computer "
        "started\"}}, \"additionalProperties\": false}, " READS "}, "
        "{\"name\": \"system_updates\", \"title\": \"Updates\", "
        "\"description\": \"The updates available for the installed "
        "packages, by the list the package manager last fetched.\", " NO_ARGS
        ", " READS "}, "
        "{\"name\": \"system_network\", \"title\": \"Network\", "
        "\"description\": \"The network interfaces, their addresses, the "
        "Wi-Fi signal, the gateway and the name servers.\", " NO_ARGS ", " READS
        "}]");
}

static const char *arg_str(const struct janas_json *args, const char *name)
{
    const char *s = janas_json_str(janas_json_get(args, name));
    return s && *s ? s : NULL;
}

static int fail(struct janas_buf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static int fail(struct janas_buf *b, const char *fmt, ...)
{
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    janas_mcps_text_result(b, line, strlen(line), 1);
    return 0;
}

static int answer(struct janas_buf *b, const char *name, struct janas_buf *d,
                  const char *layout, const char *brief)
{
    char err[300];
    int oom = d->oom;
    int r = oom ? -1
                : janas_tpl_result(b, name, d->p, d->n, layout, brief, err,
                                   sizeof err);
    janas_buf_free(d);
    if (r != 0)
        return fail(b, "The answer could not be written: %s.",
                    oom ? "out of memory" : err);
    return 0;
}

/* the structures are large: one at a time, kept here */
static union {
    struct sy_status status;
    struct sy_disks disks;
    struct sy_usage usage;
    struct sy_procs procs;
    struct sy_errors errors;
    struct sy_updates updates;
    struct sy_net net;
} R;

static int t_status(const struct janas_json *args, struct janas_buf *b)
{
    (void)args;
    char err[300] = "";
    if (sy_get_status(&R.status, err, sizeof err) != 0)
        return fail(b, "This computer could not be read: %s.", err);
    struct janas_buf d = {0};
    sy_status_data(&d, &R.status);
    return answer(b, "system_status", &d, SY_STATUS_LAYOUT, SY_STATUS_BRIEF);
}

static int t_disks(const struct janas_json *args, struct janas_buf *b)
{
    char err[1200] = "";
    const char *path = arg_str(args, "path");
    struct janas_buf d = {0};
    if (path) {
        if (sy_get_usage(path, &R.usage, err, sizeof err) != 0)
            return fail(b, "%s.", err);
        sy_usage_data(&d, &R.usage);
        return answer(b, "system_usage", &d, SY_USAGE_LAYOUT, SY_USAGE_BRIEF);
    }
    if (sy_get_disks(&R.disks, err, sizeof err) != 0)
        return fail(b, "The disks could not be read: %s.", err);
    sy_disks_data(&d, &R.disks);
    return answer(b, "system_disks", &d, SY_DISKS_LAYOUT, SY_DISKS_BRIEF);
}

static int t_procs(const struct janas_json *args, struct janas_buf *b)
{
    const char *by = arg_str(args, "by");
    if (!by || strcmp(by, "memory"))
        by = "cpu";
    char err[300] = "";
    if (sy_get_procs(by, arg_str(args, "name"), &R.procs, err, sizeof err) != 0)
        return fail(b, "The processes could not be read: %s.", err);
    struct janas_buf d = {0};
    sy_procs_data(&d, &R.procs);
    return answer(b, "system_processes", &d, SY_PROCS_LAYOUT, SY_PROCS_BRIEF);
}

static int t_errors(const struct janas_json *args, struct janas_buf *b)
{
    const struct janas_json *h = janas_json_get(args, "hours");
    double hours = h && h->type == JANAS_JSON_NUMBER ? janas_json_num(h, 0) : 0;
    if (hours < 0)
        hours = 0;
    if (hours > 24 * 31)
        hours = 24 * 31;
    char err[400] = "";
    if (sy_get_errors((int)(hours + 0.5), &R.errors, err, sizeof err) != 0)
        return fail(b, "The errors could not be read: %s.", err);
    struct janas_buf d = {0};
    sy_errors_data(&d, &R.errors);
    return answer(b, "system_errors", &d, SY_ERRORS_LAYOUT, SY_ERRORS_BRIEF);
}

static int t_updates(const struct janas_json *args, struct janas_buf *b)
{
    (void)args;
    char err[400] = "";
    if (sy_get_updates(&R.updates, err, sizeof err) != 0)
        return fail(b, "The updates could not be read: %s. Tell the user so.",
                    err);
    struct janas_buf d = {0};
    sy_updates_data(&d, &R.updates);
    return answer(b, "system_updates", &d, SY_UPDATES_LAYOUT, SY_UPDATES_BRIEF);
}

static int t_net(const struct janas_json *args, struct janas_buf *b)
{
    (void)args;
    char err[300] = "";
    if (sy_get_net(&R.net, err, sizeof err) != 0)
        return fail(b, "The network could not be read: %s.", err);
    struct janas_buf d = {0};
    sy_net_data(&d, &R.net);
    return answer(b, "system_network", &d, SY_NET_LAYOUT, SY_NET_BRIEF);
}

int sy_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    (void)ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    static const struct {
        const char *name;
        int (*fn)(const struct janas_json *, struct janas_buf *);
    } T[] = {{"system_status", t_status},   {"system_disks", t_disks},
             {"system_processes", t_procs}, {"system_errors", t_errors},
             {"system_updates", t_updates}, {"system_network", t_net}};
    for (size_t i = 0; i < sizeof T / sizeof *T; i++)
        if (janas_json_is(name, T[i].name))
            return T[i].fn(args, b);
    return -1;
}
