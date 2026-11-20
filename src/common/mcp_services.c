/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_services.c - Janas's services as a model is offered them
 * (mcp_services.h), in libjanas_mcp with the client they use.
 */
#include "common/mcp_services.h"

#include <stdlib.h>
#include <string.h>

const struct janas_service JANAS_SERVICES[] = {
    {"docs", "the user's own documents (notes, letters, contracts, "
             "manuals, papers): the passages that answer a question, found "
             "by their meaning and their words; a document read"},
    {"files", "the user's files and folders: what a folder holds, what a "
              "file says, files found by name or by the text they hold; a "
              "file written or edited, a folder made, each confirmed by the "
              "user"},
    {"flights", "flights and aircraft: where a flight is and what it is "
                "doing, today's flights of a route, what flies over a sea "
                "or near a place"},
    {"git", "the git repositories of this computer: what has changed, "
            "the commits, the branches, a diff; a commit, pull, push or "
            "switch of branch, each confirmed by the user"},
    {"github", "GitHub: a project's issues and pull requests (the new "
               "ones), its latest release, its recent commits, what it is"},
    {"maps", "roads and places: a route by car, bike or on foot, public "
             "transport between places, what is near (fuel, a pharmacy...), "
             "where an address is"},
    {"prices", "prices and finance, for information: currencies, "
               "crypto-assets, shares, inflation, central banks' rates, "
               "electricity, the cheapest fuel near a place"},
    {"quakes", "earthquakes: the latest near a place or where the user is, "
               "the strongest in the world, one in detail"},
    {"skills", "the user's skills: instructions for particular tasks, "
               "with their scripts, run in a cage"},
    {"system", "this computer itself, when the user asks about it: how it "
               "is (memory, processor, disks, battery, temperatures), what "
               "fills a folder, the programs running, the errors, the "
               "updates, the network"},
    {"weather", "the weather now and the forecast, the sea and its waves, "
                "weather warnings"},
    {"wiki", "Wikipedia, when the user asks for it: a page shown to the "
             "user; what there is to see around a place"},
};
const size_t JANAS_N_SERVICES = sizeof JANAS_SERVICES / sizeof *JANAS_SERVICES;

const char *janas_service_about(const char *name)
{
    for (size_t k = 0; k < JANAS_N_SERVICES; k++)
        if (strcmp(JANAS_SERVICES[k].name, name) == 0)
            return JANAS_SERVICES[k].about;
    return "";
}

void janas_services_catalog(struct janas_buf *o, const char *const *names,
                            const char *const *abouts, size_t n)
{
    janas_buf_puts(o,
                   "Janas's services bring data of the world. Their tools are "
                   "not listed here: to use a service, call " JANAS_OPEN_SERVICE
                   " with its name, by yourself whenever a question is on its "
                   "subject (the user need not name it), then call the tools "
                   "it gives you; an open service stays open for the "
                   "conversation. Most questions need none - writing, code, "
                   "explanations, what you know: answer those yourself, "
                   "without opening a service. Never name the services or the "
                   "tools in your reply. The services:\n");
    for (size_t i = 0; i < n; i++)
        janas_buf_printf(o, "- %s: %s\n", names[i],
                         abouts && abouts[i] ? abouts[i]
                                             : janas_service_about(names[i]));
}

void janas_services_open_tool(struct janas_buf *o, const char *const *names,
                              size_t n)
{
    janas_buf_puts(o,
                   "{\"type\": \"function\", \"function\": {\"name\": "
                   "\"" JANAS_OPEN_SERVICE "\", \"description\": \"Gives you "
                   "the instructions and the tools of one of "
                   "Janas's services, listed in the system message. "
                   "Once a service is open, call its tools.\", "
                   "\"parameters\": {\"type\": \"object\", "
                   "\"properties\": {\"service\": {\"type\": "
                   "\"string\", \"enum\": [");
    for (size_t i = 0; i < n; i++)
        janas_buf_printf(o, "%s\"%s\"", i ? ", " : "", names[i]);
    janas_buf_puts(o, "]}}, \"required\": [\"service\"]}}}");
}

void janas_service_opened(struct janas_buf *o, janas_mcp *m, const char *name,
                          const char *prefix)
{
    int32_t len = 0;
    janas_mcp_instructions(m, NULL, 0, &len);
    char *ins = len > 0 ? malloc((size_t)len + 1) : NULL;
    if (ins && janas_mcp_instructions(m, ins, len + 1, &len) == JANAS_MCP_OK)
        janas_buf_printf(o, "%s\n\n", ins);
    free(ins);
    len = 0;
    janas_mcp_tools(m, prefix, -1, NULL, 0, &len);
    char *t = malloc((size_t)len + 1);
    if (t && janas_mcp_tools(m, prefix, -1, t, len + 1, &len) == JANAS_MCP_OK) {
        /* as the system message lists tools, one a line in <tools>: given
           them as one array in the text, Qwen3.8-Flash-Next opened the
           service again and again instead of calling them (22 Oct 2026) */
        janas_buf_printf(o,
                         "The service %s is open. Its tools, to call as "
                         "any other:\n\n<tools>",
                         name);
        struct janas_json_doc *td = janas_json_parse(t, (size_t)len, NULL, 0);
        const struct janas_json *a = td ? janas_json_root(td) : NULL;
        if (a && a->type == JANAS_JSON_ARRAY)
            for (const struct janas_json *it = a->child; it; it = it->next) {
                janas_buf_puts(o, "\n");
                janas_json_write(o, it);
            }
        else
            janas_buf_printf(o, "\n%s", t);
        janas_json_free(td);
        janas_buf_puts(o, "\n</tools>\n\nNow call the one that answers "
                          "the user's question: do not answer before it "
                          "has, and never make up what it would say.");
    }
    free(t);
}
