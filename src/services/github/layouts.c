/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layouts.c - the layouts of janas-github's answers (see github.h and
 * services/common/template.h): the text the user reads, filled with the data in
 * English here and in the user's language by a client that translates the
 * layout once (janas-chat), and the brief, all the model reads of it.
 * Titles, messages and notes stay as their authors wrote them.
 */
#include "github.h"

#define MON_WORDS                                                              \
    "mon.1 = January\n"                                                        \
    "mon.2 = February\n"                                                       \
    "mon.3 = March\n"                                                          \
    "mon.4 = April\n"                                                          \
    "mon.5 = May\n"                                                            \
    "mon.6 = June\n"                                                           \
    "mon.7 = July\n"                                                           \
    "mon.8 = August\n"                                                         \
    "mon.9 = September\n"                                                      \
    "mon.10 = October\n"                                                       \
    "mon.11 = November\n"                                                      \
    "mon.12 = December\n"

/* How the repository was chosen, when the user named it alone. */
#define HOW_WORDS                                                              \
    "how.yours = one of yours\n"                                               \
    "how.found by its name = the one of that name with the most stars\n"

/* A day, with its year when not this one, and the time. */
#define WHEN(t)                                                                \
    "{{" t ".day}} {{" t ".mon|mon}}{{#" t ".year}} {{" t ".year}}{{/" t       \
    ".year}}, {{" t ".at}}"

/* The same for the model: "2/10, 3 days ago" */
#define AGO(t)                                                                 \
    "{{" t ".day}}/{{" t ".mon}}{{#" t ".year}}/{{" t ".year}}{{/" t           \
    ".year}}, {{" t ".ago_d}} days ago"

#define W_SINCE WHEN("since")
#define W_CREATED WHEN("created")
#define W_WHEN WHEN("when")
#define W_PUSHED WHEN("pushed")
#define A_SINCE AGO("since")
#define A_CREATED AGO("created")
#define A_WHEN AGO("when")
#define A_PUSHED AGO("pushed")

#define REPO "{{repo}}{{#how}} ({{how|how}}){{/how}}"
#define REPO_BRIEF                                                             \
    "{{repo}}{{#how}} ({{how}}: say which repository was taken){{/how}}"

const char GH_ISSUES_LAYOUT[] =
    REPO ": {{#total}}{{total}}{{/total}}{{^total}}no{{/total}} "
         "{{kind|kind}}{{#new}} since " W_SINCE
         "{{/new}}{{#first}} (the first time you ask: the last 7 "
         "days){{/first}}{{#more}}; the latest "
         "{{shown}}{{/more}}{{#total}}:{{/total}}{{^total}}.{{/total}}\n"
         "{{#items}}\n"
         "  #{{n}} {{title}}\n"
         "      {{status|status}}, by {{user}}, " W_CREATED
         "{{#comments}}, comments: {{comments}}{{/comments}}{{#labels}}, "
         "labels: {{labels}}{{/labels}}\n"
         "      {{url}}\n"
         "{{/items}}\n"
         "---\n" MON_WORDS HOW_WORDS "kind.open_issues = open issues\n"
         "kind.closed_issues = closed issues\n"
         "kind.all_issues = issues\n"
         "kind.new_issues = new issues\n"
         "kind.open_pulls = open pull requests\n"
         "kind.closed_pulls = closed pull requests\n"
         "kind.all_pulls = pull requests\n"
         "kind.new_pulls = new pull requests\n"
         "status.open = open\n"
         "status.closed = closed\n"
         "status.merged = merged\n"
         "status.draft = draft\n";

const char GH_ISSUES_BRIEF[] = REPO_BRIEF
    ": {{total}} {{kind}} (of those listed: {{open_n}} open, "
    "{{closed_n}} closed){{#new}} since " A_SINCE
    "{{/new}}{{#first}} (asked the first time: the last 7 "
    "days){{/first}}. {{#items}}#{{n}} {{title}} ({{status}}, " A_CREATED
    "); {{/items}}The list is shown to the user as it is: do not "
    "repeat it; answer in a line or two, in the user's language.";

const char GH_RELEASES_LAYOUT[] = REPO
    ": {{#tags}}no releases; its latest "
    "tags:{{/tags}}{{^tags}}{{#n}}{{#one}}the latest release:{{/one}}{{^one}}the latest releases:{{/one}}{{/n}}{{^n}}no "
    "releases and no tags.{{/n}}{{/tags}}\n"
    "{{#items}}\n"
    "\n{{tag}}{{#name}} - {{name}}{{/name}}{{#pre}} "
    "(pre-release){{/pre}}{{#when.day}}, " W_WHEN
    "{{/when.day}}{{#author}}, by {{author}}{{/author}}{{#assets}}, "
    "files: {{assets}}{{/assets}}\n"
    "{{#notes}}{{notes}}\n{{/notes}}"
    "{{#url}}{{url}}\n{{/url}}"
    "{{/items}}\n"
    "---\n" MON_WORDS HOW_WORDS;

const char GH_RELEASES_BRIEF[] = REPO_BRIEF
    ": {{#tags}}no releases; tags: {{/tags}}{{^n}}no releases and no "
    "tags. {{/n}}{{#items}}{{tag}}{{#pre}} "
    "(pre-release){{/pre}}{{#when.day}}, " A_WHEN
    "{{/when.day}}; {{/items}}The notes are shown to the user: do not "
    "repeat them; answer in a line or two, in the user's language.";

const char GH_COMMITS_LAYOUT[] =
    REPO ", branch {{branch}}: {{#n}}{{#more}}the latest {{n}} "
         "commits{{/more}}{{^more}}{{n}} commits{{/more}} in the last "
         "{{days}} days:{{/n}}{{^n}}no commits in the last {{days}} "
         "days.{{/n}}\n"
         "{{#items}}\n"
         "  {{sha}} {{message}}\n"
         "      {{author}}, " W_WHEN "\n"
         "{{/items}}\n"
         "---\n" MON_WORDS HOW_WORDS;

const char GH_COMMITS_BRIEF[] = REPO_BRIEF
    ", branch {{branch}}: {{#more}}more than {{/more}}{{n}} commits in "
    "the last {{days}} days. {{#items}}{{message}} ({{author}}, " A_WHEN
    "); {{/items}}The list is shown to the user as it is: do not "
    "repeat it; answer in a line or two, in the user's language.";

const char GH_REPO_LAYOUT[] = REPO
    "{{#archived}} (archived){{/archived}}{{#fork}} (a "
    "fork){{/fork}}{{#private}} (private){{/private}}\n"
    "{{#description}}{{description}}\n{{/description}}"
    "\nstars: {{stars}}, forks: {{forks}}, watching: {{watchers}}; "
    "open issues and pull requests: {{open_issues}}\n"
    "{{#language}}language: {{language}}\n{{/language}}"
    "{{#license}}licence: {{license}}\n{{/license}}"
    "main branch: {{branch}}; last push: " W_PUSHED "; created: " W_CREATED "\n"
    "{{#topics}}topics: {{topics}}\n{{/topics}}"
    "{{#homepage}}site: {{homepage}}\n{{/homepage}}"
    "{{url}}\n"
    "---\n" MON_WORDS HOW_WORDS;

const char GH_REPO_BRIEF[] = REPO_BRIEF
    "{{#archived}} (archived){{/archived}}: {{description}}; {{stars}} "
    "stars, {{forks}} forks, {{open_issues}} open issues and pull "
    "requests, {{language}}, {{license}}, last push " A_PUSHED
    ". Shown to the user: answer in a line or two, in the user's "
    "language.";
