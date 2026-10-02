/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layouts.c - the layouts of janas-git's answers (see git.h and
 * services/common/template.h): the text the user reads, filled with the data in
 * English here and in the user's language by a client that translates the
 * layout once (janas-chat), and the brief, all the model reads of it.
 * Paths, messages and diffs stay as they are; git's own output, after a
 * command that changed something, is git's (in English).
 */
#include "git.h"

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

/* A day, with its year when not this one, and the time; the same for the
   model, "2/10, 3 days ago". */
#define W_WHEN                                                                  \
    "{{when.day}} {{when.mon|mon}}{{#when.year}} {{when.year}}{{/when.year}}, " \
    "{{when.at}}"
#define A_WHEN                                                                 \
    "{{when.day}}/{{when.mon}}{{#when.year}}/{{when.year}}{{/when.year}}, "    \
    "{{when.ago_d}} days ago"

#define REPO "{{repo}} ({{path}})"

const char GT_STATUS_LAYOUT[] =
    REPO ", {{#detached}}on no branch "
         "(detached){{/detached}}{{^detached}}branch "
         "{{branch}}{{/detached}}{{#upstream}}, following "
         "{{upstream}}{{#ahead}}; commits to push: "
         "{{ahead}}{{/ahead}}{{#behind}}; commits to pull: "
         "{{behind}}{{/behind}}{{/upstream}}{{^upstream}}, following no "
         "remote branch{{/upstream}}.\n"
         "{{#clean}}Nothing to commit: the working tree is "
         "clean.{{/clean}}{{^clean}}staged: {{staged}}, changed: "
         "{{changed}}, new: {{untracked}}{{#conflicts}}, in conflict: "
         "{{conflicts}}{{/conflicts}}{{/clean}}\n"
         "{{#files}}\n"
         "  {{path}}: {{#conflict}}in conflict{{/conflict}}{{#new}}new, not "
         "tracked{{/new}}{{#in_index}}{{in_index|change}}, "
         "staged{{/in_index}}{{#in_tree}}{{#in_index}}; "
         "{{/in_index}}{{in_tree|change}}{{/in_tree}}\n"
         "{{/files}}\n"
         "{{#more}}  ... and {{more}} more files.\n{{/more}}\n"
         "---\n"
         "change.M = modified\n"
         "change.A = added\n"
         "change.D = deleted\n"
         "change.R = renamed\n"
         "change.C = copied\n"
         "change.T = type changed\n";

const char GT_STATUS_BRIEF[] =
    REPO ", {{#detached}}detached{{/detached}}{{^detached}}branch "
         "{{branch}}{{/detached}}{{#upstream}} following {{upstream}}, "
         "{{ahead}} to push, {{behind}} to pull{{/upstream}}; "
         "{{#clean}}clean{{/clean}}{{^clean}}staged {{staged}}, changed "
         "{{changed}}, new {{untracked}}, in conflict {{conflicts}}: "
         "{{#files}}{{path}} {{in_index}}{{in_tree}}{{#new}}?{{/new}}{{#conf"
         "lict}}U{{/conflict}}; {{/files}}{{/clean}}. Shown to the user: do "
         "not repeat the list; answer in a line or two, in the user's "
         "language.";

const char GT_LOG_LAYOUT[] =
    REPO ", {{branch}}: {{#n}}{{#more}}the latest commits, "
         "{{n}}{{/more}}{{^more}}commits: {{n}}{{/more}}{{/n}}{{^n}}no "
         "commits{{/n}}\n"
         "{{#items}}\n"
         "  {{sha}} {{message}}{{#refs}} ({{refs}}){{/refs}}\n"
         "      {{author}}, " W_WHEN "\n"
         "{{/items}}\n"
         "---\n" MON_WORDS;

const char GT_LOG_BRIEF[] =
    REPO ", {{branch}}: {{#more}}the latest {{/more}}{{n}} commits. "
         "{{#items}}{{sha}} {{message}} ({{author}}, " A_WHEN
         "); {{/items}}Shown to the user: do not repeat the list; answer in "
         "a line or two, in the user's language.";

const char GT_BRANCHES_LAYOUT[] =
    REPO ": the branches, the latest worked on first{{#more}} ({{n}} "
         "listed, {{more}} more){{/more}} (* the one you are on):\n"
         "{{#items}}\n"
         "{{#current}}* {{/current}}{{^current}}  "
         "{{/current}}{{name}}{{#upstream}} -> {{upstream}}{{#track}} "
         "{{track}}{{/track}}{{/upstream}}\n"
         "      " W_WHEN ": {{message}}\n"
         "{{/items}}\n"
         "---\n" MON_WORDS;

const char GT_BRANCHES_BRIEF[] =
    REPO ": {{n}} branches. {{#items}}{{#current}}(current) "
         "{{/current}}{{name}}{{#upstream}} -> {{upstream}} "
         "{{track}}{{/upstream}}, " A_WHEN
         "; {{/items}}Shown to the user: answer in a line or two, in the "
         "user's language.";

const char GT_DIFF_LAYOUT[] =
    REPO ": {{#n}}{{#staged}}what is staged for the next "
         "commit{{/staged}}{{^staged}}the changes not "
         "staged{{/staged}}{{#only}}, in {{only}}{{/only}}; files: {{n}}, "
         "lines +{{add}} -{{del}}{{/n}}{{^n}}{{#staged}}nothing "
         "staged{{/staged}}{{^staged}}no changes but those "
         "staged{{/staged}}{{#only}} in {{only}}{{/only}}{{/n}}\n"
         "{{#files}}\n"
         "  {{path}}{{#binary}} (binary){{/binary}}{{^binary}} +{{plus}} "
         "-{{minus}}{{/binary}}\n"
         "{{/files}}\n"
         "{{#more}}  ... and {{more}} more "
         "files.\n{{/more}}{{#patch}}\n{{patch}}{{/patch}}";

const char GT_DIFF_BRIEF[] =
    REPO ": {{#staged}}staged{{/staged}}{{^staged}}not "
         "staged{{/staged}}{{#only}}, in {{only}}{{/only}}: {{n}} files, "
         "+{{add}} -{{del}}. {{#files}}{{path}} +{{plus}} -{{minus}}; "
         "{{/files}}The diff is shown to the user{{#cut}}; it "
         "begins{{/cut}}:\n{{start}}";

const char GT_DONE_LAYOUT[] =
    REPO ": {{what|done}}{{#branch}}, branch {{branch}}{{/branch}}{{#sha}}, "
         "{{sha}}{{/sha}}{{#message}}: {{message}}{{/message}}\n"
         "{{#output}}\n{{output}}\n{{/output}}\n"
         "---\n"
         "done.commit = committed\n"
         "done.pull = pulled\n"
         "done.push = pushed\n"
         "done.switch = switched\n"
         "done.create = a new branch, switched to\n";

const char GT_DONE_BRIEF[] =
    REPO ": {{what}} done{{#branch}}, branch {{branch}}{{/branch}}{{#sha}}, "
         "{{sha}}{{/sha}}. Shown to the user with git's words; answer in a "
         "line, in the user's language. git said:\n{{output}}";
