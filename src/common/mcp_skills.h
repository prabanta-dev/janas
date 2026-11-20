/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_skills.h - skills in the format of Claude's, as Brebus and the
 * service janas-skills find them: a folder holding SKILL.md - a
 * frontmatter with its name and description, then the instructions - and
 * the files the instructions name. Every chat's live in skills/ beside the
 * user's mcp.json (~/.config/janas/skills, %APPDATA%\janas\skills).
 */
#ifndef JANAS_COMMON_MCP_SKILLS_H
#define JANAS_COMMON_MCP_SKILLS_H

#include <stddef.h>

/* the folder of every chat's skills, beside mcp.json: 0, or -1 */
int janas_skills_dir(char *out, size_t n);

/* whether that folder holds a skill: 1, or 0 */
int janas_skills_any(void);

/* what the service janas-skills is about for a catalog: the skills of
   every chat, each with its description, written into o (nothing when
   there are none) */
struct janas_buf;
void janas_skills_about(struct janas_buf *o);

/* a skill's name and description from dir/SKILL.md (the name: its
   frontmatter's, else folder, made fit for a tool's argument: letters,
   digits, - _ .): 0, or -1 with no SKILL.md */
int janas_skill_info(const char *dir, const char *folder, char *name,
                     size_t name_n, char *about, size_t about_n);

#endif
