/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_skills.c - janas-skills, the program beside this one with the same
 * suffix, over MCP: a skill made in a directory of its own under $TMPDIR
 * (else /tmp), listed, read, a file outside it refused - by its path and
 * by a link - and, where the cage can be built, a script of it run by
 * skill_run and by its own tool: its arguments, its input, its exit code,
 * no network, the file it wrote.
 */
#ifdef _WIN32
#include <stdio.h>

int main(void)
{
    printf("test_skills: skipped on Windows\n");
    return 0;
}
#else
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "janas/mcp.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static char text[65536];

/* a call's text into text: 0, 1 when the tool said it failed, -1 */
static int call(janas_mcp *m, const char *tool, const char *args)
{
    text[0] = 0;
    if (janas_mcp_call(m, tool, (int32_t)strlen(tool), args,
                       (int32_t)strlen(args), 60) != JANAS_MCP_OK)
        return -1;
    int32_t len = 0;
    janas_mcp_result(m, text, sizeof text - 1, &len);
    text[len < (int32_t)sizeof text ? len : 0] = 0;
    return janas_mcp_result_error(m) ? 1 : 0;
}

static void put(const char *path, const char *s)
{
    FILE *f = fopen(path, "w");
    if (f) {
        fputs(s, f);
        fclose(f);
    }
}

static void rm_rf(const char *dir)
{
    char cmd[4400];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
    if (system(cmd) != 0)
        printf("test_skills: %s left behind\n", dir);
}

int main(int argc, char **argv)
{
    (void)argc;
    char self[1024], name[1024], exe[2200];
    snprintf(self, sizeof self, "%s", argv[0]);
    const char *base = strrchr(self, '/');
    snprintf(name, sizeof name, "%s", base ? base + 1 : self);
    const char *suffix = strchr(name, '-');
    snprintf(exe, sizeof exe, "%s/janas-skills%s", dirname(self),
             suffix ? suffix : "");

    const char *t = getenv("TMPDIR");
    char root[1100], d[1400], p[1600], cfg[1200], cache[1200];
    snprintf(root, sizeof root, "%s/janas-skills-XXXXXX", t && *t ? t : "/tmp");
    if (!mkdtemp(root)) {
        printf("FAIL: no directory under %s\n", t && *t ? t : "/tmp");
        return 1;
    }
    snprintf(d, sizeof d, "%s/sk/hello/scripts", root);
    char mk[1500];
    snprintf(mk, sizeof mk, "mkdir -p '%s'", d);
    if (system(mk) != 0) {
        printf("FAIL: %s not made\n", d);
        return 1;
    }
    snprintf(p, sizeof p, "%s/sk/hello/SKILL.md", root);
    put(p, "---\nname: hello\ndescription: \"Says hello.\"\n---\n"
           "Run scripts/hi.py with a name.\n");
    snprintf(p, sizeof p, "%s/sk/hello/scripts/hi.py", root);
    put(p, "import socket, sys\n"
           "print('hello', ' '.join(sys.argv[1:]), sys.stdin.read())\n"
           "open('out.txt', 'w').write('x' * 10)\n"
           "try:\n"
           "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
           "    print('network open')\n"
           "except OSError:\n"
           "    print('no network')\n"
           "sys.exit(3 if len(sys.argv) > 2 else 0)\n");
    snprintf(p, sizeof p, "%s/secret.txt", root);
    put(p, "secret\n");
    char link[1600];
    snprintf(link, sizeof link, "%s/sk/hello/leak.txt", root);
    if (symlink(p, link) != 0)
        printf("FAIL: no link made\n"), failures++;
    snprintf(cfg, sizeof cfg, "XDG_CONFIG_HOME=%s/cfg", root);
    snprintf(cache, sizeof cache, "XDG_CACHE_HOME=%s/cache", root);
    snprintf(d, sizeof d, "%s/sk", root);

    const char *av[] = {exe, "--dir", d};
    const char *env[] = {cfg, cache};
    janas_mcp *m = NULL;
    if (janas_mcp_open_command("skills", 3, av, 2, env, 20, &m) !=
        JANAS_MCP_OK) {
        printf("FAIL: %s not started: %s\n", exe, janas_mcp_last_error());
        rm_rf(root);
        return 1;
    }
    int r = call(m, "skills_list", "{}");
    CHECK(r == 0 && strstr(text, "- hello: Says hello."), "list: %d %s", r,
          text);
    r = call(m, "skill_read", "{\"name\": \"hello\"}");
    CHECK(r == 0 && strstr(text, "Run scripts/hi.py") &&
              strstr(text, "- scripts/hi.py") && !strstr(text, "leak.txt"),
          "read: %d %s", r, text);
    r = call(m, "skill_read",
             "{\"name\": \"hello\", \"file\": \"scripts/hi.py\"}");
    CHECK(r == 0 && strstr(text, "import socket"), "read a file: %d %s", r,
          text);
    r = call(m, "skill_read",
             "{\"name\": \"hello\", \"file\": \"../../secret.txt\"}");
    CHECK(r == 1 && !strstr(text, "secret\n"), "a path out: %d %s", r, text);
    r = call(m, "skill_read", "{\"name\": \"hello\", \"file\": \"leak.txt\"}");
    CHECK(r == 1 && !strstr(text, "secret\n"), "a link out: %d %s", r, text);
    r = call(m, "skill_read", "{\"name\": \"nope\"}");
    CHECK(r == 1, "no such skill: %d %s", r, text);

    /* the tools: the three, and one for the script */
    static char tools[65536];
    int32_t len = 0;
    janas_mcp_tools(m, "", 0, tools, sizeof tools - 1, &len);
    tools[len < (int32_t)sizeof tools ? len : 0] = 0;
    CHECK(strstr(tools, "\"skill_hello_hi_py\""), "the script's tool: %s",
          tools);
    CHECK(janas_mcp_tool_read_only(m, "skill_read", 10) == 1 &&
              janas_mcp_tool_read_only(m, "skill_run", 9) != 1,
          "read-only hints");

    /* the cage is the service's to build: where it cannot, it says so */
    r = call(m, "skill_run",
             "{\"name\": \"hello\", \"script\": \"scripts/hi.py\", "
             "\"args\": [\"Ada\"], \"input\": \"from input\"}");
    if (r == 1 &&
        (strstr(text, "cannot be built") || strstr(text, "is not installed"))) {
        printf("test_skills: the runs skipped: %s\n", text);
    } else {
        CHECK(r == 0 && strstr(text, "exit code 0") &&
                  strstr(text, "hello Ada from input") &&
                  strstr(text, "no network") && strstr(text, "- out.txt (10"),
              "run: %d %s", r, text);
        r = call(m, "skill_hello_hi_py", "{\"args\": [\"a\", \"b\"]}");
        CHECK(r == 1 && strstr(text, "exit code 3") &&
                  strstr(text, "hello a b"),
              "its own tool: %d %s", r, text);
        r = call(m, "skill_run",
                 "{\"name\": \"hello\", \"script\": \"leak.txt\"}");
        CHECK(r == 1, "a link run: %d %s", r, text);
    }
    janas_mcp_close(m);
    rm_rf(root);
    if (failures) {
        printf("test_skills: %d failures\n", failures);
        return 1;
    }
    printf("test_skills: ok\n");
    return 0;
}
#endif
