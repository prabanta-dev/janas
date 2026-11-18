/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * pyinstall.c - Python for the cage on Windows (sandbox_win.c), installed
 * only when the user asks for it from the page: the embeddable package of
 * python.org and the wheels of numpy and matplotlib from PyPI, each file
 * at the version below and checked against its SHA-256 (python.org's and
 * PyPI's own, written here) before anything of it is used. Janas's
 * packages carry none of it.
 *
 * Into %LOCALAPPDATA%\janas\python\runtime.new, made whole there - the
 * interpreter unpacked, the wheels unpacked into Lib\site-packages, its
 * python314._pth told where they are - and only then named runtime, so a
 * stopped or failed install leaves no Python that half works. A name of an
 * archive's file that would land outside the folder stops it. Removed by
 * the page's button, the folder deleted. On Linux the cage runs the
 * system's python3, and nothing is installed.
 */
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/sandbox.h"
#include "common/hf_fetch.h"
#include "common/sha256.h"
#include "common/zip.h"

#ifdef _WIN32

#define PY_VERSION "3.14.10"

static const struct pyfile {
    const char *url, *sha256;
    size_t bytes;
    int wheel; /* 0: the interpreter's package; 1: a wheel */
} FILES[] = {
    {"https://www.python.org/ftp/python/3.14.10/python-3.14.10-embed-amd64.zip",
     "90905e64a840557d8c1bc4f5b74db3d7ee85e3ffb217611cac1c7813acea4159",
     12619664, 0},
    {"https://files.pythonhosted.org/packages/74/e2/494516771374b43060c1b778b371cba53d576abd3aee1275eecd73896e3b/numpy-2.5.5-cp314-cp314-win_amd64.whl",
     "20145b11776d2f7570cd61351b156a51d3a341822282095baa819d87272bc710",
     12710543, 1},
    {"https://files.pythonhosted.org/packages/ea/61/4315ef0d63937fe56f451887b4a0be3b8e8b22cf1d92a8d157caac54570d/matplotlib-3.11.2-cp314-cp314-win_amd64.whl",
     "ecea603dd2fbf8242fd31a305a8b12a4ece2de28096870c65fdd0d1e35b8d9a6",
     9505972, 1},
    {"https://files.pythonhosted.org/packages/15/41/3df8cc14bb572c8b447f9dfbc876703f0f2f2896058285b5c87a1d4a638d/contourpy-1.4.0-cp314-cp314-win_amd64.whl",
     "181bea01bc742734ae672fa00c717d855dd35e1f029536406538c52e7b0cd73d", 240101,
     1},
    {"https://files.pythonhosted.org/packages/e7/05/c19819d5e3d95294a6f5947fb9b9629efb316b96de511b418c53d245aae6/cycler-0.12.1-py3-none-any.whl",
     "85cef7cff222d8644161529808465972e51340599459b8ac3ccbac5a854e0d30", 8321,
     1},
    {"https://files.pythonhosted.org/packages/96/41/c9ca629ffe1b883c1422540a8a11a4ea09616366ab585ab4ee9833f2cd75/fonttools-4.68.0-cp314-cp314-win_amd64.whl",
     "1782f2cf952eb262c4126b1f57b5f0e027e71bf2b632350a033dec526f448428",
     2536130, 1},
    {"https://files.pythonhosted.org/packages/4e/0e/49522e1ab5788cbaf63a26fbd3b851f9028616828c961b8a31b35cb96df8/kiwisolver-1.5.1-cp314-cp314-win_amd64.whl",
     "f4167e87b397f273dc2356fcf1eaf50a6bac51e6105f45103ef7129c8efb0255", 72282,
     1},
    {"https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl",
     "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c", 129956,
     1},
    {"https://files.pythonhosted.org/packages/50/e2/7ad17747517e4c813dca78257d6bb593466f8231baf1384bc98d92745edb/pillow-13.0.0-cp314-cp314-win_amd64.whl",
     "f7cbb7a64e7ec664775195e4f81923bc8a93d9451b963f844e59118f63147a7a",
     7281830, 1},
    {"https://files.pythonhosted.org/packages/38/bb/d215ee7c73b61497b28a5503f9f53523f294fcc936762b7caf90e0c1c2b5/pyparsing-3.3.3-py3-none-any.whl",
     "ece8c00a69cf01b45d0b1dedabb469c90d8caf996d4fda40f147627a122849a4", 126420,
     1},
    {"https://files.pythonhosted.org/packages/2f/a9/52b89563d154433d79b51ed1eb4842bf7334cb2696891a7fe135618853d7/python_dateutil-4.0.0-py3-none-any.whl",
     "a5e246f79eb80475e691bf93862cead98decd8a498cf53ba0516fcdc0f50dc1c", 92853,
     1},
    {"https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl",
     "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274", 11050,
     1},
};
#define N_FILES (sizeof FILES / sizeof *FILES)

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int running;
static size_t done_bytes;
static char step[200], failed[300];

static size_t total_bytes(void)
{
    size_t t = 0;
    for (size_t i = 0; i < N_FILES; i++)
        t += FILES[i].bytes;
    return t;
}

static void dirs(char *base, size_t bn, char *rt, size_t rn)
{
    const char *la = getenv("LOCALAPPDATA");
    snprintf(base, bn, "%s/janas/python", la ? la : ".");
    janas_path_slashes(base);
    snprintf(rt, rn, "%s/runtime", base);
}

/* a folder and what it holds, gone */
static void rmtree(const char *path)
{
    DIR *d = opendir(path);
    if (d) {
        struct dirent *e;
        char p[4400];
        while ((e = readdir(d))) {
            if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
                continue;
            snprintf(p, sizeof p, "%s/%s", path, e->d_name);
            struct stat st;
            if (stat(p, &st) == 0 && S_ISDIR(st.st_mode))
                rmtree(p);
            else
                remove(p);
        }
        closedir(d);
    }
    remove(path);
}

struct unpack {
    const char *dest; /* the folder written into */
    int wheel;
};

/* an archive's file written under the folder; a name that would leave it
   stops the install */
static int put(void *ctx, const char *name, size_t nn, const uint8_t *p,
               size_t pn)
{
    struct unpack *u = ctx;
    char rel[1024], path[4400];
    if (nn == 0 || nn >= sizeof rel)
        return 0;
    memcpy(rel, name, nn);
    rel[nn] = 0;
    for (char *c = rel; *c; c++)
        if (*c == '\\')
            *c = '/';
    if (rel[0] == '/' || strchr(rel, ':') || !strncmp(rel, "../", 3) ||
        strstr(rel, "/../") || (nn >= 3 && !strcmp(rel + nn - 3, "/..")))
        return -2;
    /* a wheel's .data folder (scripts, headers) is not needed to import */
    char *first = strchr(rel, '/');
    if (u->wheel && first && first - rel > 5 && !strncmp(first - 5, ".data", 5))
        return 0;
    snprintf(path, sizeof path, "%s/%s", u->dest, rel);
    if (rel[nn - 1] == '/')
        return janas_mkdirs(path) == 0 ? 0 : -3;
    char dir[4400];
    snprintf(dir, sizeof dir, "%s", path);
    *strrchr(dir, '/') = 0;
    if (janas_mkdirs(dir) != 0)
        return -3;
    FILE *f = fopen(path, "wb");
    if (!f)
        return -3;
    int bad = fwrite(p, 1, pn, f) != pn;
    bad |= fclose(f) != 0;
    return bad ? -3 : 0;
}

static void say(const char *s)
{
    pthread_mutex_lock(&mu);
    snprintf(step, sizeof step, "%s", s);
    pthread_mutex_unlock(&mu);
}

static void fail(const char *s)
{
    pthread_mutex_lock(&mu);
    snprintf(failed, sizeof failed, "%s", s);
    running = 0;
    pthread_mutex_unlock(&mu);
}

static void *install(void *arg)
{
    (void)arg;
    char base[4096], rt[4200], tmp[4300], site[4400], why[300];
    dirs(base, sizeof base, rt, sizeof rt);
    snprintf(tmp, sizeof tmp, "%s.new", rt);
    snprintf(site, sizeof site, "%s/Lib/site-packages", tmp);
    rmtree(tmp); /* a stopped install's half, gone */
    if (janas_mkdirs(site) != 0) {
        fail("cannot make the folder for Python");
        return NULL;
    }
    for (size_t i = 0; i < N_FILES; i++) {
        const char *name = strrchr(FILES[i].url, '/') + 1;
        char s[300];
        snprintf(s, sizeof s, "%s (%zu of %zu)", name, i + 1, N_FILES);
        say(s);
        struct janas_buf b = {0};
        if (janas_hf_get_url(FILES[i].url, NULL, NULL, &b, NULL, 0, why,
                             sizeof why) != 0) {
            janas_buf_free(&b);
            snprintf(s, sizeof s, "%s: %s", name, why);
            fail(s);
            rmtree(tmp);
            return NULL;
        }
        struct janas_sha256 h;
        char hex[65];
        janas_sha256_init(&h);
        janas_sha256_update(&h, b.p, b.n);
        janas_sha256_hex(&h, hex);
        if (b.n != FILES[i].bytes || strcmp(hex, FILES[i].sha256)) {
            janas_buf_free(&b);
            snprintf(s, sizeof s,
                     "%s came different from the file expected: nothing "
                     "installed",
                     name);
            fail(s);
            rmtree(tmp);
            return NULL;
        }
        struct unpack u = {.dest = FILES[i].wheel ? site : tmp,
                           .wheel = FILES[i].wheel};
        int rc = janas_zip_each((const uint8_t *)b.p, b.n, (size_t)256 << 20,
                                put, &u, why, sizeof why);
        janas_buf_free(&b);
        if (rc != 0) {
            snprintf(s, sizeof s, "%s: %s", name,
                     rc == -2   ? "a file named outside its folder"
                     : rc == -3 ? "a file could not be written"
                                : why);
            fail(s);
            rmtree(tmp);
            return NULL;
        }
        pthread_mutex_lock(&mu);
        done_bytes += FILES[i].bytes;
        pthread_mutex_unlock(&mu);
    }
    /* where the embeddable package looks for modules: its zip, its
       folder, and the wheels' folder */
    char pth[4500];
    snprintf(pth, sizeof pth, "%s/python314._pth", tmp);
    FILE *f = fopen(pth, "w");
    int bad = !f || fputs("python314.zip\n.\nLib\\site-packages\n", f) < 0;
    if (f)
        bad |= fclose(f) != 0;
    if (bad) {
        fail("cannot write python314._pth");
        rmtree(tmp);
        return NULL;
    }
    rmtree(rt);
    if (rename(tmp, rt) != 0) {
        fail("cannot name the Python folder");
        rmtree(tmp);
        return NULL;
    }
    say("installed");
    pthread_mutex_lock(&mu);
    running = 0;
    pthread_mutex_unlock(&mu);
    return NULL;
}

#endif /* _WIN32 */

int brebus_python_installed(void)
{
#ifdef _WIN32
    char base[4096], rt[4200], exe[4300];
    dirs(base, sizeof base, rt, sizeof rt);
    snprintf(exe, sizeof exe, "%s/python.exe", rt);
    return access(exe, F_OK) == 0;
#else
    return 0;
#endif
}

void brebus_python_status(struct janas_buf *o)
{
#ifdef _WIN32
    pthread_mutex_lock(&mu);
    janas_buf_printf(o,
                     "{\"needed\": true, \"version\": \"%s\", \"installed\": "
                     "%s, \"running\": %s, \"done\": %zu, \"total\": %zu, "
                     "\"step\": ",
                     PY_VERSION, brebus_python_installed() ? "true" : "false",
                     running ? "true" : "false", done_bytes, total_bytes());
    janas_json_write_str(o, step, strlen(step));
    janas_buf_puts(o, ", \"error\": ");
    janas_json_write_str(o, failed, strlen(failed));
    janas_buf_puts(o, ", \"from\": \"python.org (the interpreter) and PyPI "
                      "(numpy, matplotlib and what they need)\"}");
    pthread_mutex_unlock(&mu);
#else
    janas_buf_puts(o, "{\"needed\": false}"); /* the system's python3 */
#endif
}

int brebus_python_install(void)
{
#ifdef _WIN32
    pthread_mutex_lock(&mu);
    if (running) {
        pthread_mutex_unlock(&mu);
        return 0;
    }
    running = 1;
    done_bytes = 0;
    step[0] = failed[0] = 0;
    pthread_mutex_unlock(&mu);
    pthread_t t;
    if (pthread_create(&t, NULL, install, NULL) != 0) {
        fail("no thread for the install");
        return -1;
    }
    pthread_detach(t);
    return 0;
#else
    return -1;
#endif
}

int brebus_python_remove(void)
{
#ifdef _WIN32
    char base[4096], rt[4200];
    pthread_mutex_lock(&mu);
    int busy = running;
    pthread_mutex_unlock(&mu);
    if (busy)
        return -1;
    dirs(base, sizeof base, rt, sizeof rt);
    rmtree(rt);
    pthread_mutex_lock(&mu); /* the last install's progress, gone with it */
    step[0] = failed[0] = 0;
    done_bytes = 0;
    pthread_mutex_unlock(&mu);
    /* the cage's own temporary folder and its AppContainer too: Python
       gone, nothing of it left */
    char tmp[4300];
    snprintf(tmp, sizeof tmp, "%s/tmp", base);
    rmtree(tmp);
    janas_sandbox_forget();
    return brebus_python_installed() ? -1 : 0;
#else
    return -1;
#endif
}
