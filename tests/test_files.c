/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_files.c - janas-files's tools (src/services/files/) on a home made
 * for the purpose under ~/tmp: what they read, list, search and write, and
 * what they refuse - the private files (keys, .env, Janas's own keys),
 * the folders outside the home, reached by a path, by "..", by a link -
 * and that a file replaced or edited is kept first, a dangling link
 * replaced rather than followed, a folder listed in files-dirs.txt reached.
 */
#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "services/files/place.c"
#include "services/files/read.c"
#include "services/files/tools.c"
#include "services/files/write.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static struct fl_roots roots;
static char base[PATH_MAX], home[PATH_MAX + 8], out[PATH_MAX + 8];
static char text[1 << 16];
static int is_error;

/* A call of the tool with its arguments (JSON); its text into text. */
static void call(const char *tool, const char *args)
{
    char req[8192];
    snprintf(req, sizeof req, "{\"name\": \"%s\", \"arguments\": %s}", tool,
             args);
    char err[200];
    struct janas_json_doc *d =
        janas_json_parse(req, strlen(req), err, sizeof err);
    text[0] = 0;
    is_error = -1;
    if (!d) {
        printf("bad test request %s: %s\n", req, err);
        failures++;
        return;
    }
    struct janas_buf b = {0};
    janas_buf_puts(&b, "{");
    int rc = fl_tools_call(&roots, janas_json_root(d), &b);
    janas_buf_puts(&b, "}");
    janas_json_free(d);
    struct janas_json_doc *a =
        rc == 0 ? janas_json_parse(b.p, b.n, err, sizeof err) : NULL;
    if (a) {
        const struct janas_json *r = janas_json_root(a);
        const struct janas_json *c = janas_json_get(r, "content");
        const char *t = c && c->child
                            ? janas_json_str(janas_json_get(c->child, "text"))
                            : NULL;
        snprintf(text, sizeof text, "%s", t ? t : "");
        const struct janas_json *e = janas_json_get(r, "isError");
        is_error = e && e->type == JANAS_JSON_TRUE;
        janas_json_free(a);
    }
    janas_buf_free(&b);
}

/* n bytes of s, all of it with n 0 */
static void put_file(const char *path, const char *s, size_t n)
{
    if (!n)
        n = strlen(s);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(s, 1, n, f);
        fclose(f);
    }
}

static char *slurp(const char *path)
{
    static char s[1 << 16];
    FILE *f = fopen(path, "rb");
    size_t n = f ? fread(s, 1, sizeof s - 1, f) : 0;
    if (f)
        fclose(f);
    s[n] = 0;
    return f ? s : NULL;
}

static int rm_one(const char *p, const struct stat *st, int flag,
                  struct FTW *ftw)
{
    (void)st, (void)flag, (void)ftw;
    return remove(p);
}

#define P(...) (snprintf(path, sizeof path, __VA_ARGS__), path)

static void make_home(void)
{
    char path[PATH_MAX + 64];
    mkdir(P("%s/proj", home), 0755);
    mkdir(P("%s/proj/ref", home), 0755);
    mkdir(P("%s/.ssh", home), 0700);
    mkdir(P("%s/.config", home), 0755);
    mkdir(P("%s/.config/janas", home), 0755);
    put_file(P("%s/proj/ref/FMain.class", home),
             "Public Sub Main()\n  Print \"hi\"\nEnd\n", 0);
    put_file(P("%s/proj/notes.txt", home),
             "line 1\nline 2\nline 3\nline 4\ntwice\ntwice\n", 0);
    put_file(P("%s/proj/bin.dat", home), "a\0b", 3);
    put_file(P("%s/proj/.env", home), "TOKEN=x\n", 0);
    put_file(P("%s/.ssh/id_rsa", home), "secret\n", 0);
    put_file(P("%s/.config/janas/key.txt", home), "key\n", 0);
    put_file(P("%s/secret.txt", out), "outside\n", 0);
    char target[PATH_MAX + 64];
    snprintf(target, sizeof target, "%s/secret.txt", out);
    symlink(target, P("%s/proj/link_out", home));
    snprintf(target, sizeof target, "%s/.ssh", home);
    symlink(target, P("%s/proj/link_ssh", home));
    snprintf(target, sizeof target, "%s/new.txt", out);
    symlink(target, P("%s/proj/dangle", home));
}

static void reads(void)
{
    call("files_list", "{}");
    CHECK(!is_error && strstr(text, "ref/") && strstr(text, "notes.txt") &&
              !strstr(text, ".env"),
          "list of the working folder:\n%s", text);
    call("files_list", "{\"path\": \"~\", \"hidden\": true, \"depth\": 2}");
    CHECK(!is_error && strstr(text, ".config/") && !strstr(text, ".ssh") &&
              !strstr(text, "janas/"),
          "the private folders listed:\n%s", text);
    call("files_list", "{\"path\": \".\", \"hidden\": true}");
    CHECK(!is_error && !strstr(text, ".env"), ".env listed:\n%s", text);

    /* a folder of names starting with a dot alone, as a Gambas project */
    char path[PATH_MAX + 64];
    mkdir(P("%s/proj/gb", home), 0755);
    mkdir(P("%s/proj/gb/.src", home), 0755);
    put_file(P("%s/proj/gb/.project", home), "Title=Notes\n", 0);
    put_file(P("%s/proj/gb/.src/FMain.form", home), "{ Form Form }\n", 0);
    call("files_list", "{\"path\": \"gb\", \"depth\": 2}");
    CHECK(!is_error && strstr(text, ".project") && strstr(text, ".src/") &&
              strstr(text, "FMain.form"),
          "a folder of dotted names alone:\n%s", text);
    put_file(P("%s/proj/.gitignore", home), "*.o\n", 0);
    call("files_list", "{}");
    CHECK(!is_error && strstr(text, "1 name starting with a dot not shown") &&
              !strstr(text, "link_ssh"),
          "the dotted names not said:\n%s", text);
    call("files_search", "{\"name\": \"*.form\"}");
    CHECK(!is_error && strstr(text, "gb/.src/FMain.form"),
          "search in a dotted folder: %s", text);

    call("files_read", "{\"path\": \"notes.txt\", \"offset\": 2, "
                       "\"lines\": 2}");
    CHECK(!is_error && strstr(text, "lines 2-3 of 6") &&
              strstr(text, "line 2\nline 3\n") && !strstr(text, "line 4\n"),
          "lines 2-3:\n%s", text);
    call("files_read", "{\"path\": \"bin.dat\"}");
    CHECK(is_error == 1 && strstr(text, "not text"), "binary read: %s", text);

    static const char *const refused[] = {
        "{\"path\": \"link_out\"}",
        "{\"path\": \"../../out/secret.txt\"}",
        "{\"path\": \"~/.ssh/id_rsa\"}",
        "{\"path\": \"link_ssh/id_rsa\"}",
        "{\"path\": \".env\"}",
        "{\"path\": \"~/.config/janas/key.txt\"}",
        "{\"path\": \"/etc/hostname\"}",
        "{\"path\": \"~/proj/../.ssh/id_rsa\"}",
    };
    for (size_t i = 0; i < sizeof refused / sizeof refused[0]; i++) {
        call("files_read", refused[i]);
        CHECK(is_error == 1 && !strstr(text, "secret\n") &&
                  !strstr(text, "outside\n") && !strstr(text, "TOKEN") &&
                  !strstr(text, "key\n"),
              "read %s not refused: %s", refused[i], text);
    }

    call("files_search", "{\"name\": \"*.CLASS\"}");
    CHECK(!is_error && strstr(text, "ref/FMain.class") &&
              strstr(text, "1 found"),
          "search by name: %s", text);
    call("files_search", "{\"path\": \"~\", \"text\": \"PRINT\", "
                         "\"hidden\": true}");
    CHECK(!is_error && strstr(text, "FMain.class:2: Print \"hi\"") &&
              strstr(text, "1 found"),
          "search by text: %s", text);
    call("files_search", "{\"path\": \"~\", \"text\": \"secret\", "
                         "\"hidden\": true}");
    CHECK(!is_error && strstr(text, "0 found"), "private files searched: %s",
          text);
}

static void writes(void)
{
    char path[PATH_MAX + 64];
    call("files_write", "{\"path\": \"new/Todo/a.txt\", \"content\": "
                        "\"one\\ntwo\\n\", \"make_dirs\": true}");
    char *s = slurp(P("%s/proj/new/Todo/a.txt", home));
    CHECK(!is_error && s && !strcmp(s, "one\ntwo\n"), "new file: %s / %s", text,
          s ? s : "(none)");

    call("files_write", "{\"path\": \"nodir/a.txt\", \"content\": "
                        "\"x\", \"make_dirs\": false}");
    CHECK(is_error == 1 && strstr(text, "folder is not there"),
          "make_dirs false made the folder: %s", text);
    call("files_write", "{\"path\": \"made2/b.txt\", \"content\": "
                        "\"y\"}");
    CHECK(!is_error && slurp(P("%s/proj/made2/b.txt", home)),
          "the folder not made by default: %s", text);
    call("files_write", "{\"path\": \"notes.txt\", \"content\": \"x\"}");
    CHECK(is_error == 1 && strstr(text, "already there"),
          "replaced without overwrite: %s", text);
    call("files_write", "{\"path\": \"notes.txt\", \"content\": \"x\\n\", "
                        "\"overwrite\": true}");
    s = slurp(P("%s/proj/notes.txt", home));
    int replaced = s && !strcmp(s, "x\n"); /* slurp's buffer is reused */
    const char *kept = strstr(text, "kept in ~/");
    char kp[2 * PATH_MAX + 16];
    snprintf(kp, sizeof kp, "%s/%s", home, kept ? kept + 10 : "-");
    size_t kn = strlen(kp);
    if (kn && kp[kn - 1] == '.') /* the sentence's end */
        kp[kn - 1] = 0;
    char *old = slurp(kp);
    CHECK(!is_error && replaced && old && strstr(old, "line 1\nline 2"),
          "overwrite and its copy: %s (copy %s)", text, kp);

    put_file(P("%s/proj/notes.txt", home), "a b\nc d\nc d\n", 0);
    call("files_edit", "{\"path\": \"notes.txt\", \"old_text\": \"a b\", "
                       "\"new_text\": \"A B\"}");
    s = slurp(P("%s/proj/notes.txt", home));
    CHECK(!is_error && s && !strcmp(s, "A B\nc d\nc d\n"), "edit: %s / %s",
          text, s ? s : "");
    call("files_edit", "{\"path\": \"notes.txt\", \"old_text\": \"c d\", "
                       "\"new_text\": \"x\"}");
    CHECK(is_error == 1 && strstr(text, "2 times"), "ambiguous edit: %s", text);
    call("files_edit", "{\"path\": \"notes.txt\", \"old_text\": \"c d\", "
                       "\"new_text\": \"x\", \"all\": true}");
    s = slurp(P("%s/proj/notes.txt", home));
    CHECK(!is_error && s && !strcmp(s, "A B\nx\nx\n"), "edit all: %s", text);
    call("files_edit", "{\"path\": \"notes.txt\", \"old_text\": \"zz\", "
                       "\"new_text\": \"x\"}");
    CHECK(is_error == 1 && strstr(text, "not in"), "edit not found: %s", text);

    static const char *const refused[] = {
        "{\"path\": \"~/.ssh/new\", \"content\": \"x\"}",
        "{\"path\": \"link_out\", \"content\": \"x\", \"overwrite\": true}",
        "{\"path\": \"../../out/new2.txt\", \"content\": \"x\"}",
        "{\"path\": \"link_ssh/new\", \"content\": \"x\"}",
        "{\"path\": \".env\", \"content\": \"x\", \"overwrite\": true}",
        "{\"path\": \"~/../out/deep/x.txt\", \"content\": \"x\", "
        "\"make_dirs\": true}",
    };
    for (size_t i = 0; i < sizeof refused / sizeof refused[0]; i++) {
        call("files_write", refused[i]);
        CHECK(is_error == 1, "write %s not refused: %s", refused[i], text);
    }
    s = slurp(P("%s/secret.txt", out));
    CHECK(s && !strcmp(s, "outside\n"), "the file outside changed");
    CHECK(!slurp(P("%s/new2.txt", out)) && !slurp(P("%s/deep/x.txt", out)),
          "a file made outside");

    /* a dangling link is replaced by the file, not followed out */
    call("files_write", "{\"path\": \"dangle\", \"content\": \"in\\n\"}");
    struct stat st;
    CHECK(!is_error && lstat(P("%s/proj/dangle", home), &st) == 0 &&
              S_ISREG(st.st_mode) && !slurp(P("%s/new.txt", out)),
          "dangling link: %s", text);

    call("files_mkdir", "{\"path\": \"made/deep\"}");
    CHECK(!is_error && stat(P("%s/proj/made/deep", home), &st) == 0 &&
              S_ISDIR(st.st_mode),
          "mkdir: %s", text);
    call("files_mkdir", "{\"path\": \"~/../out/x\"}");
    CHECK(is_error == 1 && stat(P("%s/x", out), &st) != 0, "mkdir outside: %s",
          text);
}

/* a folder listed in files-dirs.txt is reached, its private files not */
static void listed(void)
{
    char path[PATH_MAX + 64], line[PATH_MAX + 64];
    snprintf(line, sizeof line, "# my folders\n%s\n", out);
    put_file(P("%s/.config/janas/files-dirs.txt", home), line, 0);
    put_file(P("%s/id_ed25519", out), "k\n", 0);
    snprintf(line, sizeof line, "{\"path\": \"%s/secret.txt\"}", out);
    call("files_read", line);
    CHECK(!is_error && strstr(text, "outside"), "listed folder: %s", text);
    snprintf(line, sizeof line, "{\"path\": \"%s/id_ed25519\"}", out);
    call("files_read", line);
    CHECK(is_error == 1, "a private key in a listed folder: %s", text);
}

int main(void)
{
    const char *h = getenv("HOME");
    char parent[PATH_MAX - 32];
    snprintf(parent, sizeof parent, "%s/tmp", h ? h : "/tmp");
    mkdir(parent, 0700);
    snprintf(base, sizeof base, "%s/janas-files-test-XXXXXX", parent);
    if (!mkdtemp(base)) {
        perror("mkdtemp");
        return 1;
    }
    snprintf(home, sizeof home, "%s/home", base);
    snprintf(out, sizeof out, "%s/out", base);
    mkdir(home, 0755);
    mkdir(out, 0755);
    make_home();
    char proj[PATH_MAX + 16];
    snprintf(proj, sizeof proj, "%s/proj", home);
    if (setenv("HOME", home, 1) != 0 || chdir(proj) != 0 ||
        fl_roots_load(&roots) != 0) {
        printf("test_files: cannot set up the home\n");
        return 1;
    }
    reads();
    writes();
    listed();
    nftw(base, rm_one, 16, FTW_DEPTH | FTW_PHYS);
    if (failures) {
        printf("test_files: %d failures\n", failures);
        return 1;
    }
    printf("test_files: ok\n");
    return 0;
}
