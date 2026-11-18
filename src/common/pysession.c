/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * pysession.c - a Python session in the cage of sandbox.c, as a notebook:
 * the driver run by python3 -I -c, which reads pieces of code (their
 * length, a newline, their bytes), runs each keeping the variables, and
 * answers each with a JSON of what it printed, its traceback and the
 * figures it drew; and the exchange of a piece of code for its answer.
 * janas-server's code_interpreter (src/server/containers.c) and Brebus's
 * python tool use it. Linux only, as the cage.
 */
#include "common/pysession.h"

#include <stdio.h>

/* Linux's cage (sandbox.c) and Windows's (sandbox_win.c) */
#if (defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))) ||   \
    defined(_WIN32)

#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

const char JANAS_PY_DRIVER[] =
    "import sys, os, io, ast, json, traceback, contextlib\n"
    "_o = os.fdopen(os.dup(1), 'wb', 0)\n"
    "os.dup2(2, 1)\n"
    "_i = os.fdopen(0, 'rb', 0)\n"
    "_g = {'__name__': '__main__', '__builtins__': __builtins__}\n"
    "os.makedirs(os.environ.get('MPLCONFIGDIR', '/tmp/mpl'), "
    "exist_ok=True)\n"
    "def _rd(n):\n"
    "    b = b''\n"
    "    while len(b) < n:\n"
    "        c = _i.read(n - len(b))\n"
    "        if not c:\n"
    "            os._exit(0)\n"
    "        b += c\n"
    "    return b\n"
    "def _len():\n"
    "    b = b''\n"
    "    while not b.endswith(bytes([10])):\n"
    "        b += _rd(1)\n"
    "    return int(b)\n"
    "_D = os.environ.get('JANAS_PY_DATA', '/mnt/data')\n"
    "def _pngs():\n"
    "    try:\n"
    "        return {n: os.stat(os.path.join(_D, n)).st_mtime_ns for n in\n"
    "                os.listdir(_D) if n.endswith('.png')}\n"
    "    except OSError:\n"
    "        return {}\n"
    "def _figs(before):\n"
    "    m = sys.modules.get('matplotlib.pyplot')\n"
    "    out = []\n"
    "    if m is None:\n"
    "        return out\n"
    "    if _pngs() != before:\n"
    "        m.close('all')\n"
    "        return out\n"
    "    for num in m.get_fignums():\n"
    "        k = 1\n"
    "        while os.path.exists(os.path.join(_D, 'figure_%d.png' % k)):\n"
    "            k += 1\n"
    "        p = os.path.join(_D, 'figure_%d.png' % k)\n"
    "        try:\n"
    "            m.figure(num).savefig(p, format='png', dpi=100)\n"
    "            out.append(p)\n"
    "        except Exception:\n"
    "            pass\n"
    "    m.close('all')\n"
    "    return out\n"
    "while True:\n"
    "    code = _rd(_len()).decode('utf-8', 'replace')\n"
    "    before = _pngs()\n"
    "    buf = io.StringIO()\n"
    "    err = None\n"
    "    with contextlib.redirect_stdout(buf), "
    "contextlib.redirect_stderr(buf):\n"
    "        try:\n"
    "            t = ast.parse(code, '<cell>', 'exec')\n"
    "            last = None\n"
    "            if t.body and isinstance(t.body[-1], ast.Expr):\n"
    "                last = ast.Expression(t.body.pop().value)\n"
    "            exec(compile(t, '<cell>', 'exec'), _g)\n"
    "            if last is not None:\n"
    "                v = eval(compile(last, '<cell>', 'eval'), _g)\n"
    "                if v is not None:\n"
    "                    print(repr(v))\n"
    "        except SystemExit:\n"
    "            pass\n"
    "        except BaseException:\n"
    "            e = sys.exc_info()\n"
    "            tb = e[2].tb_next if e[2] is not None else None\n"
    "            err = ''.join(traceback.format_exception(e[0], e[1], tb))\n"
    "    try:\n"
    "        figs = _figs(before)\n"
    "    except Exception:\n"
    "        figs = []\n"
    "    r = json.dumps({'logs': buf.getvalue()[-200000:], 'error': err,\n"
    "                    'images': figs}).encode()\n"
    "    _o.write(str(len(r)).encode() + bytes([10]) + r)\n";

char *janas_py_exchange(struct janas_sandbox *sb, const char *code, int seconds,
                        struct janas_buf *raw, char *why, size_t n)
{
    char head[32];
    size_t len = strlen(code);
    int hl = snprintf(head, sizeof head, "%zu\n", len);
    if (write(sb->in, head, (size_t)hl) != hl ||
        write(sb->in, code, len) != (ssize_t)len) {
        snprintf(why, n, "the Python session is gone");
        return NULL;
    }
    struct janas_buf out = {0};
    time_t end = time(NULL) + seconds;
    long want = -1;
    for (;;) {
        /* the answer: its length, a newline, its bytes */
        if (want < 0) {
            char *nl = out.n ? memchr(out.p, '\n', out.n) : NULL;
            if (nl) {
                want = strtol(out.p, NULL, 10);
                size_t skip = (size_t)(nl - out.p) + 1;
                memmove(out.p, out.p + skip, out.n - skip);
                out.n -= skip;
            }
        }
        if (want >= 0 && out.n >= (size_t)want) {
            janas_buf_put(&out, "", 1);
            if (out.oom)
                break;
            return out.p;
        }
        int left = (int)(end - time(NULL));
        if (left <= 0) {
            snprintf(why, n,
                     "The code ran past %d seconds and was stopped; the "
                     "Python session started over (its variables are lost).",
                     seconds);
            janas_buf_free(&out);
            return NULL;
        }
        struct pollfd p[2] = {{sb->out, POLLIN, 0}, {sb->err, POLLIN, 0}};
        if (poll(p, 2, left * 1000) <= 0)
            continue;
        for (int i = 0; i < 2; i++) {
            if (!(p[i].revents & (POLLIN | POLLHUP)))
                continue;
            char buf[8192];
            ssize_t got = read(i ? sb->err : sb->out, buf, sizeof buf);
            if (got <= 0) {
                if (i == 0 || got < 0) {
                    /* what it said on its way out, if anything: a
                       traceback tells a broken start from the code */
                    struct pollfd e = {sb->err, POLLIN, 0};
                    while (poll(&e, 1, 300) > 0 && (e.revents & POLLIN)) {
                        ssize_t k = read(sb->err, buf, sizeof buf);
                        if (k <= 0)
                            break;
                        janas_buf_put(raw, buf, (size_t)k);
                    }
                    size_t t = raw->n > 600 ? raw->n - 600 : 0;
                    snprintf(why, n,
                             "The Python session ended (out of memory, or the "
                             "code ended it); it starts over at the next "
                             "piece of code.%s%.*s",
                             raw->n ? " Its last words: " : "",
                             (int)(raw->n - t), raw->n ? raw->p + t : "");
                    janas_buf_free(&out);
                    return NULL;
                }
                continue;
            }
            if (i)
                janas_buf_put(raw, buf, (size_t)got);
            else
                janas_buf_put(&out, buf, (size_t)got);
        }
    }
    janas_buf_free(&out);
    snprintf(why, n, "out of memory");
    return NULL;
}

#else

const char JANAS_PY_DRIVER[] = "";

char *janas_py_exchange(struct janas_sandbox *sb, const char *code, int seconds,
                        struct janas_buf *raw, char *why, size_t n)
{
    (void)sb, (void)code, (void)seconds, (void)raw;
    snprintf(why, n, "no Python session here: Linux only");
    return NULL;
}

#endif
