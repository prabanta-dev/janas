# GNU libmicrohttpd 1.0.10, as used by Janas

- Source: <https://ftp.gnu.org/gnu/libmicrohttpd/libmicrohttpd-1.0.10.tar.gz>,
  released 7 August 2026.
- sha256 of the tarball:
  `04bfe8ef75db7d629a33de767599765cecadc56274a39822d5d081030d577685`.
- Signature `libmicrohttpd-1.0.10.tar.gz.sig` checked on 23 September 2026:
  good signature by Christian Grothoff, key
  `D8423BCB326C7907033929C7939E6BE1E29FC3CC`, also in the GNU keyring.
- Licence: LGPL-2.1-or-later, or the eCos licence when built without HTTPS
  (see `COPYING`). Linked statically into `janas-server`, which is
  GPL-3.0-or-later.

## What is here

Only the files this configuration compiles, **unmodified**: 14 sources of
`src/microhttpd/`, the headers they include, and four headers of
`src/include/`.

`MHD_config.h` is not part of the tarball: it is the one `configure` wrote on
Linux x86_64 (glibc) with

```
--enable-https=no --disable-bauth --disable-dauth --disable-postprocessor
--disable-cookie --disable-httpupgrade --enable-md5=no --enable-sha256=no
--disable-sha512-256 --disable-doc --disable-examples --disable-tools
--disable-curl --with-threads=posix --disable-shared
```

`w32/MHD_config.h`, for Windows builds, is the one the same `configure`
wrote on 11 October 2026 for `--host=x86_64-w64-mingw32` (MinGW-w64, GCC
14), with the same options but `--with-threads`, which it chose itself:
Windows's own threads. It compiles the same 14 sources.

`darwin/MHD_config.h`, for macOS builds, is the one the same `configure`
wrote on 14 October 2026, natively on macOS 27.0.1 arm64 under Apple clang
21.0.0 (`/usr/bin/cc`), with the same options as the Linux run. It compiles
the same 14 sources. It carries the macOS answers: no `EPOLL_SUPPORT`, no
`HAVE_EPOLL_CREATE1`, no `HAVE_LINUX_SENDFILE`, `HAVE_ACCEPT4` or
`HAVE_SOCK_NONBLOCK`; `HAVE_DARWIN_SENDFILE` and
`HAVE_PTHREAD_SETNAME_NP_DARWIN` instead of the GNU spellings; and
`_MHD_ITC_PIPE` instead of the eventfd transport.

`build.sh` compiles the sources with `-D_GNU_SOURCE -DHAVE_CONFIG_H` and
without Janas's warning flags: warnings in code that is not Janas's are not
Janas's to fix here.

HTTPS, when it comes, needs GnuTLS, a library with dependencies of its own:
it is a separate decision.
