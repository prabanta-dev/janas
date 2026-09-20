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

`build.sh` compiles the sources with `-D_GNU_SOURCE -DHAVE_CONFIG_H` and
without Janas's warning flags: warnings in code that is not Janas's are not
Janas's to fix here.

HTTPS, when it comes, needs GnuTLS, a library with dependencies of its own:
it is a separate decision.
