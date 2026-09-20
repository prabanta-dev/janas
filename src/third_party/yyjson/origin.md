# yyjson 0.13.0, as used by Janas

- Source: <https://github.com/ibireme/yyjson>, tag `0.13.0`, released
  8 September 2026.
- sha256 of the tag's tarball
  (`https://github.com/ibireme/yyjson/archive/refs/tags/0.13.0.tar.gz`):
  `34e0f62a2bc11ab20d601e8ca1cc2b2079503aa45119a19133d89d19b94a0fae`.
- Licence: MIT (see `LICENSE`).
- Security: 0.13.0 carries the fix for the advisory of 8 September 2026
  (integer truncation bypassing the bigint digit cap). The 2024 advisory
  concerns the pool allocator, which Janas does not use.

## What is here

`yyjson.c` and `yyjson.h` from `src/`, **unmodified**, compiled into
`janas-server` without Janas's warning flags.
