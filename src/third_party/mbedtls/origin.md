# Mbed TLS 4.2.0, as used by Janas

- Source: <https://github.com/Mbed-TLS/mbedtls>, release `mbedtls-4.2.0`,
  published 7 July 2026.
- sha256 of the release tarball
  (`https://github.com/Mbed-TLS/mbedtls/releases/download/mbedtls-4.2.0/mbedtls-4.2.0.tar.bz2`),
  checked against the digest GitHub publishes for it:
  `2bed9d713b4668f76553b097e72b8aa30bc8f112a940d7ae228d524bbde6ffea`.
- Licence: Apache-2.0 OR GPL-2.0-or-later (see `LICENSE` and
  `tf-psa-crypto/LICENSE`); Janas takes it under Apache-2.0, which
  GPL-3.0-or-later can include. The ML-DSA code in
  `tf-psa-crypto/drivers/pqcp/mldsa-native` is Apache-2.0 OR ISC OR MIT
  (its `LICENSE`); everest and p256-m carry Apache-2.0 headers.
- Security: 4.2.0 carries the fixes of the advisories of July 2026 listed in
  its release notes.

## What is here

The part of the release its own Makefile builds into the library, **each
file unmodified**, at the same paths:

- `include/`, `library/` - Mbed TLS (TLS, X.509);
- `tf-psa-crypto/include`, `core`, `dispatch`, `extras`, `platform`,
  `utilities`, `drivers/builtin`, `drivers/everest`, `drivers/p256-m`,
  `drivers/pqcp/include` and `src`, and `drivers/pqcp/mldsa-native/mldsa`
  (the rest of mldsa-native - examples, proofs, tests - is left out) -
  TF-PSA-Crypto, the cryptography under it since 4.0.

The generated files the release carries (driver wrappers, configuration
checks) are used as they are: nothing is generated at build time.
`build.sh` compiles the list of sources the Makefile compiles, with the
default configuration and without Janas's warning flags, into the archive
of the libraries of others; it goes into the programs and into
`libjanas_mcp`, where its symbols stay hidden. Janas uses it for one thing:
the client side of TLS for `https://` MCP servers (`src/common/mcp_tls.c`).
