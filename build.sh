#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Maurizio Cammalleri
# Build script for Janas: ./build.sh --help says it all.
# No -march=native: CPU-specific kernels are selected at run time.
set -euo pipefail

cd "$(dirname "$0")"

usage()
{
    cat <<'EOF'
Janas build script.

  ./build.sh [variant] [action]

  variant: release (default) | debug | asan | ubsan | tsan
  action:  build (default) | test | clean | clean-build | clean-test

Variants
  release  -O2, assertions off. What is shipped, and the only build a
           speed number may come from; the only one that also makes the
           shared libraries.
  debug    -O0 -g. Slow but honest under a debugger: nothing is inlined
           away and every variable sits where the source says.
  asan     AddressSanitizer. Reads and writes past the end of a buffer,
           use after free, leaks at exit. About 2x slower, 3x the memory.
  ubsan    UndefinedBehaviorSanitizer. What the C standard leaves
           undefined and a compiler may turn into anything: signed
           overflow, a shift as wide as the type, a misaligned or null
           pointer. Stops at the first one (-fno-sanitize-recover).
  tsan     ThreadSanitizer. Data races: two threads at the same memory
           with no lock or atomic between them. The one to run after
           touching the thread pool, the expert cache or the readers.
           5-15x slower, and it holds a lot of memory.

  A sanitizer reports at run time, so it has to be the `test` action, or
  a program of the variant run by hand. Two sanitizers cannot share a
  process: run them one after the other. Their programs carry a
  -<variant> suffix (bin/<cpu>-<os>/test_pool-asan), so they never
  overwrite the release build, and each variant keeps its own objects.

Actions
  build        compile the library, the tests, the tools and the programs
  test         build, then run every test_* of the variant; exit 1 if one
               fails
  clean        remove the objects of the variant and the programs it made,
               and nothing else: the other variants stay where they are.
               On its own it frees disk space, which is why it is there
  clean-build  the two in a row, for a build from nothing: what a changed
               compiler, changed flags or a doubt about a stale object
               asks for
  clean-test   the same, then the tests. The honest answer before a commit

Output
  Objects and the static library go to lib/<cpu>-<os>-<variant>/,
  programs to bin/<cpu>-<os>/, with a -<variant> suffix for every
  variant but release. Each src/<tool>/main.c becomes janas-<tool>, with
  the other .c files of its directory; the libraries of others in
  src/third_party/ go into the programs that use them; every other .c
  under src/ goes into the library. The release build
  also links libjanas_llm.so and libjanas_mcp.so, which export the public
  APIs of include/janas/llm.h and include/janas/mcp.h and nothing else.

Environment
  CC            the compiler, gcc by default
  AR            the archiver, ar by default (gcc-ar for -flto)
  JANAS_CFLAGS  flags added to the variant, to try a compiler out:
                JANAS_CFLAGS="-O3 -flto" ./build.sh

Before a commit: ./build.sh release test and at least asan; tsan too
when concurrent code was touched.

Examples
  ./build.sh                 a release build
  ./build.sh release test    build it and run the tests
  ./build.sh asan test       the same under AddressSanitizer
  ./build.sh tsan clean-test everything again from nothing, under tsan
  ./build.sh tsan clean      throw the tsan objects away, free the disk
EOF
}

case "${1:-}" in
-h | --help | help)
    usage
    exit 0
    ;;
esac

VARIANT="${1:-release}"
ACTION="${2:-build}"
CC="${CC:-gcc}"

CPU="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
OBJ="lib/$CPU-$OS-$VARIANT"
BIN="bin/$CPU-$OS"
SUFFIX=""
[ "$VARIANT" != release ] && SUFFIX="-$VARIANT"

CFLAGS=(-std=gnu11 -Wall -Wextra -Werror -pthread -fPIC -fvisibility=hidden
    -Isrc -Iinclude)
case "$VARIANT" in
release) CFLAGS+=(-O2 -DNDEBUG) ;;
debug) CFLAGS+=(-O0 -g) ;;
asan) CFLAGS+=(-O1 -g -fsanitize=address -fno-omit-frame-pointer) ;;
ubsan) CFLAGS+=(-O1 -g -fsanitize=undefined -fno-sanitize-recover=all) ;;
tsan) CFLAGS+=(-O1 -g -fsanitize=thread) ;;
*)
    echo "build.sh: unknown variant: $VARIANT" >&2
    case "$VARIANT" in
    build | test | clean)
        echo "build.sh: the variant comes first: ./build.sh release" \
            "$VARIANT" >&2 ;;
    esac
    echo "build.sh: --help says which variants there are" >&2
    exit 2
    ;;
esac

case "$ACTION" in
build | test | clean | clean-build | clean-test) ;;
*)
    echo "build.sh: unknown action: $ACTION" >&2
    echo "build.sh: --help says which actions there are" >&2
    exit 2
    ;;
esac
LDLIBS=(-lm)
# Extra flags, for trying a compiler out: JANAS_CFLAGS="-O3 -flto" ./build.sh.
# Nothing is added when it is unset, which is how the released build is made.
# Link-time optimization needs an archiver that reads its objects, hence AR.
if [ -n "${JANAS_CFLAGS:-}" ]; then
    # shellcheck disable=SC2206
    CFLAGS+=($JANAS_CFLAGS)
    LDLIBS+=($JANAS_CFLAGS)
fi
AR="${AR:-ar}"
# the version in reports (janas-bench)
VERSION="$(git describe --always --dirty 2>/dev/null || echo unknown)"
CFLAGS+=(-DJANAS_VERSION="\"$VERSION\"")

# Throw away what this variant made, and only what this variant made: the
# objects of the variant and the programs that carry its suffix. Release has
# no suffix, so there the programs of the other variants are kept by name.
clean()
{
    rm -rf "$OBJ"
    if [ -n "$SUFFIX" ]; then
        rm -f "$BIN"/*"$SUFFIX"
        return
    fi
    for f in "$BIN"/*; do
        [ -e "$f" ] || continue
        case "$f" in
        *-debug | *-asan | *-ubsan | *-tsan) continue ;;
        esac
        rm -f "$f"
    done
}

case "$ACTION" in
clean)
    clean
    exit 0
    ;;
clean-build)
    clean
    ACTION=build
    ;;
clean-test)
    clean
    ACTION=test
    ;;
esac

mkdir -p "$OBJ" "$BIN"

# GPU shaders (src/*/shaders/*.comp) as SPIR-V in C arrays; without
# glslangValidator the library is built without GPU support
SPV="$OBJ/spv"
mkdir -p "$SPV"
if command -v glslangValidator >/dev/null 2>&1; then
    for comp in src/*/shaders/*.comp; do
        [ -f "$comp" ] || continue
        name=$(basename "${comp%.comp}")
        out="$SPV/$name.h"
        if [ ! -f "$out" ] || [ "$comp" -nt "$out" ]; then
            echo "SPV $comp"
            glslangValidator -V --target-env vulkan1.3 --vn "janas_spv_$name" \
                "$comp" -o "$out" >/dev/null
        fi
    done
    CFLAGS+=(-DJANAS_GPU_SPV -I"$SPV")
fi

# Web pages of the programs (src/<tool>/*.html) as C arrays, so that the
# program carries them: janas_<tool>_<name>_html in <tool>_<name>_html.h.
GEN="$OBJ/gen"
mkdir -p "$GEN"
for html in src/*/*.html; do
    [ -f "$html" ] || continue
    d=$(basename "$(dirname "$html")")
    n=$(basename "${html%.html}")
    out="$GEN/${d}_${n}_html.h"
    if [ ! -f "$out" ] || [ "$html" -nt "$out" ]; then
        echo "GEN $html"
        {
            echo "/* Generated by build.sh from $html - do not edit. */"
            echo "static const char janas_${d}_${n}_html[] = {"
            od -An -v -tx1 "$html" |
                awk '{ for (i = 1; i <= NF; i++) printf "0x%s,", $i; print "" }'
            echo "0};"
        } >"$out"
    fi
done
CFLAGS+=(-I"$GEN")

# one object, rebuilt when its source or any header is newer
headers_newest=$(find src include "$SPV" "$GEN" -name '*.h' -printf '%T@\n' | sort -n | tail -1)
compile()
{ # source, object, then the flags
    local src="$1" obj="$2"
    shift 2
    if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ] ||
        [ "$(stat -c %Y "$obj")" -lt "${headers_newest%.*}" ]; then
        echo "CC  $src"
        "$CC" "$@" -c "$src" -o "$obj"
    fi
}
objname() { echo "$OBJ/$(echo "${1#src/}" | tr '/' '_' | sed 's/\.c$/.o/')"; }

# Libraries of others (src/third_party/<lib>/, see its origin.md), each with
# the flags it needs and without Janas's warnings, in an archive of their
# own: they go into the programs that use them, never into libjanas_llm.
TP_CFLAGS=()
for f in "${CFLAGS[@]}"; do
    case "$f" in -Wall | -Wextra | -Werror) ;; *) TP_CFLAGS+=("$f") ;; esac
done
TP_CFLAGS+=(-w)
tp_flags()
{
    local d="src/third_party/$1"
    case "$1" in
    libmicrohttpd)
        echo "-D_GNU_SOURCE -DHAVE_CONFIG_H -I$d -I$d/src/include" \
            "-I$d/src/microhttpd" ;;
    mbedtls)
        local c="$d/tf-psa-crypto" v="$d/tf-psa-crypto/drivers"
        echo "-I$d/library -I$d/include -I$c/include -I$v/builtin/include" \
            "-I$v/everest/include" \
            "-I$v/everest/include/tf-psa-crypto/private/everest" \
            "-I$v/everest/include/tf-psa-crypto/private/everest/kremlib" \
            "-I$v/p256-m/p256-m/include -I$v/p256-m/p256-m/include/p256-m" \
            "-I$v/p256-m/p256-m_driver_interface -I$v/pqcp/include" \
            "-I$c/core -I$v/builtin/src -I$c/dispatch -I$c/extras" \
            "-I$c/platform -I$c/utilities -I$v/pqcp/src" \
            "-I$v/pqcp/mldsa-native/mldsa" ;;
    esac
}
# The sources of a library of others: all its .c files, or for Mbed TLS the
# list its own Makefile compiles (others are included by these, or are not
# part of the library).
tp_sources()
{
    local d="src/third_party/$1"
    case "$1" in
    mbedtls)
        local c="$d/tf-psa-crypto" v="$d/tf-psa-crypto/drivers"
        ls "$d"/library/*.c "$c"/core/*.c "$c"/extras/*.c "$c"/platform/*.c \
            "$c"/utilities/*.c "$v"/builtin/src/*.c "$v"/pqcp/src/*.c \
            "$v"/everest/library/x25519.c \
            "$v"/everest/library/Hacl_Curve25519_joined.c \
            "$v"/p256-m/p256-m_driver_entrypoints.c \
            "$v"/p256-m/p256-m/p256-m.c 2>/dev/null | sort
        ;;
    *) find "$d" -name '*.c' | sort ;;
    esac
}
tp_objs=()
for d in src/third_party/*/; do
    [ -d "$d" ] || continue
    lib=$(basename "$d")
    # shellcheck disable=SC2207
    extra=($(tp_flags "$lib"))
    while IFS= read -r src; do
        obj=$(objname "$src")
        tp_objs+=("$obj")
        compile "$src" "$obj" "${TP_CFLAGS[@]}" "${extra[@]}"
    done < <(tp_sources "$lib")
done
rm -f "$OBJ/libjanas_third.a"
# archives made anew: ar only adds and replaces, so the objects of a source
# since removed or renamed would stay in them and be linked twice
if [ ${#tp_objs[@]} -gt 0 ]; then
    rm -f "$OBJ/libjanas_third.a"
    "$AR" rcs "$OBJ/libjanas_third.a" "${tp_objs[@]}"
fi
# their public headers, for the programs that use them
TP_INC=(-Isrc/third_party/libmicrohttpd/src/include -Isrc/third_party/yyjson)
# Mbed TLS's, for the TLS layer of the MCP client (src/common/mcp_tls.c)
# shellcheck disable=SC2207
MBEDTLS_INC=($(tp_flags mbedtls))

# A directory with a main.c is a program: its other .c files are its own.
# Every other .c under src/ goes into the library.
tool_dirs=$(for m in src/*/main.c; do dirname "$m"; done)
objs=()
while IFS= read -r src; do
    grep -qx "$(dirname "$src")" <<<"$tool_dirs" && continue
    obj=$(objname "$src")
    objs+=("$obj")
    extra=()
    # the TLS layer reads Mbed TLS's headers, as the system's: their
    # warnings are not Janas's to fix
    [ "$src" = src/common/mcp_tls.c ] && extra=("${MBEDTLS_INC[@]/#-I/-isystem}")
    compile "$src" "$obj" "${CFLAGS[@]}" "${extra[@]}"
done < <(find src -name '*.c' ! -name main.c ! -path 'src/third_party/*' | sort)
rm -f "$OBJ/libjanas.a"
"$AR" rcs "$OBJ/libjanas.a" "${objs[@]}"
if [ "$VARIANT" = release ]; then
    # two shared libraries, each with its own public API: libjanas_mcp is
    # the MCP client (src/common/mcp_*, with the JSON reader it uses),
    # libjanas_llm all the rest; a program may use either without the other
    llm_objs=()
    mcp_objs=()
    for o in "${objs[@]}"; do
        case "$o" in
        */common_mcp_*.o) mcp_objs+=("$o") ;;
        */common_hf_*.o) ;; # for the tools, with libjanas.a only
        */llm_json.o) mcp_objs+=("$o") llm_objs+=("$o") ;;
        *) llm_objs+=("$o") ;;
        esac
    done
    echo "LD  $BIN/libjanas_llm.so"
    "$CC" -shared -pthread "${llm_objs[@]}" "${LDLIBS[@]}" \
        -o "$BIN/libjanas_llm.so"
    echo "LD  $BIN/libjanas_mcp.so"
    # with Mbed TLS from the archive of the libraries of others
    "$CC" -shared -pthread "${mcp_objs[@]}" "$OBJ/libjanas_third.a" \
        "${LDLIBS[@]}" -o "$BIN/libjanas_mcp.so"
fi
THIRD=()
[ -f "$OBJ/libjanas_third.a" ] && THIRD=("$OBJ/libjanas_third.a")

# test and benchmark programs, and the C tools
for src in tests/*.c tools/*.c src/*/main.c; do
    exe="$BIN/$(basename "${src%.c}")$SUFFIX"
    own=()
    if [ "$(basename "$src")" = main.c ]; then
        dir=$(dirname "$src")
        exe="$BIN/janas-$(basename "$dir")$SUFFIX"
        while IFS= read -r c; do
            obj=$(objname "$c")
            own+=("$obj")
            compile "$c" "$obj" "${CFLAGS[@]}" "${TP_INC[@]}"
        done < <(find "$dir" -maxdepth 1 -name '*.c' ! -name main.c | sort)
    fi
    echo "LD  $exe"
    "$CC" "${CFLAGS[@]}" "${TP_INC[@]}" "$src" "${own[@]}" "$OBJ/libjanas.a" \
        "${THIRD[@]}" "${LDLIBS[@]}" -o "$exe"
done

if [ "$ACTION" = test ]; then
    status=0
    for t in "$BIN"/test_*"$SUFFIX"; do
        # in release, skip the sanitizer builds (test_x-asan and the like)
        [ -n "$SUFFIX" ] || [[ "$(basename "$t")" != *-* ]] || continue
        echo "RUN $t"
        "$t" || status=1
    done
    exit $status
fi
