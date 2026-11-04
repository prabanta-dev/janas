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
  variant but release. Each src/<tool>/main.c and each
  src/services/<tool>/main.c becomes janas-<tool>, with
  the other .c files of its directory; the libraries of others in
  src/third_party/ go into the programs that use them; every other .c
  under src/ goes into the library. The release build
  also links libjanas_llm.so and libjanas_mcp.so, which export the public
  APIs of include/janas/llm.h and include/janas/mcp.h and nothing else.
  On Linux src/node/ becomes janas.node, the module for Node.js, with the
  library inside (janas-<variant>.node for the other variants), and its
  TypeScript declarations, janas.d.ts, beside it.

Environment
  CC            the compiler, gcc by default
  AR            the archiver, ar by default (gcc-ar for -flto)
  JANAS_CFLAGS  flags added to the variant, to try a compiler out:
                JANAS_CFLAGS="-O3 -flto" ./build.sh
  JANAS_OS      windows: the same sources for Windows x86-64, with
                MinGW-w64 (x86_64-w64-mingw32-gcc), into
                bin/x86_64-windows/ (.exe, .dll); release and debug only.
                The tests run under Wine when the build is not on Windows.
                On Windows itself, in MSYS2's MINGW64 or UCRT64 shell, it is
                set by itself, with that shell's gcc (build.bat starts it)

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
JANAS_OS="${JANAS_OS:-}"
EXE=""
# On Windows itself: MSYS2's MINGW64 or UCRT64 shell, with its gcc, glslang
# and Vulkan headers (build.bat starts it): the same build as from Linux,
# with the compilers of the shell
case "$(uname -s)" in
MINGW64_NT* | UCRT64_NT*)
    JANAS_OS=windows
    CC="${CC:-gcc}"
    AR="${AR:-ar}"
    HOSTCC="${HOSTCC:-gcc}"
    ;;
MSYS_NT* | MINGW32_NT* | CLANG*_NT* | CYGWIN_NT*)
    echo "build.sh: on Windows, build in MSYS2's MINGW64 or UCRT64 shell" \
        "(build.bat starts it)" >&2
    exit 2
    ;;
esac
if [ "$JANAS_OS" = windows ]; then
    CC="${CC:-x86_64-w64-mingw32-gcc}"
    AR="${AR:-x86_64-w64-mingw32-ar}"
    EXE=".exe"
    case "$VARIANT" in
    release | debug) ;;
    *)
        echo "build.sh: $VARIANT is not available for Windows" \
            "(release and debug are)" >&2
        exit 2
        ;;
    esac
elif [ -n "$JANAS_OS" ]; then
    echo "build.sh: unknown JANAS_OS: $JANAS_OS (windows is known)" >&2
    exit 2
fi
CC="${CC:-gcc}"

CPU="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
[ "$JANAS_OS" = windows ] && OS=windows
OBJ="lib/$CPU-$OS-$VARIANT"
BIN="bin/$CPU-$OS"
SUFFIX=""
[ "$VARIANT" != release ] && SUFFIX="-$VARIANT"

# _GNU_SOURCE for every file, empty as the files that define it write it:
# the C library of Ubuntu 22.04 (the binary package's) declares memmem,
# strcasestr and the like only with it, where a newer one does by default
# -ffp-contract=off: no multiply and add fused by the compiler on its own -
# gcc fuses them where FMA is enabled (the kernels' functions), clang within
# an expression, each version where it sees fit - so that one source gives
# the same bits with every compiler (issue #38); the FMAs the kernels write
# with intrinsics stay
CFLAGS=(-std=gnu11 -D_GNU_SOURCE= -Wall -Wextra -Werror -pthread
    -ffp-contract=off -Isrc -Iinclude)
# gcc's format-truncation is a guess that changes with each version (gcc
# 15 sees truncations gcc 14 does not, issues #50 and #51): shown, not an
# error, so that a newer compiler still builds; clang has no such option
# before 18 and would warn of it
if ! "$CC" -dM -E -x c /dev/null 2>/dev/null | grep -q __clang__; then
    CFLAGS+=(-Wno-error=format-truncation)
fi
if [ "$OS" = windows ]; then
    # Windows 10 and later; printf with the C99 formats (%zu, %lld); what
    # MinGW's C library lacks, in one header every file starts with
    # files over 2 GiB (off_t and st_size of 64 bits); the headers that
    # stand in for POSIX's (src/common/win/)
    CFLAGS+=(-D_WIN32_WINNT=0x0A00 -D__USE_MINGW_ANSI_STDIO=1
        -D_FILE_OFFSET_BITS=64 -Isrc/common/win
        -include src/common/win_compat.h)
else
    CFLAGS+=(-fPIC -fvisibility=hidden)
fi
case "$VARIANT" in
release) CFLAGS+=(-O2 -DNDEBUG) ;;
debug) CFLAGS+=(-O0 -g) ;;
# asan: aligned blocks with a tag before them, as on Windows, so that one
# given to the wrong release stops the program (src/common/aligned.h)
asan) CFLAGS+=(-O1 -g -fsanitize=address -fno-omit-frame-pointer
    -DJANAS_ALIGNED_CHECK) ;;
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
# Windows: the system's sockets, random numbers, network interfaces and
# processes; the C runtime and the threads of MinGW linked in, so that a
# program needs no DLL beyond Windows's own
WIN_LIBS=(-lws2_32 -lbcrypt -liphlpapi -lpsapi -lshell32 -lole32 -lwevtapi
    -lwlanapi -ladvapi32 -lpowrprof -lcrypt32)
[ "$OS" = windows ] && LDLIBS+=("${WIN_LIBS[@]}")
# Extra flags, for trying a compiler out: JANAS_CFLAGS="-O3 -flto" ./build.sh.
# Nothing is added when it is unset, which is how the released build is made.
# Link-time optimization needs an archiver that reads its objects, hence AR.
if [ -n "${JANAS_CFLAGS:-}" ]; then
    # shellcheck disable=SC2206
    CFLAGS+=($JANAS_CFLAGS)
    LDLIBS+=($JANAS_CFLAGS)
fi
AR="${AR:-ar}"
LDSTATIC=()
[ "$OS" = windows ] && LDSTATIC=(-static)
# A text as a C string literal for -D: backslashes and quotes escaped,
# anything but printable ASCII dropped (a tag's name, a tool's output)
c_string() {
    printf '"%s"' "$(printf '%s' "$1" | LC_ALL=C tr -cd '[:print:]' |
        sed 's/[\\"]/\\&/g')"
}
# the version in reports (janas-bench)
VERSION="$(git describe --always --dirty 2>/dev/null || echo unknown)"
CFLAGS+=(-DJANAS_VERSION="$(c_string "$VERSION")")

# Throw away what this variant made, and only what this variant made: the
# objects of the variant and the programs that carry its suffix. Release has
# no suffix, so there the programs of the other variants are kept by name.
clean()
{
    rm -rf "$OBJ"
    if [ -n "$SUFFIX" ]; then
        rm -f "$BIN"/*"$SUFFIX" "$BIN"/*"$SUFFIX".node
        return
    fi
    for f in "$BIN"/*; do
        [ -e "$f" ] || continue
        case "$f" in
        *-debug | *-asan | *-ubsan | *-tsan) continue ;;
        *-debug.node | *-asan.node | *-ubsan.node | *-tsan.node) continue ;;
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
# glslangValidator, or without the Vulkan headers, the library is built
# without GPU support - and says which is missing, here and at run time
SPV="$OBJ/spv"
mkdir -p "$SPV"
GPU_MISSING=""
XFMA_NAMES=() # the shaders with an _xfma variant
VK_INC=()
if [ "$OS" = windows ] && [ -d /usr/include/vulkan ] &&
    ! echo '#include <vulkan/vulkan.h>' | "$CC" -E -x c - >/dev/null 2>&1; then
    # the Vulkan headers are the same for every system: the host's, alone
    # in a directory of their own (never /usr/include, which is Linux's)
    mkdir -p "$OBJ/vkinc"
    for d in vulkan vk_video; do
        [ -d "/usr/include/$d" ] && ln -sfn "/usr/include/$d" "$OBJ/vkinc/$d"
    done
    VK_INC=(-I"$OBJ/vkinc")
    CFLAGS+=("${VK_INC[@]}")
fi
GLSLANG=""
if command -v glslangValidator >/dev/null 2>&1; then
    # the shaders need GL_EXT_integer_dot_product, which glslang knows from
    # 15.2 on (Ubuntu 24.04's 15.1 does not: issue #22)
    printf '#version 450\n#extension GL_EXT_integer_dot_product : require\nlayout(local_size_x = 1) in;\nvoid main() {}\n' \
        >"$SPV/probe.comp"
    if glslangValidator -V --target-env vulkan1.3 "$SPV/probe.comp" \
        -o "$SPV/probe.spv" >/dev/null 2>&1; then
        GLSLANG=yes
    else
        GPU_MISSING="a glslangValidator new enough for the shaders (this one, $(glslangValidator --version 2>/dev/null | sed -n 's/^Glslang Version: [0-9]*://p'), lacks GL_EXT_integer_dot_product: 15.2 or later)"
    fi
    rm -f "$SPV/probe.comp" "$SPV/probe.spv"
else
    GPU_MISSING="glslangValidator (glslang-tools, glslang), 15.2 or later"
fi
echo '#include <vulkan/vulkan.h>' | "$CC" "${VK_INC[@]}" -E -x c - >/dev/null 2>&1 ||
    GPU_MISSING="${GPU_MISSING:+$GPU_MISSING and }the Vulkan headers (libvulkan-dev, vulkan-headers)"
# every fma() of a shader decorated NoContraction, one operation as on the
# CPU (tools/spv_fma.c: glslang leaves them free to be split), by a tool
# built with the compiler of the machine that builds (HOSTCC, cc)
SPV_FMA="$SPV/spv_fma"
if [ -n "$GLSLANG" ] &&
    { [ ! -x "$SPV_FMA" ] || [ tools/spv_fma.c -nt "$SPV_FMA" ]; }; then
    if ! "${HOSTCC:-cc}" -O2 -o "$SPV_FMA" tools/spv_fma.c; then
        GLSLANG=""
        GPU_MISSING="${GPU_MISSING:+$GPU_MISSING and }a C compiler for the build machine (HOSTCC) to build tools/spv_fma.c"
    fi
fi
if [ -n "$GLSLANG" ]; then
    for comp in src/*/shaders/*.comp; do
        [ -f "$comp" ] || continue
        name=$(basename "${comp%.comp}")
        out="$SPV/$name.h"
        if [ ! -f "$out" ] || [ "$comp" -nt "$out" ] || [ "$SPV_FMA" -nt "$out" ]; then
            echo "SPV $comp"
            if ! glslangValidator -V --target-env vulkan1.3 \
                "$comp" -o "$SPV/$name.spv" >/dev/null ||
                ! "$SPV_FMA" "$SPV/$name.spv" "janas_spv_$name" "$out"; then
                rm -f "$out"
                GLSLANG=""
                GPU_MISSING="${GPU_MISSING:+$GPU_MISSING and }a glslangValidator that compiles $comp"
                break
            fi
            rm -f "$SPV/$name.spv"
        fi
        # a shader that calls fma() gets an _xfma variant, fma() rounded
        # once in float64 (shaders/xfma.glsl after its #extension lines),
        # for the drivers that split it (gpu_fma.c)
        grep -q 'fma(' "$comp" || continue
        xout="$SPV/${name}_xfma.h"
        if [ ! -f "$xout" ] || [ "$comp" -nt "$xout" ] ||
            [ "$SPV_FMA" -nt "$xout" ] ||
            [ src/llm/shaders/xfma.glsl -nt "$xout" ]; then
            echo "SPV $comp (xfma)"
            last=$(grep -n '^#extension\|^#version' "$comp" | tail -1 | cut -d: -f1)
            { head -n "$last" "$comp"; cat src/llm/shaders/xfma.glsl
                echo "#line $((last + 1))"; tail -n "+$((last + 1))" "$comp"; } \
                >"$SPV/${name}_xfma.comp"
            if ! glslangValidator -V --target-env vulkan1.3 \
                "$SPV/${name}_xfma.comp" -o "$SPV/${name}_xfma.spv" >/dev/null ||
                ! "$SPV_FMA" "$SPV/${name}_xfma.spv" "janas_spv_${name}_xfma" \
                    "$xout"; then
                rm -f "$xout"
                GLSLANG=""
                GPU_MISSING="${GPU_MISSING:+$GPU_MISSING and }a glslangValidator that compiles $comp with float64"
                break
            fi
            rm -f "$SPV/${name}_xfma.comp" "$SPV/${name}_xfma.spv"
        fi
        XFMA_NAMES+=("$name")
    done
fi
# the table of the variants (gpu_fma.c), rewritten only when it changes
if [ -n "$GLSLANG" ]; then
    {
        echo "/* Generated by build.sh: the shaders' _xfma variants. */"
        for name in "${XFMA_NAMES[@]}"; do
            echo "#include \"${name}_xfma.h\""
            echo "extern const uint32_t janas_spv_${name}[];"
        done
        echo "static const struct janas_xfma {"
        echo "    const uint32_t *code, *xfma;"
        echo "    size_t bytes;"
        echo "} janas_xfma_table[] = {"
        for name in "${XFMA_NAMES[@]}"; do
            echo "    {janas_spv_${name}, janas_spv_${name}_xfma, sizeof(janas_spv_${name}_xfma)},"
        done
        echo "};"
    } >"$SPV/xfma_table.h.new"
    if cmp -s "$SPV/xfma_table.h.new" "$SPV/xfma_table.h"; then
        rm -f "$SPV/xfma_table.h.new"
    else
        mv "$SPV/xfma_table.h.new" "$SPV/xfma_table.h"
    fi
fi
[ -z "$GPU_MISSING" ] ||
    echo "note: building without GPU support: it needs $GPU_MISSING"
# the reason, for the engine to give when asked for the GPU
[ -z "$GPU_MISSING" ] || CFLAGS+=(-DJANAS_GPU_WHY="$(c_string "$GPU_MISSING")")
[ -z "$GLSLANG" ] || CFLAGS+=(-DJANAS_GPU_SPV -I"$SPV")

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

# one object, rebuilt when its source or any header (or .inc, such as the
# server's routes.inc) is newer
if ! headers_newest=$(find src include "$SPV" "$GEN" \( -name '*.h' -o \
    -name '*.inc' \) -printf '%T@\n' |
    sort -n | tail -1) || [ -z "$headers_newest" ]; then
    echo "build.sh: cannot list the headers under src/ and include/" >&2
    exit 2
fi
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
# The output of a listing of sources, or a stop that names what was listed:
# read through a pipe or a process substitution its failure would be lost,
# and a program linked with fewer objects than it needs. An empty listing
# stops too, unless "may-be-empty" comes first.
sources()
{ # [may-be-empty] what was listed, then the command
    local empty="" what out
    [ "$1" = may-be-empty ] && empty=yes && shift
    what="$1"
    shift
    if ! out=$("$@"); then
        echo "build.sh: cannot list the sources of $what" >&2
        exit 2
    fi
    if [ -z "$out" ] && [ -z "$empty" ]; then
        echo "build.sh: no sources in $what" >&2
        exit 2
    fi
    [ -z "$out" ] || printf '%s\n' "$out"
}

# Libraries of others (src/third_party/<lib>/, see its origin.md), each with
# the flags it needs and without Janas's warnings, in an archive of their
# own: they go into the programs that use them, never into libjanas_llm.
TP_CFLAGS=()
skip=0
for f in "${CFLAGS[@]}"; do
    # nor Janas's stand-ins for POSIX on Windows: theirs are their own
    if [ $skip = 1 ]; then skip=0; continue; fi
    case "$f" in
    -Wall | -Wextra | -Werror | -Isrc/common/win) ;;
    -include) skip=1 ;;
    *) TP_CFLAGS+=("$f") ;;
    esac
done
TP_CFLAGS+=(-w)
tp_flags()
{
    local d="src/third_party/$1"
    case "$1" in
    libmicrohttpd)
        # Windows has a configuration of its own (see its origin.md)
        [ "$OS" = windows ] && echo "-I$d/w32"
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
    yyjson) ;;
    *)
        # without its flags it would be compiled wrong, or not at all
        echo "build.sh: no flags for the library of others $1" \
            "(src/third_party/$1): add it to tp_flags" >&2
        return 1
        ;;
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
        local s
        # a directory with no .c left is its pattern unexpanded, which is
        # no file either: both missing, both named
        for s in "$d"/library/*.c "$c"/core/*.c "$c"/extras/*.c \
            "$c"/platform/*.c "$c"/utilities/*.c "$v"/builtin/src/*.c \
            "$v"/pqcp/src/*.c "$v"/everest/library/x25519.c \
            "$v"/everest/library/Hacl_Curve25519_joined.c \
            "$v"/p256-m/p256-m_driver_entrypoints.c \
            "$v"/p256-m/p256-m/p256-m.c; do
            if [ ! -f "$s" ]; then
                echo "build.sh: $1: missing $s" >&2
                return 1
            fi
            echo "$s"
        done | sort
        ;;
    *) find "$d" -name '*.c' | sort ;;
    esac
}
tp_objs=()
tp_libs=0
for d in src/third_party/*/; do
    [ -d "$d" ] || continue
    tp_libs=$((tp_libs + 1))
    lib=$(basename "$d")
    flags=$(tp_flags "$lib") || exit 2
    # shellcheck disable=SC2206
    extra=($flags)
    tp_list=$(sources "$d" tp_sources "$lib") || exit 2
    while IFS= read -r src; do
        obj=$(objname "$src")
        tp_objs+=("$obj")
        compile "$src" "$obj" "${TP_CFLAGS[@]}" "${extra[@]}"
    done <<<"$tp_list"
done
if [ $tp_libs = 0 ]; then
    echo "build.sh: no libraries of others in src/third_party/" >&2
    exit 2
fi
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
flags=$(tp_flags mbedtls) || exit 2
# shellcheck disable=SC2206
MBEDTLS_INC=($flags)

# A directory with a main.c is a program: its other .c files are its own.
# Every other .c under src/ goes into the library, but the module for
# Node.js (src/node/), which is built on its own
# below.
tool_dirs=$(for m in src/*/main.c src/services/*/main.c; do dirname "$m"; done)
lib_list=$(sources src/ find src -name '*.c' ! -name main.c ! -name win_start.c \
    ! -path 'src/third_party/*' ! -path 'src/node/*') ||
    exit 2
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
done < <(sort <<<"$lib_list")
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
        # what the services share (src/services/common/): theirs only
        */services_*.o) ;;
        # the JSON reader and the strict reading of words, the sockets and
        # the programs started, what Windows lacks of POSIX: both
        */llm_json.o | */common_words.o | */common_sock.o | */common_proc.o | \
            */common_proc_win.o | */common_win_compat.o | */common_win_posix.o)
            mcp_objs+=("$o") llm_objs+=("$o") ;;
        *) llm_objs+=("$o") ;;
        esac
    done
    if [ "$OS" = windows ]; then
        # a DLL and its import library (libjanas_llm.dll.a) for MinGW
        for l in llm mcp; do
            if [ $l = llm ]; then lo=("${llm_objs[@]}"); else
                lo=("${mcp_objs[@]}" "$OBJ/libjanas_third.a"); fi
            echo "LD  $BIN/libjanas_$l.dll"
            "$CC" -shared -pthread "${LDSTATIC[@]}" "${lo[@]}" "${LDLIBS[@]}" \
                -Wl,--out-implib,"$BIN/libjanas_$l.dll.a" \
                -o "$BIN/libjanas_$l.dll"
        done
    else
        # --no-undefined: a function a library needs and does not have is
        # an error here, not when a program first calls it
        echo "LD  $BIN/libjanas_llm.so"
        "$CC" -shared -pthread -Wl,--no-undefined "${llm_objs[@]}" \
            "${LDLIBS[@]}" -o "$BIN/libjanas_llm.so"
        echo "LD  $BIN/libjanas_mcp.so"
        # with Mbed TLS from the archive of the libraries of others
        "$CC" -shared -pthread -Wl,--no-undefined "${mcp_objs[@]}" \
            "$OBJ/libjanas_third.a" "${LDLIBS[@]}" -o "$BIN/libjanas_mcp.so"
    fi
fi
# The module for Node.js: its own files and the library, in one janas.node
# that Node loads. The functions of Node-API are the node process's, so they
# stay undefined here - and only they: in release anything else undefined
# stops the build, as --no-undefined does for the libraries above. Linux
# only, for now: on Windows they would come from node.exe's import library.
if [ "$OS" != windows ]; then
    node_objs=()
    node_list=$(sources src/node find src/node -maxdepth 1 -name '*.c') || exit 2
    while IFS= read -r c; do
        obj=$(objname "$c")
        node_objs+=("$obj")
        compile "$c" "$obj" "${CFLAGS[@]}"
    done < <(sort <<<"$node_list")
    addon="$BIN/janas$SUFFIX.node"
    echo "LD  $addon"
    "$CC" "${CFLAGS[@]}" -shared "${node_objs[@]}" "$OBJ/libjanas.a" \
        "${LDLIBS[@]}" -o "$addon"
    if [ "$VARIANT" = release ]; then
        if ! missing=$(nm -D --undefined-only "$addon" |
            awk '$1 == "U" && $2 !~ /@/ && $2 !~ /^napi_/ { print $2 }'); then
            echo "build.sh: cannot list what $addon needs (nm)" >&2
            exit 2
        fi
        if [ -n "$missing" ]; then
            echo "build.sh: $addon needs what it does not have:" $missing >&2
            exit 2
        fi
    fi
    [ "$BIN/janas.d.ts" -nt src/node/janas.d.ts ] ||
        cp src/node/janas.d.ts "$BIN/janas.d.ts"
fi

THIRD=()
[ -f "$OBJ/libjanas_third.a" ] && THIRD=("$OBJ/libjanas_third.a")
# Windows: every program starts with the console made ready for UTF-8 and
# colours (win_start.c), and carries the manifest that makes UTF-8 its
# code page (src/common/win/janas.manifest)
START=()
if [ "$OS" = windows ]; then
    compile src/common/win_start.c "$OBJ/win_start.o" "${CFLAGS[@]}"
    res="$OBJ/janas_res.o"
    if [ ! -f "$res" ] || [ src/common/win/janas.manifest -nt "$res" ] ||
        [ src/common/win/janas.rc -nt "$res" ]; then
        echo "RC  src/common/win/janas.rc"
        "${WINDRES:-x86_64-w64-mingw32-windres}" -i src/common/win/janas.rc \
            -I src/common/win -O coff -o "$res"
    fi
    START=("$OBJ/win_start.o" "$res")
fi

# test and benchmark programs, and the C tools
for src in tests/*.c tools/*.c src/*/main.c src/services/*/main.c; do
    # a pattern that matched nothing comes here as it is
    if [ ! -f "$src" ]; then
        echo "build.sh: no sources match $src" >&2
        exit 2
    fi
    # janas-files reads POSIX paths (links, the home's private folders):
    # not on Windows yet, where a path is another thing
    [ "$OS" = windows ] && case "$src" in
    src/services/files/main.c | tests/test_files.c) continue ;;
    esac
    exe="$BIN/$(basename "${src%.c}")$SUFFIX$EXE"
    own=()
    if [ "$(basename "$src")" = main.c ]; then
        dir=$(dirname "$src")
        exe="$BIN/janas-$(basename "$dir")$SUFFIX$EXE"
        own_list=$(sources may-be-empty "$dir" \
            find "$dir" -maxdepth 1 -name '*.c' ! -name main.c) || exit 2
        while IFS= read -r c; do
            [ -n "$c" ] || continue
            obj=$(objname "$c")
            own+=("$obj")
            # the server's HTTPS reads Mbed TLS's headers, as the system's
            extra=()
            [ "$c" = src/server/tls.c ] &&
                extra=("${MBEDTLS_INC[@]/#-I/-isystem}")
            compile "$c" "$obj" "${CFLAGS[@]}" "${TP_INC[@]}" "${extra[@]}"
        done < <(sort <<<"$own_list")
    fi
    echo "LD  $exe"
    "$CC" "${CFLAGS[@]}" "${TP_INC[@]}" "$src" "${own[@]}" "${START[@]}" \
        "$OBJ/libjanas.a" "${THIRD[@]}" "${LDSTATIC[@]}" "${LDLIBS[@]}" -o "$exe"
done

if [ "$ACTION" = test ]; then
    status=0
    # the variables of this script are not the engine's, which warns of
    # a JANAS_ it does not know; a Windows build made elsewhere runs its
    # tests under Wine
    RUN=(env -u JANAS_OS -u JANAS_CFLAGS)
    if [ "$OS" = windows ] && [ "$(uname -s)" = Linux ]; then
        RUN+=(WINEDEBUG=-all wine)
    fi
    for t in "$BIN"/test_*"$SUFFIX$EXE"; do
        # in release, skip the sanitizer builds (test_x-asan and the like)
        [ -n "$SUFFIX" ] || [[ "$(basename "$t" "$EXE")" != *-* ]] || continue
        echo "RUN $t"
        "${RUN[@]}" "$t" || status=1
    done
    exit $status
fi
