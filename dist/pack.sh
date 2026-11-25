#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Maurizio Cammalleri
#
# pack.sh OS ARCH - the binary package of a release, from what build.sh put
# in bin/<ARCH>-<OS>/: janas-<os>-<arch> (.tar.gz, .zip on Windows) with the
# SHA-256 of every file inside and of the archive beside it, and the GPU's
# checks alone, janas-diag-<os>-<arch> (where there is a GPU to check). OS:
# linux, darwin (packed as macos) or windows. Run by the release workflow
# from the repository's top; the same for every system, what differs said
# where it does.
set -euo pipefail
OS="$1" ARCH="$2"
B="bin/$ARCH-$OS"
NAME=$OS
[ "$OS" = darwin ] && NAME=macos
PACKAGE="janas-$NAME-$ARCH" DIAG="janas-diag-$NAME-$ARCH"
EXE="" LIB=so
[ "$OS" = windows ] && EXE=.exe LIB=dll
[ "$OS" = darwin ] && LIB=dylib
[ -d "$B" ] || { echo "pack.sh: no $B: build first" >&2; exit 2; }

sums() { # the SHA-256 of every file of a folder, in it
    (cd "$1" && find . -type f ! -name SHA256SUMS -print0 | sort -z |
        xargs -0 sha256sum > SHA256SUMS)
}
archive() { # folder's parent, folder, archive's name: .zip or .tar.gz
    if [ "$OS" = windows ]; then
        (cd "$1" && zip -q -r -X "../$3.zip" "$2")
        sha256sum "$3.zip" > "$3.zip.sha256" && cat "$3.zip.sha256"
    else
        tar -C "$1" -czf "$3.tar.gz" "$2"
        sha256sum "$3.tar.gz" > "$3.tar.gz.sha256" && cat "$3.tar.gz.sha256"
    fi
}

rm -rf "$PACKAGE" "$DIAG"
P="$PACKAGE/janas"
mkdir -p "$P/services" "$P/lib" "$P/include/janas" "$P/docs"
progs=(janas-chat janas-server janas-get janas-setup janas-bench janas-mcp brebus
    gguf2jns jns_planes hf2jns_mtp hf2jns_ds4 hf2jns_vision hf2jns_whisper
    silero2jns wespeaker2jns jns_check vkcheck gpu_check)
# Brebus served to the users of the system: its setuid helper and its PAM
# service, Linux's
[ "$OS" = linux ] && progs+=(janas-brebus-helper)
for f in "${progs[@]}"; do
    install -m 755 "$B/$f$EXE" "$P/"
done
[ "$OS" = linux ] && install -m 644 dist/brebus.pam "$P/brebus.pam"
if [ "$OS" = windows ]; then
    # Brebus's window: WebView2's loader beside brebus.exe, from Microsoft's
    # SDK (src/third_party/webview2/README.md), checked
    curl -sfL -o wv2.nupkg \
        https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/1.0.4314.35
    echo "90778deb1b77e36292828b4b5833c9255c27533f40e0ddd55891ae718944edc4  wv2.nupkg" |
        sha256sum -c -
    # the loader of the package's processor, x64 or arm64
    wv=x64 wvsum=d3fce90bdeecb08596e377dea6a316354dd8949e32ff09b89d51be00c14e21b6
    [ "$ARCH" = aarch64 ] &&
        wv=arm64 wvsum=05429881d918fd64a3fe4955155c1af79e66d7b7d944034467874f420bc2265a
    unzip -q -j wv2.nupkg "build/native/$wv/WebView2Loader.dll" -d "$P/"
    echo "$wvsum  $P/WebView2Loader.dll" | sha256sum -c -
    install -m 644 src/third_party/webview2/LICENSE.txt "$P/LICENSE-WebView2.txt"
    rm -f wv2.nupkg
fi
for s in src/services/*/; do
    s=$(basename "$s")
    [ "$s" = common ] && continue
    # a service not built for this system stays out
    [ -f "$B/janas-$s$EXE" ] || continue
    install -m 755 "$B/janas-$s$EXE" "$P/services/"
    # each service's page, its links made to point beside it
    install -m 644 "src/services/$s/README.md" "$P/services/janas-$s.md"
    # its instructions for the model, a skill in Claude's format
    install -Dm 644 "src/services/$s/SKILL.md" "$P/skills/janas-$s/SKILL.md"
done
sed -E 's#\(([a-z]+)/README\.md\)#(janas-\1.md)#g' \
    src/services/README.md > "$P/services/README.md"
install -m 755 "$B/libjanas_llm.$LIB" "$B/libjanas_mcp.$LIB" "$P/lib/"
# MinGW's import libraries, for programs linked against the DLLs
[ "$OS" = windows ] && install -m 644 "$B/libjanas_llm.dll.a" "$B/libjanas_mcp.dll.a" "$P/lib/"
if [ "$OS" = linux ]; then
    # the library as a module for Node.js, with its types for TypeScript
    mkdir -p "$P/node"
    install -m 755 "$B/janas.node" "$P/node/"
    install -m 644 "$B/janas.d.ts" "$P/node/"
fi
install -m 644 include/janas/* "$P/include/janas/"
install -m 644 docs/*.md "$P/docs/"
case "$OS" in
windows) install -m 644 dist/README-windows.md "$P/README.md" ;;
darwin) install -m 644 dist/README-macos.md "$P/README.md" ;;
*) install -m 644 dist/README.md "$P/README.md" ;;
esac
install -m 644 LICENSE ChangeLog.md MODELS.md "$P/"
# nothing but programs, libraries and text: no script of any kind
if find "$P" -type f \( -name '*.sh' -o -name '*.py' -o -name '*.bat' \) | grep -q .; then
    echo "pack.sh: a script in the package" >&2
    exit 1
fi
if [ "$OS" = linux ]; then
    # the module, as packed, loads where the machine has Node.js
    if command -v node >/dev/null; then
        node -e 'const j = require(process.argv[1]);
          if (j.abiVersion() !== 1) process.exit(1);
          console.log("janas.node " + j.version + " loads in Node " + process.version)' \
            "$(realpath "$P/node/janas.node")"
    fi
    # the newest C library the programs ask for
    find "$P" -type f -perm -u+x ! -name '*.so' -print0 |
        xargs -0 -n 1 objdump -T | grep -o 'GLIBC_[0-9.]*' |
        sort -V | uniq | tail -1
fi
sums "$P"
archive "$PACKAGE" janas "$PACKAGE"

# vkcheck and gpu_check alone, for a tester who only has to tell whether the
# GPU works: no model, no engine, a few megabytes (not on macOS: no Vulkan
# there yet)
[ "$OS" = darwin ] && exit 0
D="$DIAG/janas-diag"
mkdir -p "$D"
install -m 755 "$B/vkcheck$EXE" "$B/gpu_check$EXE" "$D/"
install -m 644 dist/README-diag.md "$D/README.md"
install -m 644 LICENSE "$D/"
sums "$D"
archive "$DIAG" janas-diag "$DIAG"
