#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Maurizio Cammalleri
#
# janas-try.sh - try Janas on this machine and, if you agree, report how it
# went as a GitHub issue. It builds Janas, downloads a model from Hugging
# Face, converts it, checks that the answers are the ones every other
# machine gets, measures the speed, tries the HTTP server, and writes a
# report with no names or paths in it. ./tools/janas-try.sh --help says it
# all.
#
# Nothing is installed and nothing needs root: it runs the programs the
# build makes, in this directory, and asks before each long step and before
# anything leaves the machine.
set -uo pipefail
cd "$(dirname "$0")/.."

REPO=prabanta-dev/janas
BIN=bin/$(uname -m)-$(uname -s | tr '[:upper:]' '[:lower:]')
WORK=janas-try
MODELS=models
LEVELS=""
YES=0
KEEP=0
BENCH=1
REPORT=$WORK/janas-try-report.md
STATE=$WORK/state

usage()
{
    cat <<'EOF'
janas-try.sh - try Janas on this machine, and report how it went

  ./tools/janas-try.sh [options]

It builds Janas (./build.sh release test), downloads a model, converts it,
checks the model's answers against the ones every machine should give,
measures the speed with janas-bench, tries janas-server, and writes a
report. It asks before each long step, and shows you the report before
anything is sent; sending it (a GitHub issue) is always your choice.

Levels (it offers those this machine can hold):
  quick    Qwen3-4B                     2.5 GB download, ~5 GB of disk
  medium   Qwen3.6-35B-A3B              22 GB download, ~45 GB of disk
  full     Qwen3-Next-80B-A3B + MTP     52 GB download, ~100 GB of disk
  gemma    Gemma-4-E4B + its assistant  5.1 GB download, ~11 GB of disk

Options
  --level quick|medium|full|gemma
                              the level (repeat for several; default: ask)
  --dir DIR                   where the models go (default: ./models)
  --yes                       do not ask before each step (sending the
                              report is still asked)
  --keep                      keep the models at the end without asking
  --no-bench                  skip the speed measurement
  --help

Run again after an interruption: it goes on from where it stopped. What it
did and found is in ./janas-try/.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
    --level) LEVELS="$LEVELS $2"; shift ;;
    --dir) MODELS="$2"; shift ;;
    --yes) YES=1 ;;
    --keep) KEEP=1 ;;
    --no-bench) BENCH=0 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "janas-try: unknown option $1 (--help lists them)" >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "$WORK" "$MODELS"
# absolute paths: the speed test runs in its own directory
BIN=$PWD/$BIN
WORK=$PWD/$WORK
MODELS=$(cd "$MODELS" && pwd)
REPORT=$WORK/janas-try-report.md
STATE=$WORK/state
touch "$STATE"
LOG=$WORK/janas-try.log
: > "$WORK/models" # the models this run used, for the report
: > "$WORK/results" # this run's
: >> "$WORK/created" # every file this script put in $MODELS

say() { printf '\n== %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
fail() { printf '\njanas-try: %s\n' "$*" >&2; }
done_() { grep -qx "$1" "$STATE"; }
mark() { done_ "$1" || echo "$1" >> "$STATE"; }
ask()
{ # a question answered with y or n; --yes answers y (but see publish)
    [ "$YES" = 1 ] && [ "${2:-}" != always ] && return 0
    local a
    read -r -p "   $1 [y/N] " a </dev/tty || return 1
    [ "$a" = y ] || [ "$a" = Y ] || [ "$a" = yes ]
}
gib() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

# ---------------------------------------------------------------- models
# level: the model's name for janas-get (which downloads, checks and
# converts it, with the fingerprints of MODELS.md), its repository and
# GGUF (for the report), the file that runs and its prediction file ("-":
# none, or inside it), the tools that made it, then download, disk while
# converting and the memory of the smallest machine, in bytes and GB
model_of()
{
    case "$1" in
    quick) echo "qwen3-4b Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf \
qwen3-4b-q4km.jns - planes 2684354560 5368709120 8" ;;
    medium) echo "qwen3.6-35b-a3b bartowski/Qwen_Qwen3.6-35B-A3B-GGUF \
Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf qwen3.6-35b-a3b-q4km.jns - planes \
23622320128 48318382080 16" ;;
    full) echo "qwen3-next-80b-a3b Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF \
Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf qwen3-next-80b-a3b-q4km.jns \
qwen3-next-80b-a3b-mtp-q4k.jns planes 55834574848 107374182400 16" ;;
    gemma) echo "gemma-4-e4b unsloth/gemma-4-E4B-it-GGUF \
gemma-4-E4B-it-Q4_K_M.gguf gemma-4-e4b-it-q4km.jns gemma-4-e4b-it-mtp.jns \
one 5476083302 11811160064 8" ;;
    esac
}

# The answer to a fixed prompt at temperature 0, cut at 64 tokens, as a
# fingerprint of its text: the kernels give the same numbers bit for bit on
# every x86-64 machine, whatever the threads, so the text should not change
# either - free text, where one number off changes the words after it
# (checked here: the same with the defaults, AVX2 kernels forced, 4 CPUs,
# no GPU, no drafts). What the engine would choose by the memory is fixed:
# all six bits of the down matrix, all the experts, exact attention - or a
# machine with less memory would compute other numbers, rightly.
FP_PROMPT="Write a short paragraph about the island of Sardinia, its history and its landscape."
fp_expected()
{
    case "$1" in
    quick) echo "c08f0f936eeba5cb" ;;
    medium) echo "13e2f198c270b826" ;;
    full) echo "4aae064b918a0e75" ;;
    gemma) echo "b2c01def5ff6bd5c" ;;
    esac
}

# ---------------------------------------------------------------- machine
machine()
{
    CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')
    THREADS=$(nproc)
    FLAGS=$(grep -m1 '^flags' /proc/cpuinfo)
    ISA=""
    for f in avx2 avx_vnni avx512f avx512_vnni; do
        case " $FLAGS " in *" $f "*) ISA="$ISA $f" ;; esac
    done
    MEM_TOTAL=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo)
    MEM_AVAIL=$(awk '/MemAvailable/ { print $2 * 1024 }' /proc/meminfo)
    DISK_FREE=$(df -B1 --output=avail "$MODELS" | tail -1 | tr -d ' ')
    local dev
    dev=$(df --output=source "$MODELS" | tail -1)
    DISK=$(lsblk -no MODEL "$(lsblk -no PKNAME "$dev" 2>/dev/null | head -1 |
        sed 's|^|/dev/|')" 2>/dev/null | head -1 | sed 's/ *$//')
    [ -n "$DISK" ] || DISK=unknown
    case "$(lsblk -dno ROTA "$(lsblk -no PKNAME "$dev" 2>/dev/null | head -1 |
        sed 's|^|/dev/|')" 2>/dev/null)" in
    *1*) DISK="$DISK (spinning)" ;;
    esac
    KERNEL=$(uname -r)
    DISTRO=$( . /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")
    grep -qi microsoft /proc/version 2>/dev/null && DISTRO="$DISTRO (WSL)"
    # the GPU: whether the build can have it (the shaders need
    # glslangValidator) and what Vulkan sees - name, kind, version
    if ! command -v glslangValidator >/dev/null; then
        GPU_BUILD="without GPU support (glslangValidator not installed)"
    elif ! echo '#include <vulkan/vulkan.h>' |
        ${CC:-gcc} -E -x c - >/dev/null 2>&1; then
        GPU_BUILD="without GPU support (Vulkan headers not installed)"
    else
        GPU_BUILD="with GPU support"
    fi
    if command -v vulkaninfo >/dev/null; then
        GPU_VK=$(vulkaninfo --summary 2>/dev/null | awk '
            /apiVersion/ { v = $3 }
            /deviceType/ { t = $3; sub("PHYSICAL_DEVICE_TYPE_", "", t) }
            /deviceName/ { sub(/^[^=]*= */, ""); printf "%s%s (%s, Vulkan %s)",
                           n++ ? "; " : "", $0, tolower(t), v }')
        [ -n "$GPU_VK" ] || GPU_VK="Vulkan sees no device"
    else
        GPU_VK="vulkaninfo not installed"
    fi
    CC_VER=$( (${CC:-gcc} --version 2>/dev/null || clang --version 2>/dev/null) |
        head -1)
    # mains when an adapter (or a USB-C source) is plugged in, battery when
    # none is and there is a battery; a port with nothing in it says nothing
    POWER="unknown"
    local d
    for d in /sys/class/power_supply/*; do
        case "$(cat "$d/type" 2>/dev/null)" in
        Mains | USB*)
            [ "$(cat "$d/online" 2>/dev/null)" = 1 ] && POWER="mains" ;;
        Battery)
            [ "$POWER" = unknown ] && POWER="battery" ;;
        esac
    done
    LOAD=$(cut -d' ' -f1 /proc/loadavg)
    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
}

preflight()
{
    say "This machine"
    [ "$(uname -s)" = Linux ] || { fail "Linux only, for now"; exit 1; }
    [ "$(uname -m)" = x86_64 ] || { fail "x86-64 only, for now"; exit 1; }
    local missing=""
    for t in bash curl sha256sum awk df lsblk; do
        command -v "$t" >/dev/null || missing="$missing $t"
    done
    command -v "${CC:-gcc}" >/dev/null || command -v clang >/dev/null ||
        missing="$missing gcc-or-clang"
    [ -z "$missing" ] || { fail "missing:$missing"; exit 1; }
    machine
    note "CPU:    $CPU, $THREADS threads,$ISA"
    note "memory: $(gib "$MEM_TOTAL") GiB, $(gib "$MEM_AVAIL") GiB free now"
    note "disk:   $DISK, $(gib "$DISK_FREE") GiB free for the models"
    note "system: $DISTRO, Linux $KERNEL, $CC_VER"
    note "GPU:    $GPU_VK; Janas will be built $GPU_BUILD"
    note "power:  $POWER; load now $LOAD"
    case " $FLAGS " in *" avx2 "*) ;; *)
        note "no AVX2: Janas runs on its portable path, much slower";;
    esac
    [ "$POWER" = battery ] && note "on battery: the speeds will be lower"
    awk -v l="$LOAD" -v t="$THREADS" 'BEGIN { exit !(l > t / 4) }' &&
        note "the machine is busy: the speeds will say so"
    ask "Build Janas and go on?" || exit 0
}

# ---------------------------------------------------------------- build
build()
{
    done_ build && [ -x "$BIN/janas-chat" ] && return 0
    say "Building Janas (./build.sh release test): a few minutes"
    if ./build.sh release test >"$WORK/build.log" 2>&1; then
        mark build
        note "built, and its tests passed"
        return 0
    fi
    fail "the build or its tests failed: $WORK/build.log"
    BUILD_FAILED=1
    return 1
}

# ---------------------------------------------------------------- a level
level()
{
    local lv=$1
    read -r getname repo gguf jnsn mtpn steps dl disk ram <<<"$(model_of "$lv")"
    local jns=$MODELS/$jnsn converted=0
    say "Level $lv: ${gguf%.gguf}"
    machine
    # a machine sold with 16 GB shows less (the kernel and an integrated GPU
    # keep some): 85% of it is enough to count as one
    if [ "$MEM_TOTAL" -lt $((ram * 1073741824 * 85 / 100)) ]; then
        note "needs a machine with $ram GB of memory; this one has $(gib "$MEM_TOTAL") GiB: skipped"
        echo "$lv skipped (memory)" >> "$WORK/results"
        return 0
    fi
    if ! [ -f "$jns" ] && [ "$DISK_FREE" -lt "$disk" ]; then
        note "needs about $(gib "$disk") GiB of disk; $(gib "$DISK_FREE") free: skipped"
        echo "$lv skipped (disk)" >> "$WORK/results"
        return 0
    fi
    if ! [ -f "$jns" ]; then
        ask "Download $(gib "$dl") GiB and convert (about $(gib "$disk") GiB of disk while it works)?" || {
            echo "$lv declined" >> "$WORK/results"
            return 0
        }
        converted=1
    fi
    # janas-get downloads, checks the SHA-256 of every file against
    # MODELS.md, converts, and makes the prediction file beside the model;
    # what it says goes to the screen and, for the report, to a file
    [ -f "$jns" ] || echo "$jns" >> "$WORK/created"
    [ "$mtpn" = - ] || [ -f "$MODELS/$mtpn" ] || echo "$MODELS/$mtpn" >> "$WORK/created"
    local keep=()
    [ "$KEEP" = 1 ] && keep=(--keep)
    note "janas-get $getname: the GGUF goes once converted${keep:+ (kept: --keep)}"
    "$BIN/janas-get" "$getname" --dir "$MODELS" "${keep[@]}" 2>&1 | tee "$WORK/get-$lv.txt"
    local rc=${PIPESTATUS[0]} got=$WORK/get-$lv.txt
    local t
    for t in "gguf2jns" "jns_planes" "gguf2jns (assistant)" "hf2jns_mtp"; do
        local key=${t%% *}
        [ "$t" = "gguf2jns (assistant)" ] && key=assistant
        grep -qF "   $t: fingerprint as in MODELS.md" "$got" &&
            echo "$lv $key fingerprint ok" >> "$WORK/results"
        grep -qF "$t wrote" "$got" &&
            echo "$lv $key fingerprint DIFFERS" >> "$WORK/results"
    done
    if [ "$rc" != 0 ] || ! [ -f "$jns" ]; then
        fail "janas-get failed: $got"
        echo "$lv janas-get FAILED: $(grep -m1 'janas-get:' "$got")" >> "$WORK/results"
        return 1
    fi
    cat "$got" >> "$LOG"
    local mtp=()
    [ "$mtpn" = - ] || mtp=(--mtp "$MODELS/$mtpn")

    # what ran: Janas's own file, and where it came from
    local shown=${gguf%%-Q4_K_M.gguf} how="the SHA-256 of every file checked against MODELS.md by janas-get"
    shown=${shown#Qwen_}
    [ "$converted" = 1 ] || how="converted in an earlier run, checked again by janas-get"
    local tools="gguf2jns and jns_planes"
    [ "$steps" = one ] && tools="gguf2jns"
    local line="- Model ($lv): $shown, run as \`$jnsn\` - Janas's format (JNS), converted here by $tools from \`$gguf\` of \`$repo\`; $how"
    [ "$lv" = gemma ] && line="$line; with its assistant \`$mtpn\`, converted by gguf2jns"
    [ "$lv" = full ] && line="$line; with the MTP block \`$mtpn\`, made by hf2jns_mtp from the original checkpoint"
    grep -qF "$line" "$WORK/models" 2>/dev/null || echo "$line" >> "$WORK/models"

    # the answers: the same text as on every other machine
    note "checking the answers (temperature 0)"
    local text sum want
    text=$(printf '/experts 0\n%s\n' "$FP_PROMPT" |
        "$BIN/janas-chat" "$jns" "${mtp[@]}" --bits 6 --attention exact \
            --temp 0 --max 64 --system "" --no-preload --ctx 4096 2>>"$LOG")
    sum=$(printf '%s\n' "$text" | sha256sum | cut -c1-16) # as printed
    want=$(fp_expected "$lv")
    if [ "${want#@}" != "$want" ]; then
        echo "$lv answer fingerprint $sum (no reference yet)" >> "$WORK/results"
        printf '%s\n' "$text" > "$WORK/answer-$lv.txt"
    elif [ "$sum" = "$want" ]; then
        echo "$lv answer fingerprint ok ($sum)" >> "$WORK/results"
    else
        echo "$lv answer fingerprint DIFFERS ($sum, expected $want)" >> "$WORK/results"
        printf '%s\n' "$text" > "$WORK/answer-$lv.txt"
    fi

    # the speed
    if [ "$BENCH" = 1 ] && ! bench_gate; then
        echo "$lv speed not measured (the machine was busy)" >> "$WORK/results"
    elif [ "$BENCH" = 1 ]; then
        note "measuring the speed (janas-bench, 3 rounds): minutes on quick, longer on the others"
        (cd "$WORK" && "$BIN/janas-bench" "$jns" "${mtp[@]}" --rounds 3 \
            >>"$LOG" 2>&1)
        local busy="idle"
        [ "$BENCH_IDLE_BEFORE" -lt 85 ] && busy="BUSY: the speeds are lower than this machine can do"
        echo "$lv speed measured with the CPU $BENCH_IDLE_BEFORE% idle before (load $BENCH_LOAD_BEFORE over $THREADS threads; $busy)" >> "$WORK/results"
        local rep
        rep=$(ls -t "$WORK"/janas-bench-*.txt 2>/dev/null | head -1)
        [ -n "$rep" ] && mv "$rep" "$WORK/bench-$lv.txt"
    fi

    # the server, once, on the quick level
    [ "$lv" = quick ] && server_check "$jns"
    return 0
}

# How busy the machine is, in general terms only: the load average and the
# share of the CPU left idle over a few seconds, from /proc. Nothing about
# which programs run - that is nobody's business but the owner's.
idle_share()
{ # percent of CPU time idle over $1 seconds
    local a b
    a=$(awk '/^cpu / { print $5 + $6, $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9 }' /proc/stat)
    sleep "$1"
    b=$(awk '/^cpu / { print $5 + $6, $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9 }' /proc/stat)
    awk -v a="$a" -v b="$b" 'BEGIN { split(a, x, " "); split(b, y, " ");
        d = y[2] - x[2]; v = 0; if (d > 0) v = 100 * (y[1] - x[1]) / d;
        printf "%d", v }'
}

# Before measuring: a busy machine gives numbers that are not its own. Says
# so, and lets the user free it, measure all the same (the report will say
# the machine was busy), or skip. 0: measure; 1: skip.
bench_gate()
{
    local idle load
    while :; do
        note "checking that the machine is free to be measured (5 s)..."
        idle=$(idle_share 5)
        load=$(cut -d' ' -f1 /proc/loadavg)
        BENCH_IDLE_BEFORE=$idle
        BENCH_LOAD_BEFORE=$load
        [ "$idle" -ge 85 ] && return 0
        note "the CPU is only $idle% idle (load $load over $THREADS threads):"
        note "something else is running, and the speed measured now would be"
        note "lower than this machine can do. For true numbers, close what you"
        note "can (browser, builds, other models) before measuring."
        [ "$YES" = 1 ] && { note "measuring all the same (--yes); the report says so"; return 0; }
        local a
        read -r -p "   [w]ait and check again, [m]easure anyway, [s]kip the speed: " a </dev/tty ||
            return 1
        case "$a" in
        m | M) return 0 ;;
        s | S) return 1 ;;
        *) ;;
        esac
    done
}

server_check()
{
    note "trying janas-server on 127.0.0.1"
    local port=$((20000 + RANDOM % 20000)) pid ok=0
    "$BIN/janas-server" "$1" --port "$port" --ctx 4096 --no-store \
        2>>"$LOG" &
    pid=$!
    for _ in $(seq 60); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
        sleep 1
    done
    local r
    r=$(curl -sf "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' -d '{"messages":[{"role":"user",
        "content":"Say hello in one word."}],"temperature":0,"max_tokens":20}')
    echo "$r" | grep -q '"content"' && ok=$((ok + 1))
    r=$(curl -sf "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' -d '{"messages":[{"role":"user",
        "content":"What is the weather in Cagliari?"}],"temperature":0,
        "tools":[{"type":"function","function":{"name":"get_weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}},
        "required":["city"]}}}],"tool_choice":"required"}')
    echo "$r" | grep -q '"get_weather"' && ok=$((ok + 1))
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "quick server $ok/2 (chat, tool call)" >> "$WORK/results"
}

# ---------------------------------------------------------------- report
report()
{
    local f lv
    machine
    {
        echo "## Janas test report"
        echo
        echo "- Janas commit: \`$COMMIT\`"
        echo "- Levels: $(echo ${LEVELS:-none})"
        [ -f "$WORK/models" ] && cat "$WORK/models"
        echo "- CPU: $CPU ($THREADS threads;$ISA)"
        echo "- Memory: $(gib "$MEM_TOTAL") GiB"
        echo "- Disk: $DISK"
        echo "- System: $DISTRO, Linux $KERNEL"
        echo "- Compiler: $CC_VER"
        echo "- GPU seen by Vulkan: $GPU_VK; Janas built $GPU_BUILD"
        echo "- Power: $POWER"
        echo "- Load when it started: $LOAD_START over $THREADS threads (the speed lines below say how idle the CPU was before each measurement)"
        echo
        echo "### Results"
        echo
        if [ "${BUILD_FAILED:-0}" = 1 ]; then
            echo "**The build failed.** End of its log:"
            echo
            echo '```'
            tail -40 "$WORK/build.log" | sed "s|$PWD|.|g; s|$HOME|~|g"
            echo '```'
        else
            echo '```'
            cat "$WORK/results" 2>/dev/null
            echo '```'
        fi
        # only what this run measured: the files of earlier runs, of other
        # levels, stay in the directory but not in the report
        for f in "$WORK"/bench-*.txt; do
            [ -f "$f" ] || continue
            lv=$(basename "$f" .txt)
            lv=${lv#bench-}
            grep -q "^$lv speed measured" "$WORK/results" || continue
            echo
            echo "### $(basename "$f" .txt)"
            echo
            echo '```'
            sed "s|$MODELS|<models>|g; s|$PWD|.|g; s|$HOME|~|g" "$f"
            echo '```'
        done
        for f in "$WORK"/answer-*.txt; do
            [ -f "$f" ] || continue
            lv=$(basename "$f" .txt)
            lv=${lv#answer-}
            grep -q "^$lv answer fingerprint .*\(no reference\|DIFFERS\)" \
                "$WORK/results" || continue
            echo
            echo "### $(basename "$f" .txt) (no reference, or not the reference)"
            echo
            echo '```'
            cat "$f"
            echo '```'
        done
    } > "$REPORT"
}

urlencode()
{
    local LC_ALL=C s=$1 i c out=""
    for ((i = 0; i < ${#s}; i++)); do
        c=${s:i:1}
        case "$c" in
        [a-zA-Z0-9.~_-]) out+=$c ;;
        *) out+=$(printf '%%%02X' "'$c") ;;
        esac
    done
    printf '%s' "$out"
}

publish()
{
    say "The report ($REPORT)"
    echo
    cat "$REPORT"
    echo
    ask "Send it to $REPO as a new issue? (it is public)" always || {
        note "not sent; the report stays in $REPORT"
        return 0
    }
    local title="Test report: $CPU, ${LEVELS# }"
    if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
        gh issue create --repo "$REPO" --title "$title" --label test-report \
            --body-file "$REPORT" ||
            gh issue create --repo "$REPO" --title "$title" \
                --body-file "$REPORT"
        return 0
    fi
    note "gh is not here (or not logged in): open this link, paste the report"
    note "(it is in $REPORT) into the form, and press Submit:"
    echo
    local url enc
    url="https://github.com/$REPO/issues/new?template=04-test-report.yml&title=$(urlencode "$title")"
    enc=$(urlencode "$(cat "$REPORT")")
    [ "${#enc}" -lt 6000 ] && url="$url&report=$enc" # fits in a link: filled
    echo "$url"
}

cleanup()
{ # only what this script put there: the directory may hold other models
    [ "$KEEP" = 1 ] && return 0
    local f size=0 files=()
    while read -r f; do
        [ -f "$f" ] || continue
        files+=("$f")
        size=$((size + $(stat -c %s "$f")))
    done < <(sort -u "$WORK/created")
    [ "${#files[@]}" -gt 0 ] || return 0
    ask "Delete the models this script downloaded and converted ($(gib "$size") GiB)? Keep them to run again or to use Janas" || return 0
    for f in "${files[@]}"; do
        rm -f "$f" "$f.ok"
    done
    : > "$WORK/created"
}

# ---------------------------------------------------------------- main
preflight
LOAD_START=$LOAD
BUILD_FAILED=0
if build; then
    if [ -z "$LEVELS" ]; then
        say "Which levels? quick (4B), medium (35B-A3B), full (80B-A3B), gemma (Gemma-4-E4B)"
        read -r -p "   levels, space-separated [quick]: " LEVELS </dev/tty
        LEVELS=${LEVELS:-quick}
    fi
    for lv in $LEVELS; do
        case "$lv" in
        quick | medium | full | gemma) level "$lv" ;;
        *) fail "unknown level $lv" ;;
        esac
    done
fi
report
publish
cleanup
say "Done. Thank you for trying Janas."
