#!/bin/bash
# Prepares every shader of the game before play (tools/prepare_oot.py) under a memory cap. Safe to stop and to run
# again: what is done is skipped. Run it again after every translator rebuild, before the next game start: a new
# translator changes every cache key, and the game would otherwise translate everything at boot.
#   tools/prepare_oot.sh [build dir] [tools/prepare_oot.py options]     e.g. tools/prepare_oot.sh build/oot111 --stages cs
# Environment: XLATE (oot_xlate, default build/xlate/oot_xlate), ROMFS (directory with romfs.bin and index.json,
# default build/romfs-oot111), MEMORY (cap, default 14G), PYTHON (Python 3.14, default .venv-py314 or python3.14).
root=$(cd "$(dirname "$0")/.." && pwd)
build=build/oot111
case "$1" in -*|'') ;; *) build=$1; shift;; esac
cd "$root"
xlate=${XLATE:-$root/build/xlate/oot_xlate}
py=${PYTHON:-$root/.venv-py314/bin/python}
[ -x "$py" ] || py=python3.14
exec systemd-run --user --scope --quiet -p MemoryMax=${MEMORY:-14G} -p MemorySwapMax=2G -p OOMPolicy=continue \
    nice -n 10 "$py" tools/prepare_oot.py "$build" --romfs "${ROMFS:-build/romfs-oot111}" --xlate "$xlate" \
    --threads 6 --no-lock --slice 0 "$@"
