#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# DekooNut setup. Without arguments it opens the window; `./setup.sh --help` shows the terminal form.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
for py in python3 python3.14 python3.13 python3.12; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
        exec "$py" -m setup "$@"
    fi
done
echo "DekooNut setup needs Python 3.10 or newer (sudo apt install python3)." >&2
exit 1
