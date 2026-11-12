"""Refuse files that look like game content, keys or generated game code.

Runs as the pre-commit hook (.githooks/pre-commit). With no arguments it checks every tracked file; with
--staged it checks the files staged for the next commit.

Standard library only, so it runs before any environment is set up.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 1024 * 1024

# Switch containers, executables and game data formats.
BLOCKED_SUFFIXES = {
    ".xci", ".xcz", ".nsp", ".nsz", ".nca", ".ncz", ".tik", ".cert",
    ".nso", ".nro", ".nrr", ".npdm", ".keys",
    ".bfres", ".bntx", ".bnsh", ".sarc", ".szs", ".zs", ".pack", ".byml",
    ".bgyml", ".msbt", ".bars", ".bwav", ".bfwav", ".bfstm", ".bfttf",
    ".bfotf", ".sharcb", ".webm",
}

# ExeFS modules carry no extension.
BLOCKED_NAMES = {"main", "sdk", "rtld", "subsdk0", "subsdk1", "main.npdm"}

# Written by the code generator into every file it emits.
GENERATED_MARKER = "OOT_RECOMP_" + "GENERATED_FROM_GAME_CODE"

# A prod.keys / title.keys line: name containing key, kek or seed, then hex.
KEY_LINE = re.compile(
    r"^\s*[A-Za-z0-9_]*(key|kek|seed)[A-Za-z0-9_]*\s*=\s*[0-9A-Fa-f]{32,}\s*$",
    re.MULTILINE,
)


def list_files(staged: bool) -> list[Path]:
    if staged:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    else:
        # Tracked files plus untracked ones that are not ignored: everything a commit could pick up.
        cmd = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return [Path(p) for p in out.decode("utf-8").split("\0") if p]


def problems_for(path: Path, allow_large: bool = False, *, staged: bool = False) -> list[str]:
    """allow_large: files over MAX_BYTES (toolchain binaries in a release folder) are checked for the generated marker
    only, not refused. staged reads the index blob, even when the working file is missing or different."""
    found = []
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        found.append(f"blocked file type {path.suffix}")
    if path.name.lower() in BLOCKED_NAMES:
        found.append("name of a Switch executable module")
    if staged:
        revision = f":{path.as_posix()}"
        size = int(subprocess.run(["git", "cat-file", "-s", revision], check=True, capture_output=True).stdout)
    else:
        if not path.is_file():
            return found
        size = path.stat().st_size
    if size > MAX_BYTES and not allow_large:
        found.append(f"{size // 1024} KiB, over the {MAX_BYTES // 1024} KiB limit")
        return found
    data = (subprocess.run(["git", "cat-file", "blob", revision], check=True, capture_output=True).stdout
            if staged else path.read_bytes())
    if GENERATED_MARKER.encode() in data:
        found.append("generated from game code")
    if size > MAX_BYTES:
        return found
    if KEY_LINE.search(data.decode("utf-8", errors="ignore")):
        found.append("contains a key in prod.keys format")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", action="store_true", help="check staged files only")
    args = parser.parse_args()

    failed = False
    for path in list_files(args.staged):
        for problem in problems_for(path, staged=args.staged):
            print(f"{path}: {problem}")
            failed = True
    if failed:
        print("\nGame content, keys and generated game code never go into the repository.")
        print("Game files, keys and generated game code never go into the repository.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
