"""Read-only access to the original RomFS image for tools/build_gamedata.py: romfs.bin plus the index.json that
tools/pack_romfs.py wrote beside it (path, offset, size per file). Never opens the image for writing."""
import hashlib, json
from pathlib import Path


class Romfs:
    def __init__(self, folder):
        self.folder = Path(folder)
        j = json.loads((self.folder / "index.json").read_text())
        self.sha256 = j.get("romfs_sha256")
        self.files = {f["path"]: (f["offset"], f["size"]) for f in j["files"]}
        self.f = open(self.folder / "romfs.bin", "rb")

    def read(self, path):
        off, size = self.files[path]
        self.f.seek(off)
        b = self.f.read(size)
        if len(b) != size: raise IOError(f"{path}: short read")
        return b

    def sha(self, path):
        return hashlib.sha256(self.read(path)).hexdigest()
