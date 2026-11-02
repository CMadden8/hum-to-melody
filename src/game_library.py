"""Keep additional, non-destructive copies in the game's sound library.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


def load_library_path(config: Path) -> Path:
    folder = Path(json.loads(config.read_text(encoding="utf-8"))["folder"])
    if not folder.is_absolute():
        raise ValueError("The Sounds library must be an absolute folder path")
    return folder


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SoundLibrary:
    FOLDERS = {"original": "Human voice", "instrument": "Instruments"}

    def __init__(self, root: Path):
        self.root = Path(root)

    def copy(self, source: Path, kind: str) -> Path:
        """Preserve identical files and use a numbered name for changed versions."""
        source = Path(source)
        folder = self.root / self.FOLDERS[kind]
        folder.mkdir(parents=True, exist_ok=True)
        source_hash = file_hash(source)
        number = 1
        while True:
            name = source.name if number == 1 else f"{source.stem} ({number}){source.suffix}"
            destination = folder / name
            if destination.exists():
                if source.resolve() == destination.resolve() or file_hash(destination) == source_hash:
                    return destination
                number += 1
                continue
            try:
                # Exclusive creation cannot overwrite another take, even if a
                # second process creates the name between the check and copy.
                target = destination.open("xb")
            except FileExistsError:
                continue
            try:
                with target, source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, target)
                shutil.copystat(source, destination)
                if file_hash(destination) != source_hash:
                    raise OSError("The Sounds library copy could not be verified")
                return destination
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
