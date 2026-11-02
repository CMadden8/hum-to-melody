"""Install the Windows runtime from pinned upstream packages."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "Runtime"
CACHE = ROOT / ".downloads"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, digest):
    CACHE.mkdir(exist_ok=True)
    target = CACHE / url.rsplit("/", 1)[-1]
    if target.exists() and sha256(target) == digest:
        return target
    print(f"Downloading {target.name} ...", flush=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        partial = Path(temp) / "download"
        subprocess.run(["curl.exe", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "3", "--connect-timeout", "20", "--max-time", "600",
                        "--output", str(partial), url], check=True)
        if sha256(partial) != digest:
            raise RuntimeError(f"Checksum mismatch for {target.name}. Run setup again.")
        partial.replace(target)
    return target


def install():
    if os.name != "nt" or sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        raise SystemExit("Setup requires 64-bit Python 3.11 on Windows.")
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                    "--only-binary=:all:", "-r", str(ROOT / "requirements.lock.txt")], check=True)
    RUNTIME.mkdir(exist_ok=True)
    install_audio_engine()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_melody.py")], check=True)
    print("Setup complete. Open Start.cmd to launch the app.", flush=True)


def install_audio_engine():
    pack = download(
        "https://github.com/vamp-plugins/vamp-plugin-pack/releases/download/v2.0/Vamp.Plugin.Pack.Installer.2.0.exe",
        "24d12812ffde04a20e7acbeb543860552d7aa05a729d9e9123b33c6e097c4d5f")
    # Pack 2.0 stores the signed x64 DLL as an uncompressed Qt resource.
    # Both the container and the extracted resource are pinned. No installer is executed.
    with pack.open("rb") as stream:
        stream.seek(16313170)
        dll = stream.read(452960)
    if hashlib.sha256(dll).hexdigest() != "32c045fb785f8545436ef8965c115882850cc207be750c064bb25ef230ad3673":
        raise RuntimeError("Could not verify the pYIN plugin inside the distribution.")
    plugin_dir = RUNTIME / "pyin"
    plugin_dir.mkdir(exist_ok=True)
    (plugin_dir / "pyin.dll").write_bytes(dll)
    archive = download(
        "https://github.com/FluidSynth/fluidsynth/releases/download/v2.4.6/fluidsynth-2.4.6-win10-x64.zip",
        "f8344fde598c2b5c43e2dda6446e7b27e403268fb1816f4b3515cb644e6dd7b0")
    with zipfile.ZipFile(archive) as distribution:
        distribution.extractall(RUNTIME / "fluidsynth")
    font = RUNTIME / "MuseScore_General.sf2"
    digest = "ee51d2c4b1525e70f19a45909c4fd7a2e26d91d115fa89dbf5a6bc413d8b9bf3"
    if not font.exists() or sha256(font) != digest:
        download("https://ftp-osl.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2",
                 digest).replace(font)


if __name__ == "__main__":
    install()
