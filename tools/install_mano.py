"""Install the MANO hand models from the official download into model/mano (and the WSL copy).

MANO is licensed by the Max Planck Institute: register and accept the licence at
https://mano.is.tue.mpg.de, download "Models & Code" (mano_v1_2.zip), then run

    python tools/install_mano.py C:\\Users\\<you>\\Downloads\\mano_v1_2.zip

Copies MANO_LEFT.pkl and MANO_RIGHT.pkl to model/mano/ and, if the WSL distro exists, to
~/models/MANO/ there (where V2D-style configs look). Existing files are kept unless --force.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TARGET = PROJECT / "model" / "mano"
WANTED = ("MANO_LEFT.pkl", "MANO_RIGHT.pkl")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="mano_v1_2.zip, or a folder containing MANO_LEFT.pkl")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    p.add_argument("--no-wsl", action="store_true")
    a = p.parse_args()

    TARGET.mkdir(parents=True, exist_ok=True)
    found = {}
    if a.source.is_file() and zipfile.is_zipfile(a.source):
        with zipfile.ZipFile(a.source) as z:
            for name in z.namelist():
                base = name.rsplit("/", 1)[-1]
                if base in WANTED and "models/" in name.replace("\\", "/"):
                    found[base] = z.read(name)
    elif a.source.is_dir():
        for base in WANTED:
            hits = list(a.source.rglob(base))
            if hits:
                found[base] = hits[0].read_bytes()
    else:
        sys.exit(f"Not a zip file or folder: {a.source}")
    if "MANO_LEFT.pkl" not in found:
        sys.exit(f"MANO_LEFT.pkl not found in {a.source} (expected mano_v1_2/models/MANO_LEFT.pkl)")

    for base, data in found.items():
        dst = TARGET / base
        if dst.exists() and not a.force:
            print(f"keep   {dst} (exists; --force to overwrite)")
            continue
        dst.write_bytes(data)
        print(f"wrote  {dst}  sha256 {hashlib.sha256(data).hexdigest()[:16]}…")

    if not a.no_wsl and shutil.which("wsl.exe"):
        distro = os.environ.get("ZYEGO_WSL_DISTRO", "V2D-Ubuntu")
        src = "/mnt/" + str(TARGET)[0].lower() + str(TARGET)[2:].replace("\\", "/")
        flag = "" if a.force else "-n"
        cmd = f'mkdir -p "$HOME/models/MANO" && cp {flag} "{src}/MANO_LEFT.pkl" "{src}/MANO_RIGHT.pkl" "$HOME/models/MANO/" && ls -l "$HOME/models/MANO"'
        r = subprocess.run(["wsl.exe", "-d", distro, "--", "bash", "-lc", cmd],
                           env=dict(os.environ, MSYS_NO_PATHCONV="1"))
        print("WSL copy " + ("done" if r.returncode == 0 else f"failed (exit {r.returncode})"))


if __name__ == "__main__":
    main()
