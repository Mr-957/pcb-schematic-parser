"""Extract the checked platform wheelhouse from a release asset or submission ZIP."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

def prepare(archive, root):
    expected = json.loads((root / "offline_wheels_manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive) as z:
        members = z.namelist()
        for name, info in expected.items():
            matches = [m for m in members if m == "submission_wheels/" + name]
            if len(matches) != 1:
                raise ValueError("Missing/duplicate wheelhouse member: " + name)
            data = z.read(matches[0])
            if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise ValueError("Wheelhouse fingerprint mismatch: " + name)
            dest = root / "submission_wheels" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    print("Offline wheelhouse ready.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    a = p.parse_args()
    prepare(a.archive, Path(__file__).resolve().parents[1])
