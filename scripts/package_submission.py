"""Create a flat-root competition ZIP from this repository and checked wheelhouse."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from check_assets import check

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=root / "dist/submission.zip")
    a = p.parse_args()
    errors = check(root)
    wheels = json.loads((root / "offline_wheels_manifest.json").read_text(encoding="utf-8"))
    for name, info in wheels.items():
        path = root / "submission_wheels" / name
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            errors.append("submission_wheels/" + name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            errors.append("submission_wheels/" + name + " (SHA256 mismatch)")
    if errors:
        raise SystemExit("Packaging refused: " + ", ".join(errors))
    paths = [root/n for n in ["Dockerfile", "run.sh", "agent.py", "assets_manifest.json"]]
    for dirname in ["pcbmodel", "pcbeval", "10gt", "scripts", "submission_wheels"]:
        paths += [f for f in (root/dirname).rglob("*") if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(paths)):
            z.write(path, path.relative_to(root).as_posix())
    print(str(a.output.resolve()))
