"""Check required runtime files without importing PyTorch."""
import hashlib
import json
from pathlib import Path

def check(root, verify_hash=True):
    root = Path(root)
    manifest = json.loads((root / "assets_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for name, info in manifest.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            errors.append(name)
        elif verify_hash and hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != info["sha256"]:
            errors.append(name + " (SHA256 mismatch)")
    return errors

if __name__ == "__main__":
    errors = check(Path(__file__).resolve().parents[1])
    if errors:
        raise SystemExit("Asset check failed: " + ", ".join(errors))
    print("All runtime assets verified.")
