"""Bounded submission diagnostics; does not change predictor configuration."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit


def redact(text):
    def clean(match):
        try:
            u = urlsplit(match.group(0))
            host = u.hostname or "redacted"
            if u.port: host += ":" + str(u.port)
            return u.scheme + "://" + host + u.path + ("?[redacted]" if u.query else "")
        except Exception:
            return "[redacted URL]"
    return re.sub(r"https?://[^\s\"'<>]+", clean, text)


def command(label, argv, timeout=20):
    print("[diag] " + label, flush=True)
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
        print(redact(p.stdout), flush=True)
        print("[diag] exit=" + str(p.returncode), flush=True)
        return p
    except subprocess.TimeoutExpired as e:
        data = e.stdout or b""
        if isinstance(data, bytes): data = data.decode(errors="replace")
        print(redact(data), flush=True)
        print("[diag] timed out after " + str(timeout) + "s", flush=True)
    except Exception as e:
        print("[diag] " + type(e).__name__ + ": " + redact(str(e)), flush=True)


def check_wheelhouse(wheels: Path):
    """Verify the shipped offline wheel set against its manifest.  Offline, stdlib only.

    A missing or corrupt wheel means the build cannot succeed, so unlike the rest of this
    script these findings are returned to the caller to fail the build on.
    """
    bad = []
    manifest = wheels / "MANIFEST.json"
    if not manifest.is_file():
        return ["wheelhouse manifest missing: " + str(manifest)]
    data = json.loads(manifest.read_text())
    total = 0
    for w in data["wheels"]:
        p = wheels / w["filename"]
        if not p.is_file():
            bad.append("missing " + w["filename"])
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != w["sha256"]:
            bad.append("sha256 mismatch " + w["filename"])
        total += p.stat().st_size
    print("[diag] wheelhouse %d wheels, %.1f MB, %d bad"
          % (len(data["wheels"]), total / 1e6, len(bad)), flush=True)
    for b in bad:
        print("[diag] wheelhouse " + b, flush=True)
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["build", "startup"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args(argv)
    print("[diag] phase=" + a.phase + " python=" + sys.version.replace("\n", " "), flush=True)
    for name in ["pip", "torch", "torchvision", "numpy", "matplotlib",
                 "ultralytics", "scikit-image", "opencv-python-headless"]:
        try: version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: version = "NOT_REGISTERED"
        print("[diag] package " + name + "=" + version, flush=True)
    if a.phase == "build":
        # No network probing any more: the build installs from the wheelhouse shipped in the
        # ZIP, so what can still go wrong is a missing/corrupt wheel, not a bad mirror.
        # Inspect only source-related pip settings, never dump all config or environment.
        p = subprocess.run([sys.executable, "-m", "pip", "config", "list"],
                           capture_output=True, text=True, timeout=10)
        for line in p.stdout.splitlines():
            key = line.partition("=")[0].lower()
            if any(x in key for x in ["index-url", "no-index", "find-links", "trusted-host", "proxy"]):
                print("[diag] pip config " + redact(line), flush=True)
        # Resolve the real install command with the network switched off, before doing it for
        # real.  If this fails the install would fail; the rest of the diag is advisory.
        command("offline resolution (dry-run, --no-index)",
                [sys.executable, "-m", "pip", "install", "--dry-run", "--no-index",
                 "--find-links", str(a.root / "submission_wheels"),
                 "--constraint", str(a.root / "submission_constraints.txt"),
                 "--break-system-packages",
                 "ultralytics==8.4.146", "scikit-image==0.26.0"], timeout=300)
        command("base torch import", [sys.executable, "-c",
                "import torch; print('torch',torch.__version__,'cuda',torch.version.cuda)"], timeout=20)
        bad = check_wheelhouse(a.root / "submission_wheels")
        if bad:
            print("[diag] wheelhouse unusable; failing the build here instead of at pip", flush=True)
            return 1
    else:
        if a.input:
            files = sorted(p for p in a.input.iterdir() if p.is_file() and p.suffix.lower() in
                           {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
            print("[diag] input images=" + str(len(files)), flush=True)
        if a.output:
            try:
                probe = a.output / ".submission_write_probe"
                if probe.exists(): raise RuntimeError("write probe already exists")
                probe.write_text("probe"); probe.unlink()
                print("[diag] output writable", flush=True)
            except Exception as e:
                print("[diag] output write failed " + redact(str(e)), flush=True)
        need = ["detect/train/yolo11m_v2/weights/best.pt", "detect/train/text_v1/weights/best.pt",
                "seg/unet_v1/best.pt", "ocr/recog_v1_anno1694_seed6_500ep/best.pt",
                "detect/ds_v2/vocab.json", "ocr/ds_text_v1/vocab.json"]
        for n in need:
            p = a.root / "pcbmodel/runs" / n
            print("[diag] resource " + n + " bytes=" + str(p.stat().st_size if p.is_file() else "MISSING"), flush=True)
        lib = Path(os.environ.get("PCB_IC_LIB", str(a.root / "10gt/10gt/BoxICLibrary_200cases.json")))
        print("[diag] IC library exists=" + str(lib.is_file()), flush=True)
        command("CUDA visibility", [sys.executable, "-c",
                "import torch; print('cuda build',torch.version.cuda,'available',torch.cuda.is_available()); "
                "print('visible GPUs',[(torch.cuda.get_device_name(i),torch.cuda.get_device_capability(i)) "
                "for i in range(torch.cuda.device_count())])"], timeout=20)
        print("[diag] actual device selection is recorded by deliver and decision_log", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
