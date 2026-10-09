#!/usr/bin/env python3
"""Portable inference entry point. No ground truth or training data is required."""
from __future__ import annotations
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("PCBM_DESIG_RECLASS", "1")
os.environ.setdefault("PCBM_NAME_FILL", "1")
os.environ.setdefault("PCB_IC_LIB", str(HERE / "10gt/10gt/BoxICLibrary_200cases.json"))
sys.path.insert(0, str(HERE))

def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" not in args and "-h" not in args:
        from scripts.check_assets import check
        missing = check(HERE, verify_hash=False)
        if missing:
            print("Missing runtime assets: " + ", ".join(missing), file=sys.stderr)
            return 2
        if not Path(os.environ["PCB_IC_LIB"]).is_file():
            print("PCB_IC_LIB does not exist: " + os.environ["PCB_IC_LIB"], file=sys.stderr)
            return 2
    from pcbmodel.integrate.deliver import main as deliver_main
    return deliver_main(args)

if __name__ == "__main__":
    sys.exit(main())
