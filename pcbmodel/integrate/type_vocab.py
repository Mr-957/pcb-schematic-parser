# -*- coding: utf-8 -*-
"""Official 2026-09-30 output vocabulary: collapse physical-device subtypes.

Detector and pin geometry keep their fine labels internally. Canonicalization is
only for the final submission JSON, after geometry and topology are complete.
"""
from __future__ import annotations

from pathlib import Path
import json

TYPE_CANONICAL = {
    "bjt_npn": "bjt", "bjt_pnp": "bjt",
    "mosfet_npn": "mosfet", "mosfet_pnp": "mosfet",
    "crystal_2pin": "crystal", "crystal_3pin": "crystal", "crystal_4pin": "crystal",
    "amp_3pin": "amp", "amp_5pin": "amp",
}


def canonicalize_target_types(target: dict) -> int:
    """Change only component type values in a submission target; return count."""
    changed = 0
    for comp in target.get("components", {}).values():
        old = comp.get("type")
        new = TYPE_CANONICAL.get(old, old)
        if new != old:
            comp["type"] = new
            changed += 1
    return changed


def canonicalize_submission_file(path: Path) -> int:
    """Normalize a completed target file without touching pins, nets, or labels."""
    target = json.loads(path.read_text(encoding="utf-8"))
    changed = canonicalize_target_types(target)
    if changed:
        path.write_text(json.dumps(target, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed