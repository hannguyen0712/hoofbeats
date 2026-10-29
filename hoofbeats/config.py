# config: the frozen Stage 1 configuration (configs/stage1_v1.json), checked against what is on disk
from __future__ import annotations

import json
from pathlib import Path

DEFAULT = "configs/stage1_v1.json"


def load_stage1(path: str | Path = DEFAULT, ont_version: str | None = None) -> dict:
    # the configuration, with problems listed in cfg["problems"]: an HPO version or linker thresholds that differ
    # from the frozen ones mean results would not be the frozen configuration's
    cfg = json.loads(Path(path).read_text())
    problems = []
    lk = cfg["linker"]
    if ont_version and ont_version != lk["hpo_version"]:
        problems.append(f"HPO is {ont_version}, the frozen configuration used {lk['hpo_version']}")
    cal = Path(lk["calibration"])
    if cal.exists():
        c = json.loads(cal.read_text())
        diff = [k for k in ("hi", "margin", "lo", "k") if c.get(k) != lk[k]]
        if diff:
            problems.append(f"linker calibration {cal} differs from the frozen thresholds in {', '.join(diff)}")
    cfg["problems"] = problems
    return cfg
