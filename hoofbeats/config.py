# config: the frozen Stage 1 configuration (configs/stage1_v1.json), checked against what is on disk
from __future__ import annotations

import json
from pathlib import Path

DEFAULT = "configs/stage1_v1.json"


# the freeze was tagged stage1-final-dev; merging origin/main changed two frozen files without changing Stage 1 (an offline
# replay of the frozen dev evaluation was identical: 540 cached requests, 0 misses), so checks run against the merge
FROZEN_TAG = "stage1-final-dev"
TAG = "stage1-final-dev-merged"
# what the frozen configuration consists of: the readers, linker, fusion, scoring and their resources, the
# configuration and the PMC manifest (evaluation drivers such as __main__.py and llm_eval.py may change)
FROZEN_PATHS = ["configs/stage1_v1.json", "pmc_eval/manifest_v1.json", "hoofbeats/resources"] + [
    f"hoofbeats/{m}.py" for m in ("data", "numbers", "intake", "fusion", "grounding", "llm", "llm_extract", "linker",
                                   "llm_reader", "model", "pmc", "extract_eval")]


def frozen_problems(tag: str = TAG, paths=FROZEN_PATHS) -> list[str]:
    # empty if HEAD contains the tag and none of the frozen files differs from it or is uncommitted
    import subprocess
    git = lambda *x: subprocess.run(["git", *x], capture_output=True, text=True)
    if git("rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode:
        return [f"tag {tag} not found"]
    out = []
    if git("merge-base", "--is-ancestor", tag, "HEAD").returncode:
        out.append(f"HEAD does not contain {tag}")
    changed = git("diff", "--name-only", tag, "--", *paths).stdout.split()
    if changed:
        out.append(f"changed since {tag}: {', '.join(changed)}")
    dirty = git("status", "--porcelain", "--", *paths).stdout.strip()
    if dirty:
        out.append(f"uncommitted changes: {dirty}")
    return out


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
