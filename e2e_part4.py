"""End-to-end check of the review screen's pipeline on the example cases.

    python e2e_part4.py

For each example (the 3 demo cases and the 3 held-out ones in examples/heldout/), run demo_part4.py the way the
review screen does, then report:
  - reading: which gold findings (examples/*intended_findings.json) were read, missed or added
  - ranking: where the true disease lands in the rare top 3
  - grounding: every shown sentence cites IDs in its evidence pack, every number in it appears in the cited
    evidence, and every highlighted span matches the source text
  - follow-up (Marfan): FBN1 pathogenic -> supported and up; VUS -> no change and a warning; negative -> down
Problems in grounding or follow-up make it exit with status 1; reading and ranking misses are reported as notes
(they measure Stage 1 and the ranker). It overwrites results/part4_demo.json (the screen's last result).
"""
import json, os, re, subprocess, sys
from pathlib import Path

PY = sys.executable
ENV = {**os.environ, "LLM_OFFLINE": "1", "VISION_OFFLINE": "0"}
CASES = [("examples/marfan_note.json", "examples/intended_findings.json", "marfan_note.json"),
         ("examples/tsc_imaging.json", "examples/intended_findings.json", "tsc_imaging.json"),
         ("examples/wilson_labs.json", "examples/intended_findings.json", "wilson_labs.json"),
         ("examples/heldout/dm1_note.json", "examples/heldout/intended_findings.json", "dm1_note.json"),
         ("examples/heldout/fabry_note_labs.json", "examples/heldout/intended_findings.json", "fabry_note_labs.json"),
         ("examples/heldout/hht_imaging.json", "examples/heldout/intended_findings.json", "hht_imaging.json")]
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def run(case, results=None):
    cmd = [PY, "demo_part4.py", case] + (["--results", results] if results else [])
    p = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if p.returncode:
        print("   FAILED:", p.stderr[-800:])
        return None
    return json.loads(Path("results/part4_demo.json").read_text())


def cites(s):
    m = re.search(r"\[([^\]]*)\]\s*\.?\s*$", s)
    return [c.strip() for c in m.group(1).split(",")] if m else []


problems, notes = [], []
for case, gold_file, key in CASES:
    print(f"\n=== {case}")
    d = run(case)
    if d is None:
        problems.append(f"{case}: pipeline failed")
        continue
    truth = json.loads(Path(case).read_text()).get("truth", [])
    gold = [g for v in json.loads(Path(gold_file).read_text())[key].values() for g in v]
    got = {(f["hpo_id"], f["status"]) for f in d["findings"]}
    hit = [g for g in gold if any((h, g["status"]) in got for h in [g["hpo_id"], *(g.get("alternatives") or [])])]
    miss = [g for g in gold if g not in hit]
    gold_ids = {(h, g["status"]) for g in gold for h in [g["hpo_id"], *(g.get("alternatives") or [])]}
    extra = [f for f in d["findings"] if (f["hpo_id"], f["status"]) not in gold_ids]
    print(f"   reading: {len(hit)} of {len(gold)} gold findings read ({100 * len(hit) / max(len(gold), 1):.0f}%); "
          f"{len(extra)} not in the gold list")
    for g in miss:
        print(f"     missed: {g['label']} ({g['status']}) - \"{g.get('quote') or ''}\"")
    for f in extra:
        print(f"     extra:  {f['name']} ({f['status']}) - \"{f['words']}\"")
    rare = d["lists"]["rare"]
    names = [(x["name"][:45], round(x["probability"], 3)) for x in rare]
    rank = next((n for n, x in enumerate(rare, 1) if x["id"] in truth), None)
    print(f"   ranking: truth {truth} -> {'rank ' + str(rank) if rank else 'NOT in the top 3'}; top 3 {names}")
    if not rank:
        notes.append(f"{case}: true disease not in the top 3")
    probs = [x["probability"] for x in rare]
    if probs != sorted(probs, reverse=True) or not all(0 < p < 1 for p in probs):
        problems.append(f"{case}: probabilities out of order or range {probs}")
    texts = d["source"]["texts"]
    n_sent = n_bad = 0
    for x in rare + d["lists"]["common"]:
        pack = {i["id"]: i for i in x["pack"]}
        for i in pack.values():
            for sp in i.get("spans", []):
                t = texts.get(sp["field"], "")
                if t[sp["start"]:sp["end"]] != sp["quote"]:
                    n_bad += 1
                    problems.append(f"{case}: span for {i['id']} does not match the source text")
        for sec, sents in x["explanation"].items():
            for s in sents:
                n_sent += 1
                ids = cites(s["text"])
                if not ids or any(c not in pack for c in ids):
                    n_bad += 1
                    problems.append(f"{case}: {x['name'][:30]}: sentence cites unknown IDs: {s['text'][:90]}")
                    continue
                ev = " ".join(pack[c]["text"] for c in ids)
                ev_nums = {float(n) for n in NUM.findall(ev)}
                body = re.sub(r"\[[^\]]*\]\s*\.?\s*$", "", s["text"])
                body = body.replace(x["name"], "").replace(x.get("runner_up_name", "") or "@@", "")
                for n in NUM.findall(body):
                    if float(n) not in ev_nums:
                        n_bad += 1
                        problems.append(f"{case}: {x['name'][:30]}: number {n} not in its evidence: {s['text'][:90]}")
        for fit in x["fits"]:
            if not all(c in pack for c in fit["ids"]):
                problems.append(f"{case}: a 'why it fits' finding points to a missing item")
    print(f"   explanations: {n_sent} sentences shown, {n_bad} problems in IDs, numbers or highlighted spans")
    print(f"   tests: {len(d.get('separate') or [])} to tell the top diseases apart; "
          f"{sum(1 for x in rare if (x.get('tests') or {}).get('gene_test'))} of 3 top diseases have a gene test")
    if case.endswith("marfan_note.json"):
        test = next((t["test"] for x in rare for t in [(x.get("tests") or {}).get("gene_test") or {}]
                     if "FBN1" in t.get("test", "")), None)
        print(f"   follow-up with {test!r}:")
        base = {x["id"]: x["probability"] for x in rare}
        for res, cls, expect in (("positive", "pathogenic", "supported, up"), ("positive", "uncertain", "no support"),
                                 ("negative", None, "down")):
            entry = {"test": test, "result": res, "details": ""}
            if res == "positive":
                entry.update(genes=["FBN1"], classification=cls)
            Path("/tmp/e2e_results.json").write_text(json.dumps([entry]))
            f = run(case, "/tmp/e2e_results.json")
            fu = f["followup"]
            after = {x["id"]: x["probability"] for x in f["lists"]["rare"]}
            m = "OMIM:154700"
            print(f"     {res} {cls or ''}: supported {fu.get('confirmed')}, Marfan {base.get(m, 0):.1%} -> "
                  f"{after.get(m, 0):.1%}, warnings {[r.get('warning') for r in fu.get('results', []) if r.get('warning')]}"
                  f"  (expected: {expect})")
            ok = {"supported, up": fu.get("confirmed") == [m] and after.get(m, 0) > base.get(m, 0),
                  "no support": not fu.get("confirmed") and abs(after.get(m, 0) - base.get(m, 0)) < 1e-6,
                  "down": not fu.get("confirmed") and after.get(m, 0) < base.get(m, 0)}[expect]
            if not ok:
                problems.append(f"{case}: follow-up {res} {cls} did not behave as expected")
        run(case)

print("\n=== notes (reading and ranking):", len(notes))
for n in notes:
    print(" -", n)
print("=== problems (grounding and follow-up):", len(problems))
for p in problems:
    print(" -", p)
sys.exit(1 if problems else 0)
