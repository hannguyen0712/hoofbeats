# offline tests: python -m unittest tests
import csv
import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hoofbeats.__main__ import simulate
from hoofbeats.data import (DiseaseMatcher, Extractor, delabel, learn_common, load_medpix, load_mimic,
                         load_phenopackets, match_icd, orphadata_icd10, split_by_pmid)
from hoofbeats.llm import MedGemma, PolicyError, _json, _to_id, extract_findings, read_images, rerank
from hoofbeats.model import (AnnotationDB, Case, Finding, LRRanker, Ontology, Status, explain,
                          most_frequent_test, parse_frequency, suggest_tests, verify_claim)

OBO = """format-version: 1.2
data-version: hp/releases/test

[Term]
id: HP:0000001
name: All

[Term]
id: HP:0000118
name: Phenotypic abnormality
is_a: HP:0000001 ! All

[Term]
id: HP:0000707
name: Abnormality of the nervous system
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
alt_id: HP:0099999
synonym: "Epileptic seizure" EXACT []
synonym: "Fits" RELATED []
is_a: HP:0000707 ! Abnormality of the nervous system

[Term]
id: HP:0007359
name: Focal-onset seizure
is_a: HP:0001250 ! Seizure

[Term]
id: HP:0001263
name: Global developmental delay
is_a: HP:0000707 ! Abnormality of the nervous system

[Term]
id: HP:0000478
name: Abnormality of the eye
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0001083
name: Ectopia lentis
synonym: "Dislocated lens" EXACT []
is_a: HP:0000478 ! Abnormality of the eye

[Term]
id: HP:0001939
name: Abnormality of metabolism/homeostasis
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0003282
name: Decreased circulating alkaline phosphatase activity
synonym: "Low alkaline phosphatase" EXACT []
is_a: HP:0001939 ! Abnormality of metabolism/homeostasis

[Term]
id: HP:0001987
name: Hyperammonemia
is_a: HP:0001939 ! Abnormality of metabolism/homeostasis

[Term]
id: HP:0001945
name: Fever
is_a: HP:0001939 ! Abnormality of metabolism/homeostasis

[Term]
id: HP:0002151
name: Increased circulating lactate concentration
is_a: HP:0001939 ! Abnormality of metabolism/homeostasis

[Term]
id: HP:0001974
name: Increased total leukocyte count
is_a: HP:0001939 ! Abnormality of metabolism/homeostasis

[Term]
id: HP:0003259
name: Elevated circulating creatinine concentration
is_a: HP:0001939 ! Abnormality of metabolism/homeostasis

[Term]
id: HP:0003138
name: Increased blood urea nitrogen
is_a: HP:0001939 ! Abnormality of metabolism/homeostasis

[Term]
id: HP:0003155
name: Elevated circulating alkaline phosphatase concentration
is_a: HP:0001939 ! Abnormality of metabolism/homeostasis

[Term]
id: HP:0000098
name: Tall stature
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0001166
name: Arachnodactyly
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:9999999
name: obsolete Old seizure term
is_obsolete: true
replaced_by: HP:0001250

[Term]
id: HP:0040279
name: Frequency
is_a: HP:0000001 ! All

[Term]
id: HP:0040281
name: Very frequent
is_a: HP:0040279 ! Frequency

[Typedef]
id: part_of
name: part of
"""

HPOA_HEADER = "#description: test\n#version: 2026-01-01\n" \
    "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration\n"
HPOA_ROWS = [
    ("OMIM:1", "Disease Alpha", "", "HP:0000098", "PMID:100", "HP:0040281"),
    ("OMIM:1", "Disease Alpha", "", "HP:0001166", "PMID:100", "1/1"),
    ("OMIM:1", "Disease Alpha", "", "HP:0001083", "PMID:200", "HP:0040281"),
    ("OMIM:2", "Disease Beta", "", "HP:0007359", "PMID:300", "HP:0040281"),
    ("OMIM:2", "Disease Beta", "", "HP:0001263", "PMID:300", "2/2"),
    ("OMIM:3", "Hypophosphatasia Gamma", "", "HP:0003282", "PMID:400", "HP:0040280"),
    ("OMIM:3", "Hypophosphatasia Gamma", "", "HP:0001250", "PMID:400;OMIM:3", "HP:0040283"),
    ("OMIM:4", "Disease Delta", "", "HP:0001263", "PMID:600", "HP:0040281"),
    ("OMIM:4", "Disease Delta", "", "HP:0000098", "PMID:600", "HP:0040283"),
    ("ORPHA:10", "Hypophosphatasia", "", "HP:0003282", "PMID:500", "HP:0040281"),
    ("ORPHA:10", "Hypophosphatasia", "NOT", "HP:0001987", "PMID:500", ""),
    ("ORPHA:11", "Hyperammonemia syndrome X", "", "HP:0001987", "PMID:700", "HP:0040281"),
]
GENES = "ncbi_gene_id\tgene_symbol\tassociation_type\tdisease_id\tsource\n" \
        "NCBIGene:249\tALPL\tMENDELIAN\tOMIM:3\tx\nNCBIGene:249\tALPL\tMENDELIAN\tORPHA:10\tx\n" \
        "NCBIGene:2200\tFBN1\tMENDELIAN\tOMIM:1\tx\n"

PHENOPACKET = {
    "id": "PMID_100_Patient_1",
    "phenotypicFeatures": [
        {"type": {"id": "HP:0000098", "label": "Tall stature"}},
        {"type": {"id": "HP:0001166", "label": "Arachnodactyly"}},
        {"type": {"id": "HP:0001083", "label": "Ectopia lentis"}},
        {"type": {"id": "HP:0001263", "label": "Global developmental delay"}, "excluded": True},
    ],
    "interpretations": [{"diagnosis": {
        "disease": {"id": "OMIM:1", "label": "Disease Alpha"},
        "genomicInterpretations": [{"variantInterpretation": {
            "variationDescriptor": {"geneContext": {"symbol": "FBN1"}}}}]}}],
    "metaData": {"externalReferences": [{"id": "PMID:100"}]},
}

ORPHA_XML = """<?xml version="1.0"?>
<JDBOR><DisorderList count="1">
<Disorder id="1"><OrphaCode>10</OrphaCode><Name lang="en">Hypophosphatasia</Name>
<SynonymList count="1"><Synonym lang="en">HPP</Synonym></SynonymList>
<ExternalReferenceList count="3">
<ExternalReference id="a"><Source>ICD-10</Source><Reference>E83.3</Reference>
<DisorderMappingRelation id="x"><Name lang="en">E (Exact mapping: the two concepts are equivalent)</Name></DisorderMappingRelation></ExternalReference>
<ExternalReference id="b"><Source>ICD-10</Source><Reference>E83</Reference>
<DisorderMappingRelation id="y"><Name lang="en">NTBT (narrower term maps to a broader term)</Name></DisorderMappingRelation></ExternalReference>
<ExternalReference id="c"><Source>OMIM</Source><Reference>241500</Reference>
<DisorderMappingRelation id="z"><Name lang="en">E (Exact mapping: the two concepts are equivalent)</Name></DisorderMappingRelation></ExternalReference>
</ExternalReferenceList></Disorder>
</DisorderList></JDBOR>
"""

NOTE = """Name: ___ Unit No: ___
Chief Complaint:
bone pain

History of Present Illness:
Patient with known hypophosphatasia (ALPL variant) presents with low alkaline phosphatase.
No seizures. Denies fever.

Family History:
Mother had seizures.

Medications on Admission:
Hyperammonemia protocol drug

Discharge Diagnosis:
Hypophosphatasia, adult form

Discharge Instructions:
You were treated for hypophosphatasia.
"""




def build(root: Path):
    kb = root / "kb"
    kb.mkdir(parents=True, exist_ok=True)
    (kb / "hp.obo").write_text(OBO)
    rows = ["\t".join([d, n, q, h, ref, "PCS", "", f, "", "", "P", "x"]) for d, n, q, h, ref, f in HPOA_ROWS]
    (kb / "phenotype.hpoa").write_text(HPOA_HEADER + "\n".join(rows) + "\n")
    (kb / "genes_to_disease.txt").write_text(GENES)
    pp = root / "pps" / "FBN1"
    pp.mkdir(parents=True, exist_ok=True)
    (pp / "PMID_100_Patient_1.json").write_text(json.dumps(PHENOPACKET))
    (pp / "other.json").write_text("[1, 2]")
    (root / "p1.xml").write_text(ORPHA_XML)
    for name in ("mimic-iv-demo", "mimiciv/3.1"):
        hosp = root / name / "hosp"
        hosp.mkdir(parents=True, exist_ok=True)
        (hosp / "diagnoses_icd.csv").write_text(
            "subject_id,hadm_id,seq_num,icd_code,icd_version\n1,11,1,E8339,10\n1,11,2,I10,10\n2,22,1,4019,9\n3,33,9,E8339,10\n")
        (hosp / "d_icd_diagnoses.csv").write_text(
            "icd_code,icd_version,long_title\nE8339,10,Other disorders of phosphorus metabolism\n")
        (hosp / "d_labitems.csv").write_text(
            "itemid,label,fluid,category\n50863,Alkaline Phosphatase,Blood,Chem\n50866,Ammonia,Blood,Chem\n"
            "51006,Urea Nitrogen,Blood,Chem\n99999,Mystery,Blood,X\n")
        (hosp / "labevents.csv").write_text(
            "labevent_id,subject_id,hadm_id,itemid,value,valuenum,ref_range_lower,ref_range_upper,flag\n"
            "1,1,11,50863,20,20,35,105,abnormal\n3,1,11,50866,25,25,11,32,\n4,1,11,99999,9,9,0,1,abnormal\n"
            "5,2,22,51006,40,40,6,20,abnormal\n6,1,,50866,999,999,11,32,abnormal\n")
    notes = root / "note"
    notes.mkdir(exist_ok=True)
    with open(notes / "discharge.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["note_id", "subject_id", "hadm_id", "text"])
        w.writerow(["n1", 1, 11, NOTE])
    return kb


P = lambda *ids: [Finding(i, Status.PRESENT) for i in ids]
A = lambda *ids: [Finding(i, Status.ABSENT) for i in ids]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        kb = build(cls.root)
        cls.ont = Ontology(kb / "hp.obo")
        cls.db = AnnotationDB(kb / "phenotype.hpoa", cls.ont, ("OMIM",), kb / "genes_to_disease.txt")
        cls.r = LRRanker(cls.ont, cls.db)
        cls.orpha = LRRanker(cls.ont, AnnotationDB(kb / "phenotype.hpoa", cls.ont, ("ORPHA",), kb / "genes_to_disease.txt"))
        cls.ext = Extractor(cls.ont)

    # knowledge base
    def test_ontology(self):
        self.assertTrue({"HP:0001250", "HP:0000707"} <= self.ont.ancestors("HP:0007359"))
        self.assertEqual(self.ont.resolve("HP:0099999"), "HP:0001250")      # alt id
        self.assertEqual(self.ont.resolve("HP:9999999"), "HP:0001250")      # obsolete -> replaced_by
        self.assertIsNone(self.ont.resolve("HP:0040281"))                   # frequency term, not a phenotype
        self.assertNotIn("Fits", self.ont.terms["HP:0001250"].synonyms)     # RELATED synonyms dropped

    def test_annotations(self):
        self.assertAlmostEqual(parse_frequency("1/1"), 2 / 3)
        self.assertAlmostEqual(parse_frequency("HP:0040281"), 0.895)
        self.assertAlmostEqual(parse_frequency("25%"), 0.25)
        self.assertEqual(parse_frequency("junk", 0.4), 0.4)
        self.assertEqual(set(self.db.names), {"OMIM:1", "OMIM:2", "OMIM:3", "OMIM:4"})
        # an annotation also backed by another source survives exclusion of the paper
        self.assertEqual([a.hpo_id for a in self.db.annotations("OMIM:3", frozenset({"PMID:400"}))], ["HP:0001250"])

    # ranker
    def test_ranks_matching_disease_first(self):
        rk = self.r.score(Case("x", P("HP:0000098", "HP:0001166", "HP:0001083")))
        self.assertEqual(rk.rank_of(["OMIM:1"]), 1)
        self.assertAlmostEqual(rk.probs.sum(), 1)

    def test_specific_patient_term_matches_parent_annotation(self):
        con = self.r.contributions(Case("x", P("HP:0007359")), "OMIM:3")[0]
        self.assertEqual(con.kb_term, "HP:0001250")

    def test_contributions_equal_score(self):
        for c in (Case("a", P("HP:0000098", "HP:0007359") + A("HP:0001083", "HP:0001263")),
                  Case("b", P("HP:0003282", "HP:0001945") + A("HP:0001987"))):
            rk = self.r.score(c)
            for d in self.r.diseases:
                self.assertAlmostEqual(sum(x.log_lr for x in self.r.contributions(c, d)),
                                       rk.log_scores[self.r.d_index[d]], places=5)

    def test_leakage_control(self):
        c = Case("x", P("HP:0000098", "HP:0001166"), ["OMIM:1"], source_pmids=["PMID:100"])
        self.assertEqual(self.r.score(c).rank_of(c.truth), 1)
        clean = self.r.score(c, frozenset(c.source_pmids))
        self.assertGreater(clean.rank_of(c.truth), 1)
        self.assertAlmostEqual(clean.log_scores[self.r.d_index["OMIM:1"]],
                               sum(x.log_lr for x in self.r.contributions(c, "OMIM:1", frozenset({"PMID:100"}))))

    def test_absent_findings(self):
        rk = self.r.score(Case("x", A("HP:0001083")))
        self.assertLess(rk.log_scores[self.r.d_index["OMIM:1"]], 0)       # annotated: penalized
        self.assertEqual(rk.log_scores[self.r.d_index["OMIM:2"]], 0)      # not annotated: neutral
        self.r.absent_weight = 0.5
        half = self.r.score(Case("x", A("HP:0001083"))).log_scores
        self.r.absent_weight = 1.0
        np.testing.assert_allclose(half, rk.log_scores / 2)

    def test_family_uncertain_ignored_and_genes(self):
        c = Case("x", [Finding("HP:0001083", Status.FAMILY), Finding("HP:0000098", Status.UNCERTAIN)])
        self.assertTrue(np.allclose(self.r.score(c).log_scores, 0))
        self.r.use_genes = True
        self.assertEqual(self.r.score(Case("y", P("HP:0001250"), genes=["ALPL"])).rank_of(["OMIM:3"]), 1)
        self.r.use_genes = False

    def test_not_annotation(self):
        c = Case("x", P("HP:0001987"))
        self.assertAlmostEqual(self.orpha.score(c).log_scores[self.orpha.d_index["ORPHA:10"]], math.log(0.05))

    def test_ties_count_against_gold(self):
        rk = self.r.score(Case("x", []))
        self.assertEqual(rk.rank_of(["OMIM:1"]), len(self.r.diseases))

    # explanations and tests
    def test_claims_verify_and_tampering_fails(self):
        c = Case("x", P("HP:0000098", "HP:0007359") + A("HP:0001083"))
        claims = [cl for _, _, cls in explain(self.r, c, self.r.score(c), k=4) for cl in cls]
        self.assertTrue(claims and all(verify_claim(self.r, c, cl)[0] for cl in claims))
        cl = next(cl for cl in claims if cl.kb_term and cl.refs)
        self.assertFalse(verify_claim(self.r, c, cl, frozenset(cl.refs))[0])   # cites leaked paper
        cl.kb_freq = 0.123
        self.assertFalse(verify_claim(self.r, c, cl)[0])

    def test_suggest_tests(self):
        c = Case("x", P("HP:0001263"))   # fits OMIM:2 and OMIM:4
        sug = [t for t, _, _ in suggest_tests(self.r, self.r.score(c), c, top_k=4, n=10)]
        self.assertIn(sug[0], {"HP:0007359", "HP:0001250", "HP:0000098"})
        self.assertNotIn("HP:0001263", sug)
        self.assertNotIn("HP:0000707", sug)   # implied by a known finding
        self.assertIsNotNone(most_frequent_test(self.r, self.r.score(c), c))

    def test_simulate(self):
        c = Case("PMID_1_P1", P("HP:0001263", "HP:0007359") + A("HP:0000098"), ["OMIM:2"])
        for pol in ("eig", "frequent", "random"):
            ranks, answered = simulate(self.r, c, pol, 3, 0)
            self.assertEqual(len(ranks), 4)
            self.assertLessEqual(answered, 3)

    # data
    def test_phenopackets(self):
        cases = load_phenopackets(self.root / "pps")
        self.assertEqual(len(cases), 1)
        c = cases[0]
        self.assertEqual((c.truth, c.genes, c.source_pmids, c.absent), (["OMIM:1"], ["FBN1"], ["PMID:100"], ["HP:0001263"]))
        before, after = split_by_pmid([c, Case("y", [], source_pmids=["PMID:5"])], 50)
        self.assertEqual([x.case_id for x in after], [c.case_id])

    def test_extractor(self):
        st = lambda text, h: {f.hpo_id: f.status for f in self.ext.findings(text)}.get(h)
        self.assertEqual(st("No seizures were observed.", "HP:0001250"), Status.ABSENT)
        self.assertEqual(st("Exam shows dislocated lens.", "HP:0001083"), Status.PRESENT)
        self.assertEqual(st("Her mother had seizures.", "HP:0001250"), Status.FAMILY)
        self.assertEqual(st("Rule out ectopia lentis.", "HP:0001083"), Status.UNCERTAIN)
        self.assertEqual(st("No fever but has seizures.", "HP:0001250"), Status.PRESENT)
        self.assertEqual(st("Seizure was ruled out.", "HP:0001250"), Status.ABSENT)
        fs = {f.hpo_id: f.status for f in self.ext.findings(NOTE)}
        self.assertEqual((fs["HP:0003282"], fs["HP:0001945"]), (Status.PRESENT, Status.ABSENT))
        self.assertNotIn("HP:0001987", fs)    # medications section skipped

    def test_delabel(self):
        text, rep = delabel(NOTE, ["Hypophosphatasia", "Familial Mediterranean fever"], ["ALPL"], self.ext.vocab)
        self.assertNotIn("hypophosphatasia", text.lower())
        self.assertNotIn("ALPL", text)
        self.assertIn("fever", text.lower())          # phenotype word never masked
        self.assertIn("discharge diagnosis", rep["dropped_sections"])
        self.assertEqual(rep["residual"], [])

    def test_orphadata_and_mimic(self):
        icd, names = orphadata_icd10(self.root / "p1.xml")
        self.assertEqual(icd, {"E833": {"ORPHA:10"}})             # broad NTBT code excluded
        self.assertEqual(match_icd("E8339", icd), {"ORPHA:10"})
        demo = load_mimic(self.root / "mimic-iv-demo", icd)
        self.assertEqual([c.case_id for c in demo], ["mimic-11"])  # seq_num 9 not used as label
        c = demo[0]
        st = {f.hpo_id: f.status for f in c.findings}
        self.assertEqual((c.truth, c.meta["policy"]), (["ORPHA:10"], "open"))
        self.assertEqual((st["HP:0003282"], st["HP:0003155"], st["HP:0001987"]),
                         (Status.PRESENT, Status.ABSENT, Status.ABSENT))
        self.assertEqual(self.orpha.score(c).rank_of(c.truth), 1)
        full = load_mimic(self.root / "mimiciv/3.1", icd, names, self.orpha.db.disease_genes,
                          self.root / "note", self.ext)
        c = full[0]
        self.assertEqual(c.meta["policy"], "local_only")
        self.assertTrue({"Hypophosphatasia", "ALPL"} <= set(c.meta["delabel"]["masked"]))
        self.assertFalse(any("hypophosphatasia" in f.evidence.lower() for f in c.findings))
        self.assertEqual({f.hpo_id: f.status for f in c.findings}["HP:0001945"], Status.ABSENT)
        unl = load_mimic(self.root / "mimiciv/3.1", None, labeled_only=False)
        self.assertEqual({c.case_id for c in unl}, {"mimic-11", "mimic-22", "mimic-33"})


    # MedGemma (fake replies, no GPU)
    def test_medgemma_extraction_is_grounded(self):
        reply = json.dumps([
            {"finding": "low alkaline phosphatase", "status": "present", "quote": "presents with low alkaline phosphatase"},
            {"finding": "seizures", "status": "absent", "quote": "No seizures."},
            {"finding": "ectopia lentis", "status": "present", "quote": "lens dislocated on exam"},   # not in note
            {"finding": "bone pain", "status": "present", "quote": "bone pain"}])                    # no HPO match here
        llm = Fake(reply)
        found, rep = extract_findings(llm, self.ext, Case("n", []), NOTE)
        st = {f.hpo_id: f.status for f in found}
        self.assertEqual(st, {"HP:0003282": Status.PRESENT, "HP:0001250": Status.ABSENT})
        self.assertEqual((rep.proposed, rep.quote_not_in_note, rep.unmapped), (4, 1, ["bone pain"]))
        self.assertIn("NOTE", llm.prompt.upper())

    def test_medgemma_policy(self):
        remote = Fake("[]", "https://example.com/v1/chat/completions")
        with self.assertRaises(PolicyError):
            extract_findings(remote, self.ext, Case("m", [], meta={"policy": "local_only"}), NOTE)
        extract_findings(remote, self.ext, Case("p", [], meta={"policy": "open"}), NOTE)
        self.assertTrue(MedGemma().is_local and MedGemma(endpoint="http://127.0.0.1:8000/v1").is_local)
        self.assertFalse(MedGemma(endpoint="https://api.example.com/v1").is_local)

    def test_direct_diagnose_maps_answers(self):
        from hoofbeats.data import DiseaseMatcher
        from hoofbeats.llm import direct_diagnose
        d1 = sorted(self.db.names)[0]
        d2 = next(d for d in sorted(self.db.names) if "Hypophosphatasia" in self.db.names[d])
        reply = json.dumps({"diagnoses": [{"name": "made-up disorder", "omim": ""},
                                          {"name": "x", "omim": d1},
                                          {"name": self.db.names[d2], "omim": ""}]})
        slots, ok, _ = direct_diagnose(Fake(reply), self.r, Case("x", P("HP:0000098")), DiseaseMatcher(self.db.names))
        self.assertTrue(ok)
        self.assertEqual(slots[0], set())
        self.assertIn(d1, slots[1])
        self.assertIn(d2, slots[2])
        self.assertEqual(direct_diagnose(Fake("no json"), self.r, Case("x", []), DiseaseMatcher(self.db.names))[:2], ([], False))

    def test_citation_formats(self):
        from hoofbeats.llm import _cites
        self.assertEqual(_cites("A [C1][L2]."), ["C1", "L2"])
        self.assertEqual(_cites("A [C1, C2; L3]."), ["C1", "C2", "L3"])
        self.assertEqual(_cites("A [see above]."), [])

    def test_report_from_saved_runs(self):
        import pandas as pd
        from hoofbeats.report import build_report
        res, out = self.root / "res", self.root / "rep"
        (res / "runs").mkdir(parents=True)
        pd.DataFrame({"condition": ["present only"] * 4 + ["present + absent"] * 4, "case": list("abcd") * 2,
                      "truth": "OMIM:1", "rank": [1, 2, 11, None, 1, 1, 3, None], "evaluable": True}).to_csv(res / "benchmark.csv", index=False)
        pd.DataFrame({"case": list("abcd"), "truth": "OMIM:1", "ranker": [1, 2, 5, None], "rerank": [2, 1, 5, None],
                      "direct": [None, 1, None, None], "faithfulness": 0.5}).to_csv(res / "llm_rerank_m.csv", index=False)
        pd.DataFrame([{"time": "t", "model": "x/m", "task": "phenopacket", "cases": 4, "ranker_top1": 0.25, "rerank_top1": 0.25,
                       "direct_top1": 0.25, "rerank_top10": 0.75, "faithfulness": 0.5}]).to_csv(res / "llm_compare.csv", index=False)
        (res / "runs" / "r.json").write_text(json.dumps({"id": "r", "command": "benchmark", "started": "t", "metrics": {}}))
        first = build_report(res, out, latex=True)
        second = build_report(res, out)
        self.assertEqual((first.parent.name, first.name, second.name), ("rep-1", "rep-1.md", "rep-2.md"))
        self.assertIn("Key findings", first.read_text())
        d = first.parent
        self.assertTrue((d / "figures" / "llm_accuracy.png").exists() and (d / "tables" / "benchmark.tex").exists())
        self.assertTrue((d / "notes-1.md").exists())

    def test_reply_cache_and_dry_run(self):
        d = self.root / "cache"
        m = MedGemma("x/m", "http://127.0.0.1:1/v1", cache_dir=d)
        m._endpoint_chat = lambda u, s, i: (setattr(m, "requests", m.requests + 1), "reply")[1]
        self.assertEqual([m.chat("a"), m.chat("a"), m.chat("b")], ["reply"] * 3)
        self.assertEqual((m.requests, m.cache_hits), (2, 1))
        again = MedGemma("x/m", "http://127.0.0.1:1/v1", cache_dir=d, dry_run=True)
        self.assertEqual((again.chat("a"), again.chat("c")), ("reply", ""))
        self.assertEqual((again.cache_hits, again.planned, again.requests), (1, 1, 0))

    def test_offline_schema_key_and_threads(self):
        from hoofbeats.llm import chat_many, require_local
        d = self.root / "cache_offline"
        m = MedGemma("x/m", "http://127.0.0.1:1/v1", cache_dir=d)
        sent = []
        m._endpoint_chat = lambda u, s, i, schema=None: (sent.append(schema), f"r:{u}")[1]
        schema = {"type": "object"}
        self.assertEqual(chat_many(m, [{"user": str(i)} for i in range(20)] + [{"user": "0", "schema": schema}], workers=4),
                         [f"r:{i}" for i in range(20)] + ["r:0"])
        self.assertEqual((m.n_calls, m.cache_hits, sent.count(schema)), (21, 0, 1))   # the schema is part of the key
        self.assertEqual(len((d / "m.jsonl").read_text().splitlines()), 21)          # one cache file per model
        off = MedGemma("x/m", "http://127.0.0.1:1/v1", cache_dir=d, offline=True)
        off._endpoint_chat = lambda *a, **k: self.fail("offline mode sent a request")
        self.assertEqual((off.chat("3"), off.chat("0", schema=schema), off.chat("new")), ("r:3", "r:0", ""))
        self.assertEqual((off.cache_hits, off.missed, off.usage()["offline_misses"]), (2, 1, 1))
        require_local(m)
        require_local(MedGemma())
        with self.assertRaises(PolicyError):
            require_local(MedGemma("x/m", "https://api.example.com/v1"))

    def test_learned_weights(self):
        import numpy as np
        from hoofbeats.train import BOUNDS, FEATURES, candidates, default_weights, fit_weights
        c = Case("x", P("HP:0000098", "HP:0001263") + [Finding("HP:0001250", Status.ABSENT)], ["OMIM:1"])
        ds, X, pr = candidates(self.r, c, k=5, add_truth=True)
        dw = default_weights(self.r)
        w = np.array([dw[f] for f in FEATURES])
        base = self.r.score(c)
        for i, d in enumerate(ds):         # default weights reproduce the current score
            self.assertAlmostEqual(X[i] @ w + pr[i], base.log_scores[self.r.d_index[d]], places=6)
        fitted = fit_weights([X] * 3, [pr] * 3, [ds.index("OMIM:1")] * 3, l2=0.01, init=dw)
        for f, v in zip(FEATURES, fitted):  # signs never flip
            self.assertTrue(BOUNDS[f][0] - 1e-9 <= v <= BOUNDS[f][1] + 1e-9, f)

    def test_onset_and_age(self):
        from hoofbeats.data import iso_years, patient_context
        from hoofbeats.model import patient_onset_classes
        self.assertAlmostEqual(iso_years("P2Y6M"), 2.5)
        self.assertIsNone(iso_years("soon"))
        ctx = patient_context({"subject": {"sex": "FEMALE", "timeAtLastEncounter": {"age": {"iso8601duration": "P6M"}}},
                               "phenotypicFeatures": [{"onset": {"age": {"iso8601duration": "P2M"}}}]})
        self.assertEqual((ctx["sex"], ctx["age_years"]), ("female", 0.5))
        self.assertEqual(patient_onset_classes(self.ont, ctx), {3})                       # infantile
        self.assertEqual(patient_onset_classes(self.ont, {"age_years": 3}), {0, 1, 2, 3, 4})  # onset at or before 3 y
        self.assertEqual(patient_onset_classes(self.ont, {}), set())

    def test_absent_min_freq(self):
        c = Case("x", P("HP:0000098") + [Finding("HP:0001250", Status.ABSENT)], ["OMIM:1"])
        old = self.r.score(c).log_scores
        self.r.absent_min_freq = 1.01          # nothing qualifies: absent findings change nothing
        try:
            from dataclasses import replace
            same = self.r.score(replace(c, findings=c.findings[:1])).log_scores
            self.assertTrue(np.allclose(self.r.score(c).log_scores, same))
            for d in self.r.diseases[:5]:     # score() and contributions() agree under the rule
                self.assertAlmostEqual(sum(x.log_lr for x in self.r.contributions(c, d)),
                                       self.r.score(c).log_scores[self.r.d_index[d]], places=5)
        finally:
            self.r.absent_min_freq = 0.0
        self.assertEqual(old.shape, same.shape)

    def test_quota_errors(self):
        from hoofbeats.llm import quota_info
        body = json.dumps([{"error": {"code": 429, "message": "quota", "details": [
            {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]},
            {"retryDelay": "41s"}]}}]).encode()
        info = quota_info(body)
        self.assertEqual((info["quotas"][0][1], info["retry"]), ("20", 41.0))
        self.assertEqual(quota_info(b"not json")["message"], "not json")

    def test_llm_options(self):
        m = MedGemma("models/Gemini-X.Flash", "https://example.com/v1/chat/completions", api_key="k", vision=False)
        self.assertEqual(m.slug, "gemini-x.flash")
        self.assertFalse(m.sees_images or m.is_local)
        with self.assertRaises(ValueError):
            m.chat("hi", images=("a.png",))

    def test_medgemma_rerank_is_verified(self):
        c = Case("x", P("HP:0000098", "HP:0001263"))
        rk = self.r.score(c)
        shortlist = [d for d, _, _ in rk.top(4)]
        cls = {d: cl for d, _, cl in explain(self.r, c, rk, 4)}
        last, first = shortlist[-1], shortlist[0]
        n_first = len(cls[first])                      # claim ids are numbered disease by disease
        own = f"C{n_first + sum(len(cls[d]) for d in shortlist[1:-1]) + 1}"
        good = cls[last][0].text.split(".")[0].split(";")[0].replace("'", "")
        reply = "```json\n" + json.dumps({
            "ranking": ["OMIM:999", last],
            "explanations": {last: f"{good} [{own}]. Same point again [C1]. No citation here.",
                             "OMIM:999": "Invented [C1]."}}) + "\n```"
        res = rerank(Fake(reply), self.r, c, rk, k=4)
        self.assertEqual(res.ranking[0], last)
        self.assertEqual(sorted(res.ranking), sorted(shortlist))       # nothing added or lost
        self.assertEqual(len(res.explanations[last]), 1)
        self.assertEqual(len(res.dropped[last]), 2)
        self.assertAlmostEqual(res.faithfulness, 1 / 3)
        self.assertEqual(_json("noise {\"a\": 1} tail", "{"), {"a": 1})

    def test_messy_llm_replies(self):
        self.assertEqual(_json('```json\n{"a": [1, 2,], // note\n "b": "x"}\n```', "{"), {"a": [1, 2], "b": "x"})
        self.assertEqual(_json('{"a": [1], "b": {"c": "cut off mid sente', "{"), {"a": [1], "b": {"c": "cut off mid sente"}})
        self.assertEqual(_json('{"u": "http://x.org"}', "{"), {"u": "http://x.org"})
        self.assertIsNone(_json("no json", "{"))
        names = {"OMIM:1": "Disease Alpha", "OMIM:2": "Disease Beta"}
        self.assertEqual([_to_id(x, ["OMIM:1", "OMIM:2"], names) for x in ("OMIM:2", "2", "omim 1", "disease beta", "OMIM:9", "Type 2 thing")],
                         ["OMIM:2", "OMIM:2", "OMIM:1", "OMIM:2", None, None])
        c = Case("x", P("HP:0000098", "HP:0001263"))
        rk = self.r.score(c)
        short = [d for d, _, _ in rk.top(4)]
        res = rerank(Fake('Here: {"ranking": ["' + short[-1].split(":")[1] + '"]'), self.r, c, rk, k=4)
        self.assertEqual(res.ranking[0], short[-1])          # bare number and cut-off JSON still understood
        self.assertTrue(res.parsed)
        self.assertFalse(rerank(Fake("I cannot help with that."), self.r, c, rk, k=4).parsed)

    def test_mimic_with_llm_extractor(self):
        icd, names = orphadata_icd10(self.root / "p1.xml")
        seen = []
        fn = lambda text, policy: seen.append((policy, "[MASKED]" in text)) or [Finding("HP:0001083", Status.PRESENT)]
        c = load_mimic(self.root / "mimiciv/3.1", icd, names, {}, self.root / "note", self.ext, extract_fn=fn)[0]
        self.assertEqual(seen, [("local_only", True)])               # gets de-labelled text + policy
        self.assertIn("HP:0001083", {f.hpo_id for f in c.findings})


def write_mixed_mimic(root: Path) -> Path:
    # 20 patients: sepsis, CKD, and rare hypophosphatasia
    hosp = root / "mixed-demo" / "hosp"
    hosp.mkdir(parents=True, exist_ok=True)
    dx, labs = ["subject_id,hadm_id,seq_num,icd_code,icd_version"], \
        ["labevent_id,subject_id,hadm_id,itemid,value,valuenum,ref_range_lower,ref_range_upper,flag"]
    profile = {0: ("A419", [(1, 4.0, 0.5, 2), (2, 18, 4, 11), (3, 0.9, 0.5, 1.2)]),
               1: ("N184", [(3, 3.0, 0.5, 1.2), (4, 40, 6, 20), (1, 1.0, 0.5, 2)]),
               2: ("E8339", [(5, 20, 35, 105), (3, 0.9, 0.5, 1.2)])}
    for s_ in range(1, 21):
        code, vals = profile[s_ % 3]
        dx.append(f"{s_},{100 + s_},1,{code},10")
        labs += [f"{s_}{i},{s_},{100 + s_},{it},{v},{v},{lo},{hi}," for i, (it, v, lo, hi) in enumerate(vals)]
    (hosp / "diagnoses_icd.csv").write_text("\n".join(dx) + "\n")
    (hosp / "labevents.csv").write_text("\n".join(labs) + "\n")
    (hosp / "d_labitems.csv").write_text("itemid,label,fluid,category\n1,Lactate,Blood,x\n2,White Blood Cells,Blood,x\n"
                                        "3,Creatinine,Blood,x\n4,Urea Nitrogen,Blood,x\n5,Alkaline Phosphatase,Blood,x\n")
    (hosp / "d_icd_diagnoses.csv").write_text("icd_code,icd_version,long_title\nA419,10,\"Sepsis, unspecified organism\"\n"
                                             "N184,10,\"Chronic kidney disease, stage 4\"\nE8339,10,Other disorders of phosphorus metabolism\n")
    return root / "mixed-demo"


class TestRareAndCommon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp())
        kb = build(root)
        cls.mimic = write_mixed_mimic(root)
        cls.ont = Ontology(kb / "hp.obo")
        cls.prof = learn_common(cls.mimic, {"E833": {"ORPHA:10"}}, min_adm=2, test_frac=0.3)
        db = AnnotationDB(kb / "phenotype.hpoa", cls.ont, ("ORPHA",))
        db.add_common(cls.prof)
        cls.r = LRRanker(cls.ont, db, rare_mass=0.04)

    def test_learned_profiles(self):
        p = self.prof
        self.assertEqual(set(p["diseases"]), {"ICD10:A41", "ICD10:N18"})        # rare code not learned as common
        self.assertEqual(p["diseases"]["ICD10:A41"]["name"], "Sepsis, unspecified organism")
        self.assertTrue(p["test_subjects"])
        train = 20 - len(p["test_subjects"])
        self.assertEqual(p["n_admissions"], train)
        sepsis = p["diseases"]["ICD10:A41"]
        pos, meas = sepsis["counts"]["HP:0002151"]                                # high lactate
        self.assertEqual((pos, meas), (sepsis["n_adm"], sepsis["n_adm"]))
        self.assertEqual(p["diseases"]["ICD10:N18"]["counts"]["HP:0002151"][0], 0)
        self.assertEqual(p["policy"], "open")
        self.assertGreater(p["n_rare_admissions"], 0)

    def test_one_list_with_priors(self):
        sepsis = Case("s", [Finding("HP:0002151", Status.PRESENT), Finding("HP:0001974", Status.PRESENT)])
        ckd = Case("c", [Finding("HP:0003259", Status.PRESENT), Finding("HP:0003138", Status.PRESENT)])
        hpp = Case("h", [Finding("HP:0003282", Status.PRESENT)])
        self.assertEqual(self.r.score(sepsis).top(1)[0][0], "ICD10:A41")
        self.assertEqual(self.r.score(ckd).top(1)[0][0], "ICD10:N18")
        self.r.prior_weight = 0.0
        evidence = self.r.score(hpp).log_scores.copy()
        self.assertEqual(self.r.score(hpp).top(1)[0][0], "ORPHA:10")              # evidence alone: rare wins
        self.r.prior_weight = 1.0
        both = self.r.score(hpp).log_scores
        np.testing.assert_allclose(both - evidence, self.r.log_prior)
        i, j = self.r.d_index["ORPHA:10"], self.r.d_index["ICD10:A41"]
        self.assertAlmostEqual(self.r.log_prior[i], math.log(0.04 / 2))
        self.assertGreater(self.r.log_prior[j], self.r.log_prior[i])              # common diseases are a priori likelier

    def test_out_of_scope_findings_do_not_count_for_common(self):
        c = Case("x", [Finding("HP:0001083", Status.PRESENT), Finding("HP:0002151", Status.PRESENT)])
        con = {x.hpo_id: x.log_lr for x in self.r.contributions(c, "ICD10:A41")}
        self.assertEqual(con["HP:0001083"], 0.0)                                  # eye finding: labs can't see it
        self.assertGreater(con["HP:0002151"], 0)
        rk = self.r.score(c)
        for d in ("ICD10:A41", "ICD10:N18", "ORPHA:10"):
            self.assertAlmostEqual(sum(x.log_lr for x in self.r.contributions(c, d)) + self.r.log_prior[self.r.d_index[d]],
                                   rk.log_scores[self.r.d_index[d]], places=5)

    def test_mimic_eval_uses_held_out_patients_and_mixed_truth(self):
        held = set(self.prof["test_subjects"])
        cases = load_mimic(self.mimic, {"E833": {"ORPHA:10"}}, common=self.prof, subjects=held)
        self.assertEqual({c.meta["hadm_id"] - 100 for c in cases}, held)
        truths = {t for c in cases for t in c.truth}
        self.assertTrue(truths <= {"ICD10:A41", "ICD10:N18", "ORPHA:10"})

    def test_credentialed_profiles_lock_llm_to_local(self):
        db = self.r.db
        db.policy = "local_only"
        c = Case("x", [Finding("HP:0002151", Status.PRESENT)])
        with self.assertRaises(PolicyError):
            rerank(Fake("{}", "https://example.com/v1"), self.r, c, self.r.score(c), k=3)
        rerank(Fake("{}"), self.r, c, self.r.score(c), k=3)
        db.policy = "open"


class Fake:
    def __init__(self, reply, endpoint=None):
        self.reply, self.endpoint = reply, endpoint
        self.is_local = endpoint is None
        self.calls, self.images = 0, []

    def chat(self, user, system="", images=()):
        self.prompt, self.calls = user, self.calls + 1
        self.images += list(images)
        return self.reply


class TestLLMExtract(unittest.TestCase):
    # LLM finding extraction: chunks, quote check with offsets, reply parsing (fake model, no GPU)
    class ChunkFake:
        def __init__(self, answer, endpoint=None):
            self.answer, self.endpoint, self.is_local, self.sent = answer, endpoint, endpoint is None, []

        def chat(self, user, system="", images=(), schema=None):
            self.sent.append((user, schema))
            return self.answer(user)

    def test_chunks_keep_offsets(self):
        from hoofbeats.llm_extract import chunks
        text = "First paragraph here.\n\nSecond one. It has two sentences.\n\n" + "Long sentence number %d. " * 40 % tuple(range(40))
        spans = chunks(text, max_chars=200)
        self.assertEqual(spans[0], (0, text.index("sentences.") + len("sentences.")))      # short paragraphs joined
        self.assertTrue(all(b - a <= 200 for a, b in spans))
        self.assertEqual("".join(text[a:b] for a, b in spans).replace(" ", "").replace("\n", ""),
                         text.replace(" ", "").replace("\n", ""))                          # nothing lost or repeated
        self.assertEqual(chunks("   "), [])

    def test_locate_quote(self):
        from hoofbeats.llm_extract import locate
        text = "Exam: the testes were  not palpable\nin the scrotum. He had “marked” hypotonia–severe."
        s, e, how = locate(text, 0, len(text), "testes were not palpable in the scrotum.")
        self.assertEqual((text[s:e], how), ("testes were  not palpable\nin the scrotum.", "normalised"))
        s, e, _ = locate(text, 0, len(text), "the testes were not palpable in the scrotum, ")   # added comma dropped
        self.assertEqual(text[s:e], "the testes were  not palpable\nin the scrotum")
        s, e, how = locate(text, 0, len(text), 'He had "marked" Hypotonia-severe')
        self.assertEqual(text[s:e], "He had “marked” hypotonia–severe")
        self.assertEqual(locate(text, 0, len(text), "hypotonia")[2], "exact")
        self.assertIsNone(locate(text, 0, 20, "hypotonia"))                                # only inside the chunk
        self.assertIsNone(locate(text, 0, len(text), "ataxia"))

    def test_extract_candidates(self):
        from hoofbeats.llm_extract import SCHEMA, extract_candidates
        text = ("The boy had hypotonia. His testes were not palpable in the scrotum.\n\n"
                "Blood pressure was normal. His father had  seizures.")

        def answer(prompt):
            if "hypotonia" in prompt.rsplit('"""', 2)[-2]:            # the chunk, not the prompt's own example
                return json.dumps({"findings": [
                    {"phrase": "hypotonia", "status": "present", "quote": "hypotonia"},
                    {"phrase": "cryptorchidism", "status": "present", "quote": "testes were not palpable in the scrotum"},
                    {"phrase": "ataxia", "status": "present", "quote": "unsteady gait"},          # not in the text
                    {"phrase": "hypotonia", "status": "present", "quote": "hypotonia"},           # repeated
                    {"phrase": "x", "status": "maybe", "quote": "boy"}]})                         # bad status
            return 'Sure! ```json\n[{"phrase": "hypertension", "status": "absent", "quote": "Blood pressure was normal."},' \
                   '{"phrase": "seizures", "status": "family", "quote": "father had seizures"}]\n```'
        llm = self.ChunkFake(answer)
        found, rep = extract_candidates(llm, text, max_chars=80, workers=1)
        self.assertEqual(rep.chunks, 2)
        self.assertTrue(all(s is SCHEMA for _, s in llm.sent))
        self.assertEqual([(c.phrase, c.status, c.match) for c in found],
                         [("hypotonia", Status.PRESENT, "exact"), ("cryptorchidism", Status.PRESENT, "exact"),
                          ("hypertension", Status.ABSENT, "exact"), ("seizures", Status.FAMILY, "normalised")])
        self.assertTrue(all(text[c.start:c.end] == c.quote for c in found))
        self.assertEqual(found[3].quote, "father had  seizures")
        self.assertEqual((rep.proposed, rep.quote_not_found, rep.duplicates, rep.bad_item, rep.kept), (7, 1, 1, 1, 4))
        self.assertEqual(rep.not_found_examples, [{"phrase": "ataxia", "quote": "unsteady gait"}])
        v1, v2 = self.ChunkFake(lambda p: "[]"), self.ChunkFake(lambda p: "[]")
        extract_candidates(v1, text, workers=1, prompt_version=1)
        extract_candidates(v2, text, workers=1)                                            # v2 is the default
        self.assertNotIn("Example answer", v1.sent[0][0])
        self.assertIn("Example answer", v2.sent[0][0])
        self.assertTrue(v2.sent[0][0].rstrip().endswith(text.strip() + '\n"""'))           # the case text, once, at the end
        _, rep = extract_candidates(self.ChunkFake(lambda p: "I cannot help with that."), text, workers=1)
        self.assertEqual((rep.unreadable, rep.kept), (1, 0))
        with self.assertRaises(PolicyError):
            extract_candidates(self.ChunkFake(answer, "https://api.example.com/v1"), text)


class TrigramEmbedder:
    # stands in for SapBERT in tests: hashed character trigrams, L2-normalised
    name = "trigram"

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        import numpy as np
        self.calls += 1
        out = np.zeros((len(texts), 512), np.float32)
        for i, t in enumerate(texts):
            s = f"  {t.lower()} "
            for j in range(len(s) - 2):
                out[i, int(hashlib.md5(s[j:j + 3].encode()).hexdigest(), 16) % 512] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class TestLinker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        kb = build(cls.root)
        cls.ont = Ontology(kb / "hp.obo")
        cls.ext = Extractor(cls.ont)

    def index(self, cache=None):
        from hoofbeats.linker import HpoIndex
        return HpoIndex(self.ont, TrigramEmbedder(), cache or Path(tempfile.mkdtemp()))

    def test_index_strings_and_cache(self):
        from hoofbeats.linker import term_strings
        rows = term_strings(self.ont)
        ids = {t for t, _ in rows}
        self.assertIn(("HP:0001250", "Fits"), rows)                       # RELATED synonyms are searchable too
        self.assertFalse(ids & {"HP:9999999", "HP:0040279", "HP:0000118"})  # obsolete, not a phenotype, the root
        cache = Path(tempfile.mkdtemp())
        first = self.index(cache)
        again = self.index(cache)
        self.assertEqual((first.embedder.calls, again.embedder.calls), (1, 0))   # second load from the .npz
        res = again.search(again.embedder.encode(["epileptic seizures"]), k=3)[0]
        self.assertEqual((res[0][0], res[0][2]), ("HP:0001250", "Epileptic seizure"))
        self.assertEqual(len({t for t, _, _ in res}), 3)                  # terms, not strings

    def test_dictionary_needs_the_whole_phrase(self):
        from hoofbeats.linker import Linker
        lk = Linker(self.ont, self.ext)
        self.assertEqual([(x.hpo_id, x.method, x.grounding) for x in lk.dictionary("recurrent epileptic seizures")],
                         [("HP:0001250", "dict:synonym:exact", "strong")])
        self.assertEqual(lk.dictionary("seizures of the left foot"), [])  # "foot" is not accounted for
        self.assertEqual({x.hpo_id for x in lk.dictionary("fever and arachnodactyly")}, {"HP:0001945", "HP:0001166"})
        self.assertEqual(lk.link_many([("lens displaced upward", "")])[0][0].method, "unlinked")   # no index

    def test_embedding_thresholds_and_llm_choice(self):
        from hoofbeats.linker import Linker
        replies = []

        class Chooser(TestLLMExtract.ChunkFake):
            pass
        llm = Chooser(lambda p: replies.pop(0))
        lk = Linker(self.ont, self.ext, self.index(), llm, hi=0.75, lo=0.3, k=3, workers=1)
        replies[:] = [json.dumps({"choice": "HP:0001083"}), json.dumps({"choice": "none"})]
        # trigram cosine to the best string: hyperamonemia 0.90, displaced lens 0.65, lens displacement 0.37,
        # lens slipping 0.24 (hi 0.75, lo 0.3)
        out = lk.link_many([("Fever", "fever"), ("hyperamonemia", "ammonia high"), ("displaced lens", "lens off"),
                            ("lens displacement", "lens displacement"), ("lens slipping", "lens slipping")])
        self.assertEqual((out[0][0].method, out[0][0].grounding), ("dict:exact_label", "strong"))
        self.assertEqual((out[1][0].hpo_id, out[1][0].method, out[1][0].grounding), ("HP:0001987", "embedding", "supported"))
        self.assertEqual((out[2][0].hpo_id, out[2][0].method, out[2][0].grounding), ("HP:0001083", "llm_choice", "weak"))
        self.assertEqual((out[3][0].hpo_id, out[3][0].method), (None, "unlinked"))     # the LLM said none
        self.assertEqual((out[4][0].hpo_id, out[4][0].method), (None, "unlinked"))     # below lo: not asked
        self.assertEqual(len(llm.sent), 2)
        enum = llm.sent[0][1]["properties"]["choice"]["enum"]
        self.assertEqual(len(enum), 4)
        self.assertEqual(enum[-1], "none")                                 # only the 3 candidates, or none
        no_llm = Linker(self.ont, self.ext, self.index(), None, hi=0.75, lo=0.3, k=3).link_many([("displaced lens", "")])
        self.assertEqual((no_llm[0][0].hpo_id, no_llm[0][0].method), (None, "borderline"))
        with self.assertRaises(PolicyError):
            Linker(self.ont, self.ext, self.index(), Chooser(lambda p: "{}", "https://x.example/v1"), hi=0.75,
                   lo=0.3).link_many([("displaced lens", "")])

    def test_chooser_v2_sentence_and_broader_terms(self):
        from hoofbeats.linker import CHOOSE_PROMPT, Linker
        answers = []
        llm = TestLLMExtract.ChunkFake(lambda p: json.dumps({"choice": answers.pop(0)}))
        lk = Linker(self.ont, self.ext, self.index(), llm, hi=0.75, lo=0.3, k=3, workers=1, chooser_version=2)
        answers[:] = ["HP:0000478", "HP:0001083", "HP:0000118"]
        out = lk.link_many([("displaced lens", "lens off", "On exam the lens was off centre."),
                            ("displaced lens", "lens off", ""), ("displaced lens", "lens off")])
        prompt, schema = llm.sent[0]
        self.assertIn('sentence: "On exam the lens was off centre."', prompt)
        enum = schema["properties"]["choice"]["enum"]
        self.assertEqual(enum[:3], [t for t, _ in out[1][0].candidates])            # the top-k first
        self.assertIn("HP:0000478", enum)                                          # Abnormality of the eye: a parent
        self.assertNotIn("HP:0000118", enum)                                       # the root is never offered
        self.assertEqual(enum[-1], "none")
        self.assertEqual((out[0][0].hpo_id, out[0][0].method, out[0][0].grounding, out[0][0].score),
                         ("HP:0000478", "llm_choice:broader", "weak", None))
        self.assertEqual((out[1][0].hpo_id, out[1][0].method), ("HP:0001083", "llm_choice"))
        self.assertIn('sentence: "lens off"', llm.sent[1][0])                      # no sentence: the quote stands in
        self.assertIsNone(out[2][0].hpo_id)                                        # an id that was not offered
        v1 = TestLLMExtract.ChunkFake(lambda p: json.dumps({"choice": "none"}))
        Linker(self.ont, self.ext, self.index(), v1, hi=0.75, lo=0.3, k=3, workers=1).link_many(
            [("displaced lens", "lens off", "a sentence")])
        self.assertEqual(v1.sent[0][0], CHOOSE_PROMPT.format(phrase="displaced lens", quote="lens off",
                                                             options="\n".join(f"{t}: {self.ont.name(t)}" for t in
                                                                               v1.sent[0][1]["properties"]["choice"]["enum"][:-1])))
        with self.assertRaises(ValueError):
            Linker(self.ont, self.ext, chooser_version=3)

    def test_crosscheck(self):
        from hoofbeats.linker import crosscheck
        cc = lambda h, q, **kw: crosscheck(self.ont, self.ext, h, q, **kw)
        self.assertEqual(cc("HP:0001250", "had recurrent epileptic seizures").verdict, "agree")
        self.assertEqual(cc("HP:0007359", "had seizures").verdict, "agree")               # a child of the rule term
        self.assertEqual(cc("HP:0001083", "the lens had slipped upward").verdict, "no_rule_term")
        # the quote is fully accounted for by other terms: nothing supports Arachnodactyly here
        chk = cc("HP:0001166", "No fever and no seizures were noted.")
        self.assertEqual((chk.verdict, set(chk.rule_terms)), ("explained_by_other_terms", {"HP:0001945", "HP:0001250"}))
        self.assertEqual(cc("HP:0001166", "fever and very long thin fingers").verdict, "disagree")   # words remain
        low = cc("HP:0001083", "the lens had slipped upward", similarity=0.2, lo=0.62)
        self.assertEqual((low.verdict, low.low_similarity), ("no_rule_term", True))       # a flag only, not a drop

    def test_label_similarity(self):
        ix = self.index()
        sim = ix.similarity(["Hyperammonemia", "fever"], ["HP:0001987", "HP:0001987"])
        self.assertAlmostEqual(sim[0], 1.0, places=5)
        self.assertLess(sim[1], 0.5)
        self.assertEqual(ix.similarity(["x"], ["HP:9999999"]), [None])

    def test_llm_reader_merged_with_rules(self):
        from hoofbeats.intake import parse_case
        from hoofbeats.linker import Linker
        from hoofbeats.llm_reader import LLMReader
        text = "The patient had fever. No seizures. The lens had slipped upward."
        reply = json.dumps({"findings": [
            {"phrase": "fever", "status": "present", "quote": "had fever"},                  # agrees with the rules
            {"phrase": "seizures", "status": "present", "quote": "seizures"},                # rules: absent ("No")
            {"phrase": "dislocated lens", "status": "present", "quote": "The lens had slipped upward"},   # paraphrase
            {"phrase": "arachnodactyly", "status": "present", "quote": "No seizures."},      # quote explained by Seizure
            {"phrase": "zzz", "status": "present", "quote": "patient"}]})                    # links to nothing
        lk = Linker(self.ont, self.ext, self.index(), None, hi=0.75, lo=0.3, k=3)

        def run(**kw):
            reader = LLMReader(TestLLMExtract.ChunkFake(lambda p: reply), lk, self.ont, self.ext, workers=1)
            c = parse_case({"case_id": "x", "note_text": text}, self.ont, self.ext, llm_reader=reader, **kw)
            self.assertFalse([w for w in c.meta["intake_warnings"] if w.startswith("provenance")])   # spans verified
            return c, {f.hpo_id: f for f in c.findings}, reader
        c, got, reader = run()
        self.assertEqual({h: f.status for h, f in got.items()},
                         {"HP:0001945": Status.PRESENT, "HP:0001250": Status.PRESENT, "HP:0001083": Status.PRESENT})
        self.assertEqual(got["HP:0001945"].grounding, "strong")                         # rules + llm on the same words
        fever = c.meta["provenance"]["HP:0001945"]
        self.assertEqual((len(fever), fever[0]["readers"], fever[0]["grounding"]), (1, ["rules", "llm"], "strong"))
        self.assertEqual(fever[0]["other_spans"][0]["quote"], "had fever")
        conflict = c.meta["provenance"]["conflicts"][0]
        self.assertEqual((conflict["hpo_id"], conflict["status"], conflict["action"]), ("HP:0001250", "absent", "dropped"))
        self.assertEqual([e["readers"] for e in c.meta["provenance"]["HP:0001250"]], [["llm"]])
        self.assertEqual(got["HP:0001083"].link[0], "dict:synonym:exact")
        verdicts = {r["phrase"]: (r["verdict"], r["kept"]) for r in reader.log}
        self.assertEqual(verdicts["arachnodactyly"], ("explained_by_other_terms", False))
        self.assertEqual(verdicts["dislocated lens"], ("no_rule_term", True))
        self.assertEqual(verdicts["zzz"], ("unlinked", False))
        _, got, _ = run(status_policy="rules")
        self.assertEqual(got["HP:0001250"].status, Status.ABSENT)                       # the rules' "No seizures"
        c, got, _ = run(rules_text=False)
        self.assertEqual({f.reader for f in c.findings}, {"llm"})
        self.assertEqual(got["HP:0001945"].grounding, "strong")                         # dictionary link, exact label
        with self.assertRaises(ValueError):
            run(status_policy="majority")

    def test_llm_reader_prefetch_batches_requests(self):
        import threading
        from hoofbeats.linker import Linker
        from hoofbeats.llm_reader import LLMReader
        live, peak, lock = [0], [0], threading.Lock()

        class Slow(TestLLMExtract.ChunkFake):
            def chat(self, user, system="", images=(), schema=None):
                with lock:
                    live[0] += 1
                    peak[0] = max(peak[0], live[0])
                import time
                time.sleep(0.05)
                with lock:
                    live[0] -= 1
                return super().chat(user, system, images, schema)
        llm = Slow(lambda p: json.dumps({"findings": [{"phrase": "fever", "status": "present", "quote": "fever"}]}))
        reader = LLMReader(llm, Linker(self.ont, self.ext), self.ont, self.ext, workers=8)
        texts = [(f"Case {i} had fever.", "note", "note_text", f"c{i}") for i in range(6)]
        reader.prefetch(texts)
        self.assertEqual(len(llm.sent), 6)
        self.assertGreater(peak[0], 1)                                  # the six texts' requests overlapped
        self.assertEqual([f.hpo_id for f in reader("Case 3 had fever.", "note", "note_text")], ["HP:0001945"])
        self.assertEqual(len(llm.sent), 6)                              # answered from the prefetched results
        self.assertEqual([r["case"] for r in reader.reports], [f"c{i}" for i in range(6)])

    def test_llm_eval_helpers(self):
        from hoofbeats.llm_eval import absent_sample, absent_stats, configs, hand_checked
        it = lambda h, st, res, vis: {"hpo_id": h, "status": st, "result": res, "vocab_visible": vis}
        rows = [{"case": "c1", "items": [it("HP:1", "present", "exact", True), it("HP:2", "present", "missed", False),
                                         it("HP:3", "absent", "near", False), it("HP:4", "absent", "missed", True)],
                 "preds": [{"hpo_id": "HP:3", "status": "absent", "method": "llm"},
                           {"hpo_id": "HP:9", "status": "absent", "method": "llm"},
                           {"hpo_id": "HP:1", "status": "present", "method": "llm+rules"}],
                 "extras": [{"hpo_id": "HP:9", "status": "absent", "method": "llm", "quote": "no x", "sentence": "No x."}]}]
        b = lambda h, st, e: {"case_id": "c1", "hpo_id": h, "gold_status": st, "expressed": e}
        hc = hand_checked(rows, [b("HP:1", "present", "yes"), b("HP:2", "present", "yes"), b("HP:3", "absent", "yes"),
                                 b("HP:4", "absent", "no"), b("HP:7", "present", "yes")])     # HP:7 not in this run
        self.assertEqual((hc["all"]["n"], hc["all"]["strict"], hc["all"]["lenient"]), (3, 1 / 3, 2 / 3))
        self.assertEqual((hc["label"]["n"], hc["paraphrase"]["n"], hc["paraphrase"]["lenient"]), (1, 2, 0.5))
        ab = absent_stats(rows)
        self.assertEqual((ab["pred"], ab["extra"], ab["matched_share"], ab["by_method"]), (2, 1, 0.5, {"llm": 2}))
        self.assertEqual(ab["gold_recall"]["n"], 2)
        self.assertEqual([r["hpo_id"] for r in absent_sample(rows)], ["HP:9"])
        self.assertEqual(list(configs(None)), ["rules"])
        self.assertEqual(list(configs(object(), ("rules",))), ["rules", "llm", "rules+llm/status=rules"])

    def test_calibration(self):
        from hoofbeats.linker import calibrate
        cal = calibrate(self.index(), n=50, k=3)
        self.assertEqual(cal["n"], 7)     # the strings of the 3 terms with two differently worded strings (3 + 2 + 2)
        self.assertTrue(0.0 <= cal["lo"] <= cal["hi"] <= 1.0)
        self.assertEqual(sum(b["n"] for b in cal["bands"]), cal["n"])


def write_medpix(root: Path) -> Path:
    mp = root / "MedPix-2-0"
    (mp / "images").mkdir(parents=True, exist_ok=True)
    (mp / "splitted_dataset").mkdir(exist_ok=True)
    for name in ("MPX1_a", "MPX1_b", "MPX2_a"):
        (mp / "images" / f"{name}.png").write_bytes(b"\x89PNG fake")
    cases = [{"U_id": "MPX1", "TAC": ["MPX1_a"], "MRI": ["MPX1_b"],
              "Case": {"Title": "Hypophosphatasia, adult", "History": "Tall stature and dislocated lens. No seizures. Known hypophosphatasia.",
                       "Exam": "N/A", "Findings": "Low alkaline phosphatase. Hyperammonemia.",
                       "Differential Diagnosis": "x", "Case Diagnosis": "Hypophosphatasia", "Diagnosis By": "N/A"},
              "Topic": {"Title": "Hypophosphatasia"}},
             {"U_id": "MPX2", "TAC": ["MPX2_a"], "MRI": [],
              "Case": {"Title": "Unknown thing", "History": "Fever.", "Exam": "N/A", "Findings": "N/A"}, "Topic": {}}]
    desc = [{"Type": "CT", "U_id": "MPX1", "image": "MPX1_a", "Location": "Head",
             "Description": {"Modality": "CT - noncontrast", "Plane": "Axial", "Age": "40", "Sex": "male",
                             "Caption": "Ectopia lentis is seen."}},
            {"Type": "MR", "U_id": "MPX1", "image": "MPX1_b", "Location": "Head", "Description": {"Caption": ""}},
            {"Type": "CT", "U_id": "MPX2", "image": "MPX2_a", "Location": "Chest", "Description": {}}]
    (mp / "Case_topic.json").write_text(json.dumps(cases))
    (mp / "Descriptions.json").write_text(json.dumps(desc))
    (mp / "splitted_dataset" / "uid_list.json").write_text(json.dumps({"test": ["MPX2"], "train": ["MPX1"]}))
    return mp


class TestMedPix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp())
        kb = build(root)
        cls.mp = write_medpix(root)
        cls.ont = Ontology(kb / "hp.obo")
        db = AnnotationDB(kb / "phenotype.hpoa", cls.ont, ("OMIM", "ORPHA"))
        cls.r = LRRanker(cls.ont, db)
        cls.ext = Extractor(cls.ont)
        cls.m = DiseaseMatcher(db.names)

    def test_matcher(self):
        self.assertEqual(DiseaseMatcher({"X:1": "Small cell cancer of the lung"}).match(
            "Non-small Cell Lung Cancer (Large-cell variant)"), [])           # "non-small" is not "small"
        self.assertEqual(DiseaseMatcher({"X:1": "Small cell cancer of the lung"}).match("small cell lung cancer"), ["X:1"])
        self.assertEqual(self.m.match("Hypophosphatasia, adult form"), ["ORPHA:10"])
        self.assertEqual(self.m.match("Hypophosphatasia Gamma"), ["OMIM:3"])          # most specific wins
        self.assertEqual(self.m.match("Disease Alpha"), [])                            # only generic/short words
        self.assertEqual(DiseaseMatcher({"X:1": "Locked-in syndrome"}).match("jumped locked facets"), [])

    def test_load_medpix_no_leakage(self):
        cases = {c.case_id: c for c in load_medpix(self.mp, self.m, self.ext)}
        c = cases["MPX1"]
        self.assertEqual(c.truth, ["ORPHA:10"])
        st = {f.hpo_id: f.status for f in c.findings}
        self.assertEqual((st["HP:0000098"], st["HP:0001083"], st["HP:0001250"]),
                         (Status.PRESENT, Status.PRESENT, Status.ABSENT))
        self.assertNotIn("HP:0003282", st)                                  # radiologist findings are not input
        self.assertNotIn("hypophosphatasia", c.meta["history"].lower())     # diagnosis masked in history
        self.assertIn("alkaline phosphatase", c.meta["reference_text"].lower())
        self.assertEqual([i["id"] for i in c.meta["images"]], ["MPX1_a", "MPX1_b"])
        self.assertEqual(cases["MPX2"].truth, [])
        self.assertEqual([c.case_id for c in load_medpix(self.mp, self.m, self.ext, "test")], ["MPX2"])
        labels = self.mp / "labels.csv"
        labels.write_text("U_id,title,truth\nMPX2,Unknown thing,OMIM:4\n")
        self.assertEqual({c.case_id: c.truth for c in load_medpix(self.mp, self.m, self.ext, labels_csv=labels)}["MPX2"], ["OMIM:4"])

    def test_read_images_maps_free_text_to_hpo_and_caches(self):
        c = {c.case_id: c for c in load_medpix(self.mp, self.m, self.ext)}["MPX1"]
        reply = json.dumps({"report": "Dislocated lens is seen. No seizures activity markers. Hyperammonemia not assessable.",
                            "findings": [{"finding": "low alkaline phosphatase", "status": "present"},
                                         {"finding": "tall stature", "status": "absent"},
                                         {"finding": "made-up sign", "status": "present"}]})
        llm, cache = Fake(reply), {}
        found, recs = read_images(llm, self.ext, c, max_images=4, cache=cache)
        st = {f.hpo_id: f.status for f in found}
        self.assertEqual(st["HP:0001083"], Status.PRESENT)     # from free-text report
        self.assertEqual(st["HP:0003282"], Status.PRESENT)     # from findings list
        self.assertEqual(st["HP:0000098"], Status.ABSENT)
        self.assertEqual(llm.calls, 2)
        self.assertTrue(llm.images[0].endswith("MPX1_a.png"))
        self.assertIn("40-year-old male", image_prompt_of(c))
        read_images(llm, self.ext, c, cache=cache)
        self.assertEqual(llm.calls, 2)                         # cached: no new model calls
        self.assertTrue(all(r_["parsed"] for r_ in recs))
        bad, recs = read_images(Fake("not json"), self.ext, c, cache={})
        self.assertEqual((bad, [x["parsed"] for x in recs]), ([], [False, False]))

    def test_text_only_model_refuses_images(self):
        with self.assertRaises(ValueError):
            MedGemma("google/medgemma-27b-text-it").chat("x", images=("a.png",))
        self.assertTrue(MedGemma().sees_images)


class FakeVision:
    # Stage A replies (CONTEXT_SYSTEM): one finding per image; `finding` is what the image "shows"
    # the check (VERIFY_SYSTEM) answers `confirm` for every finding
    def __init__(self, model="fake-vision", endpoint=None, sees_images=True, names_diagnosis=False,
                 finding="Ectopia lentis", details="lens displaced superotemporally", confirm="yes"):
        self.model, self.endpoint, self.sees_images = model, endpoint, sees_images
        self.is_local, self.dry_run, self.names_diagnosis = endpoint is None, False, names_diagnosis
        self.finding, self.details, self.confirm = finding, details, confirm
        self.calls = []

    def chat(self, user, system="", images=()):
        from hoofbeats.llm import CLASSIFY_PROMPT, CONTEXT_SYSTEM, VERIFY_SYSTEM
        if user == CLASSIFY_PROMPT:                # an image without labels: say it is radiology
            return "radiology"
        self.calls.append((system, user, tuple(images)))
        if system == VERIFY_SYSTEM:
            n = user.count("\n1. ") + sum(f"\n{i}. " in user for i in range(2, 20))
            return json.dumps({"answers": {str(i): self.confirm for i in range(1, n + 1)}})
        if system == CONTEXT_SYSTEM:
            return json.dumps({"technique": "Axial CT without contrast.", "anatomy": "Orbits and globes.",
                               "findings": [{"finding": self.finding, "location": "right globe",
                                             "details": self.details}],
                               "pertinent_negatives": ["No orbital fracture"],
                               "summary": "One finding." + (" Hypophosphatasia." if self.names_diagnosis else "")})
        return ""


class TestConfidence(unittest.TestCase):
    def test_by_confidence(self):
        from hoofbeats.llm import by_confidence, render_context
        obj = {"technique": "CT", "findings": [{"finding": "A", "confidence": "definite"},
                                              {"finding": "B", "confidence": "probable"}, {"finding": "C"}],
               "pertinent_negatives": ["No D"], "summary": "A, B and C."}
        self.assertEqual([f["finding"] for f in by_confidence(obj, "definite")["findings"]], ["A"])
        self.assertEqual([f["finding"] for f in by_confidence(obj, "probable")["findings"]], ["A", "B"])
        self.assertNotIn("summary", by_confidence(obj, "definite"))
        self.assertIs(by_confidence(obj, "all"), obj)
        self.assertIn("No D", render_context(by_confidence(obj, "definite"), "", "Image 1"))


class TestImageKinds(unittest.TestCase):
    def test_kinds_and_prompts(self):
        from hoofbeats.llm import context_prompt, image_kind
        kind = lambda m, l="": image_kind([{"modality": m, "location": l}])
        self.assertEqual([kind("CT - noncontrast", "Head"), kind("CT w/contrast (IV)", "Eye and Orbit"),
                          kind("12-lead ECG"), kind("Fundus photograph", "Eye"), kind("Photograph", "Face"),
                          kind("Histology - H&E", "Liver"), kind("Colonoscopy"), kind("Fluoroscopy"),
                          kind("EEG"), kind("Image")],
                         ["radiology", "radiology", "ecg", "eye", "photo", "microscopy", "endoscopy", "radiology",
                          "eeg", "general"])
        ecg = context_prompt([{"modality": "12-lead ECG", "location": "Heart"}], {"age": 30, "sex": "female"})
        self.assertIn("QT and QTc interval", ecg)
        self.assertIn("one small square is 40 ms", ecg)
        self.assertNotIn("patient's right is on the left", ecg)           # radiology rules only for radiology
        self.assertNotIn("lungs (each side)", ecg)                         # no body-region checklist
        face = context_prompt([{"modality": "Clinical photograph", "location": "Face"}], {})
        self.assertIn("Elements of Morphology", face)
        self.assertIn("size, shape, colour or density", context_prompt([{"modality": "Image"}], {}))

    def test_unlabelled_image_is_classified_first(self):
        from hoofbeats.llm import CLASSIFY_PROMPT, KIND_ROLES, RADIOLOGY_SYSTEM, context_system, image_context

        class Fake:
            model, endpoint, is_local, dry_run, sees_images = "fake-vision", None, True, False, True

            def __init__(self, kind):
                self.kind, self.calls = kind, []

            def chat(self, user, system="", images=()):
                self.calls.append((user, system))
                return self.kind if user == CLASSIFY_PROMPT else json.dumps({"findings": [], "summary": "Normal."})

        case = Case("x", [], [], meta={"policy": "open", "images": [{"path": "a.png"}], "history": ""})
        ecg = Fake("ECG.")
        parts = image_context(ecg, case, 1)
        self.assertEqual([u == CLASSIFY_PROMPT for u, _ in ecg.calls], [True, False])
        self.assertEqual(parts[0].kind, "ecg")
        self.assertIn(KIND_ROLES["ecg"], ecg.calls[1][1])
        self.assertIn("QT and QTc interval", ecg.calls[1][0])
        self.assertEqual(image_context(Fake("no idea"), case, 1)[0].kind, "general")
        labelled = Case("x", [], [], meta={"policy": "open", "images": [{"path": "a.png", "modality": "CT"}],
                                           "history": ""})
        ct = Fake("ecg")
        self.assertEqual(image_context(ct, labelled, 1)[0].kind, "radiology")
        self.assertEqual([s for _, s in ct.calls], [RADIOLOGY_SYSTEM])        # labelled: no classification
        self.assertEqual(context_system("radiology"), RADIOLOGY_SYSTEM)


class TestFollowup(unittest.TestCase):
    def test_merge_and_results(self):
        from hoofbeats.followup import merge, result_kind, result_line
        orig = {"case_id": "c1", "note_text": "Tall.", "images": [{"path": "a.png"}], "genes": ["X"], "policy": "open",
                "labs": [{"name": "ALT", "value": 30}], "truth": ["OMIM:1"]}
        new = {"date": "2026-11-20", "note_text": "Echo done.", "images": [{"path": "b.png"}, {"path": "/abs/a.png"}],
               "labs": [{"name": "AST", "value": 40}], "genes": ["Y"], "policy": "local_only",
               "test_results": [{"test": "FBN1 gene testing", "result": "positive", "genes": ["FBN1"],
                                 "classification": "pathogenic"}, {"test": "angiography", "result": "negative"}]}
        m = merge(orig, new, Path("/case"), Path("/fu"))
        self.assertEqual((m["case_id"], m["note_text"]), ("c1", "Tall."))      # the original note is not re-read
        self.assertTrue(m["followup_text"].startswith("Follow-up (2026-11-20):\nEcho done.\nTest results:"))
        self.assertIn("FBN1 gene testing: positive (pathogenic variant in FBN1).", m["followup_text"])
        self.assertIn("angiography: negative.", m["followup_text"])
        self.assertEqual([x["path"] for x in m["images"]], [str(Path("/case/a.png").resolve()),
                                                             str(Path("/fu/b.png").resolve()), str(Path("/abs/a.png"))])
        self.assertEqual([x["name"] for x in m["labs"]], ["ALT", "AST"])
        self.assertEqual((m["genes"], m["policy"], m["truth"]), (["X", "Y"], "local_only", ["OMIM:1"]))
        self.assertEqual([result_kind(x) for x in ("Positive", "normal", "not detected", "4.6 cm", None)],
                         ["positive", "negative", "negative", "other", "other"])
        self.assertEqual(result_line({"test": "Echo", "result": "abnormal", "details": "root 4.6 cm"}),
                         "Echo: abnormal (root 4.6 cm).")


class TestImageContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        cls.kb = build(cls.root)
        cls.mp = write_medpix(cls.root)
        cls.ont = Ontology(cls.kb / "hp.obo")
        cls.r = LRRanker(cls.ont, AnnotationDB(cls.kb / "phenotype.hpoa", cls.ont, ("OMIM", "ORPHA")))
        cls.ext = Extractor(cls.ont)
        cls.m = DiseaseMatcher(cls.r.db.names)

    def case(self, uid="MPX1"):
        return {c.case_id: c for c in load_medpix(self.mp, self.m, self.ext)}[uid]

    def case_file(self, name="pt", **extra):
        # a case file with two images next to it (relative paths)
        d = self.root / name
        d.mkdir(exist_ok=True)
        (d / "head.png").write_bytes(b"\x89PNG fake")
        (d / "knee.jpg").write_bytes(b"\xff\xd8\xff\xe0fake jpeg")
        obj = {"case_id": name, "note_text": "Tall stature. No seizures.", "age": 40, "sex": "male",
               "details": {"setting": "clinic"}, "policy": "open",
               "images": [{"path": "head.png", "modality": "CT - noncontrast", "plane": "Axial", "location": "Head"},
                          {"path": "knee.jpg", "modality": "X-ray", "location": "Knee"}], **extra}
        (d / "case.json").write_text(json.dumps(obj))
        return d / "case.json"

    def test_context_prompt_and_rendering(self):
        from hoofbeats.llm import context_prompt, details_of, render_context
        c = self.case()
        p = context_prompt(c.meta["images"][:1], details_of(c), c.meta["history"])
        for part in ("This is one medical image: CT - noncontrast, axial plane, head.", "Patient: 40-year-old male.",
                     "Clinical question: Tall stature",
                     "the patient's right is on the left of the image", "third and fourth ventricles",
                     "check whether it is a normal structure", "must not contradict", '"confidence"'):
            self.assertIn(part, p)
        chest = context_prompt([{"modality": "XR", "location": "Chest, Pulmonary"}], {})
        self.assertIn("lungs (each side)", chest)
        self.assertNotIn("ventricles", chest)
        self.assertNotIn("Ectopia lentis is seen", p)                     # captions are never sent
        self.assertNotIn("hypophosphatasia", p.lower())                   # the history is delabelled
        text = render_context(json.loads(FakeVision().chat("", __import__("hoofbeats.llm").llm.CONTEXT_SYSTEM)),
                              "", "Image 1 (CT)")
        self.assertIn("- Ectopia lentis (right globe): lens displaced superotemporally", text)
        self.assertIn("Pertinent negatives: No orbital fracture", text)
        self.assertEqual(render_context(None, "free text reply", "Image 1"), "Image 1: free text reply")
        self.assertEqual(render_context(None, "", "Image 1"), "")

    def test_images_go_through_stage_a_into_stage1(self):
        from hoofbeats.grounding import level_of
        from hoofbeats.intake import load_case
        from hoofbeats.llm import CONTEXT_SYSTEM
        path, describer = self.case_file(), FakeVision()
        c = load_case(path, self.ont, self.ext, disease_names=self.r.db.names, describer=describer)
        self.assertEqual([(s, len(i)) for s, _, i in describer.calls], [(CONTEXT_SYSTEM, 1), (CONTEXT_SYSTEM, 1)])
        self.assertIn("Patient: 40-year-old male. Setting: clinic.", describer.calls[0][1])
        self.assertIn("Clinical question: Tall stature. No seizures.", describer.calls[0][1])
        self.assertEqual([(i["id"], i["path"]) for i in c.meta["images"]],
                         [("head", str(path.parent / "head.png")), ("knee", str(path.parent / "knee.jpg"))])
        self.assertIn("Image 1 (CT - noncontrast, axial plane, head):", c.meta["image_context"])
        self.assertEqual(c.meta["described"]["model"], "fake-vision")
        el = next(f for f in c.findings if f.hpo_id == "HP:0001083")       # Ectopia lentis, from the description
        self.assertEqual((el.status, el.field), (Status.PRESENT, "image_context"))
        self.assertEqual(c.meta["image_context"][el.start:el.end], el.quote)    # grounded in the description text
        self.assertEqual(level_of(el), "strong")
        self.assertFalse([w for w in c.meta["intake_warnings"] if w.startswith("provenance")])
        self.assertEqual(self.r.score(c).top(1)[0][0], "OMIM:1")           # tall stature + ectopia lentis
        from hoofbeats.intake import IntakeError
        with self.assertRaisesRegex(IntakeError, "has images but no image-context model"):   # never skipped silently
            load_case(path, self.ont, self.ext)
        import argparse, contextlib, io
        import hoofbeats.__main__ as hb
        from hoofbeats.__main__ import main
        from hoofbeats.context import DEFAULT_CONTEXT_ENDPOINT, DEFAULT_CONTEXT_MODEL, context_choice, reads_images
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):       # no --context-model: the default model describes them (dry run)
            main(["--kb", str(self.kb), "--common", "none", "--dry-run", "diagnose", str(path), "--out", str(self.root / "r")])
        self.assertIn(f"image context (Stage A) by {DEFAULT_CONTEXT_MODEL}", buf.getvalue())
        self.assertIn("would send 2 requests", buf.getvalue())
        ns = lambda model, vision="auto", ctx=None: argparse.Namespace(llm_model=model, llm_vision=vision, context_model=ctx,
                                                                       llm_endpoint="http://x/v1", context_endpoint=None)
        self.assertEqual(context_choice(ns("gemini-2.5-flash"))[:2], ("gemini-2.5-flash", "http://x/v1"))   # main model
        self.assertEqual(context_choice(ns("hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"))[0],
                         "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M")
        for text_only in ("Qwen/Qwen2.5-7B-Instruct", "google/medgemma-27b-text-it", None):
            self.assertEqual(context_choice(ns(text_only))[:2], (DEFAULT_CONTEXT_MODEL, DEFAULT_CONTEXT_ENDPOINT))
        self.assertTrue(reads_images(ns("Qwen/Qwen2.5-VL-7B-Instruct")))
        self.assertTrue(reads_images(ns("my-model", "yes")) and not reads_images(ns("gemini-2.5-flash", "no")))
        self.assertEqual(context_choice(ns("gemini-2.5-flash", ctx="other"))[0], "other")   # --context-model wins
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):       # a case without images builds no image model
            main(["--kb", str(self.kb), "--common", "none", "--dry-run", "diagnose",
                  str(self.case_file("noimg", images=[])), "--out", str(self.root / "r")])
        self.assertNotIn("image context (Stage A)", buf.getvalue())

    def test_existing_image_context_is_kept_and_missing_files_skipped(self):
        from hoofbeats.intake import load_case
        path = self.case_file("pt2", image_context="Arachnodactyly on hand films.",
                              images=[{"path": "head.png"}, {"path": "missing.png"}])
        c = load_case(path, self.ont, self.ext, describer=FakeVision())
        self.assertTrue(c.meta["image_context"].startswith("Arachnodactyly on hand films.\n\nImage 1 (image):"))
        self.assertTrue({"HP:0001166", "HP:0001083"} <= {f.hpo_id for f in c.findings})
        self.assertTrue(any("images[1]: file not found" in w for w in c.meta["intake_warnings"]))
        self.assertEqual(len(c.meta["images"]), 1)

    def test_timeouts_and_missing_server_fail_the_request_not_the_run(self):
        import socket, threading
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)     # accepts, never answers
        port, held = srv.getsockname()[1], []
        threading.Thread(target=lambda: held.append(srv.accept()), daemon=True).start()
        slow = MedGemma("m", f"http://127.0.0.1:{port}/v1/chat/completions", timeout=0.5)
        with self.assertRaisesRegex(RuntimeError, "no reply within 0 s|no reply within 1 s"):
            slow.chat("hi")
        for conn, _ in held:
            conn.close()
        srv.close()
        drop = socket.socket(); drop.bind(("127.0.0.1", 0)); drop.listen(1)  # accepts, then hangs up
        threading.Thread(target=lambda: drop.accept()[0].close(), daemon=True).start()
        with self.assertRaisesRegex(RuntimeError, "connection .* was lost|cannot reach"):
            MedGemma("m", f"http://127.0.0.1:{drop.getsockname()[1]}/v1/chat/completions", timeout=2).chat("hi")
        drop.close()
        free = socket.socket(); free.bind(("127.0.0.1", 0)); gone = free.getsockname()[1]; free.close()
        with self.assertRaisesRegex(RuntimeError, "cannot reach"):
            MedGemma("m", f"http://127.0.0.1:{gone}/v1/chat/completions", timeout=2).chat("hi")
        self.assertEqual(MedGemma().timeout, 600)

    def test_reply_with_a_draft_json_in_its_plan(self):
        # MedGemma 1.5 wrote its JSON twice: a draft inside its plan, then the final answer
        from hoofbeats.llm import image_context, reply_json
        keys = ("findings", "summary", "technique")
        final = {"findings": [{"finding": "Enlarged lateral ventricles"}], "summary": "final"}
        draft = {"findings": [{"finding": "draft"}], "summary": "draft"}
        raw = ("thought\nPlan...\n**Final JSON structure:**\n```json\n" + json.dumps(draft)
               + "\n```\nNow the answer.\n```json\n" + json.dumps(final, indent=2) + "\n```")
        old = _json(raw, "{") or {}                             # first "{" to last "}" spans both blocks:
        self.assertNotEqual(old.get("summary"), "final")        # nothing, or the draft
        self.assertEqual(reply_json(raw, keys)["summary"], "final")
        cut = raw[:-25]                                         # cut off inside the final JSON
        self.assertEqual(reply_json(cut, keys)["findings"][0]["finding"], "Enlarged lateral ventricles")
        llm = FakeVision()
        llm.chat = lambda user, system="", images=(): raw
        ctx = image_context(llm, self.case(), max_images=1)[0]
        self.assertEqual((ctx.parsed, ctx.cut_off), (True, False))
        self.assertIn("- Enlarged lateral ventricles", ctx.text)
        self.assertNotIn("draft", ctx.text)

    def test_unreadable_description_is_not_passed_on(self):
        from hoofbeats.intake import load_case
        from hoofbeats.llm import image_context
        loop = "thought\nThe user wants a report.\n" + "*   Lacrimal glands appear normal.\n" * 400
        llm = FakeVision()
        llm.max_new_tokens = 2048
        llm.chat = lambda user, system="", images=(): loop
        ctx = image_context(llm, self.case(), max_images=1)[0]
        self.assertEqual((ctx.text, ctx.parsed, ctx.cut_off), ("", False, True))   # stopped at the limit, no JSON
        c = load_case(self.case_file("pt3"), self.ont, self.ext, describer=llm)
        self.assertEqual(c.meta["image_context"], "")                              # nothing reaches Stage 1
        self.assertTrue(any("could not be read (stopped at the token limit)" in w for w in c.meta["intake_warnings"]))
        llm.chat = lambda user, system="", images=(): "Enlarged ventricles on both sides."    # short prose: kept
        self.assertEqual(image_context(llm, self.case(), max_images=1)[0].text, "Image 1 (CT - noncontrast, axial plane, "
                                                                               "head): Enlarged ventricles on both sides.")

    def test_cut_off_replies_are_flagged(self):
        from hoofbeats.llm import cut_off, image_context
        whole = '{"findings": [{"finding": "A"}, {"finding": "B"}], "summary": "s"}'
        cut = 'Plan: list them... ```json\n{"findings": [{"finding": "Ectopia lentis"}, {"finding": "lateral ven'
        self.assertEqual((cut_off(whole), cut_off("```json\n" + whole + "\n```"), cut_off(cut), cut_off("no json")),
                         (False, False, True, False))
        llm = FakeVision()
        llm.chat = lambda user, system="", images=(): cut
        ctx = image_context(llm, self.case(), max_images=1)[0]
        self.assertTrue(ctx.cut_off)
        self.assertIn("- Ectopia lentis", ctx.text)             # what was finished is still read

    def test_vision_model_and_credentialed_cases(self):
        from hoofbeats.intake import load_case
        from hoofbeats.llm import image_context
        with self.assertRaises(ValueError):
            image_context(FakeVision(sees_images=False), self.case())
        remote = FakeVision(endpoint="https://example.com/v1/chat/completions")
        path = self.case_file("cred", policy="local_only")
        with self.assertRaises(PolicyError):                    # credentialed images never go to a remote model
            load_case(path, self.ont, self.ext, describer=remote)
        self.assertEqual(remote.calls, [])
        self.assertEqual(len(load_case(path, self.ont, self.ext, describer=FakeVision()).meta["images"]), 2)

    def test_image_type(self):
        from hoofbeats.llm import image_type
        path = self.case_file("types")
        self.assertEqual(image_type(path.parent / "knee.jpg"), "image/jpeg")
        self.assertEqual(image_type(self.mp / "images" / "MPX1_a.png"), "image/png")

    def test_medpix_conditions(self):
        from hoofbeats.__main__ import image_agreement
        from hoofbeats.context import MM_CONDITIONS, medpix_context_case
        describer = FakeVision(finding="Low alkaline phosphatase", details="")   # the key finding for ORPHA:10
        rows = {x["condition"]: x for x in medpix_context_case(self.case(), ["text", "context", "reference"], describer,
                                                                self.r, self.ext, max_images=2, agreement=image_agreement)}
        self.assertEqual(len(describer.calls), 2)              # one description per image
        self.assertEqual({k: v["rank"] for k, v in rows.items()}, {"text": 5, "context": 3, "reference": 4})
        self.assertIn("verified", MM_CONDITIONS)
        self.assertGreater(rows["context"]["findings"], rows["text"]["findings"])
        self.assertFalse(rows["context"]["context_named_diagnosis"])
        self.assertAlmostEqual(rows["context"]["context_recall"], 1 / 3)   # low ALP; not hyperammonemia, ectopia lentis
        named = medpix_context_case(self.case(), ["context"], FakeVision(names_diagnosis=True), self.r, self.ext)[0]
        self.assertTrue(named["context_named_diagnosis"])

    def test_check_keeps_only_confirmed_findings(self):
        from hoofbeats.__main__ import image_agreement
        from hoofbeats.context import medpix_context_case
        from hoofbeats.llm import VERIFY_SYSTEM, image_context
        conds = ["text", "context", "verified"]
        no = FakeVision(finding="Low alkaline phosphatase", details="", confirm="no")
        rows = {x["condition"]: x for x in medpix_context_case(self.case(), conds, no, self.r, self.ext,
                                                                agreement=image_agreement)}
        self.assertEqual([s == VERIFY_SYSTEM for s, *_ in no.calls], [False, True, False, True])   # describe, check
        self.assertIn("1. Low alkaline phosphatase - location: right globe", no.calls[1][1])
        self.assertEqual(rows["context"]["rank"], 3)                       # the description is used as written
        self.assertEqual(rows["verified"]["rank"], rows["text"]["rank"])   # nothing confirmed: back to the text alone
        self.assertNotIn("alkaline", rows["verified"]["context"].lower())
        self.assertIn("No orbital fracture", rows["verified"]["context"])  # pertinent negatives are kept
        self.assertNotIn("Summary", rows["verified"]["context"])           # the summary restates dropped findings
        self.assertEqual((rows["verified"]["findings_checked"], rows["verified"]["findings_kept"]), (2, 0))
        yes = FakeVision(finding="Low alkaline phosphatase", details="", confirm="yes")
        rows = {x["condition"]: x for x in medpix_context_case(self.case(), conds, yes, self.r, self.ext)}
        self.assertEqual(rows["verified"]["rank"], rows["context"]["rank"])
        self.assertEqual(rows["verified"]["findings_kept"], 2)
        unsure = FakeVision(confirm="unsure")
        unsure.verify = True                                               # --context-verify, as diagnose uses it
        parts = image_context(unsure, self.case(), 1)
        self.assertEqual((parts[0].checked, parts[0].kept), (1, 0))        # only "yes" keeps a finding
        self.assertIn("Ectopia lentis", parts[0].unchecked)
        self.assertNotIn("Ectopia lentis", parts[0].text)
        self.assertEqual(len(image_context(FakeVision(), self.case(), 1)), 1)   # no check unless asked

    def test_commands_dry_run(self):
        import argparse, contextlib, io, os
        import hoofbeats.__main__ as hb
        from hoofbeats.__main__ import RUN, main
        from hoofbeats.context import context_llm
        out, buf = self.root / "results", io.StringIO()
        llm = ["--kb", str(self.kb), "--common", "none", "--dry-run", "--json-mode", "--context-model", "m-vision",
               "--context-endpoint", "http://127.0.0.1:9/v1/chat/completions"]
        with contextlib.redirect_stdout(buf):
            main(llm + ["medpix-context", "--medpix", str(self.mp), "--split", "all", "--out", str(out)])
            main(llm + ["multimodal", "--text", "Knee pain.", "--age", "34", "--sex", "female",
                        "--image", str(self.mp / "images" / "MPX1_a.png"), "--image", str(self.mp / "images" / "MPX1_b.png"),
                        "--modality", "X-ray", "--out", str(out)])
        text = buf.getvalue()
        self.assertIn("would send 2 requests to m-vision", text)      # MPX1 (labelled): 2 images; MPX2 has no KB diagnosis
        labels = self.root / "labels.csv"
        labels.write_text("U_id,truth\nMPX2,OMIM:1\nMPX9,OMIM:2\n")
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            main(llm + ["medpix-context", "--medpix", str(self.mp), "--labels", str(labels), "--labels-only",
                        "--out", str(out)])
        self.assertIn("skipped (not found or no images on disk): MPX9", buf2.getvalue())
        self.assertIn("1 cases with images", buf2.getvalue())          # only the listed MPX2, with its label
        self.assertEqual(text.count("would send 2 requests to m-vision"), 2)   # ... and the multimodal patient's 2 images
        saved = json.loads((out / "multimodal" / "patient.json").read_text())
        self.assertEqual((saved["policy"], saved["sex"], len(saved["images"])), ("open", "female", 2))
        self.assertTrue((out / "diagnose" / "patient.json").exists())
        self.assertTrue((out / "runs" / "latest.txt").exists())
        self.assertEqual(RUN["settings"]["context_model"], "m-vision")
        ollama = "http://localhost:11434/v1/chat/completions"
        a = argparse.Namespace(llm_endpoint=ollama, llm_key_env=None, llm_extra=None, llm_max_tokens=4096, llm_rpm=None,
                               dry_run=True, llm_timeout=600, json_mode=True, context_model="m2", context_endpoint=None,
                               context_key_env=None, context_extra=None, context_max_tokens=None, llm_cache="none",
                               llm_offline=True, out=str(out))
        json_only = {"response_format": {"type": "json_object"}}
        with contextlib.redirect_stdout(io.StringIO()):
            same = context_llm(hb, a)
            a.context_endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            os.environ["TEST_KEY"] = "k"
            a.context_key_env = "TEST_KEY"
            other = context_llm(hb, a)
        self.assertEqual((same.extra, other.extra), (json_only, {}))   # not sent to Gemini's endpoint
        self.assertTrue(same.offline)                                  # --llm-offline reaches the image-context model
        a.context_model, a.llm_model, a.llm_vision = None, None, "auto"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(context_llm(hb, a).model, "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M")   # the default


def efetch_xml(papers):
    arts = []
    for pmid, title, abstract, year, types in papers:
        pt = "".join(f"<PublicationType UI=\"D0\">{t}</PublicationType>" for t in types)
        arts.append(f"""<PubmedArticle><MedlineCitation Status="MEDLINE" Owner="NLM"><PMID Version="1">{pmid}</PMID>
<Article PubModel="Print"><Journal><Title>Test Journal</Title><JournalIssue><PubDate><Year>{year}</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>{title}</ArticleTitle><Abstract><AbstractText Label="BACKGROUND">Background sentence about something unrelated here.</AbstractText>
<AbstractText Label="RESULTS">{abstract}</AbstractText></Abstract><PublicationTypeList>{pt}</PublicationTypeList></Article>
</MedlineCitation></PubmedArticle>""")
    return ("<?xml version=\"1.0\"?><PubmedArticleSet>" + "".join(arts) + "</PubmedArticleSet>").encode()


class FakeNCBI:
    def __init__(self, ids, papers):
        self.ids, self.papers, self.urls = ids, papers, []

    def __call__(self, url, timeout=30):
        import io, urllib.parse
        self.urls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "esearch" in url:
            return io.BytesIO(json.dumps({"esearchresult": {"idlist": self.ids}}).encode())
        want = q["id"][0].split(",")
        return io.BytesIO(efetch_xml([p for p in self.papers if p[0] in want]))


MAXO_TSV = """#description: test
#version: 2026-01-01
hpo_id\thpo_label\tpredicate_id\tmaxo_id\tmaxo_label\tcreator_id
HP:0001083\tEctopia lentis\tis_observable_through\tMAXO:0000001\tslit-lamp examination\tx
HP:0000478\tAbnormality of the eye\tis_observable_through\tMAXO:0000002\tophthalmologist evaluation\tx
HP:0000098\tTall stature\tis_observable_through\tMAXO:0000487\tclinical assessment\tx
HP:0001166\tArachnodactyly\tis_observable_through\tMAXO:0000487\tclinical assessment\tx
HP:0001250\tSeizure\tis_observable_through\tMAXO:0000003\telectroencephalography\tx
HP:0001263\tGlobal developmental delay\tis_prenatally_observable_through\tMAXO:0000004\tfetal ultrasonography\tx
HP:0003282\tDecreased circulating alkaline phosphatase activity\tis_observable_through\tMAXO:0000005\tblood chemistry measurement\tx
HP:0003282\tDecreased circulating alkaline phosphatase activity\tis_observable_through\tMAXO:0000006\tbiopsy of bone tissue\tx
"""
LOINC_TSV = """loincId\tloincScale\toutcome\thpoTermId\tsupplementalTermId\tcuration\tcomment
6768-6\tQn\tL\tHP:0003282\t.\tx\t.
6768-6\tQn\tH\tHP:0003155\t.\tx\t.
6768-6\tQn\tN\tHP:0001939\t.\tx\t.
"""


GTR_TESTS = [   # accession, current, name, lab, country, methods, genes, MIM numbers of the conditions
    ("GTR000000001.1", "1", "FBN1 Single Gene", "Lab A", "United States",
     "Deletion/duplication analysis|Sequence analysis of the entire coding region", ["FBN1"], ["1"]),
    ("GTR000000002.1", "1", "Alpha and Delta panel", "Lab B", "Germany",
     "Sequence analysis of the entire coding region", ["FBN1", "G2", "G3", "G4"], ["1", "4"]),
    ("GTR000000003.2", "1", "Gamma enzyme activity", "Lab C", "United States", "Enzyme assay", [], ["3"]),
    ("GTR000000004.1", "0", "Old Beta test", "Lab D", "United States", "Sequence analysis of the entire coding region",
     ["OLD1"], ["2"]),
    ("GTR000000005.1", "1", "Alpha carrier screen", "Lab E", "United States", "Targeted variant analysis", ["FBN1"], ["1"]),
    ("GTR000000006.1", "1", "BETA1 sequencing", "Lab F", "United States", "Sequence analysis of the entire coding region",
     ["BETA1"], ["2"]),
]


def write_gtr(kb: Path):
    import gzip
    cols = ["test_accession_ver", "name_of_laboratory", "facility_country", "last_touch_date", "lab_test_name",
            "manufacturer_test_name", "method_categories", "now_current"]
    lines = ["\t".join(cols)] + ["\t".join([a, f'"{lab}"', f'"{country}"', "2026-10-01", f'"{name}"', "", f'"{m}"', cur])
                                  for a, cur, name, lab, country, m, genes, mims in GTR_TESTS]
    with gzip.open(kb / "test_version.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    rows = ["#accession_version\ttest_type\tobject\tGTR_identifier\tMIM_number\tobject_name\tgene_or_SNOMED_CT_ID\tgene_symbol"]
    for a, cur, name, lab, country, m, genes, mims in GTR_TESTS:
        rows += [f"{a}\tClinical\tcondition\tC{x}\t{x}\tcondition {x}\t\tN/A" for x in mims]
        rows += [f"{a}\tClinical\tgene\tC9{g}\tN/A\t{g}:gene\t1\t{g}" for g in genes]
    (kb / "test_condition_gene.txt").write_text("\n".join(rows) + "\n")


class TestWorkup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hoofbeats.workup import GTR, TestCatalog, Workupper
        cls.root = Path(tempfile.mkdtemp())
        cls.kb = build(cls.root)
        (cls.kb / "maxo_diagnostic_annotations.tsv").write_text(MAXO_TSV)
        (cls.kb / "loinc2hpo-annotations.tsv").write_text(LOINC_TSV)
        write_gtr(cls.kb)
        cls.ont = Ontology(cls.kb / "hp.obo")
        cls.r = LRRanker(cls.ont, AnnotationDB(cls.kb / "phenotype.hpoa", cls.ont))
        cls.cat = TestCatalog(cls.ont, cls.kb)
        cls.gtr = GTR.load(cls.kb)
        cls.wk = Workupper(cls.r, cls.cat, genes_path=cls.kb / "genes_to_disease.txt", gtr=cls.gtr)

    def test_gtr(self):
        from hoofbeats.workup import verify_gtr, workup_json
        g = self.gtr
        self.assertEqual(sorted(g.tests), ["GTR000000001.1", "GTR000000002.1", "GTR000000003.2", "GTR000000005.1",
                                           "GTR000000006.1"])                       # the old version is left out
        self.assertEqual({a: t.kind for a, t in g.tests.items()},
                         {"GTR000000001.1": "targeted", "GTR000000002.1": "panel", "GTR000000003.2": "biochemical",
                          "GTR000000005.1": "screening", "GTR000000006.1": "targeted"})
        self.assertEqual(g.tests["GTR000000001.1"].url, "https://www.ncbi.nlm.nih.gov/gtr/tests/1/")
        self.assertEqual(g.version, "2026-10-01")
        self.assertEqual(self.wk.gene_list("OMIM:1"), (["FBN1"], "genes_to_disease"))
        self.assertEqual(self.wk.gene_list("OMIM:2"), (["BETA1"], "GTR"))        # no HPO gene: GTR's targeted tests
        c = Case("x", P("HP:0000098"), ["OMIM:1"])
        w = self.wk.build(c, self.r.score(c), k=4)
        alpha = next(d for d in w.diseases if d.disease == "OMIM:1")
        self.assertEqual(alpha.gtr["tests"], 3)
        self.assertEqual([t.name for t in alpha.gtr["targeted"]], ["FBN1 Single Gene"])
        self.assertEqual([x["covers"] for x in w.gtr_panels], [["OMIM:1", "OMIM:4"]])
        self.assertEqual([t.name for t in g.examples("OMIM:3", "biochemical")], ["Gamma enzyme activity"])
        self.assertTrue(verify_gtr(self.wk, "OMIM:1", g.tests["GTR000000001.1"]))
        self.assertFalse(verify_gtr(self.wk, "OMIM:2", g.tests["GTR000000001.1"]))
        saved = workup_json(w)
        self.assertEqual(saved["gtr_panels"][0]["accession"], "GTR000000002.1")
        self.assertEqual(next(d for d in saved["diseases"] if d["id"] == "OMIM:1")["gtr"]["targeted"][0]["lab"], "Lab A")

    def test_gene_method_and_broad_test(self):
        from hoofbeats.workup import BROAD_FIRST, GtrTest, workup_json
        g = self.gtr
        saved_tests, saved_by = dict(g.tests), {k: list(v) for k, v in g.by_disease.items()}
        try:     # three repeat-expansion tests for OMIM:3 (GTR names them so): its gene test becomes a repeat test
            for i, name in enumerate(["GAMMA CAG Repeat Expansion", "Gamma triplet repeat analysis", "Gamma disease"]):
                acc = f"GTR00000009{i}.1"
                g.tests[acc] = GtrTest(acc, name, f"Lab {i}", "United States", ("Targeted variant analysis",),
                                       ("GAMMA",), frozenset({"OMIM:3"}))
                g.by_disease["OMIM:3"].append(acc)
            self.assertEqual((g.method("OMIM:3"), g.method("OMIM:1")), ("repeat", "sequencing"))
            c = Case("x", P("HP:0000098"), ["OMIM:1"])
            w = self.wk.build(c, self.r.score(c), k=4, shown=["OMIM:1", "OMIM:3"])
            gene3 = next(d for d in w.diseases if d.disease == "OMIM:3").gene
            self.assertIn("repeat expansion testing", gene3.test.name)
            self.assertIn("sequencing and deletion/duplication", next(d for d in w.diseases if d.disease == "OMIM:1").gene.test.name)
            b = workup_json(w)["broad_test"]
            self.assertTrue(b["test"].startswith("genome sequencing"))        # a shown disease needs repeat detection
            self.assertEqual(b["first"], b["p_in_list"] < BROAD_FIRST)
            self.assertEqual(b["repeat_diseases"], ["OMIM:3"])
        finally:
            g.tests.clear(); g.tests.update(saved_tests)
            g.by_disease.clear(); g.by_disease.update(saved_by)
        c = Case("x", P("HP:0000098"), ["OMIM:1"], genes=["FBN1"])
        self.assertIsNone(self.wk.build(c, self.r.score(c), k=4).broad)     # a gene result is already known

    def test_negative_gene_result(self):
        r = LRRanker(self.ont, AnnotationDB(self.kb / "phenotype.hpoa", self.ont, ("OMIM",), self.kb / "genes_to_disease.txt"))
        c = Case("x", P("HP:0000098"), ["OMIM:1"])
        neg = Case("x", P("HP:0000098"), ["OMIM:1"], meta={"genes_negative": ["FBN1"]})
        i = r.d_index["OMIM:1"]
        self.assertAlmostEqual(r.score(neg).log_scores[i] - r.score(c).log_scores[i], math.log(0.1))
        others = [j for j in range(len(r.diseases)) if j != i]
        self.assertTrue(np.allclose(r.score(neg).log_scores[others], r.score(c).log_scores[others]))

    def names(self, term):
        return sorted(self.cat.tests[t].name for t in self.cat.observing(term))

    def test_catalog(self):
        self.assertEqual(self.names("HP:0001083"), ["ophthalmologist evaluation", "slit-lamp examination"])  # + parent's
        self.assertEqual(self.names("HP:0007359"), ["electroencephalography"])     # nearest annotated ancestor
        self.assertEqual(self.names("HP:0001263"), [])                             # prenatal ascertainment left out
        # a LOINC lab test replaces MAxO's generic lab action; the biopsy stays. Its "normal" term (Abnormality of
        # metabolism/homeostasis) is too general to name it, so it is named after a result term
        lab = ("lab test: Decreased circulating alkaline phosphatase activity / Elevated circulating alkaline "
               "phosphatase concentration (LOINC test)")
        self.assertEqual(self.names("HP:0003282"), ["biopsy of bone tissue", lab])
        self.assertEqual(self.names("HP:0001987"), [])          # not shown by the lab test via the general term
        burden = {t.name: (t.kind, t.burden) for t in self.cat.tests.values()}
        self.assertEqual(burden["clinical assessment"], ("exam", 1))
        self.assertEqual(burden["slit-lamp examination"], ("exam", 2))
        self.assertEqual(burden[lab], ("lab", 2))
        self.assertEqual(burden["electroencephalography"], ("physiology", 3))
        self.assertEqual(burden["biopsy of bone tissue"], ("invasive", 5))
        self.assertNotIn("fetal ultrasonography", burden)
        self.assertEqual(self.cat.done_from_text(["Slit-lamp: lens normal"]), {"MAXO:0000001"})

    def test_formulas(self):
        from hoofbeats.workup import ranker_probs, score_test
        lr_pos, lr_neg, post_pos, post_neg, bits = score_test(0.5, 0.9, 0.1)
        self.assertAlmostEqual(lr_pos, 9.0)
        self.assertAlmostEqual(lr_neg, 0.1 / 0.9)
        self.assertAlmostEqual(post_pos, 0.9)
        self.assertAlmostEqual(post_neg, 0.1)
        self.assertAlmostEqual(bits, 1 + 0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
        rk = self.r.score(Case("x", P("HP:0000098")))
        p = ranker_probs(rk)
        self.assertAlmostEqual(p.sum(), 1.0)          # every disease is in the top 20 here: no outside share
        self.assertEqual(int(np.argmax(p)), rk.order[0])

    def test_workup_of_a_disease(self):
        from hoofbeats.workup import Shown, verify_result
        c = Case("x", P("HP:0000098"), ["OMIM:1"], meta={"sex": "male"})
        w = self.wk.build(c, self.r.score(c), k=2)
        dw = w.diseases[0]
        self.assertEqual(dw.disease, "OMIM:1")
        tests = {x.test.name: x for x in dw.ranked}
        self.assertIn("slit-lamp examination", tests)
        self.assertEqual([s.name for s in tests["clinical assessment"].shows], ["Arachnodactyly"])   # tall stature known
        x = tests["slit-lamp examination"]
        self.assertGreater(x.lr_pos, 1)
        self.assertGreater(x.post_pos, dw.p)
        self.assertLess(x.post_neg, dw.p)
        self.assertGreater(x.woe_confirm, 0)
        self.assertEqual(dw.gene.test.codes, ("FBN1",))
        self.assertEqual(w.panel["genes"][0], "FBN1")
        self.assertTrue(all(verify_result(self.wk, dw.disease, y)[0] for y in dw.ranked + [dw.gene]))
        bad = Shown("HP:0001083", "Ectopia lentis", 0.5, ("PMID:999",))
        x.shows = [bad]
        self.assertFalse(verify_result(self.wk, dw.disease, x)[0])     # frequency and reference not in the KB

    def test_known_gene_done_tests_and_unsure_findings(self):
        c = Case("x", P("HP:0000098") + [Finding("HP:0001166", Status.UNCERTAIN)], ["OMIM:1"], ["FBN1"],
                 meta={"done_texts": ["Slit lamp (eye clinic): lenses in place"]})
        w = self.wk.build(c, self.r.score(c), k=1)
        dw = w.diseases[0]
        self.assertIsNone(dw.gene)
        self.assertEqual(dw.genes_known, ["FBN1"])
        self.assertEqual([x.test.name for x in dw.review], ["slit-lamp examination"])   # done: review, not order
        self.assertNotIn("slit-lamp examination", [x.test.name for x in dw.ranked])
        self.assertEqual(w.done, ["slit-lamp examination"])
        self.assertEqual(w.verify[0]["term"], "Arachnodactyly")
        self.assertEqual(w.verify[0]["tests"], ["clinical assessment"])

    def test_displayed_diseases_get_tests_in_their_order(self):
        rk = self.r.score(Case("x", P("HP:0001263")))
        top2 = [d for d, *_ in rk.top(2)]
        last = [d for d, *_ in rk.top(len(self.r.diseases))][-1]          # a disease outside the ranker's top
        shown = [top2[1], top2[0], last, "OMIM:999999"]                   # e.g. a re-ranker's order; unknown id
        w = self.wk.build(Case("x", P("HP:0001263")), rk, k=2, shown=shown)
        self.assertEqual([d.disease for d in w.diseases], shown[:3])      # order kept, unknown id skipped
        self.assertEqual([d.disease for d in self.wk.build(Case("x", P("HP:0001263")), rk, k=2).diseases], top2)

    def test_note_may_already_answer_and_tests_that_tell_apart(self):
        from hoofbeats.workup import mentions, workup_json
        c = Case("x", P("HP:0000098"), meta={"history": "Exam: dislocated lens on the left. Flat feet bilat."})
        self.assertEqual(mentions(self.ont, c, {"HP:0001083"}), {"HP:0001083": "Exam: dislocated lens on the left."})
        self.assertEqual(mentions(self.ont, Case("y", [], meta={"history": "Lens is clear."}), {"HP:0001083"}), {})
        w = self.wk.build(c, self.r.score(c), k=3)
        hinted = [m for d in w.diseases for x in d.ranked + d.review for m in x.mentioned]
        self.assertIn({"finding": "Ectopia lentis", "sentence": "Exam: dislocated lens on the left."}, hinted)
        fam = Case("z", P("HP:0000098") + [Finding("HP:0001083", Status.FAMILY)], meta=c.meta)   # already recorded
        self.assertFalse([m for d in self.wk.build(fam, self.r.score(fam), k=3).diseases
                          for x in d.ranked + d.review for m in x.mentioned if m["finding"] == "Ectopia lentis"])
        self.assertTrue(w.separate)                                         # tests that tell the top 3 apart
        top = w.separate[0]
        self.assertEqual(set(top["p_pos"]), {d.disease for d in w.diseases[:3]})
        self.assertGreater(max(top["p_pos"].values()) - min(top["p_pos"].values()), 0.1)
        saved = workup_json(w)
        self.assertEqual(saved["separate"][0]["test"], top["test"])
        self.assertIn("may_already_be_answered", saved["diseases"][0]["tests"][0])

    def test_diagnose_gives_tests_per_displayed_disease(self):
        import contextlib, io, shutil
        from hoofbeats.__main__ import main
        out, case = self.root / "dx", self.root / "pps" / "FBN1" / "PMID_100_Patient_1.json"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            main(["--kb", str(self.kb), "--common", "none", "diagnose", str(case), "-k", "3", "--out", str(out)])
        saved = json.loads((out / "diagnose" / "PMID_100_Patient_1.json").read_text())
        self.assertEqual(list(saved["tests"]["diseases"]), [x["id"] for x in saved["top"]])   # one entry per disease
        first = saved["tests"]["diseases"][saved["top"][0]["id"]]
        self.assertTrue({"pre_test", "tests", "gene_test", "review"} <= first.keys())
        self.assertIn("Tests per disease", buf.getvalue())
        bare = self.root / "kb_bare"                                      # no test data: diagnose still runs
        shutil.copytree(self.kb, bare, ignore=shutil.ignore_patterns("maxo*", "loinc*"))
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            main(["--kb", str(bare), "--common", "none", "diagnose", str(case), "--out", str(out / "bare")])
        self.assertIn("test suggestions off", buf.getvalue())
        self.assertIsNone(json.loads((out / "bare" / "diagnose" / "PMID_100_Patient_1.json").read_text())["tests"])

    def test_simulation_and_commands(self):
        import contextlib, io
        from hoofbeats.__main__ import main
        from hoofbeats.workup import disease_eval, reveal, simulate_case
        pp = load_phenopackets(self.root / "pps")[0]
        found, genes = reveal(self.wk, "MAXO:0000001", P("HP:0001083", "HP:0000098"), ["FBN1"])
        self.assertEqual([f.hpo_id for f in found], ["HP:0001083"])
        self.assertEqual(reveal(self.wk, "GENE:FBN1", [], ["FBN1"])[1], ["FBN1"])
        rows = simulate_case(self.wk, pp, frozenset(), "workup", 2, 13, (2.18, 0.475), False)
        self.assertEqual([x["step"] for x in rows], [0, 1, 2])
        self.assertTrue(all(0 <= x["p_truth"] <= 1 for x in rows))
        ev = disease_eval(self.wk, pp, frozenset(), 13, (2.18, 0.475), budget=6)
        self.assertEqual({x["world"] for x in ev}, {"open", "imputed"})
        self.assertTrue(all(x["burden"] <= 6 for x in ev))
        out = self.root / "results"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            main(["--kb", str(self.kb), "--common", "none", "workup", str(self.root / "pps" / "FBN1" / "PMID_100_Patient_1.json"),
                  "-k", "2", "--out", str(out)])
            main(["--kb", str(self.kb), "--common", "none", "workup-eval", "--phenopackets", str(self.root / "pps"),
                  "--calibration-cases", "0", "--sample", "5", "--steps", "1", "--out", str(out / "eval")])
        text = buf.getvalue()
        self.assertIn("Workup for PMID_100_Patient_1", text)
        self.assertIn("suggested tests verified", text)
        saved = json.loads((out / "workup" / "PMID_100_Patient_1.json").read_text())
        self.assertEqual(saved["diseases"][0]["id"], "OMIM:1")
        self.assertTrue((out / "eval" / "workup_eval.csv").exists())


def efetch_xml(papers):
    arts = []
    for pmid, title, abstract, year, types in papers:
        pt = "".join(f"<PublicationType UI=\"D0\">{t}</PublicationType>" for t in types)
        arts.append(f"""<PubmedArticle><MedlineCitation Status="MEDLINE" Owner="NLM"><PMID Version="1">{pmid}</PMID>
<Article PubModel="Print"><Journal><Title>Test Journal</Title><JournalIssue><PubDate><Year>{year}</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>{title}</ArticleTitle><Abstract><AbstractText Label="BACKGROUND">Background sentence about something unrelated here.</AbstractText>
<AbstractText Label="RESULTS">{abstract}</AbstractText></Abstract><PublicationTypeList>{pt}</PublicationTypeList></Article>
</MedlineCitation></PubmedArticle>""")
    return ("<?xml version=\"1.0\"?><PubmedArticleSet>" + "".join(arts) + "</PubmedArticleSet>").encode()


class FakeNCBI:
    def __init__(self, ids, papers):
        self.ids, self.papers, self.urls = ids, papers, []

    def __call__(self, url, timeout=30):
        import io, urllib.parse
        self.urls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "esearch" in url:
            return io.BytesIO(json.dumps({"esearchresult": {"idlist": self.ids}}).encode())
        want = q["id"][0].split(",")
        return io.BytesIO(efetch_xml([p for p in self.papers if p[0] in want]))


class TestPubMed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp())
        kb = build(root)
        cls.root = root
        cls.ont = Ontology(kb / "hp.obo")
        cls.db = AnnotationDB(kb / "phenotype.hpoa", cls.ont, ("OMIM",))
        cls.ext = Extractor(cls.ont)
        cls.case = Case("PMID_100_P1", P("HP:0000098", "HP:0001083"), ["OMIM:1"], source_pmids=["PMID:100"])
        cls.papers = [
            ("11", "Disease Alpha in three families", "Patients had tall stature and ectopia lentis. Other text follows here.", 2015, ["Journal Article"]),
            ("12", "Retracted report of Disease Alpha", "All patients had tall stature with ectopia lentis.", 2016, ["Retracted Publication"]),
            ("100", "Disease Alpha case (the test case)", "Our patient had tall stature and ectopia lentis.", 2018, ["Case Reports"]),
            ("13", "Unrelated cardiology paper", "Tall stature was common in athletes in this cohort.", 2017, ["Journal Article"])]

    def pm(self, ids=("11", "12", "100", "13")):
        from hoofbeats.pubmed import PubMed
        return PubMed(self.root / "pm_cache.json", opener=FakeNCBI(list(ids), self.papers))

    def test_query_uses_only_labels(self):
        from hoofbeats.pubmed import build_query
        c = Case("x", [Finding("HP:0007359", Status.PRESENT, "note", "SECRET PATIENT TEXT")])
        q = build_query(self.ont, c, "Disease Beta (DB)")
        self.assertIn('"Disease Beta"[tiab]', q)
        self.assertIn("Focal-onset seizure", q)
        self.assertIn("NOT retracted publication[pt]", q)
        self.assertNotIn("SECRET", q)

    def test_literature_claims_filter_and_verify(self):
        from hoofbeats.pubmed import literature_claims, verify_lit
        pm = self.pm()
        claims = literature_claims(pm, self.ont, self.ext, self.case, "OMIM:1", "Disease Alpha", frozenset({"PMID:100"}))
        self.assertEqual([c.pmid for c in claims], ["11"])           # retracted, own paper, off-topic excluded
        c = claims[0]
        self.assertEqual(c.hpo_ids, ("HP:0000098", "HP:0001083"))
        self.assertIn("PMID:11 (2015)", c.text)
        self.assertTrue(all(u.startswith("https://eutils.ncbi.nlm.nih.gov/") for u in pm.opener.urls))
        self.assertTrue(verify_lit(pm, self.ont, self.ext, self.case, c, "Disease Alpha", frozenset({"PMID:100"}))[0])
        c.sentence = "Patients had tall stature and ectopia lentis and a made-up sign."
        self.assertFalse(verify_lit(pm, self.ont, self.ext, self.case, c, "Disease Alpha")[0])

    def test_cache_and_policy(self):
        from hoofbeats.pubmed import PubMed, PubMedPolicyError, literature_claims, parse_pubmed_xml
        pm = self.pm()
        literature_claims(pm, self.ont, self.ext, self.case, "OMIM:1", "Disease Alpha")
        pm.save()
        again = PubMed(self.root / "pm_cache.json", opener=FakeNCBI([], []))
        literature_claims(again, self.ont, self.ext, self.case, "OMIM:1", "Disease Alpha")
        self.assertEqual(again.calls, 0)                                # served from cache
        with self.assertRaises(PubMedPolicyError):
            literature_claims(pm, self.ont, self.ext, Case("m", [], meta={"policy": "local_only"}), "OMIM:1", "Disease Alpha")
        p = parse_pubmed_xml(efetch_xml([self.papers[0]]))[0]
        self.assertEqual((p.pmid, p.year, p.journal), ("11", 2015, "Test Journal"))
        self.assertIn("Patients had tall stature and ectopia lentis.", p.sentences)

    def test_rerank_can_cite_literature(self):
        from hoofbeats.pubmed import literature_claims
        r = LRRanker(self.ont, self.db)
        rk = r.score(self.case)
        lit = {"OMIM:1": literature_claims(self.pm(), self.ont, self.ext, self.case, "OMIM:1", "Disease Alpha",
                                           frozenset({"PMID:100"}))}
        reply = json.dumps({"ranking": ["OMIM:1"], "explanations": {
            "OMIM:1": "Patients with Disease Alpha had tall stature and ectopia lentis [L1]."}})
        llm = Fake(reply)
        res = rerank(llm, r, self.case, rk, k=3, literature=lit)
        self.assertNotIn("PMID:100 (2018)", llm.prompt)
        self.assertIn("[L1] PubMed PMID:11", llm.prompt)
        self.assertEqual(res.faithfulness, 1.0)


def image_prompt_of(case):
    from hoofbeats.llm import image_prompt
    return image_prompt(case.meta["images"][0])


EXTRA_TERMS = "".join(f"[Term]\nid: {i}\nname: {n}\n{s}is_a: HP:0000118\n\n" for i, n, s in (
    ("HP:0100749", "Chest pain", ""), ("HP:0002018", "Nausea", ""), ("HP:0002013", "Vomiting", ""),
    ("HP:0001653", "Mitral regurgitation", ""), ("HP:0030828", "Wheezing", ""),
    ("HP:0030212", "Collectionism", 'synonym: "Collecting" EXACT []\n'), ("HP:0012115", "Hepatitis", ""),
    ("HP:0002666", "Pheochromocytoma", ""), ("HP:9000001", "Jumping", 'synonym: "Hopping" EXACT []\n'),
    ("HP:9000002", "Atypical sorting", 'synonym: "Ordering" EXACT []\n'),
    ("HP:0033106", "Elevated circulating D-dimer concentration", 'synonym: "Elevated D dimer" EXACT []\n'),
    ("HP:0011705", "First degree atrioventricular block", "")))


class TestExtractorRules(unittest.TestCase):
    # text reader rules, on short sentences (base mini-ontology plus EXTRA_TERMS)
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        build(cls.root)
        (cls.root / "kb" / "hp.obo").write_text(OBO.replace("[Typedef]", EXTRA_TERMS + "[Typedef]"))
        cls.ont = Ontology(cls.root / "kb" / "hp.obo")
        cls.ext = Extractor(cls.ont)

    def st(self, text):
        return {(f.hpo_id, f.status) for f in self.ext.findings(text)}

    def test_patient_and_family_status_kept_apart(self):
        self.assertEqual(self.st("No seizures. Mother had seizures."),
                         {("HP:0001250", Status.ABSENT), ("HP:0001250", Status.FAMILY)})

    def test_sentence_split_keeps_abbreviations(self):
        from hoofbeats.data import sentence_spans
        cut = lambda t: [t[a:b] for a, b in sentence_spans(t)]
        self.assertEqual(cut("Seen by Dr. Smith. No seizures."), ["Seen by Dr. Smith.", "No seizures."])
        self.assertEqual(cut("Aunt also v. tall, e.g. 6 ft. Temp 38.5 today. Ref No. 5 noted"),
                         ["Aunt also v. tall, e.g. 6 ft.", "Temp 38.5 today.", "Ref No. 5 noted"])
        self.assertEqual(cut("Problems:\n- seizures\n- dislocated lens\nmild"), ["Problems:", "- seizures", "- dislocated lens\nmild"])
        text = "Mother also v. tall with dislocated lens."
        self.assertEqual(self.st(text), {("HP:0001083", Status.FAMILY)})       # not cut at "v."
        m = self.ext.mentions(text)[0]
        self.assertEqual((m.quote, text[m.start:m.end]), ("dislocated lens", "dislocated lens"))

    def test_section_headings(self):
        self.assertEqual(self.st("FHx: seizures.\nExam: dislocated lens."),
                         {("HP:0001250", Status.FAMILY), ("HP:0001083", Status.PRESENT)})
        self.assertEqual(self.st("FH: seizures"), {("HP:0001250", Status.FAMILY)})        # known 2-letter heading
        self.assertEqual(self.st("HR: 80, seizures today"), {("HP:0001250", Status.PRESENT)})   # unknown 2-letter: not a heading
        self.assertEqual(self.st("Medications: seizure prophylaxis\nPMH: dislocated lens"), {("HP:0001083", Status.PRESENT)})
        m = self.ext.mentions("Family Hx: seizures")[0]
        self.assertEqual((m.section, m.cue), ("family", ("Family Hx", 0, 9)))

    def test_unknown_line_headings_are_short_and_not_family(self):
        from hoofbeats.data import sections
        self.assertEqual([s[2] for s in sections("Mother with seizures   Pt: dislocated lens") if s[2]], [])
        self.assertEqual(self.st("Mother with seizures   Pt: dislocated lens"),
                         {("HP:0001250", Status.FAMILY), ("HP:0001083", Status.PRESENT)})
        self.assertEqual(self.st("Mother: seizures"), {("HP:0001250", Status.FAMILY)})          # family word is not a heading
        self.assertEqual([s[2] for s in sections("Slit lamp exam: lens\nOne two three four five: x") if s[2]], ["slit lamp exam"])
        self.assertEqual([s[3] for s in sections("Family history of note: seizures") if s[2]], ["family"])   # known heading

    def test_family_word_reaches_only_its_clause(self):
        self.assertEqual(self.st("Mother had seizures, pt has dislocated lens."),
                         {("HP:0001250", Status.FAMILY), ("HP:0001083", Status.PRESENT)})
        self.assertEqual(self.st("Seizures in her brother but dislocated lens noted."),
                         {("HP:0001250", Status.FAMILY), ("HP:0001083", Status.PRESENT)})

    def test_negation_scope(self):
        self.assertEqual(self.st("No fever, presents with seizures."),
                         {("HP:0001945", Status.ABSENT), ("HP:0001250", Status.PRESENT)})       # termination word
        self.assertEqual(self.st("Denies fever, seizures or dislocated lens."),
                         {("HP:0001945", Status.ABSENT), ("HP:0001250", Status.ABSENT), ("HP:0001083", Status.ABSENT)})
        self.assertEqual(self.st("Denies fever but dislocated lens on exam."),
                         {("HP:0001945", Status.ABSENT), ("HP:0001083", Status.PRESENT)})       # contrast word
        self.assertEqual(self.st("No change in seizures."), {("HP:0001250", Status.PRESENT)})  # pseudo-negation
        self.assertEqual(self.st("No family history of seizures."), set())                        # negated relative: dropped
        m = next(m for m in self.ext.mentions("Denies fever, seizures.") if m.hpo_id == "HP:0001250")
        self.assertEqual(m.cue, ("Denies", 0, 6))

    def test_related_synonyms_behind_flag(self):
        m = self.ext.mentions("He has fits at night.")[0]          # "Fits" is a RELATED synonym of Seizure
        self.assertEqual((m.hpo_id, m.link, m.grounding), ("HP:0001250", ("synonym:related", "Fits"), "weak"))
        self.assertNotIn("Fits", self.ont.terms["HP:0001250"].synonyms)          # EXACT list unchanged
        exact_only = Extractor(Ontology(self.ont.path, extra_synonyms=False))
        self.assertEqual(exact_only.mentions("He has fits at night."), [])
        self.assertEqual(exact_only.vocab, self.ext.vocab)                        # de-labelling vocabulary unchanged

    def test_lemma_matching(self):
        from hoofbeats.data import _spacy
        if _spacy() is None:
            self.skipTest("spaCy model not installed")
        text = "There are bilateral wheezes."                 # spaCy: NOUN
        m = self.ext.mentions(text)[0]
        self.assertEqual((m.hpo_id, m.quote, text[m.start:m.end], m.link, m.grounding),
                         ("HP:0030828", "wheezes", "wheezes", ("lemma", "Wheezing"), "supported"))
        self.assertEqual(Extractor(self.ont, lemmas=False).mentions(text), [])
        for verb in ("He jumped from his vehicle.", "A chest X-ray was ordered.", "She is hoping to gain control."):
            self.assertEqual(self.ext.mentions(verb), [], verb)                   # verbs never match by lemma
        self.assertTrue(list((self.root / "kb" / ".cache").glob("lemmas-*.json")))     # label lemmas cached

    def test_abbreviations_keep_original_offsets(self):
        text = "Denies CP, N/V. Echo: mild MR."
        got = {(m.hpo_id, m.status, m.quote, m.link[0]) for m in self.ext.mentions(text)}
        self.assertEqual(got, {("HP:0100749", Status.ABSENT, "CP", "abbreviation:chest pain"),
                               ("HP:0002018", Status.ABSENT, "N/V", "abbreviation:nausea and vomiting"),
                               ("HP:0002013", Status.ABSENT, "N/V", "abbreviation:nausea and vomiting"),
                               ("HP:0001653", Status.PRESENT, "MR", "abbreviation:mitral regurgitation")})
        for m in self.ext.mentions(text):
            self.assertEqual(text[m.start:m.end], m.quote)
            self.assertEqual(m.grounding, "weak" if m.quote == "CP" else "supported")   # CP (also cerebral palsy) alone in a list
        self.assertEqual(self.ext.mentions("MR scan of the brain."), [])          # MR needs a cardiac context word
        m = self.ext.mentions("r/o seizure")[0]
        self.assertEqual((m.status, m.cue), (Status.UNCERTAIN, ("r/o", 0, 3)))     # expansion feeds the hedge cue
        self.assertEqual(self.st("Fever w/o seizures"), {("HP:0001945", Status.PRESENT), ("HP:0001250", Status.ABSENT)})
        self.assertEqual(Extractor(self.ont, abbreviations=False).mentions("Denies CP."), [])

    def test_ambiguous_abbreviation_guesses_are_weak(self):
        level = lambda t: {m.quote: m.grounding for m in self.ext.mentions(t)}
        got = level("Hx of IVH, CP, MR, congenital cardiac defects.")       # CP/MR alone in a list of conditions
        self.assertEqual((got["CP"], got["MR"]), ("weak", "weak"))
        self.assertEqual(level("Echo showed mild MR.")["MR"], "supported")    # context word, not a list
        self.assertEqual(level("Echo showed trace TR, MR.")["MR"], "weak")      # in a list, own item has no context
        self.assertEqual(level("Pt has dislocated lens, echo with MR.")["MR"], "supported")   # context in its own item
        self.assertEqual(level("Chest pain since am.")["Chest pain"], "strong")
        found = self.ext.abbr.find("AMLs on renal US; leukemia hx.", 0, 30, {"amls", "on", "renal", "us", "leukemia", "hx"})
        self.assertEqual([x[2:] for x in found if x[2] in ("angiomyolipoma", "acute myeloid leukemia")],
                         [("angiomyolipoma", True)])                          # contexts of two expansions match

    def test_ambiguous_abbreviations_need_general_context(self):
        ambiguous = {"mitral regurgitation", "angiomyolipoma", "ultrasound"}
        exp = lambda s: [x[2] for x in self.ext.abbr.find(s, 0, len(s), {w.lower() for w in re.findall(r"[A-Za-z0-9]+", s)})
                         if x[2] in ambiguous]
        self.assertEqual(exp("Mild MVP, trace MR."), [])                      # "mvp" is not a context word
        self.assertEqual(exp("Echo: mild MVP, trace MR."), ["mitral regurgitation"])
        self.assertEqual(exp("Echogenic fat-containing AMLs."), [])           # nor "echogenic" or "fat"
        self.assertEqual(exp("Both kidneys with AMLs."), ["angiomyolipoma"])
        self.assertEqual(exp("Echogenic foci on US."), [])
        self.assertEqual(exp("Renal US normal."), ["ultrasound"])

    def test_blocklist_and_test_names(self):
        self.assertEqual(self.ext.mentions("No collecting system dilation."), [])     # blocklisted EXACT synonym
        self.assertEqual(self.ext.mentions("Normal hepatitis panel and CBC."), [])    # a test, not a finding
        self.assertEqual(self.st("Hepatitis noted, seizure level unknown."), {("HP:0012115", Status.PRESENT)})
        self.assertEqual(self.st("Has dislocated lens, no seizures."),
                         {("HP:0001083", Status.PRESENT), ("HP:0001250", Status.ABSENT)})   # whole tokens only
        self.assertEqual(self.ext.mentions("Shows unseizureable lens."), [])

    def test_test_names_with_a_result_are_kept(self):
        self.assertEqual(self.st("Elevated D dimer test."), {("HP:0033106", Status.PRESENT)})   # result word in the term
        self.assertEqual(self.st("Hepatitis panel negative."), set())                           # no result word: a test
        self.assertEqual(self.st("Hepatitis panel positive."), {("HP:0012115", Status.PRESENT)})
        self.assertEqual(self.st("Abnormal hepatitis serology."), {("HP:0012115", Status.PRESENT)})

    def test_grounding_of_synonym_kinds(self):
        root = Path(tempfile.mkdtemp())
        (root / "hp.obo").write_text(OBO.replace("[Typedef]", "".join(
            f"[Term]\nid: {i}\nname: {n}\n{s}is_a: HP:0000118\n\n" for i, n, s in (
                ("HP:0001382", "Joint hypermobility", 'synonym: "Double-jointed" EXACT layperson []\n'),
                ("HP:0002094", "Dyspnea", 'synonym: "Shortness of breath" RELATED []\n'),
                ("HP:0000969", "Edema", 'synonym: "Puffiness" BROAD layperson []\n'))) + "[Typedef]"))
        ext = Extractor(Ontology(root / "hp.obo"))
        level = {m.hpo_id: (m.link[0], m.grounding) for m in ext.mentions("Double-jointed, shortness of breath, puffiness.")}
        self.assertEqual(level["HP:0001382"], ("synonym:exact", "strong"))          # EXACT layperson
        self.assertEqual(level["HP:0002094"], ("synonym:related", "weak"))          # RELATED
        self.assertEqual(level["HP:0000969"], ("synonym:layperson", "supported"))   # non-EXACT layperson
        sob = ext.mentions("Pt reports SOB.")[0]                                    # abbreviation onto a RELATED synonym
        self.assertEqual((sob.hpo_id, sob.link[0], sob.grounding), ("HP:0002094", "abbreviation:shortness of breath", "weak"))

    def test_ordinals_and_av(self):
        text = "ECG: sinus rhythm, PR 236 ms (1st degree AV block)."
        m = next(m for m in self.ext.mentions(text) if m.hpo_id == "HP:0011705")
        self.assertEqual((m.status, m.quote, text[m.start:m.end]), (Status.PRESENT, "1st degree AV block", "1st degree AV block"))
        self.assertEqual(m.link[0], "abbreviation:atrioventricular")
        self.assertEqual(self.st("First degree atrioventricular block."), {("HP:0011705", Status.PRESENT)})

    def test_negation_stops_at_a_new_label(self):
        self.assertEqual(self.st("Gen: no fever    NEURO: seizures today"),
                         {("HP:0001945", Status.ABSENT), ("HP:0001250", Status.PRESENT)})
        self.assertEqual(self.st("No fever, seizures."), {("HP:0001945", Status.ABSENT), ("HP:0001250", Status.ABSENT)})
        self.assertEqual(self.st("Seen today; mother with seizures   Pt: dislocated lens"),   # family word stays in its segment
                         {("HP:0001250", Status.FAMILY), ("HP:0001083", Status.PRESENT)})
        root = Path(tempfile.mkdtemp())
        (root / "hp.obo").write_text(OBO.replace("[Typedef]", '[Term]\nid: HP:0030828\nname: Wheezing\n'
                                                 'synonym: "Wheezes" EXACT []\nis_a: HP:0000118\n\n[Typedef]'))
        ext = Extractor(Ontology(root / "hp.obo"))
        got = {(m.hpo_id, m.status) for m in ext.mentions("HEENT: no JVD/HJR    LUNGS: rhales B/L, wheezes B/L")}
        self.assertEqual(got, {("HP:0030828", Status.PRESENT)})

    def test_negated_words(self):
        text = "Pt afebrile, tall."
        m = self.ext.mentions(text)[0]
        self.assertEqual((m.hpo_id, m.status, m.quote, m.link[0], m.grounding), ("HP:0001945", Status.ABSENT, "afebrile", "negated_word", "supported"))
        self.assertEqual(text[m.start:m.end], "afebrile")


class TestGoldFiles(unittest.TestCase):
    # the evaluation gold must match its case files, and accepted alternatives only change on purpose
    ALTERNATIVES = {
        "examples": {("wilson_labs.json", "HP:0000952", "HP:0032106"), ("marfan_note.json", "HP:0001962", "HP:0001649")},
        "examples/heldout": {("fabry_note_labs.json", "HP:0000504", "HP:0000505"), ("fabry_note_labs.json", "HP:0010741", "HP:0000969"),
                             ("fabry_note_labs.json", "HP:0003774", "HP:0000083"), ("dm1_note.json", "HP:0000276", "HP:0000275"),
                             ("dm1_note.json", "HP:0200101", "HP:0001265"), ("dm1_note.json", "HP:0200101", "HP:0002600"),
                             ("dm1_note.json", "HP:0001324", "HP:0007340"), ("hht_imaging.json", "HP:0001009", "HP:0000214"),
                             ("hht_imaging.json", "HP:0001009", "HP:0007428"), ("hht_imaging.json", "HP:0006576", "HP:0006574")}}
    FIELDS = {"note": "note_text", "image_context": "image_context"}

    def test_gold_quotes_and_alternatives(self):
        root = Path(__file__).parent
        for folder, approved in self.ALTERNATIVES.items():
            gold = json.loads((root / folder / "intended_findings.json").read_text())
            alts = set()
            for fn, readers in gold.items():
                if fn.startswith("_"):
                    continue
                case = json.loads((root / folder / fn).read_text())
                for reader, items in readers.items():
                    for it in items:
                        self.assertIn(it["status"], {"present", "absent", "uncertain", "family"})
                        if reader in self.FIELDS and it["quote"]:
                            self.assertIn(it["quote"], case[self.FIELDS[reader]], (fn, it["quote"]))
                        alts |= {(fn, it["hpo_id"], a) for a in it.get("alternatives", [])}
            self.assertEqual(alts, approved, folder)


class TestGrounding(unittest.TestCase):
    # spans, cues, links and grounding levels in provenance; disease mentions
    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp())
        build(root)
        (root / "kb" / "hp.obo").write_text(OBO.replace("[Typedef]", EXTRA_TERMS + "[Typedef]"))
        cls.ont = Ontology(root / "kb" / "hp.obo")
        cls.ext = Extractor(cls.ont)

    def parse(self, obj, **kw):
        from hoofbeats.intake import parse_case
        return parse_case({"case_id": "g", **obj}, self.ont, self.ext, **kw)

    def test_spans_point_into_original_text_after_abbreviations(self):
        note = "Pt denies CP, N/V.\nFHx: szs in mother."
        c = self.parse({"note_text": note})
        cp = c.meta["provenance"]["HP:0100749"][0]
        self.assertEqual((note[cp["start"]:cp["end"]], cp["quote"], cp["field"], cp["status"]), ("CP", "CP", "note_text", "absent"))
        self.assertEqual(cp["link"], {"method": "abbreviation:chest pain", "matched_string": "Chest pain"})
        self.assertEqual((cp["grounding"], cp["readers"], cp["section"]), ("weak", ["rules"], None))   # CP alone in a list
        self.assertEqual(cp["status_cue"], {"word": "denies", "start": 3, "end": 9})
        sz = c.meta["provenance"]["HP:0001250"][0]
        self.assertEqual((sz["quote"], sz["status"], sz["status_cue"]["word"], sz["section"]), ("szs", "family", "FHx", "family"))
        self.assertFalse([w for w in c.meta["intake_warnings"] if w.startswith("provenance")])

    def test_verify_provenance_catches_a_bad_span(self):
        from hoofbeats.grounding import verify_provenance
        note = "Exam: dislocated lens."
        c = self.parse({"note_text": note})
        self.assertEqual(verify_provenance(c, {"note_text": note}), [])
        c.meta["provenance"]["HP:0001083"][0]["start"] += 1
        bad = verify_provenance(c, {"note_text": note})
        self.assertEqual(len(bad), 1)
        self.assertIn("not 'dislocated lens'", bad[0])

    def test_disease_mentions(self):
        names = {"OMIM:10": "Marfan syndrome", "OMIM:11": "Pheochromocytoma", "OMIM:12": "Familial Mediterranean fever"}
        c = self.parse({"note_text": "Pt w/ hx of Marfan. Known pheochromocytoma. Father had Marfan syndrome. "
                                     "Has familial Mediterranean fever."}, disease_names=names)
        got = [(d["quote"], d["disease_ids"], d["status"], d["also_hpo"]) for d in c.meta["disease_mentions"]]
        self.assertEqual(got, [("Marfan", ["OMIM:10"], "present", False), ("pheochromocytoma", ["OMIM:11"], "present", True),
                               ("Marfan syndrome", ["OMIM:10"], "family", False),
                               ("familial Mediterranean fever", ["OMIM:12"], "present", False)])
        ids = {f.hpo_id for f in c.findings}
        self.assertNotIn("HP:0001945", ids)                    # "fever" inside a disease name that is not an HPO term
        self.assertIn("HP:0002666", ids)                       # name that is also an HPO term: kept and flagged
        self.assertTrue(c.meta["provenance"]["HP:0002666"][0]["disease_mention"])
        self.assertFalse([w for w in c.meta["intake_warnings"] if w.startswith("provenance")])
        self.assertIn("HP:0001945", {f.hpo_id for f in self.parse({"note_text": "Has familial Mediterranean fever."}).findings})

    def test_disease_core_names_skip_ordinary_words(self):
        names = {"OMIM:20": "Cousin syndrome", "OMIM:21": "Marfan syndrome", "OMIM:22": "Weaver syndrome",
                 "OMIM:23": "Turner syndrome"}
        c = self.parse({"note_text": "Parents are 2nd cousins. Hx of Marfan. Works as a weaver. Known Turner."},
                       disease_names=names)
        self.assertEqual([d["quote"] for d in c.meta["disease_mentions"]], ["Marfan", "Turner"])
        c = self.parse({"note_text": "Father had Cousin syndrome."}, disease_names=names)
        self.assertEqual([(d["quote"], d["status"]) for d in c.meta["disease_mentions"]], [("Cousin syndrome", "family")])

    def test_numbers_conflict_with_text_and_provenance(self):
        root = Path(tempfile.mkdtemp())
        (root / "hp.obo").write_text(OBO.replace("[Typedef]", "[Term]\nid: HP:0000822\nname: Hypertension\nis_a: HP:0000118\n\n"
                                                 "[Term]\nid: HP:0002615\nname: Hypotension\nis_a: HP:0000118\n\n[Typedef]"))
        from hoofbeats.intake import parse_case
        ont = Ontology(root / "hp.obo")
        c = parse_case({"case_id": "n", "note_text": "Known hypertension. BP 118/76 today.", "vitals": {"bp": "150/95"}}, ont, Extractor(ont),
                       emit_normal_vitals=True)
        prov = c.meta["provenance"]["HP:0000822"]
        self.assertEqual(sorted((e["source"], e["status"], e["readers"][0]) for e in prov),
                         [("note", "absent", "numbers"), ("note", "present", "rules"), ("vitals:bp", "present", "vitals")])
        self.assertEqual(c.meta["provenance"]["conflicts"][0]["hpo_id"], "HP:0000822")   # text/number disagreement recorded
        num = next(e for e in prov if e["readers"] == ["numbers"])
        self.assertEqual((num["quote"], num["range_source"], num["grounding"]), ("BP 118/76", "threshold", "supported"))
        self.assertFalse([x for x in c.meta["intake_warnings"] if x.startswith("provenance")])

    def test_grounding_levels_and_merge(self):
        from hoofbeats.grounding import level_of
        from hoofbeats.intake import intake_summary
        c = self.parse({"note_text": "Exam: dislocated lens. He has fits. Pt afebrile.", "hpo_terms": [{"id": "HP:0001250"}],
                        "labs": [{"name": "Alkaline Phosphatase", "value": 20, "unit": "U/L", "ref_low": 35, "ref_high": 105}]})
        lvl = {f.hpo_id: level_of(f) for f in c.findings}
        self.assertEqual((lvl["HP:0001083"], lvl["HP:0001945"], lvl["HP:0003282"]), ("strong", "supported", "strong"))
        self.assertEqual(lvl["HP:0001250"], "strong")                                   # strongest contributor wins
        self.assertEqual(sorted(e["grounding"] for e in c.meta["provenance"]["HP:0001250"]), ["strong", "weak"])   # RELATED "fits"
        lab = c.meta["provenance"]["HP:0003282"][0]
        self.assertEqual((lab["value"], lab["unit"], lab["ref_low"], lab["ref_high"], lab["range_source"], lab["start"]),
                         (20.0, "U/L", 35.0, 105.0, "input", None))
        self.assertIn("grounding: strong 4, supported 1, weak 0", intake_summary(c, self.ont))


class TestNumbers(unittest.TestCase):
    # numbers reader: outputs before fusion, with the real HPO ids from resources/lab_reference.tsv and vital_thresholds.tsv
    def read(self, text, age=None, **kw):
        from hoofbeats.numbers import read_numbers
        warnings = []
        found = read_numbers(text, "note", "note_text", age, warnings.append, **kw)
        for f in found:
            self.assertEqual(text[f.start:f.end], f.quote)            # spans point at the original characters
        return {(f.hpo_id, f.status) for f in found}, found, warnings

    def test_lab_values_ranges_flags_and_lists(self):
        got, found, _ = self.read("Na 128 mmol/L (ref 135-145), K 5.9 H")
        self.assertIn(("HP:0002902", Status.PRESENT), got)              # hyponatremia
        self.assertIn(("HP:0002153", Status.PRESENT), got)              # hyperkalemia from the H flag
        na = next(f for f in found if f.hpo_id == "HP:0002902")
        self.assertEqual((na.grounding, dict(na.lab)["range_source"], na.quote), ("strong", "input", "Na 128 mmol/L (ref 135-145)"))
        k = next(f for f in found if f.hpo_id == "HP:0002153")
        self.assertEqual((k.grounding, k.status_cue[0]), ("weak", "H"))    # "K" is an ambiguous alias
        got, found, _ = self.read("ALB 2.6 ALK PHOS 202 AST 572 ALT 757 TBILI 0.9")
        self.assertTrue({("HP:0003073", Status.PRESENT), ("HP:0003155", Status.PRESENT), ("HP:0002910", Status.PRESENT),
                         ("HP:0002904", Status.ABSENT)} <= got)
        self.assertEqual({f.grounding for f in found}, {"weak"})          # no units: plausible values, weak

    def test_units(self):
        got, found, w = self.read("Hgb 95 g/L, creatinine 150 umol/L, glucose 3.1 mmol/L, ammonia 80 ug/dL, Na 1280")
        self.assertTrue({("HP:0001903", Status.PRESENT), ("HP:0003259", Status.PRESENT), ("HP:0001943", Status.PRESENT)} <= got)
        self.assertEqual(round(dict(next(f for f in found if f.hpo_id == "HP:0003259").lab)["converted_value"], 2), 1.7)
        self.assertFalse(any(h == "HP:0001987" for h, _ in got))        # ug/dL is not a supported ammonia unit
        self.assertFalse(any(h in ("HP:0002902", "HP:0003228") for h, _ in got))   # Na 1280 without unit: implausible
        self.assertTrue(any("not supported" in x for x in w) and any("not plausible" in x for x in w))

    def test_specimens_and_test_context(self):
        got, _, w = self.read("CSF: LP opening pressure 18, wbc=32, glc=53")
        self.assertEqual(got, set())
        got, _, _ = self.read("Urine glucose 300 mg/dL")
        self.assertEqual(got, set())

    def test_vitals(self):
        got, found, _ = self.read("Vitals: T 101.0F, HR 112, BP 120/95, RR 16, SpO2 88%", emit_normal_vitals=True)
        self.assertTrue({("HP:0001945", Status.PRESENT), ("HP:0001649", Status.PRESENT), ("HP:0000822", Status.PRESENT),
                         ("HP:0002615", Status.ABSENT), ("HP:0002789", Status.ABSENT), ("HP:0012418", Status.PRESENT)} <= got)
        t = next(f for f in found if f.hpo_id == "HP:0001945")
        self.assertEqual((t.grounding, dict(t.lab)["value"]), ("weak", 38.33))   # "T" is a one-letter label
        got, _, _ = self.read("BP 85/50 on arrival.", emit_normal_vitals=True)
        self.assertTrue({("HP:0002615", Status.PRESENT), ("HP:0000822", Status.ABSENT)} <= got)
        got, _, _ = self.read("BP 142/88.")
        self.assertIn(("HP:0000822", Status.PRESENT), got)
        self.assertEqual(self.read("Tested for P 85 deficiency.")[0], set())      # "P" only counts among vitals
        got, _, w = self.read("Vitals: HR 88, O2 sat 995 on RA, BP 70/120", emit_normal_vitals=True)   # typos: skipped with a warning
        self.assertEqual({h for h, _ in got}, {"HP:0001649", "HP:0001662"})
        self.assertEqual(len([x for x in w if "not plausible" in x]), 2)

    def test_normal_vitals_give_no_absent_findings_by_default(self):
        text = "Vitals: T 101.0F, HR 72, BP 120/95, RR 16, SpO2 98%. Na 140 mmol/L (ref 135-145). PR 160 ms."
        normals = []
        got, _, _ = self.read(text, normals=normals)
        self.assertEqual({h for h, st in got if st == Status.PRESENT}, {"HP:0001945", "HP:0000822"})
        self.assertFalse([h for h, st in got if st == Status.ABSENT and h in ("HP:0001649", "HP:0001662", "HP:0002615",
                                                                            "HP:0002789", "HP:0002793", "HP:0012418")])
        self.assertIn(("HP:0002902", Status.ABSENT), got)              # lab with an input range: normal -> absent
        self.assertIn(("HP:0011705", Status.ABSENT), got)              # PR interval is not a vital sign
        self.assertEqual({e["measure"] for e in normals}, {"heart_rate", "respiratory_rate", "spo2", "blood_pressure",
                                                          "temperature_c"})
        hr = next(e for e in normals if e["measure"] == "heart_rate")
        self.assertEqual((text[hr["start"]:hr["end"]], hr["value"], hr["status"]), ("HR 72", 72.0, "absent"))
        on, _, _ = self.read(text, emit_normal_vitals=True)
        self.assertTrue({("HP:0001649", Status.ABSENT), ("HP:0002615", Status.ABSENT)} <= on)

    def test_normal_vitals_in_case_provenance(self):
        from hoofbeats.intake import parse_case
        root = Path(tempfile.mkdtemp())
        extra = "".join(f"[Term]\nid: {i}\nname: {n}\nis_a: HP:0000118\n\n" for i, n in (
            ("HP:0000822", "Hypertension"), ("HP:0002615", "Hypotension"), ("HP:0001649", "Tachycardia"),
            ("HP:0001662", "Bradycardia"), ("HP:0012418", "Hypoxemia")) if f"id: {i}\n" not in OBO)
        (root / "hp.obo").write_text(OBO.replace("[Typedef]", extra + "[Typedef]"))
        ont = Ontology(root / "hp.obo")
        obj = {"case_id": "v", "note_text": "HR 72, BP 118/76.", "vitals": {"bp": "150/95", "hr": 70, "spo2": 97}}
        c = parse_case(obj, ont, Extractor(ont))
        self.assertEqual({(f.hpo_id, f.status) for f in c.findings}, {("HP:0000822", Status.PRESENT)})
        self.assertEqual(len(c.meta["provenance"]["normal_vitals"]), 8)   # HR and BP each give two
        self.assertEqual(c.meta["vitals"], {"bp": "150/95", "hr": 70.0, "spo2": 97.0})
        self.assertFalse([x for x in c.meta["intake_warnings"] if x.startswith("provenance")])
        on = parse_case(obj, ont, Extractor(ont), emit_normal_vitals=True)
        self.assertIn(("HP:0001649", Status.ABSENT), {(f.hpo_id, f.status) for f in on.findings})
        self.assertEqual(on.meta["provenance"]["normal_vitals"], [])

    def test_ecg_zscores_centiles_arm_span(self):
        got, found, _ = self.read("ECG: PR 236 ms. QTc 480 ms. Echo: aortic root 4.1 cm (Z +3.2). Arm span > height. "
                                  "OFC below the 3rd centile; height at the 50th percentile.")
        self.assertTrue({("HP:0011705", Status.PRESENT), ("HP:0005184", Status.PRESENT), ("HP:0002616", Status.PRESENT),
                         ("HP:0012771", Status.PRESENT), ("HP:0000252", Status.PRESENT), ("HP:0000098", Status.ABSENT),
                         ("HP:0004322", Status.ABSENT)} <= got)
        z = next(f for f in found if f.hpo_id == "HP:0002616")
        self.assertEqual(z.quote, "aortic root 4.1 cm (Z +3.2")

    def test_measure_lists_respectively(self):
        text = ("His height, weight, and head circumference were 131.7 cm (<3rd percentile), 48.6 kg (75-90th percentile), "
                "and 55.5 cm (95-97th percentile), respectively.")
        got, found, w = self.read(text)
        self.assertEqual(got, {("HP:0004322", Status.PRESENT), ("HP:0000098", Status.ABSENT),
                               ("HP:0000252", Status.ABSENT), ("HP:0000256", Status.ABSENT)})
        self.assertEqual({f.quote for f in found}, {"131.7 cm (<3rd percentile)", "55.5 cm (95-97th percentile)"})
        self.assertEqual(dict(next(f for f in found if f.hpo_id == "HP:0004322").lab)["as_written"],
                         "height: 131.7 cm (<3rd percentile)")
        got, _, _ = self.read("Height and OFC were −2.8 SD and −3.1 SD, respectively.")
        self.assertTrue({("HP:0004322", Status.PRESENT), ("HP:0000252", Status.PRESENT)} <= got)
        got, _, _ = self.read("The weight and height of this patient are 11 kg (P90) and 84 cm (P2), respectively.")
        self.assertIn(("HP:0004322", Status.PRESENT), got)
        got, _, _ = self.read("Birth weight and length were 3.35 kg and 51 cm, respectively.")     # no centile: nothing
        self.assertEqual(got, set())
        got, _, w = self.read("Height, weight and OFC were 120 cm (<3rd centile) and 20 kg, respectively.")
        self.assertEqual(got, set())                                     # 3 measures, 2 values: not paired, not misread
        self.assertTrue(any("3 measures but 2 values" in x for x in w))

    def test_centile_before_measure_list(self):
        got, found, _ = self.read("Growth was below the 3rd centile for height, weight and head circumference.")
        self.assertTrue({("HP:0004322", Status.PRESENT), ("HP:0000252", Status.PRESENT)} <= got)
        self.assertEqual(next(f for f in found if f.hpo_id == "HP:0004322").quote, "below the 3rd centile for height")
        got, _, _ = self.read("OFC was above the 97th centile, and below the 3rd percentile for length.")
        self.assertIn(("HP:0004322", Status.PRESENT), got)
        self.assertNotIn(("HP:0000252", Status.PRESENT), got)

    def test_centile_ranges(self):
        got, _, _ = self.read("Head circumference 55.5 cm (95-97th percentile); occipito-frontal circumference 29 cm (1-2nd centile).")
        self.assertEqual(got, {("HP:0000252", Status.ABSENT), ("HP:0000256", Status.ABSENT), ("HP:0000252", Status.PRESENT)})

    def test_children_get_no_adult_defaults(self):
        got, _, w = self.read("Na 128, HR 150, Na 128 mmol/L (ref 133-146), OFC below the 3rd centile", age=6)
        self.assertIn(("HP:0002902", Status.PRESENT), got)                # input range: still used
        self.assertIn(("HP:0000252", Status.PRESENT), got)                # centile: age-independent
        self.assertFalse(any(h in ("HP:0001649", "HP:0001662") for h, _ in got))
        self.assertEqual(len([x for x in w if "younger than 18" in x]), 1)

    def test_structured_labs_vitals_and_age(self):
        from hoofbeats.intake import Lab, Vitals
        from hoofbeats.numbers import age_from_text, age_years, read_structured_labs, read_structured_vitals
        w = []
        labs = read_structured_labs([Lab(name="Sodium", value=128, unit="mmol/L"), Lab(name="Ammonia", value=40, ref_high=32),
                                     Lab(name="Troponin", value=1)], w.append)
        got = {(f.hpo_id, f.status, f.grounding) for f in labs}
        self.assertIn(("HP:0002902", Status.PRESENT, "supported"), got)   # default table, unit given
        self.assertIn(("HP:0001987", Status.PRESENT, "strong"), got)      # input range
        self.assertIn("labs: 'Troponin' is not in the lab map; skipped", w)
        vit = read_structured_vitals(Vitals(bp="150/85", temp_c=36.8), w.append)
        self.assertIn(("HP:0000822", Status.PRESENT), {(f.hpo_id, f.status) for f in vit})
        self.assertEqual(next(f for f in vit if f.hpo_id == "HP:0000822").field, "vitals.bp")
        self.assertEqual(read_structured_labs([Lab(name="Sodium", value=128)], w.append, age=10), [])
        self.assertEqual((age_years("3 months"), age_years("P3Y6M"), age_years(17)), (0.25, 3.5, 17.0))
        self.assertEqual(age_from_text("38 y/o M machinist"), 38.0)


class TestPmcLocator(unittest.TestCase):
    @staticmethod
    def doc(*items):
        # (section_type, type, text)
        return {"passages": [{"infons": {"section_type": s, "type": t}, "text": x, "offset": i * 100}
                             for i, (s, t, x) in enumerate(items)]}

    def test_subject_keys(self):
        from hoofbeats.pmc import subject_keys
        self.assertEqual(subject_keys("Individual 3"), {("label", "3")})
        self.assertEqual(subject_keys("P12"), {("label", "12")})
        self.assertEqual(subject_keys("Family 2 II-3"), {("ped", "II", "3")})
        self.assertEqual(subject_keys("STX_1_Patient_4"), {("label", "4")})
        self.assertEqual(subject_keys("KBG3"), {("code", "kbg3")})
        self.assertEqual(subject_keys("proband"), set())

    def test_single_patient_paragraphs(self):
        from hoofbeats.pmc import single_patient_paragraphs
        d = self.doc(("INTRO", "paragraph", "Background on the disease."), ("CASE", "title_1", "Case presentation"),
                     ("CASE", "paragraph", "A 5-year-old boy had seizures."), ("METHODS", "title_1", "Patients and methods"),
                     ("METHODS", "paragraph", "DNA was extracted."), ("RESULTS", "title_1", "Results"),
                     ("RESULTS", "title_2", "Clinical findings"), ("RESULTS", "paragraph", "He had ataxia."),
                     ("DISCUSS", "title_1", "Clinical description"), ("DISCUSS", "paragraph", "Others report ataxia."))
        self.assertEqual([(p["text"], r) for p, r in single_patient_paragraphs(d)],
                         [("A 5-year-old boy had seizures.", "case_section"), ("He had ataxia.", "case_heading")])

    def test_multi_patient_paragraphs(self):
        from hoofbeats.pmc import multi_patient_paragraphs, subject_keys
        d = self.doc(("RESULTS", "title_1", "Clinical reports"),
                     ("RESULTS", "paragraph", "Patient 1 is a 4-year-old girl with ataxia."),
                     ("RESULTS", "paragraph", "Patient 2 had seizures. His sister, patient 1, did not."),
                     ("RESULTS", "paragraph", "Patient 2, the brother of patient 1, had ataxia."),
                     ("RESULTS", "paragraph", "Patients 1 and 2 both had short stature."),
                     ("RESULTS", "paragraph", "Patient 10 had scoliosis."),
                     ("RESULTS", "title_2", "Patient 2"), ("RESULTS", "paragraph", "He walked late."),
                     ("DISCUSS", "paragraph", "Patient 1 resembles earlier reports."))
        keys = {"a": subject_keys("Individual 1"), "b": subject_keys("Individual 2")}
        got = {k: [(p["text"][:12], r) for p, r in v] for k, v in multi_patient_paragraphs(d, keys).items()}
        self.assertEqual(got, {"a": [("Patient 1 is", "id_start")],
                               "b": [("Patient 2 ha", "id_start"), ("He walked la", "id_heading")]})

    def test_vocab_visible(self):
        from hoofbeats.extract_eval import _norm, vocab_visible
        root = Path(tempfile.mkdtemp())
        (root / "hp.obo").write_text(OBO)
        ont = Ontology(root / "hp.obo")
        self.assertTrue(vocab_visible(ont, "HP:0001250", _norm("She had two Seizures, then none.")))
        self.assertFalse(vocab_visible(ont, "HP:0001250", _norm("She had two convulsive episodes.")))

    def test_check_scores(self):
        import csv
        from hoofbeats.pmc_checks import kappa, score
        self.assertAlmostEqual(kappa(["yes", "yes", "no", "no"], ["yes", "no", "no", "no"]), 0.5)
        d = Path(tempfile.mkdtemp())

        def put(name, rows):
            with open(d / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        b = lambda i, h, e, sm="": {"row_id": f"B{i}", "case_id": "c1", "split": "dev", "hpo_id": h, "label": "", "gold_status": "present",
                                    "case_text": "", "expressed": e, "quote": "", "status_matches": sm, "notes": ""}
        put("task_b_gold_in_text.csv", [b(1, "HP:1", "yes", "yes"), b(2, "HP:2", "yes", "yes"), b(3, "HP:3", "no"), b(4, "HP:4", "maybe")])
        put("task_b_second_annotator.csv", [b(1, "HP:1", "yes"), b(2, "HP:2", "no"), b(3, "HP:3", "no")])
        put("task_c_extras.csv", [{"row_id": "C1", "verdict": "correct_not_in_gold"}, {"row_id": "C2", "verdict": "wrong_term"}])
        it = lambda h, r, m="": {"case": "c1", "kind": "gold", "hpo_id": h, "status": "present", "result": r, "method": m}
        put("items.csv", [it("HP:1", "exact", "numbers;rules"), it("HP:2", "missed"), it("HP:3", "missed")])
        (d / "summary.json").write_text(json.dumps({"pred": 10, "extra": 4, "precision": {"strict": 0.5, "lenient": 0.6}}))
        (d / "manifest.json").write_text(json.dumps({"cases": [{"case_id": "c1", "paper_kind": "multi"}]}))
        res = score(d, d / "items.csv", d / "summary.json", d / "manifest.json")
        self.assertEqual(res["task_b"]["hand_checked_recall"]["strict"], 0.5)
        self.assertEqual(res["task_b"]["hand_checked_recall_by_paper_kind"]["multi"]["n_in_run"], 2)
        self.assertEqual(res["task_b"]["hand_checked_recall_by_status"]["absent"]["n_in_run"], 0)
        self.assertEqual(res["task_b"]["hand_checked_found_by_method"], {"numbers": 1, "rules": 1})
        self.assertEqual(res["task_b"]["agreement"]["rows"], 3)
        self.assertAlmostEqual(res["task_c"]["precision_adjusted"]["strict"], 0.7)
        self.assertTrue(any("maybe" in x for x in res["problems"]))

    def test_frozen_stage1_config(self):
        from hoofbeats.config import DEFAULT, load_stage1
        cfg = load_stage1(DEFAULT, "hp/releases/2026-09-01")
        self.assertEqual((cfg["readers"], cfg["llm"]["prompt_version"], cfg["linker"]["chooser_version"],
                          cfg["status_policy"]), (["rules", "numbers", "llm"], 1, 1, "llm"))
        self.assertTrue(any("HPO is" in p for p in load_stage1(DEFAULT, "hp/releases/2099-01-01")["problems"]))
        d = Path(tempfile.mkdtemp())
        bad = json.loads(Path(DEFAULT).read_text())
        (d / "cal.json").write_text(json.dumps({"hi": 0.9, "margin": 0.05, "lo": 0.62, "k": 10}))
        bad["linker"]["calibration"] = str(d / "cal.json")
        (d / "cfg.json").write_text(json.dumps(bad))
        self.assertTrue(any("differs from the frozen thresholds in hi" in p for p in load_stage1(d / "cfg.json")["problems"]))

    def test_chooser_comparison(self):
        import csv
        from hoofbeats.chooser_eval import make_sheet, score

        def put(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        log = lambda ph, start, h, m="llm_choice": {"case": "c", "field": "note_text", "phrase": ph, "status": "present",
                                                     "start": start, "end": start + 3, "hpo_id": h or "",
                                                     "kept": str(bool(h)), "link_method": m}
        ex = lambda h, m="llm": {"case": "c", "kind": "extra", "hpo_id": h, "label": h, "status": "present", "method": m,
                                 "quote": "q", "evidence": "s"}
        gold = {"case": "c", "kind": "gold", "hpo_id": "HP:G", "label": "g", "status": "present", "method": "llm",
                "quote": "", "evidence": ""}
        d = Path(tempfile.mkdtemp())
        # old: A (chooser, kept), B (chooser, now dropped), C (chooser, will change to D), E (chooser, will hit gold),
        #      F (embedding: not a chooser extra), x (borderline, unlinked)
        put(d / "old/llm_log.csv", [log("a", 0, "HP:A"), log("b", 10, "HP:B"), log("c", 20, "HP:C"), log("e", 30, "HP:E"),
                                    log("f", 40, "HP:F", "embedding"), log("x", 50, None, "borderline")])
        put(d / "old/rules+llm_status-llm/pmc_dev_items.csv", [gold] + [ex(h) for h in ("HP:A", "HP:B", "HP:C", "HP:E", "HP:F")])
        put(d / "new/llm_log.csv", [log("a", 0, "HP:A"), log("b", 10, None, "unlinked"), log("c", 20, "HP:D"),
                                    log("e", 30, "HP:G"), log("f", 40, "HP:F", "embedding"), log("x", 50, "HP:X")])
        put(d / "new/rules+llm_status-llm/pmc_dev_items.csv", [gold] + [ex(h) for h in ("HP:A", "HP:D", "HP:F", "HP:X")])
        for run in ("old", "new"):
            (d / run / "rules+llm_status-llm" / "summary.json").write_text(
                json.dumps({"pred": 10, "precision": {"strict": 0.5, "lenient": 0.5}}))
        chk = d / "checks"
        put(chk / "task_c_llm.csv", [{"row_id": f"D{i}", "case_id": "c", "hpo_id": h, "predicted_status": "present",
                                      "verdict": v} for i, (h, v) in enumerate(
            [("HP:A", "correct_not_in_gold"), ("HP:B", "wrong_term"), ("HP:C", "wrong_term"), ("HP:E", "wrong_term"),
             ("HP:F", "correct_not_in_gold")])])
        (chk / "task_c_llm_key.json").write_text(json.dumps({"rows": {f"D{i}": [{"version": "v1", "stratum": "random"}]
                                                                      for i in range(5)}}))
        res = make_sheet(d / "old", d / "new", chk)
        self.assertEqual(res["outcomes"], {"unchanged": 1, "dropped": 1, "changed_to_judge": 1, "changed_to_gold": 1})
        self.assertEqual(res["candidates"], {"unchanged": 1, "dropped": 1, "changed": 2, "newly_linked": 1})
        sheet = list(csv.DictReader(open(chk / "task_c_chooser.csv")))
        self.assertEqual(sorted(r["hpo_id"] for r in sheet), ["HP:D", "HP:X"])     # changed term + newly linked
        put(chk / "task_c_chooser.csv", [{**r, "verdict": "correct_not_in_gold" if r["hpo_id"] == "HP:D" else "wrong_term"}
                                         for r in sheet])
        out = score(chk, 0.5)
        cp = out["chooser_precision"]
        self.assertEqual((cp["old"], cp["old_sample_after_change"], cp["newly_linked_sample"]), (0.25, 1.0, 0.0))
        # old chooser stratum: 4 extras; 3 of 4 sampled stay (B dropped) -> 3 at 100%, plus 1 fresh at 0%
        self.assertAlmostEqual(cp["new"], 3 / 4)
        self.assertEqual(out["other_llm_extras_precision"], 1.0)
        self.assertAlmostEqual(out["precision_judged"]["old"]["strict"], (5 + 0.25 * 4 + 1.0 * 1) / 10)

    def test_task_c_llm_sheet(self):
        import csv
        from hoofbeats.pmc_checks import make_task_c_llm, score_task_c_llm
        d = Path(tempfile.mkdtemp())
        ex = lambda h, st, m="llm": {"case": "c1", "kind": "extra", "hpo_id": h, "label": h, "status": st, "result": "extra",
                                     "method": m, "quote": "q", "evidence": "s"}
        for name, rows in (("v1.csv", [ex("HP:1", "present"), ex("HP:2", "present"), ex("HP:3", "absent"),
                                       ex("HP:8", "present", "rules")]),
                           ("v2.csv", [ex("HP:1", "present"), ex("HP:4", "present"), ex("HP:5", "absent")])):
            with open(d / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        make_task_c_llm({"v1": str(d / "v1.csv"), "v2": str(d / "v2.csv")}, d, n=2, n_absent=1)
        sheet = list(csv.DictReader(open(d / "task_c_llm.csv")))
        self.assertNotIn("method", sheet[0])                            # blind: no version or method on the sheet
        self.assertNotIn("HP:8", {r["hpo_id"] for r in sheet})          # only LLM-only extras
        self.assertEqual(len(sheet), len({r["hpo_id"] for r in sheet}))  # a shared extra appears once
        with open(d / "task_c_llm.csv", "w", newline="") as fh:         # judge: HP:1 and HP:3 correct, the rest wrong
            w = csv.DictWriter(fh, fieldnames=list(sheet[0]))
            w.writeheader()
            w.writerows([{**r, "verdict": "correct_not_in_gold" if r["hpo_id"] in ("HP:1", "HP:3") else "wrong_term"}
                         for r in sheet])
        (d / "s1.json").write_text(json.dumps({"pred": 10, "precision": {"strict": 0.5, "lenient": 0.6}}))
        res = score_task_c_llm(d, 0.5, {"v1": str(d / "s1.json")})
        key = json.loads((d / "task_c_llm_key.json").read_text())
        v1_random = [rid for rid, ss in key["rows"].items() for s in ss if s == {"version": "v1", "stratum": "random"}]
        self.assertEqual(res["v1"]["judged"], 2)
        self.assertEqual(res["v1"]["absent_correct"], 1.0)              # HP:3, its only absent extra
        share = res["v1"]["llm_extras_correct"]
        self.assertEqual(share, sum(next(r for r in sheet if r["row_id"] == rid)["hpo_id"] in ("HP:1", "HP:3")
                                    for rid in v1_random) / 2)
        self.assertAlmostEqual(res["v1"]["precision_adjusted"]["strict"], (5 + share * 3 + 0.5 * 1) / 10)
        self.assertNotIn("precision_adjusted", res["v2"])               # no summary given for v2


F = lambda h, st, src="x", ev="": Finding(h, st, src, ev or f"{src}: {h}")


class TestIntake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        kb = build(cls.root)
        cls.kb = kb
        cls.ont = Ontology(kb / "hp.obo")
        cls.ext = Extractor(cls.ont)

    def parse(self, obj, **kw):
        from hoofbeats.intake import parse_case
        return parse_case(obj, self.ont, self.ext, **kw)

    # schema and bad input
    def test_bad_input_is_skipped_and_reported(self):
        from hoofbeats.intake import IntakeError, load_case
        c = self.parse({"labs": [{"name": "Ammonia", "value": "high"}, {"name": "Ammonia", "value": 40, "ref_high": 32}],
                        "vitals": {"bp": "120/80", "hr": "fast"}, "sex": "Robot", "hpo_terms": "HP:0001250",
                        "image_findings": [{"status": "present"}], "foo": 1}, path="cases/pt9.json")
        w = " | ".join(c.meta["intake_warnings"])
        self.assertEqual(c.case_id, "pt9")                                      # missing case_id: file name
        for part in ("labs[0] skipped", "vitals.hr skipped", "sex skipped", "hpo_terms skipped",
                     "image_findings[0] skipped", "unknown field 'foo'", "case_id missing"):
            self.assertIn(part, w)
        self.assertEqual(c.meta["vitals"], {"bp": "120/80"})                    # good vitals kept
        self.assertEqual([(f.hpo_id, f.status) for f in c.findings], [("HP:0001987", Status.PRESENT)])   # good lab kept
        self.assertEqual((c.meta["policy"], c.meta["images"], c.meta["sex"]), ("local_only", [], None))
        bad = self.root / "bad.json"
        bad.write_text("{not json")
        with self.assertRaises(IntakeError):
            load_case(bad, self.ont, self.ext)
        with self.assertRaises(IntakeError):
            self.parse([1, 2])

    def test_each_source_is_tagged(self):
        c = self.parse({"case_id": "p1", "policy": "open", "age": 34, "sex": "Female", "genes": ["FBN1", "FBN1"],
                        "vitals": {"temp_c": 38.2}, "truth": ["OMIM:1"],
                        "hpo_terms": [{"id": "HP:0099999", "status": "absent"}, {"id": "HP:1234567"}],
                        "note_text": "Exam shows dislocated lens.",
                        "image_context": "Global developmental delay noted on referral.",
                        "image_findings": [{"finding": "tall stature", "image_id": "xr1"},
                                           {"hpo_id": "HP:0001166", "image_id": "xr2", "evidence": "long fingers"}],
                        "labs": [{"name": "alkaline phosphatase", "value": 20, "unit": "U/L", "ref_low": 35, "ref_high": 105},
                                 {"name": "Troponin", "value": 1, "ref_high": 0.04}]}, drop_ancestors=False)
        got = {f.hpo_id: (f.status, f.source, f.evidence) for f in c.findings}
        self.assertEqual(got["HP:0001250"], (Status.ABSENT, "input:hpo", "HP:0099999"))        # alt id resolved
        self.assertEqual(got["HP:0001083"][:2], (Status.PRESENT, "note"))
        self.assertEqual(got["HP:0001263"][:2], (Status.PRESENT, "image_context"))
        self.assertEqual(got["HP:0000098"], (Status.PRESENT, "image:xr1", "tall stature"))
        self.assertEqual(got["HP:0001166"], (Status.PRESENT, "image:xr2", "long fingers"))
        self.assertEqual(got["HP:0003282"], (Status.PRESENT, "lab:Alkaline Phosphatase", "alkaline phosphatase=20 U/L (ref 35-105)"))
        self.assertEqual(got["HP:0003155"][:2], (Status.ABSENT, "lab:Alkaline Phosphatase"))
        self.assertEqual((c.case_id, c.truth, c.genes, c.meta["age"], c.meta["sex"], c.meta["vitals"], c.meta["policy"]),
                         ("p1", ["OMIM:1"], ["FBN1"], 34, "female", {"temp_c": 38.2}, "open"))
        w = " | ".join(c.meta["intake_warnings"])
        self.assertIn("HP:1234567 is not a current HPO phenotype term", w)
        self.assertIn("'Troponin' is not in the lab map", w)
        self.assertEqual(c.meta["provenance"]["HP:0001250"], [{
            "source": "input:hpo", "status": "absent", "evidence": "HP:0099999", "quote": "HP:0099999", "field": "hpo_terms[0].id",
            "start": 0, "end": 10, "status_cue": None, "link": {"method": "hpo_id", "matched_string": "HP:0099999"},
            "readers": ["input"], "grounding": "strong", "section": None}])

    # fusion
    def test_fusion_status_priority_and_family(self):
        from hoofbeats.fusion import fuse
        fs, prov = fuse([[F("HP:0001083", Status.ABSENT, "note"), F("HP:0001083", Status.UNCERTAIN, "note")],
                         [F("HP:0001083", Status.PRESENT, "image:a"), F("HP:0000098", Status.UNCERTAIN, "note"),
                          F("HP:0000098", Status.ABSENT, "input:hpo"), F("HP:0000098", Status.FAMILY, "note"),
                          F("HP:0001083", Status.PRESENT, "input:hpo"), F("HP:7777777", Status.PRESENT)]], self.ont)
        got = [(f.hpo_id, f.status, f.source) for f in fs]
        self.assertEqual(got, [("HP:0001083", Status.PRESENT, "image:a;input:hpo"), ("HP:0000098", Status.ABSENT, "input:hpo"),
                               ("HP:0000098", Status.FAMILY, "note")])         # family kept beside the patient's status
        self.assertEqual([e["status"] for e in prov["HP:0001083"]], ["absent", "uncertain", "present", "present"])
        self.assertEqual(prov["conflicts"][0]["hpo_id"], "HP:0001083")         # present vs absent recorded
        self.assertEqual(prov["unresolved"][0]["hpo_id"], "HP:7777777")
        fam = Case("x", fs).terms(Status.FAMILY)
        self.assertEqual(fam, ["HP:0000098"])

    def test_fusion_drops_present_ancestors(self):
        from hoofbeats.fusion import fuse
        lists = [[F("HP:0001250", Status.PRESENT, "note")], [F("HP:0007359", Status.PRESENT, "image:a")]]
        fs, prov = fuse(lists, self.ont)
        self.assertEqual([f.hpo_id for f in fs], ["HP:0007359"])
        self.assertEqual(prov["subsumed_by"], {"HP:0001250": ["HP:0007359"]})
        self.assertEqual(len(prov["HP:0001250"]), 1)                           # dropped term keeps its provenance
        fs, prov = fuse(lists, self.ont, drop_ancestors=False)
        self.assertEqual(sorted(f.hpo_id for f in fs), ["HP:0001250", "HP:0007359"])
        self.assertEqual(prov["subsumed_by"], {})

    def test_fusion_absent_ancestor_of_present_term(self):
        from hoofbeats.fusion import fuse
        lists = [[F("HP:0007359", Status.PRESENT, "note"), F("HP:0001250", Status.ABSENT, "input:hpo")]]
        fs, prov = fuse(lists, self.ont)
        self.assertEqual([(f.hpo_id, f.status) for f in fs], [("HP:0007359", Status.PRESENT)])
        self.assertEqual((prov["conflicts"][0]["hpo_id"], prov["conflicts"][0]["contradicted_by"], prov["conflicts"][0]["action"]),
                         ("HP:0001250", ["HP:0007359"], "dropped"))
        fs, prov = fuse(lists, self.ont, drop_ancestors=False)
        self.assertEqual(len(fs), 2)
        self.assertEqual(prov["conflicts"][0]["action"], "kept")
        c = self.parse({"case_id": "c", "note_text": "Focal-onset seizure on EEG.", "hpo_terms": [{"id": "HP:0001250", "status": "absent"}]})
        self.assertTrue(any("absent term is an ancestor of a present term" in w for w in c.meta["intake_warnings"]))

    def test_fusion_keeps_most_general_absent_term(self):
        from hoofbeats.fusion import fuse
        lists = [[F("HP:0007359", Status.ABSENT, "note")], [F("HP:0001250", Status.ABSENT, "input:hpo")]]
        fs, prov = fuse(lists, self.ont)
        self.assertEqual([(f.hpo_id, f.status) for f in fs], [("HP:0001250", Status.ABSENT)])
        self.assertEqual(prov["subsumed_by"], {"HP:0007359": ["HP:0001250"]})
        fs, _ = fuse(lists, self.ont, drop_ancestors=False)
        self.assertEqual(len(fs), 2)

    def test_extract_eval_scoring(self):
        from hoofbeats.extract_eval import evaluate, summarize
        cases = self.root / "eval_cases"
        cases.mkdir(exist_ok=True)
        (cases / "c.json").write_text(json.dumps({"case_id": "c", "note_text": "Exam shows dislocated lens. Focal-onset seizure today. No fever."}))
        gold = {"c.json": {"note": [
            {"hpo_id": "HP:0001083", "label": "Ectopia lentis", "status": "present", "quote": "dislocated lens"},
            {"hpo_id": "HP:0001250", "label": "Seizure", "status": "present", "quote": "Focal-onset seizure"},
            {"hpo_id": "HP:0000098", "label": "Tall stature", "status": "absent", "quote": "No fever", "alternatives": ["HP:0001945"]},
            {"hpo_id": "HP:0001263", "label": "Global developmental delay", "status": "present", "quote": "today"}]}}
        (cases / "gold.json").write_text(json.dumps(gold))
        rows, problems = evaluate(cases, cases / "gold.json", self.ont, self.ext)
        r = rows[0]
        self.assertEqual((r["exact"], r["near"], r["missed"], r["pred"], r["extra"]), (2, 1, 1, 3, 0))   # alternative counts as exact
        self.assertEqual((r["span_checked"], r["span_overlap"]), (3, 3))
        s = summarize(rows)
        self.assertAlmostEqual(s["strict"][1], 2 / 4)
        self.assertEqual(problems, [])

    def test_extract_eval_locates_quotes_by_whole_word(self):
        from hoofbeats.extract_eval import evaluate, locate_quote
        text = "Referred by PCP. Denies CP."
        self.assertEqual(locate_quote(text, {"quote": "CP"}), 24)                  # not inside "PCP"
        self.assertEqual(locate_quote(text, {"quote": "CP", "start": 13}), 13)     # gold offsets win when they match
        self.assertIsNone(locate_quote("Referred by PCP.", {"quote": "CP"}))
        self.assertEqual(locate_quote("N/V. Denies N/V.", {"quote": "N/V."}), 0)
        cases = self.root / "eval_quotes"
        cases.mkdir(exist_ok=True)
        (cases / "c.json").write_text(json.dumps({"case_id": "c", "note_text": "Referred by PCP. No seizures."}))
        (cases / "gold.json").write_text(json.dumps({"c.json": {"note": [
            {"hpo_id": "HP:0001250", "label": "Seizure", "status": "absent", "quote": "seizures"},
            {"hpo_id": "HP:0001945", "label": "Fever", "status": "absent", "quote": "CP"}]}}))
        rows, problems = evaluate(cases, cases / "gold.json", self.ont, self.ext)
        self.assertEqual(rows[0]["span_overlap"], 1)
        self.assertEqual(problems, ["c.json/note: gold quote not in text: 'CP'"])

    # diagnose
    def test_diagnose_reads_case_file(self):
        import contextlib, io
        from hoofbeats.__main__ import main
        path = self.root / "intake_case.json"
        path.write_text(json.dumps({"case_id": "dx1", "note_text": "Tall, with dislocated lens. No seizures.",
                                    "hpo_terms": [{"id": "HP:0001166"}], "truth": ["OMIM:1"], "policy": "open"}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["--kb", str(self.kb), "--common", "none", "diagnose", str(path), "--out", str(self.root / "res")])
        text = out.getvalue()
        self.assertIn("[intake] dx1: 3 findings", text)
        self.assertIn("dx1: truth ['OMIM:1'], rank 1", text)
        self.assertIn("1. Disease Alpha (OMIM:1)", text)
        self.assertIn("[stage1] no LLM endpoint given", text)                  # rules + numbers fallback, said so

    def test_pmc_test_run_refuses_without_the_frozen_setup(self):
        # every case here must stop before any test text is read (the fixture HPO also differs from the frozen one)
        import contextlib, io
        from hoofbeats.__main__ import main
        cases = [([], "add --allow-test"), (["--prompt-version", "v2"], "--prompt-version v2 overrides"),
                 (["--llm-offline", "--llm-cache", str(self.root)], "replays a logged frozen test run")]
        for flags, why in cases:
            argv = ["--kb", str(self.kb), *flags, "pmc-test-run", "--out", str(self.root / "t")]
            if why != "add --allow-test":
                argv.append("--allow-test")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as e:
                main(argv)
            self.assertIn(why, str(e.exception))
            self.assertIn("HPO is hp/releases/test" if why != "add --allow-test" else "--allow-test", str(e.exception))
        self.assertFalse((self.root / "t" / "test").exists())

    def test_diagnose_many_cases_writes_json(self):
        import contextlib, io
        from hoofbeats.__main__ import main
        paths = []
        for i, note in enumerate(("Tall, with dislocated lens.", "No seizures. Fever.")):
            p = self.root / f"many_{i}.json"
            p.write_text(json.dumps({"case_id": f"m{i}", "note_text": note, "truth": ["OMIM:1"], "policy": "open"}))
            paths.append(str(p))
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--kb", str(self.kb), "--common", "none", "diagnose", *paths, "--out", str(self.root / "many")])
        d = json.loads((self.root / "many" / "diagnose" / "m1.json").read_text())
        self.assertFalse(d["llm_reader"])
        self.assertEqual({(x["term"], x["status"], tuple(x["readers"])) for x in d["findings"]},
                         {("Seizure", "absent", ("rules",)), ("Fever", "present", ("rules",))})
        self.assertEqual(d["findings"][0]["quote"], "Fever")
        self.assertTrue(d["top"] and {"rank", "id", "name", "posterior", "claims"} <= set(d["top"][0]))
        self.assertEqual(d["stage1"]["prompt_version"], 1)
        self.assertTrue((self.root / "many" / "diagnose" / "m0.json").exists())


if __name__ == "__main__":
    unittest.main()