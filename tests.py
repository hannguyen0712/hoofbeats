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


if __name__ == "__main__":
    unittest.main()