# offline tests: python -m unittest tests
import csv
import json
import math
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


def image_prompt_of(case):
    from hoofbeats.llm import image_prompt
    return image_prompt(case.meta["images"][0])


if __name__ == "__main__":
    unittest.main()