# numbers reader: lab values, vital signs, ECG intervals and growth measures -> grounded findings with offsets
from __future__ import annotations

import re
from dataclasses import dataclass

from .data import SKIP_SECTIONS, resource, sections, sentence_spans
from .grounding import GroundedFinding, provenance_entry
from .model import Status


@dataclass(frozen=True)
class Analyte:
    name: str
    unit: str
    low: float | None
    high: float | None
    hpo_low: str
    hpo_high: str
    plausible: tuple[float, float]
    case_sensitive: frozenset[str]


def _num(x: str) -> float | None:
    return float(x) if x not in ("", None) else None


ANALYTES: dict[str, Analyte] = {}
ALIAS: dict[str, str] = {}          # lower-case alias (or exact, if case-sensitive) -> analyte name
CS_ALIASES: set[str] = set()
for _r in resource("lab_reference.tsv"):
    _cs = frozenset(a for a in (_r.get("case_sensitive") or "").split("|") if a)
    ANALYTES[_r["analyte"]] = Analyte(_r["analyte"], _r["unit"] or "", _num(_r["low"]), _num(_r["high"]), _r["hpo_low"] or "",
                                      _r["hpo_high"] or "", (float(_r["plausible_low"]), float(_r["plausible_high"])), _cs)
    for _a in [_r["analyte"], *_r["aliases"].split("|")]:
        if _a in _cs:
            CS_ALIASES.add(_a)
            ALIAS[_a] = _r["analyte"]
        else:
            ALIAS.setdefault(_a.lower(), _r["analyte"])
    for _ab in re.findall(r"\(([^)]+)\)", _r["analyte"]):       # "Alanine Aminotransferase (ALT)" -> ALT
        ALIAS.setdefault(_ab.lower(), _r["analyte"])

THRESHOLDS: dict[str, list[tuple[str, float, str, bool, str]]] = {}
for _r in resource("vital_thresholds.tsv"):
    THRESHOLDS.setdefault(_r["measure"], []).append((_r["op"], float(_r["threshold"]), _r["hpo_id"], _r["adult_only"] == "1",
                                                     _r["source"]))

# units: identities and the only conversions supported; anything else is skipped with a warning
_SAME = {"u/l": {"u/l", "iu/l"}, "miu/l": {"miu/l", "uiu/ml", "mu/l"}, "ng/ml": {"ng/ml", "ug/l"},
         "k/ul": {"k/ul", "x10^3/ul", "10^3/ul", "x10^9/l", "10^9/l", "thou/ul", "x10e3/ul", "x10e9/l"},
         "mm/h": {"mm/h", "mm/hr"}}
MONOVALENT = {"Sodium", "Potassium", "Chloride", "Bicarbonate"}
CONVERT = {("Glucose", "mmol/l"): 18.016, ("Creatinine", "umol/l"): 1 / 88.4, ("Urea Nitrogen", "mmol/l"): 2.8,
           ("Calcium, Total", "mmol/l"): 4.008, ("Bilirubin, Total", "umol/l"): 1 / 17.1,
           ("Bilirubin, Direct", "umol/l"): 1 / 17.1, ("Albumin", "g/l"): 0.1, ("Hemoglobin", "g/l"): 0.1,
           **{(a, u): 0.001 for a in ("White Blood Cells", "Platelet Count") for u in ("/ul", "/mm3", "cells/ul", "cells/mm3")}}


def unit_key(u: str | None) -> str:
    return re.sub(r"\s+", "", u or "").replace("µ", "u").replace("μ", "u").replace("mcg", "ug").replace("*", "^").lower()


def convert(an: Analyte, value: float, unit: str | None) -> tuple[float | None, str]:
    # value in the table unit, and how: "same" | "converted" | "missing" (unit absent) | "unknown" (skip)
    u, t = unit_key(unit), unit_key(an.unit)
    if not u:
        return value, "missing"
    if u == t or u in _SAME.get(t, ()) or (an.name in MONOVALENT and u == "meq/l"):
        return value, "same"
    if (an.name, u) in CONVERT:
        return value * CONVERT[(an.name, u)], "converted"
    return None, "unknown"


def age_years(x) -> float | None:
    # years from a number or a string ("3 months", "10 mo", "2 wk", "P3Y11M", "38"); None if unknown
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    m = re.fullmatch(r"p(?:(\d+)y)?(?:(\d+)m)?(?:(\d+)d)?", s)
    if m and any(m.groups()):
        y, mo, d = (int(g or 0) for g in m.groups())
        return y + mo / 12 + d / 365
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(year|yr|y/?o|y\.o|month|mo|week|wk|day|d)", s)
    if m:
        v, unit = float(m.group(1)), m.group(2)
        return v / 12 if unit in ("month", "mo") else v / 52 if unit in ("week", "wk") else v / 365 if unit in ("day", "d") else v
    m = re.fullmatch(r"\d+(?:\.\d+)?", s)
    return float(s) if m else None


AGE_IN_TEXT = re.compile(r"\b(\d{1,3}(?:\.\d)?)\s*(?:-|\s)?(years?|yrs?|y/?o|y\.o\.|months?|mos?|weeks?|wks?|days?)(?:[- ]old)?\b", re.I)


def age_from_text(text: str | None) -> float | None:
    m = AGE_IN_TEXT.search(text or "")
    return age_years(f"{m.group(1)} {m.group(2)}") if m else None


# patterns
NUM = r"(?P<cmp>[<>]=?|≤|≥)?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
UNIT = (r"(?:(?:x\s*)?10\s*[\^*e]?\s*(?:3|9)\s*/\s*[uµμ]?L|[KkTt]/[uµμ]L|thou/[uµμ]L|cells/(?:[uµμ]L|mm3)|/(?:[uµμ]L|mm3)"
        r"|(?:mg|g|ug|µg|μg|mcg|ng|pg)/(?:dL|L|mL)|(?:m|u|µ|μ|n)?mol/L|mEq/L|(?:m|u|µ|μ)?I?U/(?:L|mL)|fL|mm/hr?|%)")
FLAG = r"(?P<flag>\(?\b(?:HH|LL|H|L|high|low)\b\)?|[↑↓])"
REF = (r"[\(\[]\s*(?:ref(?:erence)?\.?|nl|normal|nml|rr|range)?\s*:?\s*(?P<lo>\d+(?:\.\d+)?)\s*(?:-|–|to)\s*"
       r"(?P<hi>\d+(?:\.\d+)?)\s*(?:" + UNIT + r")?\s*[\)\]]")
SEP = r"(?:\s*\((?:serum|plasma|blood|s|p)\))?(?:\s+(?:level|levels|count|value))?\s*(?:[:=]|\bis\b|\bwas\b|\bof\b|\bat\b|[-–])?\s*"


def _alias_rx(aliases, flags):
    alts = sorted(aliases, key=len, reverse=True)
    body = "|".join(re.escape(a).replace(r"\ ", r"\s+") for a in alts)
    return re.compile(r"(?<![A-Za-z0-9])(?P<alias>" + body + r")(?![A-Za-z0-9])" + SEP + NUM
                      + r"(?:\s*(?P<unit>" + UNIT + r"))?(?:\s*" + FLAG + r")?(?:\s*" + REF + r")?", flags)


LAB_RX = [_alias_rx([a for a in ALIAS if a not in CS_ALIASES], re.I), _alias_rx(CS_ALIASES, 0)]
SPECIMEN = re.compile(r"\b(?:CSF|cerebrospinal|urine|urinary|urinalysis|UA|U/A|pleural|ascitic|peritoneal|synovial|stool|fecal)\b", re.I)
V = r"\s*(?:of|was|is|:|=)?\s*"
VITAL_RX = [
    ("bp", re.compile(r"\b(?P<label>BP|B/P|blood pressure)\b" + V + r"(?P<sys>\d{2,3})\s*/\s*(?P<dia>\d{2,3})(?:\s*mm\s*Hg)?", re.I)),
    ("heart_rate", re.compile(r"\b(?P<label>HR|heart rate|pulse|P)\b" + V + r"(?P<v>\d{2,3})(?!\s*/|\d)(?:\s*(?P<unit>bpm|/min))?")),
    ("heart_rate", re.compile(r"\b(?P<label>)(?P<v>\d{2,3})\s*(?P<unit>bpm)\b", re.I)),
    ("respiratory_rate", re.compile(r"\b(?P<label>RR|resp(?:iratory)?(?: rate)?|Resp)\b" + V + r"(?P<v>\d{1,2})(?!\s*/|\d)", re.I)),
    ("temperature", re.compile(r"\b(?P<label>T|Tmax|temp|temperature)\b" + V + r"(?P<v>\d{2,3}(?:\.\d)?)\s*°?\s*(?P<unit>[CF])?\b")),
    ("temperature", re.compile(r"\b(?P<label>temp|temperature|tmax)\b" + V + r"(?P<v>\d{2,3}(?:\.\d)?)\s*°?\s*(?P<unit>[CF])?\b", re.I)),
    ("spo2", re.compile(r"\b(?P<label>SpO2|SaO2|O2 sat(?:uration)?|sats?|saturation|pulse ox)\b" + V + r"(?P<v>\d{2,3})\s*(?P<unit>%)?", re.I)),
    ("pr_ms", re.compile(r"\b(?P<label>PR)(?:\s*interval)?" + V + r"(?P<v>\d{2,3})\s*(?P<unit>ms|msec)?\b")),
    ("qtc_ms", re.compile(r"\b(?P<label>QTc)(?:\s*interval)?" + V + r"(?P<v>\d{3})\s*(?P<unit>ms|msec)?\b", re.I)),
]
PLAUSIBLE = {"heart_rate": (20, 300), "respiratory_rate": (4, 80), "spo2": (50, 100), "pr_ms": (40, 800),
             "qtc_ms": (200, 800), "systolic": (50, 300), "diastolic": (20, 200)}    # outside: a typo, skipped


def plausible(measure: str, value: float) -> bool:
    lo, hi = PLAUSIBLE.get(measure, (float("-inf"), float("inf")))
    return lo <= value <= hi


VITALS_CONTEXT = re.compile(r"\b(?:BP|HR|RR|vitals?|VS|temp|pulse|SpO2|sats?)\b", re.I)
MEASURE = (r"(?P<measure>aortic\s+root|aorta|sinus(?:es)?\s+of\s+valsalva|height|stature|length|"
           r"head\s+circumference|occipito-?frontal\s+circumference|OFC|HC)")
MEASURE_KEY = {"aortic root": "aortic_root_z", "aorta": "aortic_root_z", "sinus of valsalva": "aortic_root_z",
               "sinuses of valsalva": "aortic_root_z", "height": "height_z", "stature": "height_z", "length": "height_z",
               "head circumference": "head_z", "occipitofrontal circumference": "head_z", "ofc": "head_z", "hc": "head_z"}
GAP = r"(?:[^.;\n]|\.(?=\d)){0,40}?"      # up to 40 characters within the clause; decimal points allowed ("4.1 cm")
Z_TAIL = r"\b(?:z|Z)[- ]?(?:score)?\s*(?:of|=|:)?\s*\(?(?P<z>[+\-−–]?\s*\d+(?:\.\d+)?)"
SD_TAIL = r"(?P<z>[+\-−–]\s*\d+(?:\.\d+)?)\s*(?:SD|SDS|standard deviations?)\b"
ORD = r"\s*(?:st|nd|rd|th)?\s*"
CENTILE_TAIL = (r"(?P<cmp><|>|below|above|under|over|less than|greater than)?\s*(?:the\s*)?"
                r"(?:(?P<lo>\d{1,2}(?:\.\d+)?)" + ORD + r"[-–]\s*)?(?P<c>\d{1,2}(?:\.\d+)?)" + ORD + r"(?:centile|percentile|%ile)")
P_TAIL = r"\bP\s*(?P<cmp><|>)?\s*(?P<lo>\d{1,2})(?:\s*[-–]\s*(?P<c>\d{1,2}))?\b"      # "P<5", "P90", "P25-50"
Z_RX = re.compile(MEASURE + r"\b" + GAP + Z_TAIL, re.I)
SD_RX = re.compile(MEASURE + r"\b" + GAP + SD_TAIL, re.I)
CENTILE_RX = re.compile(MEASURE + r"\b" + GAP + CENTILE_TAIL, re.I)
# lists of measures: "height, weight and OFC were 77 cm (10th centile), 11 kg and 43 cm (<3rd centile), respectively" is
# paired by order (the regexes above would give the first value to the last measure); "below the 3rd centile for height
# and head circumference" gives the centile to each measure in the list
ITEM = (r"(?:birth\s+)?(?:aortic\s+root|height|stature|length|(?:body\s+)?weight|BMI|body\s+mass\s+index|head\s+circumference|"
        r"occipito-?frontal\s+circumference|OFC|HC|chest\s+circumference|arm\s+span)(?:\s*[(\[][^)\]\n]{0,40}[)\]])?")
ITEMS = ITEM + r"(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*)" + ITEM + r")*"
LIST_RX = re.compile(ITEM + r"(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*)" + ITEM + r")+(?!\w)", re.I)
ITEM_RX = re.compile(ITEM, re.I)
RESPECTIVELY = re.compile(r"\brespectively\b", re.I)
CENTILE_FIRST_RX = re.compile(r"(?P<cmp>below|under|less than|above|over|greater than|<|>)\s*(?:the\s+)?(?:[a-z]+\s+){0,3}?"
                              r"(?P<c>\d{1,2})" + ORD + r"(?:centiles?|percentiles?)\s+(?:for|at|in|of)\s+(?:the\s+|his\s+|her\s+)?"
                              r"(?P<list>" + ITEMS + r")(?!\w)", re.I)
VALUE_Z = [(re.compile(Z_TAIL, re.I), "z"), (re.compile(SD_TAIL, re.I), "z"), (re.compile(CENTILE_TAIL, re.I), "centile"),
           (re.compile(P_TAIL), "centile")]


def measure_key(name: str) -> str | None:
    k = re.sub(r"\s+", " ", name.lower()).replace("-", "")
    return MEASURE_KEY.get(re.sub(r"\s*[(\[].*$", "", re.sub(r"^birth ", "", k)))


def centile_z(c: float, cmp: str = "", lo: float | None = None) -> float:
    # below the 3rd or above the 97th centile counts as -2.5 / +2.5 SD; a range ("10-25th") must be wholly outside
    cmp = (cmp or "").lower()
    low_side, high_side = cmp in ("<", "below", "under", "less than"), cmp in (">", "above", "over", "greater than")
    return -2.5 if (c < 3 or (low_side and c <= 3)) else 2.5 if ((c if lo is None else lo) > 97 or (high_side and c >= 97)) else 0.0


def value_z(item: str) -> float | None:
    # z from one value in a list ("43.4 cm (5-10th percentile)", "-2.5 SD", "11 kg (P90)"), or None
    for rx, how in VALUE_Z:
        m = rx.search(item)
        if not m:
            continue
        if how == "z":
            return float(re.sub(r"\s+", "", m.group("z")).replace("−", "-").replace("–", "-"))
        if m.group("c") is None:        # "P90"
            return centile_z(float(m.group("lo")), m.group("cmp"))
        return centile_z(float(m.group("c")), m.group("cmp"), float(m.group("lo")) if m.group("lo") else None)
    return None


def split_values(text: str, a: int, b: int) -> list[tuple[int, int]]:
    # the values between a measure list and "respectively", split at top-level ", " / " and " (not inside brackets,
    # not "14,600"); each span starts at its first number or sign
    parts, depth, start, i = [], 0, a, a
    while i < b:
        ch = text[i]
        depth += ch in "([" and 1 or ch in ")]" and -1 or 0
        sep = 0
        if depth == 0:
            m = re.match(r",\s*(?:and\s+)?|\s+and\s+", text[i:b])
            if m and (text[i] != "," or m.end() > 1):
                sep = m.end()
        if sep:
            parts.append((start, i))
            start = i = i + sep
        else:
            i += 1
    parts.append((start, b))
    out = []
    for s, e in parts:
        m = re.search(r"[<>+\-−–~]\s*\d|\d", text[s:e])
        if m:
            while e > s and text[e - 1] in " ,":
                e -= 1
            out.append((s + m.start(), e))
    return out
ARMSPAN_RX = [re.compile(r"\barm\s*span\s*(?:is\s*)?(?:>|greater than|exceeds|exceeding|more than)\s*(?:the\s*)?height\b", re.I),
              re.compile(r"\barm\s*span\s*(?:/|to)\s*height(?:\s*ratio)?\s*(?:of|=|:|is)?\s*(?P<r>\d\.\d+)", re.I)]


def _meets(value: float, op: str, t: float) -> bool:
    return {">": value > t, ">=": value >= t, "<": value < t, "<=": value <= t}[op]


def _finding(hpo, status, source, field, quote, start, end, link, grounding, reader, numeric, cue=None):
    return GroundedFinding(hpo, status, source, quote, quote=quote, field=field, start=start, end=end, status_cue=cue,
                           link=link, grounding=grounding, reader=reader, lab=tuple(numeric.items()))


VITAL_SIGNS = {"temperature_c", "heart_rate", "respiratory_rate", "spo2", "blood_pressure"}


class _Ctx:
    # where findings go and how to report problems; normals (if given) receives normal vital signs that are not emitted
    def __init__(self, source, field, age, warn, reader="numbers", emit_normal_vitals=False, normals=None):
        self.source, self.field, self.age, self.warn, self.reader = source, field, age, warn, reader
        self.emit_normal_vitals, self.normals, self.left_out = emit_normal_vitals, normals, 0
        # normal heart rate, blood pressure and many lab values (ALP, creatinine) depend on age in children, so adult
        # defaults would give false findings; only a range from the input is used for them
        self.child = age is not None and age < 18
        self.child_warned = False

    def skip_child(self, what):
        if not self.child_warned:
            self.warn("numbers: patient younger than 18; adult default lab ranges and vital/ECG thresholds not applied "
                      "(age bands are future work)")
            self.child_warned = True

    def keep(self, found: list[GroundedFinding]) -> list[GroundedFinding]:
        # an absent finding from a vital sign is left out by default (a normal pulse is not evidence against a disease
        # in the way a normal lab can be); the value is still recorded in normals
        if self.emit_normal_vitals:
            return found
        out = []
        for f in found:
            if f.status == Status.ABSENT and dict(f.lab)["measure"] in VITAL_SIGNS:
                self.left_out += 1
                if self.normals is not None:
                    self.normals.append(provenance_entry(f))
            else:
                out.append(f)
        return out


def lab_findings(an: Analyte, value: float, cmp: str | None, unit: str | None, ref: tuple | None, flag: str | None,
                 ctx: _Ctx, quote: str, start, end, cue=None, weak_alias=False) -> list[GroundedFinding]:
    # one finding per direction that has an HPO term: present if out of range that way, else absent
    if ref:                                     # value and input range share a unit: compare as given
        v, how = value, "same"
    else:
        v, how = convert(an, value, unit)
    if how == "unknown":
        ctx.warn(f"numbers: {an.name} unit '{unit}' is not supported; skipped ({quote!r})")
        return []
    if how == "missing" and not (an.plausible[0] <= value <= an.plausible[1]):
        ctx.warn(f"numbers: {an.name} {value:g} has no unit and is not plausible in {an.unit}; skipped")
        return []
    if ref:
        lo, hi, source = ref[0], ref[1], "input"
    elif ctx.child:
        ctx.skip_child("default lab ranges")
        lo = hi = None
        source = None
    else:
        lo, hi, source = an.low, an.high, "default_table"
    fl = (flag or "").strip("()").upper().replace("↑", "H").replace("↓", "L")
    fl = {"HIGH": "H", "LOW": "L", "HH": "H", "LL": "L"}.get(fl, fl) or None
    if source is None and not fl:
        return []
    out = []
    for hpo, bound, below in ((an.hpo_low, lo, True), (an.hpo_high, hi, False)):
        if not hpo:
            continue
        if fl:                                  # the lab's own flag decides when it is given
            status = Status.PRESENT if fl == ("L" if below else "H") else Status.ABSENT if source else None
            if source and bound is not None and status == Status.PRESENT and not (v < bound if below else v > bound):
                ctx.warn(f"numbers: {an.name} flag {fl} disagrees with the {source} range; flag used")
        elif bound is None:
            continue
        elif cmp:                               # "<0.5", ">200": only a clear side counts
            if (below and cmp.startswith("<") and v <= bound) or (not below and cmp.startswith(">") and v >= bound):
                status = Status.PRESENT
            else:
                continue
        else:
            status = Status.PRESENT if (v < bound if below else v > bound) else Status.ABSENT
        if status is None:
            continue
        grounding = ("strong" if source == "input" and how != "missing" else
                     "weak" if how == "missing" or weak_alias else "supported")
        numeric = {"measure": an.name, "value": value, "unit": unit, "converted_value": round(v, 4), "table_unit": an.unit,
                   "ref_low": lo, "ref_high": hi, "range_source": source or "flag", "flag": fl}
        out.append(_finding(hpo, status, ctx.source, ctx.field, quote, start, end, ("numeric:range", an.name), grounding,
                            ctx.reader, numeric, cue))
    return out


def threshold_findings(measure: str, value: float, ctx: _Ctx, quote, start, end, grounding, unit=None, extra=None):
    out = []
    for op, t, hpo, adult_only, src in THRESHOLDS.get(measure, []):
        if adult_only and ctx.child:
            ctx.skip_child("vital-sign and ECG thresholds")
            continue
        status = Status.PRESENT if _meets(value, op, t) else Status.ABSENT
        numeric = {"measure": measure, "value": value, "unit": unit, "threshold": f"{op} {t:g}", "range_source": "threshold",
                   "threshold_source": src, **(extra or {})}
        out.append(_finding(hpo, status, ctx.source, ctx.field, quote, start, end, ("numeric:threshold", measure), grounding,
                            ctx.reader, numeric))
    return ctx.keep(out)


def _bp(sys_, dia, ctx, quote, start, end, grounding):
    # hypertension: SBP >= 140 or DBP >= 90 (one finding); hypotension: SBP < 90
    out = []
    if not (plausible("systolic", sys_) and plausible("diastolic", dia) and sys_ > dia):
        ctx.warn(f"numbers: blood pressure {quote!r} is not plausible; skipped")
        return out
    if ctx.child:
        ctx.skip_child("vital-sign and ECG thresholds")
        return out
    hyper = [r for r in THRESHOLDS["systolic_bp"] if r[2] == "HP:0000822"][0]
    hypo = [r for r in THRESHOLDS["systolic_bp"] if r[2] == "HP:0002615"][0]
    hyper_dia = THRESHOLDS["diastolic_bp"][0]
    is_hyper = _meets(sys_, hyper[0], hyper[1]) or _meets(dia, hyper_dia[0], hyper_dia[1])
    base = {"measure": "blood_pressure", "value": f"{sys_:g}/{dia:g}", "unit": "mmHg", "range_source": "threshold"}
    out.append(_finding("HP:0000822", Status.PRESENT if is_hyper else Status.ABSENT, ctx.source, ctx.field, quote, start, end,
                        ("numeric:threshold", "blood_pressure"), grounding, ctx.reader,
                        {**base, "threshold": "SBP >= 140 or DBP >= 90", "threshold_source": hyper[4]}))
    out.append(_finding("HP:0002615", Status.PRESENT if _meets(sys_, hypo[0], hypo[1]) else Status.ABSENT, ctx.source, ctx.field,
                        quote, start, end, ("numeric:threshold", "blood_pressure"), grounding, ctx.reader,
                        {**base, "threshold": "SBP < 90", "threshold_source": hypo[4]}))
    return ctx.keep(out)


def _temperature(value, unit, ctx, quote, start, end, label_weak):
    # unit C or F; if missing, inferred from the value (weak)
    u = (unit or "").upper()
    if not u:
        u = "C" if 30 <= value <= 45 else "F" if 86 <= value <= 113 else ""
        if not u:
            ctx.warn(f"numbers: temperature {value:g} without a unit is not plausible; skipped")
            return []
        grounding = "weak"
    else:
        grounding = "weak" if label_weak else "supported"
    c = (value - 32) * 5 / 9 if u == "F" else value
    return threshold_findings("temperature_c", round(c, 2), ctx, quote, start, end, grounding, unit=u, extra={"value_as_given": value})


def measure_lists(text: str, s: int, e: int, ctx: _Ctx, used: list, taken) -> list[GroundedFinding]:
    # one sentence: "X, Y and Z were a, b and c, respectively" and "below the 3rd centile for X and Z"; their spans are
    # added to used so the single-measure regexes do not read them again
    out = []
    r = RESPECTIVELY.search(text, s, e)
    lst = LIST_RX.search(text, s, r.start()) if r else None
    if lst and not taken(lst.start(), r.end()):
        names = [m.group(0) for m in ITEM_RX.finditer(lst.group(0))]
        values = split_values(text, lst.end(), r.start())
        used.append((lst.start(), r.end()))
        if len(values) != len(names):
            ctx.warn(f"numbers: {len(names)} measures but {len(values)} values before 'respectively'; skipped "
                     f"({text[lst.start():r.end()]!r})")
        else:
            for name, (a, b) in zip(names, values):
                key, z = measure_key(name), value_z(text[a:b])
                if key and z is not None and key != "aortic_root_z":
                    out += threshold_findings(key, z, ctx, text[a:b], a, b, "supported",
                                              extra={"as_written": f"{name}: {text[a:b]}", "paired": "respectively"})
    for m in CENTILE_FIRST_RX.finditer(text, s, e):
        if taken(m.start(), m.end()):
            continue
        z = centile_z(float(m.group("c")), m.group("cmp"))
        for it in ITEM_RX.finditer(text, m.start("list"), m.end("list")):
            key = measure_key(it.group(0))
            if key and key != "aortic_root_z":
                out += threshold_findings(key, z, ctx, text[m.start():it.end()], m.start(), it.end(), "supported",
                                          extra={"as_written": text[m.start():it.end()]})
        used.append((m.start(), m.end()))
    return out


def read_numbers(text: str | None, source: str, field: str, age: float | None, warn, emit_normal_vitals: bool = False,
                 normals: list | None = None) -> list[GroundedFinding]:
    # free text: labs (with "(ref a-b)" and H/L flags, also unlabelled lists), vitals, PR/QTc, z-scores, centiles, arm span
    if not text or not text.strip():
        return []
    ctx = _Ctx(source, field, age, warn, emit_normal_vitals=emit_normal_vitals, normals=normals)
    out, used = [], []
    taken = lambda a, b: any(a < y and x < b for x, y in used)

    def trim(a, b):
        while b > a and text[b - 1].isspace():
            b -= 1
        return a, b

    for s0, e0, name, typ, head in sections(text):
        if name.startswith(SKIP_SECTIONS):
            continue
        other_specimen = bool(SPECIMEN.search(name))        # a "CSF:" or "Urine:" section
        for s, e in sentence_spans(text, s0, e0):
            sent = text[s:e]
            for rx in LAB_RX:
                for m in rx.finditer(text, s, e):
                    a = m.start("alias")
                    b = max(x for x in (m.end("num"), m.end("unit"), m.end("flag"), m.end("hi") + 1 if m.group("hi") else -1) if x >= 0)
                    a, b = trim(a, m.end() if m.group("lo") else b)
                    if taken(a, b):
                        continue
                    if other_specimen or SPECIMEN.search(text[s:a]):
                        warn(f"numbers: {m.group('alias')} in a non-blood specimen; skipped ({text[a:b]!r})")
                        continue
                    alias = m.group("alias")
                    an = ANALYTES[ALIAS.get(alias) or ALIAS[alias.lower()]]
                    value = float(m.group("num").replace(",", ""))
                    ref = (float(m.group("lo")), float(m.group("hi"))) if m.group("lo") else None
                    cue = (text[m.start("flag"):m.end("flag")], m.start("flag"), m.end("flag")) if m.group("flag") else None
                    found = lab_findings(an, value, m.group("cmp"), m.group("unit"), ref, m.group("flag"), ctx, text[a:b], a, b,
                                         cue, weak_alias=alias in CS_ALIASES)
                    if found:
                        used.append((a, b))
                        out += found
            vital_ctx = bool(VITALS_CONTEXT.search(sent))
            for kind, rx in VITAL_RX:
                for m in rx.finditer(text, s, e):
                    a, b = trim(m.start(), m.end())
                    if taken(a, b):
                        continue
                    label = m.group("label") or ""
                    if label in ("P", "T") and not vital_ctx:
                        continue
                    weak_label = label in ("P", "T")
                    before = ctx.left_out
                    if kind == "bp":
                        found = _bp(float(m.group("sys")), float(m.group("dia")), ctx, text[a:b], a, b, "supported")
                    elif kind == "temperature":
                        found = _temperature(float(m.group("v")), m.group("unit"), ctx, text[a:b], a, b, weak_label)
                    else:
                        v = float(m.group("v"))
                        if not plausible(kind, v):
                            warn(f"numbers: {kind} {v:g} is not plausible; skipped ({text[a:b]!r})")
                            continue
                        has_unit = bool(m.groupdict().get("unit"))
                        g = "weak" if weak_label or (kind in ("pr_ms", "qtc_ms") and not has_unit) else "supported"
                        found = threshold_findings(kind, v, ctx, text[a:b], a, b, g, unit=m.groupdict().get("unit"))
                    if found or ctx.left_out > before:
                        used.append((a, b))
                        out += found
            out += measure_lists(text, s, e, ctx, used, taken)
            for rx, how in ((Z_RX, "z"), (SD_RX, "z"), (CENTILE_RX, "centile")):
                for m in rx.finditer(text, s, e):
                    a, b = trim(m.start(), m.end())
                    if taken(a, b):
                        continue
                    key = measure_key(m.group("measure"))
                    if how == "z":
                        z = float(re.sub(r"\s+", "", m.group("z")).replace("−", "-").replace("–", "-"))
                    else:
                        if key == "aortic_root_z":
                            continue
                        z = centile_z(float(m.group("c")), m.group("cmp"), float(m.group("lo")) if m.group("lo") else None)
                    found = threshold_findings(key, z, ctx, text[a:b], a, b, "supported", extra={"as_written": text[a:b]})
                    if found:
                        used.append((a, b))
                        out += found
            for i, rx in enumerate(ARMSPAN_RX):
                for m in rx.finditer(text, s, e):
                    a, b = trim(m.start(), m.end())
                    if taken(a, b):
                        continue
                    ratio = 1.1 if i == 0 else float(m.group("r"))
                    out += threshold_findings("arm_span_ratio", ratio, ctx, text[a:b], a, b, "supported",
                                              extra={"as_written": text[a:b]})
                    used.append((a, b))
    return out


# structured input
def find_analyte(name: str) -> Analyte | None:
    n = name.strip()
    key = ALIAS.get(n) if n in CS_ALIASES else ALIAS.get(n.lower())
    return ANALYTES.get(key) if key else None


def read_structured_labs(items, warn, age: float | None = None) -> list[GroundedFinding]:
    # labs from the case file: the input reference range wins; otherwise the default table (adults only)
    ctx = _Ctx("lab", None, age, warn, reader="labs")
    out = []
    for lab in items:
        an = find_analyte(lab.name)
        if an is None:
            warn(f"labs: '{lab.name}' is not in the lab map; skipped")
            continue
        ref = None
        if lab.ref_low is not None or lab.ref_high is not None:
            ref = (lab.ref_low if lab.ref_low is not None else float("-inf"), lab.ref_high if lab.ref_high is not None else float("inf"))
        if ref is None and ctx.child:
            warn(f"labs: '{lab.name}' has no reference range and the patient is younger than 18; skipped")
            continue
        rng = (f"{'' if lab.ref_low is None else f'{lab.ref_low:g}'}-{'' if lab.ref_high is None else f'{lab.ref_high:g}'}"
               if ref else f"{an.low if an.low is not None else ''}-{an.high if an.high is not None else ''}, default")
        ev = f"{lab.name}={lab.value:g}{' ' + lab.unit if lab.unit else ''} (ref {rng})"
        ctx.source = f"lab:{an.name}"
        found = lab_findings(an, lab.value, None, lab.unit, ref, None, ctx, ev, None, None)
        for f in found:      # an infinite input bound means "no bound that way"
            if dict(f.lab)["ref_low"] == float("-inf") and f.hpo_id == an.hpo_low or \
               dict(f.lab)["ref_high"] == float("inf") and f.hpo_id == an.hpo_high:
                continue
            out.append(f)
        if not found:
            warn(f"labs: '{lab.name}' gave no finding")
    return out


def read_structured_vitals(vitals, warn, age: float | None = None, emit_normal_vitals: bool = False,
                           normals: list | None = None) -> list[GroundedFinding]:
    # vitals from the case file (bp "120/80", temp_c, hr, rr, spo2); the unit is given by the field
    if vitals is None:
        return []
    out = []
    opts = {"reader": "vitals", "emit_normal_vitals": emit_normal_vitals, "normals": normals}
    if vitals.bp:
        sys_, dia = (float(x) for x in re.findall(r"\d+", vitals.bp)[:2])
        ctx = _Ctx("vitals:bp", "vitals.bp", age, warn, **opts)
        out += _bp(sys_, dia, ctx, vitals.bp, 0, len(vitals.bp), "supported")
    for field, measure, unit in (("temp_c", "temperature_c", "C"), ("hr", "heart_rate", "bpm"), ("rr", "respiratory_rate", "/min"),
                                 ("spo2", "spo2", "%")):
        v = getattr(vitals, field)
        if v is None:
            continue
        if not plausible(measure, float(v)) or (field == "temp_c" and not 25 <= v <= 45):
            warn(f"vitals: {field}={v:g} is not plausible; skipped")
            continue
        ctx = _Ctx(f"vitals:{field}", None, age, warn, **opts)
        out += threshold_findings(measure, float(v), ctx, f"{field}={v:g}", None, None, "supported", unit=unit)
    return out
