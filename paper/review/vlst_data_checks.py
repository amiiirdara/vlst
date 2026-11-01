"""Read-only provenance checks on the public Wang 2020 derivation file (VLST.csv).

Standard library only. Usage:
    python3 -I vlst_data_checks.py data/raw/VLST.csv

Reproduces every data-level number quoted in VLST_editorial_review.md
(Section B, "Data-provenance audit") and in the revised manuscript (Table 2).
Nothing is written to disk.
"""

import csv
import re
import statistics as st
import sys
from collections import Counter

TARGET = "Stent thrombosis"
TSSI = "Time since stent implantation"


def load(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def flag(row, col):
    return int(float(row[col])) == 1


def ckd_epi_2009(scr_mg_dl, age, male):
    """CKD-EPI 2009 creatinine equation without the race coefficient."""
    kappa = 0.9 if male else 0.7
    alpha = -0.411 if male else -0.329
    ratio = scr_mg_dl / kappa
    sex = 1.0 if male else 1.018
    return 141 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.209 * 0.993 ** age * sex


def text_decimals(value):
    return len(value.split(".")[1]) if "." in value else 0


def section(title):
    print(f"\n=== {title}")


def main(path):
    rows = load(path)
    groups = {g: [r for r in rows if int(float(r[TARGET])) == g] for g in (0, 1)}
    print(f"rows={len(rows)} events={len(groups[1])} prevalence={len(groups[1]) / len(rows):.4f}")
    case_idx = [i for i, r in enumerate(rows) if int(float(r[TARGET])) == 1]
    print(f"case row positions: {min(case_idx)}-{max(case_idx)} (file sorted by outcome)")

    section("Event times and follow-up (TSSI column)")
    t_case = sorted(float(r[TSSI]) for r in groups[1])
    t_ctrl = sorted(float(r[TSSI]) for r in groups[0])
    print(f"events: min {t_case[0]:.0f}, median {st.median(t_case):.0f}, max {t_case[-1]:.0f}")
    print(f"controls: min {t_ctrl[0]:.0f}, median {st.median(t_ctrl):.0f}, max {t_ctrl[-1]:.0f}")
    print(f"overall median {st.median(t_case + t_ctrl):.0f}")
    for day in (730, 1095, 1241, 1461):
        print(f"  events after day {day}: {sum(t > day for t in t_case)} | "
              f"controls followed < {day} days: {sum(t < day for t in t_ctrl)}")

    section("Binary counts vs Wang Table 1")
    for col in ["Previous PCI", "Initial diagnosis-AMI", "Diabetes", "3-vessel disease", "CKD90",
                "PES", "ZES", "EVS", "1.1:1Post dilation", "No postdilation",
                "Aspirin", "Clopidogrel", "Ticagrelor", "DAPT"]:
        a = sum(flag(r, col) for r in groups[0])
        b = sum(flag(r, col) for r in groups[1])
        print(f"  {col:22s} controls {a} ({100 * a / len(groups[0]):.2f}%)  "
              f"cases {b} ({100 * b / len(groups[1]):.2f}%)")
    print("  PES+ZES+EVS per patient:",
          dict(Counter(sum(flag(r, c) for c in ("PES", "ZES", "EVS")) for r in rows)))

    section("Medication coding consistency (DAPT vs aspirin + P2Y12 inhibitor)")
    for g in (0, 1):
        dapt = [r for r in groups[g] if flag(r, "DAPT")]
        consistent = sum(flag(r, "Aspirin") and (flag(r, "Clopidogrel") or flag(r, "Ticagrelor"))
                         for r in dapt)
        print(f"  group {g}: DAPT=1 n={len(dapt)}, of which aspirin + P2Y12 recorded: {consistent}")

    section("eGFR reproducibility from Cre (umol/L), age, sex")
    for g in (0, 1):
        recorded, recomputed, matched = [], [], 0
        for r in groups[g]:
            calc = ckd_epi_2009(float(r["Cre"]) / 88.4, float(r["Age"]), flag(r, "Men"))
            rec = float(r["eGFR"])
            recorded.append(rec)
            recomputed.append(calc)
            matched += abs(rec - calc) <= 1
        n = len(groups[g])
        print(f"  group {g}: recorded mean {st.mean(recorded):.1f} (<90: {100 * sum(x < 90 for x in recorded) / n:.1f}%) | "
              f"CKD-EPI 2009 mean {st.mean(recomputed):.1f} (<90: {100 * sum(x < 90 for x in recomputed) / n:.1f}%) | "
              f"within 1 unit: {matched}/{n}")
    stage = lambda e: 1 if e >= 90 else 2 if e >= 60 else 3 if e >= 30 else 4 if e >= 15 else 5
    print("  CKD5 equals stage of recorded eGFR:",
          sum(int(float(r["CKD5"])) == stage(float(r["eGFR"])) for r in rows), "/", len(rows))
    print("  CKD90 equals recorded eGFR < 90:",
          sum(flag(r, "CKD90") == (float(r["eGFR"]) < 90) for r in rows), "/", len(rows))

    section("Continuous variables: mean (sample SD), text decimals, integer-valued count")
    for col in ["Cre", "eGFR", "LV", "LVEF", "CaI", "WBC", "HGB", "Platelet",
                "HbA1c", "Fast-Glu", "Fiberinogen", "TCL", "LDL", "HDL", "TG"]:
        for g in (0, 1):
            vals = [float(r[col]) for r in groups[g]]
            dec = Counter(text_decimals(r[col]) for r in groups[g])
            ints = sum(abs(v - round(v)) < 1e-9 for v in vals)
            print(f"  {col:11s} g{g} mean {st.mean(vals):8.2f} sd {st.stdev(vals):7.2f} | "
                  f"text decimals {dict(sorted(dec.items()))} | integer-valued {ints}")
    for g in (0, 1):
        lv = [int(round(float(r["LV"]))) for r in groups[g]]
        print(f"  LV group {g}: even last digit {sum(v % 2 == 0 for v in lv) / len(lv):.3f}; "
              f"last-digit counts {dict(sorted(Counter(v % 10 for v in lv).items()))}")
    for g in (0, 1):
        diab = [float(r["HbA1c"]) for r in groups[g] if flag(r, "Diabetes")]
        nondiab = [float(r["HbA1c"]) for r in groups[g] if not flag(r, "Diabetes")]
        print(f"  HbA1c group {g}: diabetic {st.mean(diab):.2f}, non-diabetic {st.mean(nondiab):.2f}")

    section("Stent brand string vs drug-class flag")
    def canon(s):
        s = re.sub(r"\s+", "", s.strip().replace("：", ":").lower())
        s = s.split(":")[-1]
        for sep in ("，", ",", "/"):
            s = s.split(sep)[0]
        return {"xiencex": "xiencev", "resolut": "resolute", "parnter": "partner"}.get(s, s)
    table = {}
    for r in rows:
        cls = "PES" if flag(r, "PES") else "ZES" if flag(r, "ZES") else "EVS"
        table.setdefault(canon(r["Stent type-SES"]), Counter())[cls] += 1
    for brand, counts in sorted(table.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
        print(f"  {brand:12s} {dict(counts)}")


if __name__ == "__main__":
    main(sys.argv[1])
