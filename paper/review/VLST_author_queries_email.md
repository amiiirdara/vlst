# Queries for the original authors / data custodian

**Part 1** is a ready-to-send email. **Part 2** keeps our working hypotheses apart from the email: they are not confirmed definitions and are not used as such in the manuscript.

---

## Part 1 — Email (ready to send)

**To:** Prof. Yang Zheng and Prof. Jie Du (corresponding authors, Wang et al., *Sci Rep* 2020;10:6378); Dr Xiang Wang (wangxiang17@mails.jlu.edu.cn)
**Subject:** Questions about the public VLST dataset (Figshare 10.6084/m9.figshare.7409606)

Dear Professor Zheng, Professor Du and Dr Wang,

We are carrying out a secondary analysis of the derivation-cohort dataset that you deposited on Figshare alongside your 2020 *Scientific Reports* paper on very late stent thrombosis. Thank you for making it available. Before we report anything, we want to describe the variables accurately, and several questions arose that only you can answer. They are grouped below; brief answers to any of them would help.

**A. Follow-up and outcome**
1. How was `Time since stent implantation` defined? We assume it is days from index PCI to VLST for patients with an event and to the end of follow-up for those without. In the file, follow-up for patients without an event runs from 1,241 to 1,605 days (median 1,419), and the median time to VLST is 736 days. The paper reports a median follow-up of 1,502 days and a median time to VLST of 697 days. Were these figures calculated from a different variable or date?
2. How were patients who died after discharge without VLST handled: excluded, kept with follow-up ending at death, or kept with administrative follow-up? Were deaths, early or late stent thrombosis, or repeat events recorded anywhere?
3. Twelve VLST events in the file occurred after day 1,461. Was any fixed follow-up window, such as four years, intended for the endpoint?

**B. Medication fields**
4. For `Aspirin`, `Clopidogrel`, `Ticagrelor` and `DAPT`: at what time was each recorded (12 months, last contact, or the time of VLST)? Does a value of 1 mean a prescription, actual use, or adherence?
5. Among patients without VLST, `DAPT = 1` always occurs together with aspirin and a P2Y12 inhibitor. Among patients with VLST, 8 of the 35 with `DAPT = 1` have no P2Y12 inhibitor recorded, or no aspirin. Were the medication data for patients with VLST collected or coded differently?

**C. Variable definitions, units and timing**
6. `LV`: what is its full definition, unit and measurement method, and when was it measured? (Integer values range from 37 to 73.)
7. `CaI`: is this peak cardiac troponin I? Its group means and SDs match the "peak troponin I" row of your Table 1 exactly. What are the units, and was the peak measured before or after PCI?
8. `Cre` and `eGFR`: which equation produced `eGFR`, and was it calculated from the creatinine value in `Cre` (µmol/L?) measured at the same time? For about 44% of patients without VLST and about 54% of patients with VLST, `eGFR` equals the CKD-EPI 2009 value calculated from `Cre`, age and sex. For most of the remaining patients without VLST, it is 1.3–1.9 times that value. Could some eGFR values come from a different creatinine measurement or a different source?
9. `CKD5`: is it the CKD stage (1–5) derived from `eGFR`?
10. Do `Fiberinogen` (fibrinogen, g/L?), `Fast-Glu` (fasting glucose, mmol/L?), `WBC`, `HGB`, `Platelet`, `HbA1c` and the lipid fields come from admission samples? Please give units if they differ from your Table 1.

**D. Stent and procedural coding**
11. `PES`: does this flag denote sirolimus-eluting stents? Its frequencies match your Table 1 "SES" row, and the brands with `PES = 1` are Partner, Excel, Firebird, Tivoli and Cypher. Do `ZES` and `EVS` denote zotarolimus- and everolimus-eluting stents?
12. Post-dilation: does `1.1:1Post dilation = 1` mean that post-dilation was performed, at a balloon-to-artery ratio of about 1.1:1? Does `No postdilation = 1` mean it was not performed? Your Table 1 gives "No post-dilation" as 49.01% vs 15.22%, which in the file are the frequencies of `1.1:1Post dilation`. Your Table 2 gives a hazard ratio of 0.145 alongside β = 1.93 and +4 points. Could you confirm the intended direction?

**E. Agreement between the file and the paper**
13. In the file, previous PCI is recorded in 10 patients with VLST; your Table 1 reports 13. AMI as admitting diagnosis is 3,095 and 65 in the file; your Table 1 reports 3,046 and 72. HbA1c group means appear interchanged (the file has 5.80 ± 1.61 without VLST and 6.27 ± 1.53 with VLST). Is the Figshare file the final analysis dataset, or an earlier or partially recoded version?
14. Your Results report 1,837 patients in the intermediate-risk group. Applying the score to the file gives 1,577, which together with the 3,135 low-risk and 473 high-risk patients sums to 5,185. Was 1,837 a typographical error?

**F. Data assembly**
15. In the file, patients with VLST are the last 92 rows, and their numeric values are stored with fewer decimal places than those of other patients. White-cell counts, for example, are never whole numbers in patients with VLST but are whole numbers in about 74% of the others. Were data for patients with VLST extracted, entered or exported separately (for example from adjudication records), and if so from which source?

We will acknowledge your help, and we are happy to share our analysis plan or draft with you. Thank you for your time.

With kind regards,
[Names, affiliations, contact]

---

## Part 2 — Working hypotheses (internal; not confirmed)

| Variable | Working hypothesis only | Basis | Not used as |
|---|---|---|---|
| `CaI` | Peak cardiac troponin I | Group means and sample SDs equal Wang's "peak troponin I" (37.37 ± 61.64; 40.55 ± 72.25) | A confirmed definition; prediction-time decisions beyond "discharge at earliest" |
| `LV` | A left-ventricular dimension in mm (e.g., end-diastolic diameter) | Label "LV"; integers 37–73; mean about 45; an "LV diastolic" label appears in the Methods draft items list, of unknown origin | Any definition, mechanism, title or abstract statement. The Discussion's "diastolic function" paragraph was removed |
| `PES` | Sirolimus-eluting stent flag (mislabelled) | Matches Wang's SES frequencies; all `PES = 1` brands are sirolimus-eluting products (Partner, Excel, Firebird, Tivoli, Cypher, based on general product knowledge) | The confirmed name; the manuscript says "PES (provisionally sirolimus-eluting)" |
| `1.1:1Post dilation` | Post-dilation performed (about 1.1:1 balloon:artery) | Column name; complement column `No postdilation`; Wang's abstract, Discussion and +4 points | Settled coding. The Wang Table 1 label and HR contradict it |
| `CKD5` | CKD stage from recorded eGFR | Exactly equals stage(eGFR) in 5,185/5,185 | — |
| `eGFR` mismatch | Some controls' eGFR derived from a different creatinine measurement or source | 44% of controls match CKD-EPI 2009 exactly; the rest are inflated by a variable factor | A cause. Only the inconsistency is reported |
| Medication fields | Status beyond 12 months (Wang Supplementary Table S4) | Aspirin in 62% despite mandatory 12-month DAPT; Wang's text "DAPT during follow-up" | An exact time point |
| `TSSI` | Days to VLST, or to the last follow-up contact | Event minimum 380; control minimum 1,241 | Censoring rules |
| Recording differences | Cases assembled from a separate source or entry process | Row order, decimals, digit pattern, medication coding, eGFR | A claim of misconduct or error. Reported neutrally as provenance differences |
