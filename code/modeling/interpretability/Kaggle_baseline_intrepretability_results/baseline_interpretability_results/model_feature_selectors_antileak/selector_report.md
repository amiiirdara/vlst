# Multi-Model Feature Selector Report

Generated: 2026-09-19 14:39:53

## Split check (full-cohort fit / inner-val; no unused outer test)

- n_total=5185, n_fit=4148, n_val=1037
- fit_pos=74, val_pos=18

## Models

- cat, lgb, lr, rf, rf_b, xgb, xgb_b

## All unique selected features

- count=86

## Common features across all models per algorithm and metric

- LOCO | pr_auc | n_common=2 | Cre; eGFR
- SHAP | pr_auc | n_common=4 | Cre; HGB; LDL; eGFR
- FFS | pr_auc | n_common=0 | 

## Common features across all models per algorithm (across metrics)

- LOCO | n_common=2 | Cre; eGFR
- SHAP | n_common=4 | Cre; HGB; LDL; eGFR
- FFS | n_common=0 | 

## Common features across algorithms for each model

- cat | pr_auc | LOCO∩SHAP∩FFS n_common=5 | Clopidogrel; HbA1c; LDL; LV; No postdilation
- lgb | pr_auc | LOCO∩SHAP∩FFS n_common=1 | HbA1c
- lr | pr_auc | LOCO∩SHAP∩FFS n_common=4 | Clopidogrel; Cre; Men; eGFR
- rf | pr_auc | LOCO∩SHAP∩FFS n_common=3 | HGB; LDL; LV
- rf_b | pr_auc | LOCO∩SHAP∩FFS n_common=2 | Cre; eGFR
- xgb | pr_auc | LOCO∩SHAP∩FFS n_common=2 | HGB; eGFR
- xgb_b | pr_auc | LOCO∩SHAP∩FFS n_common=4 | Cre; HGB; Stent type-SES_xiencev; eGFR

## Global common features across all models and all algorithms

- n_common=0 | 

## Priority feature ranks by model, algorithm, and metric


### cat
- **LOCO**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=50, Diabetes mellitus=NA, aspirin=NA, Hypertension=9, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=10, Current smoker=35
- **SHAP**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=12, Current smoker=14
- **FFS**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=7, Current smoker=NA

### lgb
- **LOCO**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=35, Diabetes mellitus=NA, aspirin=NA, Hypertension=36, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=6, Current smoker=17
- **SHAP**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=10, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=NA, Current smoker=28
- **FFS**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=NA, Current smoker=NA

### lr
- **LOCO**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=33, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=8, Current smoker=18
- **SHAP**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=11, Current smoker=NA
- **FFS**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=4, Current smoker=NA

### rf
- **LOCO**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=12, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=5, Current smoker=10
- **SHAP**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=11, Current smoker=29
- **FFS**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=NA, Current smoker=NA

### rf_b
- **LOCO**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=25, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=7, Current smoker=46
- **SHAP**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=28, Current smoker=NA
- **FFS**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=NA, Current smoker=NA

### xgb
- **LOCO**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=27, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=5, Current smoker=6
- **SHAP**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=NA, Current smoker=NA
- **FFS**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=NA, Current smoker=NA

### xgb_b
- **LOCO**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=52, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=21, Current smoker=NA
- **SHAP**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=36, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=29, Current smoker=NA
- **FFS**
  - pr_auc: Age, years=NA, Male sex=NA, Current drinking=NA, Diabetes mellitus=NA, aspirin=NA, Hypertension=NA, Dapt=NA, Dyslipidemia=NA, HbA1C=NA, Clopidogrel=NA, Current smoker=NA

## Union of features of all algorithms for each model

- cat | n_union_features=33 | 1.1:1Post dilation; Bifurcation; CKD5; Clopidogrel; Cre; Current smoker; Fast-Glu; Fiberinogen; HGB; HbA1c; Hypertension; Initial diagnosis-AMI; LDL; LV; Max-stent diameter; Men; NO.of vessels; No postdilation; No.of stents per lesion; P-RCA; Platelet; Pre-TIMI flow-3; STEMI; Single-vessel disease; Stent type-SES_partner; Stent type-SES_xiencev; TCL; TG; TIMI-0; Total stent length; UA; eGFR; stent overlap
- lgb | n_union_features=32 | 1.1:1Post dilation; CaI; Clopidogrel; Cre; Current drinking; Current smoker; DAPT; Diabetes; Fast-Glu; Fiberinogen; HDL; HGB; HbA1c; Initial diagnosis-AMI; LDL; LV; LVEF; Max-stent diameter; Men; Min-stent diameter; NSTEMI; No postdilation; P-LAD; PES; Platelet; Previous MI; Previous PCI; Stent type-SES_partner; Stent type-SES_xiencev; TCL; Total stent length; eGFR
- lr | n_union_features=33 | CKD5; CKD60; Chronic total occlusion; Clopidogrel; Cre; Current smoker; DAPT; Fiberinogen; HGB; History of peripheral vascualr disease; Initial diagnosis-AMI; LDL; LV; Max-stent diameter; Men; Min-stent diameter; NSTEMI; No postdilation; No.of stents per lesion; P-LM; P-RCA; Previous PCI; STEMI; Stent type-SES_other; Stent type-SES_xiencev; Stent type-SES_xx; TCL; TG; TIMI-0; Ticagrelor; Total stent length; UA; eGFR
- rf | n_union_features=31 | 1.1:1Post dilation; CKD90; CaI; Clopidogrel; Cre; Current smoker; Fast-Glu; Fiberinogen; HDL; HGB; HbA1c; Hypertension; Initial diagnosis-AMI; LDL; LV; LVEF; Max-stent diameter; Men; NO.of vessels; NSTEMI; No postdilation; No.of stents per lesion; Platelet; Proximal; STEMI; Staged PCI; Stent release pressure; Stent type-SES_xiencev; TG; Total stent length; eGFR
- rf_b | n_union_features=34 | 1.1:1Post dilation; CKD5; CKD90; CaI; Clopidogrel; Cre; DAPT; Diabetes; EVS; Fast-Glu; Fiberinogen; HDL; HGB; HbA1c; LDL; LV; LVEF; Men; Min-stent diameter; No postdilation; No.of stents per lesion; P-LCX; PES; Platelet; Previous PCI; Stent type-SES_partner; Stent type-SES_resolute; Stent type-SES_xiencev; TCL; TIMI-0; Thrombus aspiration; ZES; eGFR; stent overlap
- xgb | n_union_features=33 | 1.1:1Post dilation; Bifurcation; CaI; Cardiogenic shock; Clopidogrel; Cre; Current smoker; Dissection; Fast-Glu; Fiberinogen; HGB; HbA1c; Initial diagnosis-AMI; LDL; LV; LVEF; Lesion location-Ostial; Men; Platelet; Proximal; STEMI; Stent release pressure; Stent type-SES_other; Stent type-SES_resolute; Stent type-SES_tivoli; Stent type-SES_xiencev; Stent type-SES_xx; TCL; TG; Total stent length; UA; Visual thrombus; eGFR
- xgb_b | n_union_features=32 | 1.1:1Post dilation; Age; CKD5; CaI; Cardiogenic shock; Cre; Diabetes; Fast-Glu; Fiberinogen; HGB; HbA1c; History of HF; LDL; LV; LVEF; Lesion location-Ostial; Men; NSTEMI; No postdilation; P-RCA; PES; Platelet; Pre-TIMI flow-3; Previous PCI; Stent type-SES_partner; Stent type-SES_tivoli; Stent type-SES_xiencev; Stent type-SES_xx; TIMI-2; Total stent length; UA; eGFR

## Union of all features of all models and algorithms

- n_union_features=86 | 1.1:1Post dilation; 2-vessel disease; 3-vessel disease; Age; Aneurysm; Aspirin; Bifurcation; CKD5; CKD60; CKD90; CaI; Cardiogenic shock; Chronic renal insufficiency; Chronic total occlusion; Clopidogrel; Cre; Current drinking; Current smoker; DAPT; Diabetes; Dissection; EVS; Fast-Glu; Fiberinogen; HDL; HGB; HL; HbA1c; History of HF; History of peripheral vascualr disease; Hypertension; Initial diagnosis-AMI; LDL; LV; LVEF; Lesion location-Ostial; Max-stent diameter; Men; Min-stent diameter; Moderate/severe calcification; Moderate/severe tortuosity; Multi-vessel CAD; NO.of vessels; NSTEMI; No postdilation; No reflow; No.of stents per lesion; P-LAD; P-LCX; P-LM; P-RCA; PES; Platelet; Pre-TIMI flow-3; Previous CABG; Previous MI; Previous PCI; Proximal; STEMI; Single-vessel disease; Slow flow; Staged PCI; Stent release pressure; Stent type-SES_firebird; Stent type-SES_other; Stent type-SES_partner; Stent type-SES_resolute; Stent type-SES_tivoli; Stent type-SES_xiencev; Stent type-SES_xv; Stent type-SES_xx; Stroke/TIA; TCL; TG; TIMI-0; TIMI-1; TIMI-2; Thrombus aspiration; Ticagrelor; Total stent length; UA; Vessel dialation; Visual thrombus; ZES; eGFR; stent overlap