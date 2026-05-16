# COPD Risk Predictor

**Author:** Menna Allah AlSayed
**Student ID:** 211001864
**Course:** Machine Learning, final project (Spring 2026)

This is the runnable submission package. It contains everything needed to grade the project end-to-end: the cleaned BRFSS modelling dataset, an executed end-to-end notebook, ten figures, the saved scikit-learn pipeline plus a tuned operating point, and a small Flask web app.

---

## Contents
1. [Why this problem](#why-this-problem)
2. [Dataset at a glance](#dataset-at-a-glance)
3. [Submission folder](#submission-folder)
4. [Environment](#environment)
5. [Modelling pipeline](#modelling-pipeline)
6. [Results](#results)
7. [Web application](#web-application)
8. [Running locally](#running-locally)
9. [Reproducibility notes](#reproducibility-notes)
10. [Caveats](#caveats)

---

## Why this problem

COPD is one of the leading causes of death worldwide and is frequently diagnosed late, after lasting damage to the airways. BRFSS already collects rich self-reported lifestyle and comorbidity data, so it is reasonable to ask whether that data alone — without spirometry, FEV1, or any other lab measurement — is enough to flag who might benefit from a clinical follow-up.

This project answers that question with a screening-style classifier. Because the positive class is rare (about 8 % of BRFSS respondents report a COPD diagnosis), the workflow is built around **ROC-AUC for model selection** and **F1 for choosing the operating point**, so the deployed model is biased toward catching cases rather than chasing accuracy.

---

## Dataset at a glance

| Property | Value |
|---|---|
| Source | BRFSS 2022 (CDC) |
| Raw columns | 326 |
| Modelling rows after cleaning | 442,913 |
| Features used | 47 (6 numeric, 41 categorical) |
| Target | `COPD` — derived from BRFSS `CHCCOPD3` |
| Positive rate | ≈ 8 % |

Cleaning is already applied to the shipped `data/brfss2022_copd_model.csv`:
- BRFSS coded values translated to readable labels.
- `88` recoded to `0` for day-count fields (`PHYSHLTH`, `MENTHLTH`); `77` and `99` recoded to missing.
- BMI rescaled from `_BMI5` (BRFSS stores it × 100).
- Refusals kept as an explicit `Unknown` category instead of dropped.

---

## Submission folder

```
submission/
├── README.md                     <- this file
├── requirements.txt              <- pinned dependencies
├── run.bat                       <- Windows launcher
│
├── notebook.ipynb                <- executed end-to-end notebook
├── app.py                        <- Flask backend
├── templates/index.html
│
├── data/brfss2022_copd_model.csv <- cleaned modelling dataset
│
├── figures/                      <- 10 PNGs
│
└── deployment artefacts
    ├── copd_risk_model.pkl
    ├── copd_risk_features.pkl
    ├── copd_risk_threshold.pkl
    └── copd_risk_metadata.json
```

---

## Environment

Python 3.12 with scikit-learn 1.4, pandas 2.2, numpy 1.26, matplotlib 3.8, seaborn 0.13, joblib 1.3, Flask 3.0. Pinned versions in `requirements.txt`.

---

## Modelling pipeline

### Features

Six numeric (`Age`, `BMI`, `SleepHours`, `PhysicalHealthDays`, `MentalHealthDays`, `AvgDrinksPerOccasion`) and 41 categorical features covering demographics, self-rated health, comorbidities, lifestyle, healthcare access, and disability flags.

### Preprocessing (leak-safe)

```
ColumnTransformer
├── numeric    -> SimpleImputer(median) -> StandardScaler
└── categorical-> SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore', min_frequency=50)
```

### Sample and split

A stratified sample of 120,000 rows, then 80/20 stratified split. Both halves keep the population's positive rate.

### Model bake-off

| Model | Hyperparameters | Imbalance handling |
|---|---|---|
| Logistic Regression | `max_iter=1000` | `class_weight='balanced'` |
| Random Forest | `n_estimators=70, max_depth=13, min_samples_leaf=30` | `class_weight='balanced_subsample'` |
| Gradient Boosting | `n_estimators=80, learning_rate=0.08, max_depth=2` | tuned via threshold |

Selection metric: ROC-AUC.

### Threshold tuning

Sweep 0.10 → 0.90 in 0.01 steps. Pick the F1-maximiser.

---

## Results

Best model: **Logistic Regression**, selected on ROC-AUC.

### Final operating point (threshold = 0.77)

| Metric | Score |
|---|---:|
| Accuracy | 0.9000 |
| Precision (COPD class) | 0.4046 |
| Recall (COPD class) | 0.5140 |
| F1-score (COPD class) | 0.4528 |
| ROC-AUC | 0.8686 |

### Model comparison at default 0.5

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.787 | 0.243 | 0.783 | 0.371 | **0.869** |
| Random Forest | 0.798 | 0.251 | 0.763 | 0.378 | 0.867 |
| Gradient Boosting | 0.926 | 0.652 | 0.174 | 0.275 | 0.865 |

### Why threshold = 0.77

Logistic Regression trained with `class_weight='balanced'` shifts probability mass toward the rare class. At 0.5 the model has high recall (0.78) but precision is only 0.24. Sweeping the threshold raises precision to 0.40 while keeping recall at 0.51, a more honest screening trade-off.

### Figures

Ten PNGs in `figures/` on a unified palette: teal for the negative class, coral for the positive class, navy and slate accents, `vlag` heatmap, `crest` confusion matrix.

---

## Web application

The Flask backend in `app.py` loads the four artefacts at import time and serves a single page from `templates/index.html`.

- Three preset patient examples (low / moderate / high risk) populate the form for one-click testing.
- Result card flips green for low-risk and red for high-risk; predicted probability is shown alongside the binary label.
- Hidden categorical fields (state, marital status, etc.) are pre-filled with sensible defaults.
- Probability and threshold are both displayed, so the outcome is interpretable rather than a black-box label.

The predict route assembles a one-row DataFrame in the saved feature order, calls `MODEL.predict_proba(...)[0, 1]`, and compares the value to the saved threshold.

---

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. On Windows, double-clicking `run.bat` does the same.

To re-execute the notebook:
```bash
jupyter notebook notebook.ipynb
```

The notebook reads `data/brfss2022_copd_model.csv` and overwrites the four `copd_risk_*` artefacts in this folder.

---

## Reproducibility notes

- `RANDOM_STATE = 42` controls the stratified sample, the train/test split, and every tree-based model.
- Paths are relative.
- Categorical preprocessing uses `handle_unknown='ignore'`, so the saved pipeline does not crash on unseen labels.
- Dependencies are pinned in `requirements.txt`.

---

## Caveats

- BRFSS data is self-reported. The target captures *ever told*, not current status.
- The sample is US-only and oversamples certain states.
- No clinical measurements (no spirometry, no FEV1, no labs).
- Imbalance plus a recall-leaning threshold means a noticeable false-positive rate.

---

> Coursework deliverable. Not a clinical tool.
