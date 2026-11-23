# Cost-Related Barriers to Medical Care: A 12-Question Machine Learning Screener

**Can a short survey identify U.S. adults who cannot afford to see a doctor?**

This project uses the CDC's **Behavioral Risk Factor Surveillance System (BRFSS)** to build and validate a machine learning screener for cost-related barriers to medical care. It asks **12 routine survey questions** and returns a **calibrated risk** (a percentage that means what it says) that the person could not see a doctor in the past 12 months because of cost.

- **Development data:** BRFSS 2024 (455,997 adults)
- **External validation:** BRFSS 2025 (354,916 adults); the models were frozen, with no retraining
- **Model:** XGBoost, leakage-free pipeline, reported following **TRIPOD+AI**
- **Includes:** five reproducible notebooks, saved models, and a **web app** (FastAPI)

---

## Key results

| | Full model (35 answers) | **12-question screener** |
|---|---|---|
| ROC-AUC, 2024 held-out test | 0.845 (95% CI 0.841–0.849) | **0.839** (0.836–0.844) |
| ROC-AUC, 2025 external validation | 0.844 | **0.838** |
| Calibration (intercept / slope, 2024) | −2.22 / 1.05 (ranking only) | **0.00 / 1.03** (well calibrated) |
| Brier score, 2024 | 0.162 | **0.070** |
| Decision threshold | 0.71 | **0.22** |

- **Better than simple rules.** When both flag the same 16% of adults, the screener finds **56.3%** of adults with a cost barrier. The rule "uninsured or income below $25,000" finds **42.6%**.
- **Holds up a year later.** ROC-AUC fell by only −0.001 from 2024 to 2025, including in Tennessee, a state not in the 2024 training data.
- **Known limits.**
  - Discrimination is lower for Black and Hispanic adults (ROC-AUC ≈ 0.79) than for White adults (0.85).
  - The screener detects only about **one in three insured adults** with a cost barrier, although they make up two-thirds of everyone affected.

### The 12 questions
1. Household income
2. Main source of health coverage
3. Age group
4. Employment status
5. Days of poor mental health (past 30)
6. Self-rated general health
7. Own or rent home
8. Days of poor physical health (past 30)
9. Ever smoked 100 cigarettes?
10. Smoke now: every day, some days, not at all? (questions 9–10 together give smoking status)
11. Serious difficulty concentrating, remembering or making decisions
12. Marital status

Race and ethnicity are **not** inputs. They are used only to check fairness.

---

## Project structure

```
├── app.py                      # Web app (FastAPI): the 12-question screener
├── templates/index.html        # App page
├── static/css/style.css        # App styling
├── notebooks/
│   ├── 0_Data_Preparation_BRFSS2024.ipynb          # CDC 2024 file -> data/BRFSS2024_access.csv
│   ├── 1_EDA_BRFSS.ipynb                           # Exploratory data analysis
│   ├── 2_Feature_Engineering_and_Model_Training_BRFSS.ipynb  # Full model + 12-question screener
│   ├── 3_External_Validation_BRFSS2025.ipynb       # Frozen models tested on 2025 (also creates the 2025 CSV)
│   └── 4_Supplementary_Analyses_for_Paper.ipynb    # CIs, calibration, rule baselines, decision curves
├── models/
│   ├── brfss2024_cost_barrier_model.joblib   # Full model (37 features)
│   ├── brfss2024_short_screener.joblib       # 12-question screener (used by the app)
│   ├── training_summary.json
│   ├── fairness_subgroups.csv
│   └── paper_results.json                    # Every number reported in the paper
├── data/                       # Not in the repo (too large); see "Get the data"
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Get the data

The data files are too large for GitHub. Download them for free from the CDC:

1. **BRFSS 2024:** [cdc.gov/brfss/annual_data/annual_2024.html](https://www.cdc.gov/brfss/annual_data/annual_2024.html). Download the SAS Transport (XPT) file, unzip it, and save it as `data/LLCP2024.XPT`.
2. **BRFSS 2025:** [cdc.gov/brfss/annual_data/annual_2025.html](https://www.cdc.gov/brfss/annual_data/annual_2025.html). Do the same and save it as `data/LLCP2025.XPT`.

> Some unzipped XPT files end with a trailing space in the file name (for example `LLCP2024.XPT `). Rename them so the name ends exactly in `.XPT`.

---

## Setup

Python 3.8 (tested with 3.8.20). Using conda:

```bash
conda create -n brfss python=3.8 -y
conda activate brfss
pip install -r requirements.txt
```

---

## How to run

### 1. Reproduce the analysis
Run the notebooks **in order**, from inside the `notebooks/` folder:

| Step | Notebook | Needs | Creates |
|---|---|---|---|
| 0 | `0_Data_Preparation_BRFSS2024` | `data/LLCP2024.XPT` | `data/BRFSS2024_access.csv` |
| 1 | `1_EDA_BRFSS` | 2024 CSV | charts and statistics |
| 2 | `2_Feature_Engineering_and_Model_Training_BRFSS` | 2024 CSV | both models in `models/` |
| 3 | `3_External_Validation_BRFSS2025` | `data/LLCP2025.XPT`, `models/` | 2025 results, `data/BRFSS2025_access.csv` |
| 4 | `4_Supplementary_Analyses_for_Paper` | both CSVs, `models/` | `models/paper_results.json` |

Notebook 2 takes about 30–60 minutes on a laptop, because it compares nine algorithms and tunes three of them. All random seeds are fixed (42).

### 2. Run the web app
From the project root:

```bash
python app.py
```

Then open **http://127.0.0.1:8000**. Answer the 12 questions and you get a risk percentage and a Yes/No flag. Every question has a "Prefer not to say" option.

The app also has a JSON endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" \
  -d '{"income":"$15k-<25k","insurance_type":"No coverage","age_group":"30-34","employment":"Employed for wages","mental_unhealthy_days":"14+ days","general_health":"Fair","home_ownership":"Rent","physical_unhealthy_days":"1-13 days","smoked_100":"No","cognitive_difficulty":"No","marital_status":"Never married"}'
```

---

## Methods in brief

- **Outcome:** BRFSS item MEDCOST1, *"Was there a time in the past 12 months when you needed to see a doctor but could not because you could not afford it?"*
- **No data leakage:**
  - The data are split 80/20 **before** any preprocessing.
  - Every modelling decision is made by cross-validation on the training data: model choice, SMOTEENN vs class weights, tuning, feature selection and thresholds.
  - The test set is used **once**.
  - Two access items (personal doctor, last check-up) are excluded, because they can be consequences of the outcome.
- **Short screener:**
  - Features are ranked by SHAP importance on training data only.
  - Length is counted in **survey questions**, not model inputs.
  - The tool uses the smallest set within 0.005 ROC-AUC of the full model.
- **Evaluation:**
  - discrimination, calibration, design-based bootstrap 95% CIs and survey weights;
  - decision curve analysis;
  - comparison with simple targeting rules at the same share of people flagged;
  - fairness by race and ethnicity, income, age, sex, insurance and rural/urban residence.

---

## Limitations

- **Cross-sectional, self-reported data.** The screener identifies people **currently** reporting a cost barrier. It does not forecast the future, and its associations are not causal.
- **Same survey for validation.** It is validated on the same survey one year later, which shows it holds up over time, not on a different data source.
- **Weaker for some groups.** Performance is lower for some racial and ethnic groups and for respondents who skip questions. It also misses most insured, higher-income adults with cost barriers.
- **Not individual advice.** This is a **population-health screening aid**. It is not medical, financial or eligibility advice.

---

## Citation

If you use this work, please cite:

> Dutta S. *A Twelve-Question Machine Learning Screener for Cost-Related Barriers to Medical Care in U.S. Adults: Development in BRFSS 2024 and External Validation in BRFSS 2025.* Manuscript in preparation, 2026.

Data source: Centers for Disease Control and Prevention (CDC). Behavioral Risk Factor Surveillance System Survey Data, 2024 and 2025. Atlanta, GA: U.S. Department of Health and Human Services, CDC.

---

## License

MIT License. See [LICENSE](LICENSE).

## Author

**Shuvo Dutta**, Independent Researcher, Public Health Informatics