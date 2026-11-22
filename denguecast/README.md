# DengueCast LK 🦟 — Ensemble Learning Dashboard for Weekly Dengue Cases in Sri Lanka

**Module:** Machine Learning — Hands-on Ensemble Learning Project (10%)
**Student:** Senanayaka S.M.C.S. (224176V)
**Live dashboard:** `https://<your-app>.streamlit.app` *(add after deployment)*
**Video (≤ 3 min):** `<link>` *(add after upload)*

1-week-ahead forecasts of dengue cases for all **26 RDHS divisions** of Sri Lanka, comparing
**bagging, boosting, stacking and hybrid** ensembles on official Ministry of Health data, with an
interactive Streamlit dashboard and a next-week early-warning table.

## Why this problem

2026 is one of Sri Lanka's worst dengue years. Floodwater after Cyclone Ditwah (Nov–Dec 2025) and the
south-west monsoon pushed cases above 90,000 by mid-August 2026. Health offices plan fogging, clean-ups
and hospital beds week by week. A reliable forecast for next week, per division, gives them time to act.

## Data

| | |
|---|---|
| Source | Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health, Sri Lanka |
| Tidy version | `srilanka_weekly_data` in the [denguedatahub](https://github.com/thiyangt/denguedatahub) R package (T. S. Talagala) |
| Coverage | 26 RDHS divisions, 30 Dec 2006 – week starting 10 Aug 2026 (last 4 provisional weeks removed) |
| Rows | 26,753 raw → 26,624 clean (one row per division-week) |
| Files | `data/srilanka_weekly_data.rda`, `data/dengue_weekly_raw.csv`, `data/dengue_weekly_clean.csv` |

`python download_data.py` re-downloads and cleans the data.

## Models compared (test period: Jan 2024 – Aug 2026, includes the 2026 outbreak)

| Rank | Model | Family | MAE | RMSE | R² | Alert recall |
|---|---|---|---|---|---|---|
| 1 | **Stacking (RF+XGB+LGBM+CatBoost+Ridge)** | Stacking | **14.63** | 49.86 | 0.814 | 0.608 |
| 2 | LightGBM | Boosting | 14.89 | 50.21 | 0.811 | 0.622 |
| 3 | Bagging (Decision Trees) | Bagging | 14.99 | 50.79 | 0.807 | 0.619 |
| 4 | Hybrid A: ARIMA + RF | Hybrid | 15.01 | 48.95 | 0.821 | 0.612 |
| 5 | Hybrid B: RF + ANN | Hybrid | 15.06 | **48.02** | **0.827** | 0.568 |
| 6 | ANN (MLP) | Neural net | 15.09 | 50.45 | 0.810 | 0.572 |
| 7 | Random Forest | Bagging | 15.27 | 53.17 | 0.788 | 0.617 |
| 8 | CatBoost | Boosting | 15.33 | 55.21 | 0.772 | 0.612 |
| 9 | XGBoost | Boosting | 15.64 | 56.24 | 0.763 | 0.624 |
| 10 | Ridge regression | Baseline | 15.76 | 51.50 | 0.802 | 0.593 |
| 11 | Naive (last week) | Baseline | 16.01 | 53.50 | 0.786 | **0.698** |
| 12 | ARIMA (per division) | Time series | 16.64 | 57.34 | 0.754 | 0.624 |

MAE = cases per division-week. Alert recall = share of outbreak weeks (above the division's 90th percentile)
that the forecast also flagged. Full discussion in the notebook, Section 15.

## Repository structure

```
├── DengueCast_LK_Ensemble_Learning.ipynb   # full notebook (executed, all outputs)
├── app.py                                  # Streamlit dashboard
├── dengue_features.py                      # shared feature engineering (notebook + app)
├── hybrid_models.py                        # RF+ANN hybrid and CatBoost wrapper classes
├── Colab_Run_Dashboard.ipynb               # runs the dashboard inside Google Colab
├── download_data.py                        # downloads + cleans the dataset
├── build_notebook.py, interpretation.json  # script that generates the notebook
├── requirements.txt
├── data/                                   # raw .rda, raw CSV, clean CSV
├── models/                                 # trained models (.pkl, joblib)
├── results/                                # metrics, predictions, forecasts, figures/
└── docs/                                   # VIDEO_SCRIPT.md, COLAB_GUIDE.md
```

## Run in Google Colab

Step-by-step guide: [`docs/COLAB_GUIDE.md`](docs/COLAB_GUIDE.md). Use `DengueCast_LK_Ensemble_Learning.ipynb` for the analysis and
`Colab_Run_Dashboard.ipynb` for the dashboard.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook DengueCast_LK_Ensemble_Learning.ipynb  # Kernel → Restart & Run All (~3 min)
streamlit run app.py
```

Python 3.11 or newer is required (developed on Python 3.13). Keep the package versions in
`requirements.txt`: saved scikit-learn models only load with the same scikit-learn version.

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository (e.g. `denguecast-lk`) and push this folder:
   ```bash
   git init && git add . && git commit -m "DengueCast LK ensemble dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/denguecast-lk.git
   git push -u origin main
   ```
   All model files are under 15 MB, so no Git LFS is needed.
2. Go to <https://share.streamlit.io> → **Create app** → choose the repo, branch `main`, main file `app.py`.
3. Under **Advanced settings** choose **Python 3.13** (or 3.12), then **Deploy**.
4. Copy the `*.streamlit.app` URL into this README and into the last slide.

## AI Usage Acknowledgment

The lecturer allowed GenAI tools if their use is declared. **Claude (Anthropic)** was used as follows:

| File / part | How GenAI was used |
|---|---|
| `download_data.py` | Code generated by Claude; run and checked by me |
| `dengue_features.py` | Code generated by Claude; feature list reviewed by me |
| `hybrid_models.py` | Code generated by Claude |
| `DengueCast_LK_Ensemble_Learning.ipynb` (all modelling, evaluation and plotting code) | Code generated by Claude |
| Notebook explanations (Sections 1, 15–17, 20) | First drafts written by Claude from the real results, then reviewed and edited by me |
| `app.py` (Streamlit dashboard) | Front-end code generated by Claude |
| `fix_xgb_models.py` | Written with Claude Code to fix the XGBoost model loading error; run once to re-save models |
| README, slides, video script | First drafts by Claude, edited by me |

**Not AI-generated:** the dataset (official Ministry of Health data) and the numerical results — every number
comes from running the code on the real data. I reviewed the code and results and take responsibility for
the submitted work.

## References

1. Epidemiology Unit, Ministry of Health Sri Lanka — Weekly Epidemiological Reports. https://www.epid.gov.lk
2. Talagala, T. S. — denguedatahub R package. https://github.com/thiyangt/denguedatahub
3. Breiman (1996, 2001) — Bagging; Random Forests. Friedman (2001) — Gradient boosting.
4. Chen & Guestrin (2016) XGBoost; Ke et al. (2017) LightGBM; Prokhorenkova et al. (2018) CatBoost.
5. Wolpert (1992) Stacked generalization; Zhang (2003) hybrid ARIMA–neural network forecasting.
6. Grinsztajn et al. (2022) Why tree-based models still outperform deep learning on tabular data.
