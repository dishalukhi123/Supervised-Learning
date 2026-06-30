# Smart Outcome Predictor — Ensemble Learning Project

**Role:** Machine Learning Engineer (EdTech Analytics)
**Topic:** Supervised Learning — Ensemble Methods (Bagging, Boosting, Voting, Stacking)

## 📌 Problem Statement
Single ML models were giving unstable results for predicting student outcomes on an online learning
platform. This project builds an **ensemble-based ML system** that solves two tasks using the same
student dataset:

1. **Classification** — Predict whether a student will *successfully complete* a course (`completion_status`: 0/1)
2. **Regression** — Predict a student's *final performance score* (`final_score`: 0–100)

## 📂 Repository Structure
```
.
├── Smart_Outcome_Predictor.ipynb   # Main Jupyter Notebook (Parts A–G, fully executed)
├── Smart_Outcome_Predictor.html    # HTML export of the notebook (view without Jupyter)
├── Theory_Concepts.pdf             # Part A theory answers + glossary of key terms
├── app.py                          # Streamlit interactive web app
├── requirements.txt                # Python dependencies for the app
├── dataset.csv                     # Source dataset (5,200 student records)
└── README.md                       # This file
```

## 🖥️ Streamlit App (`app.py`)
An interactive web app to explore the whole project without writing code:
- **Data Overview** tab — preview dataset, target distributions, missing values
- **Train Models** tab — pick which ensemble models to train (Bagging, AdaBoost, Gradient Boosting,
  LightGBM, XGBoost, Voting, Stacking) for either the classification or regression task
- **Compare Results** tab — metrics table, best-model highlight, ROC curves / actual-vs-predicted plot,
  confusion matrix
- **Predict New Student** tab — enter a new student's details and get a live prediction

### Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`). Make sure `dataset.csv` is
in the same folder, or upload your CSV via the sidebar uploader.

## 🗂️ Dataset
5,200 student records with:
- **Demographics:** age, country_region, education_background
- **Engagement metrics:** sessions, time_spent_hours, videos_watched, quiz_attempts, assignments_submitted, forum_posts
- **Assessment data:** avg_quiz_score, attendance_rate
- **Targets:** `completion_status` (classification), `final_score` (regression)

A few numeric columns (`time_spent_hours`, `avg_quiz_score`, `attendance_rate`) contained a small
number of missing values, handled via median imputation inside the preprocessing pipeline.

## 🔬 Methods Implemented
| Category | Models |
|---|---|
| **Baseline** | Decision Tree (Classifier & Regressor) |
| **Bagging** | Bagging Classifier & Regressor |
| **Boosting** | AdaBoost, Gradient Boosting, LightGBM, XGBoost (Classifier & Regressor each) |
| **Voting** | Hard Voting & Soft Voting Classifier |
| **Stacking** | Stacking Classifier & Stacking Regressor (with meta-learner) |

All models share one `ColumnTransformer` preprocessing pipeline: median imputation + standard scaling
for numeric features, one-hot encoding for categorical features.

## 📊 Evaluation Metrics
- **Classification:** Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Regression:** MAE, RMSE, R² Score

All models are compared side-by-side in Part F of the notebook, with bar charts and ROC curves,
and the best model per task is automatically identified.

## ✅ Key Findings (see Part G of the notebook for full discussion)
- **Bagging** mainly reduces variance/instability versus a single Decision Tree.
- **Boosting** (especially XGBoost/LightGBM) generally achieves the highest raw accuracy/R² by
  directly reducing bias through sequential error correction.
- **Stacking** typically gives the best overall result by letting a meta-learner optimally weight
  the base models, outperforming simple voting.
- **LightGBM** offers the best speed/accuracy trade-off for deployment if low latency matters.

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib jupyter
jupyter notebook Smart_Outcome_Predictor.ipynb
```
Run all cells top-to-bottom (Run → Run All Cells).

## 🏁 Final Recommendation
- **Classification deployment:** Stacking Classifier (or XGBoost/LightGBM if speed is prioritized)
- **Regression deployment:** Stacking Regressor (or top boosting regressor)

---
*Quality is our motto — Red & White Skill Education*


## 📱 Streamlit App Link

```text
https://smart-outcome-predictor.streamlit.app/
```