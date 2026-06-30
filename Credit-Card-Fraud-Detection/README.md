# Credit Card Fraud Detection — Imbalanced Classification with Threshold Optimisation

## Project Overview
This project builds a fraud detection pipeline for a digital payments company, handling extreme class imbalance (0.17% fraud) and optimising the classification threshold for business value, rather than relying on the default 0.5 cutoff.

## Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, 30 features (V1–V28 PCA-transformed, plus Time, Amount, Class)
- A stratified sample of 50,000 rows was used for local execution, preserving the full fraud class.

## Optimal Threshold
- F1-optimal threshold: `<fill in after running, e.g. 0.37>`
- Recall>=0.90 threshold: `<fill in after running>`
- The model pipeline saved in `fraud_detection_model.pkl` was trained without the threshold baked into `.predict()` — apply the threshold manually to `.predict_proba()` outputs (see `optimal_threshold_` attribute on the pipeline object).

## How to Run
1. Download `creditcard.csv` from the Kaggle link above and place it in this folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook `FraudDetection_SupervisedLearning.ipynb` (or the equivalent `.py` script) top to bottom.
4. The trained pipeline will be saved as `fraud_detection_model.pkl`.

## Files
- `FraudDetection_SupervisedLearning.ipynb` — fully executed notebook (Steps 1–8)
- `fraud_detection_model.pkl` — saved pipeline (best model + optimal threshold)
- `summary_report.md` — business-facing summary report
- `requirements.txt` — Python dependencies
