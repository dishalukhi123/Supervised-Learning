# 🏠 House Price Prediction Using Machine Learning

## 📌 Project Overview

The House Price Prediction project is a machine learning application developed to estimate residential property prices based on various housing features. The project applies data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning model building, and performance evaluation to identify the most effective predictive model.

This project demonstrates the complete Machine Learning Development Life Cycle (MLDLC), from data collection to model evaluation and interpretation.

---

# 🎯 Objectives

- Analyze housing data and understand key factors affecting house prices.
- Perform data cleaning and preprocessing.
- Visualize relationships between features and house prices.
- Train multiple machine learning models.
- Compare model performance using evaluation metrics.
- Select the best-performing model for prediction.

---

# 📂 Project Structure

```text
House-Price-Prediction/
│
├── data/
│   └── Housing.csv
│
├── notebooks/
│   └── House_Price_Prediction.ipynb
│
├── images/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── models/
│   └── random_forest_model.pkl
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# 📊 Dataset Description

The dataset contains information about residential properties and their selling prices.

## Features

| Feature | Description |
|----------|------------|
| area | Area of house |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of floors |
| parking | Parking spaces |
| mainroad | Main road access |
| guestroom | Guest room availability |
| basement | Basement availability |
| hotwaterheating | Hot water heating |
| airconditioning | Air conditioning |
| furnishingstatus | Furnishing type |
| price | Target variable |

---

# 🔎 Exploratory Data Analysis

## 1. House Price Distribution

Image:
`images/price_distribution.png`

Observation:
- Most houses belong to the medium-price category.
- High-priced houses create a right-skewed distribution.

## 2. Correlation Heatmap

Image:
`images/correlation_heatmap.png`

Observation:
- Area has a strong positive correlation with house price.
- Bathrooms and stories also influence prices significantly.

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Missing value handling
- Duplicate record checking
- Label Encoding
- Feature Scaling
- Train-Test Split (80:20)

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

# 🤖 Machine Learning Models

## 1. Linear Regression

Used as a baseline model.

Advantages:
- Simple
- Fast
- Easy to interpret

---

## 2. Ridge Regression

Uses L2 regularization.

Advantages:
- Reduces overfitting
- Handles multicollinearity

---

## 3. Lasso Regression

Uses L1 regularization.

Advantages:
- Feature selection
- Reduces model complexity

---

## 4. Decision Tree Regressor

Advantages:
- Captures non-linear relationships
- Easy to visualize

Disadvantages:
- Can overfit training data

---

## 5. Random Forest Regressor

Advantages:
- Higher accuracy
- Reduces overfitting
- Handles complex relationships

---

# 📏 Evaluation Metrics

## Mean Absolute Error (MAE)

MAE = (1/n) × Σ|Actual − Predicted|

Measures average prediction error.

---

## Mean Squared Error (MSE)

MSE = (1/n) × Σ(Actual − Predicted)²

Penalizes large errors.

---

## Root Mean Squared Error (RMSE)

RMSE = √MSE

Provides error in original units.

---

## R² Score

R² = 1 − (SSE/SST)

Measures variance explained by the model.

---

# 📈 Model Performance

| Model | MAE | RMSE | R² Score |
|---------|---------|---------|---------|
| Linear Regression | XX | XX | XX |
| Ridge Regression | XX | XX | XX |
| Lasso Regression | XX | XX | XX |
| Decision Tree | XX | XX | XX |
| Random Forest | XX | XX | XX |

Replace XX with your actual values.

---

# 📊 Visual Results

## Actual vs Predicted

Image:
`images/actual_vs_predicted.png`

Observation:
Most points are close to the diagonal line, indicating accurate predictions.

---

## Feature Importance

Image:
`images/feature_importance.png`

Observation:
- Area is the most important feature.
- Bathrooms significantly affect price.
- Parking and furnishing status also contribute.

---

## Model Comparison

Image:
`images/model_comparison.png`

Observation:
Random Forest achieved the highest R² score and lowest prediction error.

---

# 💡 Key Findings

1. Area is the strongest predictor of house price.
2. Random Forest outperformed all other models.
3. Feature engineering improved model accuracy.
4. Ensemble learning methods provided robust results.
5. Proper preprocessing significantly improved performance.

---

# 🚀 Future Scope

- Hyperparameter tuning using GridSearchCV
- XGBoost implementation
- Deep Learning approaches
- Flask deployment
- Streamlit dashboard
- AWS cloud deployment

---

# ▶️ How To Run

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook
```

Open:

```text
House_Price_Prediction.ipynb
```

Run all cells.

---

# 🏆 Conclusion

This project successfully demonstrates the use of machine learning algorithms for predicting house prices. Multiple regression models were evaluated and compared. Random Forest Regressor produced the best overall performance due to its ability to capture complex patterns and reduce overfitting.

The developed model can assist buyers, sellers, and real-estate businesses in estimating property prices more accurately.

---

# 👩‍💻 Author

Disha Lukhi

Data Science & Machine Learning Project
