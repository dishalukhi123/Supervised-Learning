# 🏦 Risk Alert Classifier

## 📌 Project Overview

The Risk Alert Classifier is a Machine Learning web application developed using Streamlit. The project helps identify high-risk customers by applying classification techniques and evaluating model performance using multiple metrics and visualization tools.

This application performs complete data preprocessing, model training, handling imbalanced datasets, and performance evaluation through an interactive dashboard.

---

## 🚀 Features

### 📊 Part B: Dataset Understanding & Preparation

* Upload CSV dataset
* Display dataset preview
* View column names
* Detect target variable automatically
* Handle missing values using KNN Imputation
* Train-Test Split
* Class Distribution Visualization

### 📈 Part C: Baseline Classification Model

* Logistic Regression Model
* Confusion Matrix
* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Type-I Error Analysis
* Type-II Error Analysis

### ⚖️ Part D: Handling Imbalanced Data

Supports multiple balancing techniques:

* Random Under Sampling
* Random Over Sampling
* SMOTE
* ADASYN

Performance comparison before and after balancing:

* Recall
* F1 Score
* ROC-AUC Score

### 🌳 Part E: Tree-Based Classification Models

* Decision Tree Classifier
* Random Forest Classifier

Model comparison using:

* Training Accuracy
* Testing Accuracy

### 🛠️ Part F: Hyperparameter Tuning

* Randomized Search CV
* Grid Search CV

Used to optimize:

* Decision Tree
* Random Forest

### 📉 Visualizations

* Class Distribution Plot
* Confusion Matrix Heatmap
* Performance Comparison Charts
* Interactive Plotly Visualizations

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Imbalanced-Learn
* Plotly
* Matplotlib
* Seaborn

---

## 📂 Project Structure

Risk_Alert_Classifier/

│

├── demo.py

├── requirements.txt

├── README.md

├── Risk_Alert_Classifier_Dataset.csv

│

└── assets/

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/risk-alert-classifier.git
cd risk-alert-classifier
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run demo.py
```

---

## 📦 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
plotly
matplotlib
seaborn
```

---

## 📊 Evaluation Metrics

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

## 🎯 Learning Outcomes

Through this project, students learn:

* Data Preprocessing
* Missing Value Handling
* Classification Algorithms
* Imbalanced Dataset Handling
* Model Evaluation
* Hyperparameter Optimization
* Dashboard Development using Streamlit

---

## 👩‍💻 Author

**Disha Lukhi**

AI & Machine Learning Student

---

## 📜 License

This project is developed for educational and academic purposes.


## 🌐 Live Demo

The application is deployed on Streamlit Community Cloud.

### Access the Application

```text
https://your-app-name.streamlit.app
```

### Deployment Platform

Streamlit Community Cloud

### How to Use

1. Open the live application link.
2. Upload the Risk Alert Classifier dataset (.csv).
3. Explore Dataset Analysis.
4. Train and evaluate the Logistic Regression model.
5. Apply balancing techniques:

   * Under Sampling
   * Over Sampling
   * SMOTE
   * ADASYN
6. Compare model performance metrics.
7. Analyze visualizations and results.

---

## 🚀 Deployment

This project is deployed using Streamlit Community Cloud.

To deploy your own version:

1. Fork or clone this repository.
2. Push the project to GitHub.
3. Login to Streamlit Community Cloud.
4. Connect your GitHub repository.
5. Select `demo.py` as the main file.
6. Click **Deploy**.

---

## 🔗 Repository Link

```text
https://github.com/dishalukhi123/Supervised-Learning/blob/main/PR-3/Risk_Alert_Classifier.ipynb
```

---

## 📱 Streamlit App Link

```text
https://risk-alert-classifier-app.streamlit.app/
```
