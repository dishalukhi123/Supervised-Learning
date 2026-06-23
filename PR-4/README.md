# 📩 Message Intelligence System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

## 📌 Project Overview

The **Message Intelligence System** is a Machine Learning-based web application that automatically classifies digital messages as **Spam** or **Legitimate**.

This project applies concepts of **Probability Theory**, **Distance-Based Learning**, **Margin-Based Learning**, and **Probabilistic Classification** to compare different machine learning algorithms for spam detection.

---

## 🎯 Objective

The objective of this project is to design and implement a classification system that identifies whether incoming messages are:

* **0 → Legitimate Message**
* **1 → Spam Message**

---

## 🌐 Live Demo

**Streamlit Application:**
https://message-intelligence-system-pr-4.streamlit.app/

---

## 📂 Dataset Features

The dataset contains message-related features:

* Message Length
* Number of Special Characters
* Number of URLs
* Keyword Frequency Score
* Sender Activity Score
* Time-Based Features
* Target Variable: `Spam_Label`

### Target Labels

| Label | Meaning            |
| ----- | ------------------ |
| 0     | Legitimate Message |
| 1     | Spam Message       |

---

## 🛠 Technologies Used

* Python
* Jupyter Notebook
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🤖 Machine Learning Models Used

### 1. K-Nearest Neighbors (KNN)

* Distance-based classifier
* Uses nearest neighbors for prediction

### 2. Support Vector Machine (SVM)

* Margin-based classifier
* Creates optimal decision boundaries

### 3. Naive Bayes Classifier

* Probabilistic classifier
* Based on Bayes' Theorem

---

## 📊 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📁 Project Structure

```text
Message-Intelligence-System/
│
├── Message_Intelligence_Dataset(2).csv
├── Message_Intelligence_System.ipynb
├── app.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Message-Intelligence-System.git
cd Message-Intelligence-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧠 Theory Concepts Used

### Conditional Probability

P(A|B) = P(A ∩ B) / P(B)

### Bayes' Theorem

P(A|B) = [P(B|A) × P(A)] / P(B)

---

## 📈 Results

* KNN provides a simple baseline model.
* SVM generally achieves higher classification accuracy.
* Naive Bayes offers fast prediction and low computational cost.

---

## ✅ Conclusion

The Message Intelligence System successfully classifies messages as **Spam** or **Legitimate** using machine learning techniques. The project demonstrates the practical application of probability concepts and classification algorithms in communication security systems.

---

## 👩‍💻 Author

**Disha Lukhi**
B.Tech Student – Artificial Intelligence, Machine Learning & Data Science

---

⭐ If you like this project, please give it a star on GitHub.
