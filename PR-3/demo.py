import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.metrics import recall_score

from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    classification_report
)

st.markdown("""
<style>
.main {
    background-color:#0E1117;
}

h1,h2,h3 {
    color:#00C6FF;
}

[data-testid="stMetric"]{
    background:#1e293b;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)


# st.markdown("---")

# st.markdown(
# """
# <center>
# <h4>🏦 Risk Alert Classifier</h4>
# Created by Disha Lukhi
# </center>
# """,
# unsafe_allow_html=True
# )

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.sidebar.title("Risk Alert Classifier")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 Logistic Regression",
        "⚖️ Imbalanced Data"
    ]
)


# st.sidebar.title("🏦 Navigation")

# menu = st.sidebar.radio(
#     "Select Page",
#     [
#         "Home",
#         "Dataset Analysis",
#         "Baseline Model",
#         "Imbalanced Data"
#     ]
# )

if menu == "🏠 Home":

    st.title("🏦 Risk Alert Classifier Dashboard")

    c1,c2,c3 = st.columns(3)

    c1.metric("Models","4")
    c2.metric("Techniques","4")
    c3.metric("Project Parts","8")

    st.success(
        "Upload your dataset from the Dataset page."
    )

    st.stop()
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Risk Alert Classifier",
    layout="wide"
)

st.title("🏦 Risk Alert Classifier")
st.write("Dataset Understanding, Preparation & Baseline Model")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Risk_Alert_Classifier_Dataset.csv",
    type=["csv"]
)

# -----------------------------
# PROCESS DATASET
# -----------------------------
if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Clean column names
    df.columns = df.columns.str.strip()

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    # -----------------------------
    # FIND TARGET COLUMN
    # -----------------------------
    possible_targets = [
        "risk_status",
        "Risk Status",
        "Risk_Status"
    ]

    target_col = None

    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        st.error(
            "Target column not found. Please check dataset column names."
        )
        st.stop()

    # -----------------------------
    # PART B
    # -----------------------------
    st.header("📊 Part B: Dataset Understanding & Preparation")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    st.subheader("Input Features")
    st.write(list(X.columns))

    st.subheader("Target Variable")
    st.success(target_col)

    # Missing values
    st.subheader("Missing Values")

    missing_values = df.isnull().sum()
    st.dataframe(
        missing_values[missing_values > 0]
    )

    # Numeric columns
    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # KNN Imputer
    if len(numeric_cols) > 0:

        imputer = KNNImputer(
            n_neighbors=5
        )

        df[numeric_cols] = imputer.fit_transform(
            df[numeric_cols]
        )

        st.success(
            "KNN Imputation Applied Successfully"
        )

        st.subheader(
            "Missing Values After Imputation"
        )

        st.dataframe(
            df[numeric_cols].isnull().sum()
        )

    # Train Test Split
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Only numeric columns
    X_model = X.select_dtypes(
        include=["number"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_model,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    st.subheader("Train-Test Split")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Training Rows",
            X_train.shape[0]
        )

    with col2:
        st.metric(
            "Testing Rows",
            X_test.shape[0]
        )


        
    st.subheader("📊 Class Distribution")

    class_counts = y.value_counts().reset_index()
    class_counts.columns = ["Risk Status", "Count"]

    fig = px.bar(
        class_counts,
        x="Risk Status",
        y="Count",
        color="Risk Status",
        text="Count",
        title="Distribution of Risk Classes"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    # -----------------------------
    # PART C
    # -----------------------------
    st.header("📈 Part C: Baseline Classification Model")

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Logistic Regression
    model = LogisticRegression(
        max_iter=5000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    TN, FP, FN, TP = cm.ravel()

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    st.dataframe(cm_df)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)



    fig, ax = plt.subplots(figsize=(6,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Performance Metrics")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        st.metric(
            "Precision",
            f"{precision:.4f}"
        )

    with c2:
        st.metric(
            "Recall",
            f"{recall:.4f}"
        )

        st.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

    # Type Errors
    st.subheader(
        "Type-I and Type-II Errors"
    )

    st.write(
        f"🔴 Type-I Error (False Positive): {FP}"
    )

    st.write(
        f"🔵 Type-II Error (False Negative): {FN}"
    )

    st.info(
        "Type-I Error = Safe customer predicted as High Risk.\n\n"
        "Type-II Error = High Risk customer predicted as Safe."
    )

    st.header("⚖️ Part D: Handling Imbalanced Data")

    # Baseline metrics
    baseline_recall = recall
    baseline_f1 = f1

    baseline_recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    baseline_f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    try:
        baseline_auc = roc_auc_score(y_test, y_pred)
    except:
        baseline_auc = 0

    st.subheader("Before Balancing")

    st.write(f"Recall : {baseline_recall:.4f}")
    st.write(f"F1 Score : {baseline_f1:.4f}")
    st.write(f"AUC ROC : {baseline_auc:.4f}")

    sampling_method = st.selectbox(
        "Select Balancing Technique",
        [
            "Under Sampling",
            "Over Sampling",
            "SMOTE",
            "ADASYN"
        ]
    )

    if sampling_method == "Under Sampling":
        sampler = RandomUnderSampler(random_state=42)

    elif sampling_method == "Over Sampling":
        sampler = RandomOverSampler(random_state=42)

    elif sampling_method == "SMOTE":
        sampler = SMOTE(random_state=42)

    else:
        sampler = ADASYN(random_state=42)

    X_resampled, y_resampled = sampler.fit_resample(
        X_train_scaled,
        y_train
    )

    balanced_model = LogisticRegression(
        max_iter=5000
    )

    balanced_model.fit(
        X_resampled,
        y_resampled
    )

    balanced_pred = balanced_model.predict(
        X_test_scaled
    )

    balanced_recall = recall_score(
        y_test,
        balanced_pred,
        zero_division=0
    )

    balanced_f1 = f1_score(
        y_test,
        balanced_pred,
        zero_division=0
    )

    balanced_auc = roc_auc_score(
        y_test,
        balanced_pred
    )

    st.subheader("After Balancing")

    st.write(f"Recall : {balanced_recall:.4f}")
    st.write(f"F1 Score : {balanced_f1:.4f}")
    st.write(f"AUC ROC : {balanced_auc:.4f}")

else:
    st.info(
        "Please upload the dataset CSV file."
    )


st.markdown("---")

st.markdown(
"""
<center>
<h4>🏦 Risk Alert Classifier</h4>
Created by Disha Lukhi
</center>
""",
unsafe_allow_html=True
)