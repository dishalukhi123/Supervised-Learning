"""
Smart Outcome Predictor - Streamlit App
Ensemble Learning: Bagging, Boosting, Voting, Stacking
Run with: streamlit run app.py
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, StackingClassifier, StackingRegressor
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_curve, confusion_matrix
)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(page_title="Smart Outcome Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Smart Outcome Predictor")
st.caption("Ensemble Learning System — Bagging · Boosting · Voting · Stacking")

# -------------------------------------------------------------------
# Sidebar - Data loading & settings
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

uploaded = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])
default_path = "dataset.csv"

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded is not None:
    df = load_data(uploaded)
else:
    try:
        df = load_data(default_path)
        st.sidebar.info("Using default bundled dataset.csv")
    except FileNotFoundError:
        st.warning("Please upload the Smart Outcome Predictor dataset CSV to continue.")
        st.stop()

task = st.sidebar.radio(
    "Choose Task",
    ["Classification (Course Completion)", "Regression (Final Score)"]
)

test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20, step=5) / 100
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Made for: Smart Outcome Predictor Project · Red & White Skill Education")

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_overview, tab_train, tab_compare, tab_predict = st.tabs(
    ["📊 Data Overview", "🤖 Train Models", "📈 Compare Results", "🔮 Predict New Student"]
)

# -------------------------------------------------------------------
# Column setup
# -------------------------------------------------------------------
target_clf = "completion_status"
target_reg = "final_score"
drop_cols = ["student_id", "course_start_date"]

categorical_cols = [c for c in ["country_region", "device_type", "education_background",
                                 "course_level", "course_category"] if c in df.columns]
numeric_cols = [c for c in df.columns if c not in categorical_cols + drop_cols + [target_clf, target_reg]]

feature_cols = categorical_cols + numeric_cols

# -------------------------------------------------------------------
# TAB 1: Data Overview
# -------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))
    c4.metric("Completion rate", f"{df[target_clf].mean()*100:.1f}%" if target_clf in df.columns else "N/A")

    col1, col2 = st.columns(2)
    with col1:
        if target_clf in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            df[target_clf].value_counts().plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"])
            ax.set_title("Completion Status Distribution")
            ax.set_xlabel("completion_status")
            st.pyplot(fig)
    with col2:
        if target_reg in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(df[target_reg].dropna(), bins=30, color="#1f77b4", edgecolor="black")
            ax.set_title("Final Score Distribution")
            ax.set_xlabel("final_score")
            st.pyplot(fig)

    st.subheader("Missing Values")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) > 0:
        st.dataframe(miss.rename("missing_count"))
    else:
        st.success("No missing values found.")

# -------------------------------------------------------------------
# Preprocessing pipeline (shared)
# -------------------------------------------------------------------
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

is_classification = task.startswith("Classification")
target_col = target_clf if is_classification else target_reg

X = df[feature_cols]
y = df[target_col]

if is_classification:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

# -------------------------------------------------------------------
# Model zoo
# -------------------------------------------------------------------
def build_classifiers(selected):
    models = {}
    if "Decision Tree (Base)" in selected:
        models["Decision Tree (Base)"] = DecisionTreeClassifier(max_depth=8, random_state=random_state)
    if "Bagging" in selected:
        models["Bagging Classifier"] = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=8, random_state=random_state),
            n_estimators=100, random_state=random_state, n_jobs=-1)
    if "AdaBoost" in selected:
        models["AdaBoost Classifier"] = AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=random_state)
    if "Gradient Boosting" in selected:
        models["Gradient Boosting Classifier"] = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=3, random_state=random_state)
    if "LightGBM" in selected and HAS_LGBM:
        models["LightGBM Classifier"] = LGBMClassifier(n_estimators=200, learning_rate=0.1,
                                                         random_state=random_state, verbosity=-1)
    if "XGBoost" in selected and HAS_XGB:
        models["XGBoost Classifier"] = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                                                       random_state=random_state, eval_metric="logloss")
    if "Voting (Soft)" in selected:
        models["Voting Classifier (Soft)"] = VotingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=8, random_state=random_state)),
                ("lr", LogisticRegression(max_iter=1000)),
                ("knn", KNeighborsClassifier(n_neighbors=7))
            ], voting="soft")
    if "Stacking" in selected:
        base_estimators = [
            ("dt", DecisionTreeClassifier(max_depth=8, random_state=random_state)),
            ("knn", KNeighborsClassifier(n_neighbors=7)),
        ]
        if HAS_LGBM:
            base_estimators.append(("lgbm", LGBMClassifier(n_estimators=100, random_state=random_state, verbosity=-1)))
        models["Stacking Classifier"] = StackingClassifier(
            estimators=base_estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5)
    return models


def build_regressors(selected):
    models = {}
    if "Decision Tree (Base)" in selected:
        models["Decision Tree (Base)"] = DecisionTreeRegressor(max_depth=8, random_state=random_state)
    if "Bagging" in selected:
        models["Bagging Regressor"] = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=8, random_state=random_state),
            n_estimators=100, random_state=random_state, n_jobs=-1)
    if "AdaBoost" in selected:
        models["AdaBoost Regressor"] = AdaBoostRegressor(n_estimators=150, learning_rate=0.5, random_state=random_state)
    if "Gradient Boosting" in selected:
        models["Gradient Boosting Regressor"] = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=3, random_state=random_state)
    if "LightGBM" in selected and HAS_LGBM:
        models["LightGBM Regressor"] = LGBMRegressor(n_estimators=200, learning_rate=0.1,
                                                       random_state=random_state, verbosity=-1)
    if "XGBoost" in selected and HAS_XGB:
        models["XGBoost Regressor"] = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4,
                                                     random_state=random_state)
    if "Stacking" in selected:
        base_estimators = [("dt", DecisionTreeRegressor(max_depth=8, random_state=random_state))]
        if HAS_LGBM:
            base_estimators.append(("lgbm", LGBMRegressor(n_estimators=100, random_state=random_state, verbosity=-1)))
        if HAS_XGB:
            base_estimators.append(("xgb", XGBRegressor(n_estimators=100, random_state=random_state)))
        models["Stacking Regressor"] = StackingRegressor(
            estimators=base_estimators, final_estimator=LinearRegression(), cv=5)
    return models


available_options = ["Decision Tree (Base)", "Bagging", "AdaBoost", "Gradient Boosting"]
if HAS_LGBM:
    available_options.append("LightGBM")
if HAS_XGB:
    available_options.append("XGBoost")
if is_classification:
    available_options += ["Voting (Soft)", "Stacking"]
else:
    available_options += ["Stacking"]

# -------------------------------------------------------------------
# TAB 2: Train Models
# -------------------------------------------------------------------
with tab_train:
    st.subheader(f"Train ensemble models — {task}")
    selected = st.multiselect("Select models to train", available_options, default=available_options)

    if st.button("🚀 Train Selected Models", type="primary"):
        if not selected:
            st.warning("Please select at least one model.")
        else:
            models = build_classifiers(selected) if is_classification else build_regressors(selected)
            results = {}
            fitted_pipes = {}
            progress = st.progress(0)
            status = st.empty()

            for i, (name, model) in enumerate(models.items()):
                status.text(f"Training {name} ...")
                pipe = Pipeline([("prep", preprocessor), ("model", model)])
                pipe.fit(X_train, y_train)
                fitted_pipes[name] = pipe
                pred = pipe.predict(X_test)

                if is_classification:
                    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else pred
                    results[name] = {
                        "Accuracy": accuracy_score(y_test, pred),
                        "Precision": precision_score(y_test, pred),
                        "Recall": recall_score(y_test, pred),
                        "F1": f1_score(y_test, pred),
                        "ROC_AUC": roc_auc_score(y_test, proba),
                    }
                else:
                    results[name] = {
                        "MAE": mean_absolute_error(y_test, pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                        "R2": r2_score(y_test, pred),
                    }
                progress.progress((i + 1) / len(models))

            status.text("Done!")
            st.session_state["results"] = results
            st.session_state["fitted_pipes"] = fitted_pipes
            st.session_state["task_type"] = "clf" if is_classification else "reg"
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.success(f"Trained {len(models)} model(s) successfully! Go to **Compare Results** tab.")

# -------------------------------------------------------------------
# TAB 3: Compare Results
# -------------------------------------------------------------------
with tab_compare:
    if "results" not in st.session_state or st.session_state.get("task_type") != ("clf" if is_classification else "reg"):
        st.info("Train some models first in the **Train Models** tab.")
    else:
        results = st.session_state["results"]
        fitted_pipes = st.session_state["fitted_pipes"]
        X_test_s = st.session_state["X_test"]
        y_test_s = st.session_state["y_test"]

        res_df = pd.DataFrame(results).T
        sort_col = "F1" if is_classification else "R2"
        res_df = res_df.sort_values(sort_col, ascending=False)

        st.subheader("📋 Metrics Table")
        st.dataframe(res_df.round(4).style.highlight_max(axis=0, color="#d4f7d4"), use_container_width=True)

        best_model = res_df.index[0]
        st.success(f"🏆 Best model: **{best_model}** ({sort_col} = {res_df.loc[best_model, sort_col]:.4f})")

        col1, col2 = st.columns(2)
        with col1:
            metric_to_plot = sort_col
            fig, ax = plt.subplots(figsize=(6, 4))
            res_df[metric_to_plot].sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
            ax.set_title(f"{metric_to_plot} by Model")
            st.pyplot(fig)

        with col2:
            if is_classification:
                fig, ax = plt.subplots(figsize=(6, 4.2))
                for name, pipe in fitted_pipes.items():
                    if hasattr(pipe.named_steps["model"], "predict_proba"):
                        proba = pipe.predict_proba(X_test_s)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test_s, proba)
                        auc = roc_auc_score(y_test_s, proba)
                        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
                ax.set_title("ROC Curves")
                ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                ax.legend(fontsize=7)
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(6, 4.2))
                best_pred = fitted_pipes[best_model].predict(X_test_s)
                ax.scatter(y_test_s, best_pred, alpha=0.4, s=15, color="#55A868")
                lims = [min(y_test_s.min(), best_pred.min()), max(y_test_s.max(), best_pred.max())]
                ax.plot(lims, lims, "k--", alpha=0.6)
                ax.set_xlabel("Actual final_score")
                ax.set_ylabel("Predicted final_score")
                ax.set_title(f"Actual vs Predicted — {best_model}")
                st.pyplot(fig)

        if is_classification:
            st.subheader("Confusion Matrix (best model)")
            best_pred = fitted_pipes[best_model].predict(X_test_s)
            cm = confusion_matrix(y_test_s, best_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            im = ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(best_model)
            st.pyplot(fig)

# -------------------------------------------------------------------
# TAB 4: Predict New Student
# -------------------------------------------------------------------
with tab_predict:
    if "fitted_pipes" not in st.session_state or st.session_state.get("task_type") != ("clf" if is_classification else "reg"):
        st.info("Train models first, then come back here to predict on a new student.")
    else:
        st.subheader("Enter a new student's details")
        fitted_pipes = st.session_state["fitted_pipes"]
        model_choice = st.selectbox("Model to use for prediction", list(fitted_pipes.keys()))

        input_data = {}
        cols = st.columns(3)
        for i, col_name in enumerate(feature_cols):
            target_col_widget = cols[i % 3]
            if col_name in categorical_cols:
                options = sorted(df[col_name].dropna().unique().tolist())
                input_data[col_name] = target_col_widget.selectbox(col_name, options)
            else:
                col_min, col_max = float(df[col_name].min()), float(df[col_name].max())
                col_mean = float(df[col_name].median())
                input_data[col_name] = target_col_widget.number_input(
                    col_name, value=col_mean, min_value=col_min, max_value=col_max)

        if st.button("🔮 Predict", type="primary"):
            input_df = pd.DataFrame([input_data])[feature_cols]
            pipe = fitted_pipes[model_choice]
            pred = pipe.predict(input_df)[0]

            if is_classification:
                proba = pipe.predict_proba(input_df)[0, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else None
                label = "✅ Will Complete the Course" if pred == 1 else "❌ Will NOT Complete the Course"
                st.subheader(label)
                if proba is not None:
                    st.metric("Predicted probability of completion", f"{proba*100:.1f}%")
            else:
                st.subheader(f"📈 Predicted Final Score: {pred:.1f} / 100")
