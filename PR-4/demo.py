"""
Message Intelligence System — Streamlit App
Spam vs Legitimate message classification using KNN, SVM, and Naive Bayes.

Run with:
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Message Intelligence System", layout="wide", page_icon="📨")

FEATURE_COLS = [
    'message_length', 'word_count', 'num_urls', 'num_digits',
    'num_special_chars', 'spam_keyword_score', 'legit_keyword_score',
    'sender_activity_score', 'sender_account_age_days',
    'messages_sent_last_24h', 'hour_of_day', 'day_of_week'
]
TARGET_COL = 'spam_label'


# --------------------------------------------------------------------------------------
# Cached data + model helpers
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    return df


@st.cache_data
def preprocess(df, test_size, random_state):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Median-impute any missing numeric values
    for c in X.columns[X.isna().any()]:
        X[c] = X[c].fillna(X[c].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


@st.cache_resource
def train_models(X_train_scaled, X_train, y_train, knn_k, knn_metric, svm_kernel, svm_C, svm_gamma):
    knn = KNeighborsClassifier(n_neighbors=knn_k, metric=knn_metric)
    knn.fit(X_train_scaled, y_train)

    svm = SVC(kernel=svm_kernel, C=svm_C, gamma=svm_gamma, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)  # Naive Bayes uses unscaled features

    return knn, svm, nb


def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_confusion_plotly(y_true, y_pred, title, colorscale="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Legit', 'Spam']
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale=colorscale,
        x=labels, y=labels, labels=dict(x="Predicted", y="Actual", color="Count")
    )
    fig.update_layout(title=title, height=350, margin=dict(t=50, b=10, l=10, r=10))
    return fig


def plot_roc_plotly(models_preds_proba, y_true, title="ROC Curve Comparison"):
    fig = go.Figure()
    for name, proba in models_preds_proba.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random",
                              line=dict(dash='dash', color='gray')))
    fig.update_layout(title=title, xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate", height=450)
    return fig


# --------------------------------------------------------------------------------------
# Sidebar — data source & settings
# --------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

uploaded = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])
default_path = "Message_Intelligence_Dataset.csv"

if uploaded is not None:
    df = load_data(uploaded)
elif __import__("os").path.exists(default_path):
    df = load_data(default_path)
else:
    st.sidebar.warning("Upload `Message_Intelligence_Dataset.csv` to get started.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Train/Test Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.subheader("KNN Hyperparameters")
knn_k = st.sidebar.slider("K (neighbors)", 1, 25, 7, 2)
knn_metric = st.sidebar.selectbox("Distance metric", ["euclidean", "manhattan", "chebyshev"])

st.sidebar.subheader("SVM Hyperparameters")
svm_kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
svm_C = st.sidebar.slider("C (regularization)", 0.01, 10.0, 1.0, 0.01)
svm_gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess(
    df, test_size, random_state
)
knn, svm, nb = train_models(
    X_train_scaled, X_train, y_train, knn_k, knn_metric, svm_kernel, svm_C, svm_gamma
)

knn_pred = knn.predict(X_test_scaled)
svm_pred = svm.predict(X_test_scaled)
nb_pred = nb.predict(X_test)

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("📨 Message Intelligence System")
st.caption("Spam vs Legitimate message classification — KNN (distance-based) · SVM (margin-based) · Naive Bayes (probabilistic)")

tab_overview, tab_eda, tab_models, tab_compare, tab_predict = st.tabs(
    ["📋 Overview", "🔍 EDA", "🤖 Models", "📊 Comparison", "✉️ Try a Message"]
)

# --------------------------------------------------------------------------------------
# TAB: Overview
# --------------------------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total messages", len(df))
    c2.metric("Spam (%)", f"{(df[TARGET_COL].mean() * 100):.1f}%")
    c3.metric("Features used", len(FEATURE_COLS))

    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("""
    **Target variable:** `spam_label` → `0` = Legitimate, `1` = Spam

    **Features:** message length, word count, URL/digit/special-character counts,
    spam/legit keyword scores, sender activity score, sender account age,
    messages sent in last 24h, hour of day, day of week.
    """)

# --------------------------------------------------------------------------------------
# TAB: EDA
# --------------------------------------------------------------------------------------
with tab_eda:
    st.subheader("Class Distribution")
    col1, col2 = st.columns(2)

    counts = df[TARGET_COL].value_counts().rename({0: "Legitimate", 1: "Spam"})
    with col1:
        fig_bar = px.bar(
            x=counts.index, y=counts.values, color=counts.index,
            color_discrete_map={"Legitimate": "#4C72B0", "Spam": "#DD8452"},
            labels={"x": "Class", "y": "Count"}, title="Class Counts", text=counts.values
        )
        fig_bar.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        fig_pie = px.pie(
            names=counts.index, values=counts.values,
            color=counts.index,
            color_discrete_map={"Legitimate": "#4C72B0", "Spam": "#DD8452"},
            title="Class Proportion", hole=0.35
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")
    corr = df[FEATURE_COLS + [TARGET_COL]].corr().round(2)
    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto"
    )
    fig_corr.update_layout(height=600, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Feature Distributions by Class")
    feat_choice = st.selectbox("Choose a feature", FEATURE_COLS)
    plot_df = df.copy()
    plot_df['Label'] = plot_df[TARGET_COL].map({0: "Legitimate", 1: "Spam"})
    fig_hist = px.histogram(
        plot_df, x=feat_choice, color="Label", barmode="overlay", marginal="box",
        color_discrete_map={"Legitimate": "#4C72B0", "Spam": "#DD8452"},
        opacity=0.7, nbins=40
    )
    fig_hist.update_layout(height=450)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("2D PCA Projection (colored by class)")
    pca_viz = PCA(n_components=2, random_state=42)
    proj = pca_viz.fit_transform(StandardScaler().fit_transform(df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())))
    proj_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
    proj_df["Label"] = df[TARGET_COL].map({0: "Legitimate", 1: "Spam"}).values
    fig_pca = px.scatter(
        proj_df, x="PC1", y="PC2", color="Label",
        color_discrete_map={"Legitimate": "#4C72B0", "Spam": "#DD8452"},
        opacity=0.6, title=f"PCA (explained variance: {pca_viz.explained_variance_ratio_.sum()*100:.1f}%)"
    )
    fig_pca.update_layout(height=450)
    st.plotly_chart(fig_pca, use_container_width=True)

# --------------------------------------------------------------------------------------
# TAB: Models
# --------------------------------------------------------------------------------------
with tab_models:
    st.subheader("Individual Model Performance")

    m1, m2, m3 = st.tabs(["KNN", "SVM", "Naive Bayes"])

    with m1:
        st.markdown(f"**K = {knn_k}, metric = {knn_metric}**")
        metrics = get_metrics(y_test, knn_pred)
        cols = st.columns(4)
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, f"{v:.3f}")
        st.plotly_chart(plot_confusion_plotly(y_test, knn_pred, "KNN Confusion Matrix", "Blues"),
                         use_container_width=True)
        with st.expander("Classification report"):
            st.text(classification_report(y_test, knn_pred, target_names=['Legitimate', 'Spam']))

    with m2:
        st.markdown(f"**Kernel = {svm_kernel}, C = {svm_C}, gamma = {svm_gamma}**")
        metrics = get_metrics(y_test, svm_pred)
        cols = st.columns(4)
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, f"{v:.3f}")
        st.write(f"Support vectors: **{svm.support_vectors_.shape[0]}** "
                 f"({svm.support_vectors_.shape[0] / len(X_train_scaled) * 100:.1f}% of training data)")
        st.plotly_chart(plot_confusion_plotly(y_test, svm_pred, "SVM Confusion Matrix", "Oranges"),
                         use_container_width=True)
        with st.expander("Classification report"):
            st.text(classification_report(y_test, svm_pred, target_names=['Legitimate', 'Spam']))

    with m3:
        st.markdown("**Gaussian Naive Bayes (unscaled features)**")
        metrics = get_metrics(y_test, nb_pred)
        cols = st.columns(4)
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, f"{v:.3f}")
        st.write(f"Class priors → P(Legitimate) = {nb.class_prior_[0]:.3f}, "
                 f"P(Spam) = {nb.class_prior_[1]:.3f}")
        st.plotly_chart(plot_confusion_plotly(y_test, nb_pred, "Naive Bayes Confusion Matrix", "Greens"),
                         use_container_width=True)
        with st.expander("Classification report"):
            st.text(classification_report(y_test, nb_pred, target_names=['Legitimate', 'Spam']))

    st.subheader("ROC Curve Comparison")
    proba_dict = {
        "KNN": knn.predict_proba(X_test_scaled)[:, 1],
        "SVM": svm.predict_proba(X_test_scaled)[:, 1],
        "Naive Bayes": nb.predict_proba(X_test)[:, 1],
    }
    st.plotly_chart(plot_roc_plotly(proba_dict, y_test), use_container_width=True)

# --------------------------------------------------------------------------------------
# TAB: Comparison
# --------------------------------------------------------------------------------------
with tab_compare:
    st.subheader("Model Comparison")

    results = pd.DataFrame({
        'Model': ['KNN', 'SVM', 'Naive Bayes'],
        'Type': ['Distance-based', 'Margin-based', 'Probabilistic'],
        **{k: [get_metrics(y_test, p)[k] for p in [knn_pred, svm_pred, nb_pred]]
           for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score']}
    }).round(4)

    st.dataframe(results, use_container_width=True)

    melted = results.melt(id_vars=['Model', 'Type'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                           var_name='Metric', value_name='Score')
    fig_cmp = px.bar(
        melted, x='Model', y='Score', color='Metric', barmode='group',
        text=melted['Score'].round(3), title="Accuracy / Precision / Recall / F1 by Model"
    )
    fig_cmp.update_layout(yaxis_range=[0, 1.05], height=480)
    st.plotly_chart(fig_cmp, use_container_width=True)

    best_precision = results.loc[results['Precision'].idxmax()]
    best_recall = results.loc[results['Recall'].idxmax()]
    best_f1 = results.loc[results['F1 Score'].idxmax()]

    c1, c2, c3 = st.columns(3)
    c1.info(f"**Best Precision:** {best_precision['Model']} ({best_precision['Precision']:.3f})\n\n"
            "Use when false positives (blocking real messages) are costly.")
    c2.info(f"**Best Recall:** {best_recall['Model']} ({best_recall['Recall']:.3f})\n\n"
            "Use when missing actual spam (security risk) is costly.")
    c3.success(f"**Best F1 (balanced):** {best_f1['Model']} ({best_f1['F1 Score']:.3f})")

# --------------------------------------------------------------------------------------
# TAB: Try a Message (live prediction)
# --------------------------------------------------------------------------------------
with tab_predict:
    st.subheader("Classify a Custom Message")
    st.caption("Enter feature values manually, or pick a random message from the test set to auto-fill.")

    if st.button("🎲 Load random test message"):
        sample = X_test.sample(1, random_state=np.random.randint(0, 10000))
        st.session_state["sample_idx"] = sample.index[0]

    sample_idx = st.session_state.get("sample_idx", X_test.index[0])
    sample_row = X_test.loc[sample_idx]
    actual_label = y_test.loc[sample_idx]

    if 'message_text' in df.columns:
        st.text_area("Sample message text", df.loc[sample_idx, 'message_text'], height=70, disabled=True)
        st.write(f"Actual label: **{'Spam' if actual_label == 1 else 'Legitimate'}**")

    cols = st.columns(3)
    input_vals = {}
    for i, feat in enumerate(FEATURE_COLS):
        with cols[i % 3]:
            input_vals[feat] = st.number_input(
                feat, value=float(sample_row[feat]), key=f"inp_{feat}"
            )

    if st.button("🔎 Classify Message", type="primary"):
        x_input = pd.DataFrame([input_vals])[FEATURE_COLS]
        x_input_scaled = scaler.transform(x_input)

        knn_p = knn.predict(x_input_scaled)[0]
        svm_p = svm.predict(x_input_scaled)[0]
        nb_p = nb.predict(x_input)[0]
        nb_proba = nb.predict_proba(x_input)[0]

        r1, r2, r3 = st.columns(3)
        for col, name, pred in zip([r1, r2, r3], ["KNN", "SVM", "Naive Bayes"], [knn_p, svm_p, nb_p]):
            label = "🚫 Spam" if pred == 1 else "✅ Legitimate"
            col.metric(name, label)

        st.write(f"**Naive Bayes posterior probabilities:** "
                 f"P(Legitimate) = {nb_proba[0]:.4f}, P(Spam) = {nb_proba[1]:.4f}")

st.markdown("---")
st.caption("Message Intelligence System · Built with Streamlit, scikit-learn, pandas & seaborn")