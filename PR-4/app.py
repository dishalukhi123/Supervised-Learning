"""
Message Intelligence System - Streamlit App
----------------------------------------------
Spam vs Legitimate message classifier comparing KNN, SVM (Linear & RBF),
and Naive Bayes, built on top of Message_Intelligence_Dataset.csv.

Run with:
    streamlit run app.py
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

st.set_page_config(
    page_title="Message Intelligence System",
    page_icon="📨",
    layout="wide"
)

FEATURE_COLS = [
    'message_length', 'word_count', 'num_urls', 'num_digits',
    'num_special_chars', 'spam_keyword_score', 'legit_keyword_score',
    'sender_activity_score', 'sender_account_age_days',
    'messages_sent_last_24h', 'hour_of_day', 'day_of_week'
]

DATA_PATH = "Message_Intelligence_Dataset.csv"


# ----------------------------------------------------------------------
# Data + model loading (cached so it only runs once per session)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def train_models(df):
    X = df[FEATURE_COLS].copy()
    y = df['spam_label'].copy()

    for c in X.columns[X.isna().any()]:
        X[c] = X[c].fillna(X[c].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- KNN: scan K to find the best by F1, then fit final model ----
    k_values = list(range(1, 22, 2))
    knn_rows = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train_scaled, y_train)
        pred = knn.predict(X_test_scaled)
        knn_rows.append({
            'K': k,
            'Accuracy': accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred),
            'Recall': recall_score(y_test, pred),
            'F1': f1_score(y_test, pred)
        })
    knn_df = pd.DataFrame(knn_rows)
    best_k = int(knn_df.loc[knn_df['F1'].idxmax(), 'K'])

    knn_final = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn_final.fit(X_train_scaled, y_train)
    knn_pred = knn_final.predict(X_test_scaled)

    # ---- SVM: Linear + RBF ----
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    svm_linear.fit(X_train_scaled, y_train)
    pred_lin = svm_linear.predict(X_test_scaled)

    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    svm_rbf.fit(X_train_scaled, y_train)
    pred_rbf = svm_rbf.predict(X_test_scaled)

    # ---- Naive Bayes (uses unscaled features) ----
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)

    # ---- Metrics table ----
    results = pd.DataFrame({
        'Model': ['KNN (best K)', 'Linear SVM', 'RBF SVM', 'Naive Bayes'],
        'Type': ['Distance-based', 'Margin-based', 'Margin-based', 'Probabilistic'],
        'Accuracy': [accuracy_score(y_test, knn_pred), accuracy_score(y_test, pred_lin),
                     accuracy_score(y_test, pred_rbf), accuracy_score(y_test, nb_pred)],
        'Precision': [precision_score(y_test, knn_pred), precision_score(y_test, pred_lin),
                      precision_score(y_test, pred_rbf), precision_score(y_test, nb_pred)],
        'Recall': [recall_score(y_test, knn_pred), recall_score(y_test, pred_lin),
                   recall_score(y_test, pred_rbf), recall_score(y_test, nb_pred)],
        'F1 Score': [f1_score(y_test, knn_pred), f1_score(y_test, pred_lin),
                     f1_score(y_test, pred_rbf), f1_score(y_test, nb_pred)]
    }).round(4)

    predictions = {
        'KNN (best K)': knn_pred,
        'Linear SVM': pred_lin,
        'RBF SVM': pred_rbf,
        'Naive Bayes': nb_pred,
    }

    return {
        'scaler': scaler,
        'knn_final': knn_final,
        'best_k': best_k,
        'knn_scan': knn_df,
        'svm_linear': svm_linear,
        'svm_rbf': svm_rbf,
        'nb': nb,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'results': results,
        'predictions': predictions,
    }


def extract_features_from_text(text, sender_activity_score, sender_account_age_days,
                                messages_sent_last_24h, hour_of_day, day_of_week):
    """Roughly mimic the dataset's text-derived features for a typed message."""
    spam_words = ['free', 'win', 'winner', 'cash', 'prize', 'urgent', 'click',
                  'offer', 'limited', 'act now', 'congratulations', 'claim',
                  'discount', 'guarantee', 'risk-free', 'verify']
    legit_words = ['meeting', 'report', 'invoice', 'project', 'schedule',
                   'thanks', 'please', 'confirm', 'attached', 'regards']

    text_lower = text.lower()
    message_length = len(text)
    word_count = len(text.split())
    num_urls = len(re.findall(r'https?://\S+|www\.\S+', text_lower))
    num_digits = sum(c.isdigit() for c in text)
    num_special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    spam_keyword_score = sum(text_lower.count(w) for w in spam_words)
    legit_keyword_score = sum(text_lower.count(w) for w in legit_words)

    return {
        'message_length': message_length,
        'word_count': word_count,
        'num_urls': num_urls,
        'num_digits': num_digits,
        'num_special_chars': num_special_chars,
        'spam_keyword_score': spam_keyword_score,
        'legit_keyword_score': legit_keyword_score,
        'sender_activity_score': sender_activity_score,
        'sender_account_age_days': sender_account_age_days,
        'messages_sent_last_24h': messages_sent_last_24h,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
    }


def plot_confusion(y_test, y_pred, title, cmap):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.3))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Legit', 'Spam'], yticklabels=['Legit', 'Spam'])
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# App layout
# ----------------------------------------------------------------------
st.title("📨 Message Intelligence System")
st.caption("Spam vs Legitimate message classification — KNN · SVM · Naive Bayes")

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`. Place the dataset CSV in the same "
        "folder as app.py before running."
    )
    st.stop()

models = train_models(df)
results = models['results']

tab_overview, tab_predict, tab_models, tab_compare, tab_data = st.tabs(
    ["🏠 Overview", "🔮 Try a Message", "🧩 Model Details", "📊 Comparison", "📁 Dataset"]
)

# ---------------- Overview ----------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Messages", len(df))
    c2.metric("Spam %", f"{df['spam_label'].mean()*100:.1f}%")
    c3.metric("Best K (KNN)", models['best_k'])
    c4.metric("Best Model (F1)", results.loc[results['F1 Score'].idxmax(), 'Model'])

    st.subheader("Model Performance Summary")
    st.dataframe(results.set_index('Model'), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    results.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(
        kind='bar', ax=ax, colormap='Set2'
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: KNN vs SVM (Linear/RBF) vs Naive Bayes')
    plt.xticks(rotation=0)
    fig.tight_layout()
    st.pyplot(fig)

    best_precision = results.loc[results['Precision'].idxmax()]
    best_recall = results.loc[results['Recall'].idxmax()]
    colA, colB = st.columns(2)
    with colA:
        st.info(
            f"**Best for High Precision:** {best_precision['Model']} "
            f"({best_precision['Precision']:.3f})\n\n"
            "Use when false positives (flagging a real message as spam) are costly."
        )
    with colB:
        st.warning(
            f"**Best for High Recall:** {best_recall['Model']} "
            f"({best_recall['Recall']:.3f})\n\n"
            "Use when false negatives (letting spam through) are costly."
        )

# ---------------- Try a Message ----------------
with tab_predict:
    st.subheader("Classify a New Message")
    st.write("Enter a message and some sender context — all three models will vote.")

    msg_text = st.text_area(
        "Message text",
        "Congratulations! You have WON a free prize. Click here to claim now!!!",
        height=100
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        sender_activity_score = st.slider("Sender activity score", 0.0, 100.0, 25.0)
        sender_account_age_days = st.slider("Sender account age (days)", 0, 1000, 300)
    with c2:
        messages_sent_last_24h = st.slider("Messages sent last 24h", 0, 50, 5)
        hour_of_day = st.slider("Hour of day", 0, 23, 14)
    with c3:
        day_of_week = st.slider("Day of week (0=Mon)", 0, 6, 2)
        model_choice = st.selectbox(
            "Model to highlight",
            ["All Models", "KNN", "Linear SVM", "RBF SVM", "Naive Bayes"]
        )

    if st.button("🔍 Classify Message", type="primary"):
        feats = extract_features_from_text(
            msg_text, sender_activity_score, sender_account_age_days,
            messages_sent_last_24h, hour_of_day, day_of_week
        )
        x_row = pd.DataFrame([feats])[FEATURE_COLS]
        x_scaled = models['scaler'].transform(x_row)

        knn_p = models['knn_final'].predict(x_scaled)[0]
        knn_proba = models['knn_final'].predict_proba(x_scaled)[0]
        lin_p = models['svm_linear'].predict(x_scaled)[0]
        lin_proba = models['svm_linear'].predict_proba(x_scaled)[0]
        rbf_p = models['svm_rbf'].predict(x_scaled)[0]
        rbf_proba = models['svm_rbf'].predict_proba(x_scaled)[0]
        nb_p = models['nb'].predict(x_row)[0]
        nb_proba = models['nb'].predict_proba(x_row)[0]

        label = lambda v: "🚨 SPAM" if v == 1 else "✅ Legitimate"

        result_table = pd.DataFrame({
            'Model': ['KNN', 'Linear SVM', 'RBF SVM', 'Naive Bayes'],
            'Prediction': [label(knn_p), label(lin_p), label(rbf_p), label(nb_p)],
            'P(Spam)': [knn_proba[1], lin_proba[1], rbf_proba[1], nb_proba[1]],
        }).round(4)

        st.dataframe(result_table.set_index('Model'), use_container_width=True)

        votes = [knn_p, lin_p, rbf_p, nb_p]
        majority = "🚨 SPAM" if sum(votes) >= 2 else "✅ Legitimate"
        st.markdown(f"### Majority Vote: {majority}")

        with st.expander("Extracted features used for prediction"):
            st.json(feats)

# ---------------- Model Details ----------------
with tab_models:
    st.subheader("KNN — K selection")
    knn_scan = models['knn_scan']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(knn_scan['K'], knn_scan['Accuracy'], marker='o', label='Accuracy')
    ax.plot(knn_scan['K'], knn_scan['F1'], marker='s', label='F1 Score')
    ax.set_xlabel('K (number of neighbors)')
    ax.set_ylabel('Score')
    ax.set_title('KNN Performance vs K')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(f"Best K selected by F1 score: **{models['best_k']}**")
    st.dataframe(knn_scan.round(4), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("SVM — Support Vectors & Margin")
    svm_lin = models['svm_linear']
    svm_rbf = models['svm_rbf']
    n_train = models['X_train_scaled'].shape[0]
    w = svm_lin.coef_[0]
    margin_width = 2 / np.linalg.norm(w)

    c1, c2 = st.columns(2)
    c1.metric("Linear SVM support vectors",
              svm_lin.support_vectors_.shape[0],
              f"{svm_lin.support_vectors_.shape[0]/n_train*100:.1f}% of training data")
    c2.metric("RBF SVM support vectors",
              svm_rbf.support_vectors_.shape[0],
              f"{svm_rbf.support_vectors_.shape[0]/n_train*100:.1f}% of training data")
    st.write(f"**Linear SVM margin width (2/‖w‖):** {margin_width:.4f}")
    st.caption(
        "Fewer support vectors with a wide margin suggests well-separated classes. "
        "RBF typically uses more support vectors since it fits a more flexible, "
        "non-linear boundary."
    )

    if st.checkbox("Show 2D PCA decision boundary (Linear SVM)"):
        pca = PCA(n_components=2, random_state=42)
        X_train_2d = pca.fit_transform(models['X_train_scaled'])
        svm_2d = SVC(kernel='linear', C=1.0).fit(X_train_2d, models['y_train'])

        xx, yy = np.meshgrid(
            np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 300),
            np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 300)
        )
        Z = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#cfe2f3', '#f9cb9c'], alpha=0.5)
        ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
        ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=models['y_train'],
                   cmap='coolwarm', s=15, edgecolors='k', linewidths=0.3)
        ax.scatter(svm_2d.support_vectors_[:, 0], svm_2d.support_vectors_[:, 1],
                   s=80, facecolors='none', edgecolors='lime', linewidths=1.5,
                   label='Support Vectors')
        ax.set_title('Linear SVM Decision Boundary & Margin (PCA-projected, 2D)')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    st.divider()
    st.subheader("Naive Bayes — Bayes' Theorem Walkthrough")
    nb = models['nb']
    st.write(
        f"Class priors learned: **P(Legitimate) = {nb.class_prior_[0]:.4f}**, "
        f"**P(Spam) = {nb.class_prior_[1]:.4f}**"
    )
    st.latex(r"P(C \mid X) = \frac{P(X \mid C)\,P(C)}{P(X)} \;\propto\; "
             r"P(C)\prod_{i=1}^{n} P(x_i \mid C)")

    sample_idx = models['X_test'].index[:3]
    samples = models['X_test'].loc[sample_idx]
    sklearn_proba = nb.predict_proba(samples)
    actual = models['y_test'].loc[sample_idx]
    compare_table = pd.DataFrame({
        'message_id': sample_idx,
        'message_text': df.loc[sample_idx, 'message_text'].str.slice(0, 60) + "...",
        'actual': ['Spam' if v == 1 else 'Legitimate' for v in actual.values],
        'P(Legit)': sklearn_proba[:, 0].round(6),
        'P(Spam)': sklearn_proba[:, 1].round(6),
    })
    st.dataframe(compare_table.set_index('message_id'), use_container_width=True)
    st.caption(
        "Manually computing the Gaussian likelihoods per feature, multiplying by the "
        "class prior, and normalizing reproduces these same posterior probabilities "
        "(see notebook for the hand-derived calculation)."
    )

# ---------------- Comparison ----------------
with tab_compare:
    st.subheader("Confusion Matrices")
    y_test = models['y_test']
    preds = models['predictions']
    cols = st.columns(4)
    cmaps = ['Blues', 'Purples', 'Oranges', 'Greens']
    for col, (name, pred), cmap in zip(cols, preds.items(), cmaps):
        with col:
            fig = plot_confusion(y_test, pred, name, cmap)
            st.pyplot(fig)

    st.divider()
    st.subheader("Classification Reports")
    chosen = st.selectbox("Choose a model", list(preds.keys()))
    report = classification_report(
        y_test, preds[chosen], target_names=['Legitimate', 'Spam'], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    st.divider()
    st.subheader("Final Recommendation")
    st.markdown(
        """
- **Naive Bayes** is fast, interpretable, and gives calibrated probabilities — good baseline.
- **KNN** is simple and non-parametric but slows down on large data and is sensitive to scaling.
- **SVM (Linear/RBF)** often gives the best margin-based separation; RBF captures non-linear
  patterns at the cost of interpretability and more support vectors.
- For a production spam filter, prioritize **recall** to catch as much spam as possible,
  but monitor **precision** so legitimate messages aren't lost — an ensemble/voting
  approach (as shown in the *Try a Message* tab) balances both.
        """
    )

# ---------------- Dataset ----------------
with tab_data:
    st.subheader("Raw Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Feature Correlation with Spam Label")
    corr = df[FEATURE_COLS + ['spam_label']].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Class Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        df['spam_label'].value_counts().plot(kind='bar', ax=ax, color=['#4C72B0', '#DD8452'])
        ax.set_xticklabels(['Legitimate (0)', 'Spam (1)'], rotation=0)
        ax.set_title('Class Distribution (counts)')
        fig.tight_layout()
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        df['spam_label'].value_counts().plot(
            kind='pie', autopct='%1.1f%%', ax=ax,
            labels=['Legitimate', 'Spam'], colors=['#4C72B0', '#DD8452']
        )
        ax.set_ylabel('')
        ax.set_title('Class Distribution (%)')
        fig.tight_layout()
        st.pyplot(fig)
