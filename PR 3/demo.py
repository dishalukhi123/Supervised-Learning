import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

st.set_page_config(page_title="Risk Alert Classifier", layout="wide")

st.title("🏦 Risk Alert Classifier")
st.write("Dataset Understanding & Preparation")

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload Risk_Alert_Classifier_Dataset.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    # Remove extra spaces
    df.columns = df.columns.str.strip()

    # Find target column automatically
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
        st.error("Target column not found.")
        st.stop()

    # Features and Target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    st.subheader("Input Features")
    st.write(list(X.columns))

    st.subheader("Target Variable")
    st.write(target_col)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    st.subheader("Train-Test Split")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training Rows", X_train.shape[0])

    with col2:
        st.metric("Testing Rows", X_test.shape[0])

    # Missing Values
    st.subheader("Missing Values")

    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0])

    # KNN Imputation
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if len(numeric_cols) > 0:

        imputer = KNNImputer(n_neighbors=5)

        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        st.success("KNN Imputation Applied Successfully")

        st.subheader("Missing Values After Imputation")
        st.dataframe(df[numeric_cols].isnull().sum())

    st.subheader("Final Dataset")
    st.dataframe(df.head())

else:
    st.info("Please upload the dataset file.")