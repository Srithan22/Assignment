import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Fraud Intelligence Dashboard", layout="wide")

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("AI Fraud Intelligence Dashboard")

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Class" in df.columns:

        fraud_pct = (df["Class"] == 1).mean() * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fraud %", f"{fraud_pct:.2f}%")

        with col2:
            st.metric("Legitimate %", f"{100-fraud_pct:.2f}%")

    fig = px.histogram(
        df,
        x="Amount",
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(fig)

    X = df.drop(columns=["Class"], errors="ignore")

    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[:, 1]

    df["Fraud_Probability"] = probs

    st.subheader("Top High Risk Transactions")

    st.dataframe(
        df.sort_values(
            "Fraud_Probability",
            ascending=False
        ).head(20)
    )

    high_risk = df[df["Fraud_Probability"] > 0.8]

    st.subheader("High Risk Transactions")
    st.dataframe(high_risk)

    fig2 = px.histogram(
        df,
        x="Fraud_Probability",
        nbins=50
    )

    st.plotly_chart(fig2)