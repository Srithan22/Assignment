import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px

st.set_page_config(page_title="Fraud Intelligence Dashboard", layout="wide")

model = tf.keras.models.load_model("lstm_attention_model.keras")
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
            st.metric(
                "Fraud %",
                f"{fraud_pct:.2f}%"
            )

        with col2:
            st.metric(
                "Legitimate %",
                f"{100-fraud_pct:.2f}%"
            )

    fig = px.histogram(
        df,
        x="Amount",
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    features = df.drop(
        columns=["Class"],
        errors="ignore"
    )

    features_scaled = scaler.transform(features)

    preds = model.predict(
        features_scaled,
        verbose=0
    )

    df["Fraud_Probability"] = preds

    st.subheader("Fraud Predictions")

    st.dataframe(
        df[
            ["Fraud_Probability"]
        ].sort_values(
            "Fraud_Probability",
            ascending=False
        ).head(20)
    )

    high_risk = df[
        df["Fraud_Probability"] > 0.8
    ]

    st.subheader("High Risk Transactions")

    st.dataframe(high_risk)

    st.subheader("Fraud Probability Distribution")

    fig2 = px.histogram(
        df,
        x="Fraud_Probability",
        nbins=50
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.sidebar.header("Project")

st.sidebar.write(
    """
    Dense Network
    LSTM
    LSTM + Attention
    Fraud Detection Dashboard
    """
)