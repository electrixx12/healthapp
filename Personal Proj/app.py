import streamlit as st
import joblib
import numpy as np

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(layout="centered")
st.title("🩺 Multi-Disease Predictor (Logistic Regression)")

# ─── LOAD MODEL & METADATA ────────────────────────────────────
data = joblib.load("disease_multi_model_lr.pkl", mmap_mode="r")
model         = data["model"]
label_encoder = data["label_encoder"]
feature_cols  = data["feature_cols"]

# ─── UI ────────────────────────────────────────────────────────
st.markdown("Select **all** the symptoms you’re experiencing:")
selected = st.multiselect("Your symptoms:", options=feature_cols)

if st.button("🔍 Predict Disease"):
    if not selected:
        st.warning("Please pick at least one symptom.")
    else:
        # Build the 0/1 feature vector
        X_input = np.array([[1 if feat in selected else 0 for feat in feature_cols]])
        # Predict
        pred    = model.predict(X_input)[0]
        proba   = model.predict_proba(X_input)[0]
        disease    = label_encoder.inverse_transform([pred])[0]
        confidence = np.max(proba) * 100

        st.success(f"**Predicted Disease:** {disease}")
        st.info(f"Confidence: {confidence:.1f}%")


