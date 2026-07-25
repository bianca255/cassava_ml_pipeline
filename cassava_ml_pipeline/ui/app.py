"""
ui/app.py
---------
Streamlit UI for the Cassava Leaf Disease Classification pipeline.

Tabs:
    1. Predict       - upload a single leaf image, get a prediction
    2. Visualizations - dataset class distribution + per-class image stats
    3. Upload & Retrain - bulk-upload new images and trigger retraining
    4. Model Uptime      - API/service uptime + latest eval metrics

Run with:
    streamlit run ui/app.py

Set API_URL env var if the FastAPI backend is not on localhost:8000.
"""

import os
import time

import requests
import streamlit as st
import pandas as pd
import plotly.express as px

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Cassava Disease Classifier", layout="wide")
st.title("🌿 Cassava Leaf Disease Classification")

tab_predict, tab_viz, tab_retrain, tab_uptime = st.tabs(
    ["🔍 Predict", "📊 Visualizations", "📤 Upload & Retrain", "⏱️ Model Uptime"]
)

# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict a single leaf image")
    uploaded_file = st.file_uploader("Upload a cassava leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", width=300)
        if st.button("Run Prediction"):
            with st.spinner("Contacting model..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(f"Prediction: **{result['predicted_class']}** "
                                   f"({result['confidence']*100:.1f}% confidence)")
                        probs_df = pd.DataFrame(
                            list(result["all_class_probabilities"].items()),
                            columns=["Class", "Probability"],
                        ).sort_values("Probability", ascending=False)
                        st.bar_chart(probs_df.set_index("Class"))
                    else:
                        st.error(f"API error: {resp.status_code} - {resp.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach API at {API_URL}: {e}")

# ---------------------------------------------------------------------------
# Tab 2: Visualizations
# ---------------------------------------------------------------------------
with tab_viz:
    st.subheader("Dataset insights")

    if st.button("Refresh visualizations"):
        st.rerun()

    try:
        dist_resp = requests.get(f"{API_URL}/visualizations/class-distribution", timeout=15)
        stats_resp = requests.get(f"{API_URL}/visualizations/image-stats", timeout=15)

        if dist_resp.status_code == 200:
            dist = dist_resp.json()
            df_dist = pd.DataFrame(list(dist.items()), columns=["Class", "Count"])
            st.markdown("**1. Class distribution** — reveals class imbalance, which explains "
                        "why the model uses class-weighted loss / augmentation during training.")
            fig = px.bar(df_dist, x="Class", y="Count", color="Class")
            st.plotly_chart(fig, use_container_width=True)

        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            if stats:
                df_stats = pd.DataFrame(stats).T.reset_index().rename(columns={"index": "Class"})
                st.markdown("**2. Average brightness per class** — some diseases (e.g. CBSD) "
                            "cause visible discoloration, which shows up as brightness shifts.")
                fig2 = px.bar(df_stats, x="Class", y="avg_brightness", color="Class")
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("**3. Average image resolution per class** — checks whether image "
                            "capture conditions were consistent across classes (a potential data quality issue).")
                fig3 = px.bar(df_stats, x="Class", y=["avg_width", "avg_height"], barmode="group")
                st.plotly_chart(fig3, use_container_width=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API at {API_URL}: {e}")

# ---------------------------------------------------------------------------
# Tab 3: Upload & Retrain
# ---------------------------------------------------------------------------
with tab_retrain:
    st.subheader("Bulk upload new training images")
    label = st.selectbox(
        "Label for uploaded images",
        [
            "Cassava Bacterial Blight (CBB)",
            "Cassava Brown Streak Disease (CBSD)",
            "Cassava Green Mottle (CGM)",
            "Cassava Mosaic Disease (CMD)",
            "Healthy",
        ],
    )
    bulk_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if st.button("Upload batch") and bulk_files:
        files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in bulk_files]
        resp = requests.post(f"{API_URL}/upload", data={"label": label}, files=files_payload, timeout=60)
        if resp.status_code == 200:
            st.success(resp.json()["message"] + f" ({resp.json()['ingested']} images ingested)")
        else:
            st.error(f"Upload failed: {resp.text}")

    st.divider()
    st.subheader("Trigger retraining")
    epochs = st.slider("Epochs", min_value=1, max_value=20, value=5)
    if st.button("🚀 Retrain model now"):
        resp = requests.post(f"{API_URL}/retrain", params={"epochs": epochs}, timeout=15)
        if resp.status_code == 200:
            st.info("Retraining triggered in the background. Poll status below.")
        else:
            st.error(f"Could not trigger retraining: {resp.text}")

    if st.button("Check retraining status"):
        status_resp = requests.get(f"{API_URL}/retrain/status", timeout=15)
        st.json(status_resp.json())

# ---------------------------------------------------------------------------
# Tab 4: Model Uptime
# ---------------------------------------------------------------------------
with tab_uptime:
    st.subheader("Service uptime")
    try:
        up_resp = requests.get(f"{API_URL}/uptime", timeout=10)
        if up_resp.status_code == 200:
            st.metric("Uptime", up_resp.json()["uptime_human"])
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API at {API_URL}: {e}")

    st.subheader("Latest evaluation metrics")
    try:
        m_resp = requests.get(f"{API_URL}/metrics", timeout=10)
        if m_resp.status_code == 200:
            m = m_resp.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            c2.metric("Precision (macro)", f"{m['precision_macro']*100:.1f}%")
            c3.metric("Recall (macro)", f"{m['recall_macro']*100:.1f}%")
            c4.metric("F1 (macro)", f"{m['f1_macro']*100:.1f}%")
            st.caption(f"Last evaluated: {m['evaluated_at']}")
        else:
            st.info("No metrics available yet. Train or retrain the model first.")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API at {API_URL}: {e}")
