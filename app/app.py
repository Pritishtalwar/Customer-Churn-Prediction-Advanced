from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "customer_churn.csv"
MODEL = ROOT / "models" / "churn_model.pkl"
METRICS = ROOT / "model_metrics.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL)


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


st.set_page_config(
    page_title="Churn Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL.exists():
    st.error("Model not found. Run `python scripts/train_models.py` first.")
    st.stop()

model = load_model()
df = load_data()

with st.sidebar:
    st.title("Churn Intelligence")
    page = st.radio(
        "Navigation",
        ["Overview", "Customer Prediction", "Batch Prediction"],
        label_visibility="collapsed",
    )
    st.caption("Customer retention analytics")

st.title("Customer Churn Prediction")

feature_columns = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
    c3.metric("Model Features", len(feature_columns))
    c4.metric("Target", "Churn")

    left, right = st.columns([1.3, 1])
    with left:
        counts = df["Churn"].value_counts().sort_index()
        fig = px.bar(
            x=["Stay", "Churn"],
            y=[int(counts.get(0, 0)), int(counts.get(1, 0))],
            text=[int(counts.get(0, 0)), int(counts.get(1, 0))],
            labels={"x": "Outcome", "y": "Customers"},
            color=["Stay", "Churn"],
            color_discrete_map={"Stay": "#94a3b8", "Churn": "#ef4444"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model status")
        if METRICS.exists():
            import json

            metrics = json.loads(METRICS.read_text(encoding="utf-8"))
            st.success(f"Selected: {metrics['selected_model']}")
            st.metric("Test Accuracy", f"{metrics['final_test']['accuracy']:.1%}")
            st.metric("Test ROC-AUC", f"{metrics['final_test']['roc_auc']:.1%}")
        else:
            st.warning("Run model training to generate metrics.")

elif page == "Customer Prediction":
    st.subheader("Customer Details")
    left, right = st.columns(2)

    customer = {}
    with left:
        customer["Age"] = st.number_input(
            "Age",
            int(df["Age"].min()),
            int(df["Age"].max()),
            int(df["Age"].median()),
        )
        customer["Tenure"] = st.number_input(
            "Tenure",
            int(df["Tenure"].min()),
            int(df["Tenure"].max()),
            int(df["Tenure"].median()),
        )
        customer["Usage Frequency"] = st.number_input(
            "Usage Frequency",
            int(df["Usage Frequency"].min()),
            int(df["Usage Frequency"].max()),
            int(df["Usage Frequency"].median()),
        )
        customer["Support Calls"] = st.number_input(
            "Support Calls",
            int(df["Support Calls"].min()),
            int(df["Support Calls"].max()),
            int(df["Support Calls"].median()),
        )
        customer["Payment Delay"] = st.number_input(
            "Payment Delay",
            int(df["Payment Delay"].min()),
            int(df["Payment Delay"].max()),
            int(df["Payment Delay"].median()),
        )

    with right:
        customer["Total Spend"] = st.number_input(
            "Total Spend",
            float(df["Total Spend"].min()),
            float(df["Total Spend"].max()),
            float(df["Total Spend"].median()),
        )
        customer["Last Interaction"] = st.number_input(
            "Last Interaction",
            int(df["Last Interaction"].min()),
            int(df["Last Interaction"].max()),
            int(df["Last Interaction"].median()),
        )
        customer["Gender"] = st.selectbox(
            "Gender",
            sorted(df["Gender"].dropna().astype(str).unique()),
        )
        customer["Subscription Type"] = st.selectbox(
            "Subscription Type",
            sorted(df["Subscription Type"].dropna().astype(str).unique()),
        )
        customer["Contract Length"] = st.selectbox(
            "Contract Length",
            sorted(df["Contract Length"].dropna().astype(str).unique()),
        )

    if st.button("Predict Churn", type="primary"):
        row = pd.DataFrame([customer])[feature_columns]
        probability = float(model.predict_proba(row)[0, 1])
        threshold = 0.5
        if METRICS.exists():
            import json

            threshold = float(
                json.loads(METRICS.read_text(encoding="utf-8"))
                .get("threshold_selection", {})
                .get("threshold", 0.5)
            )

        a, b = st.columns(2)
        a.metric("Churn Probability", f"{probability:.1%}")
        b.metric("Prediction", "Likely to Churn" if probability >= threshold else "Likely to Stay")

        if probability >= threshold:
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is likely to stay.")

else:
    st.subheader("Batch CSV Prediction")
    uploaded = st.file_uploader("Upload a customer CSV", type=["csv"])
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            missing = [column for column in feature_columns if column not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                probabilities = model.predict_proba(batch[feature_columns])[:, 1]
                threshold = 0.5
                if METRICS.exists():
                    import json

                    threshold = float(
                        json.loads(METRICS.read_text(encoding="utf-8"))
                        .get("threshold_selection", {})
                        .get("threshold", 0.5)
                    )
                predictions = (probabilities >= threshold).astype(int)
                result = batch.copy()
                result["churn_probability"] = probabilities
                result["predicted_churn"] = predictions
                st.success(f"{len(result):,} customers predicted")
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download Results",
                    result.to_csv(index=False).encode("utf-8"),
                    "churn_predictions.csv",
                    "text/csv",
                )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
