"""Smoke-test the Streamlit prediction flows with Streamlit's AppTest harness."""

import io
import json
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "app.py"
FEATURES = [
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


def assert_no_app_errors(app):
    errors = [(item.type, item.message) for item in app.exception]
    assert not errors, errors


def main():
    model = joblib.load(ROOT / "models" / "churn_model.pkl")
    data = pd.read_csv(ROOT / "dataset" / "customer_churn.csv")
    metrics = json.loads((ROOT / "model_metrics.json").read_text(encoding="utf-8"))
    threshold = float(metrics["threshold_selection"]["threshold"])

    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert_no_app_errors(app)
    assert app.radio[0].value == "Overview"
    print("PASS: app starts, model and overview load")

    app.radio[0].set_value("Customer Prediction").run()
    assert_no_app_errors(app)
    customer = {widget.label: widget.value for widget in app.number_input}
    customer.update({widget.label: widget.value for widget in app.selectbox})
    single_row = pd.DataFrame([{column: customer[column] for column in FEATURES}])
    probability = float(model.predict_proba(single_row)[0, 1])
    prediction = "Likely to Churn" if probability >= threshold else "Likely to Stay"

    app.button[0].click().run()
    assert_no_app_errors(app)
    shown_metrics = {item.label: item.value for item in app.metric}
    assert shown_metrics["Churn Probability"] == f"{probability:.1%}"
    assert shown_metrics["Prediction"] == prediction
    assert any(
        item.value == f"Decision threshold: {threshold:.3f}"
        for item in app.caption
    )
    print("PASS: individual probability, decision, and saved threshold display")

    batch = data.iloc[:5][FEATURES].copy()
    app.radio[0].set_value("Batch Prediction").run()
    app.file_uploader[0].set_value(
        ("sample_customers.csv", batch.to_csv(index=False).encode("utf-8"), "text/csv")
    ).run()
    assert_no_app_errors(app)
    assert app.success[0].value == "5 customers predicted"
    result = app.dataframe[0].value
    expected_probability = model.predict_proba(batch)[:, 1]
    expected_prediction = (expected_probability >= threshold).astype(int)
    assert list(result["predicted_churn"]) == expected_prediction.tolist()
    assert all(
        abs(float(actual) - float(expected)) < 1e-12
        for actual, expected in zip(result["churn_probability"], expected_probability)
    )
    assert any(
        item.value == f"Decision threshold: {threshold:.3f}"
        for item in app.caption
    )

    download = app.get("download_button")[0]
    assert download.proto.label == "Download Results"
    assert download.proto.url.endswith(".csv")
    download.click().run()
    assert_no_app_errors(app)
    assert app.get("download_button")[0].value is True
    exported = pd.read_csv(io.BytesIO(result.to_csv(index=False).encode("utf-8")))
    assert len(exported) == 5
    assert list(exported.columns[-2:]) == ["churn_probability", "predicted_churn"]
    print("PASS: batch upload, predictions, threshold, and CSV download registration")

    invalid_cases = [
        (
            "missing.csv",
            batch.drop(columns=["Age"]).to_csv(index=False),
            "Missing required columns",
        ),
        (
            "nonnumeric.csv",
            batch.assign(Age="invalid").to_csv(index=False),
            "must contain finite numeric values",
        ),
        (
            "out_of_range.csv",
            batch.assign(Age=10000).to_csv(index=False),
            "values must be between",
        ),
        (
            "unknown_category.csv",
            batch.assign(Gender="unknown").to_csv(index=False),
            "Unknown 'Gender' value",
        ),
        (
            "empty.csv",
            ",".join(FEATURES) + "\n",
            "no customer rows",
        ),
        ("blank.csv", "", "Unable to read the uploaded CSV"),
    ]
    for filename, content, expected_message in invalid_cases:
        app.file_uploader[0].set_value(
            (filename, content.encode("utf-8"), "text/csv")
        ).run()
        assert_no_app_errors(app)
        errors = [item.value for item in app.error]
        assert errors and expected_message in errors[-1], (filename, errors)
    print(f"PASS: {len(invalid_cases)} invalid-upload cases show clear validation errors")
    print("All Streamlit AppTest checks passed.")


if __name__ == "__main__":
    main()
