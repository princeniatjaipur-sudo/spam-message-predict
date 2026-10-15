from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, HISTORY_DB_PATH, MAX_BATCH_ROWS, MAX_FILE_SIZE_MB
from src.database import delete_history, fetch_history, init_history_db, save_prediction_record
from src.data_loader import load_dataset
from src.explainability import explain_message
from src.feedback import approve_feedback, list_feedback, record_feedback, reject_feedback, retrain_with_approved_feedback
from src.predict import batch_predict, predict_single_message
from src.risk_scoring import compute_risk_score
from src.train_model import compare_classifiers, load_model_bundle, train_default_model

st.set_page_config(page_title="SpamShield AI", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: linear-gradient(135deg, #030814 0%, #0a1120 38%, #111c2d 100%);
        color: #edf6ff;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .stApp {
        background-image: radial-gradient(circle at top left, rgba(13, 184, 255, 0.12), transparent 24%),
                          radial-gradient(circle at bottom right, rgba(53, 146, 255, 0.12), transparent 30%);
    }

    h1, h2, h3, h4, h5 {
        color: #f4f8ff;
        letter-spacing: -0.03em;
    }

    .stMetric {
        background: rgba(18, 27, 42, 0.8);
        border: 1px solid rgba(120, 170, 255, 0.18);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.14);
    }

    div[data-testid="stSidebar"] {
        background: rgba(12, 17, 28, 0.88);
        border-right: 1px solid rgba(130, 164, 255, 0.18);
    }

    .stButton > button {
        background: linear-gradient(135deg, #22d3ee, #3b82f6);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        box-shadow: 0 10px 24px rgba(37, 150, 255, 0.25);
    }

    .stButton > button:hover {
        filter: brightness(1.08);
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stFileUploader > div {
        background: rgba(16, 22, 33, 0.8);
        color: #edf6ff;
        border: 1px solid rgba(120, 170, 255, 0.2);
        border-radius: 12px;
    }

    .stDataFrame, .stTable {
        border-radius: 14px;
        overflow: hidden;
    }

    .stAlert {
        border-radius: 14px;
        border: 1px solid rgba(125, 211, 252, 0.22);
        background: rgba(15, 23, 42, 0.85);
    }

    .css-1d391kg, .css-1v0mbdj {
        background: rgba(10, 15, 25, 0.72);
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_model_bundle():
    try:
        return load_model_bundle(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        train_default_model(DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH)
        return load_model_bundle(DEFAULT_MODEL_PATH)


@st.cache_data
def get_dataset_summary():
    df = load_dataset(DEFAULT_DATASET_PATH)
    return df


@st.cache_data
def get_eval_snapshot():
    bundle = get_model_bundle()
    metrics = bundle["metadata"]["metrics"]
    return metrics


def store_prediction(result: dict, category: str = "single") -> None:
    init_history_db(HISTORY_DB_PATH)
    record = {
        "id": result["prediction_id"],
        "timestamp": result["timestamp"],
        "message": result["message"],
        "label": result["prediction"],
        "confidence": result["confidence"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "indicators": result.get("signals", []),
        "metadata": {"category": category, "summary": result.get("summary", "")},
    }
    save_prediction_record(record, HISTORY_DB_PATH)


def home_page() -> None:
    st.markdown(
        """
        <div style="padding: 1.2rem 0 1.6rem 0;">
            <div style="display: inline-block; padding: 0.35rem 0.8rem; border-radius: 999px; background: rgba(34, 211, 238, 0.12); border: 1px solid rgba(34, 211, 238, 0.2); color: #7dd3fc; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.72rem;">
                Security Intelligence • Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("SpamShield AI")
    st.subheader("Explainable message security platform")
    st.write(
        "SpamShield AI analyzes SMS and message text for spam indicators using a trained TF-IDF + Multinomial Naive Bayes classifier. "
        "It surfaces important warning signals while keeping model output separate from heuristic risk scoring."
    )

    col1, col2, col3 = st.columns(3)
    metrics = [
        ("Model", "TF-IDF + Multinomial NB"),
        ("Dataset", "UCI SMS Spam Collection"),
        ("Risk signal", "Transparent heuristic review"),
    ]
    for idx, (label, value) in enumerate(metrics):
        col = [col1, col2, col3][idx]
        col.metric(label, value)

    st.markdown("### Key capabilities")
    st.markdown(
        "- Real spam/ham prediction from a trained pipeline\n"
        "- Explainable security indicators\n"
        "- Risk scores from 0 to 100\n"
        "- Local prediction history and privacy controls\n"
        "- Batch CSV scanning and model performance dashboard"
    )


def message_page() -> None:
    st.title("Message Analyzer")
    message = st.text_area("Paste or type a message to analyze:", height=180, placeholder="Example: WIN a FREE cash bonus now!")

    if st.button("Analyze message") and message.strip():
        with st.spinner("Running the analysis..."):
            result = predict_single_message(message)
            st.success(f"Prediction: {result['prediction'].upper()}")
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            st.metric("Risk score", result["risk_score"])
            st.info(result["summary"])

            if result["signals"]:
                st.markdown("### Detected risk indicators")
                for signal in result["signals"]:
                    st.markdown(f"- {signal['indicator']} ({signal['evidence']})")
            else:
                st.markdown("### Risk indicators")
                st.write("No obvious heuristic alerts were detected for this message.")

            st.markdown("### Recommendation")
            st.warning(
                "If a message asks for passwords, verification codes, payment details, or contains a suspicious link, treat it with caution and verify independently."
            )
            store_prediction(result, category="single")


def bulk_page() -> None:
    st.title("Bulk Scanner")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        return

    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File is too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.")
        return

    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse the CSV file: {exc}")
        return

    if "message" not in data.columns:
        st.error("CSV must include a 'message' column.")
        return

    if len(data) > MAX_BATCH_ROWS:
        st.error(f"Too many rows. Maximum is {MAX_BATCH_ROWS}.")
        return

    messages = data["message"].fillna("").astype(str).tolist()
    with st.spinner("Scanning batch..."):
        results = batch_predict(messages)
    data["prediction"] = [item["prediction"] for item in results]
    data["confidence"] = [item["confidence"] for item in results]
    data["risk_score"] = [item["risk_score"] for item in results]
    st.dataframe(data.head(50), use_container_width=True)
    if results:
        st.download_button(
            "Download result CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="spamshield_batch_results.csv",
            mime="text/csv",
        )


def dashboard_page() -> None:
    st.title("Analytics Dashboard")
    bundle = get_model_bundle()
    metrics = bundle["metadata"].get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    col2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    col3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    col4.metric("F1", f"{metrics.get('f1', 0):.3f}")

    try:
        labels = ["ham", "spam"]
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        cm_df = pd.DataFrame(cm, index=["Actual Ham", "Actual Spam"], columns=["Predicted Ham", "Predicted Spam"])
        st.dataframe(cm_df, use_container_width=True)
    except Exception:
        st.write("Confusion matrix unavailable.")

    history = fetch_history(HISTORY_DB_PATH)
    if history:
        df = pd.DataFrame(history)
        fig = px.histogram(df, x="label", title="Recent prediction distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Prediction history is empty. Run at least one analysis to populate charts.")

    most_recent = fetch_history(HISTORY_DB_PATH, limit=10)
    if most_recent:
        st.subheader("Recent prediction history")
        st.dataframe(pd.DataFrame(most_recent), use_container_width=True)

    feature_names = bundle["pipeline"].named_steps["tfidf"].get_feature_names_out().tolist()
    top_terms = explain_message("free prize claim winner urgent", model=bundle["pipeline"].named_steps["classifier"], feature_names=feature_names)["top_terms"][:10]
    if top_terms:
        term_df = pd.DataFrame(top_terms, columns=["term", "spam_relevance"])
        st.plotly_chart(px.bar(term_df, x="term", y="spam_relevance", title="Most informative spam-associated terms"), use_container_width=True)


def history_page() -> None:
    st.title("Prediction History")
    search = st.text_input("Search messages or labels")
    records = fetch_history(HISTORY_DB_PATH, search=search if search else None)
    if not records:
        st.info("No stored prediction records match the current filter.")
        return

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)

    selected = st.multiselect("Select records to delete", df["id"].tolist())
    if st.button("Delete selected") and selected:
        delete_history(HISTORY_DB_PATH, ids=selected)
        st.rerun()
    if st.button("Clear history"):
        delete_history(HISTORY_DB_PATH)
        st.rerun()

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", data=csv_bytes, file_name="spamshield_history.csv", mime="text/csv")


def feedback_page() -> None:
    st.title("Learning Lab")
    st.write(
        "Approved feedback is stored separately from the benchmark training data. Retraining only occurs after explicit review and approval."
    )

    with st.form("feedback_form"):
        st.subheader("Submit a review")
        message = st.text_area("Message under review", height=160)
        predicted = st.selectbox("Prediction given by the model", ["ham", "spam"])
        correction = st.selectbox("Correct label", ["ham", "spam"])
        notes = st.text_area("Notes or context")
        submitted = st.form_submit_button("Submit feedback")
        if submitted and message.strip():
            record_feedback(message=message.strip(), prediction=predicted, correction=correction, notes=notes)
            st.success("Feedback stored as pending review.")

    st.subheader("Review queue")
    pending = list_feedback(status="pending")
    if not pending:
        st.info("No pending feedback yet.")
    else:
        for item in pending:
            with st.container():
                st.markdown(f"**Message:** {item['message']}")
                st.write(f"Prediction: {item['prediction']} | Correction: {item['correction']} | Notes: {item['notes']}")
                col1, col2 = st.columns(2)
                if col1.button("Approve", key=f"approve_{item['id']}"):
                    approve_feedback(item["id"])
                    st.rerun()
                if col2.button("Reject", key=f"reject_{item['id']}"):
                    reject_feedback(item["id"])
                    st.rerun()

    approved = list_feedback(status="approved")
    st.subheader("Approved feedback")
    if not approved:
        st.info("No approved examples are available.")
    else:
        st.dataframe(pd.DataFrame(approved), use_container_width=True)

    if st.button("Retrain model with approved feedback") and approved:
        try:
            result = retrain_with_approved_feedback(DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, DEFAULT_MODEL_PATH)
            st.success(f"Retraining finished. Selected model: {result['selected_model']}")
            st.json(result)
        except Exception as exc:
            st.error(f"Retraining failed: {exc}")


def model_page() -> None:
    st.title("Model Information")
    bundle = get_model_bundle()
    metadata = bundle["metadata"]
    st.write("### Model metadata")
    st.json(metadata)
    st.write("### Dataset details")
    dataset = load_dataset(DEFAULT_DATASET_PATH)
    st.write(f"Rows: {len(dataset)}")
    st.write(dataset["label"].value_counts().to_dict())
    st.write("### Reliability notes")
    st.warning(
        "Machine learning predictions can be mistaken. This system is a decision-support tool and should not be treated as a guarantee of safety."
    )


def main() -> None:
    pages = {
        "Home": home_page,
        "Message Analyzer": message_page,
        "Bulk Scanner": bulk_page,
        "Analytics Dashboard": dashboard_page,
        "Prediction History": history_page,
        "Learning Lab": feedback_page,
        "Model Information": model_page,
    }
    page_name = st.sidebar.selectbox("Navigation", list(pages.keys()))
    pages[page_name]()


if __name__ == "__main__":
    main()
