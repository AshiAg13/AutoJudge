import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AutoJudge - Programming Difficulty Predictor",
    layout="centered"
)

# --- BASE DIRECTORY ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOAD MODELS & ASSETS ---
@st.cache_resource
def load_assets():
    tfidf = joblib.load(os.path.join(BASE_DIR, "autojudge_tfidf.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "autojudge_scaler.pkl"))
    classifier = joblib.load(os.path.join(BASE_DIR, "autojudge_classifier.pkl"))
    regressor = joblib.load(os.path.join(BASE_DIR, "autojudge_regressor.pkl"))
    return tfidf, scaler, classifier, regressor

tfidf, scaler, clf, reg = load_assets()

# --- FEATURE EXTRACTION ---
keywords = ["dp", "graph", "recursion", "tree", "greedy", "bfs", "dfs"]

def extract_features(text):
    length = len(text.split())

    math_symbols = "+-*/=<>%"
    sym_count = sum(text.count(sym) for sym in math_symbols)

    kw_counts = [text.lower().count(kw) for kw in keywords]

    numeric_feats = np.array([[length, sym_count] + kw_counts])
    numeric_feats = scaler.transform(numeric_feats)

    text_tfidf = tfidf.transform([text])

    return hstack([text_tfidf, numeric_feats])

# --- UI ---
st.title("AutoJudge")
st.markdown("### Predict Programming Problem Difficulty")

st.divider()

prob_title = st.text_input("Problem Title")
prob_desc = st.text_area("Problem Description", height=150)
input_desc = st.text_area("Input Description", height=100)
output_desc = st.text_area("Output Description", height=100)

# --- PREDICTION ---
if st.button("Predict Difficulty", type="primary"):
    if prob_desc.strip() == "":
        st.warning("Please enter a problem description.")
    else:
        combined_text = f"{prob_title} {prob_desc} {input_desc} {output_desc}"

        features = extract_features(combined_text)

        pred_class = clf.predict(features)[0]
        pred_score = reg.predict(features)[0]

        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", pred_class.upper())
        with col2:
            st.metric("Predicted Score", f"{pred_score:.2f}")
