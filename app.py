"""Streamlit dashboard for label-noise-type-fingerprint.

Three tabs:
1. Diagnose -- pick a planted noise type, rate and seed; the dataset
   is generated and diagnosed live, with its fingerprint chart.
2. Signatures -- the disagreement-vs-margin profile of the chosen
   dataset and the bank distances behind the verdict.
3. Evaluation -- the shipped aggregate results from
   results/evaluation.json: per-type accuracy vs the noise-matrix
   baseline, rate-estimation error, diagnosis confusion.

Run:  streamlit run app.py
"""

import io
import json
import os

import pandas as pd
import streamlit as st

from noise_fingerprint.diagnose import diagnose
from noise_fingerprint.fingerprint import margin_profile
from noise_fingerprint.generate import build_dataset
from noise_fingerprint.models import FINGERPRINT_KEYS, NOISE_TYPES

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_results():
    path = os.path.join(HERE, "results", "evaluation.json")
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def _diagnose_cached(noise_type, rate, seed):
    dataset = build_dataset(seed, noise_type, rate)
    verdict = diagnose(dataset.features, dataset.labels)
    profile = margin_profile(dataset.features, dataset.labels)
    return {
        "n": dataset.n,
        "actual_rate": dataset.actual_rate,
        "predicted_type": verdict.predicted_type,
        "confidence": verdict.confidence,
        "estimated_rate": verdict.estimated_rate,
        "fingerprint": verdict.fingerprint,
        "distances": verdict.distances,
        "remedy": verdict.remedy,
        "profile": profile,
    }


def tab_diagnose(result, noise_type, rate):
    st.subheader("Diagnosis")
    cols = st.columns(4)
    cols[0].metric("Planted type", noise_type)
    cols[1].metric("Diagnosed type", result["predicted_type"])
    cols[2].metric("Confidence", "%.2f" % result["confidence"])
    cols[3].metric("Estimated rate", "%.3f" % result["estimated_rate"])
    st.write("Actual flipped share: %.3f across %d points."
             % (result["actual_rate"], result["n"]))
    if result["predicted_type"] == noise_type:
        st.success("Diagnosis matches the planted noise process.")
    else:
        st.warning("Diagnosis differs from the planted process -- "
                   "low rates and instance-dependent noise are the "
                   "hard cases; see the evaluation tab.")
    st.write("**Remedy:** %s" % result["remedy"])
    st.subheader("Fingerprint")
    frame = pd.DataFrame(
        {"component": list(FINGERPRINT_KEYS),
         "value": [result["fingerprint"][k] for k in FINGERPRINT_KEYS]}
    )
    st.bar_chart(frame.set_index("component"))
    st.dataframe(frame)


def tab_signatures(result):
    st.subheader("Disagreement vs margin profile")
    st.write("Share of points whose label their neighbours outvote, "
             "by neighbourhood-margin decile (lowest margin first). "
             "Boundary noise concentrates the disagreement in the "
             "first deciles; uniform noise spreads it across all "
             "deciles.")
    frame = pd.DataFrame(
        {"decile": list(range(1, 11)),
         "mean_margin": [p[0] for p in result["profile"]],
         "suspect_rate": [p[1] for p in result["profile"]],
         "points": [p[2] for p in result["profile"]]}
    )
    st.line_chart(frame.set_index("decile")["suspect_rate"])
    st.dataframe(frame)
    st.subheader("Bank distances (lower = closer match)")
    frame = pd.DataFrame(
        {"noise_type": list(NOISE_TYPES),
         "distance": [result["distances"][t] for t in NOISE_TYPES]}
    )
    st.bar_chart(frame.set_index("noise_type"))


def tab_evaluation():
    st.subheader("Shipped evaluation (planted ground truth)")
    results = _load_results()
    if results is None:
        st.write("No results yet -- run "
                 "`python -m noise_fingerprint evaluate`.")
        return
    agg = results["aggregate"]
    cols = st.columns(3)
    cols[0].metric("Type accuracy", "%.3f" % agg["accuracy"])
    cols[1].metric("Matrix baseline", "%.3f" % agg["baseline_accuracy"])
    cols[2].metric("Rate MAE", "%.3f" % agg["rate_mae"])
    rows = []
    for noise_type in NOISE_TYPES:
        entry = agg["per_type"][noise_type]
        rows.append({
            "noise_type": noise_type,
            "n": entry["n"],
            "fingerprinter": entry["accuracy"],
            "baseline": entry["baseline_accuracy"],
            "rate_mae": entry["rate_mae"],
        })
    st.dataframe(pd.DataFrame(rows))
    st.subheader("Diagnosis confusion (true -> predicted)")
    rows = []
    for noise_type in NOISE_TYPES:
        row = {"true_type": noise_type}
        counts = agg["confusion"].get(noise_type, {})
        for pred in NOISE_TYPES:
            row[pred] = counts.get(pred, 0)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows))


def main():
    st.title("Label Noise Type Fingerprint")
    st.write("Diagnose *which kind* of label noise a dataset has -- "
             "uniform, class-dependent, instance-dependent or "
             "boundary-concentrated -- because each kind needs a "
             "different remedy.")
    noise_type = st.selectbox("Planted noise type", list(NOISE_TYPES),
                              index=3)
    rate = st.slider("Planted noise rate", min_value=0.05,
                     max_value=0.30, value=0.20, step=0.05)
    seed = st.number_input("Seed", min_value=1, max_value=999999,
                           value=7)
    result = _diagnose_cached(noise_type, float(rate), int(seed))
    tab1, tab2, tab3 = st.tabs(["Diagnose", "Signatures", "Evaluation"])
    with tab1:
        tab_diagnose(result, noise_type, float(rate))
    with tab2:
        tab_signatures(result)
    with tab3:
        tab_evaluation()


if __name__ == "__main__":
    main()
