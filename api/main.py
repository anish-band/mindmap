import asyncio
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import engineer_features

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")

LABEL_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# ── Global state (populated at startup) ──────────────────────────────────────
state: dict = {}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MindMap EEG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    # Load CSV — get raw feature names and build simulation samples
    df = pd.read_csv(os.path.join(DATA_DIR, "emotions.csv"))
    df.columns = [c.lstrip("# ") for c in df.columns]

    raw_feature_cols = [c for c in df.columns if c != "label"]

    # Preprocess 200 held-out samples for simulation (last 200 rows)
    sim_df = df.tail(210).head(200).reset_index(drop=True)   # held-out slice
    sim_labels  = sim_df["label"].tolist()
    sim_X_raw   = sim_df[raw_feature_cols].copy()
    sim_X_eng   = engineer_features(sim_X_raw.copy())
    sim_X_eng   = sim_X_eng[scaler.feature_names_in_]
    sim_X_scaled = pd.DataFrame(
        scaler.transform(sim_X_eng),
        columns=scaler.feature_names_in_,
    )

    # SHAP explainer (TreeExplainer on RF — no background needed for trees)
    explainer = shap.TreeExplainer(rf_model)

    state.update({
        "rf_model":        rf_model,
        "scaler":          scaler,
        "raw_feature_cols": raw_feature_cols,
        "sim_labels":      sim_labels,
        "sim_X_scaled":    sim_X_scaled,
        "sim_X_raw":       sim_X_raw,
        "explainer":       explainer,
    })
    print("MindMap API ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _assign_region(col: str) -> str:
    col = col.lower()
    if "ratio" in col:
        return "band_ratios"
    if "coher" in col or "asym" in col:
        return "interhemispheric"
    if col.endswith("_a") or "_a2" in col:
        if col.startswith("mean_d") or col.startswith("fft"):
            return "temporal_parietal_left"
        return "frontal_left"
    if col.endswith("_b") or "_b2" in col:
        if col.startswith("mean_d") or col.startswith("fft"):
            return "temporal_parietal_right"
        return "frontal_right"
    return "band_ratios"


def _preprocess(features: list[float]) -> pd.DataFrame:
    """Raw 2548 floats → scaled 2564-feature DataFrame (1 row)."""
    raw_cols = state["raw_feature_cols"]
    if len(features) != len(raw_cols):
        raise ValueError(
            f"Expected {len(raw_cols)} features, got {len(features)}"
        )
    df_row = pd.DataFrame([features], columns=raw_cols)
    df_eng = engineer_features(df_row.copy())
    df_eng = df_eng[state["scaler"].feature_names_in_]
    df_scaled = pd.DataFrame(
        state["scaler"].transform(df_eng),
        columns=state["scaler"].feature_names_in_,
    )
    return df_scaled


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: list[float]

class ExplainRequest(BaseModel):
    features: list[float]


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":   "ok",
        "model":    "random_forest",
        "accuracy": 98.58,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    X = _preprocess(req.features)
    rf = state["rf_model"]

    pred  = rf.predict(X)[0]
    proba = rf.predict_proba(X)[0]
    conf  = {cls: round(float(p), 4) for cls, p in zip(rf.classes_, proba)}

    return {
        "prediction": pred,
        "confidence": conf,
    }


@app.get("/sample")
def sample():
    """Return one random held-out sample for the Explainability demo."""
    df = pd.read_csv(os.path.join(DATA_DIR, "emotions.csv"))
    df.columns = [c.lstrip("# ") for c in df.columns]
    row = df.sample(1, random_state=np.random.randint(0, 1000))
    feat_cols = [c for c in df.columns if c != "label"]
    return {
        "features": row[feat_cols].values[0].tolist(),
        "true_label": row["label"].values[0],
    }


@app.get("/compare")
def compare():
    return {
        "models": [
            {
                "name":     "SVM",
                "accuracy": 97.87,
                "f1":       97.87,
                "split":    "random_80_20",
            },
            {
                "name":     "Random Forest",
                "accuracy": 98.58,
                "f1":       98.58,
                "split":    "random_80_20",
            },
            {
                "name":     "1D CNN",
                "accuracy": 98.82,
                "f1":       98.82,
                "split":    "random_80_20",
            },
            {
                "name":     "LSTM",
                "accuracy": 76.49,
                "f1":       77.10,
                "split":    "subject_independent",
            },
        ]
    }


@app.post("/explain")
def explain(req: ExplainRequest):
    X = _preprocess(req.features)
    rf          = state["rf_model"]
    explainer   = state["explainer"]
    feat_names  = list(state["scaler"].feature_names_in_)

    explanation = explainer(X)                  # Explanation [1, n_features, n_classes]
    vals = explanation.values                   # [1, n_features, n_classes]

    # Mean |SHAP| across classes for global importance on this sample
    mean_abs = np.abs(vals[0]).mean(axis=1)     # [n_features]

    top_idx    = np.argsort(mean_abs)[-20:][::-1]
    top_feats  = [
        {
            "feature":      feat_names[i],
            "importance":   round(float(mean_abs[i]), 6),
            "shap_positive": round(float(vals[0, i, 2]), 6),  # POSITIVE class
            "brain_region": _assign_region(feat_names[i]),
        }
        for i in top_idx
    ]

    pred  = rf.predict(X)[0]
    proba = rf.predict_proba(X)[0]
    conf  = {cls: round(float(p), 4) for cls, p in zip(rf.classes_, proba)}

    return {
        "prediction":        pred,
        "confidence":        conf,
        "top_features":      top_feats,
    }


# ── WebSocket simulation ──────────────────────────────────────────────────────
@app.websocket("/simulate")
async def simulate(ws: WebSocket):
    await ws.accept()

    rf          = state["rf_model"]
    sim_labels  = state["sim_labels"]
    sim_X       = state["sim_X_scaled"]
    sim_X_raw   = state["sim_X_raw"]

    n_correct = 0

    try:
        for i in range(len(sim_labels)):
            x_row   = sim_X.iloc[[i]]
            pred    = rf.predict(x_row)[0]
            proba   = rf.predict_proba(x_row)[0]
            conf    = max(float(p) for p in proba)
            true_lbl = sim_labels[i]

            n_correct += int(pred == true_lbl)
            rolling_acc = round(n_correct / (i + 1), 4)

            eeg_snapshot = sim_X_raw.iloc[i, :50].tolist()

            msg = {
                "sample_index":    i,
                "true_label":      true_lbl,
                "predicted_label": pred,
                "confidence":      round(conf, 4),
                "rolling_accuracy": rolling_acc,
                "eeg_snapshot":    [round(v, 4) for v in eeg_snapshot],
            }
            await ws.send_text(json.dumps(msg))
            await asyncio.sleep(0.05)

        await ws.send_text(json.dumps({"done": True, "final_accuracy": rolling_acc}))

    except WebSocketDisconnect:
        pass
