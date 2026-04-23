# MindMap — EEG-Based Emotion Classification

An end-to-end machine learning system that classifies emotional states (Positive / Neutral / Negative) from EEG brainwave signals. Built with multiple ML architectures, SHAP explainability, real-time inference simulation, and a full-stack web interface.

🌐 **Live Demo:** [mindmap-bsp9apen9-anishbandapelli-1733s-projects.vercel.app](https://mindmap-bsp9apen9-anishbandapelli-1733s-projects.vercel.app)

---

## What It Does

MindMap covers the full ML lifecycle — from raw EEG data to a deployed web application:

- **Real-Time Classification** — streams EEG signals through a trained model and predicts emotional state live with confidence scores
- **Multi-Model Benchmarking** — trains and compares SVM, Random Forest, 1D CNN, and LSTM architectures
- **Explainable AI** — uses SHAP values to reveal which brain regions and frequency features drive each prediction
- **Full-Stack Web App** — FastAPI backend + Next.js frontend with a live WebSocket demo

---

## Results

| Model | Accuracy | F1 (weighted) | Split |
|---|---|---|---|
| SVM (RBF) | 97.87% | 97.87% | Random 80/20 |
| Random Forest | 98.58% | 98.58% | Random 80/20 |
| 1D CNN | 98.82% | 98.82% | Random 80/20 |
| LSTM | 76.49% | 77.10% | Subject-independent |

> The LSTM uses a subject-independent split (train on subject A, test on subject B) — a harder, more realistic generalization test. The accuracy drop from ~98% to ~76% reveals how much models can overfit to individual brain signal patterns.

---

## Key Findings

- Band power ratios (theta/alpha, alpha/beta) are the most discriminative features
- Left-right hemisphere asymmetry in frontal channels captures emotional valence — consistent with Davidson's frontal asymmetry theory (1998)
- Subject-independent generalization drops to ~76%, highlighting the core challenge of cross-subject EEG transfer learning

---

## Dataset

**EEG Brainwave Dataset: Feeling Emotions** by Jordan J. Bird  
2,132 samples · 2,548 statistical EEG features · Muse headband (AF7, AF8, TP9, TP10) · Balanced across 3 classes  
Downloaded automatically at runtime via `kagglehub`.

---

## How It Was Built

### ML Pipeline (`python main.py` runs all 5 phases)

**Phase 1 — Data Pipeline**  
Loads dataset via kagglehub, inspects shape/dtypes/class distribution, removes outliers, scales with StandardScaler, and engineers 16 additional features: theta/alpha and alpha/beta band power ratios, left-right hemisphere asymmetry indices, and FFT spectral statistics.

**Phase 2 — Models**  
Trains four architectures: SVM with RBF kernel, Random Forest (200 estimators), 1D CNN treating the feature vector as a signal, and LSTM on the FFT spectrum (750 frequency bins × 2 electrode groups as a sequence). Reports accuracy, F1, confusion matrix, and training time for each.

**Phase 3 — Explainability**  
Runs SHAP TreeExplainer on the Random Forest. Generates a beeswarm summary plot, force plot for individual predictions, and a bar chart mapping feature importance to brain regions.

**Phase 4 — Real-Time Simulation**  
Streams 200 held-out test samples through the best model at 50ms intervals, tracking predicted emotion, confidence, and rolling accuracy live.

**Phase 5 — Dashboard**  
Assembles a single comprehensive figure: all confusion matrices, model comparison, brain region importance chart, and rolling accuracy curve.

### Backend (`api/main.py`)
FastAPI server wrapping the trained pipeline with REST endpoints and a WebSocket for real-time streaming.

| Endpoint | Type | Description |
|---|---|---|
| `GET /health` | REST | Status + model accuracy |
| `POST /predict` | REST | 2548 features → label + confidence |
| `GET /compare` | REST | All 4 models, accuracy, F1, split |
| `POST /explain` | REST | SHAP top 20 features + brain region mapping |
| `WS /simulate` | WebSocket | Streams 200 samples at 50ms with rolling accuracy |

### Frontend (`frontend/`)
Next.js + Tailwind dark-themed single page app with three interactive sections: Live Demo (WebSocket simulation), Model Comparison (interactive charts), and Explainability (SHAP visualization).

---

## Project Structure
```
MindMap/
├── api/
│   └── main.py             # FastAPI backend
├── src/
│   ├── data_loader.py      # Download, inspect, subject split
│   ├── preprocessor.py     # Outlier removal, StandardScaler
│   ├── features.py         # Band ratios, asymmetry, FFT stats
│   ├── models.py           # SVM, Random Forest, 1D CNN, LSTM
│   ├── evaluate.py         # Accuracy, F1, confusion matrices
│   ├── explainability.py   # SHAP summary, force plot, brain regions
│   ├── realtime_sim.py     # 200-sample streaming simulation
│   └── dashboard.py        # 6-panel results figure
├── frontend/               # Next.js web application
├── main.py                 # Runs all 5 phases end-to-end
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

All outputs are saved to `results/` and `models/`.

To run the API:
```bash
cd api && uvicorn main:app --reload
```

To run the frontend:
```bash
cd frontend && npm install && npm run dev
```

---

## Built By

Anish Bandapelli  
Dataset: [EEG Brainwave Dataset: Feeling Emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions)
