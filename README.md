# MindMap — EEG-Based Emotion Classification

Classify emotional states (Positive / Neutral / Negative) from EEG brainwave signals using multiple ML approaches, with SHAP explainability and real-time inference simulation.

## Results

| Model | Accuracy | F1 (weighted) | Split |
|---|---|---|---|
| SVM (RBF) | 97.87% | 97.87% | Random 80/20 |
| Random Forest | 98.58% | 98.58% | Random 80/20 |
| **1D CNN** | **98.82%** | **98.82%** | Random 80/20 |
| LSTM | 76.49% | 77.10% | Subject-independent |

The LSTM uses a harder subject-independent split (train on subject A, test on subject B) — the realistic generalization test.

## Dataset

[EEG Brainwave Dataset: Feeling Emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions) by Jordan J. Bird — 2132 samples, 2548 statistical EEG features from a Muse headband (AF7, AF8, TP9, TP10 channels), perfectly balanced across 3 classes.

Downloaded automatically at runtime via `kagglehub`.

## Project Structure

```
MindMap/
├── src/
│   ├── data_loader.py      # Download, inspect, subject split, class distribution
│   ├── preprocessor.py     # Missing values, outlier removal, StandardScaler
│   ├── features.py         # Band power ratios, hemisphere asymmetry, FFT stats
│   ├── models.py           # SVM, Random Forest, 1D CNN, LSTM (PyTorch)
│   ├── evaluate.py         # Accuracy, F1, confusion matrices, summary table
│   ├── explainability.py   # SHAP summary, force plot, brain region importance
│   ├── realtime_sim.py     # 200-sample streaming simulation
│   └── dashboard.py        # Comprehensive 6-panel results figure
├── main.py                 # Runs all 5 phases end-to-end
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

All outputs (plots, trained models) are saved to `results/` and `models/`.

## Phases

**Phase 1 — Data Pipeline**
Loads the dataset, inspects shape/dtypes/class distribution, removes outliers, scales with `StandardScaler`, and engineers additional features: theta/alpha and alpha/beta band power ratios, left-right hemisphere asymmetry indices, and FFT spectral statistics.

**Phase 2 — Models**
Trains and evaluates four models: SVM with RBF kernel, Random Forest (200 estimators), 1D CNN treating the full feature vector as a signal, and an LSTM on the FFT spectrum (750 frequency bins × 2 electrode groups as a sequence). Reports accuracy, F1, confusion matrix, and training time for each.

**Phase 3 — Explainability**
Runs SHAP (`TreeExplainer`) on the Random Forest. Generates a beeswarm summary plot (top 30 features), a force plot for an individual prediction, and a bar chart mapping feature importance to brain regions (frontal, temporal-parietal, band ratios, inter-hemispheric coherence).

**Phase 4 — Real-Time Simulation**
Streams 200 held-out test samples through the best model one by one, tracking predicted emotion, confidence score, and rolling accuracy over time. Saves a snapshot figure.

**Phase 5 — Dashboard**
Assembles a single comprehensive figure: all confusion matrices, model comparison bar chart, brain region importance, and the real-time simulation rolling accuracy curve.

## Key Findings

- **Band power ratios** (theta/alpha, alpha/beta) are the most discriminative features
- **Left-right hemisphere asymmetry** in frontal channels captures valence (positive vs. negative affect) — consistent with frontal asymmetry theory (Davidson, 1998)
- Subject-independent generalization drops to ~76% (LSTM), highlighting the challenge of cross-subject EEG transfer
