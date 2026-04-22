"""
MindMap — EEG-based Emotion Classification
==========================================
Runs all 5 phases end-to-end and saves results to results/.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

from data_loader     import load_data, random_split, subject_independent_split
from preprocessor    import preprocess, scale_features
from features        import engineer_features, get_feature_cols, get_fft_matrix
from models          import train_svm, train_rf, train_cnn, train_lstm
from evaluate        import (eval_sklearn, eval_torch, plot_confusion_matrix,
                              plot_model_comparison, print_summary_table, best_model_name)
from explainability  import run_explainability
from realtime_sim    import run_realtime_sim
from dashboard       import build_dashboard

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Phase 1: Data Pipeline ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1 — Data Pipeline")
    print("="*60)

    df = load_data()
    df = preprocess(df)
    df = engineer_features(df)

    # Random 80/20 split
    X_train_r, X_test_r, y_train_r, y_test_r = random_split(df)

    # Subject-independent split
    X_train_s, X_test_s, y_train_s, y_test_s = subject_independent_split(df)

    # Scale features (use random split for main models)
    X_train_sc, X_test_sc, scaler = scale_features(X_train_r, X_test_r)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # FFT matrix for LSTM (subject-independent split, scaled)
    X_train_s_sc, X_test_s_sc, _ = scale_features(X_train_s, X_test_s)
    fft_train = get_fft_matrix(X_train_s_sc)
    fft_test  = get_fft_matrix(X_test_s_sc)

    print(f"\n  Random split    → train: {len(X_train_r)}, test: {len(X_test_r)}")
    print(f"  Subject split   → train: {len(X_train_s)}, test: {len(X_test_s)}")
    print(f"  Total features  → {X_train_sc.shape[1]}")
    print(f"  FFT matrix      → {fft_train.shape} (seq_len × 2 channels)")

    # ── Phase 2: Models ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 2 — Model Training & Evaluation")
    print("="*60)

    results, times = {}, {}

    # SVM
    print("\n--- SVM ---")
    svm_clf, svm_time = train_svm(X_train_sc, y_train_r)
    joblib.dump(svm_clf, os.path.join(MODELS_DIR, "svm.pkl"))
    svm_res = eval_sklearn(svm_clf, X_test_sc, y_test_r)
    results["SVM"], times["SVM"] = svm_res, svm_time

    # Random Forest
    print("\n--- Random Forest ---")
    rf_clf, rf_time = train_rf(X_train_sc, y_train_r)
    joblib.dump(rf_clf, os.path.join(MODELS_DIR, "rf.pkl"))
    rf_res = eval_sklearn(rf_clf, X_test_sc, y_test_r)
    results["Random Forest"], times["Random Forest"] = rf_res, rf_time

    # 1D CNN (random split)
    print("\n--- 1D CNN ---")
    cnn_model, cnn_time, cnn_test_dl, cnn_y_te, cnn_le = train_cnn(
        X_train_sc, y_train_r, X_test_sc, y_test_r, n_epochs=30
    )
    import torch
    torch.save(cnn_model.state_dict(), os.path.join(MODELS_DIR, "cnn.pt"))
    cnn_res = eval_torch(cnn_model, cnn_test_dl, cnn_y_te, cnn_le)
    results["CNN"], times["CNN"] = cnn_res, cnn_time

    # LSTM (subject-independent split, FFT input)
    print("\n--- LSTM ---")
    lstm_model, lstm_time, lstm_test_dl, lstm_y_te, lstm_le = train_lstm(
        fft_train, y_train_s, fft_test, y_test_s, n_epochs=30
    )
    torch.save(lstm_model.state_dict(), os.path.join(MODELS_DIR, "lstm.pt"))
    lstm_res = eval_torch(lstm_model, lstm_test_dl, lstm_y_te, lstm_le)
    results["LSTM"], times["LSTM"] = lstm_res, lstm_time

    # Save confusion matrices individually
    for name, res in results.items():
        plot_confusion_matrix(
            res["cm"], title=name,
            save_path=os.path.join(RESULTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
        )

    plot_model_comparison(results, os.path.join(RESULTS_DIR, "model_comparison.png"))
    print_summary_table(results, times)

    # ── Phase 3: Explainability (RF — best sklearn model for SHAP speed) ────
    print("\n" + "="*60)
    print("  PHASE 3 — Explainability")
    print("="*60)
    region_scores = run_explainability(rf_clf, X_train_sc, X_test_sc)

    # ── Phase 4: Real-time Simulation ───────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 4 — Real-time Simulation")
    print("="*60)
    best = best_model_name(results)
    print(f"  Best model: {best}")

    if best == "SVM":
        sim_model, sim_X, sim_y, sim_type = svm_clf, X_test_sc, y_test_r, "sklearn"
    elif best == "Random Forest":
        sim_model, sim_X, sim_y, sim_type = rf_clf, X_test_sc, y_test_r, "sklearn"
    elif best == "CNN":
        sim_model, sim_X, sim_y, sim_type = cnn_model, X_test_sc, y_test_r, "torch"
    else:
        sim_model, sim_X, sim_y, sim_type = lstm_model, X_test_s_sc, y_test_s, "torch"

    _, _, rolling_acc = run_realtime_sim(
        sim_model, sim_X, sim_y, model_type=sim_type, n_samples=200
    )

    # ── Phase 5: Dashboard ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 5 — Results Dashboard")
    print("="*60)
    build_dashboard(results, times, region_scores, rolling_acc)

    print("\n" + "="*60)
    print("  MindMap complete. All outputs saved to results/")
    print("="*60)


if __name__ == "__main__":
    main()
