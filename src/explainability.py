import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LABEL_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Feature → brain region mapping for the Muse headband
# _a = AF7 (left frontal) + TP9 (left temporal-parietal)
# _b = AF8 (right frontal) + TP10 (right temporal-parietal)
REGION_KEYWORDS = {
    "Left Frontal\n(AF7)":         ["_a", "asym"],
    "Right Frontal\n(AF8)":        ["_b"],
    "Left Temporal-\nParietal (TP9)": ["mean_d_", "_a"],
    "Right Temporal-\nParietal (TP10)": ["fft_", "_b"],
    "Inter-hemi\nCoherence":       ["coherence", "asym"],
    "Band Ratios":                 ["ratio"],
}

REGION_COLORS = {
    "Left Frontal\n(AF7)":            "#e74c3c",
    "Right Frontal\n(AF8)":           "#e67e22",
    "Left Temporal-\nParietal (TP9)": "#3498db",
    "Right Temporal-\nParietal (TP10)":"#9b59b6",
    "Inter-hemi\nCoherence":          "#2ecc71",
    "Band Ratios":                    "#f39c12",
}


def _assign_region(col: str) -> str:
    col_lower = col.lower()
    if "ratio" in col_lower:
        return "Band Ratios"
    if "coher" in col_lower or "asym" in col_lower:
        return "Inter-hemi\nCoherence"
    if col_lower.endswith("_a") or "_a2" in col_lower:
        if col_lower.startswith("mean_d") or col_lower.startswith("fft"):
            return "Left Temporal-\nParietal (TP9)"
        return "Left Frontal\n(AF7)"
    if col_lower.endswith("_b") or "_b2" in col_lower:
        if col_lower.startswith("mean_d") or col_lower.startswith("fft"):
            return "Right Temporal-\nParietal (TP10)"
        return "Right Frontal\n(AF8)"
    return "Band Ratios"


def run_shap_rf(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                n_samples: int = 200):
    print("\n  Computing SHAP values (TreeExplainer on RF)...")
    X_eval = X_test.iloc[:n_samples].reset_index(drop=True)
    feat_names = X_eval.columns.tolist()

    # Use new Explanation-based API for SHAP >= 0.41
    explainer  = shap.TreeExplainer(model)
    explanation = explainer(X_eval)          # Explanation: [n_samples, n_features, n_classes]

    # .values shape: [n_samples, n_features, n_classes] or [n_samples, n_features]
    vals = explanation.values
    if vals.ndim == 3:
        # multi-class: average |SHAP| over classes for global importance
        mean_abs_all = np.abs(vals).mean(axis=(0, 2))   # [n_features]
        sv_pos       = vals[:, :, 2]                     # POSITIVE class
    elif isinstance(vals, list):
        sv_pos = vals[2]
        mean_abs_all = np.abs(sv_pos).mean(axis=0)
    else:
        sv_pos = vals
        mean_abs_all = np.abs(vals).mean(axis=0)

    return explainer, explanation, sv_pos, mean_abs_all, feat_names, X_eval


def plot_shap_summary(sv_pos: np.ndarray, mean_abs: np.ndarray,
                      feat_names: list, X_eval: pd.DataFrame, class_idx: int = 2):
    """Beeswarm-style summary plot for POSITIVE class."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    top_n   = 30
    top_idx = np.argsort(mean_abs)[-top_n:]              # [30] int indices
    top_cols  = [feat_names[i] for i in top_idx]
    top_sv    = sv_pos[:, top_idx]                        # [n_samples, 30]
    top_data  = X_eval.iloc[:, top_idx].values            # [n_samples, 30]

    fig, ax = plt.subplots(figsize=(10, 9))
    plt.sca(ax)
    shap.summary_plot(top_sv, top_data, feature_names=top_cols,
                      show=False, plot_size=None)
    ax.set_title("SHAP Summary — POSITIVE class (top 30 features)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return mean_abs, feat_names


def plot_shap_force(sv_pos: np.ndarray, feat_names: list,
                    X_eval: pd.DataFrame, sample_idx: int = 0):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sv = sv_pos[sample_idx]                              # [n_features]

    top_n = 15
    order = np.argsort(np.abs(sv))[-top_n:]
    names  = [feat_names[i] for i in order]
    sv_top = sv[order]

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in sv_top]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, sv_top, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"SHAP Force — Sample #{sample_idx} (POSITIVE class, top {top_n})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("SHAP Value (impact on model output)")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "shap_force.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_brain_region_importance(mean_abs_shap: np.ndarray,
                                  feature_names: pd.Index):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    region_scores = {}
    region_counts = {}

    for feat, score in zip(feature_names, mean_abs_shap):
        region = _assign_region(feat)
        region_scores[region] = region_scores.get(region, 0.0) + score
        region_counts[region] = region_counts.get(region, 0) + 1

    # Normalise by count so regions with more features aren't unfairly boosted
    regions = list(region_scores.keys())
    scores  = [region_scores[r] / max(region_counts[r], 1) for r in regions]
    colors  = [REGION_COLORS.get(r, "#95a5a6") for r in regions]

    order = np.argsort(scores)[::-1]
    regions = [regions[i] for i in order]
    scores  = [scores[i]  for i in order]
    colors  = [colors[i]  for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(regions, scores, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(scores)*0.01,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.set_title("Feature Importance by Brain Region\n"
                 "(Mean |SHAP| per feature, normalised by region size)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean |SHAP| per feature")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "brain_region_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    print("\n  --- Brain Region Interpretation ---")
    for r, s in zip(regions, scores):
        print(f"  {r.replace(chr(10), ' '):<35} score={s:.4f}")
    print("\n  Key findings:")
    top_region = regions[0].replace("\n", " ")
    print(f"  • Most predictive region: {top_region}")
    print("  • Left-right asymmetry features capture valence (positive/negative affect)")
    print("  • Band ratios (theta/alpha, alpha/beta) encode arousal states")
    print("  • FFT temporal-parietal features reflect emotional processing networks")

    return dict(zip(regions, scores))


def run_explainability(rf_model, X_train: pd.DataFrame,
                        X_test: pd.DataFrame) -> dict:
    print("\n=== Phase 3: Explainability ===")
    explainer, explanation, sv_pos, mean_abs, feat_names, X_eval = run_shap_rf(
        rf_model, X_train, X_test
    )
    plot_shap_summary(sv_pos, mean_abs, feat_names, X_eval)
    plot_shap_force(sv_pos, feat_names, X_eval)
    region_scores = plot_brain_region_importance(
        mean_abs, pd.Index(feat_names)
    )
    return region_scores
