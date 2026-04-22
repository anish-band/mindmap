import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LABEL_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def build_dashboard(results: dict, times: dict, region_scores: dict,
                    sim_rolling_acc: list):
    """Single comprehensive figure saved to results/dashboard.png."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_names = list(results.keys())
    n_models    = len(model_names)

    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, n_models, figure=fig, hspace=0.55, wspace=0.4)

    # ── Row 0: confusion matrices ────────────────────────────────────────────
    for col, name in enumerate(model_names):
        ax = fig.add_subplot(gs[0, col])
        cm = results[name]["cm"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
            ax=ax, cbar=False, annot_kws={"size": 9},
        )
        acc = results[name]["accuracy"]
        ax.set_title(f"{name}\nAcc={acc:.1%}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.tick_params(labelsize=7)

    # ── Row 1: model comparison bar chart (spans all cols) ──────────────────
    ax_cmp = fig.add_subplot(gs[1, :2])
    x   = np.arange(n_models)
    w   = 0.35
    acc = [results[n]["accuracy"] * 100 for n in model_names]
    f1  = [results[n]["f1"]        * 100 for n in model_names]
    b1  = ax_cmp.bar(x - w/2, acc, w, label="Accuracy %",       color="#3498db", edgecolor="white")
    b2  = ax_cmp.bar(x + w/2, f1,  w, label="F1 (weighted) %",  color="#2ecc71", edgecolor="white")
    for bar in list(b1) + list(b2):
        ax_cmp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                    f"{bar.get_height():.1f}", ha="center", fontsize=8)
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(model_names, fontsize=10)
    ax_cmp.set_ylim(0, 115)
    ax_cmp.set_ylabel("Score (%)")
    ax_cmp.set_title("Model Comparison", fontsize=11, fontweight="bold")
    ax_cmp.legend(fontsize=9)
    ax_cmp.spines[["top", "right"]].set_visible(False)

    # ── Row 1: training times ────────────────────────────────────────────────
    ax_time = fig.add_subplot(gs[1, 2:])
    t_vals  = [times.get(n, 0) for n in model_names]
    colors  = ["#e74c3c", "#e67e22", "#9b59b6", "#1abc9c"][:n_models]
    bars    = ax_time.barh(model_names, t_vals, color=colors, edgecolor="white")
    for bar, t in zip(bars, t_vals):
        ax_time.text(bar.get_width() + max(t_vals)*0.01, bar.get_y() + bar.get_height()/2,
                     f"{t:.1f}s", va="center", fontsize=9)
    ax_time.set_xlabel("Training Time (s)")
    ax_time.set_title("Training Time", fontsize=11, fontweight="bold")
    ax_time.spines[["top", "right"]].set_visible(False)

    # ── Row 2: brain region importance ──────────────────────────────────────
    ax_region = fig.add_subplot(gs[2, :2])
    if region_scores:
        regions = list(region_scores.keys())
        scores  = list(region_scores.values())
        region_colors = [
            "#e74c3c", "#e67e22", "#3498db", "#9b59b6", "#2ecc71", "#f39c12"
        ][:len(regions)]
        order = np.argsort(scores)[::-1]
        ax_region.bar(
            [regions[i].replace("\n", " ") for i in order],
            [scores[i] for i in order],
            color=[region_colors[i % len(region_colors)] for i in order],
            edgecolor="white",
        )
        ax_region.set_title("Feature Importance by Brain Region\n(mean |SHAP| per feature)",
                             fontsize=11, fontweight="bold")
        ax_region.set_ylabel("Mean |SHAP|")
        ax_region.tick_params(axis="x", labelsize=8, rotation=15)
        ax_region.spines[["top", "right"]].set_visible(False)
    else:
        ax_region.text(0.5, 0.5, "SHAP not available", ha="center", va="center")

    # ── Row 2: real-time sim rolling accuracy ────────────────────────────────
    ax_sim = fig.add_subplot(gs[2, 2:])
    if sim_rolling_acc:
        ax_sim.plot(sim_rolling_acc, color="#3498db", linewidth=1.5)
        ax_sim.axhline(sim_rolling_acc[-1], color="#e74c3c", linestyle="--",
                       linewidth=1, label=f"Final: {sim_rolling_acc[-1]:.1%}")
        ax_sim.set_ylim(0, 1.05)
        ax_sim.set_title("Real-Time Sim — Rolling Accuracy", fontsize=11, fontweight="bold")
        ax_sim.set_xlabel("Sample")
        ax_sim.set_ylabel("Accuracy")
        ax_sim.legend(fontsize=9)
        ax_sim.spines[["top", "right"]].set_visible(False)

    fig.suptitle("MindMap — EEG Emotion Classification Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)

    out = os.path.join(RESULTS_DIR, "dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Dashboard saved: {out}")
