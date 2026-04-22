import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LABEL_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL_COLORS = {
    "NEGATIVE": "#e74c3c",
    "NEUTRAL":  "#3498db",
    "POSITIVE": "#2ecc71",
}
N_SAMPLES   = 200
DELAY       = 0.05   # seconds between frames


def _predict_sample(model, x_row, model_type="sklearn"):
    if model_type == "sklearn":
        pred  = model.predict(x_row.reshape(1, -1))[0]
        proba = model.predict_proba(x_row.reshape(1, -1))[0]
    else:
        import torch
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            t = torch.tensor(x_row, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(t)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = LABEL_ORDER[proba.argmax()]
    return pred, proba


def run_realtime_sim(model, X_test, y_test, model_type="sklearn",
                     n_samples=N_SAMPLES, delay=DELAY, save_snapshot=True):
    print(f"\n=== Phase 4: Real-Time Simulation ({n_samples} samples) ===")

    # Take held-out samples from test set
    X_arr = X_test.values[:n_samples] if hasattr(X_test, "values") else X_test[:n_samples]
    y_arr = list(y_test)[:n_samples]

    predictions   = []
    confidences   = []
    rolling_accs  = []
    n_correct     = 0

    # ── Static run: collect all predictions first for the saved figure ──
    for i in range(n_samples):
        pred, proba = _predict_sample(model, X_arr[i], model_type)
        predictions.append(pred)
        confidences.append(proba.max())
        n_correct += int(pred == y_arr[i])
        rolling_accs.append(n_correct / (i + 1))

    # ── Build static snapshot figure ────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Sample EEG signal (first 32 features as proxy waveform)
    ax_eeg = fig.add_subplot(gs[0, :2])
    n_ch = min(32, X_arr.shape[1])
    eeg_snippet = X_arr[:50, :n_ch]
    for ch in range(min(4, n_ch)):
        ax_eeg.plot(eeg_snippet[:, ch] + ch * 15, linewidth=0.8, alpha=0.85)
    ax_eeg.set_title("EEG Signal — 4 channels (first 50 samples)", fontsize=11)
    ax_eeg.set_xlabel("Sample index")
    ax_eeg.set_ylabel("Amplitude (offset per channel)")
    ax_eeg.spines[["top", "right"]].set_visible(False)

    # 2. Last predicted label
    ax_label = fig.add_subplot(gs[0, 2])
    last_pred  = predictions[-1]
    last_conf  = confidences[-1]
    color = LABEL_COLORS[last_pred]
    ax_label.set_facecolor(color + "22")
    ax_label.text(0.5, 0.55, last_pred, ha="center", va="center",
                  fontsize=20, fontweight="bold", color=color,
                  transform=ax_label.transAxes)
    ax_label.text(0.5, 0.3, f"Confidence: {last_conf:.1%}",
                  ha="center", va="center", fontsize=12,
                  transform=ax_label.transAxes)
    ax_label.set_title("Predicted Emotion (last sample)", fontsize=11)
    ax_label.set_xticks([]); ax_label.set_yticks([])

    # 3. Rolling accuracy
    ax_acc = fig.add_subplot(gs[1, :2])
    ax_acc.plot(rolling_accs, color="#3498db", linewidth=1.5, label="Rolling accuracy")
    ax_acc.axhline(rolling_accs[-1], color="#e74c3c", linestyle="--", linewidth=1,
                   label=f"Final: {rolling_accs[-1]:.1%}")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title("Rolling Accuracy Over Time", fontsize=11)
    ax_acc.set_xlabel("Sample")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=9)
    ax_acc.spines[["top", "right"]].set_visible(False)

    # 4. Confidence per prediction coloured by label
    ax_conf = fig.add_subplot(gs[1, 2])
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        ax_conf.scatter(i, conf, color=LABEL_COLORS[pred], s=8, alpha=0.6)
    ax_conf.set_title("Confidence per Prediction", fontsize=11)
    ax_conf.set_xlabel("Sample")
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_ylim(0, 1.05)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(color=v, label=k) for k, v in LABEL_COLORS.items()]
    ax_conf.legend(handles=legend_elements, fontsize=8, loc="lower left")
    ax_conf.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"MindMap — Real-Time EEG Emotion Simulation\n"
                 f"Final accuracy: {rolling_accs[-1]:.1%} over {n_samples} samples",
                 fontsize=13, fontweight="bold")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if save_snapshot:
        out = os.path.join(RESULTS_DIR, "realtime_sim_snapshot.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")

    print(f"  Simulation complete. Final rolling accuracy: {rolling_accs[-1]:.1%}")
    return predictions, confidences, rolling_accs
