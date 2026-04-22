import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LABEL_ORDER  = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
PALETTE      = {"NEGATIVE": "#e74c3c", "NEUTRAL": "#3498db", "POSITIVE": "#2ecc71"}


def eval_sklearn(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cm  = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    print(f"\n{classification_report(y_test, y_pred, target_names=LABEL_ORDER)}")
    return {"accuracy": acc, "f1": f1, "cm": cm, "y_pred": y_pred}


def eval_torch(model, test_dl, y_true_enc, le):
    model.eval()
    preds, proba = [], []
    with torch.no_grad():
        for xb, _ in test_dl:
            logits = model(xb.to(next(model.parameters()).device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            proba.append(p)
            preds.append(p.argmax(axis=1))
    y_pred_enc = np.concatenate(preds)
    y_pred = le.inverse_transform(y_pred_enc)
    y_true = le.inverse_transform(y_true_enc)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    cm  = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    print(f"\n{classification_report(y_true, y_pred, target_names=LABEL_ORDER)}")
    return {"accuracy": acc, "f1": f1, "cm": cm, "y_pred": y_pred,
            "proba": np.concatenate(proba, axis=0)}


def plot_confusion_matrix(cm, title: str, ax=None, save_path: str = None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
        ax=ax, cbar=not standalone,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if standalone:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


def plot_model_comparison(results: dict, save_path: str):
    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]
    f1s   = [results[n]["f1"] * 100        for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy %", color="#3498db", edgecolor="white")
    b2 = ax.bar(x + w/2, f1s,  w, label="F1 (weighted) %", color="#2ecc71", edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Accuracy & F1", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_summary_table(results: dict, times: dict):
    print("\n" + "=" * 70)
    print(f"{'MODEL':<18} {'ACCURACY':>10} {'F1 (weighted)':>14} {'TRAIN TIME':>12}")
    print("=" * 70)
    best_acc  = max(results[n]["accuracy"] for n in results)
    for name in results:
        acc  = results[name]["accuracy"]
        f1   = results[name]["f1"]
        t    = times.get(name, 0)
        flag = " ← best" if acc == best_acc else ""
        print(f"{name:<18} {acc*100:>9.2f}%  {f1*100:>13.2f}%  {t:>10.1f}s{flag}")
    print("=" * 70)


def best_model_name(results: dict) -> str:
    return max(results, key=lambda n: results[n]["accuracy"])
