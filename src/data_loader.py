import os
import shutil
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

DATASET_SLUG = "birdy654/eeg-brainwave-dataset-feeling-emotions"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Bands: 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
# Electrode groups from Muse headband: _a = AF7+TP9, _b = AF8+TP10
GROUP_A = "AF7/TP9 (left)"
GROUP_B = "AF8/TP10 (right)"


def download_dataset() -> str:
    path = kagglehub.dataset_download(DATASET_SLUG)
    dest = os.path.join(DATA_DIR, "emotions.csv")
    src  = os.path.join(path, "emotions.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(dest):
        shutil.copy(src, dest)
    return dest


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Fix the leading '# ' on the first column
    df.columns = [c.lstrip("# ") for c in df.columns]
    return df


def inspect(df: pd.DataFrame) -> None:
    print(f"\nShape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes.value_counts()}")
    print(f"\nUnique labels: {df['label'].unique()}")
    print(f"\nClass distribution:\n{df['label'].value_counts()}")
    feat_cols = [c for c in df.columns if c not in ("label", "subject_id")]
    print(f"\nDescriptive stats (first 10 feature cols):\n"
          f"{df[feat_cols[:10]].describe().round(2)}")


def plot_class_distribution(df: pd.DataFrame) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    counts = df["label"].value_counts().reindex(["NEGATIVE", "NEUTRAL", "POSITIVE"])
    palette = {"NEGATIVE": "#e74c3c", "NEUTRAL": "#3498db", "POSITIVE": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[palette[l] for l in counts.index], edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 8, str(val),
                ha="center", fontsize=12, fontweight="bold")
    ax.set_title("MindMap — Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Emotional State")
    ax.set_ylabel("Sample Count")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def _assign_subjects(df: pd.DataFrame) -> pd.DataFrame:
    # Dataset has 2 subjects; split data in half (temporal ordering per class)
    df = df.copy()
    subject_ids = []
    for label in df["label"].unique():
        idx = df.index[df["label"] == label].tolist()
        half = len(idx) // 2
        for i, j in enumerate(idx):
            subject_ids.append((j, 0 if i < half else 1))
    subject_series = pd.Series(dict(subject_ids), name="subject_id")
    df["subject_id"] = subject_series
    return df


def random_split(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("label", "subject_id")]
    X = df[feature_cols]
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def subject_independent_split(df: pd.DataFrame):
    train = df[df["subject_id"] == 0]
    test  = df[df["subject_id"] == 1]
    cols  = [c for c in df.columns if c not in ("label", "subject_id")]
    return (
        train[cols].reset_index(drop=True),
        test[cols].reset_index(drop=True),
        train["label"].reset_index(drop=True),
        test["label"].reset_index(drop=True),
    )


def load_data() -> pd.DataFrame:
    csv_path = download_dataset()
    df = load_raw(csv_path)
    df = _assign_subjects(df)
    print("\n=== Phase 1: Data Inspection ===")
    inspect(df)
    plot_class_distribution(df)
    return df
