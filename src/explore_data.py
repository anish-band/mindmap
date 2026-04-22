import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os

path = kagglehub.dataset_download("samnikolas/eeg-dataset")
print("Dataset path:", path)

csv_path = os.path.join(path, "features_raw.csv")
df = pd.read_csv(csv_path)

shutil.copy(csv_path, "/Users/anishbandapelli/Documents/Code/machine_learning/MindMap/data/features_raw.csv")
print("Copied to data/features_raw.csv\n")

print("=== Shape ===")
print(df.shape)

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== Dtypes ===")
print(df.dtypes)

print("\n=== Label unique values ===")
label_col = [c for c in df.columns if "label" in c.lower()]
print("Label column(s):", label_col)
for col in label_col:
    print(df[col].unique())

print("\n=== Describe ===")
print(df.describe())

# Class distribution plot
if label_col:
    col = label_col[0]
    counts = df[col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.values.max() * 0.01, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    out = "/Users/anishbandapelli/Documents/Code/machine_learning/MindMap/results/class_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved class distribution plot to {out}")
    plt.show()
