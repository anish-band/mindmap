import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number).columns
    n_missing = df[numeric].isna().sum().sum()
    if n_missing:
        df[numeric] = df[numeric].fillna(df[numeric].median())
        print(f"  Filled {n_missing} missing values with column medians")
    else:
        print("  No missing values found")
    return df


def remove_outliers(df: pd.DataFrame, n_std: float = 5.0,
                     max_extreme_frac: float = 0.05) -> pd.DataFrame:
    """Remove rows where >max_extreme_frac of features exceed n_std σ.

    With thousands of features, requiring ALL to be in range removes too many
    rows due to the curse of dimensionality. Instead we allow a small fraction
    of features to be extreme before flagging a whole row.
    """
    skip = {"label", "subject_id"}
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c not in skip]
    means = df[numeric].mean()
    stds  = df[numeric].std().replace(0, 1)
    extreme = ((df[numeric] - means).abs() > n_std * stds)
    frac_extreme = extreme.sum(axis=1) / len(numeric)
    mask = frac_extreme <= max_extreme_frac
    n_removed = (~mask).sum()
    if n_removed:
        print(f"  Outlier removal (>{n_std}σ in >{max_extreme_frac:.0%} of features): "
              f"dropped {n_removed} rows ({n_removed / len(df) * 100:.1f}%)")
    else:
        print(f"  No outliers found at >{n_std}σ / >{max_extreme_frac:.0%} threshold")
    return df[mask].reset_index(drop=True)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_te = pd.DataFrame(scaler.transform(X_test),       columns=X_test.columns)
    print(f"  Scaled {X_tr.shape[1]} features with StandardScaler")
    return X_tr, X_te, scaler


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Preprocessing ===")
    df = handle_missing(df)
    df = remove_outliers(df)
    return df
