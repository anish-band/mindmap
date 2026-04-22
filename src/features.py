import numpy as np
import pandas as pd

# Muse headband: _a = AF7/TP9 (left), _b = AF8/TP10 (right)
# Band indices: 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma
BANDS = {0: "delta", 1: "theta", 2: "alpha", 3: "beta", 4: "gamma"}

BRAIN_REGIONS = {
    "frontal":       [c for b in range(5) for c in [f"mean_{b}_a", f"mean_{b}_b"]],
    "temporal":      [c for b in range(5) for c in [f"mean_d_{b}_a", f"mean_d_{b}_b"]],
    "theta_alpha":   ["ratio_theta_alpha_a", "ratio_theta_alpha_b"],
    "alpha_beta":    ["ratio_alpha_beta_a",  "ratio_alpha_beta_b"],
    "asymmetry":     [f"asym_band{b}" for b in range(5)],
}


def _safe_cols(df, cols):
    return [c for c in cols if c in df.columns]


def add_band_power_ratios(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-8
    for grp in ["a", "b"]:
        theta = df.get(f"mean_1_{grp}", pd.Series(eps, index=df.index))
        alpha = df.get(f"mean_2_{grp}", pd.Series(eps, index=df.index))
        beta  = df.get(f"mean_3_{grp}", pd.Series(eps, index=df.index))

        df[f"ratio_theta_alpha_{grp}"] = theta.abs() / (alpha.abs() + eps)
        df[f"ratio_alpha_beta_{grp}"]  = alpha.abs() / (beta.abs()  + eps)
    return df


def add_hemisphere_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    for band_idx in range(5):
        col_a = f"mean_{band_idx}_a"
        col_b = f"mean_{band_idx}_b"
        if col_a in df.columns and col_b in df.columns:
            df[f"asym_band{band_idx}"] = df[col_a] - df[col_b]
    return df


def add_rolling_fft_stats(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    fft_a = [c for c in df.columns if c.startswith("fft_") and c.endswith("_a")]
    fft_b = [c for c in df.columns if c.startswith("fft_") and c.endswith("_b")]

    if fft_a:
        fft_vals_a = df[fft_a].values
        df["fft_mean_a"] = fft_vals_a.mean(axis=1)
        df["fft_std_a"]  = fft_vals_a.std(axis=1)
        df["fft_peak_a"] = np.abs(fft_vals_a).max(axis=1)

    if fft_b:
        fft_vals_b = df[fft_b].values
        df["fft_mean_b"] = fft_vals_b.mean(axis=1)
        df["fft_std_b"]  = fft_vals_b.std(axis=1)
        df["fft_peak_b"] = np.abs(fft_vals_b).max(axis=1)

    # Cross-group FFT correlation as a proxy for interhemispheric coherence
    if fft_a and fft_b:
        n = min(len(fft_a), len(fft_b))
        corr = np.array([
            np.corrcoef(df[fft_a[:n]].values[i], df[fft_b[:n]].values[i])[0, 1]
            for i in range(len(df))
        ])
        df["fft_coherence"] = np.nan_to_num(corr, nan=0.0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Feature Engineering ===")
    n_before = df.shape[1]
    df = add_band_power_ratios(df)
    df = add_hemisphere_asymmetry(df)
    df = add_rolling_fft_stats(df)
    n_after = df.shape[1]
    print(f"  Added {n_after - n_before} engineered features "
          f"({n_before} → {n_after} total)")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ("label", "subject_id")]


def get_fft_matrix(df: pd.DataFrame):
    """Return FFT spectrum features shaped as [n_samples, 750, 2] for LSTM."""
    import re
    _is_fft = re.compile(r"^fft_\d+_[ab]$")
    fft_a = sorted([c for c in df.columns if _is_fft.match(c) and c.endswith("_a")],
                   key=lambda x: int(x.split("_")[1]))
    fft_b = sorted([c for c in df.columns if _is_fft.match(c) and c.endswith("_b")],
                   key=lambda x: int(x.split("_")[1]))
    n = min(len(fft_a), len(fft_b))
    a_vals = df[fft_a[:n]].values  # [samples, n]
    b_vals = df[fft_b[:n]].values  # [samples, n]
    return np.stack([a_vals, b_vals], axis=2)  # [samples, n, 2]
