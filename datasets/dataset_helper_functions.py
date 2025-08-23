# --- Standard imports ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Kaggle download ---
import kagglehub

# --- ML / Splits ---
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

# --- Stats ---
from scipy.stats import median_abs_deviation, pearsonr, entropy, pointbiserialr

# --- Hugging Face Hub ---
from huggingface_hub import HfApi, upload_file


# =========================
# Download / Load
# =========================

def download_kaggle_dataset(kaggle_id: str) -> str:
    """Download a Kaggle dataset via kagglehub and return the local path."""
    return kagglehub.dataset_download(kaggle_id)


def load_first_csv_in(path: str) -> pd.DataFrame:
    """Find and load the first CSV file in a directory."""
    csv_file = None
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_file = os.path.join(path, file)
            break
    if csv_file is None:
        raise FileNotFoundError(f"No CSV file found in: {path}")
    return pd.read_csv(csv_file)

# =========================
# Features
# =========================

def get_feature_lists(df: pd.DataFrame, label_col: str):
    """
    Infer numerical and categorical feature lists from dtypes, excluding the label.
    """
    numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if label_col in numerical:
        numerical.remove(label_col)
    if label_col in categorical:
        categorical.remove(label_col)
    return numerical, categorical

# =========================
# EDA / Plots
# =========================

def plot_label_pie(df: pd.DataFrame, label_col: str, colors=None, title=None):
    """Simple pie chart for label distribution."""
    if colors is None:
        colors = ["lightsteelblue", "dimgrey"]
    counts = df[label_col].value_counts(dropna=False)
    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    plt.title(title)
    plt.axis("equal")
    plt.show()


# =========================
# Balancing + Splitting
# =========================

def undersample_to_balance(df: pd.DataFrame, label_col: str, seed: int = 42) -> pd.DataFrame:
    """
    Undersample the majority class to match the minority class size.
    Returns a shuffled balanced DataFrame.
    """
    counts = df[label_col].value_counts()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    n_min = counts.min()

    df_min = df[df[label_col] == minority_class]
    df_maj = df[df[label_col] == majority_class].sample(n=n_min, random_state=seed)

    df_balanced = pd.concat([df_min, df_maj]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_balanced

def undersample_to_balance_multiclass(df: pd.DataFrame, label_col: str, seed: int = 42) -> pd.DataFrame:
    """
    Undersample each class to match the size of the smallest class.
    Works for binary and multi-class datasets.
    """
    np.random.seed(seed)
    counts = df[label_col].value_counts()
    n_min = counts.min()
    dfs = []
    for cls in counts.index:
        dfs.append(df[df[label_col] == cls].sample(n=n_min, random_state=seed))
    df_balanced = pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_balanced

def stratified_splits(
    df_balanced: pd.DataFrame,
    label_col: str,
    seed: int = 42,
    train_ratio: float = 0.5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.4
):
    """
    Make stratified splits with ratios that sum to 1.0.
    The implementation follows your 50/10/40 pattern.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    # First split: train vs temp
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df_balanced,
        test_size=temp_ratio,
        stratify=df_balanced[label_col],
        random_state=seed
    )
    # Second split: val vs test
    test_share_of_temp = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_share_of_temp,
        stratify=temp_df[label_col],
        random_state=seed
    )
    return train_df, val_df, test_df


# =========================
# Upload CSVs to HF
# =========================

def upload_csv_splits_to_hub(train_df, val_df, test_df, repo_id: str):
    """
    Save splits to CSV locally and upload to an existing HF dataset repo.
    Assumes you're already logged in (huggingface_hub.login()).
    """
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    api = HfApi()
    for fname in ["train.csv", "val.csv", "test.csv"]:
        upload_file(
            path_or_fileobj=fname,
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset"
        )

# =========================
# Correlations (value-wise)
# =========================

def valuewise_correlation_dict_on_test(
    test_df: pd.DataFrame,
    label_col: str,
    categorical_features: list,
    numerical_features: list,
    pos_label: str = ">50K"
):
    """
    Reproduces your value-wise point-biserial correlation per feature/value (on test only).
    Returns: dict {feature: {value: corr}}
    """
    df = test_df.copy()
    df["__y__"] = (df[label_col].astype(str) == pos_label).astype(int)

    corr_dict = {}

    # Categorical: per value corr
    for col in categorical_features:
        if col not in df.columns:
            continue
        vals = df[col].dropna().unique()
        v_corrs = {}
        for v in vals:
            bin_vec = (df[col] == v).astype(int)
            corr, _ = pointbiserialr(bin_vec, df["__y__"])
            v_corrs[str(v)] = float(0.0 if pd.isna(corr) else corr)
        corr_dict[col] = dict(sorted(v_corrs.items(), key=lambda x: x[1], reverse=True))

    # Numerical: per value corr (as in your original approach)
    for col in numerical_features:
        if col not in df.columns:
            continue
        vals = sorted(df[col].dropna().unique())
        v_corrs = {}
        for v in vals:
            bin_vec = (df[col] == v).astype(int)
            corr, _ = pointbiserialr(bin_vec, df["__y__"])
            # cast int keys for nicer JSON if integers; keep as str if not
            key = int(v) if (isinstance(v, (int, np.integer)) or float(v).is_integer()) else float(v)
            v_corrs[key] = float(0.0 if pd.isna(corr) else corr)
        corr_dict[col] = dict(sorted(v_corrs.items(), key=lambda x: x[0]))

    # --- Pretty print ---
    print("\n=== Value-wise Correlation Dictionary ===")
    for feature, value_corrs in corr_dict.items():
        print(f"\nFeature: {feature}")
        for val, corr in value_corrs.items():
            print(f"  {val} -> {corr:.3f}")

    return corr_dict


def feature_mutual_info(df: pd.DataFrame, label_col: str, categorical_features: list, numerical_features: list):
    """
    Compute mutual information between each feature and the (multi-class) label.
    Returns: dict {feature: score}
    """
    X = pd.get_dummies(df[categorical_features + numerical_features], drop_first=False)
    y = df[label_col]
    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    return dict(zip(X.columns, mi))


# =========================
# Sigma selection
# =========================

def compute_sigmas(
    df: pd.DataFrame,
    label_col: str = "income",
    pos_label: str = ">50K",
    numerical_features: list = None,
    categorical_features: list = None,
    # knobs:
    numeric_scale: float = 0.2,      # global multiplier for numeric sigma
    min_numeric_sigma: float = 1e-3, # floor for numeric
    corr_shrink: float = 0.6,        # shrink by |corr| (0=no shrink, 1=full)
    p0: float = 0.30,                # baseline flip prob for categorical
    min_flip: float = 0.01,          # floor for categorical flip prob
    use_mad: bool = True             # True: MAD; False: IQR-based
):
    """
    Returns: dict {feature_name: sigma_value}
      * numerical features -> numeric sigma
      * categorical features -> flip probability in [0,1]
    """
    y = (df[label_col].astype(str) == pos_label).astype(int)

    # Detect lists if not provided
    if numerical_features is None:
        numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Exclude the label if present
    numerical_features = [c for c in numerical_features if c != label_col]
    categorical_features = [c for c in categorical_features if c != label_col]

    sigmas = {}

    # ---- Numerical ----
    for col in numerical_features:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        x_valid = x.dropna()
        if len(x_valid) < 3 or x_valid.nunique() <= 1:
            sigmas[col] = float(min_numeric_sigma)
            continue

        if use_mad:
            # MAD scaled to be comparable to std
            robust_spread = float(median_abs_deviation(x_valid, scale="normal", nan_policy="omit"))
        else:
            q95, q5 = np.percentile(x_valid, [95, 5])
            robust_spread = (q95 - q5) / 2.0

        if robust_spread == 0:
            robust_spread = float(x_valid.std() or 1.0)

        mask = x.notna() & y.notna()
        r, _ = (pearsonr(x[mask], y[mask]) if mask.sum() >= 3 else (0.0, None))
        r = 0.0 if pd.isna(r) else float(r)

        shrink = max(0.0, 1.0 - corr_shrink * min(abs(r), 1.0))
        sig = max(min_numeric_sigma, numeric_scale * robust_spread * shrink)
        sigmas[col] = float(sig)

    # ---- Categorical ----
    for col in categorical_features:
        if col not in df.columns:
            continue
        s = df[col].astype("object")
        values = [v for v in s.dropna().unique()]
        if len(values) == 0:
            sigmas[col] = float(min_flip)
            continue

        max_abs_r = 0.0
        for v in values:
            bin_vec = (s == v).astype(int)
            mask = bin_vec.notna() & y.notna()
            if mask.sum() < 3:
                r = 0.0
            else:
                r, _ = pointbiserialr(bin_vec[mask], y[mask])
                r = 0.0 if pd.isna(r) else float(r)
            max_abs_r = max(max_abs_r, abs(r))

        p = s.value_counts(normalize=True)
        ent = float(entropy(p, base=2))
        ent_norm = ent / np.log2(len(p)) if len(p) > 1 else 0.0

        flip_prob = p0 * (1.0 - max_abs_r) * (0.8 + 0.4 * ent_norm)
        flip_prob = float(np.clip(flip_prob, min_flip, 0.9))
        sigmas[col] = flip_prob

    # Final guard: never include label in sigmas
    if label_col in sigmas:
        sigmas.pop(label_col)
    return sigmas


def compute_sigmas_multiclass(
    df: pd.DataFrame,
    label_col: str,
    numerical_features: list,
    categorical_features: list,
    numeric_scale: float = 0.2,
    min_numeric_sigma: float = 1e-3,
    p0: float = 0.3,
    min_flip: float = 0.01,
):
    """
    Multi-class version of compute_sigmas.
    Uses feature spread (numeric) and mutual information (categorical/numeric) instead of correlation.
    """
    from sklearn.feature_selection import mutual_info_classif

    sigmas = {}

    # mutual info for all features
    X = pd.get_dummies(df[categorical_features + numerical_features], drop_first=False)
    y = df[label_col]
    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_scores = dict(zip(X.columns, mi))

    # ---- Numerical ----
    for col in numerical_features:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x) < 3 or x.nunique() <= 1:
            sigmas[col] = min_numeric_sigma
            continue
        robust_spread = float(median_abs_deviation(x, scale="normal", nan_policy="omit"))
        if robust_spread == 0:
            robust_spread = float(x.std() or 1.0)

        shrink = 1 - min(1.0, mi_scores.get(col, 0))  # more MI → less noise
        sigmas[col] = max(min_numeric_sigma, numeric_scale * robust_spread * shrink)

    # ---- Categorical ----
    for col in categorical_features:
        p = df[col].value_counts(normalize=True)
        ent = float(entropy(p, base=2))
        ent_norm = ent / np.log2(len(p)) if len(p) > 1 else 0.0

        shrink = 1 - min(1.0, mi_scores.get(col, 0))
        flip_prob = p0 * shrink * (0.8 + 0.4 * ent_norm)
        sigmas[col] = float(np.clip(flip_prob, min_flip, 0.9))

    return sigmas


