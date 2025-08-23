# import the functions you defined
from datasets.dataset_helper_functions import *

# -----------------------------
# Hugging Face Authentication
# -----------------------------
from huggingface_hub import login, whoami

HF_TOKEN = "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"

# Log in
login(token=HF_TOKEN)
print("Logged in as:", whoami()["name"])

# ----------------------------
# 1. Parameters
# ----------------------------
KAGGLE_ID = "mysticvalley/balance-scale"
LABEL_COL = "Class"
POS_LABEL = "B"
HF_REPO_ID = "yuvalira/Balance-Scale_balanced"
SEED = 42

# ----------------------------
# 2. Download + load CSV
# ----------------------------
path = download_kaggle_dataset(KAGGLE_ID)
df = load_first_csv_in(path)

print("Dataset shape:", df.shape)
print(df.head())

# ----------------------------
# 3. Feature lists + EDA
# ----------------------------
numerical_features, categorical_features = get_feature_lists(df, LABEL_COL)
print("Numerical:", numerical_features)
print("Categorical:", categorical_features)

plot_label_pie(df, LABEL_COL)

# ----------------------------
# 4. Balance + splits
# ----------------------------
df_bal = undersample_to_balance_multiclass(df, LABEL_COL, seed=SEED)
train_df, val_df, test_df = stratified_splits(df_bal, LABEL_COL, seed=SEED)

print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
print("Train dist:\n", train_df[LABEL_COL].value_counts(normalize=True))
print("Val dist:\n", val_df[LABEL_COL].value_counts(normalize=True))
print("Test dist:\n", test_df[LABEL_COL].value_counts(normalize=True))

# ----------------------------
# 5. Upload splits to HF Hub
# ----------------------------

upload_csv_splits_to_hub(train_df, val_df, test_df, repo_id=HF_REPO_ID)

# ----------------------------
# 6. Correlations (on test)
# ----------------------------

corr_dict = feature_mutual_info(
    test_df,
    label_col=LABEL_COL,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
)

# ----------------------------
# 7. Sigmas (on test)
# ----------------------------
sigmas = compute_sigmas_multiclass(
    test_df,
    label_col=LABEL_COL,
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    numeric_scale=0.2,
    min_numeric_sigma=1e-3,
    p0=0.30,
    min_flip= 0.01
)

print("=== Sigmas per feature ===")
for feat, sigma in sigmas.items():
    print(f"{feat:16s} -> {sigma:.4f}")
