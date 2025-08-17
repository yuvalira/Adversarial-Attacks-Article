import numpy as np
# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def print_weighted_correlation_summary(distances, batch_sizes):

    print(f"Total number of batches: {len(distances)}")
    print("\nFinal correlation distance from original samples:")

    weighted_sum = 0
    total_size = 0

    for i, batch_distances in enumerate(distances):
        if batch_distances and i < len(batch_sizes):
            final_distance = float(batch_distances[-1])
            batch_size = batch_sizes[i]
            weighted_sum += final_distance * batch_size
            total_size += batch_size
            print(f"Batch {i + 1} - {final_distance:.6f}")
        else:
            print(f"Batch {i + 1} - No data")

    if total_size > 0:
        weighted_avg = weighted_sum / total_size
        print(f"\nOverall weighted average distance across all adversarial samples: {weighted_avg:.6f}")
    else:
        print("\nNo valid distances to summarize.")


def summarize_feature_changes(original_df, adversarial_df, features):
    # Align rows
    original_df = original_df.reset_index(drop=True)
    adversarial_df = adversarial_df.reset_index(drop=True)

    # Count changes per feature
    change_counts = {}
    for feature in features:
        changes = original_df[feature] != adversarial_df[feature]
        change_counts[feature] = int(changes.sum())

    # Compute summary stats
    total_samples = len(adversarial_df)
    attempted_attacks = (adversarial_df['IsCorrect'] == True).sum()
    successful_attacks = adversarial_df['IsAdversarial'].sum()
    total_feature_changes = sum(change_counts.values())
    success_rate = (successful_attacks / attempted_attacks) * 100 if attempted_attacks else 0

    # Print global summary
    print("=== Adversarial Change Summary ===\n")
    print(f"Total datapoints: {total_samples}")
    print(f"Attempted attacks (on correct predictions): {attempted_attacks}")
    print(f"Successful attacks (prediction flipped):    {successful_attacks}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Total number of feature-level changes:      {total_feature_changes}\n")

    # Print feature-level summary
    print("=== Feature-Level Change Stats ===\n")
    for feature in features:
        count = change_counts[feature]
        # How many of the successful attacks involved a change in this feature
        successful_change_rate = count / successful_attacks if successful_attacks else 0
        fraction_of_total_change = count / total_feature_changes if total_feature_changes else 0

        print(f"{feature}:")
        print(f"  - Changed in {count} rows ({successful_change_rate:.2%} of successful attacks)")
        print(f"  - Fraction of total changes: {fraction_of_total_change:.2%}\n")


def display_attacks(original_df, adversarial_df, correlation_dict, all_features, n=5):
    count = 0

    for i in range(len(original_df)):
        adversarial_dp = adversarial_df.iloc[i]

        # Only show successful attacks
        if not adversarial_dp.get("IsAdversarial", False):
            continue

        original_dp = original_df.iloc[i]

        print(f"\n=== Attack #{count + 1} ===")
        print("Original:")
        print(original_dp[all_features])

        print("\nAdversarial:")
        print(adversarial_dp[all_features + ['original_label', 'original_prediction', 'final_prediction']])

        # Features that changed
        changed_features = [
            feature for feature in all_features
            if original_dp[feature] != adversarial_dp.get(feature, None)
        ]
        print("\nChanged features:")
        print(changed_features if changed_features else "None")

        # Compute L2 distance using correlation_dict
        squared_diffs = []
        for feature in all_features:
            orig_val = original_dp[feature]
            adv_val = adversarial_dp.get(feature, None)
            orig_corr = correlation_dict[feature].get(orig_val, 0.0)
            adv_corr = correlation_dict[feature].get(adv_val, 0.0)
            squared_diffs.append((orig_corr - adv_corr) ** 2)

        distance_score = np.sqrt(np.sum(squared_diffs))
        print(f"\nDistance score: {distance_score:.6f}")

        count += 1
        if count >= n:
            break


def compare_attack_results(original_df, adversarial_df, y_column='income'):

    # Prepare original labels (y = 1 if >50K, else 0)
    y_original = original_df[y_column].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)

    # Predictions (already stored in adversarial_df)
    y_pred_original = adversarial_df['original_prediction']
    y_pred_attacked = adversarial_df['final_prediction']
    y_attacked = adversarial_df['original_label']  # true labels for the attacked set

    # Compute metrics
    original_accuracy = accuracy_score(y_original, y_pred_original)
    original_precision = precision_score(y_original, y_pred_original)
    original_recall = recall_score(y_original, y_pred_original)
    original_f1 = f1_score(y_original, y_pred_original)

    attacked_accuracy = accuracy_score(y_attacked, y_pred_attacked)
    attacked_precision = precision_score(y_attacked, y_pred_attacked)
    attacked_recall = recall_score(y_attacked, y_pred_attacked)
    attacked_f1 = f1_score(y_attacked, y_pred_attacked)

    # Print comparison
    print("\n=== Model Performance Comparison ===")
    print(f"Original Accuracy:  {original_accuracy:.4f} | Attacked Accuracy:  {attacked_accuracy:.4f}")
    print(f"Original Precision: {original_precision:.4f} | Attacked Precision: {attacked_precision:.4f}")
    print(f"Original Recall:    {original_recall:.4f} | Attacked Recall:    {attacked_recall:.4f}")
    print(f"Original F1-Score:  {original_f1:.4f} | Attacked F1-Score:  {attacked_f1:.4f}")

    # Accuracy drop
    acc_drop = original_accuracy - attacked_accuracy
    print(f"\n Accuracy Drop due to Attack: {acc_drop:.4f} ({acc_drop * 100:.2f}% decrease)")

def plot_roc_comparison(df_attacked, label_col='original_label', original_score_col='original_score', attacked_score_col='attacked_score'):
    # Extract true labels
    y_true = df_attacked[label_col]

    # Extract scores
    y_score_orig = df_attacked[original_score_col]
    y_score_att = df_attacked[attacked_score_col]

    # Compute ROC and AUC
    fpr_orig, tpr_orig, _ = roc_curve(y_true, y_score_orig)
    auc_orig = roc_auc_score(y_true, y_score_orig)

    fpr_att, tpr_att, _ = roc_curve(y_true, y_score_att)
    auc_att = roc_auc_score(y_true, y_score_att)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_orig, tpr_orig, label=f"Original (AUC = {auc_orig:.4f})", linewidth=2)
    plt.plot(fpr_att, tpr_att, label=f"Attacked (AUC = {auc_att:.4f})", linestyle='--', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Original vs Attacked")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_pca_projection(df_original,df_attacked,encoders,title_prefix=''):
    class_map={0: '<= 50k', 1: '> 50k'}
    custom_colors={'<= 50k': 'lightsteelblue', '> 50k': 'dimgrey'}

    # Define feature groups
    numerical_features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    all_features = numerical_features + categorical_features

    def encode_df(df, label_column):
        df = df.copy()

        # Apply label encoders to categorical features
        for col in categorical_features:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])
            else:
                raise ValueError(f"No encoder found for column: {col}")

        features = df[all_features]
        labels_raw = df[label_column].values
        labels = np.vectorize(class_map.get)(labels_raw)
        return features.values, labels

    # Encode both datasets
    X_orig, y_orig = encode_df(df_original, label_column='income_binary')
    X_attacked, y_attacked = encode_df(df_attacked, label_column='final_prediction')

    # Normalize
    scaler = StandardScaler()
    X_orig_scaled = scaler.fit_transform(X_orig)
    X_att_scaled = scaler.transform(X_attacked)

    # PCA
    pca = PCA(n_components=2)
    X_orig_pca = pca.fit_transform(X_orig_scaled)
    X_att_pca = pca.transform(X_att_scaled)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for label in np.unique(y_orig):
        idx = y_orig == label
        axes[0].scatter(X_orig_pca[idx, 0], X_orig_pca[idx, 1],
                        c=custom_colors[label], label=label, alpha=0.6)
    axes[0].set_title(f'{title_prefix}Original Data')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend(title='Income')

    for label in np.unique(y_attacked):
        idx = y_attacked == label
        axes[1].scatter(X_att_pca[idx, 0], X_att_pca[idx, 1],
                        c=custom_colors[label], label=label, alpha=0.6)
    axes[1].set_title(f'{title_prefix}Attacked Data')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].legend(title='Income')

    plt.tight_layout()
    plt.show()


