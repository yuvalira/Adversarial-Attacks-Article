# System and file handling
import os
import importlib

# Transformers and model loading
from huggingface_hub import hf_hub_download
import torch

from attack_helper_functions import *
from evaluation_helper_functions import *
from attack_function import generate_adversarial_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# set specific dataset and model to attack
dataset_name = "adult_income"
model_name = "GBT"

# Attack parameters
batch_size = 32
num_perturb_iterations = 5
num_minimize_iterations = 50


# load parameters for this permutation
parameters_module_path = f"{dataset_name}.parameters"
parameters_module = importlib.import_module(parameters_module_path)

sigmas               = parameters_module.sigmas
numerical_features   = parameters_module.numerical_features
categorical_features = parameters_module.categorical_features
all_features         = numerical_features + categorical_features
label_col            = parameters_module.label_col
positive_value       = parameters_module.positive_value

# set paths for this permutation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
figures_path = os.path.join(project_root, dataset_name, 'figures')
results_path = os.path.join(project_root, dataset_name, 'results')

# download dataset for this permutation
csv_path = hf_hub_download(
    repo_id="ETdanR/" + dataset_name,
    filename="experiment_data.csv",
    repo_type="dataset"
)
experiment_data = pd.read_csv(csv_path)
df = experiment_data.copy()

# # Create binary target label (1 for >50K, 0 otherwise)
# df['income_binary'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
#
# # Separate target and input features
# target = df['income_binary']
# features_main = df.drop(columns=['income', 'income_binary'])

# Create binary target label using the parameters file
df['target_binary'] = df[label_col].apply(lambda x: 1 if positive_value in str(x) else 0)

# Separate target and input features
target = df['target_binary']
features_main = df.drop(columns=[label_col, 'target_binary'])

correlation_dict = compute_point_biserial_correlation_dictionary(df, features_main, target, categorical_features)
# print_correlation_dict(correlation_dict)

# Define subset for attack
df_to_attack = df.copy()
df_to_attack = df.sample(n=50, random_state=42).copy()

# Run the adversarial attack and collect results
adversarial_df, accuracies, distances, batch_sizes = generate_adversarial_dataset(
    df_to_attack,
    model_name,
    correlation_dict,
    sigmas,
    numerical_features,
    categorical_features,
    label_col,
    positive_value,
    num_perturb_iterations,
    num_minimize_iterations,
    batch_size=32,
    random_seed=42
)

# Save adversarial dataset to CSV file
file_path = os.path.join(results_path, 'adversarial_df.csv')
adversarial_df.to_csv(file_path)


# print_weighted_correlation_summary(distances, batch_sizes)
# summarize_feature_changes(original_df=df_to_attack, adversarial_df=adversarial_df, features=all_features)
# display_attacks(original_df=df_to_attack, adversarial_df=adversarial_df, correlation_dict=correlation_dict, all_features=all_features, n=5)
compare_attack_results(df_to_attack, adversarial_df)
plot_roc_comparison(figures_path,
                    df_attacked=adversarial_df,
                    label_col='original_label',
                    original_score_col='score_before_attack',
                    attacked_score_col='score_after_attack'
                    )

# # Load label encoders for decoding categorical features
# encoders = 'label_encoders.pkl'
# encoders_path = dataset_name + '/' + encoders
# with open(encoders_path, 'rb') as f:
#     encoders = pickle.load(f)
# print("Label encoders loaded successfully.")

# plot_pca_projection(
#     df_original=df_to_attack,
#     df_attacked=adversarial_df_GBT,
#     encoders=encoders,
#     title_prefix='GBT '
# )

# plot_adversarial_shift_arrows(
#     df_original=df_to_attack,
#     df_attacked=adversarial_df_GBT,
#     encoders=encoders,
#     title_prefix='GBT '
# )


