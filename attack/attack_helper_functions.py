import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr



def extract_batch(df, batch_length, last_row_handled):
    start = last_row_handled # Start index of the current batch

    # If all rows have been handled, return None
    if start >= len(df):
        return None, None

    end = min(start + batch_length, len(df))  # End index of the batch

    # Extract batch from DataFrame
    batch = df.iloc[start:end]

    # Update the last handled row index
    new_last_row_handled = end

    return batch, new_last_row_handled

def categorical_gaussian_perturb(datapoint, correlation_dict, sigmas, categorical_features):
    """
    Applies Gaussian-based perturbation to categorical features based on correlation scores.
    """

    # Create a copy to avoid modifying the original input
    perturbed_datapoint = datapoint.copy()

    # Iterate over all features
    for feature in datapoint.columns:
        # Only perturb categorical features
        if feature not in categorical_features:
            continue

        # Current value of the feature
        current_value = datapoint[feature].iloc[0]
        category_scores = correlation_dict[feature]
        categories = list(category_scores.keys())

        # Get correlation score positions
        positions = np.array([category_scores[cat] for cat in categories])
        center = category_scores[current_value]

        # Standard deviation for this feature
        feature_sigma = sigmas[feature]

        # Compute Gaussian weights based on correlation distance
        distances = positions - center
        weights = np.exp(-0.5 * (distances / feature_sigma) ** 2)
        weights /= weights.sum()

        # Sample new category based on weights
        sampled_index = np.random.choice(len(categories), p=weights)
        new_value = categories[sampled_index]

        # Update the feature with the new value
        perturbed_datapoint[feature] = new_value

    return perturbed_datapoint

def numerical_gaussian_perturb(datapoint, correlation_dict, sigmas, numerical_features):
    """
    Applies Gaussian-based perturbation to numerical features based on correlation distance.
    """

    # Copy to avoid modifying the original input
    perturbed_datapoint = datapoint.copy()

    # Iterate over all features
    for feature in datapoint.columns:
        # Only perturb numerical features
        if feature not in numerical_features:
            continue

        current_value = datapoint[feature].iloc[0] # Current numerical value

        # Possible values based on correlation dictionary
        possible_values = [float(val) for val in correlation_dict[feature].keys()]
        positions = np.array(possible_values)
        center = current_value

        # Standard deviation for this feature
        feature_sigma = sigmas[feature]

        # Compute Gaussian weights
        distances = positions - center
        weights = np.exp(-0.5 * (distances / feature_sigma) ** 2)
        weights /= weights.sum()

        # Sample new value using the computed weights
        sampled_value = np.random.choice(possible_values, p=weights, size=1)[0]

        # Update feature with the sampled value
        perturbed_datapoint[feature] = sampled_value

    return perturbed_datapoint

def find_feature_to_minimize(current_dp, original_dp, correlation_dict, saturated_features):
    max_score_diff = -float('inf')
    feature_to_minimize = None

    for feature in current_dp.columns:
        if feature in saturated_features:
            continue

        # get features
        current_val = current_dp.iloc[0][feature]
        original_val = original_dp.iloc[0][feature]

        # Get correlation scores
        current_score = correlation_dict[feature].get(current_val, 0.0)
        original_score = correlation_dict[feature].get(original_val, 0.0)

        score_diff = abs(original_score - current_score)

        if score_diff > max_score_diff:
            max_score_diff = score_diff
            feature_to_minimize = feature

    return feature_to_minimize

def minimize_feature(current_dp, feature, correlation_dict, original_dp):
    # Get current and original values
    current_val = current_dp.iloc[0][feature]
    original_val = original_dp.iloc[0][feature]

    # Sort feature values by correlation score
    ordered_values = list(correlation_dict[feature].keys())

    current_index = ordered_values.index(current_val)
    original_index = ordered_values.index(original_val)

    # Step one position toward the original
    sign = (original_index - current_index) // abs(original_index - current_index)
    new_index = current_index + sign

    new_val = ordered_values[new_index]
    current_dp.loc[current_dp.index[0], feature] = new_val
    return current_dp

def batch_distance_score(batch_original, batch_minimized, correlation_dict):

    total_squared_diffs = []

    for i in range(len(batch_original)):
        original_dp = batch_original.iloc[[i]]
        minimized_dp = batch_minimized.iloc[[i]]

        squared_diffs_per_dp = []

        for feature in batch_minimized.columns:
            original_val = original_dp.at[original_dp.index[0], feature]
            minimized_val = minimized_dp.at[minimized_dp.index[0], feature]

            original_corr_score = correlation_dict[feature].get(original_val, 0.0)
            minimized_corr_score = correlation_dict[feature].get(minimized_val, 0.0)

            # Calculate squared absolute difference
            squared_diff = abs(original_corr_score - minimized_corr_score)**2
            squared_diffs_per_dp.append(squared_diff)


        total_squared_diff_dp = np.sum(squared_diffs_per_dp)
        distance_for_dp = np.sqrt(total_squared_diff_dp)
        total_squared_diffs.append(distance_for_dp)


    # Average score across all datapoints
    average_score_across_datapoints = np.mean(total_squared_diffs)

    return average_score_across_datapoints


def compute_point_biserial_correlation_dictionary(df, features_main, target, categorical_features):
    # Initialize correlation dictionary
    correlation_dict = {}

    # Compute point-biserial correlation for each value in each feature
    for col in features_main.columns:
        value_corrs = {}

        if df[col].dtype == 'object':
            # For categorical features: compute correlation for each category
            for val in df[col].unique():
                binary_vec = df[col].apply(lambda x: 1 if x == val else 0)
                corr, _ = pointbiserialr(binary_vec, target)
                value_corrs[val] = float(corr if pd.notnull(corr) else 0)
        else:
            # For numerical features: treat each value as a separate binary indicator
            for val in sorted(df[col].unique()):
                binary_vec = df[col].apply(lambda x: 1 if x == val else 0)
                corr, _ = pointbiserialr(binary_vec, target)
                value_corrs[int(val)] = float(corr if pd.notnull(corr) else 0)

        # Sort values: categorical by descending correlation, numerical by value
        if col in categorical_features:
            correlation_dict[col] = dict(sorted(value_corrs.items(), key=lambda x: x[1], reverse=True))
        else:
            correlation_dict[col] = dict(sorted(value_corrs.items(), key=lambda x: x[0]))

    return correlation_dict


def print_correlation_dict(correlation_dict):
    # Print correlation scores for each feature and value
    for feature, value_corrs in correlation_dict.items():
        print(f"\nFeature: {feature}")
        for val, corr in value_corrs.items():
            print(f"  {val} → {corr:.3f}")