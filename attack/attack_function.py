from attack_helper_functions import *
from prediction_functions import *
import pandas as pd

def generate_adversarial_dataset(
    df,
    modelArch,  # "LM" for predict_LM, "GBT" for predict_GBT
    correlation_dict,
    sigmas,
    numerical_features,
    categorical_features,
    num_perturb_iterations,
    num_minimize_iterations,
    batch_size: int = 32,
    random_seed = 42
):
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Initialize state variables
    last_row_handled = 0
    total_rows = len(df)
    all_adversarial_batches = []
    all_perturbation_accuracies = []
    all_minimization_distances = []
    batch_sizes = []

    # Process the dataset in batches
    while last_row_handled < total_rows:
        print(f"\n--- Processing batch starting from row {last_row_handled} ---")

        # Extract current batch
        true_batch, last_row_handled = extract_batch(df, batch_size, last_row_handled)
        batch = true_batch.copy()
        batch_sizes.append(len(batch))

        # Get true labels and prepare input features
        true_labels = true_batch['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0).to_numpy()
        batch.drop(columns=['income'], inplace=True)
        batch.drop(columns=['income_binary'], inplace=True)

        # Get initial predictions
        current_labels, original_scores = predict_LM(batch) if modelArch == "LM" else predict_GBT(batch)
        original_predictions = current_labels.copy()
        original_incorrect_predictions = (original_predictions != true_labels)

        # -------------------------------------------------------------------------------
        #                               Perturbation Loop
        # -------------------------------------------------------------------------------
        print(f"Original accuracy: {sum(current_labels==true_labels)/len(batch)}")

        perturbation_accuracies = []

        for perturb_iteration in range(num_perturb_iterations):
            # Record accuracy before perturbation
            acc = sum(current_labels == true_labels) / len(batch)
            perturbation_accuracies.append(acc)

            # Apply perturbations to correctly classified samples
            for index in range(len(batch)):
                datapoint = batch.iloc[[index]]
                if true_labels[index] == current_labels[index]:
                    temp_dp = categorical_gaussian_perturb(datapoint, correlation_dict, sigmas, categorical_features)
                    temp_dp = numerical_gaussian_perturb(temp_dp, correlation_dict, sigmas, numerical_features)
                    batch.iloc[[index]] = temp_dp

            # Get predictions after perturbation
            current_labels,_ = predict_LM(batch) if modelArch == "LM" else predict_GBT(batch)

            # Revert datapoints that failed to flip prediction
            for index in range(len(batch)):
                if true_labels[index] == current_labels[index]:
                    batch.iloc[[index]] = true_batch.iloc[[index]]

            print(f"{perturb_iteration + 1}/{num_perturb_iterations} Accuracy: {sum(current_labels==true_labels)/len(batch)}")

        # Mark successful adversarial examples (flipped predictions)
        successfully_flipped = (current_labels != true_labels) & (original_incorrect_predictions == False)

        # -------------------------------------------------------------------------------
        #                             Minimization Loop
        # -------------------------------------------------------------------------------
        # Track saturation of features per sample
        feature_saturation_map = [set() for _ in range(len(batch))]
        datapoint_saturated = ~successfully_flipped
        minimization_distances = []

        # Initialize saturated features (those already equal to original)
        for datapoint_index in range(len(batch)):
            for feature in batch.columns:
                current_val = batch.iloc[datapoint_index][feature]
                original_val = true_batch.iloc[datapoint_index][feature]
                if current_val == original_val:
                    feature_saturation_map[datapoint_index].add(feature)

        # Run minimization steps
        for minimize_iteration in range(num_minimize_iterations):
            # Measure current perturbation distance
            distance = batch_distance_score(true_batch, batch, correlation_dict)
            minimization_distances.append(distance)

            # Backup batch before minimization
            batch_before_minimization = batch.copy()
            labels_before_minimization,_ = predict_LM(batch_before_minimization) if modelArch == "LM" else predict_GBT(batch_before_minimization)

            for datapoint_index in range(len(batch)):
                if datapoint_saturated[datapoint_index]:
                    continue  # Skip already saturated samples

                datapoint = batch.iloc[[datapoint_index]]
                original_datapoint = true_batch.iloc[[datapoint_index]]

                # Select and minimize the most changed feature
                feature = find_feature_to_minimize(datapoint, original_datapoint, correlation_dict, feature_saturation_map[datapoint_index])
                datapoint_minimized = minimize_feature(datapoint, feature, correlation_dict, true_batch.iloc[[datapoint_index]])
                batch.iloc[[datapoint_index]] = datapoint_minimized

                # Mark feature as saturated if it reached original value
                if datapoint_minimized[feature].iloc[0] == original_datapoint[feature].iloc[0]:
                    feature_saturation_map[datapoint_index].add(feature)

            # Get predictions after minimization
            current_labels,_ = predict_LM(batch) if modelArch == "LM" else predict_GBT(batch)

            for datapoint_index in range(len(batch)):
                if datapoint_saturated[datapoint_index]:
                    continue

                if current_labels[datapoint_index] == true_labels[datapoint_index]:
                    # Prediction reverted — undo minimization
                    batch.iloc[[datapoint_index]] = batch_before_minimization.iloc[[datapoint_index]]

                    # Mark the attempted feature as saturated
                    last_feature = find_feature_to_minimize(
                        batch_before_minimization.iloc[[datapoint_index]],
                        original_datapoint,
                        correlation_dict,
                        feature_saturation_map[datapoint_index]
                    )
                    feature_saturation_map[datapoint_index].add(last_feature)

                # Check if all features are now saturated
                if len(feature_saturation_map[datapoint_index]) >= len(batch.columns):
                    datapoint_saturated[datapoint_index] = True

            # Early exit if all samples are saturated
            if all(datapoint_saturated):
                print(f"All datapoints saturated. Exiting minimization loop early. Iteration num: {minimize_iteration}")
                break

        all_perturbation_accuracies.append(perturbation_accuracies)
        all_minimization_distances.append(minimization_distances)

        # Final predictions after perturbation + minimization
        final_predictions, final_scores = predict_LM(batch) if modelArch == "LM" else predict_GBT(batch)

        # Annotate adversarial success per row
        batch['original_label'] = true_labels
        batch['original_prediction'] = original_predictions
        batch['final_prediction'] = final_predictions
        batch['IsCorrect'] = batch['original_label'] == batch['original_prediction']
        batch['IsAdversarial'] = (batch['original_label'] != batch['final_prediction']) & batch['IsCorrect']

        # Add score columns
        batch['score_before_attack'] = original_scores
        batch['score_after_attack'] = final_scores

        # Store final adversarial batch
        all_adversarial_batches.append(batch)

    # Combine all batches into a single DataFrame
    full_adversarial_dataset = pd.concat(all_adversarial_batches, ignore_index=True)
    print("\nAdversarial dataset generation complete!")

    return full_adversarial_dataset, all_perturbation_accuracies, all_minimization_distances, batch_sizes