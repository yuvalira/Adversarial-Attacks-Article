import pandas as pd
import numpy as np
import torch
import os
import pickle
import joblib
import importlib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_name = "adult_income"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)


parameters_module_path = f"{dataset_name}.parameters"
parameters_module = importlib.import_module(parameters_module_path)

sigmas               = parameters_module.sigmas
numerical_features   = parameters_module.numerical_features
categorical_features = parameters_module.categorical_features
all_features         = numerical_features + categorical_features

# Read GBT model (sklearn)
model_path = os.path.join(project_root, dataset_name, 'sklearn_GBT_model.joblib')
print(model_path)
assert os.path.exists(model_path), f"Model file not found at {model_path}"
model_gbt = joblib.load(model_path)
print("sklearn GBT model loaded successfully.")

# Load label encoders for decoding categorical features
encoders = 'label_encoders.pkl'
encoders_path = os.path.join(project_root, dataset_name, encoders)
with open(encoders_path, 'rb') as f:
    encoders = pickle.load(f)
print("Label encoders loaded successfully.")

print("Encoder keys:", encoders.keys())
print(f"Type of encoder for 'workclass': {type(encoders['workclass'])}")



def predict(input_df: pd.DataFrame, categorical_features=categorical_features, encoders=encoders, model_gbt=model_gbt) -> tuple[np.ndarray, np.ndarray]:
    """
    Makes predictions using the loaded GBT model, handling label encoding for
    categorical features.
    Args: input_df (pd.DataFrame): DataFrame containing the features for prediction.
                                 Can contain string categorical values.
    Returns: Tuple: Predicted class labels (0 or 1), Confidence scores (probability of class 1, i.e., > 50K)
    """
    # Create a copy of the input to avoid modifying the original DataFrame
    processed_df = input_df.copy()

    # Apply label encoding to each categorical feature using the saved encoders
    for feature in categorical_features:
        if feature in processed_df.columns and feature in encoders:
            le = encoders[feature]
            try:
                # Transform known categories
                processed_df[feature] = le.transform(processed_df[feature])
            except ValueError as e:
                # Handle unseen categories by mapping them to 0 (first known class)
                print(f"Warning: Could not transform feature '{feature}' due to unknown category. Error: {e}")

                unknown_cats = processed_df[feature][~processed_df[feature].isin(le.classes_)]
                if not unknown_cats.empty:
                    print(f"Mapping unknown categories {unknown_cats.unique().tolist()} in '{feature}' to 0 (first class).")
                    processed_df[feature] = processed_df[feature].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )

    # Get predicted class labels
    predictions = model_gbt.predict(processed_df)

    # Get probabilities of class 1 (i.e., > 50K)
    probabilities = model_gbt.predict_proba(processed_df)[:, 1]

    return predictions, probabilities





