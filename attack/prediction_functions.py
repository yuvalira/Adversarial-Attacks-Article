import pandas as pd
import numpy as np
import torch
import os
import pickle
import joblib
import importlib
from transformers import RobertaTokenizer, RobertaForMaskedLM



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_name = "adult_income"


parameters_module_path = f"{dataset_name}.parameters"
parameters_module = importlib.import_module(parameters_module_path)

sigmas               = parameters_module.sigmas
numerical_features   = parameters_module.numerical_features
categorical_features = parameters_module.categorical_features
all_features         = numerical_features + categorical_features

# Read GBT model (sklearn)
model = 'sklearn_GBT_model.joblib'
model_path = dataset_name + '/' + model
assert os.path.exists(model_path), f"Model file not found at {model_path}"
model_gbt = joblib.load(model_path)
print("sklearn GBT model loaded successfully.")

# Load label encoders for decoding categorical features
encoders = 'label_encoders.pkl'
encoders_path = dataset_name + '/' + encoders
with open(encoders_path, 'rb') as f:
    encoders = pickle.load(f)
print("Label encoders loaded successfully.")

print("Encoder keys:", encoders.keys())
print(f"Type of encoder for 'workclass': {type(encoders['workclass'])}")




# Load fine-tuned model directly from Hugging Face Hub
model_path = "ETdanR/RoBERTa_FT_adult"  # model repo
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path)
model.eval()
model.to(device)


def predict_GBT(input_df: pd.DataFrame, categorical_features=categorical_features, encoders=encoders, model_gbt=model_gbt) -> tuple[np.ndarray, np.ndarray]:
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







def predict_LM(df: pd.DataFrame, tokenizer = tokenizer, model = model, device = device):
    """
    Generates predictions for a DataFrame of input features using a Language Model.
    Args: df (pd.DataFrame): DataFrame with tabular features per row.
    Returns: np.ndarray: Array of predictions (0 for <=50K, 1 for >50K, -1 for unknown).
    """

    # Define mask token info
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id

    # Token IDs for income prediction (example IDs from your snippet)
    greater_than_id = 9312  # " Greater"
    less_than_id = 10862  # " Less"

    # Check if model and tokenizer are loaded
    if tokenizer is None or model is None:
        print("Error: Hugging Face tokenizer or model not loaded. Returning random predictions.")
        random_preds = np.random.randint(0, 2, size=len(df))
        random_scores = np.random.rand(len(df))
        return random_preds, random_scores

    # Convert each row into a pseudo-sentence with a masked income token
    sentences = []
    for _, row in df.iterrows():
        sentence_parts = [
            f"age: {row['age']}",
            f"workclass: {row['workclass']}",
            f"education: {row['education']}",
            f"educational-num: {row['educational-num']}",
            f"marital-status: {row['marital-status']}",
            f"occupation: {row['occupation']}",
            f"relationship: {row['relationship']}",
            f"race: {row['race']}",
            f"gender: {row['gender']}",
            f"capital-gain: {row['capital-gain']}",
            f"capital-loss: {row['capital-loss']}",
            f"hours-per-week: {row['hours-per-week']}",
            f"native-country: {row['native-country']}"
        ]
        sentence_parts.append(f"income: {mask_token} than 50k")
        sentences.append(", ".join(sentence_parts))

    # Tokenize sentences and move tensors to device
    encoded = tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    # Run the model to get logits (no gradient calculation)
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    # Locate the positions of mask tokens in the input
    mask_batch_indices, mask_token_indices = (encoded['input_ids'] == mask_token_id).nonzero(as_tuple=True)

    # Initialize result arrays
    pred_tensor = torch.full((len(df),), -1, dtype=torch.long, device=device)
    score_tensor = torch.full((len(df),), -1.0, dtype=torch.float, device=device)

    if mask_batch_indices.numel() > 0:
        # Extract logits at masked positions
        logits_at_mask = logits[mask_batch_indices, mask_token_indices]
        probs = torch.softmax(logits_at_mask, dim=-1)

        # Predict token IDs at mask and map to classes
        predicted_token_ids_at_mask = torch.argmax(logits_at_mask, dim=-1)
        is_greater = (predicted_token_ids_at_mask == greater_than_id)
        is_less = (predicted_token_ids_at_mask == less_than_id)

        mapped_predictions = torch.full_like(predicted_token_ids_at_mask, -1)
        mapped_predictions[is_greater] = 1
        mapped_predictions[is_less] = 0

        # Assign predictions to the correct batch positions
        pred_tensor[mask_batch_indices] = mapped_predictions
        score_tensor[mask_batch_indices] = probs[:, greater_than_id]

    # Return predictions and scores as NumPy arrays
    return pred_tensor.cpu().numpy(), score_tensor.cpu().numpy()