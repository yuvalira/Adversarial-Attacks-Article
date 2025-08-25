from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

model_id = "google/gemma-3-1b-it"

model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    torch_dtype="float32",  # or torch.float32 explicitly
    device_map="cpu"      # force CPU
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

csv_path = hf_hub_download(
    repo_id="ETdanR/adult_income",
    filename="experiment_data.csv",
    repo_type="dataset"
)
experiment_data = pd.read_csv(csv_path)


def predict(df: pd.DataFrame, tokenizer, model, device='cpu'):
    """
    Generates income predictions (0 for <=50K, 1 for >50K) using Gemma causal LM.
    Processes the entire DataFrame at once.

    Args:
        df (pd.DataFrame): Tabular input (already a batch).
        tokenizer: Hugging Face tokenizer for Gemma.
        model: Gemma model (Gemma3ForCausalLM).
        device: Device to run on ('cpu' or 'cuda').

    Returns:
        preds (np.ndarray): 0 or 1 predictions.
        confidences (np.ndarray): confidence score placeholder for ">50K" prediction.
    """
    # Build sentences for all rows
    all_sentences = []
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
        sentence_parts.append("income: __BLANK__ than 50k")
        all_sentences.append(", ".join(sentence_parts))

    # Build chat messages for the whole batch
    messages_batch = [
        [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": f"Context: {sentence}. Should the __BLANK__ be 'Greater' or 'Less'? Provide a one-word answer."}]},
        ]
        for sentence in all_sentences
    ]

    # Tokenize entire batch
    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,  # Add this line
        truncation=True # Add this line
    ).to(device)
    # Generate outputs
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=8)

    # Decode and map to 0/1
    decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(decoded_batch)
    preds, confidences = [], []
    for decoded in decoded_batch:
        # Split the string at the "model\n" delimiter and take the part after it
        # to isolate the model's output.
        print(decoded)
        try:
            model_output = decoded.split("model\n")[1].strip().lower()
        except IndexError:
            # Handle cases where the split doesn't work as expected
            print(f"Warning: Could not parse model output for decoded string: {decoded}")
            model_output = ""
        print(model_output)
        if "greater" in model_output:
            preds.append(1)
            confidences.append(1.0)
        elif "less" in model_output:
            preds.append(0)
            confidences.append(1.0)
        else:
            # Assign a neutral value like -1 or 0 for unparsed outputs
            preds.append(-1)
            confidences.append(-1.0)

    return np.array(preds), np.array(confidences)

batch_size = 32
batch_df = experiment_data.sample(n=batch_size, random_state=42)
predictions, confidence = predict_Gemma(batch_df, tokenizer, model)
print(predictions)