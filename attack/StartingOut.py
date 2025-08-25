from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch
from huggingface_hub import hf_hub_download
import pandas as pd

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

for index in range (20):
    row = experiment_data.iloc[index]

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
    sentence_parts.append(f"income: __BLANK__ than 50k")
    sentence = ", ".join(sentence_parts)


    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text":  f"Context: {sentence}. Based on this, should the __BLANK__ be 'Greater' or 'Less'? Please provide a one-word answer."},]
            },
        ],
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=64)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(decoded)

