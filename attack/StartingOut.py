from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch
from huggingface_hub import hf_hub_download
import pandas as pd

model_id = "google/gemma-3-1b-it"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

csv_path = hf_hub_download(
    repo_id="ETdanR/Balanced_Adult_Split_DS",
    filename="experiment_data.csv",
    repo_type="dataset"
)
experiment_data = pd.read_csv(csv_path)

row = experiment_data.iloc[0]

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
).to(model.device).to(torch.bfloat16)


with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64)

outputs = tokenizer.batch_decode(outputs)

print(outputs)
