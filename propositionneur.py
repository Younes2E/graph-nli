from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

model_id = "Zual/MPropositioneur-V2-large"

tokenizer = AutoTokenizer.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

texte = "狗和猫在厨房里。"

prompt = f"<|im_start|>user\nAtomize: {texte}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

generated_ids = outputs[0][inputs.input_ids.shape[1]:]
result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# La sortie est une liste JSON : ["p1", "p2", ...]
propositions = json.loads(result)
for p in propositions:
    print(f"• {p}")
