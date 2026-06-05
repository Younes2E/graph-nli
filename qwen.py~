import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

dataset = load_dataset("stanfordnlp/snli")
train_set = dataset["train"].filter(lambda x: x["label"] != -1)
#labels = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


model_id = "Zual/MPropositioneur-V2-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)


def extract_atomic_propositions(text):
    prompt = f"<|im_start|>user\nAtomize: {text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    propositions = json.loads(result)
    
    return propositions


def main():
    lines = ["Two dogs are eating and playing football","Le chien de Louis XIV aime manger.", "Il mange une salade ou un burger.", "Il peut manger une salade ou un burger.", "Il mange soit une salade soit un burger.","Les enclos bretons sont une église entourée d’un placître."]
    for l in lines:
        print(f"Entrée : {l}\nRéponse du modèle : {extract_atomic_propositions(l)}")

if __name__=="__main__":
    main()
    
    
    
    
    
