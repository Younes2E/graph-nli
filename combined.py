import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = "cpu"

model_id = "Zual/MPropositioneur-V2-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)

def extract_propositions(text):
    prompt = f"<|im_start|>user\nAtomize: {text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    propositions = json.loads(result)
    
    return set(propositions)

qwen_model_id = "Qwen/Qwen2.5-7B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    dtype=torch.float16).to(device)

with open('data/relations.json', encoding='utf-8') as file:
    relations = json.load(file)

with open('data/exemples.json', encoding='utf-8') as file:
    exemples_raw = json.load(file)
    lines = []
    for line in exemples_raw:
        lines.append(f'Input: "{line["input"]}"')
        lines.append("Output:")
        for triple in line['output']:
            lines.append(" | ".join(triple))
        lines.append("\n")
    exemples = "\n".join(lines)




def extract_triples(text):
    prompt = f"""
    Extract all factual ([SUBJECT], [RELATION], [OBJECT]) triples from sentence. 
    One triple per line in the format: 
    [SUBJECT] | [RELATION] | [OBJECT]

    No explanations. If no triple can be extracted, write nothing. 

    Allowed Relations and Type Constraints:
    {json.dumps(relations, indent = 1)}
    Do NOT use any other relation.

    ADDITIONAL RULES: 
    1. If an action is negated in the sentence (e.g., "didn't call", "is not eating"), you MUST capture the negation inside the [VERB] node using 'not' (e.g., 'not call', 'not eating').
    2. When an action involves multiple elements at once (e.g., an actor, a target, a recipient, a tool, or a location), do NOT link the secondary elements to each other. Instead, make the action the central hub and ALL links must involve the action.

    EXEMPLES:
    {exemples}

    
    Sentence: {text}"""

    messages = [
        {"role": "system", "content": "You are a deterministic Information Extraction expert."},
        {"role": "user", "content": prompt}
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        do_sample=False
        ##enable_thinking=False

    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(device)

    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    content = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    triples = []
    for line in content.split("\n"):
        t = line.split(" | ")
        try :
            triples.append((t[0].strip(), t[1].strip(), t[2].strip()))
        except : 
            print(f"\nMauvais format de triplet : {t}.\n")

    return triples


def extract_triples_propositions(text): 
    propositions = extract_propositions(text)
    triplets_qwen = []
    for p in propositions:
        triplets_qwen += extract_triples(p)
            
    return set(triplets_qwen)

    
    
    
    
    
    
    
    
    
