import json
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

dataset = load_dataset("stanfordnlp/snli")
train_set = dataset["train"].filter(lambda x: x["label"] != -1)
#labels = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


model_id = "Zual/MPropositioneur-V2-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)

qwen_model_id = "Qwen/Qwen2.5-7B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    torch_dtype="auto").to(device)




def extract_atomic_propositions(text):
    prompt = f"<|im_start|>user\nAtomize: {text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    propositions = json.loads(result)
    
    return set(propositions)


def extract_triplets_qwen(text):
    prompt = f"Extract all factual (subject, predicate, object) triples from sentence. One triple per line in the format: subject | predicate | object. No explanations. If no triple can be extracted, write nothing. Sentence: {text}"

    messages = [
    {"role": "user", "content": prompt}
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(device)

    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    triples = []
    for line in response.split("\n"):
        t = line.split(" | ")
        triples.append((t[0], t[1], t[2]))
    
    return triples


def pipeline(text): 
    propositions = extract_atomic_propositions(text)
    triplets_qwen = []
    for p in propositions:
        triplets_qwen+=extract_triplets_qwen(p)
            
    return propositions, triplets_qwen




texts = ["Frank's dog doesn't eat fruits, he is allergic","Simon didn't call me back, he is busy.", "Younes is working on a project. His friend is playing a video game.", "I have never seen anyone like Frank, he must be gifted.","He must be sick.", "He is sick."]
texts_snli = [train_set[i][j] for j in ["premise", "hypothesis"] for i in [4131,2656,922,1384,7048]]
for t in texts:
    texts_snli.append(t)

for t in texts_snli:
    prop_t, t_qwen = pipeline(t)
    print(f"{'-'*50}\nPhrase : {t} \nAvant Atomisation :")
    print(f"Triplet Qwen : {extract_triplets_qwen(t)}")
    print(f"Apres Atomisation : {prop_t}")
    print(f"Triplets Qwen : {t_qwen}")

    
    
    
    
    
    
    
    
    
    
