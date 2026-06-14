import json
import torch
from datasets import load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = "cpu"

dataset = load_dataset("stanfordnlp/snli")
snli = dataset["test"].filter(lambda x: x["label"] != -1)

dataset = pd.read_csv('data/sick_clean.csv')
labels = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


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
    
    return set(propositions)



qwen_model_id = "Qwen/Qwen2.5-7B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    dtype=torch.float16).to(device)


list_relations = {
'FormOf' : "'subject' is an inflected form of 'object'; 'object' is the root word of 'subject'" ,
'IsA' : "'subject' is a subtype or a specific instance of 'object'" ,
'PartOf' : "'subject' is a part of 'object'" ,
'HasA' : "'object' belongs to 'subject'" ,
'Contains': "'subject' contains 'object'",
'UsedFor' : "'subject' is used for 'object'; the purpose of 'subject' is 'object'" ,
'CapableOf' : "Something that 'subject' can typically do is 'object'" ,
'AtLocation' : "'subject' is located at 'object', 'subject' can be an event taking place at 'object'" ,
'Causes' : "'subject' and 'object' are events, and 'subject' causes 'object'" ,
'HasSubevent' : "'subject' and 'object' are events, and 'object' happens as a subevent of 'subject'" ,
'HasPrerequisite' : "In order for 'subject' to happen, 'object' needs to happen; 'object' is a dependency of 'subject'" ,
'HasProperty' : "'subject' has 'object' as a property; 'subject' can be described as 'object'" ,
'MotivatedByGoal' : "Someone does 'subject' because they want result 'object'; 'subject' is a step toward accomplishing the goal 'object'" ,
'CreatedBy' : "'object' is a process or agent that creates 'subject'" ,
'Synonym' : "'subject' and 'object' have very similar meanings. Symmetric" ,
'Antonym' : "'subject' and 'object' are opposites in some relevant way" ,
'SymbolOf' : "'subject' symbolically represents 'object'" ,
'SimilarTo' : "'subject' is similar to 'object'. Symmetric" ,
'MadeOf' : "'subject' is made of 'object'" ,
'ReceivesAction' : "'object' can be done to 'subject'",
'PerformsAction' : "'subject' is doing the action 'object', 'object' is usually a verb. If 'subject' is NOT performing action 'object' then 'subject' PerformsAction 'not object'"}



def extract_triplets_qwen(text):
    prompt = f"""
    Extract all factual (subject, relation, object) triples from sentence. 
    One triple per line in the format: 
    subject | relation | object

    No explanations. If no triple can be extracted, write nothing. 

    Here are the only relations available and how to use them : {list_relations}
    Do not use any other relation.

    Sentence: {text}"""

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
        max_new_tokens=512 ##4096 pour thinking
    )
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    content = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    triples = []
    for line in content.split("\n"):
        t = line.split(" | ")
        try :
            triples.append((t[0], t[1], t[2]))
        except : 
            print(f"{t}, Mauvais format de triplet.")

    return triples


def pipeline(text): 
    propositions = extract_atomic_propositions(text)
    triplets_qwen = []
    for p in propositions:
        triplets_qwen+=extract_triplets_qwen(p)
            
    return propositions, triplets_qwen





"""
np_data = dataset.to_numpy()
sentence_data = [np_data[i][j] for j in [1, 2] for i in [5,45,2450]]
data_supp = ["John is eating a ham pizza.", "He plays because he wants to be a football player."]
for l in data_supp:
    sentence_data.append(l)
"""
texts_snli = [snli[i][j] for j in ["premise", "hypothesis"] for i in [4031,265,9202,1389,9028, 3551]]
texts = ["Frank is in New York, but he plays football in Manchester.","The cat and the dog are in the kitchen.", "A cat and a dog are in a kitchen."]##["Frank's dog doesn't eat fruits, he is allergic","Simon didn't call me back, he is busy.", "Younes is working on a project. His friend is playing a video game.", "I have never seen anyone like Frank, he must be gifted.","He must be sick.", "He is sick.", "He played football after eating.","A cat and a dog are in the kitchen","Two dogs are eating and playing football"]
for t in texts:
    texts_snli.append(t)

for t in texts_snli:
    prop_t, t_qwen = pipeline(t)
    print(f"{'-'*50}\nPhrase : {t}")
    print(f"Triplets Qwen avant atomisation :\n{extract_triplets_qwen(t)}")
    print(f"Atomisation :\n{prop_t}")
    print(f"Triplets Qwen apres atomisation :\n{t_qwen}")

    
    
    
    
    
    
    
    
    
    
