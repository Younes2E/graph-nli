import json
import torch
from datasets import load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

##dataset = load_dataset("stanfordnlp/snli")
##train_set = dataset["train"].filter(lambda x: x["label"] != -1)

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



qwen_model_id = "Qwen/Qwen3-4B-Thinking-2507"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    dtype="auto").to(device)


list_relations = {'RelatedTo' : "The most general relation. There is some positive relationship between A and B. Symmetric." ,
'FormOf' : "A is an inflected form of B; B is the root word of A." ,
'IsA' : "A is a subtype or a specific instance of B." ,
'PartOf' : "A is a part of B." ,
'HasA' : "B belongs to A." ,
'UsedFor' : "A is used for B; the purpose of A is B." ,
'CapableOf' : "Something that A can typically do is B." ,
'AtLocation' : "A is a location for B." ,
'Causes' : "A and B are events, and it is typical for A to cause B." ,
'HasSubevent' : "A and B are events, and B happens as a subevent of A." ,
'HasPrerequisite' : "In order for A to happen, B needs to happen; B is a dependency of A." ,
'HasProperty' : "A has B as a property; A can be described as B." ,
'MotivatedByGoal' : "Someone does A because they want result B; A is a step toward accomplishing the goal B." ,
'CreatedBy' : "B is a process or agent that creates A." ,
'Synonym' : "A and B have very similar meanings. Symmetric." ,
'Antonym' : "A and B are opposites in some relevant way, such as being opposite ends of a scale, or a key difference between them. Symmetric." ,
'DistinctFrom' : "A and B are distinct member of a set; something that is A is not B. Symmetric." ,
'DerivedFrom' : "A is a word or phrase that appears within B and contributes to B's meaning." ,
'SymbolOf' : "A symbolically represents B." ,
'DefinedAs' : "A and B overlap considerably in meaning, and B is a more explanatory version of A." ,
'MannerOf' : "A is a specific way to do B. Similar to 'IsA', but for verbs." ,
'LocatedNear' : "A and B are typically found near each other. Symmetric." ,
'HasContext' : "A is a word used in the context of B." ,
'SimilarTo' : "A is similar to B. Symmetric." ,
'CausesDesire' : "A makes someone want B." ,
'MadeOf' : "A is made of B." ,
'ReceivesAction' : "B can be done to A.",
'PerformsAction' : "A is performing the action B"}


def extract_triplets_deepseek(text):
    print("todo")

def extract_triplets_qwen(text):
    prompt = f"""
    Extract all factual (subject, relation, object) triples from sentence. 
    One triple per line in the format: 
    subject | relation | object

    If no triple can be extracted, write nothing. 

    Here are the only relations available : {list_relations.keys()}
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

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

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




#texts_snli = [train_set[i][j] for j in ["premise", "hypothesis"] for i in [4131,2656,922,1384,7048]]
#texts = ["Frank's dog doesn't eat fruits, he is allergic","Simon didn't call me back, he is busy.", "Younes is working on a project. His friend is playing a video game.", "I have never seen anyone like Frank, he must be gifted.","He must be sick.", "He is sick.", "He played football after eating.","A cat and a dog are in the kitchen","Two dogs are eating and playing football"]
#for t in texts:
    #texts_snli.append(t)

np_data = dataset.to_numpy()
sentence_data = [np_data[i][j] for j in [1, 2] for i in [5,45,2450]]

for t in sentence_data:
    prop_t, t_qwen = pipeline(t)
    print(f"{'-'*50}\nPhrase : {t}")
    print(f"Triplets Qwen avant atomisation :\n{extract_triplets_qwen(t)}")
    print(f"Atomisation :\n{prop_t}")
    print(f"Triplets Qwen apres atomisation :\n{t_qwen}")

    
    
    
    
    
    
    
    
    
    
