import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

dataset = load_dataset("stanfordnlp/snli")
snli_train = dataset["train"].filter(lambda x: x["label"] != -1)
snli_test = dataset["test"].filter(lambda x: x["label"] != -1)


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


list_relations  = {
"FormOf" : "[SUBJECT] is an inflected form of [OBJECT]" ,
"IsA" : "[SUBJECT] is a strict taxonomic subtype, class, or specific instance of [OBJECT]" ,
"CouldBe" : "[SUBJECT] is conditionally, hypothetically, or possibly described as [OBJECT], categorized as [OBJECT], or executing [OBJECT]",
"PartOf": "[SUBJECT] is an intrinsic physical component, member, or structural part of [OBJECT]",
"HasA" : "[SUBJECT] physically possesses or owns [OBJECT]",
"Contains" : "[SUBJECT] physically encloses, holds inside, or is chemically/structurally composed of [OBJECT] inside it",
"HasQuantity" : "[SUBJECT] has a quantity of [OBJECT]",
"UsedFor" : "[SUBJECT] is used for [OBJECT]; the purpose of [SUBJECT] is [OBJECT]" ,
"CapableOf" : "Something that [SUBJECT] can typically do is [OBJECT]" ,
"AtLocation" : "[SUBJECT] is located at [OBJECT], [SUBJECT] can be an event taking place at [OBJECT]" ,
"AtTime" : "[SUBJECT] took place in, during, or relative to the temporal frame, date, or event [OBJECT]",
"Causes" : "[SUBJECT] and [OBJECT] are events, and [SUBJECT] causes [OBJECT]" ,
"HasSubevent" : "[SUBJECT] and [OBJECT] are events, and [OBJECT] happens as a subevent of [SUBJECT]" ,
"HasPrerequisite" : "In order for [SUBJECT] to happen, [OBJECT] needs to happen; [OBJECT] is a dependency of [SUBJECT]" ,
"HasProperty" : "[SUBJECT] has [OBJECT] as a property; [SUBJECT] can be described as [OBJECT]" ,
"MotivatedByGoal" : "Someone does [SUBJECT] because they want result [OBJECT]; [SUBJECT] is a step toward accomplishing the goal [OBJECT]" ,
"CreatedBy" : "[OBJECT] is a process or agent that creates [SUBJECT]" ,
"Synonym" : "[SUBJECT] and [OBJECT] have very similar meanings. Symmetric" ,
"Antonym" : "[SUBJECT] and [OBJECT] are opposites in some relevant way" ,
"HasContext" : "[SUBJECT] is inherently associated with, defined by, or belongs to the domain, or category [OBJECT]" ,
"SimilarTo" : "[SUBJECT] is similar to [OBJECT]. Symmetric" ,
"MadeOf" : "[SUBJECT] is made of [OBJECT]" ,
"ReceivesAction" : "[OBJECT] can be done to [SUBJECT]",
"PerformsAction" : "[SUBJECT] is doing the action [OBJECT], [OBJECT] is usually a verb"}


def extract_triplets_qwen(text):
    prompt = f"""
    Extract all factual ([SUBJECT], [RELATION], [OBJECT]) triples from sentence. 
    One triple per line in the format: 
    [SUBJECT] | [RELATION] | [OBJECT]

    No explanations. If no triple can be extracted, write nothing. 

    Allowed Relations and Type Constraints:
    {json.dumps(list_relations, indent=2)}
    Do not use any other relation.

    RULES: 
    1. If an action is negated in the sentence (e.g., "didn't call", "is not eating"), you MUST capture the negation inside the [VERB] node using 'not' (e.g., 'not call', 'not eating').
    2. When an action involves multiple elements at once (e.g., an actor, a target, a recipient, a tool, or a location), do NOT link the secondary elements to each other. Instead, make the action the central hub and link ALL participants directly to that action.
    
    EXEMPLES:
    Input: John is eating a ham sandwich
    Output:
    John | PerformsAction | eating
    sandwich | ReceivesAction | eating
    sandwich | Contains | ham

    Input: A man is playing an instrument
    Output:
    man | PerformsAction | playing
    instrument | ReceivesAction | playing

    Input: Simon didn't call me back. He must be busy.
    Output:
    Simon | PerformsAction | not call
    me | ReceivesAction | not call
    Simon | CanBe | busy

    
    Sentence: {text}"""

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful information extraction system."},
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
            print(f"\n{t}, Mauvais format de triplet.\n")

    return triples


def build_graph(text): 
    propositions = extract_atomic_propositions(text)
    triplets_qwen = []
    for p in propositions:
        triplets_qwen+=extract_triplets_qwen(p)
            
    return set(triplets_qwen)

    
    
    
    
    
    
    
    
    
