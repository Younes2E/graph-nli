import json
import matplotlib.pyplot as plt
import networkx as nx
import torch
import spacy
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer



nlp = spacy.load("en_core_web_sm")

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
    
    return set(propositions)



def extract_triplets(text):
    doc = nlp(text)
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = spacy.util.filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    triples = []

    for token in doc:
        if token.pos_ in ["VERB", "AUX"] or token.dep_ == "ROOT":
            sujet = None
            objet = None

            # Exploration des dépendances directes du verbe
            for child in token.children:
                # 1. Capture du sujet
                if "subj" in child.dep_:
                    sujet = child.text

                # 2. Capture de l'objet direct, de l'attribut ou du complément
                if child.dep_ in ["dobj", "attr", "acomp"]:
                    obj_text = child.text
                    # Optionnel : Si l'objet a lui-même une préposition (ex: "president of the USA")
                    # on la récupère pour garder le contexte complet du nœud
                    preps = [c for c in child.children if c.dep_ == "prep"]
                    for p in preps:
                        pobjs = [
                            cc.text for cc in p.children if cc.dep_ == "pobj"
                        ]
                        if pobjs:
                            obj_text += f" {p.text} {pobjs[0]}"
                    objet = obj_text

                # 3. Capture si la relation passe par une préposition directe (ex: "jumping over...")
                elif child.dep_ == "prep":
                    pobjs = [c.text for c in child.children if c.dep_ == "pobj"]
                    if pobjs:
                        objet = f"{child.text} {pobjs[0]}"

            # Si on a trouvé un couple Sujet/Objet valide, on extrait le triplet
            if sujet and objet:
                # Gestion de la négation ("don't", "never", "not") pour ne pas fausser le sens
                neg = "".join(
                    [c.text + " " for c in token.children if c.dep_ == "neg"]
                )
                relation = f"{neg}{token.text}"

                triples.append((sujet, relation, objet))

    return triples



def graph_build(text):
    propositions = extract_atomic_propositions(text)
    relations = []
    for p in propositions:
        triplets = extract_triplets(p)
        relations.append(triplets)
            
    return propositions, relations




texts = ["Frank's dog doesn't eat fruits, he is allergic","Simon didn't call me back, he is busy."]
texts_snli = [train_set[i][j] for j in ["premise", "hypothesis"] for i in [0,15,56,78,150]]

for t in texts_snli:
    prop_t, graph_t = graph_build(t)
    print(f"\nPhrase : {t} :")
    print(f"Propositions : {prop_t}")
    print(f"Triplets avant atomisation : {extract_triplets(t)}")
    print(f"Triplets apres atomisation : {graph_t}")
    
    
    
    
    
    
    
    
    
    
    
    
