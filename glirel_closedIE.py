import json
import torch
import spacy
from glirel import GLiREL
from gliner import GLiNER
from argparse import Namespace
from huggingface_hub import hf_hub_download

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_glirel(model_id, device):
    print(f"Loading GLiREL on {device}...")
    model_file  = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
    config_file = hf_hub_download(repo_id=model_id, filename="glirel_config.json")
    with open(config_file) as f:
        cfg = Namespace(**json.load(f))
    if not hasattr(cfg, "span_mode"):
        cfg.span_mode = getattr(cfg, "rel_mode", "marker")
    model = GLiREL(cfg)
    sd = torch.load(model_file, map_location=device)
    w_ck = sd["token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight"]
    w_mo = model.token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight
    if w_ck.shape != w_mo.shape:
        model.token_rep_layer.bert_layer.model.resize_token_embeddings(w_ck.shape[0])
    model.load_state_dict(sd, strict=True)
    return model.to(device).float().eval()

nlp = spacy.load('en_core_web_sm')

ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1").to(device) #knowledgator/gliner-multitask-large-v0.5

rel_model = load_glirel("jackboyla/glirel-large-v0", device)



def find_token_span(doc, entity):
    char_start = doc.text.lower().find(entity.lower())
    if char_start == -1:
        return None
    char_end = char_start + len(entity) - 1
    start_tok = end_tok = None
    for i, tok in enumerate(doc):
        if start_tok is None and tok.idx <= char_start < tok.idx + len(tok.text):
            start_tok = i
        if tok.idx <= char_end < tok.idx + len(tok.text):
            end_tok = i
    if start_tok is not None and end_tok is not None:
        return (start_tok, end_tok)
    return None

LABEL_DESCRIPTIONS = {
    "FormOf":        "is an inflected form of",
    "IsA":           "is a subtype of",
    "PartOf":        "is a part of",
    "HasA":          "has",
    "Contains":      "contains",
    "UsedFor":       "is used for",
    "CapableOf":     "is capable of",
    "AtLocation":    "is located at",
    "Causes":        "causes",
    "HasSubevent":   "has subevent",
    "HasPrerequisite": "has prerequisite",
    "HasProperty":   "has property",
    "MotivatedByGoal": "is motivated by goal",
    "CreatedBy":     "was created by",
    "Synonym":       "is a synonym of",
    "Antonym":       "is an antonym of",
    "SymbolOf":      "is a symbol of",
    "SimilarTo":     "is similar to",
    "MadeOf":        "is made of",
    "ReceivesAction": "receives the action",
    "PerformsAction": "performs the action",
}

DESC_TO_LABEL = {v: k for k, v in LABEL_DESCRIPTIONS.items()}


LABELS = list(LABEL_DESCRIPTIONS.values())

def extract_triplets_glirel(text):
    doc = nlp(text)
    tokens = [t.text for t in doc]

    entity_labels = ["person", "object", "action", "location", "organization"]
    entities = ner_model.predict_entities(text, entity_labels, threshold=0.0)


    if len(entities) < 2:
        return ""

    # Conversion positions caractères → positions tokens
    ner = []
    for ent in entities:
        span = find_token_span(doc, ent["text"])
        if span is not None:
            ner.append([span[0], span[1], ent["label"], ent["text"]])


    if len(ner) < 2:
        return ""

    with torch.no_grad():
        relations = rel_model.predict_relations(
            tokens, LABELS, threshold=0.3, ner=ner
        )

    if not relations:
        return ""

    triplets = []
    for rel in sorted(relations, key=lambda x: x['score'], reverse=True):
        label_name = DESC_TO_LABEL.get(rel['label'], rel['label'])
        triplets.append(f"{rel['head_text']} | {label_name} | {rel['tail_text']} : {rel['score']}")

    return "\n".join(triplets)

phrases = [
    "Steve Jobs founded Apple.",
    "He doesn't play baseball.", 
    "Chris doesn't like staying with us.",
    "John is eating a ham sandwich", 
    "Two boys are playing outside", 
    "A man is playing an instrument.", 
    "Steve Jobs founded Apple.", 
    "He and his friend played at the park, they played football and basketball.",
    "Frank is in New York, but he plays football in Manchester.",
    "The cat and the dog are in the kitchen.", 
    "A cat and a dog are in a kitchen.", 
    "Frank's dog doesn't eat fruits, he is allergic",
    "Simon didn't call me back, he is busy.", 
    "Younes is working on a project. His friend is playing a video game.", 
    "I have never seen anyone like Frank, he must be gifted.","He must be sick.", 
    "He is sick.", "He played football after eating.",
    "A cat and a dog are in the kitchen",
    "Two dogs are eating and playing football"]


for p in phrases:
    print("-"*50)
    print(f"{p} :")
    print(extract_triplets_glirel(p))