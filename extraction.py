import numpy as np
import pandas as pd
import json
from argparse import Namespace
import spacy
import torch
from huggingface_hub import hf_hub_download
from glirel import GLiREL


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

model = load_glirel("jackboyla/glirel-large-v0","cuda")

text = "Jack Dorsey's father, Tim Dorsey, is a licensed pilot. Jack met his wife Sarah Paulson in New York in 2003. They have one son, Edward."

nlp = spacy.load('en_core_web_sm')

labels = ['RelatedTo', 'FormOf', 'isA', 'PartOf', 'HasA','UsedFor', 'CapableOf', 'AtLocation','Causes','HasSubevent','HasFirstSubevent','HasLastSubevent', 'HasPrerequisite', 'HasProperty', 'MotivatedByGoal', 'ObstructedBy', 'Desires', 'CreatedBy', 'Synonym', 'Antonym', 'DistinctFrom', 'DerivedFrom', 'SymbolOf', 'DefinedAs', 'MannerOf', 'LocatedNear', 'HasContext', 'SimilarTo', 'EtymologicallyRelatedTo', 'EtymologicallyDerivedFrom', 'CausesDesire', 'MadeOf', 'ReceivesAction']

doc = nlp(text)
tokens = [token.text for token in doc]
ner = [[ent.start, ent.end, ent.label_, ent.text] for ent in doc.ents]
relations = model.predict_relations(tokens, labels, threshold=0.01, ner=ner)
sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
for item in sorted_data_desc:
    print(f"{item['head_text']} --> {item['label']} --> {item['tail_text']} | socre: {item['score']}")
